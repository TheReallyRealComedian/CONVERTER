"""DOC-API P1 — the document-conversion service surface.

Covers the sprint's stop conditions: the auth failure paths (fail-closed 503,
constant-time 401, session path independent of the env token), the job states
(pending / ready / failed / transient-Redis-stays-pending), the size check
(header gate before parsing + on-disk backstop), and the end-to-end flow
submit → poll → result for a PDF **and** a DOCX.

RQ is mocked at the ``task_queue`` / ``Job`` singletons (``mock_redis_queue``);
the worker task runs in-process against the monkeypatched shared-volume dir.
The PDF end-to-end runs the REAL local extraction path (PyMuPDF, no API key —
the deterministic degradation ``PDFExtractionService`` ships); the DOCX
end-to-end runs the REAL serializer (``elements_to_markdown`` is pure) over a
stubbed ``partition`` (unstructured is not installed on the dev box, see
conftest).
"""
import hashlib
import io
import json
import sys
from types import SimpleNamespace

import pytest
from rq.exceptions import NoSuchJobError

import fitz

from app_pkg.config import (
    TIMEOUT_DOC_JOB_BASE_SECONDS,
    TIMEOUT_DOC_JOB_PER_PAGE_SECONDS,
    TIMEOUT_RQ_JOB_HARD_CAP,
    doc_convert_job_timeout_for,
)
from models import Conversion, User, db
from services import document_conversions as doc_lib
from tasks import convert_document_task

DOC_URL = '/api/document-conversions'
DOC_TOKEN = 'doc-test-token-91af'


def _auth(token=DOC_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _pdf_bytes(text='Hallo Konvertierung. Dies ist ein Testdokument.'):
    """A real one-page PDF with a text layer (PyMuPDF), as bytes."""
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 100), text)
    data = pdf.tobytes()
    pdf.close()
    return data


def _upload(data, filename):
    return {'file': (io.BytesIO(data), filename)}


def _post(client, data=b'%PDF-fake', filename='doc.pdf', headers=None):
    return client.post(DOC_URL, data=_upload(data, filename), headers=headers,
                       content_type='multipart/form-data')


def _conversion_count(app):
    with app.app_context():
        return Conversion.query.filter_by(
            conversion_type='document_conversion').count()


@pytest.fixture
def doc_convert_dir(tmp_path, monkeypatch):
    """Point the shared-volume namespace dir at a tmp dir.

    One patch point: routes and task call the path *functions*, which read
    the module global at call time.
    """
    d = tmp_path / 'doc_conversions'
    monkeypatch.setattr(doc_lib, 'DOC_CONVERT_DIR', str(d))
    return d


@pytest.fixture
def doc_token(monkeypatch):
    monkeypatch.setenv('DOC_CONVERT_TOKEN', DOC_TOKEN)
    return DOC_TOKEN


# --- auth matrix: 503 fail-closed / 401 / session path -----------------------

def test_post_fail_closed_without_token(app, client, test_user, monkeypatch,
                                        mock_redis_queue, doc_convert_dir):
    monkeypatch.delenv('DOC_CONVERT_TOKEN', raising=False)
    # Unset → 503 even with a Bearer presented (config check precedes auth).
    assert _post(client, headers=_auth()).status_code == 503
    monkeypatch.setenv('DOC_CONVERT_TOKEN', '')
    assert _post(client, headers=_auth()).status_code == 503
    assert _conversion_count(app) == 0


def test_post_401_missing_and_wrong_token(app, client, test_user, doc_token,
                                          mock_redis_queue, doc_convert_dir):
    assert _post(client).status_code == 401
    assert _post(client, headers=_auth('the-wrong-token')).status_code == 401
    assert _conversion_count(app) == 0


def test_get_auth_matrix(app, client, test_user, monkeypatch,
                         mock_redis_queue, doc_convert_dir):
    monkeypatch.delenv('DOC_CONVERT_TOKEN', raising=False)
    assert client.get(f'{DOC_URL}/1', headers=_auth()).status_code == 503
    monkeypatch.setenv('DOC_CONVERT_TOKEN', DOC_TOKEN)
    assert client.get(f'{DOC_URL}/1').status_code == 401
    assert client.get(f'{DOC_URL}/1',
                      headers=_auth('the-wrong-token')).status_code == 401


def test_post_503_without_target_user(app, client, doc_token,
                                      mock_redis_queue, doc_convert_dir):
    # Token valid, but no user exists at all → 503, nothing created.
    assert _post(client, headers=_auth()).status_code == 503
    assert _conversion_count(app) == 0


def test_session_path_works_without_env_token(app, authenticated_client,
                                              monkeypatch, mock_redis_queue,
                                              doc_convert_dir):
    # The web/app path must not depend on the service token being configured.
    monkeypatch.delenv('DOC_CONVERT_TOKEN', raising=False)
    resp = _post(authenticated_client, data=_pdf_bytes())
    assert resp.status_code == 202
    body = resp.get_json()
    assert body['status'] == 'pending'
    assert body['job_id'] == 'test-job-123'
    # Poll works on the same session.
    mock_redis_queue['fetch'].return_value.is_failed = False
    poll = authenticated_client.get(f"{DOC_URL}/{body['id']}")
    assert poll.status_code == 200
    assert poll.get_json()['status'] == 'pending'


# --- upload validation ---------------------------------------------------------

def test_post_400_on_missing_file_empty_name_bad_ext(app, client, test_user,
                                                     doc_token, mock_redis_queue,
                                                     doc_convert_dir):
    assert client.post(DOC_URL, data={}, headers=_auth(),
                       content_type='multipart/form-data').status_code == 400
    assert _post(client, filename='', headers=_auth()).status_code == 400
    assert _post(client, filename='malware.exe', headers=_auth()).status_code == 400
    assert _post(client, filename='README', headers=_auth()).status_code == 400
    assert _post(client, data=b'', filename='empty.pdf',
                 headers=_auth()).status_code == 400
    assert _conversion_count(app) == 0


def test_post_413_header_gate_before_parse(app, client, test_user, doc_token,
                                           mock_redis_queue, doc_convert_dir,
                                           monkeypatch):
    import app_pkg.document_api as mod
    monkeypatch.setattr(mod, 'MAX_DOCUMENT_UPLOAD_BYTES', 1000)
    resp = _post(client, data=b'x' * 5000, headers=_auth())
    assert resp.status_code == 413
    assert _conversion_count(app) == 0
    # Nothing was parsed onto the volume — no source file, no leftover tmp.
    assert not doc_convert_dir.exists() or list(doc_convert_dir.iterdir()) == []


def test_post_413_disk_backstop_when_header_lies(app, client, test_user,
                                                 doc_token, mock_redis_queue,
                                                 doc_convert_dir, monkeypatch):
    # Disable the header gate to simulate a missing/lying Content-Length; the
    # on-disk check after the save must still reject and leave no residue.
    import app_pkg.document_api as mod
    monkeypatch.setattr(mod, 'MAX_DOCUMENT_UPLOAD_BYTES', 1000)
    monkeypatch.setattr(mod, '_oversize_response', lambda: None)
    resp = _post(client, data=b'x' * 5000, headers=_auth())
    assert resp.status_code == 413
    assert _conversion_count(app) == 0
    assert list(doc_convert_dir.iterdir()) == []


# --- submit mechanics ----------------------------------------------------------

def test_post_creates_pending_row_and_enqueues(app, client, test_user, doc_token,
                                               mock_redis_queue, doc_convert_dir):
    pdf = _pdf_bytes()
    resp = _post(client, data=pdf, filename='Bericht Q3.pdf', headers=_auth())
    assert resp.status_code == 202
    body = resp.get_json()
    assert body['status'] == 'pending'
    assert body['job_id'] == 'test-job-123'
    cid = body['id']

    with app.app_context():
        row = db.session.get(Conversion, cid)
        assert row.conversion_type == 'document_conversion'
        assert row.user_id == test_user['id']
        assert row.title == 'Bericht Q3.pdf'
        assert row.content == ''
        assert row.source_filename == 'Bericht Q3.pdf'
        assert row.source_size_bytes == len(pdf)
        assert row.lifecycle_status == 'archive'
        metadata = json.loads(row.metadata_json)
    assert metadata['doc_status'] == 'pending'
    assert metadata['source_format'] == 'pdf'
    assert metadata['page_count'] == 1
    assert metadata['source_sha256'] == hashlib.sha256(pdf).hexdigest()
    assert metadata['job_id'] == 'test-job-123'

    # Source landed under the id-derived name with the exact upload bytes.
    source = doc_convert_dir / f'source_{cid}.pdf'
    assert source.read_bytes() == pdf
    # No leftover tmp spool file.
    assert sorted(p.name for p in doc_convert_dir.iterdir()) == [source.name]

    # Enqueue carried the task, the id-args and the page-scaled envelope.
    args, kwargs = mock_redis_queue['queue'].enqueue.call_args
    assert args == (convert_document_task, cid, 'pdf')
    assert kwargs['job_timeout'] == doc_convert_job_timeout_for(1)
    assert kwargs['meta'] == {'user_id': test_user['id'], 'conversion_id': cid}


def test_doc_job_timeout_scales_from_pages():
    floor = TIMEOUT_DOC_JOB_BASE_SECONDS + TIMEOUT_DOC_JOB_PER_PAGE_SECONDS
    assert doc_convert_job_timeout_for(None) == floor
    assert doc_convert_job_timeout_for(0) == floor
    assert doc_convert_job_timeout_for(1) == floor
    assert doc_convert_job_timeout_for(10) == (
        TIMEOUT_DOC_JOB_BASE_SECONDS + 10 * TIMEOUT_DOC_JOB_PER_PAGE_SECONDS)
    assert doc_convert_job_timeout_for(10_000) == TIMEOUT_RQ_JOB_HARD_CAP


# --- job states via reconcile ---------------------------------------------------

def _submit(client, app, data=None, filename='doc.pdf'):
    resp = _post(client, data=data if data is not None else _pdf_bytes(),
                 filename=filename, headers=_auth())
    assert resp.status_code == 202
    return resp.get_json()['id']


def test_get_stays_pending_while_job_runs(app, client, test_user, doc_token,
                                          mock_redis_queue, doc_convert_dir):
    cid = _submit(client, app)
    mock_redis_queue['fetch'].return_value.is_failed = False
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'pending'
    assert body['markdown'] is None
    assert body['error'] is None


def test_get_ready_reads_structured_result(app, client, test_user, doc_token,
                                           mock_redis_queue, doc_convert_dir):
    cid = _submit(client, app, filename='Bericht.pdf')
    doc_lib.write_result_file(cid, {'markdown': '# Hallo\n\nWelt.',
                                    'warnings': ['Tabelle degradiert']})
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'ready'
    assert body['markdown'] == '# Hallo\n\nWelt.'
    assert body['warnings'] == ['Tabelle degradiert']
    assert body['error'] is None
    assert body['source']['filename'] == 'Bericht.pdf'
    assert body['source']['format'] == 'pdf'
    assert body['source']['page_count'] == 1
    assert body['source']['size_bytes'] > 0
    assert body['created_at']

    # Markdown persisted into the row; scratch files discarded post-commit.
    with app.app_context():
        row = db.session.get(Conversion, cid)
        assert row.content == '# Hallo\n\nWelt.'
    assert list(doc_convert_dir.iterdir()) == []

    # Terminal state is idempotent — second poll serves from the DB.
    again = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert again['status'] == 'ready'
    assert again['markdown'] == '# Hallo\n\nWelt.'


def test_get_failed_from_rq_keeps_exc_tail(app, client, test_user, doc_token,
                                           mock_redis_queue, doc_convert_dir):
    cid = _submit(client, app)
    job = mock_redis_queue['fetch'].return_value
    job.is_failed = True
    job.exc_info = 'x' * 3000 + 'ValueError: kaputt'
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'failed'
    assert body['markdown'] is None
    # Tail, not head — the exception line lives at the end (NARR-FAIL).
    assert body['error'].endswith('ValueError: kaputt')
    assert len(body['error']) == 2000
    # Source file cleaned up on the failed flip.
    assert list(doc_convert_dir.iterdir()) == []


def test_get_failed_when_job_gone(app, client, test_user, doc_token,
                                  mock_redis_queue, doc_convert_dir):
    cid = _submit(client, app)
    mock_redis_queue['fetch'].side_effect = NoSuchJobError()
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'failed'
    assert body['error'] == 'Job nicht mehr auffindbar.'


def test_get_stays_pending_on_transient_redis_error(app, client, test_user,
                                                    doc_token, mock_redis_queue,
                                                    doc_convert_dir):
    cid = _submit(client, app)
    mock_redis_queue['fetch'].side_effect = ConnectionError('redis down')
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'pending'
    assert body['error'] is None
    # Next poll with Redis back → resolves normally.
    mock_redis_queue['fetch'].side_effect = None
    mock_redis_queue['fetch'].return_value.is_failed = False
    assert client.get(f'{DOC_URL}/{cid}',
                      headers=_auth()).get_json()['status'] == 'pending'


def test_get_failed_on_unreadable_result_file(app, client, test_user, doc_token,
                                              mock_redis_queue, doc_convert_dir):
    cid = _submit(client, app)
    (doc_convert_dir / f'result_{cid}.json').write_text('{not json', encoding='utf-8')
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'failed'
    assert body['error'] == 'Ergebnisdatei unlesbar.'
    # The broken file is kept for diagnosis.
    assert (doc_convert_dir / f'result_{cid}.json').exists()


def test_get_owner_and_type_scoped_404(app, client, authenticated_client,
                                       test_user, doc_token, mock_redis_queue,
                                       doc_convert_dir):
    with app.app_context():
        bob = User(username='bob')
        bob.set_password('password1234')
        db.session.add(bob)
        db.session.flush()
        foreign = Conversion(user_id=bob.id, conversion_type='document_conversion',
                             title='fremd', content='x')
        wrong_type = Conversion(user_id=test_user['id'],
                                conversion_type='markdown_input',
                                title='typfremd', content='x')
        db.session.add_all([foreign, wrong_type])
        db.session.commit()
        foreign_id, wrong_type_id = foreign.id, wrong_type.id

    # Token path resolves to the first user (alice) → bob's row is 404.
    assert client.get(f'{DOC_URL}/{foreign_id}',
                      headers=_auth()).status_code == 404
    # Wrong conversion_type is an indistinguishable 404 (no type leak).
    assert client.get(f'{DOC_URL}/{wrong_type_id}',
                      headers=_auth()).status_code == 404
    # Session path (alice) sees the same 404s.
    assert authenticated_client.get(f'{DOC_URL}/{foreign_id}').status_code == 404
    assert authenticated_client.get(f'{DOC_URL}/{wrong_type_id}').status_code == 404


# --- end-to-end: submit → run task → poll → result ------------------------------

def test_pdf_end_to_end_with_real_local_extraction(app, client, test_user,
                                                   doc_token, mock_redis_queue,
                                                   doc_convert_dir, monkeypatch):
    """Submit → worker task (REAL PyMuPDF extraction, no API key) → poll → ready."""
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    cid = _submit(client, app, data=_pdf_bytes('Hallo Konvertierung.'),
                  filename='echt.pdf')

    # Run the DB-free worker task in-process, exactly as RQ would call it.
    convert_document_task(cid, 'pdf')

    # Structured result on the volume, source consumed by the task's finally.
    assert (doc_convert_dir / f'result_{cid}.json').exists()
    assert not (doc_convert_dir / f'source_{cid}.pdf').exists()

    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'ready'
    assert 'Hallo Konvertierung.' in body['markdown']
    assert body['source']['format'] == 'pdf'
    assert body['source']['page_count'] == 1


def test_docx_end_to_end_with_real_serializer(app, client, test_user, doc_token,
                                              mock_redis_queue, doc_convert_dir,
                                              monkeypatch):
    """Submit → worker task (stubbed partition, REAL serializer) → poll → ready.

    ``unstructured`` is stubbed (conftest), so the partition output is a
    synthetic element list in the measured shape; ``elements_to_markdown``
    runs for real and must produce heading + paragraph + a table warning.
    """
    def fake_partition(filename=None, strategy=None):
        def el(category, text='', depth=None, html=None):
            return SimpleNamespace(
                category=category, text=text,
                metadata=SimpleNamespace(category_depth=depth, page_number=1,
                                         text_as_html=html))
        return [
            el('Title', 'Kapitel Eins', depth=0),
            el('NarrativeText', 'Ein Absatz mit Inhalt.'),
            el('Table', 'Stoff Menge'),  # no text_as_html → warning
        ]

    monkeypatch.setattr(sys.modules['unstructured.partition.auto'],
                        'partition', fake_partition)

    cid = _submit(client, app, data=b'PK\x03\x04 fake docx bytes',
                  filename='bericht.docx')
    convert_document_task(cid, 'docx')

    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'ready'
    assert '# Kapitel Eins' in body['markdown']
    assert 'Ein Absatz mit Inhalt.' in body['markdown']
    assert body['warnings'] == [
        'Tabelle ohne text_as_html — als Fliesstext ausgegeben']
    assert body['source']['format'] == 'docx'
    assert body['source']['page_count'] is None


def test_task_failure_leaves_no_result_and_consumes_source(app, client,
                                                           test_user, doc_token,
                                                           mock_redis_queue,
                                                           doc_convert_dir,
                                                           monkeypatch):
    def boom(filename=None, strategy=None):
        raise RuntimeError('partition explodiert')

    monkeypatch.setattr(sys.modules['unstructured.partition.auto'],
                        'partition', boom)
    cid = _submit(client, app, data=b'kaputt', filename='defekt.docx')

    with pytest.raises(RuntimeError):
        convert_document_task(cid, 'docx')

    # No result file → reconcile keys on the RQ job; source is consumed.
    assert not (doc_convert_dir / f'result_{cid}.json').exists()
    assert not (doc_convert_dir / f'source_{cid}.docx').exists()

    job = mock_redis_queue['fetch'].return_value
    job.is_failed = True
    job.exc_info = 'Traceback ...\nRuntimeError: partition explodiert'
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'failed'
    assert body['error'].endswith('RuntimeError: partition explodiert')
