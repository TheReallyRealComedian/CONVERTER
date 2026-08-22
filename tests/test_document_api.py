"""DOC-API — the document-conversion service surface (P1 + P2).

P1 stop conditions: auth failure paths (fail-closed 503, constant-time 401,
session path independent of the env token), job states (pending / ready /
failed / transient-Redis-stays-pending), the size check (header gate before
parsing + on-disk backstop), and end-to-end submit → poll → result for a PDF
**and** a DOCX.

P2 stop conditions: the answer shape (provenance per unit, degradations,
usage, mode, budget), strict per-job mode with a settings default (own
namespace in the shared settings blob — coexistence with the learn keys is
sentinel-tested in BOTH directions), content-hash idempotency, and the
budget-cap mechanic **proven against placeholder backends**: the provenance
of the affected pages demonstrably flips mid-document, and the pipeline's
payload flows through the existing reconcile unchanged.

RQ is mocked at the ``task_queue`` / ``Job`` singletons (``mock_redis_queue``);
the worker task runs in-process against the monkeypatched shared-volume dir.
The PDF end-to-end runs the REAL local extraction path (PyMuPDF, no API key —
provably deterministic); the DOCX end-to-end runs the REAL serializer
(``elements_to_markdown`` is pure) over a stubbed ``partition`` (unstructured
is not installed on the dev box, see conftest).
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
    DOC_CONVERT_BUDGET_EUR,
    TIMEOUT_DOC_JOB_BASE_SECONDS,
    TIMEOUT_DOC_JOB_LOCAL_BASE_SECONDS,
    TIMEOUT_DOC_JOB_LOCAL_PER_PAGE_SECONDS,
    TIMEOUT_DOC_JOB_PER_PAGE_SECONDS,
    TIMEOUT_RQ_JOB_HARD_CAP,
    doc_convert_job_timeout_for,
)
from models import Conversion, User, db
from services import document_conversions as doc_lib
from services.document_conversions import build_result_payload, degradation
from services.document_pipeline import run_paged_conversion
from tasks import convert_document_task

DOC_URL = '/api/document-conversions'
SETTINGS_URL = '/api/document-conversions/settings'
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


def _upload(data, filename, mode=None):
    fields = {'file': (io.BytesIO(data), filename)}
    if mode is not None:
        fields['mode'] = mode
    return fields


def _post(client, data=b'%PDF-fake', filename='doc.pdf', headers=None, mode=None):
    return client.post(DOC_URL, data=_upload(data, filename, mode),
                       headers=headers, content_type='multipart/form-data')


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
    assert body['mode'] == 'lokal'  # default without a stored setting (DOC-WEB)
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
    assert metadata['mode'] == 'lokal'
    assert metadata['budget_eur'] == DOC_CONVERT_BUDGET_EUR
    assert metadata['source_format'] == 'pdf'
    assert metadata['page_count'] == 1
    assert metadata['source_sha256'] == hashlib.sha256(pdf).hexdigest()
    assert metadata['job_id'] == 'test-job-123'

    # Source landed under the id-derived name with the exact upload bytes.
    source = doc_convert_dir / f'source_{cid}.pdf'
    assert source.read_bytes() == pdf
    # No leftover tmp spool file.
    assert sorted(p.name for p in doc_convert_dir.iterdir()) == [source.name]

    # Enqueue carried the task, the resolved job args and the page-scaled
    # envelope.
    args, kwargs = mock_redis_queue['queue'].enqueue.call_args
    assert args == (convert_document_task, cid, 'pdf', 'lokal',
                    DOC_CONVERT_BUDGET_EUR, 1)
    assert kwargs['job_timeout'] == doc_convert_job_timeout_for(1, 'lokal')
    assert kwargs['meta'] == {'user_id': test_user['id'], 'conversion_id': cid}


def test_doc_job_timeout_scales_from_pages():
    floor = TIMEOUT_DOC_JOB_BASE_SECONDS + TIMEOUT_DOC_JOB_PER_PAGE_SECONDS
    assert doc_convert_job_timeout_for(None) == floor
    assert doc_convert_job_timeout_for(0) == floor
    assert doc_convert_job_timeout_for(1) == floor
    assert doc_convert_job_timeout_for(10) == (
        TIMEOUT_DOC_JOB_BASE_SECONDS + 10 * TIMEOUT_DOC_JOB_PER_PAGE_SECONDS)
    assert doc_convert_job_timeout_for(10_000) == TIMEOUT_RQ_JOB_HARD_CAP


def test_doc_job_timeout_lokal_rides_the_mineru_curve():
    """DOC-LOCAL: mode=lokal switches to the measured mineru envelope
    (61 s + 2,5 s/Seite gemessen; Container-Deadline 300 + 10 s/Seite) —
    and the envelope must stay ABOVE the module's container deadline for
    every page count, or RQ could kill a run its own deadline still allows."""
    from services.pdf_local import mineru_run_timeout_for

    floor = (TIMEOUT_DOC_JOB_LOCAL_BASE_SECONDS
             + TIMEOUT_DOC_JOB_LOCAL_PER_PAGE_SECONDS)
    assert doc_convert_job_timeout_for(None, 'lokal') == floor
    assert doc_convert_job_timeout_for(1, 'lokal') == floor
    # 12_grosses-pdf, the sprint's named case: 280 pages → 3400 s (~57 min)
    # envelope over a measured ~766 s run.
    assert doc_convert_job_timeout_for(280, 'lokal') == 3400
    assert doc_convert_job_timeout_for(10_000, 'lokal') == TIMEOUT_RQ_JOB_HARD_CAP
    for n in (1, 12, 280, 1000):
        assert (doc_convert_job_timeout_for(n, 'lokal')
                > mineru_run_timeout_for(n))
    # Cloud mode is byte-identical to the pre-DOC-LOCAL envelope:
    assert doc_convert_job_timeout_for(10, 'cloud') == doc_convert_job_timeout_for(10)


# --- P2: per-job mode, strictly read --------------------------------------------

def test_mode_strict_read(app, client, test_user, doc_token,
                          mock_redis_queue, doc_convert_dir):
    # Only the exact values switch (house pattern since LEARN-MORE).
    for bad in ('Cloud', 'CLOUD', '', 'auto', 'local', ' lokal'):
        resp = _post(client, headers=_auth(), mode=bad)
        assert resp.status_code == 400, f'mode={bad!r} must 400'
    assert _conversion_count(app) == 0
    ok = _post(client, headers=_auth(), mode='lokal', data=_pdf_bytes())
    assert ok.status_code == 202
    assert ok.get_json()['mode'] == 'lokal'


def test_mode_default_from_settings(app, authenticated_client, monkeypatch,
                                    mock_redis_queue, doc_convert_dir):
    monkeypatch.delenv('DOC_CONVERT_TOKEN', raising=False)
    # No stored setting → lokal (DOC-WEB default: no money without a choice).
    r1 = _post(authenticated_client, data=_pdf_bytes('Erstes Dokument.'))
    assert r1.get_json()['mode'] == 'lokal'
    # Stored default flips the resolution for mode-less submits.
    put = authenticated_client.put(SETTINGS_URL, json={'default_mode': 'cloud'})
    assert put.status_code == 200
    r2 = _post(authenticated_client, data=_pdf_bytes('Zweites Dokument.'))
    assert r2.get_json()['mode'] == 'cloud'
    # An explicit mode still wins over the default.
    r3 = _post(authenticated_client, data=_pdf_bytes('Drittes Dokument.'),
               mode='lokal')
    assert r3.get_json()['mode'] == 'lokal'


# --- P2: settings (own namespace in the shared blob) -----------------------------

def test_settings_roundtrip_strict_and_auth(app, authenticated_client,
                                            test_user):
    assert authenticated_client.get(SETTINGS_URL).get_json() == {
        'default_mode': 'lokal'}
    # Strict write: unknown key / invalid value / non-object → 400.
    assert authenticated_client.put(
        SETTINGS_URL, json={'default_mode': 'auto'}).status_code == 400
    assert authenticated_client.put(
        SETTINGS_URL, json={'quatsch': 1}).status_code == 400
    assert authenticated_client.put(SETTINGS_URL, json='nope').status_code == 400
    # Roundtrip.
    assert authenticated_client.put(
        SETTINGS_URL, json={'default_mode': 'lokal'}).status_code == 200
    assert authenticated_client.get(SETTINGS_URL).get_json() == {
        'default_mode': 'lokal'}
    # The service token is NOT a session — settings are a session surface.
    # Fresh client: authenticated_client IS the shared `client` fixture with
    # a session cookie, so a session-less check needs its own instance.
    anonymous = app.test_client()
    assert anonymous.get(SETTINGS_URL, headers=_auth()).status_code == 401


def test_settings_namespaces_coexist(app, authenticated_client, test_user):
    """Sentinel: neither feature's save may drop the other's keys."""
    assert authenticated_client.put(
        '/api/learn/settings', json={'daily_new_limit': 7}).status_code == 200
    assert authenticated_client.put(
        SETTINGS_URL, json={'default_mode': 'lokal'}).status_code == 200
    # Learn write AFTER the doc write must keep the doc namespace (the
    # learn.py merge fix) …
    assert authenticated_client.put(
        '/api/learn/settings', json={'daily_new_limit': 9}).status_code == 200
    assert authenticated_client.get(SETTINGS_URL).get_json() == {
        'default_mode': 'lokal'}
    # … and the doc write kept the learn keys.
    learn = authenticated_client.get('/api/learn/settings').get_json()
    assert learn['daily_new_limit'] == 9
    with app.app_context():
        blob = json.loads(db.session.get(User, test_user['id']).settings_json)
    assert blob['document_api'] == {'default_mode': 'lokal'}
    assert blob['daily_new_limit'] == 9


# --- P2: idempotency over the content hash ---------------------------------------

def test_dedup_same_file_same_mode(app, client, test_user, doc_token,
                                   mock_redis_queue, doc_convert_dir):
    pdf = _pdf_bytes()
    first = _post(client, data=pdf, headers=_auth())
    assert first.status_code == 202
    cid = first.get_json()['id']

    mock_redis_queue['fetch'].return_value.is_failed = False
    # Same bytes, same (default) mode — the filename does not matter.
    second = _post(client, data=pdf, filename='anders-benannt.pdf',
                   headers=_auth())
    assert second.status_code == 200
    body = second.get_json()
    assert body['deduped'] is True
    assert body['id'] == cid
    assert body['status'] == 'pending'
    # No second job, no second row.
    assert mock_redis_queue['queue'].enqueue.call_count == 1
    assert _conversion_count(app) == 1


def test_dedup_serves_stored_ready_result(app, client, test_user, doc_token,
                                          mock_redis_queue, doc_convert_dir):
    pdf = _pdf_bytes()
    cid = _post(client, data=pdf, headers=_auth()).get_json()['id']
    doc_lib.write_result_file(cid, build_result_payload(
        '# Gespeichert', provenance_unit='page', provenance=['deterministisch'],
        usage={'model_calls': 0, 'cost_eur': 0.0}))
    second = _post(client, data=pdf, headers=_auth())
    assert second.status_code == 200
    body = second.get_json()
    assert body['deduped'] is True
    assert body['status'] == 'ready'
    assert body['markdown'] == '# Gespeichert'
    assert mock_redis_queue['queue'].enqueue.call_count == 1


def test_no_dedup_across_engine_generations(app, client, test_user, doc_token,
                                            mock_redis_queue, doc_convert_dir):
    """DOC-LOCAL P3: same file + same mode, but the stored row comes from an
    OLDER engine generation (pre-generation rows carry no field → count as 1)
    → NO dedup, a fresh job runs. Live-hit that motivated this: the legacy
    lokal row (deterministisch×280) answered for the freshly deployed mineru
    engine, with no user path around it."""
    pdf = _pdf_bytes()
    first = _post(client, data=pdf, headers=_auth())
    assert first.status_code == 202
    cid = first.get_json()['id']
    # Age the row to a pre-DOC-LOCAL shape: no engine_generation at all.
    with app.app_context():
        row = db.session.get(Conversion, cid)
        metadata = json.loads(row.metadata_json)
        del metadata['engine_generation']
        row.metadata_json = json.dumps(metadata)
        db.session.commit()

    second = _post(client, data=pdf, headers=_auth())
    assert second.status_code == 202  # fresh job, not the stored answer
    assert second.get_json()['id'] != cid
    assert mock_redis_queue['queue'].enqueue.call_count == 2
    # The new row carries the current generation → a THIRD submit dedups
    # against it (the key still works within one generation).
    mock_redis_queue['fetch'].return_value.is_failed = False
    third = _post(client, data=pdf, headers=_auth())
    assert third.status_code == 200
    assert third.get_json()['deduped'] is True
    assert third.get_json()['id'] == second.get_json()['id']


def test_no_dedup_on_other_mode_or_failed(app, client, test_user, doc_token,
                                          mock_redis_queue, doc_convert_dir):
    pdf = _pdf_bytes()
    first = _post(client, data=pdf, headers=_auth(), mode='cloud')
    assert first.status_code == 202
    cid = first.get_json()['id']

    # Different mode → a different quality claim → new job.
    other_mode = _post(client, data=pdf, headers=_auth(), mode='lokal')
    assert other_mode.status_code == 202
    assert other_mode.get_json()['id'] != cid

    # Flip the first job to failed → re-submitting IS the retry path.
    job = mock_redis_queue['fetch'].return_value
    job.is_failed = True
    job.exc_info = 'RuntimeError: kaputt'
    assert client.get(f'{DOC_URL}/{cid}',
                      headers=_auth()).get_json()['status'] == 'failed'
    retry = _post(client, data=pdf, headers=_auth(), mode='cloud')
    assert retry.status_code == 202
    assert retry.get_json()['id'] != cid


# --- job states via reconcile ---------------------------------------------------

def _submit(client, app, data=None, filename='doc.pdf', mode=None):
    resp = _post(client, data=data if data is not None else _pdf_bytes(),
                 filename=filename, headers=_auth(), mode=mode)
    assert resp.status_code == 202
    return resp.get_json()['id']


def test_get_stays_pending_while_job_runs(app, client, test_user, doc_token,
                                          mock_redis_queue, doc_convert_dir):
    cid = _submit(client, app)
    mock_redis_queue['fetch'].return_value.is_failed = False
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'pending'
    assert body['markdown'] is None
    assert body['provenance'] is None
    assert body['error'] is None


def test_get_ready_reads_structured_result(app, client, test_user, doc_token,
                                           mock_redis_queue, doc_convert_dir):
    cid = _submit(client, app, filename='Bericht.pdf')
    doc_lib.write_result_file(cid, build_result_payload(
        '# Hallo\n\nWelt.',
        provenance_unit='page',
        provenance=['deterministisch'],
        degradations=[degradation('serializer', 'Tabelle degradiert')],
        usage={'model_calls': 0, 'cost_eur': 0.0}))
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'ready'
    assert body['markdown'] == '# Hallo\n\nWelt.'
    assert body['provenance_unit'] == 'page'
    assert body['provenance'] == ['deterministisch']
    assert body['degradations'] == [
        {'code': 'serializer', 'message': 'Tabelle degradiert', 'pages': None}]
    assert body['usage'] == {'model_calls': 0, 'cost_eur': 0.0}
    assert body['mode'] == 'lokal'  # mode-less submit → DOC-WEB default
    assert body['budget_eur'] == DOC_CONVERT_BUDGET_EUR
    assert body['error'] is None
    assert body['source']['filename'] == 'Bericht.pdf'
    assert body['source']['format'] == 'pdf'
    assert body['source']['page_count'] == 1
    assert body['source']['size_bytes'] > 0
    assert body['created_at']
    # The full contract surface, nothing implicit.
    assert set(body) == {'id', 'status', 'mode', 'markdown', 'provenance_unit',
                         'provenance', 'degradations', 'usage', 'budget_eur',
                         'error', 'source', 'created_at'}

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


# --- P2: worker task — mode routing, honest provenance, budget pre-flight --------

def _plant_source(cid, ext, data=b'x'):
    doc_lib.ensure_doc_convert_dir()
    path = doc_lib.doc_source_path(cid, ext)
    with open(path, 'wb') as f:
        f.write(data)
    return path


def test_task_cloud_routes_to_paged_backend(monkeypatch, doc_convert_dir):
    """Cloud + key + pre-flight ok → the page-wise gemini backend runs with
    the job's key and frozen budget; the legacy engine is never touched
    (DOC-ENGINE P2 — replaced the blackbox branch and its provenance
    round-up)."""
    monkeypatch.setenv('GEMINI_API_KEY', 'k-123')
    calls = {}

    def fake_run_cloud_pdf(source_path, api_key, budget_eur, model_name=None):
        calls['args'] = (api_key, budget_eur)
        return build_result_payload(
            '# Cloud', provenance_unit='page', provenance=['modell'] * 3,
            usage={'model_calls': 3, 'cost_eur': 0.044})

    monkeypatch.setattr('services.pdf_cloud.run_cloud_pdf', fake_run_cloud_pdf)
    _plant_source(901, 'pdf')
    convert_document_task(901, 'pdf', 'cloud', 5.0, 3)
    payload = doc_lib.read_result_file(901)
    assert calls['args'] == ('k-123', 5.0)
    assert payload['provenance_unit'] == 'page'
    assert payload['provenance'] == ['modell'] * 3
    assert payload['usage'] == {'model_calls': 3, 'cost_eur': 0.044}


@pytest.fixture
def fake_local_run(monkeypatch):
    """Replace the DOC-LOCAL engine run (services.pdf_local.run_local_pdf):
    mineru-style payload — per-page ``modell``, 0 €, no degradations. The
    container mechanics have their own suite (test_pdf_local); these tests
    prove the TASK routing."""
    calls = []

    def fake_run_local_pdf(source_path, page_count):
        calls.append((source_path, page_count))
        return build_result_payload(
            '# Lokal', provenance_unit='page',
            provenance=['modell'] * page_count,
            usage={'model_calls': 0, 'cost_eur': 0.0})

    monkeypatch.setattr('services.pdf_local.run_local_pdf',
                        fake_run_local_pdf)
    return calls


def test_task_budget_preflight_degrades_to_local(monkeypatch, doc_convert_dir,
                                                 fake_local_run):
    # 3 pages × 1.48 ct ≈ 0.044 € > cap 0.01 € → the key is NEVER used, the
    # run goes to the mineru engine (DOC-LOCAL: provenance ``modell``, 0 €)
    # and the degradation names the numbers.
    monkeypatch.setenv('GEMINI_API_KEY', 'k-123')
    _plant_source(902, 'pdf')
    convert_document_task(902, 'pdf', 'cloud', 0.01, 3)
    payload = doc_lib.read_result_file(902)
    assert fake_local_run == [(doc_lib.doc_source_path(902, 'pdf'), 3)]
    assert payload['provenance_unit'] == 'page'
    assert payload['provenance'] == ['modell'] * 3
    assert [d['code'] for d in payload['degradations']] == ['budget_exceeded']
    assert 'Kostendeckel 0.01 €' in payload['degradations'][0]['message']
    assert payload['usage'] == {'model_calls': 0, 'cost_eur': 0.0}


def test_task_cloud_without_key_degrades(monkeypatch, doc_convert_dir,
                                         fake_local_run):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    _plant_source(903, 'pdf')
    convert_document_task(903, 'pdf', 'cloud', 5.0, 2)
    payload = doc_lib.read_result_file(903)
    assert fake_local_run == [(doc_lib.doc_source_path(903, 'pdf'), 2)]
    assert payload['provenance'] == ['modell'] * 2
    assert [d['code'] for d in payload['degradations']] == ['cloud_unavailable']


def test_task_local_mode_never_touches_the_key(monkeypatch, doc_convert_dir,
                                               fake_local_run):
    """mode=lokal routes straight to the mineru engine — whose call signature
    carries no API key at all (key-free by construction), and the cloud
    backend is never built."""
    monkeypatch.setenv('GEMINI_API_KEY', 'k-123')
    _plant_source(904, 'pdf')
    convert_document_task(904, 'pdf', 'lokal', 5.0, 2)
    payload = doc_lib.read_result_file(904)
    assert fake_local_run == [(doc_lib.doc_source_path(904, 'pdf'), 2)]
    assert payload['provenance_unit'] == 'page'
    assert payload['provenance'] == ['modell'] * 2
    assert payload['degradations'] == []
    assert payload['usage'] == {'model_calls': 0, 'cost_eur': 0.0}


# --- P2: the budget-cap mechanic, proven against placeholder backends ------------

def _cloud_page(index):
    return {'markdown': f'C{index}', 'origin': 'modell', 'cost_eur': 0.5}


def _local_page(index):
    return {'markdown': f'L{index}', 'origin': 'deterministisch', 'cost_eur': 0.0}


def test_pipeline_budget_switch_flips_provenance():
    """The sprint's core proof: the cap hits mid-document and the provenance
    of every page after the switch demonstrably changes, with ONE degradation
    naming cap, spend and affected pages."""
    payload = run_paged_conversion(4, _cloud_page, _local_page, budget_eur=1.0)
    assert payload['provenance'] == [
        'modell', 'modell', 'deterministisch', 'deterministisch']
    assert payload['markdown'] == 'C0\n\nC1\n\nL2\n\nL3'
    assert len(payload['degradations']) == 1
    entry = payload['degradations'][0]
    assert entry['code'] == 'budget_exceeded'
    assert entry['pages'] == [3, 4]
    assert 'Kostendeckel 1.00 €' in entry['message']
    assert payload['usage'] == {'model_calls': 2, 'cost_eur': 1.0}


def test_pipeline_within_budget_never_switches():
    payload = run_paged_conversion(3, _cloud_page, _local_page, budget_eur=10.0)
    assert payload['provenance'] == ['modell'] * 3
    assert payload['degradations'] == []
    assert payload['usage'] == {'model_calls': 3, 'cost_eur': 1.5}


def test_pipeline_payload_flows_through_reconcile(app, client, test_user,
                                                  doc_token, mock_redis_queue,
                                                  doc_convert_dir):
    """The pipeline's output IS the worker result shape: written as
    result_<id>.json, the existing reconcile serves the mixed provenance
    unchanged — the follow-up engine sprint plugs in without touching the
    contract."""
    payload = run_paged_conversion(4, _cloud_page, _local_page, budget_eur=1.0)
    cid = _submit(client, app)
    doc_lib.write_result_file(cid, payload)
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'ready'
    assert body['provenance_unit'] == 'page'
    assert body['provenance'] == [
        'modell', 'modell', 'deterministisch', 'deterministisch']
    assert body['degradations'][0]['code'] == 'budget_exceeded'
    assert body['degradations'][0]['pages'] == [3, 4]
    assert body['usage'] == {'model_calls': 2, 'cost_eur': 1.0}
    assert body['markdown'] == 'C0\n\nC1\n\nL2\n\nL3'


# --- end-to-end: submit → run task → poll → result ------------------------------

def test_pdf_end_to_end_local_engine_failure_degrades(app, client, test_user,
                                                      doc_token,
                                                      mock_redis_queue,
                                                      doc_convert_dir,
                                                      monkeypatch, tmp_path):
    """Submit (mode lokal) → worker task with an UNAVAILABLE mineru engine
    (docker mocked to rc=1) → the REAL PyMuPDF text layer serves the pages,
    the switch is a named backend_fallback on a ready result — DOC-LOCAL's
    sprint-1.3 failure path, end to end through submit/task/reconcile."""
    from types import SimpleNamespace as NS

    import services.pdf_local as pdf_local_mod

    monkeypatch.setenv('DOC_LOCAL_EXCHANGE_DIR', str(tmp_path))
    monkeypatch.setattr(
        pdf_local_mod.subprocess, 'run',
        lambda *a, **k: NS(returncode=1, stdout='', stderr='kein docker'))

    cid = _submit(client, app, data=_pdf_bytes('Hallo Konvertierung.'),
                  filename='echt.pdf', mode='lokal')

    # Run the DB-free worker task in-process, exactly as RQ would call it.
    convert_document_task(cid, 'pdf', 'lokal', DOC_CONVERT_BUDGET_EUR, 1)

    # Structured result on the volume, source consumed by the task's finally.
    assert (doc_convert_dir / f'result_{cid}.json').exists()
    assert not (doc_convert_dir / f'source_{cid}.pdf').exists()

    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'ready'
    assert 'Hallo Konvertierung.' in body['markdown']  # real text layer
    assert body['mode'] == 'lokal'
    assert body['provenance_unit'] == 'page'
    assert body['provenance'] == ['deterministisch']
    assert [d['code'] for d in body['degradations']] == ['backend_fallback']
    assert 'kein docker' in body['degradations'][0]['message']
    assert body['usage'] == {'model_calls': 0, 'cost_eur': 0.0}
    assert body['source']['format'] == 'pdf'
    assert body['source']['page_count'] == 1


def test_eml_end_to_end_with_real_serializer(app, client, test_user, doc_token,
                                             mock_redis_queue, doc_convert_dir,
                                             monkeypatch):
    """Submit → worker task (stubbed partition, REAL serializer) → poll → ready.

    EML is the canonical unstructured-path format since DOC-ENGINE (DOCX/PPTX/
    HTML route to their measured winners). No page concept → document-level
    provenance, and a serializer warning becomes a structured degradation on a
    ready (not failed) result — partial success is a 200 with a list, never a
    500.
    """
    def fake_partition(filename=None, strategy=None, paragraph_grouper=None):
        def el(category, text='', depth=None, html=None):
            return SimpleNamespace(
                category=category, text=text,
                metadata=SimpleNamespace(category_depth=depth, page_number=1,
                                         text_as_html=html))
        return [
            el('Title', 'Kapitel Eins', depth=0),
            el('NarrativeText', 'Ein Absatz mit Inhalt.'),
            el('Table', 'Stoff Menge'),  # no text_as_html → degradation
        ]

    monkeypatch.setattr(sys.modules['unstructured.partition.auto'],
                        'partition', fake_partition)

    cid = _submit(client, app, data=b'From: a@b.de\n\nHallo',
                  filename='mail.eml')
    convert_document_task(cid, 'eml', 'cloud', DOC_CONVERT_BUDGET_EUR, None)

    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'ready'
    assert '# Kapitel Eins' in body['markdown']
    assert 'Ein Absatz mit Inhalt.' in body['markdown']
    assert body['provenance_unit'] == 'document'
    assert body['provenance'] == ['deterministisch']
    assert body['degradations'] == [{
        'code': 'serializer',
        'message': 'Tabelle ohne text_as_html — als Fliesstext ausgegeben',
        'pages': None}]
    assert body['usage'] == {'model_calls': 0, 'cost_eur': 0.0}
    assert body['source']['format'] == 'eml'
    assert body['source']['page_count'] is None


def test_task_failure_leaves_no_result_and_consumes_source(app, client,
                                                           test_user, doc_token,
                                                           mock_redis_queue,
                                                           doc_convert_dir,
                                                           monkeypatch):
    def boom(filename=None, strategy=None, paragraph_grouper=None):
        raise RuntimeError('partition explodiert')

    monkeypatch.setattr(sys.modules['unstructured.partition.auto'],
                        'partition', boom)
    cid = _submit(client, app, data=b'kaputt', filename='defekt.eml')

    with pytest.raises(RuntimeError):
        convert_document_task(cid, 'eml', 'cloud', DOC_CONVERT_BUDGET_EUR, None)

    # No result file → reconcile keys on the RQ job; source is consumed.
    assert not (doc_convert_dir / f'result_{cid}.json').exists()
    assert not (doc_convert_dir / f'source_{cid}.eml').exists()

    job = mock_redis_queue['fetch'].return_value
    job.is_failed = True
    job.exc_info = 'Traceback ...\nRuntimeError: partition explodiert'
    body = client.get(f'{DOC_URL}/{cid}', headers=_auth()).get_json()
    assert body['status'] == 'failed'
    assert body['error'].endswith('RuntimeError: partition explodiert')
