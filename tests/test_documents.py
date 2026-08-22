"""Document→Markdown characterization tests (web path, DOC-WEB P1).

Since DOC-WEB the ``/transform-document`` non-PDF branch runs through the
SHARED ``services.document_router.convert_non_pdf`` — the same backends the
API task uses (DOCX → pandoc, PPTX → markitdown, HTML/HTM → trafilatura,
EML/TXT/MD → unstructured + serializer). The tests mock at the same boundary
as ``tests/test_office_backends.py``: backend functions on
``services.office_backends``, ``partition`` on the ``sys.modules`` stub
(see ``conftest.py``) — so the REAL router routing runs in every test.

PDF (DOC-WEB-ASYNC) is NOT converted on this route anymore: the browser
submits PDFs to ``POST /api/document-conversions`` and polls (covered in
``tests/test_document_api.py``, session path included); a PDF posted here
is answered with a pointer to the service before any engine is touched —
asserted against the same engine seams (``services.pdf_cloud.run_cloud_pdf``
/ ``services.pdf_local.run_local_pdf``).
"""
import sys
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from tests.test_unstructured_markdown import El


def _post_document(client, filename, payload=b'irrelevant bytes'):
    return client.post(
        '/transform-document',
        data={'document_file': (BytesIO(payload), filename)},
        content_type='multipart/form-data',
    )


def test_transform_document_docx_routes_to_pandoc(authenticated_client):
    """DOCX in the browser hits the SAME backend as the API: pandoc — not
    the legacy unstructured pipeline (DOC-WEB: one quality, not two)."""
    with patch('services.office_backends.convert_docx_pandoc',
               return_value=('# Via pandoc\n\nAbsatztext.', [])) as backend:
        resp = _post_document(authenticated_client, 'sample.docx')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['markdown'] == '# Via pandoc\n\nAbsatztext.'
    assert body['filename'] == 'sample.md'
    assert body['degradations'] == []
    backend.assert_called_once()


def test_transform_document_pptx_routes_to_markitdown(authenticated_client):
    with patch('services.office_backends.convert_pptx_markitdown',
               return_value=('# Folie 1\n\nSprechernotiz', [])) as backend:
        resp = _post_document(authenticated_client, 'deck.pptx')
    assert resp.status_code == 200
    assert 'Sprechernotiz' in resp.get_json()['markdown']
    assert resp.get_json()['filename'] == 'deck.md'
    backend.assert_called_once()


def test_transform_document_html_routes_to_trafilatura(authenticated_client):
    with patch('services.office_backends.convert_html_trafilatura',
               return_value=('# Artikel\n\nHaupttext.', [])) as backend:
        resp = _post_document(authenticated_client, 'seite.html')
    assert resp.status_code == 200
    assert resp.get_json()['markdown'].startswith('# Artikel')
    backend.assert_called_once()


def test_transform_document_txt_stays_on_serializer_with_source_ext(
        authenticated_client):
    """TXT keeps the unstructured path, and ``source_ext`` travels: TXT makes
    every paragraph a ``Title`` with depth ``None`` — without the ext rule the
    whole file would render as ``#`` headings."""
    fake_elements = [El('Title', 'Nur ein Absatz.'),
                     El('Title', 'Noch ein Absatz.')]
    with patch.object(sys.modules['unstructured.partition.auto'], 'partition',
                      return_value=fake_elements):
        resp = _post_document(authenticated_client, 'notiz.txt')
    assert resp.status_code == 200
    md = resp.get_json()['markdown']
    assert 'Nur ein Absatz.' in md
    assert '# Nur ein Absatz.' not in md


def test_transform_document_eml_renders_tables_as_pipes(authenticated_client):
    """The serializer is actually wired into the web route — a Table with
    ``text_as_html`` reaches the download as a GFM pipe table."""
    fake_elements = [El(
        'Table', 'Stoff Menge Ethanol 12,5',
        text_as_html='<table><tr><td>Stoff</td><td>Menge</td></tr>'
                     '<tr><td>Ethanol</td><td>12,5</td></tr></table>',
    )]
    with patch.object(sys.modules['unstructured.partition.auto'], 'partition',
                      return_value=fake_elements):
        resp = _post_document(authenticated_client, 'mail.eml')
    assert resp.status_code == 200
    assert '| Stoff | Menge |\n| --- | --- |\n| Ethanol | 12,5 |' in resp.get_json()['markdown']


def test_transform_document_html_fallback_still_serves_markdown(
        authenticated_client):
    """trafilatura finds no main content → the router's named fallback path
    serves the unstructured result through the web route (no 500)."""
    fake_elements = [El('NarrativeText', 'Roher Seitentext.')]
    with patch('services.office_backends.convert_html_trafilatura',
               return_value=(None, [])):
        with patch.object(sys.modules['unstructured.partition.auto'],
                          'partition', return_value=fake_elements):
            resp = _post_document(authenticated_client, 'leer.html')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['markdown'] == 'Roher Seitentext.'
    # The degradation reaches the USER (DOC-WEB P3), not only the log.
    assert [d['code'] for d in body['degradations']] == ['backend_fallback']


def test_transform_document_pdf_is_pointed_to_the_service(
        authenticated_client, fixtures_dir, monkeypatch):
    """DOC-WEB-ASYNC: the synchronous route runs NO PDF engine. The web
    container holds no Docker socket anymore (``lokal`` would silently serve
    the bare text layer), and at the time a synchronous conversion stalled
    the then-single request thread for every other request (measured 78 s;
    SYNC-FREEZE has since put sync views on a thread pool — the engine
    argument stands on its own). A PDF here is a caller on the wrong route:
    named 400 with the service, engines untouched."""
    cloud = MagicMock()
    monkeypatch.setattr('services.pdf_cloud.run_cloud_pdf', cloud)
    local = MagicMock()
    monkeypatch.setattr('services.pdf_local.run_local_pdf', local)
    with open(fixtures_dir / 'sample.pdf', 'rb') as fh:
        resp = _post_document(authenticated_client, 'sample.pdf', payload=fh.read())
    assert resp.status_code == 400
    body = resp.get_json()
    assert '/api/document-conversions' in body['error']
    cloud.assert_not_called()
    local.assert_not_called()


def test_converter_page_still_accepts_pdf_for_the_service_path(
        authenticated_client):
    """The page keeps ``pdf`` in its accept list — the JS routes it to the
    service — and hands the JS the service URL it submits to."""
    resp = authenticated_client.get('/document-converter')
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert '.pdf' in html
    assert "documentConversionsUrl: '/api/document-conversions'" in html


def test_transform_document_missing_file_returns_400(authenticated_client):
    resp = authenticated_client.post(
        '/transform-document',
        data={},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 400
    assert resp.get_json()['error'].lower().startswith('no file')


def test_transform_document_unsupported_extension_returns_400(authenticated_client):
    """Files with an extension outside ACCEPTED_EXTENSIONS get rejected before
    they reach the router (Cluster D / Pattern 6 backstop)."""
    resp = _post_document(authenticated_client, 'evil.xyz')
    assert resp.status_code == 400
    body = resp.get_json()
    assert 'nicht unterstützt' in body['error']
    assert 'PDF, DOCX, PPTX, EML, HTML, TXT, MD' in body['error']
