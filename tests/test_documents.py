"""Document→Markdown characterization tests (web path, DOC-WEB P1).

Since DOC-WEB the ``/transform-document`` non-PDF branch runs through the
SHARED ``services.document_router.convert_non_pdf`` — the same backends the
API task uses (DOCX → pandoc, PPTX → markitdown, HTML/HTM → trafilatura,
EML/TXT/MD → unstructured + serializer). The tests mock at the same boundary
as ``tests/test_office_backends.py``: backend functions on
``services.office_backends``, ``partition`` on the ``sys.modules`` stub
(see ``conftest.py``) — so the REAL router routing runs in every test.

PDF still goes through ``app.pdf_extraction_service`` (falls in P2).
"""
import sys
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import app as app_module

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
    assert resp.mimetype == 'text/markdown'
    assert b'# Via pandoc' in resp.data
    assert b'Absatztext.' in resp.data
    assert 'sample.md' in resp.headers.get('Content-Disposition', '')
    backend.assert_called_once()


def test_transform_document_pptx_routes_to_markitdown(authenticated_client):
    with patch('services.office_backends.convert_pptx_markitdown',
               return_value=('# Folie 1\n\nSprechernotiz', [])) as backend:
        resp = _post_document(authenticated_client, 'deck.pptx')
    assert resp.status_code == 200
    assert b'Sprechernotiz' in resp.data
    assert 'deck.md' in resp.headers.get('Content-Disposition', '')
    backend.assert_called_once()


def test_transform_document_html_routes_to_trafilatura(authenticated_client):
    with patch('services.office_backends.convert_html_trafilatura',
               return_value=('# Artikel\n\nHaupttext.', [])) as backend:
        resp = _post_document(authenticated_client, 'seite.html')
    assert resp.status_code == 200
    assert b'# Artikel' in resp.data
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
    assert b'Nur ein Absatz.' in resp.data
    assert b'# Nur ein Absatz.' not in resp.data


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
    assert b'| Stoff | Menge |\n| --- | --- |\n| Ethanol | 12,5 |' in resp.data


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
    assert b'Roher Seitentext.' in resp.data


def test_transform_document_pdf_uses_pdf_extraction_service(authenticated_client, fixtures_dir):
    """PDF uploads bypass the router and go through ``pdf_extraction_service``."""
    mock_pdf_svc = MagicMock()
    mock_pdf_svc.extract_markdown.return_value = '# Extracted PDF\n\nbody text'
    original = app_module.pdf_extraction_service
    app_module.pdf_extraction_service = mock_pdf_svc
    try:
        with open(fixtures_dir / 'sample.pdf', 'rb') as fh:
            resp = _post_document(authenticated_client, 'sample.pdf',
                                  payload=fh.read())
    finally:
        app_module.pdf_extraction_service = original
    assert resp.status_code == 200
    assert resp.mimetype == 'text/markdown'
    assert b'Extracted PDF' in resp.data
    mock_pdf_svc.extract_markdown.assert_called_once()


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
