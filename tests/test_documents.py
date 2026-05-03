"""Document→Markdown characterization tests.

Locks in: a successful POST with a docx file returns a markdown download,
and a POST without a file part returns 400.

``unstructured.partition.auto.partition`` is stubbed at the sys.modules
level (see ``conftest.py``); for the docx test we additionally patch
``app.partition`` so the route receives a deterministic element list.
"""
from io import BytesIO
from unittest.mock import patch, MagicMock

import app as app_module


def test_transform_document_docx_returns_markdown(authenticated_client, fixtures_dir):
    # ``unstructured.partition.auto.partition`` returns a list of "elements"
    # whose ``.text`` attribute the route joins with double-newlines.
    fake_elements = [MagicMock(text='Hello docx fixture'), MagicMock(text='Second paragraph')]
    with patch.object(app_module, 'partition', return_value=fake_elements):
        with open(fixtures_dir / 'sample.docx', 'rb') as fh:
            resp = authenticated_client.post(
                '/transform-document',
                data={'document_file': (BytesIO(fh.read()), 'sample.docx')},
                content_type='multipart/form-data',
            )
    assert resp.status_code == 200
    assert resp.mimetype == 'text/markdown'
    assert b'Hello docx fixture' in resp.data
    assert b'Second paragraph' in resp.data
    assert 'sample.md' in resp.headers.get('Content-Disposition', '')


def test_transform_document_pdf_uses_pdf_extraction_service(authenticated_client, fixtures_dir):
    """PDF uploads bypass ``partition`` and go through ``pdf_extraction_service``."""
    mock_pdf_svc = MagicMock()
    mock_pdf_svc.extract_markdown.return_value = '# Extracted PDF\n\nbody text'
    original = app_module.pdf_extraction_service
    app_module.pdf_extraction_service = mock_pdf_svc
    try:
        with open(fixtures_dir / 'sample.pdf', 'rb') as fh:
            resp = authenticated_client.post(
                '/transform-document',
                data={'document_file': (BytesIO(fh.read()), 'sample.pdf')},
                content_type='multipart/form-data',
            )
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
    they reach the unstructured/PDF pipelines (Cluster D / Pattern 6 backstop)."""
    resp = authenticated_client.post(
        '/transform-document',
        data={'document_file': (BytesIO(b'irrelevant bytes'), 'evil.xyz')},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert 'nicht unterstützt' in body['error']
    assert 'PDF, DOCX, PPTX, EML, HTML, TXT, MD' in body['error']
