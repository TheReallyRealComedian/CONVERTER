"""Document→Markdown characterization tests.

Locks in: a successful POST with a docx file returns a markdown download,
and a POST without a file part returns 400.

``unstructured.partition.auto.partition`` is stubbed at the sys.modules
level (see ``conftest.py``); for the docx test we additionally patch
``app.partition`` so the route receives a deterministic element list.

Since DOC-FIX P2 the non-PDF branch runs the element list through
``services.unstructured_markdown.elements_to_markdown`` instead of
``"\\n\\n".join(el.text ...)``, so the stand-ins here carry the fields that
serializer reads (see ``tests/test_unstructured_markdown.py`` for its own,
much denser coverage).
"""
from io import BytesIO
from unittest.mock import patch, MagicMock

import app as app_module

from tests.test_unstructured_markdown import El


def test_transform_document_docx_returns_markdown(authenticated_client, fixtures_dir):
    fake_elements = [El('Title', 'Hello docx fixture', category_depth=0),
                     El('NarrativeText', 'Second paragraph')]
    with patch.object(app_module, 'partition', return_value=fake_elements):
        with open(fixtures_dir / 'sample.docx', 'rb') as fh:
            resp = authenticated_client.post(
                '/transform-document',
                data={'document_file': (BytesIO(fh.read()), 'sample.docx')},
                content_type='multipart/form-data',
            )
    assert resp.status_code == 200
    assert resp.mimetype == 'text/markdown'
    assert b'# Hello docx fixture' in resp.data
    assert b'Second paragraph' in resp.data
    assert 'sample.md' in resp.headers.get('Content-Disposition', '')


def test_transform_document_docx_renders_tables_as_pipes(authenticated_client, fixtures_dir):
    """The serializer is actually wired into the route — a Table with
    ``text_as_html`` reaches the download as a GFM pipe table, not as the
    flattened ``el.text`` the old ``join`` produced."""
    fake_elements = [El(
        'Table', 'Stoff Menge Ethanol 12,5',
        text_as_html='<table><tr><td>Stoff</td><td>Menge</td></tr>'
                     '<tr><td>Ethanol</td><td>12,5</td></tr></table>',
    )]
    with patch.object(app_module, 'partition', return_value=fake_elements):
        with open(fixtures_dir / 'sample.docx', 'rb') as fh:
            resp = authenticated_client.post(
                '/transform-document',
                data={'document_file': (BytesIO(fh.read()), 'sample.docx')},
                content_type='multipart/form-data',
            )
    assert resp.status_code == 200
    assert b'| Stoff | Menge |\n| --- | --- |\n| Ethanol | 12,5 |' in resp.data


def test_transform_document_passes_source_ext_to_serializer(authenticated_client, fixtures_dir):
    """A ``.pptx`` upload must reach the serializer as ``source_ext='pptx'`` —
    otherwise every short slide-body line would become an ``##`` heading."""
    fake_elements = [El('Title', 'Folientitel', category_depth=0),
                     El('Title', 'Kurze Body-Zeile', category_depth=1)]
    with patch.object(app_module, 'partition', return_value=fake_elements):
        with open(fixtures_dir / 'sample.docx', 'rb') as fh:
            resp = authenticated_client.post(
                '/transform-document',
                data={'document_file': (BytesIO(fh.read()), 'deck.pptx')},
                content_type='multipart/form-data',
            )
    assert resp.status_code == 200
    assert b'# Folientitel' in resp.data
    assert b'## Kurze Body-Zeile' not in resp.data


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
