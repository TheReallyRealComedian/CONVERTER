"""Markdown→PDF characterization tests.

Locks in: a successful POST returns a PDF download (200 + application/pdf),
and a POST with neither pasted text nor an uploaded file flashes an error
and re-renders the form (302 → /).

Playwright is patched at the ``app.async_playwright`` import boundary so the
test does not require a running browser.
"""
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import app as app_module


class _FakeAsyncCM:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_playwright_mock():
    """Build a chain of AsyncMocks shaped like the playwright API surface used
    by ``convert_markdown``.  ``page.pdf(path=…)`` writes a stub PDF byte
    string so the route can read it back from disk.
    """
    page = MagicMock()
    page.set_content = AsyncMock()
    page.evaluate = AsyncMock()

    async def fake_pdf(path=None, **_kw):
        with open(path, 'wb') as fh:
            fh.write(b'%PDF-1.4\n% stub pdf bytes for tests\n%%EOF\n')

    page.pdf = AsyncMock(side_effect=fake_pdf)

    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)
    browser.close = AsyncMock()

    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)

    pw = MagicMock()
    pw.chromium = chromium
    return _FakeAsyncCM(pw)


def test_convert_markdown_returns_pdf(authenticated_client):
    with patch.object(app_module, 'async_playwright', return_value=_make_playwright_mock()):
        resp = authenticated_client.post(
            '/convert-markdown',
            data={
                'markdown_text': '# Hello\n\nA paragraph.',
                'output_filename': 'test_output',
                'orientation': 'portrait',
                'style_theme': 'none',
            },
            content_type='multipart/form-data',
        )
    assert resp.status_code == 200
    assert resp.mimetype == 'application/pdf'
    assert resp.data.startswith(b'%PDF')
    assert 'attachment' in resp.headers.get('Content-Disposition', '')
    assert 'test_output.pdf' in resp.headers.get('Content-Disposition', '')


def test_convert_markdown_no_input_redirects_with_flash(authenticated_client):
    resp = authenticated_client.post(
        '/convert-markdown',
        data={
            'markdown_text': '',
            'output_filename': 'whatever',
            'orientation': 'portrait',
            'style_theme': 'none',
        },
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    # The route flashes "No Markdown content provided" and redirects to /.
    assert resp.status_code == 302
    assert resp.headers['Location'].endswith('/')


def test_convert_markdown_invalid_filename_redirects(authenticated_client):
    with patch.object(app_module, 'async_playwright', return_value=_make_playwright_mock()):
        resp = authenticated_client.post(
            '/convert-markdown',
            data={
                'markdown_text': '# Some content',
                'output_filename': '',
                'orientation': 'portrait',
                'style_theme': 'none',
            },
            content_type='multipart/form-data',
            follow_redirects=False,
        )
    # secure_filename('') -> '' -> flash + redirect.
    assert resp.status_code == 302
