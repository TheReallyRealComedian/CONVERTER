"""Markdown→PDF characterization tests.

Locks in: a successful POST returns a PDF download (200 + application/pdf),
and a POST with neither pasted text nor an uploaded file flashes an error
and re-renders the form (302 → /).

Playwright is patched at the ``app.async_playwright`` import boundary so the
test does not require a running browser.
"""
from io import BytesIO
from unittest.mock import ANY, AsyncMock, MagicMock, patch

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
    # SEC-SSRF: convert_markdown now installs a default-deny route handler
    # (``await page.route(...)``) before set_content. Without an AsyncMock the
    # ``await`` would fail on the auto-created MagicMock attribute.
    page.route = AsyncMock()

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


def test_convert_markdown_missing_filename_field_handled(authenticated_client):
    """F-007: a POST without an ``output_filename`` form field used to crash
    inside the broad ``except`` (``secure_filename(None)`` raised
    AttributeError, which surfaced as a generic "Could not generate PDF"
    flash). Now the missing field is treated like an empty one, so the
    invalid-filename branch handles it cleanly with a 302 redirect.
    """
    resp = authenticated_client.post(
        '/convert-markdown',
        data={
            'markdown_text': '# Some content',
            # 'output_filename' intentionally omitted
            'orientation': 'portrait',
            'style_theme': 'none',
        },
        content_type='multipart/form-data',
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert resp.headers['Location'].endswith('/')


def test_convert_markdown_pdf_gen_error_redirects_with_flash(authenticated_client):
    """P8: when Playwright (or anything else inside the PDF-gen ``try``) raises,
    the route must flash a German error and ``redirect`` back to the form. The
    pre-P8 code re-rendered the template directly, which then crashed on the
    missing ``themes`` / ``accepted_extensions`` context.
    """
    def _raising_playwright():
        raise RuntimeError('simulated browser launch failure')

    with patch.object(app_module, 'async_playwright', side_effect=_raising_playwright):
        resp = authenticated_client.post(
            '/convert-markdown',
            data={
                'markdown_text': '# Some content',
                'output_filename': 'test_output',
                'orientation': 'portrait',
                'style_theme': 'none',
            },
            content_type='multipart/form-data',
            follow_redirects=False,
        )
    assert resp.status_code == 302
    assert resp.headers['Location'].endswith('/')

    follow = authenticated_client.get('/')
    assert follow.status_code == 200
    assert 'PDF-Erstellung fehlgeschlagen' in follow.get_data(as_text=True)


def test_convert_markdown_unsupported_extension_returns_400(authenticated_client):
    """F-006: file uploads with extensions outside ACCEPTED_EXTENSIONS must be
    rejected with 400 + DE-JSON. Previously, .read().decode('utf-8') on a
    binary upload raised UnicodeDecodeError that the broad except masked as
    'Could not generate PDF' — wrong error class, wrong status code."""
    fake_binary = BytesIO(b'\x89PNG\r\n\x1a\n\x00\x00')
    resp = authenticated_client.post(
        '/convert-markdown',
        data={
            'markdown_text': '',
            'output_filename': 'whatever',
            'orientation': 'portrait',
            'style_theme': 'none',
            'markdown_file': (fake_binary, 'image.png'),
        },
        content_type='multipart/form-data',
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert 'nicht unterstützt' in body['error']
    assert '.md' in body['error']


def test_convert_markdown_routes_before_set_content_and_launches_with_belt(
        authenticated_client):
    """SEC-SSRF: the render path must (1) launch Chromium with the dead-proxy
    belt args and (2) install the default-deny route handler BEFORE
    set_content — a route installed after would leave the first content load
    unguarded. Both are asserted from the recorded call order on a shared
    manager mock.
    """
    from unittest.mock import call
    from app_pkg.pdf_egress import PDF_BROWSER_ARGS

    manager = MagicMock()

    page = MagicMock()

    async def fake_pdf(path=None, **_kw):
        with open(path, 'wb') as fh:
            fh.write(b'%PDF-1.4\n%%EOF\n')

    page.set_content = AsyncMock()
    page.evaluate = AsyncMock()
    page.route = AsyncMock()
    page.pdf = AsyncMock(side_effect=fake_pdf)

    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=page)
    browser.close = AsyncMock()
    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)

    # Record ordering of route vs set_content on one manager.
    manager.attach_mock(page.route, 'route')
    manager.attach_mock(page.set_content, 'set_content')

    pw = MagicMock()
    pw.chromium = chromium

    with patch.object(app_module, 'async_playwright', return_value=_FakeAsyncCM(pw)):
        resp = authenticated_client.post(
            '/convert-markdown',
            data={
                'markdown_text': '# Hi',
                'output_filename': 'out',
                'orientation': 'portrait',
                'style_theme': 'none',
            },
            content_type='multipart/form-data',
        )
    assert resp.status_code == 200

    # (1) belt args on launch
    chromium.launch.assert_awaited_once_with(args=PDF_BROWSER_ARGS)

    # (2) page.route(...) happened before page.set_content(...)
    names = [c[0] for c in manager.mock_calls]
    assert 'route' in names and 'set_content' in names
    assert names.index('route') < names.index('set_content')
    # the handler is installed with a glob catch-all
    assert manager.mock_calls[names.index('route')] == call.route('**/*', ANY)
