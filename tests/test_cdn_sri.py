"""SEC-AUDIT P2 — every external <script> on the app's pages is SRI-pinned.

Before: ``markdown-it@14.1.0`` without SRI on the markdown converter,
``mermaid@10`` FLOATING without SRI on the mermaid converter (the reader
loads ``10.9.8`` with SRI since RICH-MEDIA). A compromised or re-published
CDN file would have run in every session of the page.

One named exception: the Tailwind Play CDN (``cdn.tailwindcss.com``) is a
JIT compiler that injects <style> at runtime — it cannot carry a stable
hash. Replacing it with build-time CSS is the prerequisite of CSP-BASELINE;
until then it is listed here, not silently skipped.
"""
import re
from pathlib import Path

import pytest

from models import Conversion, db

REPO = Path(__file__).resolve().parent.parent
PLAY_CDN = 'https://cdn.tailwindcss.com'  # CSP-BASELINE prerequisite, see docstring

PAGES = ['/', '/document-converter', '/audio-converter', '/mermaid-converter',
         '/library', '/review', '/tags']

_SCRIPT_TAG = re.compile(r'<script\b([^>]*)>', re.IGNORECASE)
_ATTR = re.compile(r'([a-zA-Z-]+)="([^"]*)"')


def _external_scripts(html):
    for match in _SCRIPT_TAG.finditer(html):
        attrs = dict(_ATTR.findall(match.group(1)))
        src = attrs.get('src', '')
        if src.startswith(('https://', 'http://', '//')):
            yield src, attrs


def _rendered_pages(app, test_user, authenticated_client):
    with app.app_context():
        conv = Conversion(user_id=test_user['id'], conversion_type='markdown_input',
                          title='doc', content='# body')
        db.session.add(conv)
        db.session.commit()
        detail = f'/library/{conv.id}'
    for page in PAGES + [detail]:
        resp = authenticated_client.get(page)
        assert resp.status_code == 200, page
        yield page, resp.get_data(as_text=True)


def test_every_external_script_is_pinned(app, test_user, authenticated_client, client):
    pages = list(_rendered_pages(app, test_user, authenticated_client))
    authenticated_client.get('/logout')
    pages.append(('/login', client.get('/login').get_data(as_text=True)))
    seen = set()
    for page, html in pages:
        for src, attrs in _external_scripts(html):
            seen.add(src)
            if src == PLAY_CDN:
                continue
            assert attrs.get('integrity', '').startswith('sha384-'), f'{page}: {src}'
            assert attrs.get('crossorigin') == 'anonymous', f'{page}: {src}'
            # An exact version, never a floating major (@10/) or @latest.
            assert re.search(r'@\d+\.\d+\.\d+/', src), f'{page}: unpinned version {src}'
    # The sentinel really looked at the two CDN scripts it is about.
    assert any('markdown-it@' in s for s in seen)
    assert any('mermaid@' in s for s in seen)


def test_mermaid_page_uses_the_readers_pin(authenticated_client):
    reader_js = (REPO / 'static' / 'js' / 'reader_figures.js').read_text()
    reader_src = re.search(r"MERMAID_SRC = '([^']+)'", reader_js).group(1)
    reader_sri = re.search(r"MERMAID_SRI = '([^']+)'", reader_js).group(1)
    html = authenticated_client.get('/mermaid-converter').get_data(as_text=True)
    [(src, attrs)] = [(s, a) for s, a in _external_scripts(html) if 'mermaid@' in s]
    assert src == reader_src
    assert attrs['integrity'] == reader_sri
