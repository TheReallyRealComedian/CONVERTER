"""Sprint KINDLE Phase 1 — the pure EPUB builder.

``build_epub`` wraps sanitized HTML (from ``render_markdown_to_html``) into a
single-chapter reflowable EPUB via ``ebooklib``. Network-free, no Flask. Tests
assert non-empty bytes, a real ``read_epub`` round-trip (title + content
survive), and that an empty/None body yields a valid EPUB instead of crashing.
"""
import zipfile

import ebooklib
from ebooklib import epub

from app_pkg.markdown_render import render_markdown_to_html
from services.epub_service import build_epub


def _read_back(epub_bytes, tmp_path):
    """Round-trip the bytes through ebooklib's reader via a temp file."""
    path = tmp_path / 'book.epub'
    path.write_bytes(epub_bytes)
    return epub.read_epub(str(path))


def _doc_text(book):
    """Concatenate the text of all XHTML document items."""
    items = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    return b''.join(it.get_content() for it in items).decode('utf-8', 'replace')


def test_build_epub_returns_nonempty_zip_bytes():
    data = build_epub('Titel', '<p>Inhalt</p>')
    assert isinstance(data, bytes)
    assert len(data) > 0
    # EPUB is a ZIP container — magic bytes prove a real archive came back.
    assert data[:2] == b'PK'


def test_build_epub_roundtrip_preserves_title_and_content(tmp_path):
    data = build_epub('Mein Dokument', '<h1>Überschrift</h1><p>Suchbegriff-äöü</p>')
    book = _read_back(data, tmp_path)

    titles = [value for value, _attrs in book.get_metadata('DC', 'title')]
    assert 'Mein Dokument' in titles

    text = _doc_text(book)
    assert 'Suchbegriff-äöü' in text
    assert 'Überschrift' in text


def test_build_epub_empty_body_does_not_crash(tmp_path):
    data = build_epub('Leer', '')
    assert data[:2] == b'PK'
    book = _read_back(data, tmp_path)
    titles = [value for value, _attrs in book.get_metadata('DC', 'title')]
    assert 'Leer' in titles


def test_build_epub_none_body_does_not_crash():
    data = build_epub('Leer', None)
    assert isinstance(data, bytes) and data[:2] == b'PK'


def test_build_epub_blank_title_falls_back():
    # Empty title must still produce a valid book with a non-empty identifier.
    data = build_epub('', '<p>x</p>')
    assert data[:2] == b'PK'


# --- Sprint KINDLE-MATH: server-side LaTeX→MathML in the EPUB build ---------

def _opf_text(epub_bytes, tmp_path):
    """Return the raw OPF (manifest) XML from the EPUB zip."""
    path = tmp_path / 'opf.epub'
    path.write_bytes(epub_bytes)
    with zipfile.ZipFile(path) as z:
        opf_name = next(n for n in z.namelist() if n.endswith('.opf'))
        return z.read(opf_name).decode('utf-8', 'replace')


def test_build_epub_renders_math_and_sets_mathml_property(tmp_path):
    # End-to-end: real render_markdown_to_html math spans → EPUB3 MathML that
    # survives ebooklib's re-parse/re-serialize (well-formed XML), plus the OPF
    # manifest item carrying properties="mathml". The read-back is the real
    # proof — not a string-contains on the pre-build HTML.
    html = render_markdown_to_html('Vorwort.\n\n$$\\frac{1}{2}$$\n\n$a+b$ inline.')
    data = build_epub('Mathe', html)

    book = _read_back(data, tmp_path)
    content = _doc_text(book)
    assert '<math' in content
    assert 'display="block"' in content   # the $$…$$ display branch
    assert 'display="inline"' in content  # the $…$ inline branch

    assert 'properties="mathml"' in _opf_text(data, tmp_path)


def test_build_epub_math_free_body_has_no_mathml_property(tmp_path):
    # Byte-stability regression guard: math-free EPUBs must not gain the OPF
    # mathml property vs. today's behavior.
    html = render_markdown_to_html('# Titel\n\nGanz normaler Text ohne Mathe.')
    data = build_epub('Prosa', html)

    book = _read_back(data, tmp_path)
    assert '<math' not in _doc_text(book)
    assert 'properties="mathml"' not in _opf_text(data, tmp_path)


def test_build_epub_math_mode_off_is_passthrough(tmp_path, monkeypatch):
    # EPUB_MATH_MODE=off is the kill-switch: today's behavior, raw LaTeX spans
    # stay, no MathML, no mathml property.
    monkeypatch.setenv('EPUB_MATH_MODE', 'off')
    html = render_markdown_to_html('$$\\frac{1}{2}$$')
    data = build_epub('Aus', html)

    book = _read_back(data, tmp_path)
    content = _doc_text(book)
    assert '<math' not in content
    assert 'math-display' in content  # raw span survives untouched
    assert 'properties="mathml"' not in _opf_text(data, tmp_path)
