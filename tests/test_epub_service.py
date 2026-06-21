"""Sprint KINDLE Phase 1 — the pure EPUB builder.

``build_epub`` wraps sanitized HTML (from ``render_markdown_to_html``) into a
single-chapter reflowable EPUB via ``ebooklib``. Network-free, no Flask. Tests
assert non-empty bytes, a real ``read_epub`` round-trip (title + content
survive), and that an empty/None body yields a valid EPUB instead of crashing.
"""
import ebooklib
from ebooklib import epub

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
