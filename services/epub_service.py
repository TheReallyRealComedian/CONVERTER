# services/epub_service.py
"""Build an EPUB from rendered HTML — in-memory, network-free.

Used by the Send-to-Kindle flow: a library ``Conversion`` (Markdown) is
rendered through the shared ``render_markdown_to_html`` and wrapped into a
single-chapter reflowable EPUB via ``ebooklib`` — so the Kindle output is the
same HTML the in-app reader shows. Pure function: no Flask, no SMTP, no network.
"""
import os
import re
import tempfile

from ebooklib import epub

from services.epub_math import latex_spans_to_mathml


def _slug(text: str) -> str:
    """ASCII slug for the EPUB identifier; falls back to ``document``."""
    slug = re.sub(r'[^a-z0-9]+', '-', (text or '').lower()).strip('-')
    return slug or 'document'


def build_epub(
    title: str,
    html_body: str,
    *,
    author: str = 'CONVERTER',
    language: str = 'de',
    identifier: str | None = None,
) -> bytes:
    """Build a single-chapter reflowable EPUB and return its bytes.

    ``html_body`` is sanitized HTML (from ``render_markdown_to_html``). An
    empty/None body still yields a valid (empty) EPUB rather than crashing.
    """
    safe_title = title or 'Dokument'

    book = epub.EpubBook()
    book.set_identifier(identifier or f'converter-{_slug(title)}')
    book.set_title(safe_title)
    book.set_language(language)
    book.add_author(author)

    chapter = epub.EpubHtml(title=safe_title, file_name='chapter.xhtml', lang=language)

    # Sprint KINDLE-MATH: server-side LaTeX→MathML so math survives on JS-less
    # e-readers (KaTeX runs only in the in-app reader/preview/PDF surfaces).
    # EPUB_MATH_MODE='mathml' (default) transforms the math spans; 'off' is the
    # kill-switch (today's passthrough), 'image' is the documented-but-unbuilt
    # escape-hatch (also passthrough until the device smoke says MathML is bad).
    math_mode = os.environ.get('EPUB_MATH_MODE', 'mathml')
    has_math = False
    if math_mode == 'mathml':
        html_body, has_math = latex_spans_to_mathml(html_body)
    if has_math:
        # OPF manifest item must carry properties="mathml" for EPUB3 MathML.
        # Attribute-append form only — EpubHtml(..., properties=[...]) raises
        # TypeError. math-free EPUBs leave properties untouched → byte-stable.
        chapter.properties.append('mathml')

    # render_markdown_to_html emits HTML5 (e.g. <br> not <br/>). ebooklib and
    # Amazon's converter are generally tolerant; if Kindle rendering misbehaves
    # in smoke, serialize html_body through an XHTML void-tag fixup here.
    # An empty body would make ebooklib's EPUB3-nav page scan feed lxml an empty
    # document (ParserError); a placeholder paragraph keeps the EPUB valid.
    inner = (html_body or '').strip() or '<p></p>'
    chapter.content = f'<html><body>{inner}</body></html>'
    book.add_item(chapter)

    book.toc = (chapter,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ['nav', chapter]

    # write_epub targets a path (zipfile under the hood); mirror the PDF flow's
    # temp-file pattern and return the bytes so nothing leaks on disk.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as tmp:
            tmp_path = tmp.name
        epub.write_epub(tmp_path, book, {'raise_exceptions': True})
        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
