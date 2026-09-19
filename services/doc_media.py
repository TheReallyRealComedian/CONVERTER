# services/doc_media.py
"""Media inside Markdown documents — finding it, budgeting it, hiding it from
previews (Sprint RICH-MEDIA).

Since RICH-MEDIA a document may carry figures: inline ``<svg>``, ``data:``
image URIs, Mermaid fences. Three places need to know where that media sits
in a TEXT without rendering it:

* ``app_pkg/markdown_render.py`` cuts ``<svg>`` islands out of the rendered
  HTML (``find_svg_spans``) and sends each through the one SVG policy
  (``services/svg_sanitize.py``).
* the content write paths budget it (``check_media_limits``) — 2 MB per data
  URI, 10 MB of media per document, answered with a 413 before anything is
  written.
* the library list shows ``content_preview`` from a media-free text
  (``strip_media_for_preview``), so a document that opens with a figure does
  not preview as source code.

Pure ``str -> …`` module — no Flask, no SDK, no nh3 (Vorbild
``services/markdown_sections.py``). ⚠️ Nothing here is a security boundary:
the scanners are deliberately simple and may mis-cut hostile markup. That is
safe by construction, because every piece the renderer cuts goes through an
``nh3`` pass of its own — a wrong cut changes what is *shown*, never what is
*allowed*. The limits guard against a runaway agent, not an adversary (a
holder of the write token can already store plain text of any length).
"""
import re

MAX_DATA_URI_BYTES = 2 * 1024 * 1024
MAX_DOCUMENT_MEDIA_BYTES = 10 * 1024 * 1024

# German, ≤ 2 Sätze (Microcopy-Regel) — the write paths return these verbatim.
DATA_URI_TOO_LARGE = ('Ein eingebettetes Bild ist zu groß. '
                      'Maximal 2 MB je data-URI.')
DOCUMENT_MEDIA_TOO_LARGE = ('Das Dokument enthält zu viele eingebettete Medien. '
                            'Maximal 10 MB je Dokument.')

# An ``<svg`` open edge or a ``</svg>`` close edge. The lookahead keeps
# ``<svgfoo`` out; case-insensitive because the HTML parser is.
SVG_EDGE_RE = re.compile(r'<svg(?=[\s>/])|</svg\s*>', re.IGNORECASE)

# What the span scanner stops at: a comment, an SVG edge, or the start of any
# other tag (which it then skips WHOLE, attribute values included).
_MARKUP_RE = re.compile(r'<!--|<svg(?=[\s>/])|</svg\s*>|<(?=[a-zA-Z/])',
                        re.IGNORECASE)
# From a tag's ``<`` to its closing ``>``, quote-aware: ``title="a>b"`` does
# not end the tag. Unrolled loop, each alternative starts on a distinct
# character → linear, also on a 2-MB ``src="data:…"``.
_TAG_END_RE = re.compile(
    r'<[^"\'>]*(?:"[^"]*"[^"\'>]*|\'[^\']*\'[^"\'>]*)*>')


def find_svg_spans(text: str):
    """Locate ``<svg>`` markup in ``text``.

    Returns ``(islands, dangling)`` — two lists of ``(start, end)`` slices:

    * ``islands``: the OUTERMOST balanced ``<svg …>…</svg>`` runs (a nested
      ``<svg>`` belongs to its parent), plus self-closed ``<svg …/>`` roots.
    * ``dangling``: the open TAGS (tag only, not what follows) of ``<svg``
      that never meet a ``</svg>`` — the extent of such a figure is unknowable.

    Tag-aware: an ``<svg`` inside another tag's attribute value is not an edge
    — ``<img alt="<svg>…</svg>">`` and the common hand-written
    ``<img src="data:image/svg+xml;utf8,<svg …>">`` stay whole. Comments are
    skipped. All returned spans are pairwise disjoint.
    """
    stack = []  # (start, open_tag_end) of opens still waiting for a close
    pairs = []  # every balanced pair, any depth
    pos = 0
    while True:
        hit = _MARKUP_RE.search(text, pos)
        if hit is None:
            break
        token = hit.group(0)
        if token == '<!--':
            close = text.find('-->', hit.end())
            if close < 0:
                break
            pos = close + 3
            continue
        if token[:2] == '</':
            if stack:
                start, _ = stack.pop()
                pairs.append((start, hit.end()))
            pos = hit.end()
            continue
        tag = _TAG_END_RE.match(text, hit.start())
        if tag is None:
            pos = hit.start() + 1  # a lone "<", not a tag
            continue
        pos = tag.end()
        if token == '<':
            continue  # some other tag — skipped whole
        if text[tag.end() - 2] == '/':
            pairs.append((hit.start(), tag.end()))  # <svg …/> — complete, empty
        else:
            stack.append((hit.start(), tag.end()))

    # Pairs are properly nested or disjoint, so after sorting by start a pair
    # is outermost exactly when it begins past the end of the last kept one.
    # (A pair nested only inside DANGLING opens is outermost too.)
    islands = []
    for start, end in sorted(pairs):
        if not islands or start >= islands[-1][1]:
            islands.append((start, end))
    return islands, list(stack)


# A data: URI of any type. Preceded by a quote it runs to that quote (hand-made
# SVG data URIs carry spaces and the other quote kind); bare — as in
# ``![alt](data:…)`` — it runs to whitespace or a closing delimiter.
_DATA_URI_RE = re.compile(
    r'''(?P<q>["'])(?P<quoted>data:[a-z0-9.+-]+/[a-z0-9.+-]+[;,].*?)(?P=q)'''
    r'''|(?P<bare>data:[a-z0-9.+-]+/[a-z0-9.+-]+[;,][^\s"'()<>]*)''',
    re.IGNORECASE | re.DOTALL,
)


def _data_uri_spans(text: str):
    for m in _DATA_URI_RE.finditer(text):
        group = 'quoted' if m.group('quoted') is not None else 'bare'
        yield m.span(group)


def check_media_limits(content):
    """Budget the media in a document's Markdown. Returns a German error
    sentence (the caller answers 413 with it, before writing anything) or
    ``None`` if the text is within budget.

    * a single ``data:`` URI above ``MAX_DATA_URI_BYTES`` → refused;
    * media in total above ``MAX_DOCUMENT_MEDIA_BYTES`` → refused. Media =
      ``data:`` URIs plus inline ``<svg>`` islands, overlaps counted once
      (a data URI inside an inline SVG is not billed twice).

    Bytes are those of the text AS WRITTEN (utf-8) — a base64 payload counts
    inflated, which is exactly what the Text column and the reader carry.
    Fences are not special: a data URI shown as code costs the same bytes.
    Mermaid source is text, not media. Non-strings are not this function's
    business (the routes validate their types) → ``None``.
    """
    if not isinstance(content, str):
        return None
    # No single URI and no total can exceed a limit the whole text is under.
    if len(content.encode('utf-8')) <= MAX_DATA_URI_BYTES:
        return None

    spans = []
    for start, end in _data_uri_spans(content):
        if len(content[start:end].encode('utf-8')) > MAX_DATA_URI_BYTES:
            return DATA_URI_TOO_LARGE
        spans.append((start, end))
    islands, _dangling = find_svg_spans(content)
    spans.extend(islands)

    total = 0
    covered_until = 0
    for start, end in sorted(spans):
        start = max(start, covered_until)
        if end > start:
            total += len(content[start:end].encode('utf-8'))
            covered_until = end
    if total > MAX_DOCUMENT_MEDIA_BYTES:
        return DOCUMENT_MEDIA_TOO_LARGE
    return None


# A Mermaid fence, CommonMark-shaped: ≤ 3 spaces indent, ``` or ~~~ (3+), the
# info string starts with ``mermaid``; closed by the same marker run or — as
# CommonMark has it for an unclosed fence — by the end of the document.
_MERMAID_FENCE_RE = re.compile(
    r'^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*mermaid\b[^\n]*\n'
    r'.*?'
    r'(?:^ {0,3}(?P=fence)[`~]*[ \t]*$|\Z)',
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_MEDIA_HINT_RE = re.compile(r'<svg|data:|mermaid', re.IGNORECASE)
_IMG_TAG_WITH_DATA_RE = re.compile(r'<img\b[^>]*\bdata:[^>]*>', re.IGNORECASE)
_MD_IMAGE_WITH_DATA_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\(\s*<?data:[^)]*\)',
                                    re.IGNORECASE)


def strip_media_for_preview(content: str) -> str:
    """The document text without its media source: inline ``<svg>`` markup,
    ``data:`` URIs (a Markdown image keeps its alt text) and Mermaid fences
    are removed, so a list preview reads as prose instead of as markup.

    A text without media comes back as the SAME string — previews of the
    existing library do not change. Preview-grade, not a parser: an ``<svg>``
    shown as code inside a fence is dropped as well.
    """
    if not content or not _MEDIA_HINT_RE.search(content):
        return content

    text = _MERMAID_FENCE_RE.sub('', content)

    # Closed figures only. A dangling ``<svg`` in raw Markdown is far more
    # often prose — a heading that says "inline `<svg>`" — than a broken figure.
    islands, _dangling = find_svg_spans(text)
    if islands:
        kept = []
        cursor = 0
        for start, end in islands:
            kept.append(text[cursor:start])
            cursor = end
        kept.append(text[cursor:])
        text = ''.join(kept)

    text = _IMG_TAG_WITH_DATA_RE.sub('', text)
    text = _MD_IMAGE_WITH_DATA_RE.sub(lambda m: m.group('alt'), text)
    kept = []
    cursor = 0
    for start, end in _data_uri_spans(text):
        kept.append(text[cursor:start])
        cursor = end
    kept.append(text[cursor:])
    text = ''.join(kept)

    if text == content:
        return content
    # Removing a figure block leaves its surrounding blank lines stacked up.
    return re.sub(r'\n{3,}', '\n\n', text).lstrip('\n')
