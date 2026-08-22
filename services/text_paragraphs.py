"""Hyphen-safe paragraph grouping for ``partition_text`` (TXT-BINDESTRICH, KLEINKRAM 2026-08-22).

The ``unstructured`` branch of the document router (TXT/EML/MD and the HTML
fallback) tore ``SYNC-FREEZE multi-process RQ check`` into three paragraphs.
Measured on the pinned ``unstructured`` 0.18.32, at real element objects —
not read from the docs (house rule of ``services/unstructured_markdown.py``):

* ``partition(filename=..., strategy="fast")`` → ``partition_text`` with the
  default ``paragraph_grouper=None`` → ``auto_paragraph_grouper``: fewer than
  10 % blank lines → ``new_line_grouper`` (splits at ``LINE_BREAK_RE`` only —
  never at a hyphen); otherwise → ``blank_line_grouper`` →
  ``group_broken_paragraphs``.
* ``group_broken_paragraphs`` splits every paragraph with
  ``PARAGRAPH_PATTERN_RE = ((?:BULLETS)|\\s*\\n\\s*)(?!BULLETS|$)`` and
  ``UNICODE_BULLETS`` contains ``-`` (U+002D) **and** ``–`` (U+2013) —
  **unanchored**, so a hyphen anywhere in a line is a split point. Two
  consequences, both measured:

  - a paragraph whose pieces are all shorter than five words
    (``all_lines_short``) is emitted piecewise: ``SYNC`` / ``FREEZE multi`` /
    ``process RQ check``; ``Kurz-Update:`` → ``Kurz`` / ``Update:``;
  - a bullet paragraph goes through ``group_bullet_paragraph`` →
    ``UNICODE_BULLETS_RE_0W.split`` → ``- erster Punkt mit Binde-Strich``
    becomes ListItem ``erster Punkt mit Binde`` + ListItem ``Strich``.

  Paragraphs with at least one long line are joined with spaces
  (``PARAGRAPH_PATTERN`` only) and survive — which is why an EML quote chain
  of long lines looked fine while short lines (greetings, signatures,
  subject-like lines, ``2024-2026``, ``E-Mail``) did not. EML is hit exactly
  like TXT: ``partition_email`` hands the body to the same ``partition_text``.

What this module does: the **same three heuristics** as upstream — the 10 %
blank-line ratio, "short" = fewer than five words, bullets open a paragraph —
with bullets recognised **only at the start of a line and only when followed
by whitespace** (a hyphen inside a word or a ``*`` that opens emphasis is
text). It is handed to ``partition()`` as ``paragraph_grouper`` by
``services/document_router.py``; element classification (Title /
NarrativeText / ListItem) stays ``unstructured``'s. Not replicated on purpose:
upstream's pytesseract hack that turns a leading ``e`` into a bullet — an OCR
artefact, irrelevant for text files, and it would claim structure. Also not
replicated: splitting at bullet characters *inside* a line (``• a • b``) —
mid-line markers are far more often hyphens or asterisks in prose.

Pure module (no ``unstructured`` import — the dev machine has none; the
container-only sentinel in ``tests/test_text_paragraphs.py`` pins the upstream
defect this module exists for, so a dependency bump that fixes it upstream is
noticed instead of carrying this grouper forever).
"""
import re

# Upstream ``auto_paragraph_grouper`` defaults (unstructured 0.18.32).
NEW_LINE_THRESHOLD = 0.1   # blank-line ratio below which every line is a paragraph
MAX_LINE_COUNT = 2000      # only the first N lines decide the ratio
# Upstream ``group_broken_paragraphs``: a line with fewer words is "short".
SHORT_LINE_WORDS = 5

# ``unstructured.nlp.patterns.UNICODE_BULLETS`` (0.18.32), as characters.
# ``*`` is the regex-escaped entry there; \x95 appears twice upstream.
_BULLET_CHARS = ('\x95•‣⁃ㅤ⁌⁍∙○●'
                 '◘◦☙❥❧⦾⦿-–*·')
# A bullet opens a line: optional indent, ONE marker, then whitespace or end.
_BULLET_LINE = re.compile(r'^[^\S\n]*[' + re.escape(_BULLET_CHARS) + r'](?:\s|$)')

_LINE_BREAK = re.compile(r'(?<=\n)')            # upstream LINE_BREAK_RE
_LINE_SPLIT = re.compile(r'\s*\n\s*')            # upstream PARAGRAPH_PATTERN
_PARAGRAPH_SPLIT = re.compile(r'(?:\s*\n\s*){2}')  # upstream DOUBLE_PARAGRAPH_PATTERN_RE


def is_bullet_line(line):
    """True if ``line`` opens with a bullet marker followed by whitespace/end."""
    return _BULLET_LINE.match(line) is not None


def _is_short(line):
    # Verbatim upstream: ``len(line.strip().split(" ")) < 5``.
    return len(line.strip().split(' ')) < SHORT_LINE_WORDS


def _group_blank_line_paragraph(paragraph):
    """One blank-line-delimited paragraph → list of output paragraphs."""
    lines = [line.strip() for line in _LINE_SPLIT.split(paragraph) if line.strip()]
    if not lines:
        return []
    if any(is_bullet_line(line) for line in lines):
        # Bullet items: a marker line opens an item, the following non-marker
        # lines continue it (upstream joins wrapped bullet lines the same way).
        items = []
        for line in lines:
            if is_bullet_line(line) or not items:
                items.append(line)
            else:
                items[-1] = items[-1] + ' ' + line
        return items
    if all(_is_short(line) for line in lines):
        # Address blocks, signatures, license headers: lines stay separate.
        return lines
    # Hard-wrapped prose: one paragraph, line breaks become spaces.
    return [' '.join(lines)]


def group_paragraphs(text):
    """``paragraph_grouper`` for ``unstructured.partition.text.partition_text``.

    Returns the text re-grouped so that each output paragraph is one line,
    separated by blank lines — the shape ``partition_text`` then splits at
    newlines and classifies element by element.
    """
    if not text or not text.strip():
        return ''
    lines = _LINE_BREAK.split(text)
    considered = lines[:min(len(lines), MAX_LINE_COUNT)]
    blank = sum(1 for line in considered if not line.strip())
    if blank / len(considered) < NEW_LINE_THRESHOLD:
        # Upstream ``new_line_grouper``: one paragraph per line, nothing else.
        return '\n\n'.join(line.strip() for line in lines if line.strip())
    paragraphs = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraphs.extend(_group_blank_line_paragraph(paragraph))
    return '\n\n'.join(paragraphs)
