# services/markdown_sections.py
"""Replace a single Markdown section addressed by its heading text.

Pure, Flask-free. The endpoint layer (``app_pkg/docwrite.py``) maps the two
exceptions raised here onto HTTP 404 / 409.

A "section" is an ATX heading line plus everything below it up to (but not
including) the next heading of the **same or higher** level — so subsections
(deeper headings) belong to their parent section. The match is **level-
agnostic on the text**: ``# Intro`` and ``### Intro`` both count as "Intro".
More than one match is an error (never guess which one) → ``SectionAmbiguous``;
zero matches → ``SectionNotFound``.

Fenced-code-aware: a line that starts (after optional leading whitespace) with
``` ``` ``` or ``~~~`` toggles "inside a fenced code block", and ``#`` lines
inside a fence are **not** treated as headings (a ``# comment`` in a Python
block is not a section). This guards both directions: such a line is neither a
valid target nor a false section boundary.

**Out of scope v1:** Setext headings (``===`` / ``---`` underlines) are not
recognised — ATX (``#`` … ``######``) only. Also no multi-section / regex /
range addressing: exactly one heading-addressed section per call.
"""
import re


# ATX heading: 1–6 leading hashes, mandatory space, text, optional closed-ATX
# trailing hashes. ``group(2).strip()`` is the heading text (closing #s and
# surrounding whitespace are stripped by the tail ``\s*#*\s*$``).
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$')

# A fence marker: ``` ``` ``` or ``~~~`` after optional leading whitespace.
_FENCE_RE = re.compile(r'^\s*(```|~~~)')


class SectionError(Exception):
    """Base class for section-addressing failures."""


class SectionNotFound(SectionError):
    """No heading matched the requested target text."""


class SectionAmbiguous(SectionError):
    """More than one heading matched the requested target text."""


def _iter_headings(lines):
    """Yield ``(index, level, text)`` for each ATX heading outside fenced code.

    Lines inside a ``` ``` ``` / ``~~~`` fence are skipped, so ``#`` comments in
    a code block are never mistaken for headings.
    """
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            yield i, len(m.group(1)), m.group(2).strip()


# An HTML comment, single- or multi-line. PDF→Markdown decks open with page
# markers like ``<!-- Seite 1 -->`` / ``<!-- Grafik: … -->``; ``DOTALL`` lets a
# single pattern span newlines, and the non-greedy ``.*?`` plus the global
# substitution below close each comment independently (so ``# foo`` *inside* a
# multi-line comment is removed, never read as a heading — ``_iter_headings``
# knows code fences but not HTML comments).
_HTML_COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)


def derive_title(markdown_text: str) -> str:
    """Derive a human title from Markdown: first ATX heading, else first line.

    Used to rescue degenerate titles (e.g. a deck whose first line is the page
    marker ``<!-- Seite 1 -->``). HTML comments are stripped first so a heading
    *after* a leading comment wins, and a ``#`` *inside* a multi-line comment is
    never mistaken for one. The first heading's text has surrounding emphasis
    markers stripped (``# *Opt*`` → ``Opt``). Falls back to the first non-empty
    line of the comment-stripped content, then ``'Untitled'``.

    Pure; **not** truncated — the caller clips to its own column/UI limit.
    """
    stripped = _HTML_COMMENT_RE.sub('', markdown_text or '')
    for _i, _level, text in _iter_headings(stripped.split('\n')):
        # Strip surrounding emphasis markers (and any whitespace they leave).
        title = text.strip().strip('*_ ').strip()
        if title:
            return title
    for line in stripped.split('\n'):
        line = line.strip()
        if line:
            return line
    return 'Untitled'


def _is_degenerate_title(title) -> bool:
    """True when ``title`` carries no real information and should be re-derived.

    Degenerate = empty/blank, a literal placeholder (``Untitled`` /
    ``Untitled Markdown``), or a leftover HTML comment marker (``<!-- … -->``)
    that the old "first line" heuristic captured verbatim. A real client title
    is left untouched by the callers — the trigger is solely degeneracy.
    """
    t = (title or '').strip()
    return (not t
            or t.lower() in ('untitled', 'untitled markdown')
            or t.startswith('<!--'))


def replace_section(markdown_text: str, heading: str, new_section: str) -> str:
    """Return ``markdown_text`` with the section under ``heading`` replaced.

    ``heading`` is matched against heading text only (leading ``#`` and
    surrounding whitespace are stripped from it first), level-agnostically.
    ``new_section`` is spliced in verbatim — it carries its own heading (the
    agent may rename or drop it; no heading is enforced).

    Raises ``SectionNotFound`` (0 matches) or ``SectionAmbiguous`` (>1 match).
    """
    target = heading.lstrip('#').strip()
    # split('\n') + '\n'.join() round-trips exactly, preserving the document's
    # trailing-newline state and never gluing adjacent lines together.
    lines = markdown_text.split('\n')
    headings = list(_iter_headings(lines))

    matches = [(i, level) for (i, level, text) in headings if text == target]
    if not matches:
        raise SectionNotFound(target)
    if len(matches) > 1:
        raise SectionAmbiguous(target)

    start, level = matches[0]
    # Section end = first later heading with level <= the matched level (a
    # deeper heading is a subsection and stays inside); otherwise EOF.
    end = len(lines)
    for i, lvl, _text in headings:
        if i > start and lvl <= level:
            end = i
            break

    result = lines[:start] + new_section.split('\n') + lines[end:]
    return '\n'.join(result)
