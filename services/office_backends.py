"""Office/web conversion backends behind the DOC-API result shape (DOC-ENGINE P1).

Pure module — no Flask, no SDK singleton, no top-level heavy imports (the dev
box has neither markitdown nor trafilatura; imports live in the functions,
like the in-task engine imports in ``tasks.py``). Each backend is an adapter
``(source_path) -> (markdown, warnings)``; ``tasks.convert_document_task``
routes by extension and wraps the pair into the shared document-level
deterministic ``build_result_payload``.

The invocations replicate the bake-off's measured calls VERBATIM
(``corpus/bakeoff/harness/adapters.py``) — the per-format winners of the
decision doc (2026-08-08) won with exactly these calls, so a deviating call
would void the measurement:

* DOCX -> ``pandoc -f docx -t gfm --wrap=none`` (subprocess; the binary ships
  in the image). The only candidate carrying the image-footnote-link chain
  (rule 3: 4/4 vs 0/4). ``gfm`` output has the ``smart`` extension DISABLED
  by default, so original typography (– „…“) passes through verbatim — the
  bake-off's "smart typography" attribution for the 0.941 text score is
  refuted (the harness normalizes dashes/quotes before scoring; the delta is
  tokenization/structure artifacts), hence no typography switch is set.
* PPTX -> ``markitdown`` (measured 0.1.7) — recall 1.0, the only candidate
  carrying speaker notes.
* HTML -> ``trafilatura`` markdown extraction (measured 2.2.0, <2 %
  boilerplate) plus the metadata head below.

EML (and TXT/MD) deliberately stay on the unstructured serializer path
(decision doc: functional, without competition for EML).

HTML metadata head (Oli's locked decision: title/author/date from the HTML
head — trafilatura's body extraction drops them, and CONVERTER derives the
library title from the first heading, TITLE-FIX): the ``<title>`` tag becomes
a leading ``# `` heading, author/date from ``trafilatura.extract_metadata``
become one italic line. Measured on the corpus exemplar: the raw ``<title>``
tag carries kicker + headline ("Korruptes Web 2.0: Verraten und verkauft -
…") while trafilatura's own title field yields the bare sitename ("SPIEGEL
ONLINE") — so the tag wins and the metadata field is only the fallback.
No site-specific body heuristics (h3/h4/byline classes): never claim more
structure than the source declares (DOC-FIX rule).
"""
import shutil
import subprocess
from html.parser import HTMLParser

# Per-call deadline for the pandoc subprocess (NARR-TIMEOUT doctrine: every
# external call carries its own deadline). Harness value.
PANDOC_TIMEOUT_SECONDS = 600

# The measured pandoc invocation (bake-off adapter) — kept as a module
# constant so the sentinel test can pin the exact argument vector.
PANDOC_ARGS = ('-f', 'docx', '-t', 'gfm', '--wrap=none')


def convert_docx_pandoc(source_path):
    """DOCX → Markdown via pandoc. Returns ``(markdown, warnings)``.

    Raises ``RuntimeError`` (→ task failure, honest) when the binary is
    missing, exits nonzero or yields empty output; nonzero stderr on success
    becomes a warning (pandoc reports dropped constructs there).
    """
    exe = shutil.which('pandoc')
    if exe is None:
        raise RuntimeError('pandoc nicht im PATH — DOCX-Backend nicht verfügbar.')
    proc = subprocess.run(
        [exe, *PANDOC_ARGS, source_path],
        capture_output=True, text=True, timeout=PANDOC_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'pandoc rc={proc.returncode}: {proc.stderr[:400]}')
    markdown = proc.stdout
    if not markdown.strip():
        raise RuntimeError('pandoc lieferte leeres Markdown.')
    warnings = [proc.stderr.strip()[:300]] if proc.stderr.strip() else []
    return markdown, warnings


def convert_pptx_markitdown(source_path):
    """PPTX → Markdown via markitdown. Returns ``(markdown, warnings)``."""
    from markitdown import MarkItDown

    result = MarkItDown().convert(source_path)
    markdown = getattr(result, 'text_content', '') or ''
    if not markdown.strip():
        raise RuntimeError('markitdown lieferte leeres Markdown.')
    return markdown, []


class _TitleTagParser(HTMLParser):
    """Collects the text of the FIRST ``<title>`` element (the head title;
    later ``<title>`` occurrences, e.g. inside inline SVG, are ignored)."""

    def __init__(self):
        super().__init__()
        self._in_title = False
        self._done = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'title' and not self._done:
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == 'title' and self._in_title:
            self._in_title = False
            self._done = True

    def handle_data(self, data):
        if self._in_title:
            self.parts.append(data)


def _html_title(html_text):
    """The first ``<title>`` tag's text, whitespace-collapsed; ``''`` if none.

    stdlib HTMLParser (entities decoded via convert_charrefs) — no lxml
    dependency for one head tag.
    """
    parser = _TitleTagParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        return ''
    return ' '.join(''.join(parser.parts).split())


def _decode_html(raw_bytes):
    """utf-8 → cp1252 → latin-1 decode chain (the measured harness procedure;
    latin-1 cannot fail, so the chain always yields text)."""
    for encoding in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode('latin-1', errors='replace')  # unreachable guard


def convert_html_trafilatura(source_path):
    """HTML/HTM → Markdown via trafilatura + metadata head.

    Returns ``(markdown, warnings)``; ``markdown`` is ``None`` when
    trafilatura finds no main content — the CALLER decides the fallback
    (the task degrades to the unstructured path instead of failing: the
    legacy path could always serve some text for HTML, so a hard fail here
    would be a capability regression).
    """
    import trafilatura

    with open(source_path, 'rb') as f:
        raw = f.read()
    html_text = _decode_html(raw)

    body = trafilatura.extract(
        html_text, output_format='markdown', include_tables=True,
        include_links=True, include_formatting=True, include_comments=False,
    )
    if not body or not body.strip():
        return None, []

    head_lines = []
    title = _html_title(html_text)
    metadata = None
    try:
        metadata = trafilatura.extract_metadata(html_text)
    except Exception:
        pass  # metadata is an add-on; the body result stands without it
    if not title and metadata is not None:
        title = (getattr(metadata, 'title', None) or '').strip()
    if title:
        head_lines.append(f'# {title}')
    byline_parts = []
    for attr in ('author', 'date'):
        value = (getattr(metadata, attr, None) or '').strip() if metadata else ''
        if value:
            byline_parts.append(value)
    if byline_parts:
        head_lines.append(f"*{' · '.join(byline_parts)}*")

    if head_lines:
        markdown = '\n\n'.join(head_lines + [body.strip()])
    else:
        markdown = body.strip()
    return markdown, []
