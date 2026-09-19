"""Single source of truth for Markdown→HTML in the project.

The Markdown→PDF pipeline (``app_pkg/markdown.py``), the library reading-view
(``app_pkg/library.py``) and the EPUB/Kindle flow (``app_pkg/kindle.py``) all
render through ``render_markdown_to_html`` so the MarkdownIt config,
``pygments`` highlight callback, and ``nh3`` allow-list stay in one place.

Figures (Sprint RICH-MEDIA) — how a document gets pictures without a second
SVG policy:

* **Inline ``<svg>``** never meets the Markdown allow-list. ⚠️ Wildcard-Falle:
  that list carries ``'*': {class, id, style}``; adding the SVG tags to it
  would hand every SVG element the ``style`` and ``class`` the SVG doctrine
  bans (``url()`` in CSS, collision with app utilities). So the SVG islands
  are cut out of the rendered HTML, each goes through
  ``services.svg_sanitize.sanitize_svg`` — the very nh3 call the card figures
  use, one policy in one place — and comes back into a slot after the main
  pass. Whatever the cut misses still hits the main pass, where ``svg`` is not
  allowed: a wrong cut changes what is shown, never what is allowed.
* **``data:`` images** need two gates opened: markdown-it's ``validateLink``
  (stock: no ``svg+xml``) and nh3's ``url_schemes``. ⚠️ data:-Falle:
  ``url_schemes`` is global — opening ``data`` opens ``<a href="data:…">``
  too. ``_filter_markdown_attribute`` closes it again everywhere except
  ``img@src`` with one of five image types.
* **Mermaid** is a plain fence here; the reader renders it client-side.
"""
import re
import secrets

import nh3
from markdown_it import MarkdownIt
from markdown_it.common import normalize_url
from markdown_it.common.utils import escapeHtml
from mdit_py_plugins.dollarmath import dollarmath_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

from services.doc_media import SVG_EDGE_RE, find_svg_spans
from services.svg_sanitize import ensure_viewbox, sanitize_svg, svg_has_drawable


def highlight_code(code, lang, _):
    try:
        lexer = get_lexer_by_name(lang, stripall=True)
    except ClassNotFound:
        lexer = get_lexer_by_name('text', stripall=True)
    formatter = HtmlFormatter(style='default', cssclass='highlight', noclasses=True)
    return highlight(code, lexer, formatter)


# --- LaTeX-Mathe (MATH-RENDER) ---------------------------------------------
# dollarmath *schützt* die Mathe: es tokenisiert ``$…$``/``$$…$$`` bevor der
# Inline-Parser ``_``/``\``/``{}`` zerlegt. Wir rendern das rohe LaTeX als
# class-getaggten Span (``math-inline`` / ``math-display``) — KaTeX rendert
# clientseitig pro Fläche (Reader-JS · Preview-iframe · Playwright). Die Spans
# überstehen nh3 (span/class erlaubt), der escapeHtml'te LaTeX-Body bleibt als
# Text-Content erhalten (KaTeX liest ``textContent``, das die Entities zurück
# dekodiert). Konservativ konfiguriert: ``allow_space``/``allow_digits`` aus →
# Streu-``$`` (Preise wie „5$"/„$ 5 $") bleibt Text, wird nicht zu Mathe.

def _render_math_inline(_self, tokens, idx, _options, _env):
    return f'<span class="math-inline">{escapeHtml(tokens[idx].content.strip())}</span>'


def _render_math_display(_self, tokens, idx, _options, _env):
    return f'<span class="math-display">{escapeHtml(tokens[idx].content.strip())}</span>'


_md = MarkdownIt(
    'default',
    {'breaks': True, 'html': True, 'highlight': highlight_code},
)
_md.use(
    dollarmath_plugin,
    allow_space=False,
    allow_digits=False,
    allow_labels=False,
    double_inline=False,
)
# Eigene Render-Rules statt der Plugin-Defaults (``math inline``/``math block``):
# class-Namen ``math-inline``/``math-display``, auf die das KaTeX-Render-Script
# pro Fläche zielt.
_md.add_render_rule('math_inline', _render_math_inline)
_md.add_render_rule('math_block', _render_math_display)


# --- Figuren: SVG-Block (RICH-MEDIA) ----------------------------------------
# CommonMark kennt ``<svg>`` nur als HTML-Block Typ 7: der Open-Tag muss ALLEIN
# auf seiner Zeile stehen, der Block endet an der ERSTEN Leerzeile und kann
# keinen Absatz unterbrechen. Ein Agent, der sein SVG „schön formatiert"
# (Leerzeile zwischen Gruppen, Attribute über mehrere Zeilen, erstes Kind auf
# der Tag-Zeile), bekäme Absätze voller ``<path>``-Text. Diese Regel liest ein
# ``<svg`` am Zeilenanfang wie CommonMark ``<pre>``/``<script>`` liest (Typ 1):
# der Block läuft bis zur Zeile, die das ``</svg>`` trägt — tiefenbewusst, ein
# verschachteltes ``<svg>`` schließt den Block nicht. Ohne schließendes
# ``</svg>`` greift sie NICHT (sonst fräße ein offenes Tag den Rest des
# Dokuments); dann gilt Stock-CommonMark, und die Insel-Logik unten setzt einen
# Platzhalter mit Grund. Emittiert ein normales ``html_block`` — ab hier ist
# ein Block-SVG nichts Besonderes mehr. Dokumente ohne ``<svg`` am
# Zeilenanfang sehen diese Regel nie.

_SVG_BLOCK_OPEN_RE = re.compile(r'<svg(?=[\s>/]|$)', re.IGNORECASE)

# Inside a container (blockquote, list item) the figure's lines have to be
# checked against the container's indent; that check is bounded by this many
# lines, beyond it the rule declines and Stock-CommonMark takes the lines. At
# the document root there is nothing to check and no bound.
_SVG_BLOCK_MAX_NESTED_LINES = 2000
_SVG_INDEX_KEY = '_svg_block_close_lines'


def _svg_close_lines(state):
    """``close[s]`` = the line at whose end an ``<svg`` opened on line ``s`` is
    closed again (depth-matched), ``-1`` if it never is. Built ONCE per
    document, in one pass, and kept in ``state.env``.

    ⚠️ Why a table and not a scan from ``s``: the rule is a paragraph
    terminator, so markdown-it asks it again for EVERY line of a paragraph. A
    forward scan per ask made 4 000 lines of never-closed ``<svg`` cost 6 s and
    8 000 lines over 20 s (×4 per doubling). "Remember the last ``</svg>``" is
    no fix — ``<svg><svg></svg>`` per line has closers everywhere and still
    never balances. With per-line net depth ``net`` and its prefix sums ``P``,
    "closed at the end of line e" is ``P[e+1] <= P[s]``: the NEXT
    SMALLER-OR-EQUAL element to the right, which one monotonic-stack pass
    answers for all lines at once."""
    cached = state.env.get(_SVG_INDEX_KEY)
    if cached is not None and cached[0] is state.src:
        return cached[1]
    src = state.src
    starts = [0]
    find = src.find
    at = find('\n')
    while at >= 0:
        starts.append(at + 1)
        at = find('\n', at + 1)
    n_lines = len(starts)

    prefix = [0] * (n_lines + 1)  # first as per-line net (shifted by one) …
    line = 0
    for edge in SVG_EDGE_RE.finditer(src):
        at = edge.start()
        while line + 1 < n_lines and starts[line + 1] <= at:
            line += 1
        prefix[line + 1] += -1 if edge.group(0)[1] == '/' else 1
    for i in range(n_lines):      # … then summed up in place
        prefix[i + 1] += prefix[i]

    close = [-1] * n_lines
    waiting = []  # indices still without an answer; their prefix values rise
    for j, value in enumerate(prefix):
        while waiting and value <= prefix[waiting[-1]]:
            close[waiting.pop()] = j - 1
        if j < n_lines:
            waiting.append(j)
    state.env[_SVG_INDEX_KEY] = (src, close)
    return close


def _svg_block(state, startLine, endLine, silent):
    if state.is_code_block(startLine):
        return False
    if not state.md.options.get('html', None):
        return False
    pos = state.bMarks[startLine] + state.tShift[startLine]
    if not _SVG_BLOCK_OPEN_RE.match(state.src, pos, state.eMarks[startLine]):
        return False

    lastLine = _svg_close_lines(state)[startLine]
    if lastLine < 0 or lastLine >= endLine:
        return False
    # The enclosing container must not end before the figure does. At the root
    # (level 0: blkIndent 0, no negative sCount ahead) that cannot happen —
    # O(1) there. Nested, it is one C-level min over a bounded slice.
    if state.level > 0 and lastLine > startLine:
        if lastLine - startLine > _SVG_BLOCK_MAX_NESTED_LINES:
            return False
        if min(state.sCount[startLine + 1:lastLine + 1]) < state.blkIndent:
            return False
    if silent:
        return True  # may terminate a paragraph, like the Typ-1 blocks

    state.line = lastLine + 1
    token = state.push('html_block', '', 0)
    token.map = [startLine, state.line]
    token.content = state.getLines(startLine, state.line, state.blkIndent, True)
    return True


_md.block.ruler.before(
    'html_block', 'svg_block', _svg_block,
    {'alt': ['paragraph', 'reference', 'blockquote']},
)


# --- Figuren: data:-Bilder, Tor 1 von 2 (markdown-it) ------------------------
# Stock-``validateLink`` lässt ``data:image/(gif|png|jpeg|webp);`` durch, aber
# kein ``svg+xml`` und nicht die Komma-Form ohne Parameter
# (``data:image/svg+xml,%3Csvg…``) — ``![](data:image/svg+xml…)`` bliebe
# Literal-Text. ``validateLink`` gilt für Links UND Bilder: ein
# ``[x](data:image/…)`` wird hier zum ``<a href>`` und fällt dann an Tor 2.

# What a browser's URL parser drops before it reads the scheme: tab/LF/CR
# anywhere, C0 controls and space up front. ``da\tta:`` IS ``data:``.
_URL_NOISE_RE = re.compile(r'[\t\n\r]')
_URL_LEADING = ''.join(map(chr, range(0x21)))
_IMAGE_DATA_URI_RE = re.compile(r'data:image/(?:svg\+xml|png|jpeg|webp|gif)[;,]')


def _scheme_view(url: str) -> str:
    # 96 chars hold any scheme + media type; never lowercase a 2-MB payload.
    return _URL_NOISE_RE.sub('', url[:96]).lstrip(_URL_LEADING).lower()


def _validate_link(url: str) -> bool:
    if _IMAGE_DATA_URI_RE.match(_scheme_view(url)):
        return True
    return normalize_url.validateLink(url)


_md.validateLink = _validate_link


_ALLOWED_TAGS = {
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'br', 'hr', 'blockquote', 'pre', 'code',
    'ul', 'ol', 'li', 'dl', 'dt', 'dd',
    'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption', 'colgroup', 'col',
    'a', 'img', 'figure', 'figcaption',
    'strong', 'em', 'b', 'i', 'u', 's', 'del', 'ins', 'mark',
    'sub', 'sup', 'small', 'abbr', 'cite', 'q', 'kbd', 'var', 'samp',
    'details', 'summary',
    'div', 'span', 'section', 'article', 'aside', 'header', 'footer', 'nav', 'main',
}

_ALLOWED_ATTRIBUTES = {
    '*': {'class', 'id', 'style'},
    'a': {'href', 'title', 'target'},
    'img': {'src', 'alt', 'title', 'width', 'height'},
    'th': {'colspan', 'rowspan', 'scope'},
    'td': {'colspan', 'rowspan'},
    'col': {'span'},
    'colgroup': {'span'},
}


# nh3 replaces its scheme list when ``url_schemes`` is passed, so the default
# has to be restated to add ``data``. 0.2.18 (the repo pin) does not export it;
# this is ammonia's list, identical to ``nh3.ALLOWED_URL_SCHEMES`` in 0.3.x
# (sentinel in tests/test_markdown_media.py).
_URL_SCHEMES = {
    'bitcoin', 'ftp', 'ftps', 'geo', 'http', 'https', 'im', 'irc', 'ircs',
    'magnet', 'mailto', 'mms', 'mx', 'news', 'nntp', 'openpgp4fpr', 'sip',
    'sms', 'smsto', 'ssh', 'tel', 'url', 'webcal', 'wtai', 'xmpp',
    'data',
}

# ``style`` is wildcard-allowed on the HTML tags (pygments writes inline
# styles), and nh3 0.2.18 does not look inside CSS: ``background:url(https://…)``
# is a tracking pixel in every reader open. Anything that can name a resource
# kills the attribute — incl. a backslash, because ``\75rl(`` IS ``url(`` after
# CSS escape processing. Pygments output carries none of these.
_STYLE_FETCH_RE = re.compile(
    r'url|image-set|image\(|cross-fade|element\(|expression|@import|\\',
    re.IGNORECASE)


def _filter_markdown_attribute(element: str, attribute: str, value: str):
    # Tor 2 von 2 für data:-Bilder — und das Schließen der data:-Falle:
    # ``data`` steht global in ``_URL_SCHEMES``, erlaubt ist es genau auf
    # ``img@src`` mit einem der fünf Bildtypen. href/src sind die einzigen
    # URL-Attribute der Allow-List.
    if attribute in ('href', 'src'):
        view = _scheme_view(value)
        if view.startswith('data:'):
            if (element == 'img' and attribute == 'src'
                    and _IMAGE_DATA_URI_RE.match(view)):
                return value
            return None
    elif attribute == 'style' and _STYLE_FETCH_RE.search(value):
        return None
    return value


def _placeholder(reason: str) -> str:
    # Visible, selectable text — the document counterpart of the CARD-SVG rule
    # "400 statt stumm leer". Static German only, never author content.
    return f'<span class="media-placeholder">[Abbildung nicht darstellbar: {reason}]</span>'


_NO_INK = 'nach der Sicherheitsprüfung bleibt kein zeichenbares Element übrig.'
_NOT_CLOSED = 'das SVG ist nicht geschlossen (&lt;/svg&gt; fehlt).'
_BLANK_LINE = 'das SVG enthält eine Leerzeile, die Markdown als Absatzgrenze liest.'

# Inside a paragraph (``<svg`` not at the start of a line) markdown-it turns
# every newline into ``<br>`` (``breaks: True``) — and ``<br>`` makes the HTML
# parser LEAVE the SVG. It is never part of a figure; drop it before the clean.
_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)
_PARAGRAPH_EDGE_RE = re.compile(r'</?p[\s>]', re.IGNORECASE)


def _render_svg_island(island: str) -> str:
    if _PARAGRAPH_EDGE_RE.search(island):
        # A blank line inside a figure the block rule could not take (mid-line
        # ``<svg``, or wrapped in a Typ-6 block like ``<figure>``): markdown-it
        # has cut it into paragraphs, the parser would drop out of the SVG at
        # the first ``<p>`` and scatter the labels. Say so instead.
        return _placeholder(_BLANK_LINE)
    cleaned = sanitize_svg(_BR_RE.sub('', island))
    if not cleaned or not svg_has_drawable(cleaned):
        return _placeholder(_NO_INK)
    return ensure_viewbox(cleaned)


def _cut_svg_islands(rendered: str, nonce: str):
    """Swap every ``<svg>`` in the rendered HTML for a slot element and return
    ``(html_with_slots, replacements)``.

    The slot is an ELEMENT whose serialization contains a ``"`` and a per-call
    nonce. After the main nh3 pass its exact string can only exist where nh3
    parsed it as a real element in element content: inside an attribute value
    the ``"`` comes back as ``&quot;``, inside text/RCDATA the ``<`` comes back
    as ``&lt;``, inside a comment it is gone — and the nonce keeps the author
    from planting one. So the figure can only land where a figure may stand,
    never inside ``alt="…"`` where its own quotes would break out."""
    islands, dangling = find_svg_spans(rendered)
    if not islands and not dangling:
        return rendered, []
    spans = sorted([(s, e, True) for s, e in islands]
                   + [(s, e, False) for s, e in dangling])
    parts, replacements, cursor = [], [], 0
    for start, end, closed in spans:
        parts.append(rendered[cursor:start])
        parts.append(f'<span class="svgslot-{nonce}-{len(replacements)}"></span>')
        if closed:
            replacements.append(_render_svg_island(rendered[start:end]))
            cursor = end
        else:
            # The placeholder goes IN FRONT of the unclosed tag; the tag itself
            # stays for the main pass. Taken out, its children would be orphans
            # in HTML context, where ``<rect …/>`` does not self-close and
            # swallows the rest of the document — which nh3 0.3.x then drops
            # WITH contents (unknown SVG-named element). Left in, the parser is
            # in foreign content, ``<rect/>`` closes, the next ``<h2>``/``<p>``
            # breaks out, and the document goes on — on 0.2.18 and 0.3.x alike.
            replacements.append(_placeholder(_NOT_CLOSED))
            cursor = start
    parts.append(rendered[cursor:])
    return ''.join(parts), replacements


def _clean_markdown_html(rendered: str, nonce: str) -> str:
    """The main nh3 pass (Markdown allow-list — no SVG tags in it)."""
    # https images load lazily and send no referrer. nh3 can force attribute
    # values, but two forced attributes come out in per-process random order
    # (HashMap) — the same document would render to different bytes in two
    # workers. So ONE attribute is forced, with the nonce in its value, and the
    # exact string is swapped for the pair afterwards (same unforgeable-marker
    # argument as the slots). Neither attribute is in the allow-list: an
    # author's ``loading="eager"`` / ``referrerpolicy="unsafe-url"`` falls.
    lazy_marker = f'lazy-{nonce}'
    cleaned = nh3.clean(
        rendered,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        attribute_filter=_filter_markdown_attribute,
        url_schemes=_URL_SCHEMES,
        set_tag_attribute_values={'img': {'loading': lazy_marker}},
    )
    return cleaned.replace(f' loading="{lazy_marker}"',
                           ' loading="lazy" referrerpolicy="no-referrer"')


def render_markdown_to_html(markdown_text: str) -> str:
    """Render Markdown to sanitized HTML. Empty/None input returns ''."""
    if not markdown_text:
        return ''
    rendered = _md.render(markdown_text)
    nonce = secrets.token_hex(8)
    rendered, figures = _cut_svg_islands(rendered, nonce)
    cleaned = _clean_markdown_html(rendered, nonce)
    if figures:
        cleaned = re.sub(
            rf'<span class="svgslot-{nonce}-(\d+)"></span>',
            lambda m: figures[int(m.group(1))],
            cleaned,
        )
        # A slot the scanner placed where no figure may stand (RCDATA like
        # <textarea>, an attribute value) came back escaped and was not
        # filled. Take the debris out instead of showing a nonce to the reader.
        cleaned = re.sub(
            rf'(?:<|&lt;)span class=(?:"|&quot;)svgslot-{nonce}-\d+(?:"|&quot;)'
            r'(?:>|&gt;)(?:<|&lt;)/span(?:>|&gt;)',
            '',
            cleaned,
        )
    return cleaned
