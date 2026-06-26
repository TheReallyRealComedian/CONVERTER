"""Single source of truth for Markdown→HTML in the project.

Both the Markdown→PDF pipeline (``app_pkg/markdown.py``) and the library
reading-view (``app_pkg/library.py``) render through ``render_markdown_to_html``
so the MarkdownIt config, ``pygments`` highlight callback, and ``nh3`` allow-list
stay in one place.
"""
import nh3
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from mdit_py_plugins.dollarmath import dollarmath_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound


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


def render_markdown_to_html(markdown_text: str) -> str:
    """Render Markdown to sanitized HTML. Empty/None input returns ''."""
    if not markdown_text:
        return ''
    rendered = _md.render(markdown_text)
    return nh3.clean(rendered, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRIBUTES)
