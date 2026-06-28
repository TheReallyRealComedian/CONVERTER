# services/epub_math.py
"""Server-side LaTeX→MathML for the EPUB/Kindle build (Sprint KINDLE-MATH).

The shared ``render_markdown_to_html`` leaves math as class-tagged spans
(``math-inline`` / ``math-display``) holding the raw (escapeHtml'd) LaTeX, which
KaTeX renders client-side per surface. E-readers run no reliable JS, so in the
EPUB those spans would stay as bare LaTeX text. This pure module does a one-pass
transform over the rendered HTML body — *before* it becomes an EPUB chapter —
turning each span into embedded EPUB3 MathML via the pure-Python ``latex2mathml``.

Pure ``str -> (str, bool)`` transform: no Flask, no SMTP, no EPUB. ``has_math``
reports whether ≥1 ``<math>`` was emitted (drives the OPF ``mathml`` property).

Per-equation safety is load-bearing: ``latex2mathml.convert`` *raises* on broken
or partial LaTeX (unlike KaTeX's ``throwOnError:false``). A single bad formula
without try/except would crash the whole EPUB build → 502 on send. On any
exception (or empty/whitespace LaTeX) we leave the original span untouched, so
the worst case is visible raw LaTeX — never a failed send.
"""
import lxml.html as LH
from latex2mathml.converter import convert
from lxml import etree

_MATH_CLASSES = ('math-inline', 'math-display')


def latex_spans_to_mathml(html_body: str) -> tuple[str, bool]:
    """Replace ``math-inline``/``math-display`` spans with EPUB3 MathML.

    Returns ``(transformed_body, has_math)``. If no ``<math>`` is emitted (no
    spans, or every span empty/unparseable), the *original* body is returned
    byte-identical and ``has_math`` is ``False`` — math-free EPUBs stay
    byte-stable vs. today.
    """
    body = html_body or ''
    # Cheap early-out: avoid parsing (and any lxml renormalization) when there
    # is provably no math to transform.
    if 'math-inline' not in body and 'math-display' not in body:
        return body, False

    # create_parent='div' avoids the implicit html/body shell; we serialize only
    # the wrapper's inner content at the end, never the wrapper itself.
    root = LH.fragment_fromstring(body, create_parent='div')

    emitted = 0
    # Snapshot the node list: we mutate the tree (replace) while iterating.
    for el in list(root.iter()):
        classes = (el.get('class') or '').split()
        is_display = 'math-display' in classes
        if not is_display and 'math-inline' not in classes:
            continue

        latex = el.text_content()  # lxml decodes the 4 escaped entities → exact LaTeX
        if not latex.strip():
            continue  # empty/whitespace span: leave untouched

        try:
            mathml = convert(latex, display='block' if is_display else 'inline')
        except Exception:
            continue  # broken LaTeX: leave the visible raw-LaTeX span in place

        math_el = LH.fragment_fromstring(mathml)
        math_el.set('alttext', latex)  # accessibility / recovery floor

        parent = el.getparent()
        if is_display:
            # Source span is bare (no <p>/<div> wrap) → without a block container
            # the display math would sit inline. Wrap it.
            wrap = LH.fragment_fromstring('<p class="math-display"></p>')
            wrap.append(math_el)
            wrap.tail = el.tail
            parent.replace(el, wrap)
        else:
            math_el.tail = el.tail
            parent.replace(el, math_el)
        emitted += 1

    if not emitted:
        # Nothing changed in a way we want to keep — return the original body so
        # math-free (or all-failed) bodies stay byte-identical.
        return body, False

    inner = root.text or ''
    for child in root:
        inner += etree.tostring(child, encoding='unicode', method='html')
    return inner, True
