"""LaTeX-Mathe-Schutz im geteilten Renderer (MATH-RENDER).

``render_markdown_to_html`` tokenisiert ``$…$``/``$$…$$`` über das
dollarmath-Plugin, *bevor* der Markdown-Inline-Parser ``_``/``\\``/``{}``
zerlegt. Diese Tests pinnen: rohes LaTeX bleibt intakt, Streu-``$`` (Preise)
bleibt Text, Nicht-Mathe-Content unverändert, nh3 sanitisiert weiter.
"""
from app_pkg.markdown_render import render_markdown_to_html


def test_block_math_stays_intact():
    html = render_markdown_to_html(r'$$\frac{dC}{dt}$$')
    assert 'class="math-display"' in html
    # \frac{dC}{dt} darf nicht markdown-zerlegt werden (kein <em> aus _).
    assert r'\frac{dC}{dt}' in html
    assert '<em>' not in html


def test_inline_math_stays_intact():
    html = render_markdown_to_html(r'Wert $C_{\text{in}}$ hier')
    assert 'class="math-inline"' in html
    assert r'C_{\text{in}}' in html
    # Der Unterstrich darf kein Emphasis öffnen.
    assert '<em>' not in html


def test_full_accumulation_formula():
    """Olis exakter Reader-Inhalt — Block + Inline gemischt."""
    src = (
        r'$$\frac{dC}{dt} = \frac{F}{V}\,(C_{\text{in}} - C) + r$$'
        '\n\nMit $C$, $F$ und $V$.'
    )
    html = render_markdown_to_html(src)
    assert r'\frac{F}{V}\,(C_{\text{in}} - C) + r' in html
    assert html.count('class="math-inline"') == 3


def test_stray_dollar_prices_stay_text():
    """„kostet 5$ und 10$" darf nicht zu Mathe werden (allow_digits aus)."""
    html = render_markdown_to_html('Das kostet 5$ und 10$ zusammen.')
    assert 'math-inline' not in html
    assert 'math-display' not in html
    assert '5$ und 10$' in html


def test_spaced_dollar_stays_text():
    """„$ 5 $" mit Spaces bleibt Text (allow_space aus)."""
    html = render_markdown_to_html('Preis $ 5 $ heute.')
    assert 'math-inline' not in html
    assert '$ 5 $' in html


def test_non_math_content_unchanged():
    html = render_markdown_to_html('Normaler **fetter** Text und `code`.')
    assert '<strong>fetter</strong>' in html
    assert '<code>code</code>' in html
    assert 'math-' not in html


def test_nh3_still_sanitizes_no_xss():
    """nh3 entfernt weiter aktive Inhalte — auch neben Mathe."""
    html = render_markdown_to_html(
        r'$x$ <script>alert(1)</script> <img src=x onerror=alert(1)>'
    )
    assert '<script>' not in html
    assert 'onerror' not in html
    assert 'class="math-inline"' in html


def test_empty_input_returns_empty():
    assert render_markdown_to_html('') == ''
    assert render_markdown_to_html(None) == ''
