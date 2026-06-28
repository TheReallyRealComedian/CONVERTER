"""Sprint KINDLE-MATH Phase 1 — the pure LaTeX→MathML span transform.

``latex_spans_to_mathml`` is a pure ``str -> (str, bool)`` transform run over the
rendered HTML body before it becomes an EPUB chapter. No Flask, no SMTP, no EPUB.
The load-bearing test is malformed-LaTeX-does-not-raise: ``latex2mathml.convert``
throws on broken input, and a single bad formula must not crash the build.
"""
from services.epub_math import latex_spans_to_mathml


def test_inline_span_becomes_inline_math():
    body = '<p>Vor <span class="math-inline">a+b</span> nach.</p>'
    out, has_math = latex_spans_to_mathml(body)
    assert has_math is True
    assert '<math' in out
    assert 'display="inline"' in out
    # Surrounding non-math text survives the round-trip.
    assert 'Vor ' in out and ' nach.' in out
    # The original raw-LaTeX span is gone.
    assert 'math-inline">a+b<' not in out


def test_display_span_becomes_block_math_and_is_block_wrapped():
    body = '<span class="math-display">\\frac{1}{2}</span>'
    out, has_math = latex_spans_to_mathml(body)
    assert has_math is True
    assert 'display="block"' in out
    # Bare source span must be wrapped in a block container, not left inline.
    assert '<p class="math-display"><math' in out


def test_alttext_carries_original_latex():
    body = '<span class="math-inline">x^2</span>'
    out, _ = latex_spans_to_mathml(body)
    assert 'alttext="x^2"' in out


def test_malformed_latex_does_not_raise_and_keeps_visible_span():
    # latex2mathml.convert raises on this; without try/except the whole EPUB
    # build would 502. The visible raw-LaTeX span must remain instead.
    body = '<p><span class="math-inline">\\frac{1}{</span></p>'
    out, has_math = latex_spans_to_mathml(body)
    assert has_math is False
    assert '\\frac{1}{' in out
    assert '<math' not in out


def test_empty_or_whitespace_span_is_left_untouched():
    body = '<span class="math-inline">   </span>'
    out, has_math = latex_spans_to_mathml(body)
    assert has_math is False
    assert out == body  # byte-identical


def test_math_free_body_is_byte_identical_and_has_math_false():
    body = '<h1>Titel</h1><p>Kein <em>Math</em> hier &amp; fertig.</p>'
    out, has_math = latex_spans_to_mathml(body)
    assert has_math is False
    assert out == body  # byte-identical, untouched


def test_non_math_html_around_spans_survives():
    body = ('<h2>Kopf</h2><p>Text <span class="math-inline">a+b</span> mehr</p>'
            '<ul><li>eins</li><li>zwei</li></ul>')
    out, _ = latex_spans_to_mathml(body)
    assert '<h2>Kopf</h2>' in out
    assert '<ul><li>eins</li><li>zwei</li></ul>' in out
    assert 'Text ' in out and ' mehr' in out


def test_mixed_valid_and_invalid_emits_one_keeps_other():
    body = ('<span class="math-inline">a+b</span>'
            '<span class="math-inline">\\frac{1}{</span>')
    out, has_math = latex_spans_to_mathml(body)
    assert has_math is True
    assert '<math' in out
    # The broken one stays as a visible raw-LaTeX span.
    assert '\\frac{1}{' in out


def test_none_body_does_not_crash():
    out, has_math = latex_spans_to_mathml(None)
    assert out == '' and has_math is False
