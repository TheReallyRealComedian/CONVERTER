"""RICH-MEDIA Phase 1 — figures in documents: inline SVG, data: images, https
images, through the one renderer (reader, PDF and EPUB share it).

Pure tests against ``app_pkg.markdown_render.render_markdown_to_html``. The
probe text is Olis Render-Probe #240 byte for byte
(``tests/fixtures/rich_media_probe_240.md``); the malicious fixture
(``tests/fixtures/rich_media_malicious.md``) is the same document the browser
smoke loads in Phase 2 — every external target in it points at
``evil.example``, so "the host is gone" is the whole assertion.

Doctrine under test: ONE SVG policy (``services/svg_sanitize``) for cards and
documents, reached by cutting the SVG islands out and cleaning them with the
card call — not by merging SVG tags into the Markdown allow-list, whose
``'*': {class, id, style}`` wildcard would hand them what the policy bans.
"""
import os
import time
from html.parser import HTMLParser

import nh3
import pytest

from app_pkg import markdown_render
from app_pkg.markdown_render import render_markdown_to_html
from services.svg_sanitize import sanitize_card_svg

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def _fixture(name):
    with open(os.path.join(FIXTURES, name), encoding='utf-8') as f:
        return f.read()


class _Walker(HTMLParser):
    """Parse the OUTPUT the way a consumer would and record what is really an
    element / an attribute — the oracle for "did something break out of its
    quotes", which a substring check cannot answer."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.attrs = []  # (tag, name, value)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.extend((tag, name, value) for name, value in attrs)


def _walk(html):
    walker = _Walker()
    walker.feed(html)
    return walker


def _old_renderer(text):
    """The renderer as it stood before RICH-MEDIA — same MarkdownIt instance,
    the bare nh3 call. The pytest-sized stand-in for Gate 6."""
    return nh3.clean(markdown_render._md.render(text),
                     tags=markdown_render._ALLOWED_TAGS,
                     attributes=markdown_render._ALLOWED_ATTRIBUTES)


# --- Olis Probe #240: the variants ------------------------------------------


def test_probe_v1_inline_svg_renders_with_its_labels():
    html = render_markdown_to_html(_fixture('rich_media_probe_240.md'))
    assert '<svg xmlns="http://www.w3.org/2000/svg" width="260" height="150"' in html
    assert 'viewBox="0 0 260 150"' in html
    # Root-level font defaults survive (the probe sets them on <svg>).
    assert 'font-family="sans-serif" font-size="18"' in html
    assert '<text x="122" y="90" text-anchor="middle">C</text>' in html
    assert 'Variante 1: inline svg</text>' in html
    assert html.count('<line ') == 4
    assert '</svg>' in html


def test_probe_v2_img_tag_with_data_uri_keeps_its_src():
    html = render_markdown_to_html(_fixture('rich_media_probe_240.md'))
    assert '<img alt="Variante 2" src="data:image/svg+xml;utf8,%3Csvg' in html


def test_probe_v3_markdown_image_with_svg_data_uri_becomes_an_img():
    # Two gates: markdown-it's validateLink (stock: no svg+xml → literal text)
    # and nh3's url_schemes.
    html = render_markdown_to_html(_fixture('rich_media_probe_240.md'))
    assert '![Variante 3]' not in html
    assert '<img src="data:image/svg+xml;utf8,%3Csvg' in html
    assert 'alt="Variante 3"' in html


def test_probe_v4_mermaid_stays_a_fence_with_its_source():
    # Phase 1 does not touch Mermaid: the reader renders it client-side and
    # keeps this <pre> in the DOM (anchor stability). Same bytes as before.
    text = _fixture('rich_media_probe_240.md')
    html = render_markdown_to_html(text)
    assert '<pre><code class="language-mermaid">' in html
    assert 'flowchart LR' in html
    start = html.index('<pre><code class="language-mermaid">')
    end = html.index('</code></pre>', start)
    assert html[start:end] in _old_renderer(text)


def test_inline_svg_inside_a_paragraph_keeps_the_text_around_it():
    # Before: '<p>vor  nach</p>' — figure AND label silently gone.
    html = render_markdown_to_html(
        'vor <svg viewBox="0 0 10 10"><text x="1" y="5">Lbl</text></svg> nach')
    assert html.startswith('<p>vor <svg viewBox="0 0 10 10">')
    assert '<text x="1" y="5">Lbl</text></svg> nach</p>' in html


# --- one policy: the document path IS the card path -------------------------


def test_document_svg_equals_the_card_sanitizer_output():
    raw = ('<svg viewBox="0 0 10 10" class="hidden"><script>alert(1)</script>'
           '<rect x="1" width="5" height="5" style="fill:red" id="r"/>'
           '<a href="https://evil.example/"><text x="1" y="9">T</text></a></svg>')
    html = render_markdown_to_html(raw)
    assert html.strip() == sanitize_card_svg(raw)


def test_wildcard_trap_style_class_id_do_not_reach_svg_elements():
    """The Markdown list's '*' wildcard must not leak onto SVG — while it
    keeps serving the HTML tags it was written for."""
    html = render_markdown_to_html(
        '<svg viewBox="0 0 10 10" class="hidden" style="display:none" id="s">'
        '<rect x="1" width="5" height="5" class="hidden" style="fill:red" id="r"/></svg>\n\n'
        '<div class="note" style="color:red" id="d">HTML behält seine Attribute.</div>')
    svg = html[html.index('<svg'):html.index('</svg>')]
    assert 'class' not in svg and 'style' not in svg and 'id=' not in svg
    assert '<div class="note" style="color:red" id="d">' in html


def test_svg_tags_are_not_in_the_markdown_allow_list():
    # SENTINEL: whoever "simplifies" this into one merged nh3 pass reopens the
    # wildcard trap. The Markdown list stays SVG-free.
    from services.svg_sanitize import SVG_ALLOWED_TAGS
    assert not (SVG_ALLOWED_TAGS & markdown_render._ALLOWED_TAGS)
    assert '*' in markdown_render._ALLOWED_ATTRIBUTES


# --- the malicious fixture ---------------------------------------------------


def test_malicious_fixture_every_banned_part_falls_the_rest_renders():
    html = render_markdown_to_html(_fixture('rich_media_malicious.md'))
    walked = _walk(html)

    # The legitimate part is there.
    assert 'Text vor der Figur.' in html and 'Text nach der Figur.' in html
    assert '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 80">' in html
    assert walked.tags.count('rect') == 2
    assert 'Legitimes Label</text>' in html
    assert 'Link-Label' in html  # the <a> falls, its label stays
    assert 'Ein div mit Tracking-Hintergrund.' in html
    assert 'Ende des Fixtures.' in html

    # Nothing executable, nothing embedding, nothing referencing.
    for tag in ('script', 'style', 'image', 'use', 'foreignobject', 'animate',
                'iframe'):
        assert tag not in walked.tags, tag
    assert not [a for a in walked.attrs if a[1].startswith('on')]
    assert not [a for a in walked.attrs if a[1] in ('class', 'style')]
    hrefs = [value for _tag, name, value in walked.attrs if name == 'href']
    assert not hrefs, hrefs  # javascript:, data:text/html, data:image/svg — all gone

    # data: survives on img@src with an image type, nowhere else.
    data_attrs = [(tag, name) for tag, name, value in walked.attrs
                  if value and 'data:' in value]
    assert data_attrs == [('img', 'src')] * 2
    assert 'src="data:image/png;base64,iVBORw0KGgo="' in html

    # The host shows up ONLY as inert text — escaped payloads of things that
    # fell, and the alt text that quotes SVG source. Never in an attribute a
    # browser would act on.
    assert not [a for a in walked.attrs
                if a[1] != 'alt' and a[2] and 'evil.example' in a[2]]


def test_svg_inside_an_attribute_value_is_not_a_figure():
    # The span scanner is tag-aware: alt text that quotes SVG source, and the
    # common hand-written unencoded SVG data URI, stay whole attributes.
    html = render_markdown_to_html(
        '<img alt="<svg viewBox=\'0 0 1 1\'><rect width=\'1\' height=\'1\' '
        'fill=\'x onerror=alert(1) y\'/></svg>" '
        'src="data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\'></svg>">')
    walked = _walk(html)
    assert walked.tags == ['img']
    assert [a[1] for a in walked.attrs] == ['alt', 'src', 'loading', 'referrerpolicy']
    assert 'media-placeholder' not in html


def test_slot_string_survives_nh3_only_in_element_content():
    """SENTINEL for the re-insert. The scanner is a text scan and can be
    fooled into cutting an ``<svg>`` out of a place where no figure may stand.
    A sanitized figure carries its own double quotes: put back inside
    ``alt="…"`` it would close that attribute early and turn a hostile
    ``fill`` value into attributes of the host tag. So the slot is an ELEMENT
    with a ``"`` in its serialization, and it is only filled where its exact
    string survives the main pass — which must be element content and nothing
    else. If an nh3 bump stops escaping ``"`` in attribute values or ``<`` in
    text, this goes red before the reader does."""
    slot = '<span class="svgslot-0123456789abcdef-0"></span>'
    clean = lambda html: markdown_render._clean_markdown_html(html, '0123456789abcdef')  # noqa: E731
    assert slot in clean(f'<p>vor {slot} nach</p>')
    assert slot in clean(f'<figure>{slot}<figcaption>Abb.</figcaption></figure>')
    for hostile in (
        f"<img alt='{slot}' src='https://x.example/a.png'>",
        f"<p title='{slot}'>x</p>",
        f'<textarea>{slot}</textarea>',
        f'<title>{slot}</title>',
        f'<xmp>{slot}</xmp>',
        f'<!-- {slot} -->',
        f'<script>{slot}</script>',
        f'<style>{slot}</style>',
    ):
        assert slot not in clean(hostile), hostile


def test_figure_cut_out_of_rcdata_is_dropped_without_debris():
    # <textarea> content is text to the HTML parser but markup to the scanner.
    html = render_markdown_to_html(
        '<textarea><svg viewBox="0 0 1 1"><rect width="1" height="1" '
        'fill="x onerror=alert(1) y"/></svg></textarea>\n\nDanach.')
    walked = _walk(html)
    assert not [a for a in walked.attrs if a[1].startswith('on')]
    assert 'svgslot' not in html and '<svg' not in html
    assert '<p>Danach.</p>' in html


def test_no_marker_or_nonce_leaks_and_output_is_stable():
    text = ('![a](https://x.example/a.png)\n\n'
            '<svg viewBox="0 0 10 10"><rect x="1" width="5" height="5"/></svg>\n')
    first = render_markdown_to_html(text)
    assert 'svgslot-' not in first and 'lazy-' not in first
    assert all(render_markdown_to_html(text) == first for _ in range(10))


# --- data: only on img@src, only image types --------------------------------


@pytest.mark.parametrize('mime', ['svg+xml', 'png', 'jpeg', 'webp', 'gif'])
def test_the_five_image_types_pass_on_img_src(mime):
    uri = f'data:image/{mime};base64,AAAA'
    assert f'src="{uri}"' in render_markdown_to_html(f'![x]({uri})')
    assert f'src="{uri}"' in render_markdown_to_html(f'<img alt="x" src="{uri}">')


def test_parameterless_comma_form_of_an_svg_data_uri_passes():
    uri = 'data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%2F%3E'
    assert f'<img src="{uri}"' in render_markdown_to_html(f'![x]({uri})')


@pytest.mark.parametrize('uri', [
    'data:text/html,<b>x</b>',
    'data:image/bmp;base64,AAAA',
    'data:application/pdf;base64,AAAA',
    'data:image/svg+xmlx,oops',
])
def test_other_data_types_fall_on_img_src(uri):
    html = render_markdown_to_html(f'<img alt="x" src="{uri}">')
    assert 'data:' not in html
    assert '<img alt="x"' in html


def test_data_trap_anchor_href_never_carries_data():
    # url_schemes is global in nh3: opening `data` for img opens it for <a>.
    # The attribute filter closes it again — also for an image type, also for
    # the Markdown link form, also with a tab hidden in the scheme.
    for source in (
        '<a href="data:text/html,<script>alert(1)</script>">x</a>',
        '<a href="data:image/png;base64,AAAA">x</a>',
        '[x](data:image/png;base64,AAAA)',
        '[x](data:image/svg+xml,%3Csvg%2F%3E)',
        '<a href="da&#9;ta:text/html,x">x</a>',
        '<a href="  DATA:text/html,x">x</a>',
    ):
        walked = _walk(render_markdown_to_html(source))
        assert [a for a in walked.attrs if a[1] == 'href'] == [], source


def test_url_scheme_list_is_the_nh3_default_plus_data():
    """SENTINEL. Passing ``url_schemes`` REPLACES nh3's default; the pin
    (0.2.18) does not export it, so the renderer restates it. A scheme missing
    there would silently strip links from the existing library."""
    schemes = markdown_render._URL_SCHEMES - {'data'}
    default = getattr(nh3, 'ALLOWED_URL_SCHEMES', None)
    if default is not None:
        assert schemes == set(default)
    for scheme in schemes:
        assert f'href="{scheme}:x"' in nh3.clean(f'<a href="{scheme}:x">a</a>'), scheme


# --- style on HTML tags may not fetch ----------------------------------------


@pytest.mark.parametrize('style', [
    'background:url(https://evil.example/t.png)',
    'background: URL( "https://evil.example/t.png" )',
    'background:\\75rl(https://evil.example/t.png)',
    'background-image:image-set("https://evil.example/t.png" 1x)',
    'list-style:image("https://evil.example/t.png")',
])
def test_style_that_can_name_a_resource_falls(style):
    html = render_markdown_to_html(f'<div style=\'{style}\'>Text bleibt.</div>')
    assert 'evil.example' not in html
    assert '<div>Text bleibt.</div>' in html


def test_honest_inline_styles_survive_including_pygments():
    html = render_markdown_to_html(
        '<span style="color: #c00; font-weight: bold">rot</span>\n\n'
        '```python\nx = 1  # Kommentar\n```\n')
    assert '<span style="color: #c00; font-weight: bold">rot</span>' in html
    assert 'style="background: #f8f8f8"' in html  # pygments wrapper


# --- https images: lazy, no referrer -----------------------------------------


def test_https_images_carry_lazy_and_no_referrer_in_fixed_order():
    for source in ('![a](https://x.example/a.png)',
                   '<img alt="a" src="https://x.example/a.png">'):
        html = render_markdown_to_html(source)
        assert ' loading="lazy" referrerpolicy="no-referrer">' in html, source
        assert html.count('loading=') == 1 and html.count('referrerpolicy=') == 1


def test_author_cannot_override_the_forced_image_attributes():
    html = render_markdown_to_html(
        '<img alt="a" src="https://x.example/a.png" loading="eager" '
        'referrerpolicy="unsafe-url">')
    assert 'eager' not in html and 'unsafe-url' not in html
    assert ' loading="lazy" referrerpolicy="no-referrer">' in html


def test_prose_about_lazy_loading_is_not_rewritten():
    text = 'Schreib `<img loading="lazy">` — oder loading="lazy-abc" im Text.'
    assert render_markdown_to_html(text) == _old_renderer(text)


# --- viewBox, placeholder ----------------------------------------------------


def test_viewbox_is_derived_from_width_and_height():
    html = render_markdown_to_html(
        '<svg width="300" height="120"><rect x="1" width="5" height="5"/></svg>')
    assert '<svg width="300" height="120" viewBox="0 0 300 120">' in html


def test_svg_that_sanitizes_to_nothing_leaves_a_visible_placeholder():
    # Before: silently swallowed. Now: selectable text with the reason.
    html = render_markdown_to_html(
        'Davor.\n\n<svg viewBox="0 0 1 1"><script>alert(1)</script>'
        '<image href="https://evil.example/t.png"/></svg>\n\nDanach.')
    assert ('<span class="media-placeholder">[Abbildung nicht darstellbar: nach der '
            'Sicherheitsprüfung bleibt kein zeichenbares Element übrig.]</span>') in html
    assert '<svg' not in html and 'evil.example' not in html
    assert '<p>Davor.</p>' in html and '<p>Danach.</p>' in html


def test_unclosed_svg_gets_a_placeholder_and_does_not_eat_the_document():
    html = render_markdown_to_html(
        'Davor.\n\n<svg viewBox="0 0 10 10">\n<rect x="1" width="5" height="5"/>\n\n'
        '## Nächste Überschrift\n\nDanach.')
    assert 'das SVG ist nicht geschlossen (&lt;/svg&gt; fehlt).' in html
    assert '<h2>Nächste Überschrift</h2>' in html
    assert '<p>Danach.</p>' in html
    assert '<svg' not in html


# --- the CommonMark trap: blank lines, pretty-printed SVG --------------------

_PRETTY_SVG = '''<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 100 60"><title>Pretty</title>

  <g stroke="#222">

    <path d="M10 10 L90 10"/>
    <text x="50" y="40" text-anchor="middle">Label A</text>

  </g>

  <svg x="0" y="0"><circle cx="5" cy="5" r="2"/></svg>

</svg>'''


def test_pretty_printed_block_svg_with_blank_lines_renders_as_one_figure():
    """CommonMark reads a lone ``<svg>`` line as an HTML block Typ 7, which
    ENDS at the first blank line — the inner lines became paragraphs full of
    ``<path>`` text. The svg block rule reads it like a Typ-1 block: up to the
    line carrying its (depth-matched) ``</svg>``. Multi-line open tag, first
    child on the tag line and a nested ``<svg>`` included."""
    html = render_markdown_to_html(f'Davor.\n\n{_PRETTY_SVG}\n\nDanach.')
    walked = _walk(html)
    assert walked.tags.count('svg') == 2
    assert walked.tags.count('p') == 2  # Davor. / Danach. — nothing in between
    assert '<path d="M10 10 L90 10"></path>' in html
    assert '<text x="50" y="40" text-anchor="middle">Label A</text>' in html
    assert 'media-placeholder' not in html
    assert '<p>Davor.</p>' in html and '<p>Danach.</p>' in html


def test_block_svg_may_interrupt_a_paragraph():
    html = render_markdown_to_html(
        'Satz direkt davor.\n<svg viewBox="0 0 10 10">\n<rect x="1" width="5" height="5"/>\n</svg>\nSatz direkt danach.')
    assert '<p>Satz direkt davor.</p>' in html
    assert '<rect x="1" width="5" height="5"></rect>' in html
    assert '<br' not in html[html.index('<svg'):html.index('</svg>')]


def test_midline_multiline_svg_survives_the_br_of_breaks_true():
    # Not at the start of a line → paragraph content; `breaks: True` writes a
    # <br> per newline, and <br> makes the HTML parser leave the SVG.
    html = render_markdown_to_html(
        'Skizze: <svg viewBox="0 0 10 10">\n<rect x="1" width="5" height="5"/>\n'
        '<text x="1" y="9">L</text>\n</svg> Ende.')
    svg = html[html.index('<svg'):html.index('</svg>')]
    assert '<rect x="1" width="5" height="5"></rect>' in svg
    assert '<text x="1" y="9">L</text>' in svg
    assert '<br' not in svg


# nh3 0.3.x drops a disallowed SVG-named element WITH its text; the repo pin
# (0.2.18) strips the tag and keeps the text, as the pre-sprint renderer did.
# Probed, not version-compared.
_NH3_KEEPS_STRIPPED_SVG_TEXT = 'x' in nh3.clean('<p><text>x</text></p>', tags={'p'})


def test_midline_svg_with_a_blank_line_is_explained_and_costs_no_text():
    """The one blank-line case the block rule cannot take: markdown-it has cut
    the figure into paragraphs. There is no figure to be had — the placeholder
    says why, and the run stays for the main pass, where the tags fall."""
    html = render_markdown_to_html(
        'Skizze: <svg viewBox="0 0 10 10">\n<rect x="1" width="5" height="5"/>\n\n'
        '<text x="1" y="9">Verwaistes Label</text>\n</svg>\n\nDanach.')
    assert 'das SVG enthält eine Leerzeile, die Markdown als Absatzgrenze liest.' in html
    assert html.index('Skizze:') < html.index('media-placeholder')
    assert '<svg' not in html and '<rect' not in html
    assert '<p>Danach.</p>' in html
    if _NH3_KEEPS_STRIPPED_SVG_TEXT:
        assert 'Verwaistes Label' in html  # loose text, exactly as before the sprint


# --- ein Figur-Fehler kostet die Figur, nie den Text um sie herum ------------


def test_prose_between_an_svg_tag_and_its_closer_is_not_hidden():
    """REGRESSION (Master, nach Phase 2). An "island" is whatever lies between
    an ``<svg`` and the next ``</svg>`` — in a text ABOUT SVG that is prose.
    The first version replaced the failed island whole: the middle paragraph
    vanished behind the placeholder, where the old renderer had merely
    stripped two tags. Backticks in the authoring convention do not replace
    this: the next document about SVG will forget them."""
    html = render_markdown_to_html(
        'Das Tag <svg> öffnet die Figur.\n\n'
        'WICHTIGER ABSATZ dazwischen.\n\n'
        'Und </svg> schließt sie. Ende.')
    assert '<p>WICHTIGER ABSATZ dazwischen.</p>' in html
    assert 'öffnet die Figur.' in html
    assert 'schließt sie. Ende.' in html
    assert html.count('media-placeholder') == 1  # and it still says what happened
    assert html.index('Das Tag') < html.index('media-placeholder') < html.index('öffnet die Figur.')
    assert '<svg' not in html


def test_prose_after_an_unclosed_svg_tag_is_not_hidden():
    html = render_markdown_to_html(
        'Das Tag <svg> öffnet die Figur.\n\n'
        'WICHTIGER ABSATZ dazwischen.\n\n'
        'Ende ohne Schließer.')
    assert 'das SVG ist nicht geschlossen' in html
    for prose in ('öffnet die Figur.', '<p>WICHTIGER ABSATZ dazwischen.</p>',
                  '<p>Ende ohne Schließer.</p>'):
        assert prose in html, prose
    assert '<svg' not in html


def test_prose_inside_an_inkless_one_line_svg_is_not_hidden():
    # Both tags on one line: a closed island without a drawable element. The
    # words between the tags are the author's text, not the figure's ink.
    html = render_markdown_to_html(
        'Das Tag <svg> öffnet und </svg> schließt die Figur.\n\n'
        'WICHTIGER ABSATZ danach.')
    assert 'bleibt kein zeichenbares Element übrig.' in html
    for prose in ('öffnet und', 'schließt die Figur.', '<p>WICHTIGER ABSATZ danach.</p>'):
        assert prose in html, prose
    assert '<svg' not in html


def test_a_failed_figure_left_in_is_still_sanitized_by_the_main_pass():
    # "Left in" is not "let through": the run meets the Markdown allow-list.
    html = render_markdown_to_html(
        'Davor.\n\n<svg viewBox="0 0 1 1" onload="alert(1)"><script>alert(2)</script>'
        '<image href="https://evil.example/t.png"/>'
        '<a href="javascript:alert(3)">Linktext bleibt</a></svg>\n\nDanach.')
    walked = _walk(html)
    assert 'media-placeholder' in html
    if _NH3_KEEPS_STRIPPED_SVG_TEXT:
        assert 'Linktext bleibt' in html  # the <a> falls, its text stays
    assert not [a for a in walked.attrs if a[1].startswith('on') or a[1] == 'href']
    for tag in ('svg', 'script', 'image', 'img'):
        assert tag not in walked.tags, tag
    assert 'evil.example' not in html and 'alert' not in html
    assert '<p>Davor.</p>' in html and '<p>Danach.</p>' in html


def test_svg_shown_as_code_stays_code():
    text = ('Inline `<svg viewBox="0 0 1 1"></svg>` und als Block:\n\n'
            '```html\n<svg viewBox="0 0 10 10">\n\n<rect x="1"/>\n</svg>\n```\n\n'
            '    <svg viewBox="0 0 1 1"></svg>\n')
    html = render_markdown_to_html(text)
    assert html == _old_renderer(text)
    assert '<svg' not in html and 'media-placeholder' not in html


# --- runtime on pathological input (Pflicht-Nachtrag nach Phase 1) -----------

# The svg block rule is a paragraph TERMINATOR: markdown-it asks it again for
# every line of a paragraph. Its first version scanned forward per ask —
# 4 000 lines cost 5.7 s (form A) / 10.7 s (form B), ×4 per doubling, in a
# synchronous WSGI thread on every reader open, PDF and EPUB alike. Measured
# after the fix: 47 ms / 77 ms, the plain-text control 30 ms. The bound leaves
# a slow machine ~20× headroom and a quadratic regression none.
_RUNTIME_LINES = 4000
_RUNTIME_BOUND_S = 1.5

_PATHOLOGICAL = {
    # A — never closed. A "remember the last </svg>" memo fixes THIS one …
    'A_never_closed': '<svg \n' * _RUNTIME_LINES,
    # B — … and not this: closers everywhere, the depth still never returns.
    'B_openers_outnumber_closers': '<svg><svg></svg>\n' * _RUNTIME_LINES,
    'C_tag_closed_element_open': '<svg>\n' * _RUNTIME_LINES,
    # Containers take the bounded-slice path (state.level > 0).
    'G_in_a_blockquote': '> <svg \n' * _RUNTIME_LINES,
    'H_list_openers_then_closers': ('- x\n' + '  <svg>\n' * _RUNTIME_LINES
                                    + '\n' + '</svg>\n' * _RUNTIME_LINES),
    'K_control_plain_text': 'Zeile mit Text.\n' * _RUNTIME_LINES,
}


@pytest.mark.parametrize('form', sorted(_PATHOLOGICAL))
def test_render_time_is_bounded_on_pathological_input(form):
    started = time.perf_counter()
    html = render_markdown_to_html(_PATHOLOGICAL[form])
    elapsed = time.perf_counter() - started
    assert elapsed < _RUNTIME_BOUND_S, f'{form}: {elapsed:.2f}s'
    assert html  # it answered with something, not with nothing


def test_figure_inside_containers_is_still_one_figure():
    quoted = render_markdown_to_html(
        '> Zitat davor.\n>\n> <svg viewBox="0 0 10 10">\n>\n'
        '>   <rect x="1" width="5" height="5"/>\n> </svg>\n>\n> Zitat danach.')
    assert _walk(quoted).tags.count('svg') == 1
    assert '<rect x="1" width="5" height="5"></rect>' in quoted
    assert 'media-placeholder' not in quoted

    listed = render_markdown_to_html(
        '- Punkt\n\n  <svg viewBox="0 0 10 10">\n\n'
        '    <rect x="1" width="5" height="5"/>\n  </svg>\n\n- Nächster Punkt')
    assert _walk(listed).tags.count('svg') == 1
    assert '<rect x="1" width="5" height="5"></rect>' in listed
    assert listed.count('<li>') == 2


def test_block_rule_declines_when_the_container_ends_before_the_figure():
    # The </svg> exists — but outside the list item. Taking the block would
    # pull the following top-level lines into the item. Asserted on the
    # markdown-it output: that is where the rule decides.
    rendered = markdown_render._md.render(
        '- Punkt\n  <svg viewBox="0 0 10 10">\n  <rect x="1" width="5" height="5"/>\n\n'
        'Absatz außerhalb der Liste.\n\n</svg>\n\nEnde.')
    assert rendered.index('</ul>') < rendered.index('<p>Absatz außerhalb der Liste.</p>')
    assert '<p>Ende.</p>' in rendered


def test_nested_figure_beyond_the_line_bound_falls_back_to_stock_commonmark():
    # No blank line inside → Stock-CommonMark (Typ 7) still yields one raw
    # block and the island logic renders it: the bound costs nothing here.
    body = '> <rect x="1" width="5" height="5"/>\n' * (markdown_render._SVG_BLOCK_MAX_NESTED_LINES + 5)
    html = render_markdown_to_html('> <svg viewBox="0 0 10 10">\n' + body + '> </svg>\n')
    walked = _walk(html)
    assert walked.tags.count('svg') == 1
    assert walked.tags.count('rect') == markdown_render._SVG_BLOCK_MAX_NESTED_LINES + 5


# --- the existing library does not move (pytest-sized Gate 6) ----------------


@pytest.mark.parametrize('text', [
    _fixture('sample.md'),
    '# Titel\n\nText mit **fett**, `code`, [Link](https://example.org) und '
    '[Mail](mailto:a@example.org).\n\n| a | b |\n|---|---|\n| 1 | 2 |\n',
    'Platzhalter in Prosa: <uuid>, <api-pod> und <id> verschwinden wie bisher.',
    'Mathe $a_1 + b$ und\n\n$$\\frac{1}{2}$$\n\nPreis 5$ bleibt Text.',
    '<details><summary>Mehr</summary>\n\nInhalt <mark>markiert</mark>.\n\n</details>',
    '<div class="callout" id="c1" style="color: red">Hinweis</div>\n\n<!-- Kommentar -->',
    '[relativ](Kapitel-8/abb.md) [anker](#ziel) [js](javascript:alert(1)) <a href="ftp://x.example/f">ftp</a>',
    '```mermaid\nflowchart LR\n  A --> B\n```\n',
    '> Zitat\n> <b>fett</b>\n\n1. eins\n2. zwei\n   - innen\n',
])
def test_documents_without_media_render_byte_identical(text):
    assert render_markdown_to_html(text) == _old_renderer(text)
