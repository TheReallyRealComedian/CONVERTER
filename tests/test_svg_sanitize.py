"""CARD-SVG Phase 1 — the card-figure SVG sanitizer.

Pure unit tests for ``services.svg_sanitize.sanitize_card_svg``. The twelve
Master probe vectors (2026-07-25, verified against the repo pin nh3==0.2.18)
live here as regression tests: three preservation cases (camelCase survives
the SVG foreign-content adjustment), eight attack vectors that must fall by
*omission* from the allow-list, and one realistic diagram that must pass
intact. Plus input guards (non-string / cap / non-SVG) and the local-only
``url(...)`` value filter.

Doctrine note: dangerous constructs are dropped because they are ABSENT from
the allow-list, not by special-casing — if one of the "falls" tests ever goes
red, someone widened the list.
"""
from services.svg_sanitize import MAX_CARD_SVG_BYTES, sanitize_card_svg


# --- Preservation: camelCase survives nh3 (Probe 1-3) ----------------------


def test_sentinel_viewbox_camelcase_survives_nh3_bump():
    """SENTINEL — fires on an nh3/ammonia bump that loses SVG camelCase.

    The entire design rests on ammonia/html5ever running the SVG
    foreign-content attribute adjustment: ``viewBox`` must come back exactly
    camelCased, because a lowercased ``viewbox`` is dead — nothing scales.
    Same doctrine as the Flask-WTF sentinels in ``tests/test_csrf_inversion.py``:
    when this goes red after a dependency bump, re-verify the whole probe table
    in the module docstring before shipping.
    """
    out = sanitize_card_svg(
        '<svg viewBox="0 0 100 50" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="1" y="1" width="10" height="10"/></svg>'
    )
    assert 'viewBox="0 0 100 50"' in out
    assert 'viewbox' not in out  # the lowercased corpse must not appear


def test_preserveaspectratio_and_marker_attrs_survive():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10" preserveAspectRatio="xMidYMid meet">'
        '<marker id="m" markerWidth="5" markerHeight="5" refX="2" refY="2">'
        '<path d="M0 0L5 2"/></marker></svg>'
    )
    assert 'preserveAspectRatio="xMidYMid meet"' in out
    assert 'markerWidth="5"' in out
    assert 'refX="2"' in out


def test_lineargradient_tag_and_gradientunits_survive():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10">'
        '<linearGradient id="lg" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#123"/></linearGradient></svg>'
    )
    assert '<linearGradient' in out
    assert 'gradientUnits="userSpaceOnUse"' in out
    assert 'stop-color="#123"' in out


# --- The eight attack vectors fall (Probe 4-11) -----------------------------


def test_script_tag_falls():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10"><script>alert(1)</script><rect x="1"/></svg>'
    )
    assert 'script' not in out.lower()
    assert 'alert' not in out
    assert '<rect' in out  # the harmless sibling survives


def test_onload_onclick_handlers_fall():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10" onload="evil()"><rect x="1" onclick="evil()"/></svg>'
    )
    assert 'onload' not in out
    assert 'onclick' not in out
    assert 'evil' not in out
    assert '<rect' in out


def test_foreignobject_falls_including_onerror_payload():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10"><foreignObject>'
        '<img src="x" onerror="evil()"></foreignObject></svg>'
    )
    assert 'foreignobject' not in out.lower()
    assert 'onerror' not in out
    assert '<img' not in out


def test_anchor_with_javascript_href_falls():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10"><a xlink:href="javascript:evil()">'
        '<rect x="1"/></a></svg>'
    )
    assert '<a' not in out.replace('<animate', '')  # no anchor tag
    assert 'javascript:' not in out
    assert '<rect' in out  # content of the stripped <a> survives


def test_use_with_external_ref_falls():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10"><use href="https://evil.example/x.svg#p"/></svg>'
    )
    assert '<use' not in out
    assert 'evil.example' not in out


def test_animate_attributename_href_falls():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10"><animate attributeName="href" to="javascript:evil()"/></svg>'
    )
    assert 'animate' not in out.lower()
    assert 'javascript:' not in out


def test_image_with_external_href_falls():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10"><image href="https://evil.example/t.png"/></svg>'
    )
    assert '<image' not in out
    assert 'evil.example' not in out


def test_style_tag_falls():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10"><style>rect{fill:red}</style><rect x="1"/></svg>'
    )
    assert '<style' not in out
    assert 'fill:red' not in out
    assert '<rect' in out


# --- Probe 12: a realistic diagram passes intact ----------------------------


def test_realistic_diagram_passes_intact():
    raw = (
        '<svg viewBox="0 0 320 180" xmlns="http://www.w3.org/2000/svg">'
        '<g transform="translate(10,10)" font-family="sans-serif">'
        '<rect x="0" y="0" width="120" height="40" rx="6" fill="#e8f0fe" '
        'stroke="#1a56b0" stroke-width="2"/>'
        '<text x="60" y="25" text-anchor="middle" font-size="14" fill="#111">Protein A</text>'
        '<path d="M120 20 H 180" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>'
        '</g></svg>'
    )
    out = sanitize_card_svg(raw)
    for fragment in (
        'viewBox="0 0 320 180"',
        'transform="translate(10,10)"',
        'rx="6"',
        'fill="#e8f0fe"',
        'text-anchor="middle"',
        '>Protein A</text>',
        'd="M120 20 H 180"',
        'marker-end="url(#arrow)"',
    ):
        assert fragment in out


# --- Input guards -----------------------------------------------------------


def test_none_empty_and_non_string_inputs_yield_empty():
    assert sanitize_card_svg(None) == ''
    assert sanitize_card_svg('') == ''
    assert sanitize_card_svg('   \n  ') == ''
    # truthy non-strings must yield '' and never raise
    # (Präzedenz reference_strict_bool_isinstance_destructive_writes)
    assert sanitize_card_svg(123) == ''
    assert sanitize_card_svg(True) == ''
    assert sanitize_card_svg(['<svg/>']) == ''
    assert sanitize_card_svg({'svg': 1}) == ''


def test_over_byte_cap_yields_empty():
    filler = '<rect x="1"/>' * (MAX_CARD_SVG_BYTES // 10)
    raw = f'<svg viewBox="0 0 10 10">{filler}</svg>'
    assert len(raw.encode('utf-8')) > MAX_CARD_SVG_BYTES
    assert sanitize_card_svg(raw) == ''


def test_cap_measures_utf8_bytes_not_characters():
    # Multi-byte chars: just under the cap in characters, over it in bytes.
    label = 'ü' * (MAX_CARD_SVG_BYTES // 2)  # 2 bytes each in utf-8
    raw = f'<svg viewBox="0 0 10 10"><text x="1" y="1">{label}</text></svg>'
    assert len(raw) < MAX_CARD_SVG_BYTES
    assert sanitize_card_svg(raw) == ''


def test_non_svg_input_yields_empty():
    # nh3 reduces <p>hi</p> to bare text 'hi' — the <svg>-root guard must
    # turn that into '' instead of letting a naked fragment through.
    assert sanitize_card_svg('<p>hi</p>') == ''
    assert sanitize_card_svg('kein markup') == ''


# --- url() value filter: local fragments only -------------------------------


def test_local_url_fragment_references_survive():
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10">'
        '<rect x="1" fill="url(#lg)" stroke="url( \'#lg2\' )"/></svg>'
    )
    assert 'fill="url(#lg)"' in out
    assert 'stroke=' in out  # quoted/spaced local form survives too


def test_external_url_paint_server_is_dropped():
    # Same doctrine as banning <image>/<use>: no external loads, ever.
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10"><rect x="1" fill="url(https://evil.example/p.svg#g)"/></svg>'
    )
    assert 'evil.example' not in out
    assert 'fill' not in out  # attribute dropped entirely
    assert '<rect' in out  # element itself stays


def test_mixed_local_and_external_url_list_is_dropped():
    # A paint fallback list smuggling an external ref behind a local one must
    # fall as a whole — EVERY url( occurrence has to be a local fragment.
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10">'
        '<rect x="1" fill="url(#g) url(https://evil.example/x)"/></svg>'
    )
    assert 'evil.example' not in out
    assert 'fill' not in out
    assert '<rect' in out


# --- class is banned everywhere (collision with app utilities) --------------


def test_class_attribute_is_stripped_from_svg_root():
    # class grants the agent zero capability (no CSS path) but full collision
    # surface: class="hidden" is the exact class review.js hides the figure
    # containers with — an invisible figure with no findable cause.
    out = sanitize_card_svg(
        '<svg viewBox="0 0 10 10" class="hidden"><rect x="1" width="5" height="5"/></svg>'
    )
    assert 'class' not in out
    assert 'hidden' not in out
    assert 'viewBox="0 0 10 10"' in out  # SVG otherwise intact
    assert '<rect' in out
