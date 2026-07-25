# services/svg_sanitize.py
"""Sanitizer for agent-authored card figures (Sprint CARD-SVG).

Cards may carry an SVG figure on front and back (``Card.front_svg`` /
``Card.back_svg``), written exclusively by the external card agent via
``CARD_TOKEN``. House style: the SVG is stored *raw* and sanitized on every
read (``Card.to_dict``); the write path additionally calls this function to
reject inputs that would sanitize to nothing (400 with a reason instead of a
silently empty figure).

Pure ``str -> str`` module — no Flask, no SDK (Vorbild ``services/epub_math.py``
/ ``services/markdown_sections.py``). It deliberately does NOT import from
``app_pkg/markdown_render.py``: the shared Markdown allow-list serves
PDF/EPUB/Library/Reader and knows no SVG; this one serves exactly the two
figure containers in the review UI. Own list, zero blast radius — and unlike
the Markdown list there is NO ``'*'`` wildcard entry and NO ``style``
attribute here.

Security doctrine (Master-Probe 2026-07-25, verified against nh3==0.2.18 —
the repo pin — and 0.3.5): ammonia/html5ever runs the SVG foreign-content
attribute adjustment, so camelCase attributes (``viewBox``,
``preserveAspectRatio``, ``markerWidth``, ``refX``, ``gradientUnits``) survive
``nh3.clean`` exactly. Dangerous constructs fall **by omission from the
allow-list**, not by special-casing — the never-include list below is the
actual security boundary. ``tests/test_svg_sanitize.py`` carries a sentinel
test that fires if an nh3 bump ever loses the camelCase preservation.

NEVER include (each falls today because it is absent — keep it that way):
  - ``script``          — direct code execution
  - ``style`` (tag AND attribute) — CSS can pull external resources via
                          ``url()`` and leak/track; also style injection
  - ``foreignObject``    — smuggles arbitrary HTML (iframe/script/img@onerror)
                          back into the HTML parsing context
  - ``use``              — external document references (``href``)
  - ``image``            — external loads = LAN egress/tracking pixel
  - ``a``                — ``javascript:`` / ``xlink:href`` vectors
  - ``animate``/``set``  — ``attributeName="href"`` can rewrite an allowed
                          attribute into a reference at runtime
  - ``iframe``/``audio``/``video`` — foreign embedding / network loads
"""
import re

import nh3

# Hard byte cap for a single figure (measured on the utf-8 encoded raw input).
# Keeps a runaway agent from parking megabytes in a Text column; the authoring
# convention (docs/card_svg_authoring.md) names the same number.
MAX_CARD_SVG_BYTES = 100_000

# Presentation attributes shared by the drawing tags. ``font-weight`` goes
# beyond the sprint's start list: labeled boxes are the core use case and bold
# emphasis is pure presentation with no reference semantics. The three
# ``marker-*`` attributes are what makes the allowed ``<marker>`` tag reachable
# at all (arrowheads); their ``url(...)`` values are constrained to local
# fragments by ``_filter_attribute`` below.
_PRESENTATION = {
    'fill', 'stroke', 'stroke-width', 'stroke-linecap', 'stroke-dasharray',
    'opacity', 'transform', 'font-size', 'font-family', 'font-weight',
    'text-anchor', 'dominant-baseline',
    'marker-start', 'marker-mid', 'marker-end',
}

_ALLOWED_TAGS = {
    'svg', 'g', 'defs', 'title', 'desc',
    'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon',
    'text', 'tspan',
    'marker', 'linearGradient', 'radialGradient', 'stop',
}

# Per-tag only — deliberately no ``'*'`` entry. ``id`` exists solely on the
# three referenceable defs (marker/gradients) so ``url(#...)`` targets work;
# shapes get none. No ``class`` anywhere: the agent cannot ship CSS (``style``
# tag and attribute are both banned) and no app stylesheet targets
# agent-chosen classes — ``class`` grants zero capability at full collision
# surface with the app utilities (``class="hidden"`` would meet the exact
# class review.js hides the figure containers with → invisible figure with no
# findable cause; Präzedenz feedback_css_class_collision_in_markdown_views).
_ALLOWED_ATTRIBUTES = {
    'svg': {'viewBox', 'width', 'height', 'xmlns', 'preserveAspectRatio'},
    'g': _PRESENTATION,
    'path': {'d'} | _PRESENTATION,
    'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'} | _PRESENTATION,
    'circle': {'cx', 'cy', 'r'} | _PRESENTATION,
    'ellipse': {'cx', 'cy', 'rx', 'ry'} | _PRESENTATION,
    'line': {'x1', 'y1', 'x2', 'y2'} | _PRESENTATION,
    'polyline': {'points'} | _PRESENTATION,
    'polygon': {'points'} | _PRESENTATION,
    'text': {'x', 'y', 'dx', 'dy'} | _PRESENTATION,
    'tspan': {'x', 'y', 'dx', 'dy'} | _PRESENTATION,
    'marker': {'id', 'viewBox', 'markerWidth', 'markerHeight', 'refX', 'refY',
               'orient', 'markerUnits'} | _PRESENTATION,
    'linearGradient': {'id', 'x1', 'y1', 'x2', 'y2', 'gradientUnits',
                       'gradientTransform'},
    'radialGradient': {'id', 'cx', 'cy', 'r', 'fx', 'fy', 'gradientUnits',
                       'gradientTransform'},
    'stop': {'offset', 'stop-color', 'stop-opacity'},
}

# ``fill``/``stroke``/``marker-*`` accept ``url(...)`` values that browsers may
# resolve as resource references. Local fragments (``url(#id)``, optionally
# quoted) are required for gradients/markers; anything else (external paint
# servers, ``url(https://...)``) is a network load and gets dropped — same
# doctrine as banning ``image``/``use``. EVERY ``url(`` occurrence in the
# value must be a local fragment, or the whole attribute falls — a mixed
# ``url(#g) url(https://...)`` fallback list dies too, so the "no external
# references" guarantee holds without asterisks.
_URL_VALUED = {'fill', 'stroke', 'marker-start', 'marker-mid', 'marker-end'}
_URL_TOKEN = re.compile(r"url\(", re.IGNORECASE)
_LOCAL_URL = re.compile(r"url\(\s*['\"]?#", re.IGNORECASE)


def _filter_attribute(element: str, attribute: str, value: str):
    if attribute in _URL_VALUED:
        n_urls = len(_URL_TOKEN.findall(value))
        if n_urls and len(_LOCAL_URL.findall(value)) != n_urls:
            return None  # drop the attribute entirely
    return value


def sanitize_card_svg(raw: str) -> str:
    """Bereinigt agent-geschriebenes SVG auf eine enge Allow-List.

    Gibt '' zurück, wenn nichts Renderbares übrig bleibt (kein String, leer,
    über MAX_CARD_SVG_BYTES, oder nach dem Clean ohne ``<svg``-Wurzel).
    """
    # isinstance before anything else: a truthy non-string must yield '' and
    # never a 500 (Präzedenz reference_strict_bool_isinstance_destructive_writes).
    if not isinstance(raw, str) or not raw.strip():
        return ''
    if len(raw.encode('utf-8')) > MAX_CARD_SVG_BYTES:
        return ''
    cleaned = nh3.clean(
        raw,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        attribute_filter=_filter_attribute,
    )
    # No <svg> root left (e.g. plain HTML input reduced to its text) → nothing
    # renderable; don't let bare fragments through. html5ever lowercases tag
    # names, so the lowercase check is exact.
    if '<svg' not in cleaned:
        return ''
    return cleaned
