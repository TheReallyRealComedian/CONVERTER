# services/svg_sanitize.py
"""THE SVG policy — one allow-list for card figures and document figures.

Sprint CARD-SVG introduced it for the two card figure fields
(``Card.front_svg`` / ``Card.back_svg``, written by the external card agent via
``CARD_TOKEN``): the SVG is stored *raw* and sanitized on every read
(``Card.to_dict``); the write path additionally calls ``sanitize_card_svg`` to
reject inputs that would sanitize to nothing (400 with a reason instead of a
silently empty figure).

Sprint RICH-MEDIA made it the single definition of what ANY agent-authored SVG
may carry: ``app_pkg/markdown_render.py`` cuts inline ``<svg>`` islands out of
the rendered document and sends each through ``sanitize_svg`` — the very same
``nh3`` call the cards use. The import direction is one-way (the renderer
consumes this module, never the reverse), and the Markdown allow-list stays
SVG-free on purpose: that list carries a ``'*': {class, id, style}`` wildcard,
which would hand every SVG element the ``style`` and ``class`` this policy
bans. Here there is NO ``'*'`` entry and NO ``style`` attribute. A second SVG
list anywhere in the repo is a defect, not a variant.

Pure ``str -> str`` module — no Flask, no SDK (Vorbild ``services/epub_math.py``
/ ``services/markdown_sections.py``).

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
  - ``mask``             — (RICH-MEDIA, benannt statt stillschweigend) wäre
                          unter demselben ``url(#…)``-Filter sicher, aber als
                          CSS-Property nimmt ``mask`` auch Bildquellen — eine
                          weitere referenztragende Fläche ohne Anwendungsfall
                          in Strich-Figuren. Fällt durch Abwesenheit.
  - ``filter`` + ``fe*``  — ~25 Elemente, ``feImage`` trägt ``href`` (externer
                          Load), große Blur-Radien sind ein Render-DoS.

``clipPath`` IS in the policy (RICH-MEDIA 1.1): clipping is pure geometry —
the element holds shapes, the reference travels as ``clip-path="url(#id)"``
and meets the very same local-fragment filter as markers and gradients, so it
adds no fetch surface the policy did not already constrain.
"""
import re

import nh3

# Hard byte cap for a single CARD figure (measured on the utf-8 encoded raw
# input). Keeps a runaway agent from parking megabytes in a Text column; the
# authoring convention (docs/card_svg_authoring.md) names the same number.
# Document figures have no per-figure cap here — their budget is enforced at
# the write paths (services/doc_media.py), not at render time.
MAX_CARD_SVG_BYTES = 100_000

# Presentation attributes shared by the drawing tags. ``font-weight`` goes
# beyond the sprint's start list: labeled boxes are the core use case and bold
# emphasis is pure presentation with no reference semantics. The three
# ``marker-*`` attributes are what makes the allowed ``<marker>`` tag reachable
# at all (arrowheads), ``clip-path`` does the same for ``<clipPath>``; their
# ``url(...)`` values are constrained to local fragments by
# ``filter_svg_attribute`` below.
_PRESENTATION = {
    'fill', 'stroke', 'stroke-width', 'stroke-linecap', 'stroke-dasharray',
    'opacity', 'transform', 'font-size', 'font-family', 'font-weight',
    'text-anchor', 'dominant-baseline',
    'marker-start', 'marker-mid', 'marker-end',
    'clip-path', 'clip-rule',
}

SVG_ALLOWED_TAGS = {
    'svg', 'g', 'defs', 'title', 'desc',
    'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon',
    'text', 'tspan',
    'marker', 'linearGradient', 'radialGradient', 'stop',
    'clipPath',
}

# Tags that put ink on the page. A figure that keeps its ``<svg>`` root but none
# of these renders as nothing — ``svg_has_drawable`` lets the document renderer
# say so instead of showing an empty box.
SVG_DRAWABLE_TAGS = {
    'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text',
}

# Per-tag only — deliberately no ``'*'`` entry. ``id`` exists solely on the
# referenceable defs (marker/gradients/clipPath) so ``url(#...)`` targets work;
# shapes get none. No ``class`` anywhere: the agent cannot ship CSS (``style``
# tag and attribute are both banned) and no app stylesheet targets
# agent-chosen classes — ``class`` grants zero capability at full collision
# surface with the app utilities (``class="hidden"`` would meet the exact
# class review.js hides the figure containers with → invisible figure with no
# findable cause; Präzedenz feedback_css_class_collision_in_markdown_views).
# In the reader the same holds: Tailwind utilities are global there too.
#
# The ``svg`` root carries the presentation set since RICH-MEDIA: root-level
# ``font-family``/``font-size``/``fill`` are inherited defaults (Olis Probe
# #240 sets them there). No new capability — the same attributes already work
# one level down on a wrapping ``<g>``.
SVG_ALLOWED_ATTRIBUTES = {
    'svg': {'viewBox', 'width', 'height', 'xmlns', 'preserveAspectRatio'}
           | _PRESENTATION,
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
    'clipPath': {'id', 'clipPathUnits', 'transform'},
}

# ``fill``/``stroke``/``marker-*``/``clip-path`` accept ``url(...)`` values that
# browsers may resolve as resource references. Local fragments (``url(#id)``,
# optionally quoted) are required for gradients/markers/clip paths; anything
# else (external paint servers, ``url(https://...)``) is a network load and
# gets dropped — same doctrine as banning ``image``/``use``. EVERY ``url(``
# occurrence in the value must be a local fragment, or the whole attribute
# falls — a mixed ``url(#g) url(https://...)`` fallback list dies too, so the
# "no external references" guarantee holds without asterisks.
# A backslash kills the attribute outright: presentation attributes are parsed
# as CSS, where ``\75rl(https://…)`` IS ``url(…)`` after escape processing —
# the literal ``url(`` scan would never see it, and no honest figure needs a
# CSS escape in a paint or reference value.
_URL_VALUED = {'fill', 'stroke', 'marker-start', 'marker-mid', 'marker-end',
               'clip-path'}
_URL_TOKEN = re.compile(r"url\(", re.IGNORECASE)
_LOCAL_URL = re.compile(r"url\(\s*['\"]?#", re.IGNORECASE)


def filter_svg_attribute(element: str, attribute: str, value: str):
    if attribute in _URL_VALUED:
        if '\\' in value:
            return None
        n_urls = len(_URL_TOKEN.findall(value))
        if n_urls and len(_LOCAL_URL.findall(value)) != n_urls:
            return None  # drop the attribute entirely
    return value


def sanitize_svg(raw: str, max_bytes=None) -> str:
    """The one ``nh3`` call behind every agent-authored SVG (cards, documents).

    Gibt '' zurück, wenn nichts Renderbares übrig bleibt (kein String, leer,
    über ``max_bytes``, oder nach dem Clean ohne ``<svg``-Wurzel).
    """
    # isinstance before anything else: a truthy non-string must yield '' and
    # never a 500 (Präzedenz reference_strict_bool_isinstance_destructive_writes).
    if not isinstance(raw, str) or not raw.strip():
        return ''
    if max_bytes is not None and len(raw.encode('utf-8')) > max_bytes:
        return ''
    cleaned = nh3.clean(
        raw,
        tags=SVG_ALLOWED_TAGS,
        attributes=SVG_ALLOWED_ATTRIBUTES,
        attribute_filter=filter_svg_attribute,
    )
    # No <svg> root left (e.g. plain HTML input reduced to its text) → nothing
    # renderable; don't let bare fragments through. html5ever lowercases tag
    # names, so the lowercase check is exact.
    if '<svg' not in cleaned:
        return ''
    return cleaned


def sanitize_card_svg(raw: str) -> str:
    """Bereinigt agent-geschriebenes Karten-SVG auf die eine Allow-List.

    Gibt '' zurück, wenn nichts Renderbares übrig bleibt (kein String, leer,
    über MAX_CARD_SVG_BYTES, oder nach dem Clean ohne ``<svg``-Wurzel).
    """
    return sanitize_svg(raw, max_bytes=MAX_CARD_SVG_BYTES)


_DRAWABLE_RE = re.compile(
    r'<(?:' + '|'.join(sorted(SVG_DRAWABLE_TAGS)) + r')[\s>]')


def svg_has_drawable(cleaned: str) -> bool:
    """True if a SANITIZED figure still holds at least one ink-bearing element.

    Operates on ``sanitize_svg`` output only: nh3 serializes every element as
    an explicit ``<tag …>`` with lowercase drawable names. ⚠️ nh3 does NOT
    escape ``<`` inside attribute values, so ``font-family="<rect "`` can fake
    a hit — the cost is an empty box instead of a placeholder, never a safety
    question (this decides a message, not what is allowed)."""
    return bool(_DRAWABLE_RE.search(cleaned))


# nh3 serializes attributes as `` name="value"`` with ``"`` escaped to
# ``&quot;`` — but it leaves ``<`` and ``>`` inside values alone (measured on
# 0.2.18 and 0.3.5). The root tag therefore has to be read QUOTE-AWARE: a lazy
# ``<svg[^>]*>`` would stop at a ``>`` inside ``font-family="a>…"`` and the
# viewBox insertion below would close that attribute early, turning the rest
# of the value into live markup. Never runs on raw author markup.
_ROOT_TAG_RE = re.compile(r'<svg((?:\s+[^\s="<>/]+="[^"]*")*)\s*>')
_ROOT_ATTR_RE = re.compile(r'\s+([^\s="<>/]+)="([^"]*)"')
_PLAIN_LENGTH_RE = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*(?:px)?\s*$')


def ensure_viewbox(cleaned: str) -> str:
    """Give a sanitized figure a ``viewBox`` when it only has ``width``/``height``.

    Without one, the reader's ``max-width: 100%; height: auto`` would CROP a
    figure wider than the column instead of scaling it. Derived server-side,
    once, so reader, PDF and EPUB agree. Only plain numbers / ``px`` lengths
    qualify; ``%``/``em`` roots and roots that already carry a ``viewBox`` are
    returned untouched. Document path only — card figures are not rewritten.
    """
    match = _ROOT_TAG_RE.match(cleaned)
    if match is None:
        return cleaned
    attrs = dict(_ROOT_ATTR_RE.findall(match.group(1)))
    if 'viewBox' in attrs:
        return cleaned
    w = _PLAIN_LENGTH_RE.match(attrs.get('width', ''))
    h = _PLAIN_LENGTH_RE.match(attrs.get('height', ''))
    if not w or not h:
        return cleaned
    root = match.group(0)
    new_root = f'{root[:-1]} viewBox="0 0 {w.group(1)} {h.group(1)}">'
    return new_root + cleaned[match.end():]
