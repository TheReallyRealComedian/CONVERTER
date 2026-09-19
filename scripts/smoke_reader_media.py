#!/usr/bin/env python3
"""Browser smoke for figures in the reader (RICH-MEDIA).

pytest sees the renderer (``tests/test_markdown_media.py``); it renders no
template and runs no JS. This is the REAL-browser half: Playwright INSIDE the
web container against the deployed app, as a throwaway user whose documents
are written through the ORM. It MEASURES the claims instead of looking at
them:

1. Figures document (inline SVG, ``<img data:>``, ``![](data:)``, Mermaid,
   text before/between/after):
   - ``readerRawText`` — the highlight coordinate system — is BYTE-EQUAL
     before and after Mermaid renders (the fence source stays in the DOM, the
     diagram and all UI text live in a shadow root), measured with the CDN
     request held back so "before" really is before;
   - ``readerRawText`` of the document under the PRE-SPRINT renderer (on the
     container's nh3 pin the old renderer left SVG ``<text>`` content behind
     as loose text) equals the live one — whitespace between elements
     included: figures moved no anchor of an existing document;
   - a highlight set by REAL MOUSE DRAG immediately before and immediately
     after each of the four figures (Memory
     ``feedback_selection_anchor_coordinate_system``: synthetic ranges do not
     reproduce a drag) wraps exactly its phrase; after a reload all eight sit
     at the same raw offset; ``/api/highlights/recent`` delivers the exact
     phrases;
   - a highlight INSIDE a Mermaid source (Oli has three on #133): the figure
     carries the marker, ``scrollToHighlight`` opens the source.
2. Malicious fixture as a real document: no request to ``evil.example``, no
   origin the plain baseline document does not also load, no dialog, nothing
   executable left in the reader DOM.
3. Mermaid syntax error: source visible, hint shown, the valid fence next to
   it and the rest of the document rendered, no "Syntax error" graphic parked
   in ``<body>``.
4. Pathological input (4 000 lines of never-closed ``<svg ``; 4 000 lines of
   ``<svg><svg></svg>``): the reader page and the library list answer in
   finite time — the browser half of the time-boxed pytest.
5. Layout: light/dark × 375 px/desktop — no figure wider than the reader
   column, ``currentColor`` follows the theme (screenshots:
   ``SMOKE_OUT_<theme>_<width>.png``).
6. PDF through the real form (vector drawing of the inline SVG in the file)
   and the EPUB build (no exception, the chapter carries the ``<svg>``).

Documents are written through the ORM under the throwaway user's own
``user_id`` — never through ``POST /api/ingest/conversion``, which writes to
the INGEST_USER = Oli's account. The script REFUSES to run for user id 1 or
the INGEST_USER and removes the documents it created (highlights cascade),
strictly by ``user_id`` + the ids it wrote; the user itself is created and
removed by the operator (recipe below).

How to run (Mintbox, ~2 min):

    # 1. throwaway user — NEVER Oli's account
    docker exec markdown-converter-web flask --app app create-user zz_media --password '<random>'
    docker cp scripts/smoke_reader_media.py markdown-converter-web:/tmp/smoke_media.py
    docker cp tests/fixtures/rich_media_malicious.md markdown-converter-web:/tmp/rich_media_malicious.md
    # 2. run — screenshots/PDF land in the container as /tmp/smoke_media_*
    docker exec -e SMOKE_USER=zz_media -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_media.py
    # 3. clean up STRICTLY by user_id (the api_token table carries Oli's iOS
    #    tokens): the script already removed its documents; delete the User row
    #    via the ORM filtered by that user_id, then rm /tmp/smoke_media* and
    #    /tmp/rich_media_malicious.md.

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SMOKE_OUT (/tmp/smoke_media), SMOKE_MALICIOUS (/tmp/rich_media_malicious.md),
SMOKE_APP_ROOT (default: cwd = /app in the container). Exit 0 = every check
passed; every measured value is printed so a failure is diagnosable from the
output alone.
"""
import importlib.metadata
import io
import os
import sys
import time
import zipfile

import fitz  # PyMuPDF — ships in the image (the PDF service uses it)
import nh3
from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
OUT = os.environ.get('SMOKE_OUT', '/tmp/smoke_media')
MALICIOUS_PATH = os.environ.get('SMOKE_MALICIOUS', '/tmp/rich_media_malicious.md')

nh3_version = importlib.metadata.version('nh3')

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


# --- documents ---------------------------------------------------------------
_SVG_URI = ('data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20'
            'width%3D%22900%22%20height%3D%2260%22%3E%3Crect%20width%3D%22900%22%20height%3D%2260%22%20'
            'fill%3D%22%23{fill}%22%2F%3E%3C%2Fsvg%3E')

# The inline SVG is 900 wide WITHOUT a viewBox (the server derives one — without
# it max-width would crop, not scale) and pretty-printed WITH blank lines (the
# CommonMark trap). #336699 is the colour the PDF check looks for.
FIGURES_DOC = f'''# Smoke: Figuren im Dokument

Der erste Absatz steht vor dem Inline-SVG und endet mit Marke A1 Ende

<svg xmlns="http://www.w3.org/2000/svg" width="900" height="200" font-family="sans-serif">

  <rect x="10" y="10" width="880" height="180" fill="#336699" stroke="#222"/>

  <text x="450" y="90" text-anchor="middle" fill="currentColor">SvgLabel Eins</text>
  <text x="450" y="130" text-anchor="middle" fill="#fff">SvgLabel Zwei</text>
</svg>

Marke A2 Anfang steht direkt nach dem Inline-SVG, und dieser Absatz endet mit Marke B1 Ende

<img alt="img mit data-URI" src="{_SVG_URI.format(fill='eeeeff')}">

Marke B2 Anfang steht direkt nach dem img, und dieser Absatz endet mit Marke C1 Ende

![Markdown-Bild mit data-URI]({_SVG_URI.format(fill='eeffee')})

Marke C2 Anfang steht direkt nach dem Markdown-Bild, und dieser Absatz endet mit Marke D1 Ende

```mermaid
flowchart LR
  QuelleA --> QuelleB
  QuelleB --> QuelleC
```

Marke D2 Anfang steht direkt nach dem Mermaid-Diagramm. Schluss des Dokuments.
'''
PHRASES = ['Marke A1 Ende', 'Marke A2 Anfang', 'Marke B1 Ende', 'Marke B2 Anfang',
           'Marke C1 Ende', 'Marke C2 Anfang', 'Marke D1 Ende', 'Marke D2 Anfang']
MERMAID_SOURCE_PHRASE = 'QuelleB --> QuelleC'

PLAIN_DOC = '# Smoke: Baseline\n\nEin Dokument ohne jede Figur.\n'

# The shape of an SVG document that RENDERED under the old renderer (Olis #240:
# tag alone on its line, no blank line inside) plus an inline figure — the
# reference for "did figures move the text nodes of an existing document".
COMPACT_SVG_DOC = '''Absatz davor.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40" font-family="sans-serif">
  <title>Titel der Figur</title>
  <rect x="1" y="1" width="98" height="38" fill="#eee" stroke="#999"/>
  <text x="50" y="18" text-anchor="middle">Label Eins</text>
  <text x="50" y="32" text-anchor="middle">Label <tspan font-weight="bold">Zwei</tspan></text>
</svg>

Mitte <svg viewBox="0 0 10 10"><text x="1" y="5">Inline-Label</text></svg> Ende.

Absatz danach.
'''

BROKEN_MERMAID_DOC = '''# Smoke: Mermaid-Syntaxfehler

Absatz vor dem kaputten Diagramm.

```mermaid
flowchart LR
  A -->
  ((( das ist kein Mermaid
```

Absatz zwischen den Diagrammen.

```mermaid
flowchart LR
  Heil --> Ganz
```

Absatz nach beiden Diagrammen ist gerendert.
'''

PATHOLOGICAL = {
    'A nie geschlossen': '# Smoke: pathologisch A\n\n' + '<svg \n' * 4000,
    'B Oeffner ueberwiegen': '# Smoke: pathologisch B\n\n' + '<svg><svg></svg>\n' * 4000,
}
PAGE_BOUND_S = 10.0   # the first block rule took 5.7 s / 10.7 s for the RENDER alone
LIST_BOUND_S = 5.0

# --- test data through the ORM (same container, same DB) ---------------------
sys.path.insert(0, os.environ.get('SMOKE_APP_ROOT', os.getcwd()))
from app import app  # noqa: E402  (bootstrap shim; the CLI imports it the same way)
from app_pkg import markdown_render  # noqa: E402
from models import Conversion, Highlight, User, db  # noqa: E402
from services.epub_service import build_epub  # noqa: E402


def resolve_user():
    with app.app_context():
        user = User.query.filter_by(username=USER).first()
        if user is None:
            sys.exit(f'user {USER!r} does not exist — create it with flask create-user')
        if user.id == 1 or USER == os.environ.get('INGEST_USER'):
            sys.exit(f'refusing to run as user id {user.id} ({USER!r}): that is the '
                     'INGEST/first user = Oli\'s account')
        return user.id


def create_documents(user_id, docs):
    ids = {}
    with app.app_context():
        for key, (title, content) in docs.items():
            conv = Conversion(user_id=user_id, conversion_type='markdown_input',
                              title=title, content=content)
            db.session.add(conv)
            db.session.flush()
            ids[key] = conv.id
        db.session.commit()
    return ids


def remove_documents(ids, user_id):
    with app.app_context():
        rows = Conversion.query.filter(Conversion.user_id == user_id,
                                       Conversion.id.in_(ids)).all()
        for conv in rows:
            db.session.delete(conv)  # highlights cascade ORM-side
        db.session.commit()
        left = Conversion.query.filter(Conversion.user_id == user_id,
                                       Conversion.id.in_(ids)).count()
        orphans = Highlight.query.filter(Highlight.conversion_id.in_(ids)).count()
        return len(rows), left, orphans


def pre_sprint_html(text):
    """The renderer as it stood before RICH-MEDIA: same MarkdownIt core rules
    minus the svg block rule, the bare nh3 call."""
    md = markdown_render._md
    md.block.ruler.disable('svg_block')
    try:
        rendered = md.render(text)
    finally:
        md.block.ruler.enable('svg_block')
    return nh3.clean(rendered, tags=markdown_render._ALLOWED_TAGS,
                     attributes=markdown_render._ALLOWED_ATTRIBUTES)


# --- page helpers ------------------------------------------------------------
RAW_TEXT = "() => readerRawText(highlightReaderEl())"

RAW_TEXT_OF_HTML = """html => {
  const holder = document.createElement('article');
  holder.innerHTML = html;
  return readerRawText(holder);
}"""

MERMAID_STATES = """() => [...document.querySelectorAll('.reader-view .reader-mermaid')].map(h => ({
  state: h.dataset.mermaidState,
  marked: h.classList.contains('reader-mermaid--marked'),
  hasSvg: !!h.shadowRoot.querySelector('.figure svg'),
  hint: h.shadowRoot.querySelector('.hint').hidden ? null : h.shadowRoot.querySelector('.hint').textContent,
  toggle: h.shadowRoot.querySelector('.toggle').hidden ? null : h.shadowRoot.querySelector('.toggle').textContent,
  sourceVisible: h.nextElementSibling.offsetParent !== null,
}))"""

HIGHLIGHT_STATE = """() => {
  const reader = highlightReaderEl();
  const out = {};
  reader.querySelectorAll('span.highlight[data-highlight-id]').forEach(span => {
    const id = span.dataset.highlightId;
    if (!out[id]) out[id] = { text: '', start: rawOffsetForPoint(reader, span.firstChild, 0) };
    out[id].text += span.textContent;
  });
  return out;
}"""

PHRASE_RECTS = """phrase => {
  const reader = highlightReaderEl();
  const walker = document.createTreeWalker(reader, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const at = node.nodeValue.indexOf(phrase);
    if (at < 0 || !node.parentElement || node.parentElement.offsetParent === null) continue;
    const range = document.createRange();
    range.setStart(node, at);
    range.setEnd(node, at + phrase.length);
    node.parentElement.scrollIntoView({ block: 'center' });
    const rects = [...range.getClientRects()];
    const first = rects[0], last = rects[rects.length - 1];
    return { x1: first.left, y1: first.top + first.height / 2,
             x2: last.right, y2: last.top + last.height / 2, lines: rects.length };
  }
  return null;
}"""

LAYOUT = """() => {
  const reader = highlightReaderEl();
  const box = reader.getBoundingClientRect();
  const figures = [...reader.querySelectorAll('svg, img, .reader-mermaid')].map(el => {
    const r = el.getBoundingClientRect();
    return { tag: el.tagName.toLowerCase(), width: Math.round(r.width), right: Math.round(r.right) };
  });
  const inline = reader.querySelector('svg');
  return {
    readerRight: Math.round(box.right), readerWidth: Math.round(box.width),
    overflow: reader.scrollWidth - reader.clientWidth,
    figures,
    svgColor: inline ? getComputedStyle(inline).color : null,
    textColor: getComputedStyle(reader).color,
    theme: document.documentElement.getAttribute('data-global-theme') || 'light',
  };
}"""


def wait_mermaid(page, expected, timeout_ms=30000):
    """Wait until every fence host has left pending/rendering."""
    page.wait_for_function(
        """n => { const hosts = [...document.querySelectorAll('.reader-view .reader-mermaid')];
                  return hosts.length === n && hosts.every(h =>
                      ['rendered', 'failed'].includes(h.dataset.mermaidState)); }""",
        arg=expected, timeout=timeout_ms)


def open_doc(page, doc_id, theme='light'):
    page.goto(f'{BASE}/login')
    page.evaluate("t => localStorage.setItem('globalTheme', t)", theme)
    started = time.perf_counter()
    # domcontentloaded, not load: a Mermaid script injected before `load`
    # delays that event — and 1a holds exactly that request back.
    page.goto(f'{BASE}/library/{doc_id}', wait_until='domcontentloaded')
    page.wait_for_selector('.reader-view')
    return time.perf_counter() - started


def drag_phrase(page, phrase):
    """Select ``phrase`` with a REAL mouse drag and wait for the highlight
    POST the mouseup triggers. Returns the created highlight (JSON)."""
    rect = page.evaluate(PHRASE_RECTS, phrase)
    if rect is None:
        return None
    page.wait_for_timeout(250)  # let the scrollIntoView settle
    rect = page.evaluate(PHRASE_RECTS, phrase)
    with page.expect_response(lambda r: r.url.endswith('/highlights')
                              and r.request.method == 'POST', timeout=10000) as info:
        page.mouse.move(rect['x1'] + 1, rect['y1'])
        page.mouse.down()
        page.mouse.move((rect['x1'] + rect['x2']) / 2, rect['y2'], steps=6)
        page.mouse.move(rect['x2'] - 1, rect['y2'], steps=6)
        page.mouse.up()
    created = info.value.json()
    page.wait_for_timeout(900)  # one click per second (keep-alive race, house rule)
    return created


def pdf_facts(path):
    doc = fitz.open(path)
    fills, text, images = set(), '', 0
    for pg in doc:
        text += pg.get_text()
        images += len(pg.get_images())
        for drawing in pg.get_drawings():
            if drawing.get('fill'):
                fills.add(tuple(round(c, 2) for c in drawing['fill']))
    pages = doc.page_count
    doc.close()
    return pages, fills, text, images


# --- run -----------------------------------------------------------------------
user_id = resolve_user()
with open(MALICIOUS_PATH, encoding='utf-8') as fh:
    MALICIOUS_DOC = fh.read()

docs = {
    'figures': ('Smoke RICH-MEDIA Figuren', FIGURES_DOC),
    'plain': ('Smoke RICH-MEDIA Baseline', PLAIN_DOC),
    'malicious': ('Smoke RICH-MEDIA Malicious', MALICIOUS_DOC),
    'broken': ('Smoke RICH-MEDIA Mermaid-Fehler', BROKEN_MERMAID_DOC),
}
for label, content in PATHOLOGICAL.items():
    docs[f'patho:{label}'] = (f'Smoke RICH-MEDIA pathologisch {label}', content)
ids = create_documents(user_id, docs)
print(f'user_id={user_id} documents={ids}')

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={'width': 1200, 'height': 900}, accept_downloads=True)
        page = ctx.new_page()
        dialogs = []
        page.on('dialog', lambda d: (dialogs.append(d.message), d.dismiss()))

        page.goto(f'{BASE}/login')
        page.fill('input[name=username]', USER)
        page.fill('input[name=password]', PASSWORD)
        page.click('button[type=submit]')
        page.wait_for_url(lambda url: '/login' not in url)

        print('=== 1a. readerRawText is byte-equal before/after Mermaid renders ===')
        held = []
        page.route('**/mermaid@*/**', lambda route: held.append(route))
        open_doc(page, ids['figures'])
        page.wait_for_function("() => document.querySelectorAll('.reader-view .reader-mermaid').length === 1")
        page.evaluate("() => document.querySelector('.reader-view .reader-mermaid').nextElementSibling"
                      ".scrollIntoView({block: 'center'})")
        page.wait_for_timeout(1500)
        before_states = page.evaluate(MERMAID_STATES)
        raw_before = page.evaluate(RAW_TEXT)
        print(f'[before] mermaid={before_states} held CDN requests={len(held)} raw chars={len(raw_before)}')
        check(len(held) == 1 and before_states[0]['state'] == 'rendering' and not before_states[0]['hasSvg'],
              'measured BEFORE the diagram exists: CDN request held, host in state "rendering", no svg')
        check(before_states[0]['sourceVisible'], 'before rendering the fence source is visible')
        for route in held:  # release first — unroute() settles held routes itself
            route.continue_()
        page.unroute('**/mermaid@*/**')
        wait_mermaid(page, 1)
        after_states = page.evaluate(MERMAID_STATES)
        raw_after = page.evaluate(RAW_TEXT)
        print(f'[after]  mermaid={after_states} raw chars={len(raw_after)}')
        check(after_states[0]['state'] == 'rendered' and after_states[0]['hasSvg'],
              'the diagram rendered into the shadow root')
        check(not after_states[0]['sourceVisible'] and after_states[0]['toggle'] == 'Quelltext zeigen',
              'the fence source is hidden, the toggle offers "Quelltext zeigen"')
        check(raw_before == raw_after,
              f'readerRawText byte-equal before/after Mermaid ({len(raw_before)} == {len(raw_after)} chars)')
        check('flowchart LR' in raw_after and MERMAID_SOURCE_PHRASE in raw_after,
              'the hidden fence source is still part of readerRawText')
        shadow_text = page.evaluate("() => document.querySelector('.reader-view .reader-mermaid')"
                                    ".shadowRoot.querySelector('.figure').textContent")
        check('QuelleA' in shadow_text and raw_after.count('QuelleA') == 1,
              f'the diagram\'s own label text lives in the shadow tree only '
              f'(shadow has it: {"QuelleA" in shadow_text}; readerRawText count: {raw_after.count("QuelleA")})')

        print('=== 1b. text nodes under the PRE-SPRINT renderer vs now (SVG text, whitespace) ===')
        # Both HTML strings go through the same detached <article> and the
        # page's own readerRawText — the template's indentation plays no part.
        for label, text, gate in (('compact SVG document (the #240 shape)', COMPACT_SVG_DOC, True),
                                  ('figures document (blank lines INSIDE the svg)', FIGURES_DOC, False)):
            raw_old = page.evaluate(RAW_TEXT_OF_HTML, pre_sprint_html(text))
            raw_new = page.evaluate(RAW_TEXT_OF_HTML, markdown_render.render_markdown_to_html(text))
            first_diff = next((i for i, (a, b) in enumerate(zip(raw_old, raw_new)) if a != b),
                              None if len(raw_old) == len(raw_new) else min(len(raw_old), len(raw_new)))
            print(f'[nh3 {nh3_version}] {label}: old={len(raw_old)} chars, new={len(raw_new)} chars, '
                  f'first difference at={first_diff}')
            if first_diff is not None:
                print(f'    old: {raw_old[max(0, first_diff - 30):first_diff + 40]!r}')
                print(f'    new: {raw_new[max(0, first_diff - 30):first_diff + 40]!r}')
            if not gate:
                # A blank line inside an <svg> broke the OLD renderer (Typ 7:
                # the inner lines became paragraphs) — there is no intact
                # "before" to be equal to. Printed, not judged.
                continue
            if 'Label Eins' in raw_old:
                # The repo pin (0.2.18): the old renderer stripped the SVG TAGS
                # and left their text behind as loose text nodes — figures
                # bring no new text, it only moves into <text> elements.
                check(raw_old == raw_new,
                      'readerRawText is byte-equal under the pre-sprint and the live renderer '
                      '(SVG text and the whitespace between elements included)')
            else:
                # nh3 0.3.x drops a disallowed SVG subtree WITH its text; there
                # the old renderer is no reference. Not the deployed pin.
                print(f'  n/a  nh3 {nh3_version} drops the SVG subtree with its text under the old '
                      'renderer; only meaningful on the repo pin (the container)')

        print('=== 1c. real mouse-drag highlights immediately before/after each figure ===')
        created = {}
        for phrase in PHRASES:
            hl = drag_phrase(page, phrase)
            ok = bool(hl) and hl.get('exact') == phrase
            print(f'[drag] {phrase!r} -> id={hl and hl.get("id")} exact={hl and hl.get("exact")!r}')
            check(ok, f'drag over {phrase!r} stored exactly that phrase')
            if hl:
                created[str(hl['id'])] = phrase
        state_1 = page.evaluate(HIGHLIGHT_STATE)
        print(f'[spans] {state_1}')
        check(all(state_1.get(i, {}).get('text') == phrase for i, phrase in created.items()),
              'every drag highlight is wrapped as exactly its phrase')
        check(all(raw_after[state_1[i]['start']:state_1[i]['start'] + len(phrase)] == phrase
                  for i, phrase in created.items() if i in state_1),
              'every span sits at the raw offset of its phrase')
        check(page.evaluate(RAW_TEXT) == raw_after, 'wrapping highlights left readerRawText unchanged')

        recent = page.request.get(f'{BASE}/api/highlights/recent?limit=50').json()
        recent_exacts = {str(h['id']): h['exact'] for h in recent}
        check(all(recent_exacts.get(i) == phrase for i, phrase in created.items()),
              f'/api/highlights/recent delivers the exact phrases ({len(created)} of {len(created)})')

        print('=== 1d. a highlight INSIDE the Mermaid source (the #133 case) ===')
        source_hl = page.evaluate(
            """async ([id, exact]) => {
                const r = await fetch(`/api/conversions/${id}/highlights`, {method: 'POST',
                    headers: {'Content-Type': 'application/json'}, body: JSON.stringify({exact, prefix: '', suffix: ''})});
                return r.ok ? r.json() : {error: r.status};
            }""", [ids['figures'], MERMAID_SOURCE_PHRASE])
        check('id' in source_hl, f'highlight inside the fence source stored ({source_hl})')
        created[str(source_hl.get('id'))] = MERMAID_SOURCE_PHRASE

        print('=== 1e. reload: every highlight re-anchors at the same raw offset ===')
        open_doc(page, ids['figures'])
        wait_mermaid(page, 1)
        page.wait_for_function("n => new Set([...document.querySelectorAll('.reader-view span.highlight[data-highlight-id]')]"
                               ".map(s => s.dataset.highlightId)).size === n", arg=len(created))
        state_2 = page.evaluate(HIGHLIGHT_STATE)
        raw_reload = page.evaluate(RAW_TEXT)
        print(f'[spans after reload] {state_2}')
        check(raw_reload == raw_after, 'readerRawText after the reload is byte-equal')
        check(all(state_2.get(i, {}).get('text') == phrase for i, phrase in created.items()),
              'after the reload every highlight wraps exactly its phrase again')
        check(all(state_2[i]['start'] == state_1[i]['start'] for i in state_1),
              'after the reload the eight drag highlights sit at the SAME raw offsets')
        marked = page.evaluate(MERMAID_STATES)[0]
        check(marked['marked'] and not marked['sourceVisible'],
              f'the figure carries the marker while its hidden source holds a highlight ({marked})')
        page.evaluate("id => scrollToHighlight(id)", source_hl.get('id'))
        page.wait_for_timeout(800)
        opened = page.evaluate(MERMAID_STATES)[0]
        check(opened['sourceVisible'] and opened['toggle'] == 'Quelltext verbergen',
              f'scrollToHighlight opened the source ({opened})')
        check(page.evaluate(RAW_TEXT) == raw_after, 'opening the source left readerRawText unchanged')

        print('=== 2. malicious fixture: no outgoing request, nothing executable ===')
        requests = []
        page.route('**/*', lambda route: (requests.append(route.request.url), route.continue_()))
        open_doc(page, ids['plain'])
        page.wait_for_load_state('networkidle')
        baseline = list(requests)
        requests.clear()
        open_doc(page, ids['malicious'])
        page.wait_for_load_state('networkidle')
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)
        page.unroute('**/*')
        origin = lambda url: '/'.join(url.split('/')[:3])  # noqa: E731
        strip_id = lambda url: url.replace(f'/{ids["malicious"]}', '/ID').replace(f'/{ids["plain"]}', '/ID')  # noqa: E731
        evil = [u for u in requests if 'evil.example' in u]
        new_origins = sorted({origin(u) for u in requests} - {origin(u) for u in baseline})
        new_urls = sorted({strip_id(u) for u in requests if not u.startswith('data:')}
                          - {strip_id(u) for u in baseline})
        print(f'[requests] baseline={len(baseline)} malicious={len(requests)} '
              f'evil={len(evil)} new origins={new_origins} new urls={new_urls}')
        check(not evil, f'requests to evil.example: {len(evil)}')
        check(not new_origins, 'the malicious document loads from no origin the plain document does not')
        # The one request a LONGER document adds is the app's own reading
        # progress (scrolling) — same origin, not caused by the markup.
        off_origin = [u for u in new_urls if not u.startswith(BASE)]
        unexpected = [u for u in new_urls if u.startswith(BASE) and not u.endswith('/progress')]
        check(not off_origin and not unexpected,
              f'beyond the baseline only the app\'s own reading-progress request (off-origin: {off_origin}, '
              f'unexpected: {unexpected})')
        dom = page.evaluate("""() => { const r = highlightReaderEl(); return {
            scripts: r.querySelectorAll('script, iframe, foreignObject, image, use, style').length,
            handlers: [...r.querySelectorAll('*')].filter(e => [...e.attributes].some(a => a.name.startsWith('on'))).length,
            hrefs: [...r.querySelectorAll('[href]')].map(e => e.getAttribute('href')),
            styled: r.querySelectorAll('svg [style], svg[style], div[style]').length,
            rects: r.querySelectorAll('svg rect').length,
            label: r.textContent.includes('Legitimes Label'),
        }; }""")
        print(f'[dom] {dom}')
        check(dom['scripts'] == 0 and dom['handlers'] == 0 and not dom['hrefs'] and dom['styled'] == 0,
              'reader DOM: no script/iframe/foreignObject/image/use/style, no on*-handler, no href, no style')
        check(dom['rects'] == 2 and dom['label'], 'the legitimate part of the figure rendered (2 rects, label)')
        check(not dialogs, f'no dialog opened ({dialogs})')

        print('=== 3. Mermaid syntax error: source visible, hint, the rest rendered ===')
        open_doc(page, ids['broken'])
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        wait_mermaid(page, 2)
        broken, healthy = page.evaluate(MERMAID_STATES)
        print(f'[broken] {broken}\n[healthy] {healthy}')
        check(broken['state'] == 'failed' and broken['sourceVisible'] and not broken['hasSvg'],
              'broken fence: state failed, source visible, no diagram')
        check(bool(broken['hint']) and 'Syntaxfehler' in broken['hint'], f'broken fence shows the hint ({broken["hint"]!r})')
        check(healthy['state'] == 'rendered' and healthy['hasSvg'] and not healthy['sourceVisible'],
              'the valid fence in the same document rendered')
        rest = page.evaluate("""() => ({
            last: highlightReaderEl().textContent.includes('Absatz nach beiden Diagrammen ist gerendert.'),
            parked: document.querySelectorAll('body > [id^="dreader-mermaid"], body > svg').length })""")
        check(rest['last'], 'the paragraph after both diagrams is rendered')
        check(rest['parked'] == 0, f'no Mermaid error graphic parked in <body> ({rest["parked"]})')

        print('=== 4. pathological input: reader page and library list answer in finite time ===')
        for label in PATHOLOGICAL:
            elapsed = open_doc(page, ids[f'patho:{label}'])
            placeholders = page.evaluate("() => document.querySelectorAll('.reader-view .media-placeholder').length")
            print(f'[patho] {label}: reader page in {elapsed:.2f}s, placeholders={placeholders}')
            check(elapsed < PAGE_BOUND_S, f'{label}: reader page answered in {elapsed:.2f}s (< {PAGE_BOUND_S:.0f}s)')
        started = time.perf_counter()
        listing = page.request.get(f'{BASE}/api/conversions?limit=100')
        elapsed = time.perf_counter() - started
        titles = [i['title'] for i in listing.json()['items']]
        print(f'[patho] /api/conversions (previews of all {len(titles)} documents) in {elapsed:.2f}s')
        check(listing.ok and any('pathologisch' in t for t in titles) and elapsed < LIST_BOUND_S,
              f'the list with the pathological documents in it answered in {elapsed:.2f}s (< {LIST_BOUND_S:.0f}s)')
        started = time.perf_counter()
        page.goto(f'{BASE}/library')
        elapsed = time.perf_counter() - started
        check(elapsed < PAGE_BOUND_S, f'the library page answered in {elapsed:.2f}s')

        print('=== 5. layout: light/dark x 375/1200 ===')
        colors = {}
        for theme in ('light', 'dark'):
            for width in (375, 1200):
                page.set_viewport_size({'width': width, 'height': 900})
                open_doc(page, ids['figures'], theme)
                page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
                wait_mermaid(page, 1)
                page.evaluate("() => window.scrollTo(0, 0)")
                page.wait_for_timeout(400)
                lay = page.evaluate(LAYOUT)
                colors[(theme, width)] = lay['svgColor']
                page.screenshot(path=f'{OUT}_{theme}_{width}.png', full_page=True)
                print(f'[layout {theme} {width}] {lay}')
                check(lay['theme'] == theme, f'{theme}/{width}: theme applied')
                check(lay['overflow'] <= 1, f'{theme}/{width}: reader has no horizontal overflow ({lay["overflow"]}px)')
                check(all(f['right'] <= lay['readerRight'] + 1 and f['width'] > 0 for f in lay['figures'])
                      and len(lay['figures']) == 4,
                      f'{theme}/{width}: all 4 figures lie inside the reader column')
                check(lay['svgColor'] == lay['textColor'], f'{theme}/{width}: svg colour = reader text colour')
        check(colors[('light', 1200)] != colors[('dark', 1200)],
              f'currentColor follows the theme ({colors[("light", 1200)]} vs {colors[("dark", 1200)]})')
        page.set_viewport_size({'width': 1200, 'height': 900})

        print('=== 6. PDF through the real form, EPUB build ===')
        page.goto(f'{BASE}/login')
        page.evaluate("() => localStorage.setItem('globalTheme', 'light')")
        page.goto(f'{BASE}/')
        page.wait_for_selector('#markdown_text')
        page.fill('#markdown_text', FIGURES_DOC)
        page.fill('#output_filename', 'smoke-media')
        with page.expect_download(timeout=90000) as dl:
            page.click('#convert-form button[type=submit]')
        pdf_path = f'{OUT}_figures.pdf'
        dl.value.save_as(pdf_path)
        pages, fills, text, images = pdf_facts(pdf_path)
        print(f'[pdf] pages={pages} images={images} fills={sorted(fills)} label={"SvgLabel Eins" in text}')
        check(pages >= 1 and 'SvgLabel Eins' in text, 'PDF built, the SVG label is in its text layer')
        check((0.2, 0.4, 0.6) in fills, 'PDF carries the inline SVG as drawing (rect fill #336699)')
        check((0.93, 0.93, 1.0) in fills or (0.93, 1.0, 0.93) in fills or images >= 2,
              'PDF carries the data-URI images (their fills as drawings, or as images)')

        browser.close()

    epub = build_epub('Smoke RICH-MEDIA', markdown_render.render_markdown_to_html(FIGURES_DOC))
    with zipfile.ZipFile(io.BytesIO(epub)) as z:
        chapter = next(n for n in z.namelist() if n.endswith('chapter.xhtml'))
        body = z.read(chapter).decode('utf-8')
    print(f'[epub] bytes={len(epub)} chapter={chapter} svg={"<svg" in body} data-img={"data:image/svg+xml" in body}')
    check(epub[:2] == b'PK' and '<svg' in body and 'data:image/svg+xml' in body,
          'EPUB built without exception; the chapter carries the inline SVG and the data-URI images')
finally:
    removed, left, orphans = remove_documents(list(ids.values()), user_id)
    print(f'[cleanup] removed {removed} documents, left={left}, orphaned highlights={orphans}')
    check(left == 0 and orphans == 0, 'test documents and their highlights are gone')

print()
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('ALL PASSED')
