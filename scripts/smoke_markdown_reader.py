#!/usr/bin/env python3
"""Browser smoke for the markdown-converter reader mode (READER-SCOPE + READER-STIL).

The pytest suite renders no templates, runs no JS and computes no CSS — this
is the one REAL-browser check of the reader scope in
``templates/markdown_converter.html`` + ``static/css/style.css`` +
``static/js/markdown_converter.js``. It runs INSIDE the web container (the
playwright base image ships Chromium — nothing to install) against the
deployed app, logs in as a throwaway user and drives the reader through the
states a person can reach, MEASURING instead of looking:

1. Reader-mode ON, five states — global light × reader {follows, dark},
   global dark × reader {follows global, explicit light, dark}. For each it
   prints ``getComputedStyle`` background colours of ``.main-container``,
   ``.preview-pane``, ``.preview-container``, the ``.preview-iframe`` element
   and the iframe document, plus the four corner pixels of a screenshot, and
   asserts the property READER-SCOPE fixed: **every surface between <body>
   and the iframe document carries the document's own paper tone** (the old
   rim was the container's global-token padding + the surface card whose
   ``::before`` selector never matched).
2. Reader-mode OFF, global light and dark — the container keeps its global
   padding/background/card: the reader scope leaks nowhere when it is off.
3. Flash probe — a CDP screencast while 30 keystrokes rebuild the iframe's
   srcdoc in the dark reader; every compositor frame's mean luminance over
   the iframe area is printed and no frame may be blank-white (the iframe
   ELEMENT used to be ``#fff`` and showed through during every rebuild:
   measured 4–5 white frames per 30 keystrokes before the fix).
4. Styles in the dark reader (READER-STIL) — each PDF style is picked through
   the Aa-popover; paper, h1 colour, table line, code colour and font are
   measured inside the document, the surfaces + screenshot corners must
   equal that paper (the style's dark twin sets the paper, the scope follows
   it), and the three styles must be pairwise DISTINCT on paper and h1
   (before the twins every one of these values was identical in dark).
5. The popover drives the form's <select>: pick Bodoni in the reader, leave
   the reader through the popover, generate the PDF — the downloaded file's
   text spans must differ from a default-style PDF: no Liberation Sans (the
   default's body face) and a drop-cap span (Bodoni's ::first-letter — Chromium
   embeds the Google web fonts as unnamed Type3 programs, so the signature is
   the typography, not a font name). The reader's look proves nothing; the
   artefact does.
6. The library reader shows no style group (it has no PDF styles).
7. Scroll bar in the preview iframe (READER-SCROLLBAR) — in Chromium AND
   WebKit, eight states (dark × three styles + the twin-less fallback, global
   light × reader dark, light, explicit light under a dark global theme,
   reader off): computed ``color-scheme`` + ``scrollbar-width`` at the iframe's
   <html>, the bar's width, and one pixel column down the bar, run-length
   encoded — the track must BE the paper, the thumb must stand against it
   within a named contrast band (WebKit on Linux draws overlay bars: 0 px,
   nothing at rest — there the computed values and the track carry the proof). Before the fix: 15 px, track rgb(252,252,252)
   in every state (Chromium); in WebKit under a dark global theme the app's
   own ::-webkit-scrollbar rule styled the frame's bar through the OWNER
   <iframe> element (8 px, --nm-bg track on the paper) and beat the framed
   document's standard properties — hence ``:not(iframe)`` in style.css.
   Playwright's Firefox forces ``scrollbar-width: none``; not measurable here.

How to run (Mintbox, ~4 min):

    # 1. throwaway user — NEVER Oli's account
    docker exec markdown-converter-web flask --app app create-user zz_smoke --password '<random>'
    docker cp scripts/smoke_markdown_reader.py markdown-converter-web:/tmp/smoke_reader.py
    # 2. run — screenshots + PDFs land in the container as /tmp/smoke_reader_*
    docker exec -e SMOKE_USER=zz_smoke -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_reader.py
    # 3. clean up STRICTLY by user_id (the api_token table carries Oli's iOS
    #    tokens): delete the user's Conversion/ApiToken rows and the User row
    #    via the ORM, then remove /tmp/smoke_reader* from the container.

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SMOKE_OUT (/tmp/smoke_reader), SMOKE_LIBRARY_ID (optional: a conversion id
of the throwaway user for step 6), SMOKE_ONLY=scrollbar (step 7 alone). Exit 0 = every check passed; every
measured value is printed so a failure is diagnosable from the output alone.
"""
import base64
import io
import itertools
import json
import os
import re
import sys

import fitz  # PyMuPDF — in the image (pdf_local)
from PIL import Image
from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
OUT = os.environ.get('SMOKE_OUT', '/tmp/smoke_reader')
LIBRARY_ID = os.environ.get('SMOKE_LIBRARY_ID')

WHITE = 'rgb(255, 255, 255)'
GLOBAL_BG = {'light': 'rgb(224, 229, 236)', 'dark': 'rgb(42, 45, 58)'}  # --nm-bg
STYLES = ('default', 'academic-latex', 'newspaper-bodoni')

# (global theme, readerPrefs.dark, click the dark toggle once after load, label, expected kind)
READER_STATES = [
    ('light', False, False, 'G-hell_R-folgt', 'light'),
    ('light', True, False, 'G-hell_R-dunkel', 'dark'),
    ('dark', False, False, 'G-dunkel_R-folgt', 'dark'),
    ('dark', True, True, 'G-dunkel_R-hell-explizit', 'light'),
    ('dark', True, False, 'G-dunkel_R-dunkel', 'dark'),
]

MEASURE = """() => {
  const q = s => document.querySelector(s);
  const bg = el => el ? getComputedStyle(el).backgroundColor : null;
  const main = q('.main-container');
  const ifr = q('#preview-iframe');
  const doc = ifr && ifr.contentDocument;
  return {
    global: document.documentElement.getAttribute('data-global-theme'),
    reader: document.documentElement.getAttribute('data-theme'),
    readerMode: main.classList.contains('reader-mode'),
    body: bg(document.body),
    main: bg(main),
    main_before: getComputedStyle(main, '::before').display,
    main_padding: getComputedStyle(main).padding,
    pane: bg(q('.preview-pane')),
    container: bg(q('.preview-container')),
    iframe_el: bg(ifr),
    doc_html: doc && doc.documentElement ? bg(doc.documentElement) : null,
  };
}"""

MEASURE_STYLE = """() => {
  const d = document.querySelector('#preview-iframe').contentDocument;
  const cs = (s, p) => { const el = d.querySelector(s); return el ? getComputedStyle(el)[p] : null; };
  return {
    select: document.getElementById('style_theme').value,
    paper: getComputedStyle(d.documentElement).backgroundColor,
    h1: cs('h1', 'color'),
    td_line: cs('td', 'borderBottomColor'),
    pre_bg: cs('pre', 'backgroundColor'),
    code_color: cs('pre code', 'color'),
    font: cs('body', 'fontFamily').split(',')[0],
  };
}"""

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


def rgb(px):
    return f'rgb({px[0]}, {px[1]}, {px[2]})'


def luminance(css_rgb):
    r, g, b = (int(v) for v in re.findall(r'\d+', css_rgb)[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def corner_pixels(png_bytes):
    """Four screenshot corners; the top-right one sits below the fixed 'Aa'
    trigger (a UI overlay, not a surface between body and iframe)."""
    im = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    w, h = im.size
    pts = {'top-left': (3, 3), 'bottom-left': (3, h - 4),
           'bottom-right': (w - 4, h - 4), 'right-middle': (w - 4, h // 2)}
    return {k: rgb(im.getpixel(v)) for k, v in pts.items()}


def surfaces_follow_paper(label, m, corners):
    paper = m['doc_html']
    wrong = {k: m[k] for k in ('main', 'pane', 'container', 'iframe_el') if m[k] != paper}
    check(not wrong, f'{label}: main/pane/container/iframe all carry the paper {paper}'
                     + (f' — off: {wrong}' if wrong else ''))
    check(m['main_before'] == 'none' and m['main_padding'] == '0px',
          f'{label}: no surface card (::before none) and no global padding')
    wrong_px = {k: v for k, v in corners.items() if v != paper}
    check(not wrong_px, f'{label}: screenshot corners are the paper tone'
                        + (f' — off: {wrong_px}' if wrong_px else ''))


def load(page, global_theme, mode_on, reader_dark):
    page.evaluate("""([g, on, d]) => {
        localStorage.setItem('globalTheme', g);
        localStorage.setItem('readerPrefs', JSON.stringify({modeOn: on, dark: d, width: 'medium'}));
    }""", [global_theme, mode_on, reader_dark])
    page.goto(f'{BASE}/')
    page.wait_for_function(
        "() => document.querySelector('#preview-iframe')?.contentDocument?.body?.children.length > 0")
    page.wait_for_timeout(500)


def pick_style_in_popover(page, theme):
    """The user's path: open the Aa-popover, click the style's button."""
    if not page.evaluate("() => document.getElementById('reader-aa-popover').classList.contains('is-open')"):
        page.click('#reader-aa-trigger')
    page.click(f'[data-reader-style="{theme}"]')
    page.wait_for_function(f"() => document.getElementById('style_theme').value === '{theme}'")
    page.wait_for_timeout(1200)  # style + dark twin fetch, srcdoc rebuild, paper sync


def pdf_spans(page, filename):
    """Generate the PDF through the real form + download; return
    {span font name: [texts]} — the font NAMES a text run was set in."""
    page.fill('#output_filename', filename)
    with page.expect_download() as dl:
        page.click('#convert-form button[type=submit]')
    path = f'{OUT}_{filename}.pdf'
    dl.value.save_as(path)
    doc = fitz.open(path)
    spans = {}
    for pg in doc:
        for block in pg.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    spans.setdefault(span['font'], []).append(span['text'])
    doc.close()
    return spans


def flash_probe(page, label, keystrokes=30):
    box = page.evaluate("() => { const r = document.querySelector('#preview-iframe')"
                        ".getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }")
    vp = page.viewport_size
    cdp = page.context.new_cdp_session(page)
    frames = []

    def on_frame(ev):
        cdp.send('Page.screencastFrameAck', {'sessionId': ev['sessionId']})
        frames.append(ev['data'])

    cdp.on('Page.screencastFrame', on_frame)
    cdp.send('Page.startScreencast', {'format': 'png', 'maxWidth': vp['width'],
                                      'maxHeight': vp['height'], 'everyNthFrame': 1})
    page.wait_for_timeout(300)
    n0 = len(frames)
    for _ in range(keystrokes):
        # The reader hides the editor pane, so type the way the page's own
        # listener sees it: value change + 'input' event on the textarea.
        page.evaluate("() => { const t = document.getElementById('markdown_text');"
                      " t.value += 'x'; t.dispatchEvent(new Event('input')); }")
        page.wait_for_timeout(90)
    page.wait_for_timeout(500)
    cdp.send('Page.stopScreencast')
    cdp.detach()
    x, y, w, h = box
    lums = []
    for data in frames[n0:]:
        im = Image.open(io.BytesIO(base64.b64decode(data))).convert('L')
        crop = im.crop((int(x + w * 0.1), int(y + h * 0.1), int(x + w * 0.9), int(y + h * 0.9)))
        lums.append(round(sum(crop.getdata()) / (crop.width * crop.height)))
    blank = sum(1 for v in lums if v >= 254)
    print(f'[flash {label}] compositor frames={len(lums)} · blank-white={blank} · luminance={lums}')
    check(blank == 0, f'{label}: no blank-white frame while srcdoc is rebuilt')


# --- 7. scroll bar (READER-SCROLLBAR) ---------------------------------------
# The preview iframe is its OWN document: the app's `color-scheme: dark` and
# its `::-webkit-scrollbar` rules stop at the iframe border, so the browser
# drew its light default bar (15 px, white track) into a dark paper — and in
# the reader the iframe itself scrolls, so that bar stands at the right edge
# of the TEXT COLUMN, not at the window. Headless Chromium draws classic
# (space-taking) bars, so this is measurable, not just visible.
MAX_SCROLLBAR_PX = 12      # a `thin` classic bar; the default measured 15
MIN_THUMB_CONTRAST = 3.0   # WCAG 1.4.11 non-text: the thumb must stay findable
MAX_THUMB_CONTRAST = 4.5   # … and must not shout (the old white track: > 15:1)
LONG_DOC = '# Scroll-Probe\n\n' + '\n\n'.join(
    f'Absatz {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do '
    'eiusmod tempor incididunt ut labore et dolore magna aliqua.' for i in range(1, 121))

# (global theme, readerPrefs.dark, toggle dark once, style, kill the dark twin, reader on, label, kind)
SCROLL_STATES = [
    ('dark', True, False, 'default', False, True, 'dunkel_default', 'dark'),
    ('dark', True, False, 'academic-latex', False, True, 'dunkel_academic-latex', 'dark'),
    ('dark', True, False, 'newspaper-bodoni', False, True, 'dunkel_newspaper-bodoni', 'dark'),
    ('dark', True, False, 'default', True, True, 'dunkel_fallback-ohne-zwilling', 'dark'),
    ('light', True, False, 'default', False, True, 'G-hell_R-dunkel', 'dark'),
    ('light', False, False, 'default', False, True, 'hell_default', 'light'),
    ('dark', True, True, 'default', False, True, 'G-dunkel_R-hell-explizit', 'light'),
    ('dark', False, False, 'default', False, False, 'reader-aus_G-dunkel', 'dark'),
]

MEASURE_SCROLLBAR = """() => {
  const ifr = document.querySelector('#preview-iframe');
  const w = ifr.contentWindow, html = ifr.contentDocument.documentElement;
  const cs = w.getComputedStyle(html);
  const r = ifr.getBoundingClientRect();
  return {
    color_scheme: cs.colorScheme,
    scrollbar_width: cs.scrollbarWidth === undefined ? 'unsupported' : cs.scrollbarWidth,
    scrollbar_color: cs.scrollbarColor === undefined ? 'unsupported' : cs.scrollbarColor,
    supports_width: w.CSS.supports('scrollbar-width', 'thin'),
    supports_color: w.CSS.supports('scrollbar-color', 'red blue'),
    bar_px: w.innerWidth - html.clientWidth,
    scroll_h: html.scrollHeight, client_h: html.clientHeight,
    paper: cs.backgroundColor,
    iframe_el: getComputedStyle(ifr).backgroundColor,
    host_rule_on_body: getComputedStyle(document.body, '::-webkit-scrollbar').width,
    host_rule_on_iframe: getComputedStyle(ifr, '::-webkit-scrollbar').width,
    box: [r.x, r.y, r.width, r.height],
  };
}"""


def contrast(a, b):
    def rel(css):
        ch = []
        for v in (int(x) for x in re.findall(r'\d+', css)[:3]):
            c = v / 255
            ch.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    hi, lo = sorted((rel(a), rel(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def bar_strip(png_bytes, m, out_path):
    """The measurement: one pixel column down the middle of the bar, run-length
    encoded top to bottom — track, thumb and (if the engine draws them) arrow
    buttons all show up as runs. Also saves a 60-px crop of the iframe's right
    edge for the eye."""
    im = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    x0, y0, w, h = m['box']
    right = int(round(x0 + w))
    x = right - max(m['bar_px'], 8) // 2  # an overlay bar (0 px) would sit in the last 8
    top, bottom = max(int(y0) + 1, 0), min(int(y0 + h) - 1, im.size[1])
    im.crop((right - 60, top, min(right + 20, im.size[0]), bottom)).save(out_path)
    runs = []
    for y in range(top, bottom):
        px = rgb(im.getpixel((x, y)))
        if runs and runs[-1][0] == px:
            runs[-1][1] += 1
        else:
            runs.append([px, 1])
    return runs


def scrollbar_section(p):
    print('=== 7. scroll bar in the preview iframe: own document, own rules (READER-SCROLLBAR) ===')
    for engine in ('chromium', 'webkit'):
        # Playwright starts headless Chromium with --hide-scrollbars: no bar is
        # drawn and none takes space — every check below would pass on air.
        browser = getattr(p, engine).launch(
            **({'ignore_default_args': ['--hide-scrollbars']} if engine == 'chromium' else {}))
        page = browser.new_page(viewport={'width': 1200, 'height': 900})
        page.goto(f'{BASE}/login')
        page.fill('#username', USER)
        page.fill('#password', PASSWORD)
        page.click('button[type=submit]')
        page.wait_for_url(lambda url: '/login' not in url)
        for g, d, click_dark, theme, kill_twin, reader_on, label, kind in SCROLL_STATES:
            label = f'{engine} {label}'
            twin_url = f'**/static/css/pdf_styles/dark/{theme}.css'
            if kill_twin:
                page.route(twin_url, lambda route: route.fulfill(status=404, body=''))
            load(page, g, reader_on, d)
            if click_dark:
                page.evaluate("() => toggleDarkMode()")
                page.wait_for_timeout(600)
            if theme != 'default':
                pick_style_in_popover(page, theme)
                page.keyboard.press('Escape')
            page.evaluate("""(t) => { const el = document.getElementById('markdown_text');
                el.value = t; el.dispatchEvent(new Event('input')); }""", LONG_DOC)
            page.wait_for_timeout(1200)
            m = page.evaluate(MEASURE_SCROLLBAR)
            shot = page.screenshot()
            runs = bar_strip(shot, m, f'{OUT}_bar_{label.replace(" ", "_")}.png')
            if kill_twin:
                page.unroute(twin_url)
            paper = m['paper']
            long_runs = sorted(runs, key=lambda r: -r[1])
            track = long_runs[0][0]
            others = [r for r in long_runs if r[0] != track and r[1] >= 8]
            thumb = others[0] if others else None
            print(f'[bar {label}] ' + json.dumps(m))
            print(f'[bar {label}] column runs (colour × px, top→bottom): '
                  + ' | '.join(f'{c}×{n}' for c, n in runs if n >= 3))
            check(m['scroll_h'] > m['client_h'] * 3, f'{label}: document scrolls ({m["scroll_h"]} > 3×{m["client_h"]})')
            check(m['color_scheme'] == kind, f'{label}: iframe html color-scheme is {kind} ({m["color_scheme"]})')
            if m['supports_width']:
                check(m['scrollbar_width'] == 'thin', f'{label}: iframe html scrollbar-width is thin ({m["scrollbar_width"]})')
            else:
                print(f'  NOTE {label}: engine has no scrollbar-width — the ::-webkit-scrollbar fallback carries it')
            check(m['bar_px'] <= MAX_SCROLLBAR_PX, f'{label}: bar is {m["bar_px"]} px wide (≤ {MAX_SCROLLBAR_PX})')
            if reader_on:
                check(track == paper, f'{label}: track pixel {track} == paper {paper}')
            else:
                # Outside the reader nothing syncs the iframe ELEMENT to the
                # paper (style.css paints it #1a1a2e, the twin's paper differs),
                # and Chromium shows the element through a transparent track.
                # The invariant that holds everywhere: no track tone of its own.
                check(track in (paper, m['iframe_el']),
                      f'{label}: track pixel {track} is the paper {paper} or the iframe element '
                      f'behind it {m["iframe_el"]} — no tone of its own')
                if track != paper:
                    print(f'  NOTE {label}: element tone shows through the track, '
                          f'{contrast(track, paper):.2f}:1 against the paper (known residue outside the reader)')
            if g == 'dark':
                check(m['host_rule_on_body'] == '8px' and m['host_rule_on_iframe'] != '8px',
                      f'{label}: the app\'s dark ::-webkit-scrollbar rule still reaches <body> '
                      f'({m["host_rule_on_body"]}) and no longer the <iframe> element ({m["host_rule_on_iframe"]})')
            if m['bar_px'] == 0 and thumb is None:
                print(f'  NOTE {label}: overlay bar — takes no space, draws nothing at rest; no thumb to measure')
            else:
                ratio = contrast(thumb[0], paper) if thumb else 0
                check(thumb is not None and MIN_THUMB_CONTRAST <= ratio <= MAX_THUMB_CONTRAST,
                      f'{label}: thumb {thumb[0] if thumb else None} × {thumb[1] if thumb else 0} px stands '
                      f'{ratio:.2f}:1 against the paper ({MIN_THUMB_CONTRAST} ≤ x ≤ {MAX_THUMB_CONTRAST})')
        browser.close()


def finish():
    print()
    if failures:
        print(f'{len(failures)} FAILED:')
        for f in failures:
            print('  -', f)
        sys.exit(1)
    print('ALL PASSED')
    sys.exit(0)


with sync_playwright() as p:
    if os.environ.get('SMOKE_ONLY') == 'scrollbar':
        scrollbar_section(p)
        finish()
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1200, 'height': 900}, accept_downloads=True)
    page.goto(f'{BASE}/login')
    page.fill('#username', USER)
    page.fill('#password', PASSWORD)
    page.click('button[type=submit]')
    page.wait_for_url(lambda url: '/login' not in url)
    print('logged in as', USER)

    print('=== 1. reader-mode ON: every surface between <body> and the iframe document ===')
    for g, d, click_dark, label, kind in READER_STATES:
        load(page, g, True, d)
        if click_dark:
            page.evaluate("() => toggleDarkMode()")
            page.wait_for_timeout(600)
        m = page.evaluate(MEASURE)
        shot = page.screenshot(path=f'{OUT}_{label}.png')
        corners = corner_pixels(shot)
        print(f'[{label}] kind={kind} ' + json.dumps(m) + ' corners=' + json.dumps(corners))
        check(m['readerMode'], f'{label}: reader-mode is on')
        if kind == 'light':
            check(m['doc_html'] == WHITE, f'{label}: document paper is white')
        else:
            check(luminance(m['doc_html']) < 60, f'{label}: document paper is dark ({m["doc_html"]})')
        surfaces_follow_paper(label, m, corners)

    print('=== 2. reader-mode OFF: the reader scope leaks nowhere ===')
    for g in ('light', 'dark'):
        load(page, g, False, False)
        m = page.evaluate(MEASURE)
        print(f'[off G-{g}] ' + json.dumps(m))
        check(not m['readerMode'] and m['reader'] is None, f'off G-{g}: reader-mode off, no data-theme')
        check(m['main'] == GLOBAL_BG[g] and m['main_padding'] != '0px' and m['main_before'] == 'block',
              f'off G-{g}: container keeps global --nm-bg, padding and surface card')

    print('=== 3. flash probe: srcdoc rebuild in the dark reader ===')
    for g in ('dark', 'light'):
        load(page, g, True, True)
        flash_probe(page, f'G-{g}_R-dunkel')

    print('=== 4. styles in the dark reader: picked through the popover, each one itself ===')
    load(page, 'dark', True, True)
    seen = {}
    for theme in STYLES:
        pick_style_in_popover(page, theme)
        st = page.evaluate(MEASURE_STYLE)
        m = page.evaluate(MEASURE)
        page.keyboard.press('Escape')  # close the popover for a clean screenshot
        page.wait_for_timeout(200)
        shot = page.screenshot(path=f'{OUT}_style-dark_{theme}.png')
        corners = corner_pixels(shot)
        print(f'[style-dark {theme}] ' + json.dumps(st) + ' surfaces=' + json.dumps(
            {k: m[k] for k in ('main', 'pane', 'container', 'iframe_el', 'doc_html')})
            + ' corners=' + json.dumps(corners))
        check(st['select'] == theme, f'{theme}: popover button drove the <select>')
        check(luminance(st['paper']) < 60, f'{theme}: paper is dark ({st["paper"]})')
        surfaces_follow_paper(f'style-dark {theme}', m, corners)
        seen[theme] = st
    for key in ('paper', 'h1', 'td_line'):
        same = [(a, b) for a, b in itertools.combinations(STYLES, 2) if seen[a][key] == seen[b][key]]
        check(not same, f'dark {key} is pairwise distinct across the three styles'
                        + (f' — same: {same}' if same else ''))
    check(seen['default']['code_color'] != seen['newspaper-bodoni']['code_color'],
          'dark code colour differs between default and bodoni')

    print('=== 5. the <select> is source of truth: the generated PDF carries the reader pick ===')
    # One PDF per page load: the submit button stays disabled ("Wird
    # vorbereitet …") until a navigation, and a download is none (P13 in the
    # submit handler) — so the reader pick is generated first on a fresh page,
    # the default reference after a reload.
    load(page, 'light', False, False)
    page.evaluate("() => toggleReaderMode()")
    page.wait_for_timeout(500)
    pick_style_in_popover(page, 'newspaper-bodoni')
    page.click('[data-reader-exit]')  # leave the reader the user's way
    page.wait_for_timeout(500)
    picked = page.evaluate("() => document.getElementById('style_theme').value")
    check(picked == 'newspaper-bodoni', f'select still carries the reader pick after exit ({picked})')
    spans_bodoni = pdf_spans(page, 'smoke-bodoni')
    print(f'[pdf bodoni]  span fonts={ {k: v[:3] for k, v in spans_bodoni.items()} }')
    load(page, 'light', False, False)
    check(page.evaluate("() => document.getElementById('style_theme').value") == 'default',
          'a fresh page starts on the default style (the pick was not persisted)')
    spans_default = pdf_spans(page, 'smoke-default')
    print(f'[pdf default] span fonts={ {k: v[:3] for k, v in spans_default.items()} }')
    check(set(spans_bodoni) != set(spans_default), 'the Bodoni PDF is set in different fonts than the default PDF')
    check(not any('LiberationSans' in f for f in spans_bodoni),
          "the Bodoni PDF has no Liberation Sans run (the default's body face)")
    check(any(t == 'T' for texts in spans_bodoni.values() for t in texts),
          "the Bodoni PDF carries the drop cap as its own run ('T' of 'This is a sample …')")

    if LIBRARY_ID:
        print('=== 6. the library reader shows no style group ===')
        page.goto(f'{BASE}/library/{LIBRARY_ID}')
        page.wait_for_selector('.library-reader-enter')
        n = page.evaluate("() => document.querySelectorAll('[data-reader-style]').length")
        check(n == 0, f'library reader popover has no style buttons ({n})')

    browser.close()
    scrollbar_section(p)

finish()
