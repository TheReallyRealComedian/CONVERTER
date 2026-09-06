#!/usr/bin/env python3
"""Browser smoke for the markdown-converter reader mode (READER-SCOPE, LESEMODUS).

The pytest suite renders no templates, runs no JS and computes no CSS — this
is the one REAL-browser check of the reader scope in
``templates/markdown_converter.html`` + ``static/css/style.css`` +
``static/js/markdown_converter.js``. It runs INSIDE the web container (the
playwright base image ships Chromium — nothing to install) against the
deployed app, logs in as a throwaway user and drives the reader through the
theme combinations a person can reach, MEASURING instead of looking:

1. Reader-mode ON, five states — global light × reader {follows, dark},
   global dark × reader {follows global, explicit light, dark}. For each it
   prints ``getComputedStyle`` background colours of ``.main-container``,
   ``.preview-pane``, ``.preview-container``, the ``.preview-iframe`` element
   and the iframe document, plus the four corner pixels of a screenshot, and
   asserts the property READER-SCOPE fixed: **no element between <body> and
   the iframe document carries a colour other than the reader tone** (the
   old rim was the container's global-token padding + the surface card whose
   ``::before`` selector never matched).
2. Reader-mode OFF, global light and dark — the container keeps its global
   padding/background/card: the reader scope leaks nowhere when it is off.
3. Flash probe — a CDP screencast while 30 keystrokes rebuild the iframe's
   srcdoc in the dark reader; every compositor frame's mean luminance over
   the iframe area is printed and no frame may be blank-white (the iframe
   ELEMENT used to be ``#fff`` and showed through during every rebuild:
   measured 4–5 white frames per 30 keystrokes before the fix).

How to run (Mintbox, ~1 min):

    # 1. throwaway user — NEVER Oli's account
    docker exec markdown-converter-web flask --app app create-user zz_smoke --password '<random>'
    docker cp scripts/smoke_markdown_reader.py markdown-converter-web:/tmp/smoke_reader.py
    # 2. run — screenshots land in the container as /tmp/smoke_reader_<state>.png
    docker exec -e SMOKE_USER=zz_smoke -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_reader.py
    # 3. clean up STRICTLY by user_id (the api_token table carries Oli's iOS
    #    tokens): delete the user's Conversion/ApiToken rows and the User row
    #    via the ORM, then remove /tmp/smoke_reader* from the container.

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SMOKE_OUT (/tmp/smoke_reader). Exit 0 = every check passed; every measured
value is printed so a failure is diagnosable from the output alone.
"""
import base64
import io
import json
import os
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
OUT = os.environ.get('SMOKE_OUT', '/tmp/smoke_reader')

# Mirrors buildIframeDoc() in markdown_converter.js and the --reader-bg tokens
# at .main-container.reader-mode in style.css.
READER_DARK = 'rgb(26, 26, 46)'      # #1a1a2e
READER_LIGHT = 'rgb(255, 255, 255)'  # #ffffff
GLOBAL_BG = {'light': 'rgb(224, 229, 236)', 'dark': 'rgb(42, 45, 58)'}  # --nm-bg

# (global theme, readerPrefs.dark, click the dark toggle once after load, label, expected reader tone)
READER_STATES = [
    ('light', False, False, 'G-hell_R-folgt', READER_LIGHT),
    ('light', True, False, 'G-hell_R-dunkel', READER_DARK),
    ('dark', False, False, 'G-dunkel_R-folgt', READER_DARK),
    ('dark', True, True, 'G-dunkel_R-hell-explizit', READER_LIGHT),
    ('dark', True, False, 'G-dunkel_R-dunkel', READER_DARK),
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

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


def rgb(px):
    return f'rgb({px[0]}, {px[1]}, {px[2]})'


def corner_pixels(png_bytes):
    """Four screenshot corners; the top-right one sits below the fixed 'Aa'
    trigger (a UI overlay, not a surface between body and iframe)."""
    im = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    w, h = im.size
    pts = {'top-left': (3, 3), 'bottom-left': (3, h - 4),
           'bottom-right': (w - 4, h - 4), 'right-middle': (w - 4, h // 2)}
    return {k: rgb(im.getpixel(v)) for k, v in pts.items()}


def load(page, global_theme, mode_on, reader_dark):
    page.evaluate("""([g, on, d]) => {
        localStorage.setItem('globalTheme', g);
        localStorage.setItem('readerPrefs', JSON.stringify({modeOn: on, dark: d, width: 'medium'}));
    }""", [global_theme, mode_on, reader_dark])
    page.goto(f'{BASE}/')
    page.wait_for_function(
        "() => document.querySelector('#preview-iframe')?.contentDocument?.body?.children.length > 0")
    page.wait_for_timeout(400)


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


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1200, 'height': 900})
    page.goto(f'{BASE}/login')
    page.fill('#username', USER)
    page.fill('#password', PASSWORD)
    page.click('button[type=submit]')
    page.wait_for_url(lambda url: '/login' not in url)
    print('logged in as', USER)

    print('=== 1. reader-mode ON: every surface between <body> and the iframe document ===')
    for g, d, click_dark, label, expected in READER_STATES:
        load(page, g, True, d)
        if click_dark:
            page.evaluate("() => toggleDarkMode()")
            page.wait_for_timeout(300)
        m = page.evaluate(MEASURE)
        shot = page.screenshot(path=f'{OUT}_{label}.png')
        corners = corner_pixels(shot)
        print(f'[{label}] expected={expected} ' + json.dumps(m) + ' corners=' + json.dumps(corners))
        check(m['readerMode'], f'{label}: reader-mode is on')
        surfaces = {k: m[k] for k in ('main', 'pane', 'container', 'iframe_el', 'doc_html')}
        wrong = {k: v for k, v in surfaces.items() if v != expected}
        check(not wrong, f'{label}: main/pane/container/iframe/doc all {expected}'
                         + (f' — off: {wrong}' if wrong else ''))
        check(m['main_before'] == 'none' and m['main_padding'] == '0px',
              f'{label}: no surface card (::before none) and no global padding')
        wrong_px = {k: v for k, v in corners.items() if v != expected}
        check(not wrong_px, f'{label}: screenshot corners are the reader tone'
                            + (f' — off: {wrong_px}' if wrong_px else ''))

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

    browser.close()

print()
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('ALL PASSED')
