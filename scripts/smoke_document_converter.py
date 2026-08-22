#!/usr/bin/env python3
"""Browser smoke for the document-converter page (DOC-WEB-ASYNC P2/P3).

The pytest suite renders no templates and runs no JS — this is the one
REAL-browser check of ``templates/document_converter.html`` +
``static/js/document_converter.js``. It runs INSIDE the web container (the
playwright base image ships Chromium — nothing to install, nothing typed into
a browser a person uses) against the deployed app and drives the page like a
user: login, PDF → service job + poll, the same PDF again → dedup note,
"In Library speichern" → the job row moves to the inbox, then a TXT through
the synchronous route. In its first sprint it caught two things pytest can
never see: the missing dedup note and the asgiref keep-alive deadlock
(Backlog SYNC-FREEZE).

How to run (Mintbox, ~2 min — one real mineru run on a fresh user):

    # 1. throwaway user — NEVER Oli's account; the password only ever
    #    reaches this headless Chromium via -e
    docker exec markdown-converter-web flask --app app create-user zz_smoke --password '<random>'
    # 2. inputs + script into the web container (the corpus is not in the
    #    image; any PDF that converts in ~1 min does, e.g. 12 pages of
    #    corpus/05_scan-sauber)
    docker cp scan12.pdf markdown-converter-web:/tmp/scan12.pdf
    printf 'Notiz zum Smoke.\\n' > note.txt && docker cp note.txt markdown-converter-web:/tmp/note.txt
    docker cp scripts/smoke_document_converter.py markdown-converter-web:/tmp/smoke.py
    # 3. run — screenshots land in the container as SMOKE_OUT_<step>.png
    #    (docker cp them out); exit 0 = every step passed
    docker exec -e SMOKE_USER=zz_smoke -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke.py
    # 4. clean up STRICTLY by user_id — the api_token table carries Oli's
    #    iOS tokens: delete the user's Conversion rows, ApiToken rows and the
    #    User row via the ORM (see the DOC-WEB-ASYNC sprint report), then
    #    remove /tmp/smoke* and /tmp/scan12.pdf from the container.

Env: BASE_URL (default http://localhost:5000 — the container's own port),
SMOKE_PDF (/tmp/scan12.pdf), SMOKE_TXT (/tmp/note.txt), SMOKE_OUT (/tmp/smoke).
Every step prints what the page showed (result size, poll count, note,
degradations, alert text, button label) — a failure is diagnosable from the
output alone. On a user who already converted the PDF, the first run dedups
(polls=0): the job path is then NOT exercised — the script says so.

⚠️ The save click is deliberately human-paced (1 s): a click within
milliseconds of the previous response on the same keep-alive connection hits
the asgiref/uvicorn ``would deadlock`` 500 (SYNC-FREEZE) — a server defect
this smoke documents, not a page bug.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
PDF = os.environ.get('SMOKE_PDF', '/tmp/scan12.pdf')
TXT = os.environ.get('SMOKE_TXT', '/tmp/note.txt')
OUT = os.environ.get('SMOKE_OUT', '/tmp/smoke')

DEDUP_NOTE = 'Diese Datei war schon umgewandelt'
RESULT_OR_ALERT = ("() => !document.getElementById('result-area').classList.contains('hidden')"
                   " || !!document.querySelector('#alert-container .c-alert')")
failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


def text_if_visible(page, selector):
    return page.inner_text(selector).strip() if page.is_visible(selector) else ''


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1100, 'height': 1000})
    polls = []
    page.on('request', lambda req: polls.append(req.url)
            if req.method == 'GET' and '/api/document-conversions/' in req.url else None)

    page.goto(f'{BASE}/login')
    page.fill('#username', USER)
    page.fill('#password', PASSWORD)
    page.click('button[type=submit]')
    page.wait_for_url(lambda url: '/login' not in url)
    page.goto(f'{BASE}/document-converter')
    print('logged in, on', page.url)

    def convert(path, label, timeout_ms, sample_button_after=None):
        page.set_input_files('#document_file', path)
        polls.clear()
        t0 = time.time()
        page.click('#convert-btn')
        if sample_button_after:
            page.wait_for_timeout(sample_button_after)
            print(f'[{label}] button after {sample_button_after} ms: '
                  f'{page.inner_text("#convert-btn")!r}')
        page.wait_for_function(RESULT_OR_ALERT, timeout=timeout_ms)
        dt = time.time() - t0
        result = page.inner_text('#result-content') if page.is_visible('#result-area') else ''
        state = {
            'seconds': round(dt, 1), 'result_chars': len(result), 'polls': len(polls),
            'note': text_if_visible(page, '#result-note'),
            'degradations': text_if_visible(page, '#degradation-list'),
            'alert': text_if_visible(page, '#alert-container'),
            'button': page.inner_text('#convert-btn').strip(),
        }
        print(f'[{label}] ' + ' · '.join(f'{k}={v!r}' for k, v in state.items()))
        page.screenshot(path=f'{OUT}_{label}.png', full_page=True)
        return result, state

    first, st = convert(PDF, 'pdf_first', 15 * 60 * 1000, sample_button_after=4000)
    check(bool(first) and not st['alert'], 'PDF converts through the service')
    check(st['button'] == 'Dokument umwandeln', 'convert button resets after the job')
    if st['note'].startswith(DEDUP_NOTE):
        print('  NOTE first run was already deduped for this user — the job path '
              '(poll + worker) was NOT exercised; use a fresh throwaway user')
    else:
        check(st['polls'] >= 1, 'job was polled (GET /api/document-conversions/<id>)')

    page.click('#clear-file')
    second, st = convert(PDF, 'pdf_dedup', 60 * 1000)
    check(st['note'].startswith(DEDUP_NOTE), 'dedup is SAID on the result')
    check(st['polls'] == 0 and second == first, 'dedup served the stored result without a job')

    # Human-paced click — see the module docstring (SYNC-FREEZE race).
    page.wait_for_timeout(1000)
    page.click('#save-btn')
    try:
        page.wait_for_function(
            "() => document.getElementById('save-btn').textContent.includes('Gespeichert')",
            timeout=30 * 1000)
        saved = True
    except Exception:
        saved = False
    print(f'[save] button={page.inner_text("#save-btn").strip()!r} · '
          f'alert={text_if_visible(page, "#alert-container")!r}')
    check(saved, '"In Library speichern" moves the job row into the inbox')
    page.screenshot(path=f'{OUT}_saved.png', full_page=True)

    page.click('#clear-file')
    txt, st = convert(TXT, 'txt_sync', 60 * 1000)
    check(bool(txt) and st['polls'] == 0 and not st['alert'],
          'TXT still converts synchronously (no job)')
    browser.close()

if failures:
    sys.exit(f'{len(failures)} step(s) failed: ' + '; '.join(failures))
print('all steps passed')
