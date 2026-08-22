#!/usr/bin/env python3
"""Browser smoke for the audio-converter file tab (SYNC-FREEZE P3).

The pytest suite renders no templates and runs no JS — this is the one
REAL-browser check of ``templates/audio_converter.html`` +
``static/js/audio_converter.js`` after the transcription became a job on the
worker. It runs INSIDE the web container (the playwright base image ships
Chromium) against the deployed app and drives the page like a user:

1. login, file tab, language Deutsch, a real recording → job + poll until
   the transcript is on the page (elapsed counter sampled on the button);
2. the same file again → the dedup note, zero polls;
3. "In Library speichern" → the job row moves into the inbox (checked via
   the API);
4. a second recording → the page is LEFT while the job runs (what a closed
   tab does) → the library detail of the new row shows "Wird transkribiert …",
   polls, and reloads with the transcript — the tab-close survival the job
   was built for.

How to run (Mintbox; costs Deepgram money — two recordings):

    # 1. throwaway user — NEVER Oli's account
    docker exec markdown-converter-web flask --app app create-user zz_smoke --password '<random>'
    # 2. inputs + script into the web container
    docker cp long.wav  markdown-converter-web:/tmp/smoke_long.wav
    docker cp short.wav markdown-converter-web:/tmp/smoke_short.wav
    docker cp scripts/smoke_audio_converter.py markdown-converter-web:/tmp/smoke_audio.py
    # 3. run — screenshots land in the container as SMOKE_OUT_<step>.png
    docker exec -e SMOKE_USER=zz_smoke -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_audio.py
    # 4. clean up STRICTLY by user_id (the api_token table carries Oli's iOS
    #    tokens): the user's Conversion rows, ApiToken rows, the User row via
    #    the ORM; then /tmp/smoke* from the container.

Env: BASE_URL (default http://localhost:5000 — the container's own port),
SMOKE_LONG (/tmp/smoke_long.wav), SMOKE_SHORT (/tmp/smoke_short.wav),
SMOKE_OUT (/tmp/smoke_audio). Every step prints what the page showed; a
failure is diagnosable from the output alone. Clicks are human-paced (1 s).
On a user who already transcribed SMOKE_LONG, step 1 dedups (polls=0) — the
job path is then NOT exercised by step 1; the script says so.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
LONG = os.environ.get('SMOKE_LONG', '/tmp/smoke_long.wav')
SHORT = os.environ.get('SMOKE_SHORT', '/tmp/smoke_short.wav')
OUT = os.environ.get('SMOKE_OUT', '/tmp/smoke_audio')

DEDUP_NOTE = 'Diese Datei war schon umgewandelt'
RESULT_OR_ALERT = ("() => (document.getElementById('transcription-result-container').textContent.trim().length > 200"
                   " && !document.getElementById('transcription-result-container').textContent.includes('wird verarbeitet'))"
                   " || document.querySelector('#file-alert-container .c-alert') !== null")

failures = []


def check(ok, what):
    print(('  ✓ ' if ok else '  ✗ ') + what, flush=True)
    if not ok:
        failures.append(what)


def text_of(page, selector):
    el = page.query_selector(selector)
    return el.inner_text().strip() if el else ''


def api_json(page, path):
    return page.evaluate("(p) => fetch(p).then(r => r.json())", path)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        polls = []
        submits = []
        page.on('request', lambda req: (polls.append(req.url)
                                        if req.method == 'GET' and '/api/transcriptions/' in req.url else None))
        page.on('response', lambda resp: (submits.append(resp)
                                          if resp.request.method == 'POST' and resp.url.endswith('/api/transcriptions') else None))

        page.goto(f'{BASE}/login')
        page.fill('input[name=username]', USER)
        page.fill('input[name=password]', PASSWORD)
        page.click('button[type=submit]')
        page.wait_for_url(lambda url: '/login' not in url)
        page.goto(f'{BASE}/audio-converter')
        page.click('button[data-tab=file]')
        page.click('button.language-btn[data-lang=de]')
        page.wait_for_timeout(500)
        page.screenshot(path=f'{OUT}_0_page.png')

        def transcribe(path, label, timeout_ms):
            polls.clear()
            submits.clear()
            page.set_input_files('#file-upload-input', path)
            page.wait_for_timeout(1000)
            t0 = time.monotonic()
            page.click('#transcribe-file-btn')
            page.wait_for_timeout(3000)
            label_sample = text_of(page, '#transcribe-file-btn')
            page.wait_for_function(RESULT_OR_ALERT, timeout=timeout_ms)
            elapsed = time.monotonic() - t0
            page.wait_for_timeout(500)
            result = text_of(page, '#transcription-result-container')
            alert = text_of(page, '#file-alert-container')
            note = page.query_selector('#transcription-dedup-note')
            note_visible = bool(note) and note.is_visible()
            page.screenshot(path=f'{OUT}_{label}.png')
            submit_status = submits[0].status if submits else None
            print(f'[{label}] elapsed={elapsed:.1f}s polls={len(polls)} submit={submit_status} '
                  f'button@3s={label_sample!r} result_chars={len(result)} '
                  f'dedup_note={note_visible} alert={alert!r}', flush=True)
            return {'result': result, 'alert': alert, 'polls': len(polls), 'elapsed': elapsed,
                    'note': note_visible, 'button': label_sample, 'submit': submit_status}

        # 1. the long recording → job + poll
        r1 = transcribe(LONG, '1_long', 900_000)
        check(r1['result'] and not r1['alert'], 'long recording: transcript on the page, no alert')
        if r1['polls'] == 0 and r1['note']:
            print('  (this user had already transcribed the file — step 1 dedup-ed, job path not exercised)')
        else:
            check(r1['polls'] >= 1, f'long recording ran as a job ({r1["polls"]} polls)')
            check(r1['button'].startswith('Wird umgewandelt'), f'elapsed counter on the button: {r1["button"]!r}')
            check(r1['submit'] == 202, 'submit answered 202')
        check(text_of(page, '#transcribe-file-btn') == 'Datei umwandeln', 'button label restored')
        check(page.query_selector('#save-transcription-btn').is_visible(), 'save button visible')
        first_result = r1['result']

        # 2. the same file again → dedup note, no polling
        r2 = transcribe(LONG, '2_dedup', 60_000)
        check(r2['note'], 'dedup note shown on the second submit')
        check(r2['polls'] == 0 and r2['submit'] == 200, 'no polling, submit answered 200 (stored result)')
        check(r2['result'] == first_result, 'same transcript text')

        # 3. save → /place inbox
        page.wait_for_timeout(1000)
        page.click('#save-transcription-btn')
        page.wait_for_function("() => document.getElementById('save-transcription-btn').textContent.includes('Gespeichert')",
                               timeout=15_000)
        check(True, f'save button: {text_of(page, "#save-transcription-btn")!r}')
        saved = api_json(page, '/api/conversions?type=audio_transcription&status=inbox&limit=5')
        items = saved.get('items') if isinstance(saved, dict) else saved
        check(bool(items), f'job row now in the inbox ({len(items or [])} inbox transcript(s))')
        page.screenshot(path=f'{OUT}_3_saved.png')

        # 4. second recording, leave the page mid-job → library detail survives
        page.click('#clear-file-btn')
        page.wait_for_timeout(1000)
        submits.clear()
        page.set_input_files('#file-upload-input', SHORT)
        page.wait_for_timeout(1000)
        page.click('#transcribe-file-btn')
        deadline = time.monotonic() + 30
        while not submits and time.monotonic() < deadline:
            page.wait_for_timeout(200)
        check(bool(submits) and submits[0].status == 202, 'second recording submitted as a job')
        job_id = submits[0].json().get('id') if submits else None
        page.wait_for_timeout(1500)
        page.goto(f'{BASE}/library')  # the tab is "gone" — the page's poll dies here
        page.wait_for_timeout(1000)
        t0 = time.monotonic()
        page.goto(f'{BASE}/library/{job_id}')
        pending = page.query_selector('#transcription-status-pending')
        pending_visible = bool(pending) and pending.is_visible()
        print(f'[4_detail] pending block visible on first open: {pending_visible} '
              f'({text_of(page, "#transcription-status-pending")!r})', flush=True)
        page.screenshot(path=f'{OUT}_4_pending.png')
        page.wait_for_function("() => document.querySelector('article.reader-view') && "
                               "document.querySelector('article.reader-view').innerText.trim().length > 100 && "
                               "!document.getElementById('transcription-status-pending')",
                               timeout=600_000)
        elapsed = time.monotonic() - t0
        body = page.query_selector('article.reader-view').inner_text().strip()
        page.screenshot(path=f'{OUT}_4_ready.png')
        print(f'[4_detail] transcript visible after {elapsed:.1f}s, {len(body)} chars', flush=True)
        check(len(body) > 100, 'library detail reloaded with the transcript of the closed-tab job')
        state = api_json(page, f'/api/transcriptions/{job_id}')
        check(state.get('status') == 'ready' and state.get('lifecycle_status') == 'archive',
              f'row {job_id}: status={state.get("status")} place={state.get("lifecycle_status")}')

        browser.close()

    print(f'\n{len(failures)} failure(s)' if failures else '\nall steps passed', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
