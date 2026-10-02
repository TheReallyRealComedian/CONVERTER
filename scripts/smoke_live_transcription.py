#!/usr/bin/env python3
"""Browser smoke for the LIVE transcription (SEC-DG-TOKEN).

The pytest suite runs no JS and talks to no Deepgram — this is the one check
of the live path against the REAL stack: the deployed view, Deepgram's grant,
a real Chromium with a FAKE microphone (a speech WAV played in a loop) and
the real WebSocket to ``api.deepgram.com``. It runs INSIDE the web container
(the playwright base image ships Chromium) as a THROWAWAY user.

What it shows:

A. server — ``GET /api/get-deepgram-token`` answers 200 with a JWT-shaped
   token whose sha256 differs from the sha256 of the configured key, the key
   is nowhere in the body, ``expires_in`` is the server TTL; with that token
   Deepgram's management API (``/v1/projects``) and a further grant refuse.
B. browser — the token is fetched AFTER the mic permission resolved, the
   WebSocket opens with first subprotocol ``bearer``, transcript text arrives,
   the recording runs LONGER than the token lives and text keeps arriving
   after the expiry; stop ends cleanly; a second recording fetches a second,
   different token.
C. browser whose microphone cannot be started — no token is requested, no
   socket opened. (Headless Chromium without the fake-UI flag: getUserMedia
   rejects. A real permission denial, ``NotAllowedError``, cannot be produced
   there — CDP ``Browser.setPermission`` leaves the state at ``prompt``, on
   the page and on the browser session, measured — but every getUserMedia
   error leaves through the same catch block, before the token fetch.)

No key, token or JWT is ever printed: lengths, shapes, hash comparisons and
claim NAMES only. The WebSocket recorder keeps ``protocols[0]`` and nothing
else of the subprotocol list.

How to run (Mintbox; costs about 1.5 min of Deepgram streaming):

    # 1. throwaway user — NEVER Oli's account; the password only travels via -e
    docker exec markdown-converter-web flask --app app create-user zz_smoke --password '<random>'
    # 2. a speech WAV for the fake mic, cut inside the container from an
    #    existing recording (ffmpeg is in the image), and the script — streamed
    #    in, never docker cp (SEC-NONROOT)
    docker exec markdown-converter-web ffmpeg -loglevel error -ss 5 -t 50 \
        -i /app/output_podcasts/narration_<id>.wav -ar 48000 -ac 1 -c:a pcm_s16le /tmp/smoke_live.wav
    docker exec -i markdown-converter-web sh -c 'cat > /tmp/smoke_live.py' < scripts/smoke_live_transcription.py
    # 3. run — exit 0 = every check held, 1 = a check failed
    docker exec -e SMOKE_USER=zz_smoke -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_live.py
    # 4. clean up STRICTLY by user_id (the api_token table carries Oli's iOS
    #    tokens): the User row via the ORM (the smoke saves nothing, so there
    #    are no Conversion rows); then /tmp/smoke_live* and the /tmp/pulse-*
    #    directory Chromium's audio leaves behind from the container.

Measured twice on 2026-10-02 (identical in every check): an open connection
outlives its token — 75 s of recording against a 30 s token, text arriving
throughout; Deepgram checks the token when the socket is opened. In the web
log each grant shows as one httpx INFO line (URL and status, no token).
⚠️ gunicorn is PID 1 in the web container: Chromium helper processes orphaned
at ``browser.close()`` are reaped by it and logged as
``Worker (pid:N) was sent SIGTERM!`` — those are not gunicorn workers.

Env: BASE_URL (default http://localhost:5000 — the container's own port; a
``localhost`` origin is a secure context, which getUserMedia needs),
SMOKE_WAV (/tmp/smoke_live.wav), SMOKE_RECORD_SECONDS (TTL + 45),
SMOKE_APP_ROOT (cwd — docker exec starts in /app). Every step prints what it
measured; a failure is diagnosable from the output alone.
"""
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
WAV = os.environ.get('SMOKE_WAV', '/tmp/smoke_live.wav')

# The script is streamed to /tmp; the app lives in the container's WORKDIR
# (/app) — docker exec starts there, so the cwd is the app root.
sys.path.insert(0, os.environ.get('SMOKE_APP_ROOT', os.getcwd()))
from app_pkg.config import DEEPGRAM_LIVE_TOKEN_TTL_SECONDS as TTL  # noqa: E402

RECORD_SECONDS = int(os.environ.get('SMOKE_RECORD_SECONDS', str(TTL + 45)))
SECOND_RECORD_SECONDS = 10
TOKEN_ROUTE = '/api/get-deepgram-token'
JWT_SHAPE = re.compile(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+')

if not os.path.exists(WAV):
    sys.exit(f'{WAV} missing — cut a speech WAV first (see the docstring)')
API_KEY = os.environ.get('DEEPGRAM_API_KEY') or sys.exit(
    'DEEPGRAM_API_KEY not in this container — run inside the web container')
KEY_SHA = hashlib.sha256(API_KEY.encode()).hexdigest()

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what, flush=True)
    if not ok:
        failures.append(what)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def deepgram_status(method, path, token, body=None):
    """HTTP status Deepgram answers when the granted token is presented."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        'https://api.deepgram.com' + path, method=method, data=data,
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def claims_of(token):
    """Claim names of the JWT and the seconds until its own ``exp`` — never
    another claim value. (The token carries no ``iat``; the remaining
    lifetime is read against this container's clock.)"""
    try:
        payload = token.split('.')[1]
        payload += '=' * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None, None
    exp = claims.get('exp')
    remaining = round(exp - time.time(), 1) if isinstance(exp, (int, float)) else None
    return sorted(claims), remaining


# Runs before any page script. Records WHEN the mic resolved, WHEN the token
# was requested / answered and how the WebSocket was opened — timestamps,
# statuses and the FIRST subprotocol only; never a token.
INIT_SCRIPT = r"""
(() => {
  const log = { gum: [], token: [], ws: [] };
  window.__liveSmoke = log;

  const md = navigator.mediaDevices;
  if (md && md.getUserMedia) {
    const nativeGum = md.getUserMedia.bind(md);
    md.getUserMedia = async (constraints) => {
      const e = { t_call: Date.now(), t_resolved: null, error: null };
      log.gum.push(e);
      try {
        const stream = await nativeGum(constraints);
        e.t_resolved = Date.now();
        return stream;
      } catch (err) {
        e.error = (err && err.name) || 'error';
        throw err;
      }
    };
  }

  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : ((input && input.url) || '');
    if (!url.includes('/api/get-deepgram-token')) return nativeFetch(input, init);
    const e = { t_request: Date.now(), t_response: null, status: null };
    log.token.push(e);
    const resp = await nativeFetch(input, init);
    e.t_response = Date.now();
    e.status = resp.status;
    return resp;
  };

  const Native = window.WebSocket;
  window.WebSocket = class extends Native {
    constructor(url, protocols) {
      super(url, protocols);
      const u = new URL(url);
      const list = Array.isArray(protocols) ? protocols : (protocols == null ? [] : [protocols]);
      const first = list.length ? String(list[0]) : null;
      const e = {
        t_new: Date.now(), host: u.host, path: u.pathname,
        // a scheme word, never a credential: anything else is redacted
        protocol0: first === null ? null : (/^[a-z]{1,16}$/.test(first) ? first : '<redacted>'),
        n_protocols: list.length,
        t_open: null, t_close: null, close_code: null, close_reason: null, was_clean: null,
        messages: 0, t_first_message: null, t_last_message: null,
      };
      log.ws.push(e);
      this.addEventListener('open', () => { e.t_open = Date.now(); });
      this.addEventListener('message', () => {
        e.messages += 1;
        const now = Date.now();
        if (e.t_first_message === null) e.t_first_message = now;
        e.t_last_message = now;
      });
      this.addEventListener('close', (ev) => {
        e.t_close = Date.now();
        e.close_code = ev.code;
        e.close_reason = String(ev.reason || '').slice(0, 120);
        e.was_clean = ev.wasClean;
      });
    }
  };
})();
"""

STATE_JS = """() => {
  const mic = document.getElementById('mic-button');
  const out = document.getElementById('live-transcript-output');
  const alertBox = document.getElementById('live-alert-container');
  return {
    now: Date.now(),
    log: window.__liveSmoke,
    chars: out ? out.value.length : -1,
    head: out ? out.value.slice(0, 60) : '',
    recording: !!mic && mic.classList.contains('recording'),
    loading: !!mic && mic.classList.contains('mic-loading'),
    alert: alertBox ? alertBox.innerText.trim() : '',
  };
}"""

STARTED_OR_ALERT = """() => document.getElementById('mic-button').classList.contains('recording')
  || document.getElementById('live-alert-container').innerText.trim().length > 0"""


def login(page):
    page.goto(f'{BASE}/login')
    page.fill('input[name=username]', USER)
    page.fill('input[name=password]', PASSWORD)
    page.click('button[type=submit]')
    page.wait_for_url(lambda url: '/login' not in url)


def open_live_tab(page):
    page.goto(f'{BASE}/audio-converter')
    page.click('button.language-btn[data-lang=de]')
    page.wait_for_timeout(1000)


def seconds(ms_a, ms_b):
    return None if ms_a is None or ms_b is None else round((ms_a - ms_b) / 1000, 2)


def server_evidence(context):
    print('\n[A] server: the view and what the token can do', flush=True)
    resp = context.request.get(f'{BASE}{TOKEN_ROUTE}')
    body_text = resp.text()
    check(resp.status == 200, f'view answered 200 (got {resp.status})')
    if resp.status != 200:
        print(f'  body: {body_text[:200]!r}')
        return
    body = json.loads(body_text)
    token = body.get('deepgram_token') or ''
    names, remaining = claims_of(token)
    print(f'  fields={sorted(body)} token_len={len(token)} parts={len(token.split("."))} '
          f'expires_in={body.get("expires_in")!r} cache_control={resp.headers.get("cache-control")!r}\n'
          f'  jwt claim names={names} exp in {remaining} s', flush=True)
    check(bool(JWT_SHAPE.fullmatch(token)), 'deepgram_token has JWT shape (three base64url parts)')
    check(sha(token) != KEY_SHA and token != API_KEY,
          'sha256(token) differs from sha256(configured key)')
    check(API_KEY not in body_text, 'the configured key is nowhere in the response body')
    check(body.get('expires_in') == TTL, f'expires_in is the server TTL ({TTL} s)')
    check(resp.headers.get('cache-control') == 'no-store', 'response is Cache-Control: no-store')

    projects = deepgram_status('GET', '/v1/projects', token)
    regrant = deepgram_status('POST', '/v1/auth/grant', token, {'ttl_seconds': 3600})
    print(f'  with the token: GET /v1/projects -> {projects}, POST /v1/auth/grant -> {regrant}',
          flush=True)
    check(projects == 403, 'the token cannot read the project management API (403)')
    check(regrant == 403, 'the token cannot mint a further token (403)')


def record(page, label, run_seconds):
    """One live recording: start, sample, stop. Returns the measurements."""
    before = page.evaluate(STATE_JS)
    n_tokens, n_ws, n_gum = (len(before['log']['token']), len(before['log']['ws']),
                             len(before['log']['gum']))
    page.click('#mic-button')
    page.wait_for_function(STARTED_OR_ALERT, timeout=30_000)
    state = page.evaluate(STATE_JS)
    if not state['recording']:
        print(f'[{label}] did not start: alert={state["alert"]!r} gum={state["log"]["gum"][n_gum:]} '
              f'token={state["log"]["token"][n_tokens:]} ws={state["log"]["ws"][n_ws:]}', flush=True)
        check(False, f'{label}: recording started')
        return None

    gum = state['log']['gum'][n_gum]
    tok = state['log']['token'][n_tokens]
    ws = state['log']['ws'][n_ws]
    t_token = tok['t_response']
    print(f'[{label}] started: mic resolved {seconds(gum["t_resolved"], gum["t_call"])} s after the click path, '
          f'token requested {seconds(tok["t_request"], gum["t_resolved"])} s after the mic, '
          f'answered in {seconds(tok["t_response"], tok["t_request"])} s (status {tok["status"]}), '
          f'socket created {seconds(ws["t_new"], tok["t_response"])} s after the token, '
          f'open {seconds(ws["t_open"], ws["t_new"])} s later\n'
          f'  ws host={ws["host"]} path={ws["path"]} protocols[0]={ws["protocol0"]!r} '
          f'n_protocols={ws["n_protocols"]}', flush=True)

    samples = []
    while True:
        page.wait_for_timeout(3000)
        state = page.evaluate(STATE_JS)
        cur = state['log']['ws'][n_ws]
        since = seconds(state['now'], t_token)
        samples.append({'t': since, 'chars': state['chars'], 'messages': cur['messages'],
                        'open': cur['t_close'] is None, 'recording': state['recording']})
        print(f'  t=+{since:5.1f}s after the token: chars={state["chars"]:5d} '
              f'ws_messages={cur["messages"]:4d} ws_open={cur["t_close"] is None} '
              f'recording={state["recording"]}', flush=True)
        if since >= run_seconds or not state['recording']:
            break

    ran = samples[-1]['t']
    still_recording = samples[-1]['recording']
    if still_recording:
        page.click('#mic-button')  # stop
        page.wait_for_function(
            "() => !document.getElementById('mic-button').classList.contains('recording')",
            timeout=10_000)
    page.wait_for_timeout(1500)
    end = page.evaluate(STATE_JS)
    ws_end = end['log']['ws'][n_ws]
    print(f'[{label}] stopped after {ran} s: chars={end["chars"]} head={end["head"]!r}\n'
          f'  ws closed {seconds(ws_end["t_close"], t_token)} s after the token, '
          f'code={ws_end["close_code"]} clean={ws_end["was_clean"]} reason={ws_end["close_reason"]!r} '
          f'messages={ws_end["messages"]} last message at +{seconds(ws_end["t_last_message"], t_token)} s\n'
          f'  alert={end["alert"]!r} token requests this recording='
          f'{len(end["log"]["token"]) - n_tokens} sockets={len(end["log"]["ws"]) - n_ws}', flush=True)

    return {'gum': gum, 'tok': tok, 'ws': ws, 'ws_end': ws_end, 'samples': samples,
            'chars_before': before['chars'], 'chars_end': end['chars'], 'alert': end['alert'],
            'still_recording': still_recording, 'ran': ran,
            'tokens_this_recording': len(end['log']['token']) - n_tokens,
            'sockets_this_recording': len(end['log']['ws']) - n_ws}


def common_checks(label, r):
    check(r['gum']['t_resolved'] is not None
          and r['gum']['t_resolved'] <= r['tok']['t_request'] <= r['tok']['t_response'] <= r['ws']['t_new'],
          f'{label}: order is mic → token → socket')
    check(r['tok']['status'] == 200, f'{label}: token fetch answered 200')
    check(r['ws']['host'] == 'api.deepgram.com' and r['ws']['path'] == '/v1/listen',
          f'{label}: WebSocket goes to api.deepgram.com/v1/listen')
    check(r['ws']['protocol0'] == 'bearer' and r['ws']['n_protocols'] == 2,
          f'{label}: WebSocket opened with first subprotocol "bearer"')
    check(r['ws']['t_open'] is not None, f'{label}: WebSocket opened')
    check(r['chars_end'] > r['chars_before'], f'{label}: transcript text arrived '
          f'({r["chars_before"]} → {r["chars_end"]} chars)')
    check(r['tokens_this_recording'] == 1 and r['sockets_this_recording'] == 1,
          f'{label}: exactly one token and one socket')
    check(r['alert'] == '', f'{label}: no alert on the page')
    check(r['ws_end']['t_close'] is not None, f'{label}: stop closed the socket')


def live_evidence(p):
    print(f'\n[B] browser: live recording with a fake mic ({WAV} in a loop), '
          f'{RECORD_SECONDS} s against a {TTL} s token', flush=True)
    browser = p.chromium.launch(args=[
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        f'--use-file-for-fake-audio-capture={WAV}',
    ])
    try:
        context = browser.new_context()
        context.add_init_script(INIT_SCRIPT)
        page = context.new_page()
        token_responses = []
        page.on('response', lambda resp: (token_responses.append(resp)
                                          if TOKEN_ROUTE in resp.url else None))
        login(page)
        server_evidence(context)

        print('\n[B] continued', flush=True)
        open_live_tab(page)

        first = record(page, 'recording 1', RECORD_SECONDS)
        if first is not None:
            common_checks('recording 1', first)
            after = [s for s in first['samples'] if s['t'] > TTL]
            at_expiry = max((s for s in first['samples'] if s['t'] <= TTL),
                            key=lambda s: s['t'], default=None)
            closed_at = seconds(first['ws_end']['t_close'], first['tok']['t_response'])
            check(first['still_recording'] and first['ran'] > TTL,
                  f'recording 1 ran past the token lifetime without stopping itself '
                  f'({first["ran"]} s > {TTL} s; socket closed at +{closed_at} s, by the stop)')
            check(bool(after) and all(s['open'] for s in after),
                  'recording 1: the socket stayed open after the token expired')
            check(at_expiry is not None and bool(after)
                  and after[-1]['chars'] > at_expiry['chars']
                  and after[-1]['messages'] > at_expiry['messages'],
                  'recording 1: text kept arriving after the expiry '
                  + (f'({at_expiry["chars"]} chars at +{at_expiry["t"]} s → '
                     f'{after[-1]["chars"]} chars at +{after[-1]["t"]} s)'
                     if at_expiry and after else '(no samples)'))

        page.wait_for_timeout(2000)
        second = record(page, 'recording 2', SECOND_RECORD_SECONDS)
        if second is not None:
            common_checks('recording 2', second)

        hashes = []
        for resp in token_responses:
            try:
                token = resp.json().get('deepgram_token') or ''
            except Exception:
                token = ''
            hashes.append(sha(token) if token else None)
        print(f'\n[B] token fetches by the page: {len(token_responses)}, '
              f'statuses={[r.status for r in token_responses]}, '
              f'distinct hashes={len(set(hashes))}, any equal to the key hash='
              f'{any(h == KEY_SHA for h in hashes)}', flush=True)
        check(len(token_responses) == 2 and all(h for h in hashes) and len(set(hashes)) == 2,
              'two recordings fetched two tokens with two different hashes')
        check(all(h != KEY_SHA for h in hashes), 'neither fetched token is the configured key')
    finally:
        browser.close()


def no_mic_evidence(p):
    print('\n[C] browser whose mic cannot be started: no grant, no socket', flush=True)
    browser = p.chromium.launch(args=['--use-fake-device-for-media-stream'])
    try:
        context = browser.new_context()
        context.add_init_script(INIT_SCRIPT)
        page = context.new_page()
        login(page)
        open_live_tab(page)
        page.click('#mic-button')
        try:
            page.wait_for_function(STARTED_OR_ALERT, timeout=10_000)
        except Exception:
            pass  # a getUserMedia that never settles shows no alert either
        page.wait_for_timeout(1000)
        state = page.evaluate(STATE_JS)
        log = state['log']
        print(f'  gum={[(g["error"], g["t_resolved"] is not None) for g in log["gum"]]} '
              f'token requests={len(log["token"])} sockets={len(log["ws"])} '
              f'recording={state["recording"]} alert={state["alert"]!r}', flush=True)
        check(len(log['gum']) == 1 and log['gum'][0]['t_resolved'] is None,
              'the mic was asked for and did not start')
        check(not state['recording'] and state['alert'] != '',
              'no recording, the page shows an alert')
        check(len(log['token']) == 0, 'no token was requested')
        check(len(log['ws']) == 0, 'no WebSocket was opened')
    finally:
        browser.close()


def main():
    with sync_playwright() as p:
        live_evidence(p)
        no_mic_evidence(p)

    print()
    if failures:
        print(f'{len(failures)} FAILED:')
        for f in failures:
            print('  -', f)
        sys.exit(1)
    print('ALL PASSED')


if __name__ == '__main__':
    main()
