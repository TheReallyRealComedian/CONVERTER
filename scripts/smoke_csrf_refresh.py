#!/usr/bin/env python3
"""Browser smoke for CSRF-REFRESH — the CSRF token lives with the session, the
fetch wrapper refreshes once.

The pytest suite never runs the fetch wrapper in ``templates/base.html`` —
this is the one REAL-browser check of what a page that has been standing
for a while does on its first mutation. It runs with Playwright INSIDE the
app image (the playwright base image ships Chromium) as a throwaway user
with ONE own ``markdown_input`` row written through the ORM, opens
``/library/<id>`` and triggers the title save ``updateField('title', …)``
(``static/js/library_detail.js``) through ``page.evaluate`` — no click, so
the page needs no layout (``--network none``, no Tailwind). Every run
records the requests the page makes as a sequence of ``(method, path,
status)`` (``page.on('response')``, same-origin, no ``/static/``) and proves
"not reloaded" twice: a ``window.__smokeMarker`` set before the mutation is
still there afterwards, and ``performance.getEntriesByType('navigation')``
stays at 1. Four scenarios, each on a freshly loaded page:

S1 stale token — ``window.CSRF_TOKEN = 'stale'``, then the mutation.
   head:  exactly ``PUT 400``, NO ``GET /api/csrf-token``, the page's own
          failure alert ("Titel konnte nicht gespeichert werden. …"), no toast.
   fixed: ``PUT 400 → GET /api/csrf-token 200 → PUT 200``, success toast
          "Titel gespeichert", no alert, ``window.CSRF_TOKEN`` is a non-empty
          value ≠ ``'stale'`` afterwards, the title is in the DB.
S2 freshly minted session — ``context.clear_cookies(name='session')`` (the
   ``remember_token`` stays), the page's token unchanged, then the mutation.
   The mechanics are proved first: the ``session`` cookie is really gone and
   the ``remember_token`` really there (names and flags only, never values).
   head:  ``PUT 400`` ("The CSRF session token is missing."), no GET; the
          next page load is NOT a login redirect (the remember cookie mints
          a new session) and the ``session`` cookie is back.
   fixed: ``PUT 400 → GET 200 → PUT 200`` and a ``session`` cookie exists in
          the context again right after the chain.
S3 second failure (gate only; observed in head) — ``page.route`` fulfils
   ``/api/csrf-token`` with ``{"csrf_token": "bogus"}`` 200, token ``'stale'``,
   the mutation → ``PUT 400 → GET 200 → PUT 400`` and NOTHING further (three
   requests, not four; the route was hit exactly once), the failure alert
   with 400, no reload. head: ``PUT 400``, route never hit.
S4 multipart — ``fetch('/api/transcriptions', {method: 'POST', body:
   FormData})`` with a ``.txt`` as ``audio_file`` (``app_pkg/audio.py``): the
   route rejects the extension AFTER the CSRF check with its own 400
   ("Dieses Dateiformat wird nicht unterstützt. …") before anything touches
   the volume — no row, no file, no job. A positive control with the valid
   token proves that text (the file arrived) first.
   head:  stale token → exactly ``POST 400 csrf_expired``, no GET.
   fixed: ``POST 400 csrf_expired → GET 200 → POST 400 <format text>`` — the
          FormData body arrived on the second attempt (a non-replayable body
          breaks exactly here). Zero ``audio_transcription`` rows afterwards.

Two modes, chosen with ``--expect``:

``--expect head`` (Phase 1): the Master's derivations FROM THE CODE; the
smoke prints "Ist bestätigt" / "Ist widerlegt" per derivation and exits 0
either way — it is the evidence, not the gate. Every measured value is
printed (request sequences, the JSON ``message`` behind each 400, alert and
toast texts, cookie names).

``--expect fixed`` (Phase 2 gate, Phase 3 / acceptance): S1–S4 as above,
PASS/FAIL, exit 1 on any FAIL.

The row is written through the ORM under the throwaway user's own
``user_id`` — never through ``POST /api/conversions``. The script REFUSES to
run for user id 1 or the INGEST_USER. Everything it wrote is removed at the
end, strictly by ``user_id`` + the id it wrote (and any
``audio_transcription`` row of that user, which must be none); the user
itself is created and removed by the operator.

How to run — (a) WITHOUT a deploy, on a throwaway instance from the deployed
image with the Mac working tree streamed in (own SQLite under /tmp/w, no
volume, no network; ~1 min). ``zz_first`` is a placeholder so the smoke user
is not user id 1 — the id-1 guard below is the same on every instance.
``DEEPGRAM_API_KEY`` is a FAKE value: ``POST /api/transcriptions`` sits
behind ``require_service('deepgram')`` (503 without a key) and S4 never
reaches Deepgram — the ``.txt`` is rejected before any job exists:

    cd ~/CODE/CONVERTER && git ls-files -co --exclude-standard | grep -v -E '^(corpus|docs)/' > /tmp/cr_files \
    && COPYFILE_DISABLE=1 tar --no-xattrs -cf - -T /tmp/cr_files | ssh mintbox 'docker run -i --rm --network none --entrypoint sh converter-app:latest -c "
        mkdir /tmp/w && tar -xf - -C /tmp/w && cd /tmp/w \
        && export DATABASE_URL=sqlite:////tmp/w/smoke.db SECRET_KEY=smoke-only REDIS_URL=redis://127.0.0.1:9/0 DEEPGRAM_API_KEY=smoke-fake-key \
        && flask --app app create-user zz_first --password placeholder-1234 >/dev/null \
        && flask --app app create-user zz_csrf --password smoke-pass-1234 >/dev/null \
        && (gunicorn --bind 127.0.0.1:5000 --workers 1 --worker-class uvicorn.workers.UvicornWorker app:asgi_app >/tmp/w/gunicorn.log 2>&1 &) \
        && sleep 5 \
        && SMOKE_USER=zz_csrf SMOKE_PASSWORD=smoke-pass-1234 BASE_URL=http://127.0.0.1:5000 python scripts/smoke_csrf_refresh.py --expect head"'

(b) against the DEPLOYED web container (Phase 3 / acceptance, ``--expect fixed``;
S4 creates nothing by construction — no row, no file, no job):

    docker exec markdown-converter-web flask --app app create-user zz_csrf --password '<random>'
    docker exec -i markdown-converter-web sh -c 'cat > /tmp/smoke_csrf_refresh.py' < scripts/smoke_csrf_refresh.py
    docker exec -e SMOKE_USER=zz_csrf -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_csrf_refresh.py --expect fixed
    # clean up STRICTLY by user_id (api_token carries Oli's iOS tokens): the
    # script removed its rows; delete the user's remaining Conversion/ApiToken
    # rows and the User row via the ORM by that user_id, then:
    #   docker exec markdown-converter-web sh -c 'rm -f /tmp/smoke_csrf_refresh.py; rm -rf /tmp/pulse-*'

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SMOKE_APP_ROOT (default: cwd — where ``app.py`` lives). No screenshots are
written, no cookie value or token is ever printed (names, flags, lengths).
"""
import argparse
import json
import os
import secrets
import sys

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--expect', choices=('head', 'fixed'), default='head',
                    help="'head' = confirm/refute the derivations (Phase 1, exit 0); "
                         "'fixed' = the gate after the fixes (PASS/FAIL, exit 1 on FAIL)")
ARGS = parser.parse_args()

BASE = os.environ.get('BASE_URL', 'http://localhost:5000').rstrip('/')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')

CSRF_PATH = '/api/csrf-token'
TRANSCRIPTIONS_PATH = '/api/transcriptions'
# The page's own texts (static/js/library_detail.js SAVE_MESSAGES.title) —
# the sprint adds no microcopy, so these are the texts before and after.
TITLE_SAVED = 'Titel gespeichert'
TITLE_FAILED = 'Titel konnte nicht gespeichert werden. Verbindung prüfen und erneut versuchen.'
# app_pkg/audio.py: the extension gate's own 400, after CSRF, before any write.
FORMAT_REJECTED = ('Dieses Dateiformat wird nicht unterstützt. '
                   'Erlaubt: MP3, WAV, M4A, OGG, FLAC, WEBM.')
MUTATION_TIMEOUT_MS = 15000
SETTLE_MS = 1000          # trailing window after the UI settled: no further request may arrive

failures = []
verdicts = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


def verdict(ok, finding):
    """Phase-1 form: the Master derived a finding from the code; the browser
    confirms or refutes it. Informational — never fails the run."""
    print(('  Ist bestätigt  ' if ok else '  Ist widerlegt  ') + finding)
    verdicts.append((ok, finding))


def note(what):
    print('  beobachtet    ' + what)


# --- test data through the ORM (same container, same DB) -------------------
sys.path.insert(0, os.environ.get('SMOKE_APP_ROOT', os.getcwd()))
from app import app  # noqa: E402
from models import Conversion, User, db  # noqa: E402

TRANSCRIPTION_TYPE = 'audio_transcription'


def resolve_user():
    with app.app_context():
        user = User.query.filter_by(username=USER).first()
        if user is None:
            sys.exit(f'user {USER!r} does not exist — create it with flask create-user')
        if user.id == 1 or USER == os.environ.get('INGEST_USER'):
            sys.exit(f'refusing to run as user id {user.id} ({USER!r}): that is the '
                     'INGEST/first user = Oli\'s account')
        return user.id


def create_row(user_id):
    with app.app_context():
        row = Conversion(user_id=user_id, conversion_type='markdown_input',
                         title='CSRF-REFRESH Smoke',
                         content='# CSRF-REFRESH Smoke\n\nEin Absatz, damit der Reader etwas zeigt.\n',
                         lifecycle_status='archive', metadata_json='{}')
        db.session.add(row)
        db.session.commit()
        return row.id


def db_title(cid, user_id):
    with app.app_context():
        row = Conversion.query.filter_by(id=cid, user_id=user_id).first()
        return row.title if row else None


def transcription_rows(user_id):
    with app.app_context():
        return Conversion.query.filter_by(user_id=user_id,
                                          conversion_type=TRANSCRIPTION_TYPE).count()


def remove_rows(cid, user_id):
    with app.app_context():
        rows = Conversion.query.filter(
            Conversion.user_id == user_id,
            (Conversion.id == cid) | (Conversion.conversion_type == TRANSCRIPTION_TYPE)).all()
        for r in rows:
            db.session.delete(r)
        db.session.commit()
        left = Conversion.query.filter_by(user_id=user_id).count()
        return len(rows), left


# --- page helpers ----------------------------------------------------------
def cookie_names(ctx):
    """Names + flags only — never a value."""
    return sorted((c['name'], 'secure' if c['secure'] else 'plain',
                   'httpOnly' if c['httpOnly'] else 'script', len(c['value']))
                  for c in ctx.cookies())


def has_cookie(ctx, name):
    return any(c['name'] == name for c in ctx.cookies())


class RequestLog:
    """Same-origin, non-static responses in arrival order; the Response
    objects are kept so the JSON ``message`` behind a 400 can be read after
    the scenario settled (never inside the event handler)."""

    def __init__(self, page):
        self.entries = []
        page.on('response', self._on_response)

    def _on_response(self, resp):
        url = resp.url
        if not url.startswith(BASE):
            return
        path = url[len(BASE):].split('?')[0]
        if path.startswith('/static/'):
            return
        self.entries.append((resp.request.method, path, resp.status, resp))

    def clear(self):
        self.entries = []

    def sequence(self):
        return [(m, p, s) for m, p, s, _ in self.entries]

    def describe(self):
        parts = []
        for m, p, s, resp in self.entries:
            msg = ''
            if p.startswith('/api/') and s >= 400:
                try:
                    body = json.loads(resp.text())
                except Exception:  # noqa: BLE001 — body not readable: say so, do not fail
                    body = None
                if isinstance(body, dict):
                    message = body.get('message')
                    msg = f' [{body.get("error")}' + (f': {message}' if message else '') + ']'
                else:
                    msg = ' [body not JSON]'
            parts.append(f'{m} {p} {s}{msg}')
        return ' → '.join(parts) if parts else '<keine Anfrage>'


def login(page):
    page.goto(f'{BASE}/login')
    page.fill('#username', USER)
    page.fill('#password', PASSWORD)
    page.click('button[type=submit]')
    page.wait_for_url(lambda url: '/login' not in url)


def open_detail(page, log, cid, marker):
    page.goto(f'{BASE}/library/{cid}')
    page.wait_for_function("() => typeof window.updateField === 'function' && document.readyState === 'complete'")
    page.wait_for_load_state('networkidle')
    page.evaluate(f"window.__smokeMarker = {marker!r}")
    log.clear()


def page_state(page):
    return page.evaluate("""() => ({
        marker: window.__smokeMarker,
        navigations: performance.getEntriesByType('navigation').length,
        token: window.CSRF_TOKEN,
        toast: (document.querySelector('.toast-notification') || {textContent: ''}).textContent.trim(),
        alert: (document.querySelector('#detail-alert-container .c-alert__message') || {textContent: ''}).textContent.trim(),
    })""")


def save_title(page, log, value):
    """Trigger the page's own title save and wait until its UI settled —
    the success toast or the failure alert — then one trailing window in
    which no further request may show up."""
    page.evaluate(f"() => updateField('title', {value!r})")
    page.wait_for_function(
        "() => document.querySelector('.toast-notification') || document.querySelector('#detail-alert-container .c-alert')",
        timeout=MUTATION_TIMEOUT_MS)
    state = page_state(page)
    page.wait_for_timeout(SETTLE_MS)
    return state, log.sequence()


S4_FETCH = """async () => {
    const fd = new FormData();
    fd.append('audio_file', new File(['CSRF-REFRESH Smoke: kein Audio'], 'csrf-refresh-probe.txt', {type: 'text/plain'}));
    fd.append('language', 'de');
    const r = await fetch('/api/transcriptions', {method: 'POST', body: fd});
    const text = await r.text();
    let body;
    try { body = JSON.parse(text); } catch (e) { body = text; }
    return {status: r.status, body};
}"""


def post_multipart(page, log):
    result = page.evaluate(S4_FETCH)
    page.wait_for_timeout(SETTLE_MS)
    return result, log.sequence()


def mark_stale(page):
    page.evaluate("window.CSRF_TOKEN = 'stale'")


def fmt(seq):
    return ' → '.join(f'{m} {p} {s}' for m, p, s in seq) if seq else '<keine Anfrage>'


def seq_put(cid, *statuses):
    return [('PUT', f'/api/conversions/{cid}', s) for s in statuses]


def not_reloaded(state, marker):
    return state['marker'] == marker and state['navigations'] == 1


# --- the four scenarios ----------------------------------------------------
def run(browser, cid, user_id):
    ctx = browser.new_context(viewport={'width': 1200, 'height': 900})
    page = ctx.new_page()
    log = RequestLog(page)
    login(page)
    print(f'cookies after login: {cookie_names(ctx)}')
    m = {}

    # S1 — stale token
    nonce = secrets.token_hex(3)
    open_detail(page, log, cid, 'S1')
    mark_stale(page)
    v1 = f'CSRF-REFRESH S1 {nonce}'
    s1, seq1 = save_title(page, log, v1)
    m['S1'] = dict(state=s1, seq=seq1, desc=log.describe(), value=v1, db=db_title(cid, user_id))
    print(f'S1 requests: {m["S1"]["desc"]}')
    print(f'S1 page: toast={s1["toast"]!r} alert={s1["alert"]!r} marker={s1["marker"]!r} '
          f'navigations={s1["navigations"]} token_len={len(s1["token"] or "")} '
          f'token_is_stale={s1["token"] == "stale"}; DB title={m["S1"]["db"]!r}')

    # S2 — freshly minted session (remember cookie only)
    open_detail(page, log, cid, 'S2')
    before = cookie_names(ctx)
    ctx.clear_cookies(name='session')
    after = cookie_names(ctx)
    print(f'S2 cookies before clear: {before}')
    print(f'S2 cookies after  clear: {after}')
    v2 = f'CSRF-REFRESH S2 {nonce}'
    s2, seq2 = save_title(page, log, v2)
    cookies_after_mutation = cookie_names(ctx)
    m['S2'] = dict(state=s2, seq=seq2, desc=log.describe(), value=v2, db=db_title(cid, user_id),
                   session_gone=not any(c[0] == 'session' for c in after),
                   remember_kept=any(c[0] == 'remember_token' for c in after),
                   session_back=any(c[0] == 'session' for c in cookies_after_mutation))
    print(f'S2 requests: {m["S2"]["desc"]}')
    print(f'S2 page: toast={s2["toast"]!r} alert={s2["alert"]!r} marker={s2["marker"]!r} '
          f'navigations={s2["navigations"]}; cookies after mutation: {cookies_after_mutation}; '
          f'DB title={m["S2"]["db"]!r}')
    # The remember cookie's own proof, independent of the wrapper: the next
    # page load must NOT be a login redirect, and the session cookie returns.
    page.goto(f'{BASE}/library/{cid}')
    page.wait_for_load_state('networkidle')
    m['S2']['reload_url_is_login'] = '/login' in page.url
    m['S2']['session_after_reload'] = has_cookie(ctx, 'session')
    print(f'S2 next load: url_is_login={m["S2"]["reload_url_is_login"]} '
          f'session_cookie_after_reload={m["S2"]["session_after_reload"]}')

    # S3 — second failure: the refresh hands out a bogus token
    route_hits = {'n': 0}

    def bogus(route):
        route_hits['n'] += 1
        route.fulfill(status=200, content_type='application/json',
                      body=json.dumps({'csrf_token': 'bogus'}))

    open_detail(page, log, cid, 'S3')
    page.route(f'**{CSRF_PATH}', bogus)
    mark_stale(page)
    v3 = f'CSRF-REFRESH S3 {nonce}'
    s3, seq3 = save_title(page, log, v3)
    page.unroute(f'**{CSRF_PATH}')
    m['S3'] = dict(state=s3, seq=seq3, desc=log.describe(), value=v3, db=db_title(cid, user_id),
                   route_hits=route_hits['n'])
    print(f'S3 requests: {m["S3"]["desc"]} (route hits: {route_hits["n"]})')
    print(f'S3 page: toast={s3["toast"]!r} alert={s3["alert"]!r} marker={s3["marker"]!r} '
          f'navigations={s3["navigations"]}; DB title={m["S3"]["db"]!r}')

    # S4 — multipart: positive control with the valid token, then stale
    open_detail(page, log, cid, 'S4')
    ctrl, seq_ctrl = post_multipart(page, log)
    print(f'S4 control requests: {log.describe()}')
    print(f'S4 control: status={ctrl["status"]} body={ctrl["body"]!r}')
    log.clear()
    mark_stale(page)
    res, seq4 = post_multipart(page, log)
    s4 = page_state(page)
    m['S4'] = dict(control=ctrl, control_seq=seq_ctrl, result=res, seq=seq4, desc=log.describe(),
                   state=s4, rows=transcription_rows(user_id))
    print(f'S4 requests: {m["S4"]["desc"]}')
    print(f'S4 result: status={res["status"]} body={res["body"]!r}; marker={s4["marker"]!r} '
          f'navigations={s4["navigations"]}; audio_transcription rows of user: {m["S4"]["rows"]}')

    ctx.close()
    return m


def error_of(result):
    body = result['body']
    return body.get('error') if isinstance(body, dict) else None


def judge_head(m, cid):
    print('--- Befunde, hergeleitet vs. gesehen ---')
    s1 = m['S1']
    verdict(s1['seq'] == seq_put(cid, 400) and s1['state']['alert'] == TITLE_FAILED
            and s1['state']['toast'] == '' and not_reloaded(s1['state'], 'S1'),
            f'H1 S1 alter Token: genau PUT 400, kein GET {CSRF_PATH}, Fehler-Alert der Seite, '
            f'kein Toast, kein Reload — gesehen {fmt(s1["seq"])}, alert={s1["state"]["alert"]!r}')
    s2 = m['S2']
    verdict(s2['session_gone'] and s2['remember_kept'],
            f'H2a S2 Mechanik: session-Cookie weg, remember_token bleibt — '
            f'gesehen session_gone={s2["session_gone"]} remember_kept={s2["remember_kept"]}')
    verdict(s2['seq'] == seq_put(cid, 400) and not_reloaded(s2['state'], 'S2'),
            f'H2b S2 neu geprägte Session: genau PUT 400, kein GET — gesehen {fmt(s2["seq"])}')
    verdict((not s2['reload_url_is_login']) and s2['session_after_reload'],
            f'H2c remember_token prägt beim nächsten Laden eine neue Session (kein Login-Redirect, '
            f'session-Cookie wieder da) — gesehen url_is_login={s2["reload_url_is_login"]} '
            f'session_back={s2["session_after_reload"]}')
    s4 = m['S4']
    verdict(s4['control']['status'] == 400 and error_of(s4['control']) == FORMAT_REJECTED
            and s4['control_seq'] == [('POST', TRANSCRIPTIONS_PATH, 400)],
            f'H3 S4 Positivkontrolle (gültiger Token): POST 400 mit dem Format-Text — die Datei kam an; '
            f'gesehen status={s4["control"]["status"]} error={error_of(s4["control"])!r}')
    verdict(s4['seq'] == [('POST', TRANSCRIPTIONS_PATH, 400)] and error_of(s4['result']) == 'csrf_expired'
            and not_reloaded(s4['state'], 'S4') and s4['rows'] == 0,
            f'H4 S4 alter Token: genau POST 400 csrf_expired, kein GET, 0 Zeilen — gesehen {fmt(s4["seq"])}, '
            f'error={error_of(s4["result"])!r}, rows={s4["rows"]}')
    s3 = m['S3']
    note(f'S3 (nur Gate): {fmt(s3["seq"])}, route hits {s3["route_hits"]}, alert={s3["state"]["alert"]!r}')
    note(f'DB-Titel nach S1/S2/S3: {s1["db"]!r} / {s2["db"]!r} / {s3["db"]!r}')


def judge_fixed(m, cid):
    print('--- gate ---')
    s1 = m['S1']
    check(s1['seq'] == seq_put(cid, 400) + [('GET', CSRF_PATH, 200)] + seq_put(cid, 200),
          f'S1: PUT 400 → GET {CSRF_PATH} 200 → PUT 200 ({fmt(s1["seq"])})')
    check(s1['state']['toast'] == TITLE_SAVED and s1['state']['alert'] == '',
          f'S1: success toast, no alert (toast={s1["state"]["toast"]!r} alert={s1["state"]["alert"]!r})')
    check(bool(s1['state']['token']) and s1['state']['token'] != 'stale',
          f'S1: window.CSRF_TOKEN refreshed (len {len(s1["state"]["token"] or "")}, not "stale")')
    check(s1['db'] == s1['value'], f'S1: title in the DB ({s1["db"]!r})')
    check(not_reloaded(s1['state'], 'S1'),
          f'S1: not reloaded (marker={s1["state"]["marker"]!r}, navigations={s1["state"]["navigations"]})')
    s2 = m['S2']
    check(s2['session_gone'] and s2['remember_kept'],
          f'S2: session cookie cleared, remember_token kept (gone={s2["session_gone"]} kept={s2["remember_kept"]})')
    check(s2['seq'] == seq_put(cid, 400) + [('GET', CSRF_PATH, 200)] + seq_put(cid, 200),
          f'S2: PUT 400 → GET 200 → PUT 200 ({fmt(s2["seq"])})')
    check(s2['session_back'], 'S2: a session cookie exists in the context again after the chain')
    check(s2['state']['toast'] == TITLE_SAVED and s2['db'] == s2['value'],
          f'S2: success toast and title in the DB ({s2["db"]!r})')
    check(not_reloaded(s2['state'], 'S2'),
          f'S2: not reloaded (marker={s2["state"]["marker"]!r}, navigations={s2["state"]["navigations"]})')
    s3 = m['S3']
    check(s3['seq'] == seq_put(cid, 400) + [('GET', CSRF_PATH, 200)] + seq_put(cid, 400),
          f'S3: PUT 400 → GET 200 → PUT 400 and nothing further ({fmt(s3["seq"])})')
    check(s3['route_hits'] == 1, f'S3: the refresh route was hit exactly once ({s3["route_hits"]})')
    check(s3['state']['alert'] == TITLE_FAILED and s3['state']['toast'] == '',
          f'S3: the existing failure alert, no toast (alert={s3["state"]["alert"]!r})')
    check(s3['db'] == s2['value'], f'S3: title unchanged in the DB ({s3["db"]!r})')
    check(not_reloaded(s3['state'], 'S3'),
          f'S3: not reloaded (marker={s3["state"]["marker"]!r}, navigations={s3["state"]["navigations"]})')
    s4 = m['S4']
    check(s4['control']['status'] == 400 and error_of(s4['control']) == FORMAT_REJECTED,
          f'S4 control: valid token → 400 with the format text ({error_of(s4["control"])!r})')
    check(s4['seq'] == [('POST', TRANSCRIPTIONS_PATH, 400), ('GET', CSRF_PATH, 200),
                        ('POST', TRANSCRIPTIONS_PATH, 400)],
          f'S4: POST 400 → GET 200 → POST 400 ({fmt(s4["seq"])})')
    check(s4['result']['status'] == 400 and error_of(s4['result']) == FORMAT_REJECTED,
          f'S4: the second POST carried the file — its own 400 ({error_of(s4["result"])!r})')
    check(s4['rows'] == 0, f'S4: zero audio_transcription rows for the user ({s4["rows"]})')
    check(not_reloaded(s4['state'], 'S4'),
          f'S4: not reloaded (marker={s4["state"]["marker"]!r}, navigations={s4["state"]["navigations"]})')


user_id = resolve_user()
cid = create_row(user_id)
print(f'mode=--expect {ARGS.expect} user_id={user_id} row={cid}')

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        measured = run(browser, cid, user_id)
        browser.close()
finally:
    removed, left = remove_rows(cid, user_id)
    print(f'cleanup: removed {removed} rows of user_id={user_id} (left: {left})')

print()
if ARGS.expect == 'head':
    judge_head(measured, cid)
    print()
    confirmed = sum(1 for ok, _ in verdicts if ok)
    print(f'{confirmed} von {len(verdicts)} Herleitungen bestätigt, {len(verdicts) - confirmed} widerlegt (kein Gate — exit 0)')
    sys.exit(0)
judge_fixed(measured, cid)
print()
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('ALL PASSED')
