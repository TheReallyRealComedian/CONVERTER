#!/usr/bin/env python3
"""Browser smoke for the "An Notion senden" dialog (NOTION-MEETING-LINK).

pytest renders no template and runs no JS — this is the REAL-browser check of
the Notion panel in ``templates/library_detail.html`` + its JS. It runs INSIDE
the web container (the playwright base image ships Chromium) against the
deployed app, as a throwaway user whose two documents are written through the
ORM, and MEASURES what the dialog does instead of looking at it.

Phase 0 = characterization of the dialog as it stood BEFORE the Notion code
moved out of ``library_detail.js`` (the criteria below were read off the
running prod state, measured twice):

1. Before the first open: panel hidden, no fields, no suggestions request.
2. First open on an ``audio_transcription`` row: target "Meeting" is active,
   the six meeting fields with their labels/defaults; values typed BEFORE the
   suggestions arrive survive the re-render that fills the datalists (the
   suggestions request is held back so "before" really is before).
3. Target switch Meeting → Notiz → Inbox → Meeting: which values carry over
   (same key, or the shared body slot summary/description), which fall back to
   their default, the live-region text, the tags field fed from the row's tags.
4. Close/reopen keeps the inputs and sends no second suggestions request; a
   click on the ALREADY active target re-renders the defaults.
5. The send button and its states — WITHOUT SENDING: every
   ``…/send-to-notion`` request is intercepted in the browser and answered by
   the script (held → "Sende …"/disabled; 200; 200 with url → ``window.open``;
   400 with error; 500 without; non-JSON; network abort). The request body the
   page WOULD have sent is read off the intercepted request. The script fails
   if a send request ever reached the server.
6. A ``markdown_input`` row opens on target "Notiz".

Nothing is written to Notion. ``/api/notion/suggestions`` is the one real
(read-only) call; only COUNTS of its lists are printed, never the names.

Test rows are written through the ORM under the throwaway user's own
``user_id``. The script REFUSES to run for user id 1 or the INGEST_USER and
removes the rows it created (strictly by ``user_id`` + the ids it wrote); the
user itself is created and removed by the operator (recipe below).

How to run (Mintbox, ~1 min):

    # 1. throwaway user — NEVER Oli's account
    docker exec markdown-converter-web flask --app app create-user zz_notion --password '<random>'
    # 2. stream the script in (SEC-NONROOT: written as the container user)
    docker exec -i markdown-converter-web sh -c 'cat > /tmp/smoke_notion.py' < scripts/smoke_notion_dialog.py
    # 3. run
    docker exec -e SMOKE_USER=zz_notion -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_notion.py
    # 4. clean up STRICTLY by user_id (the api_token table carries Oli's iOS
    #    tokens): the script already removed its rows; delete the User row via
    #    the ORM filtered by that user_id, then rm /tmp/smoke_notion.py.

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SMOKE_APP_ROOT (default: cwd = /app in the container). Exit 0 = every check
passed; every measured value is printed so a failure is diagnosable from the
output alone.

SMOKE_FAKE_SUGGESTIONS=1 is for a run WITHOUT Notion (a dev instance with no
NOTION_TOKEN): the script then answers ``/api/notion/suggestions`` itself with
canned lists. Never set it for the run that counts — against the deployed app
the real endpoint is the one being measured.
"""
import json
import os
import re
import sys

from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
FAKE_SUGGESTIONS = os.environ.get('SMOKE_FAKE_SUGGESTIONS') == '1'
CANNED_SUGGESTIONS = json.dumps({'people': ['Anna Beispiel', 'Bert Probe'], 'projects': ['Smoke-Projekt'],
                                 'meeting_types': ['Jour fixe'], 'note_types': ['Idee', 'Protokoll']})

AUDIO_TITLE = 'Smoke NOTION Audio'
AUDIO_CONTENT = '**Sprecher 1:** Smoke-Transkript, erste Zeile.\n\n**Sprecher 2:** Zweite Zeile mit <spitzen> & Klammern.'
DOC_TITLE = 'Smoke NOTION Dokument'
DOC_CONTENT = '# Smoke\n\nEin Absatz.'
TAG_NAME = 'zz-smoke-notion'

SUBMIT_IDLE = 'An Notion senden'
SUBMIT_BUSY = 'Sende …'
MSG_GENERIC = 'Senden an Notion fehlgeschlagen. Erneut versuchen oder Server-Konfiguration prüfen.'
MSG_NETWORK = 'Verbindung zu Notion fehlgeschlagen. Netzwerk und Notion-MCP-Server-Status prüfen.'

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


# --- test data through the ORM (same container, same DB) -------------------
sys.path.insert(0, os.environ.get('SMOKE_APP_ROOT', os.getcwd()))
from app import app  # noqa: E402  (bootstrap shim; the CLI imports it the same way)
from models import Conversion, Tag, User, db  # noqa: E402


def resolve_user():
    with app.app_context():
        user = User.query.filter_by(username=USER).first()
        if user is None:
            sys.exit(f'user {USER!r} does not exist — create it with flask create-user')
        if user.id == 1 or USER == os.environ.get('INGEST_USER'):
            sys.exit(f'refusing to run as user id {user.id} ({USER!r}): that is the '
                     'INGEST/first user = Oli\'s account')
        return user.id


def create_rows(user_id):
    with app.app_context():
        tag = Tag(user_id=user_id, name=TAG_NAME)
        audio = Conversion(user_id=user_id, conversion_type='audio_transcription',
                           title=AUDIO_TITLE, content=AUDIO_CONTENT)
        audio.tag_refs.append(tag)
        doc = Conversion(user_id=user_id, conversion_type='markdown_input',
                         title=DOC_TITLE, content=DOC_CONTENT)
        db.session.add_all([tag, audio, doc])
        db.session.commit()
        return {'audio': audio.id, 'doc': doc.id}, tag.id


def remove_rows(ids, tag_id, user_id):
    with app.app_context():
        rows = Conversion.query.filter(Conversion.user_id == user_id,
                                       Conversion.id.in_(ids)).all()
        for conv in rows:
            db.session.delete(conv)  # the conversion_tags rows go with it (ORM secondary)
        tag = Tag.query.filter_by(id=tag_id, user_id=user_id).first()
        if tag is not None:
            db.session.delete(tag)
        db.session.commit()
        left = Conversion.query.filter(Conversion.user_id == user_id,
                                       Conversion.id.in_(ids)).count()
        tags_left = Tag.query.filter_by(user_id=user_id).count()
        return len(rows), left, tags_left


# --- page helpers ----------------------------------------------------------
STATE = """() => {
  const q = s => document.querySelector(s);
  const btn = q('#notion-submit-btn');
  const fields = [...document.querySelectorAll('#notion-fields input, #notion-fields textarea')].map(el => ({
    key: el.id.replace('nf-', ''),
    tag: el.tagName.toLowerCase(),
    type: el.getAttribute('type'),
    value: el.value,
    list: el.getAttribute('list'),
    placeholder: el.getAttribute('placeholder') || '',
    label: el.parentElement.querySelector('label').textContent,
  }));
  const datalists = {};
  document.querySelectorAll('#notion-fields datalist').forEach(d => {
    datalists[d.id] = d.querySelectorAll('option').length;
  });
  const alertEl = q('#notion-alert-container .c-alert');
  const toastEl = q('.toast-notification');
  return {
    panelHidden: q('#notion-panel').classList.contains('hidden'),
    expanded: q('#notion-toggle-btn').getAttribute('aria-expanded'),
    icon: q('#notion-toggle-icon').textContent,
    primary: [...document.querySelectorAll('#notion-target-group button')]
      .filter(b => b.classList.contains('c-btn--primary')).map(b => b.dataset.target),
    fields, datalists,
    status: q('#notion-target-status').textContent,
    submit: { disabled: btn.disabled, text: btn.textContent },
    alert: alertEl ? { cls: alertEl.className, text: alertEl.querySelector('.c-alert__message').textContent } : null,
    toast: toastEl ? toastEl.textContent : null,
    opens: window.__opens.slice(),
  };
}"""

# window.open is recorded instead of executed: a success answer with a url
# would otherwise open a real Notion tab from inside the container.
INIT_SCRIPT = "window.__opens = []; window.open = (...args) => { window.__opens.push(args); return null; };"


def state(page):
    return page.evaluate(STATE)


def values(st):
    return {f['key']: f['value'] for f in st['fields']}


def keys(st):
    return [f['key'] for f in st['fields']]


def labels(st):
    return [f['label'] for f in st['fields']]


def show(tag, st):
    """Print a state without the datalist CONTENT (names from Notion) — the
    STATE query only ever carries option counts."""
    print(f'[{tag}] ' + json.dumps(st, ensure_ascii=False))


def open_detail(page, conv_id):
    page.goto(f'{BASE}/library/{conv_id}')
    page.wait_for_selector('#notion-toggle-btn')
    page.wait_for_load_state('networkidle')


def set_field(page, key, value):
    page.fill(f'#nf-{key}', value)


def release_suggestions(route):
    if FAKE_SUGGESTIONS:
        route.fulfill(status=200, content_type='application/json', body=CANNED_SUGGESTIONS)
    else:
        route.continue_()


user_id = resolve_user()
ids, tag_id = create_rows(user_id)
print(f'user_id={user_id} rows={ids} tag={tag_id}' + (' — FAKE SUGGESTIONS (not a run against Notion)' if FAKE_SUGGESTIONS else ''))

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={'width': 1200, 'height': 900})
        ctx.add_init_script(INIT_SCRIPT)
        page = ctx.new_page()
        page_errors = []
        page.on('pageerror', lambda e: page_errors.append(str(e)))
        suggestion_requests = []
        send_requests_seen = []       # every send-to-notion request the PAGE issued
        page.on('request', lambda r: suggestion_requests.append(r.url)
                if '/api/notion/suggestions' in r.url else None)

        # Every send request stops here; the script answers it itself. Nothing
        # is ever continued to the server.
        held_sends = []

        def on_send(route):
            send_requests_seen.append(route.request.post_data)
            held_sends.append(route)

        ctx.route('**/api/conversions/*/send-to-notion', on_send)

        page.goto(f'{BASE}/login')
        page.fill('input[name=username]', USER)
        page.fill('input[name=password]', PASSWORD)
        page.click('button[type=submit]')
        page.wait_for_url(lambda url: '/login' not in url)
        print('logged in as', USER)

        print('=== 1. before the first open ===')
        open_detail(page, ids['audio'])
        s = state(page)
        show('closed', s)
        check(s['panelHidden'] and s['expanded'] == 'false' and s['icon'] == '▾',
              'panel hidden, aria-expanded=false, icon ▾')
        check(s['fields'] == [] and s['primary'] == [], 'no fields rendered, no target marked')
        check(len(suggestion_requests) == 0, f'no suggestions request before the open ({len(suggestion_requests)})')
        check(s['submit'] == {'disabled': False, 'text': SUBMIT_IDLE}, f'submit button idle ({s["submit"]})')

        print('=== 2. first open (audio row): Meeting fields, typed values survive the suggestions re-render ===')
        held_suggestions = []
        page.route('**/api/notion/suggestions', lambda route: held_suggestions.append(route))
        page.click('#notion-toggle-btn')
        page.wait_for_selector('#nf-title')
        page.wait_for_timeout(600)
        s = state(page)
        show('open, suggestions held', s)
        check(not s['panelHidden'] and s['expanded'] == 'true' and s['icon'] == '▴',
              'panel visible, aria-expanded=true, icon ▴')
        check(s['primary'] == ['meetings'], f'target "Meeting" is the active one for an audio row ({s["primary"]})')
        check(keys(s) == ['title', 'datum', 'project', 'people', 'type', 'summary'],
              f'meeting fields in order ({keys(s)})')
        check(labels(s) == ['Titel *', 'Datum', 'Projekt', 'Personen', 'Typ', 'Zusammenfassung'],
              f'meeting labels ({labels(s)})')
        v = values(s)
        check(v['title'] == AUDIO_TITLE, f'title defaults to the row title ({v["title"]!r})')
        datum = next(f for f in s['fields'] if f['key'] == 'datum')
        check(datum['type'] == 'datetime-local' and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', datum['value']) is not None,
              f'datum is a datetime-local with a now-default ({datum["type"]}, {datum["value"]!r})')
        check([v['project'], v['people'], v['type'], v['summary']] == ['', '', '', ''],
              'project, people, type, summary start empty')
        summary = next(f for f in s['fields'] if f['key'] == 'summary')
        people = next(f for f in s['fields'] if f['key'] == 'people')
        check(summary['tag'] == 'textarea' and people['placeholder'] == 'kommagetrennt',
              'summary is a textarea, people carries the placeholder "kommagetrennt"')
        check(s['datalists'] == {} and len(held_suggestions) == 1,
              f'suggestions still held: no datalist yet ({s["datalists"]}, held={len(held_suggestions)})')
        check(s['status'] == '', f'no live-region text on the initial render ({s["status"]!r})')
        set_field(page, 'title', 'Smoke Titel geändert')
        set_field(page, 'project', 'Smoke-Projekt')
        set_field(page, 'people', 'Anna Beispiel, Bert Probe')
        set_field(page, 'type', 'Smoke-Typ')
        set_field(page, 'summary', 'Smoke-Zusammenfassung')
        for route in held_suggestions:
            release_suggestions(route)
        page.unroute('**/api/notion/suggestions')
        if FAKE_SUGGESTIONS:
            ctx.route('**/api/notion/suggestions', release_suggestions)
        page.wait_for_function("() => document.querySelectorAll('#notion-fields datalist').length > 0", timeout=30000)
        page.wait_for_timeout(300)
        s = state(page)
        show('open, suggestions arrived', s)
        v = values(s)
        check(sorted(s['datalists']) == ['dl-people', 'dl-project', 'dl-type'] and all(n > 0 for n in s['datalists'].values()),
              f'datalists for project, people, type are filled ({s["datalists"]})')
        lists = {f['key']: f['list'] for f in s['fields']}
        check(lists == {'title': None, 'datum': None, 'project': 'dl-project', 'people': 'dl-people',
                        'type': 'dl-type', 'summary': None},
              f'the three inputs point at their datalist ({lists})')
        check([v['title'], v['project'], v['people'], v['type'], v['summary']]
              == ['Smoke Titel geändert', 'Smoke-Projekt', 'Anna Beispiel, Bert Probe', 'Smoke-Typ', 'Smoke-Zusammenfassung'],
              'values typed before the suggestions arrived survived the re-render')
        check(len(suggestion_requests) == 1, f'exactly one suggestions request ({len(suggestion_requests)})')
        meeting_type_options = s['datalists']['dl-type']

        print('=== 3. target switch: Meeting → Notiz → Inbox → Meeting ===')
        page.click('#notion-target-group button[data-target=notes]')
        page.wait_for_timeout(400)
        s = state(page)
        show('notes', s)
        v = values(s)
        check(s['primary'] == ['notes'], f'target "Notiz" active ({s["primary"]})')
        check(keys(s) == ['title', 'project', 'type', 'tags', 'people', 'summary'], f'note fields in order ({keys(s)})')
        check(labels(s) == ['Titel *', 'Projekt', 'Typ', 'Tags', 'Personen', 'Zusammenfassung'], f'note labels ({labels(s)})')
        check([v['title'], v['project'], v['type'], v['people'], v['summary']]
              == ['Smoke Titel geändert', 'Smoke-Projekt', 'Smoke-Typ', 'Anna Beispiel, Bert Probe', 'Smoke-Zusammenfassung'],
              'title, project, type, people, summary carried over (same keys)')
        check(v['tags'] == TAG_NAME, f'tags field is fed from the row\'s tags ({v["tags"]!r})')
        check(s['status'] == 'Ziel gewechselt zu Notiz — passende Felder übernommen.', f'live region ({s["status"]!r})')
        check(sorted(s['datalists']) == ['dl-people', 'dl-project', 'dl-type'], f'note datalists ({s["datalists"]})')
        note_type_options = s['datalists'].get('dl-type')
        print(f'[type options] meetings={meeting_type_options} notes={note_type_options}')

        page.click('#notion-target-group button[data-target=inbox]')
        page.wait_for_timeout(400)
        s = state(page)
        show('inbox', s)
        v = values(s)
        check(s['primary'] == ['inbox'], f'target "Inbox" active ({s["primary"]})')
        check(keys(s) == ['name', 'description', 'source', 'project', 'people'], f'inbox fields in order ({keys(s)})')
        check(labels(s) == ['Name *', 'Beschreibung', 'Quelle', 'Projekt', 'Personen'], f'inbox labels ({labels(s)})')
        check(v['name'] == AUDIO_TITLE,
              f'name falls back to the ROW title — the edited title is not carried (different key) ({v["name"]!r})')
        check(v['description'] == 'Smoke-Zusammenfassung', 'description takes the summary (shared body slot)')
        check(v['source'] == 'CONVERTER', f'source defaults to CONVERTER ({v["source"]!r})')
        check([v['project'], v['people']] == ['Smoke-Projekt', 'Anna Beispiel, Bert Probe'], 'project and people carried over')
        check(s['status'] == 'Ziel gewechselt zu Inbox — passende Felder übernommen.', f'live region ({s["status"]!r})')
        check(sorted(s['datalists']) == ['dl-people', 'dl-project'], f'inbox datalists ({s["datalists"]})')

        page.click('#notion-target-group button[data-target=meetings]')
        page.wait_for_timeout(400)
        s = state(page)
        show('meetings again', s)
        v = values(s)
        check(s['primary'] == ['meetings'] and keys(s) == ['title', 'datum', 'project', 'people', 'type', 'summary'],
              'back on "Meeting" with its six fields')
        check(v['title'] == AUDIO_TITLE and v['type'] == '', 'title and type are back at their defaults (not in the inbox snapshot)')
        check(v['summary'] == 'Smoke-Zusammenfassung', 'summary takes the description back (shared body slot)')
        check([v['project'], v['people']] == ['Smoke-Projekt', 'Anna Beispiel, Bert Probe'], 'project and people carried over')
        check(s['status'] == 'Ziel gewechselt zu Meeting — passende Felder übernommen.', f'live region ({s["status"]!r})')

        print('=== 4. close/reopen keeps inputs; a click on the active target resets them ===')
        page.click('#notion-toggle-btn')
        page.wait_for_timeout(200)
        s = state(page)
        check(s['panelHidden'] and s['expanded'] == 'false' and s['icon'] == '▾', 'panel closed again')
        page.click('#notion-toggle-btn')
        page.wait_for_timeout(400)
        s = state(page)
        v = values(s)
        check(not s['panelHidden'] and [v['project'], v['people'], v['summary']]
              == ['Smoke-Projekt', 'Anna Beispiel, Bert Probe', 'Smoke-Zusammenfassung'],
              'reopened: the inputs are still there')
        check(len(suggestion_requests) == 1, f'no second suggestions request on reopen ({len(suggestion_requests)})')
        page.click('#notion-target-group button[data-target=notes]')
        page.wait_for_timeout(300)
        set_field(page, 'project', 'wird verworfen')
        page.click('#notion-target-group button[data-target=notes]')
        page.wait_for_timeout(300)
        s = state(page)
        v = values(s)
        show('notes clicked twice', s)
        check(v['project'] == '' and v['title'] == AUDIO_TITLE and v['summary'] == '' and v['tags'] == TAG_NAME,
              'a click on the ALREADY active target re-renders the defaults (inputs gone)')
        page.click('#notion-target-group button[data-target=meetings]')
        page.wait_for_timeout(300)

        print('=== 5. send button states — intercepted in the browser, nothing reaches the server ===')

        def click_send_and_hold():
            n = len(held_sends)
            page.click('#notion-submit-btn')
            page.wait_for_function("() => document.getElementById('notion-submit-btn').disabled")
            for _ in range(50):
                if len(held_sends) > n:
                    break
                page.wait_for_timeout(100)
            return held_sends[-1], state(page)

        def settle():
            page.wait_for_function("() => !document.getElementById('notion-submit-btn').disabled")
            page.wait_for_timeout(200)
            return state(page)

        set_field(page, 'project', 'Smoke-Projekt')
        set_field(page, 'people', 'Anna Beispiel,  Bert Probe ,')
        set_field(page, 'type', '')
        set_field(page, 'summary', '  Smoke-Zusammenfassung  ')
        datum_value = page.input_value('#nf-datum')

        # 5a: held → busy; 200 without url → toast, no window.open
        route, busy = click_send_and_hold()
        show('5a busy', busy)
        check(busy['submit'] == {'disabled': True, 'text': SUBMIT_BUSY}, f'in flight: disabled, "{SUBMIT_BUSY}" ({busy["submit"]})')
        body = json.loads(route.request.post_data)
        print('[5a body keys] ' + json.dumps({'target': body['target'], 'fields': sorted(body['fields'])})
              + f' transcript_len={len(body["fields"].get("transcript", ""))}')
        check(body['target'] == 'meetings' and sorted(body) == ['fields', 'target'], 'body is {target: "meetings", fields}')
        check(sorted(body['fields']) == ['datum', 'people', 'project', 'summary', 'title', 'transcript'],
              f'fields: filled inputs + transcript, the empty "type" is omitted ({sorted(body["fields"])})')
        check(body['fields']['transcript'] == AUDIO_CONTENT, 'transcript is the raw row content (content-source), byte-equal')
        check(body['fields']['people'] == ['Anna Beispiel', 'Bert Probe'], f'people is split into a trimmed list ({body["fields"]["people"]})')
        check(body['fields']['summary'] == 'Smoke-Zusammenfassung' and body['fields']['datum'] == datum_value,
              'values are trimmed; datum goes out as the zone-less form value')
        route.fulfill(status=200, content_type='application/json', body='{"success": true}')
        s = settle()
        show('5a done', s)
        check(s['submit'] == {'disabled': False, 'text': SUBMIT_IDLE}, 'after the answer: enabled, label restored')
        check(s['toast'] == 'An Notion gesendet' and s['alert'] is None and s['opens'] == [],
              f'200 without url: toast "An Notion gesendet", no alert, no window.open ({s["toast"]!r}, {s["opens"]})')

        # 5b: 200 with url → window.open(url, _blank, noopener,noreferrer)
        route, _ = click_send_and_hold()
        route.fulfill(status=200, content_type='application/json',
                      body='{"success": true, "url": "https://app.notion.com/p/smoke"}')
        s = settle()
        check(s['opens'] == [['https://app.notion.com/p/smoke', '_blank', 'noopener,noreferrer']],
              f'200 with url: window.open(url, "_blank", "noopener,noreferrer") ({s["opens"]})')

        # 5c: 400 with error → danger alert with the server's text
        route, _ = click_send_and_hold()
        route.fulfill(status=400, content_type='application/json', body='{"error": "Smoke-Fehler"}')
        s = settle()
        show('5c 400', s)
        check(s['alert'] is not None and s['alert']['text'] == 'Senden fehlgeschlagen: Smoke-Fehler.'
              and 'c-alert--danger' in s['alert']['cls'],
              f'400 + error: danger alert "Senden fehlgeschlagen: <text>." ({s["alert"]})')
        check(s['submit'] == {'disabled': False, 'text': SUBMIT_IDLE}, 'button restored after the error')

        # 5d: the next send clears the alert at its start; 500 without error text → generic sentence
        route, busy = click_send_and_hold()
        check(busy['alert'] is None, 'a new send clears the previous alert before the answer arrives')
        route.fulfill(status=500, content_type='application/json', body='{}')
        s = settle()
        check(s['alert'] is not None and s['alert']['text'] == MSG_GENERIC, f'500 without error text: generic sentence ({s["alert"]})')

        # 5e: `detail` is the fallback key
        route, _ = click_send_and_hold()
        route.fulfill(status=422, content_type='application/json', body='{"detail": "Smoke-Detail"}')
        s = settle()
        check(s['alert'] is not None and s['alert']['text'] == 'Senden fehlgeschlagen: Smoke-Detail.',
              f'422 + detail: the detail text is shown ({s["alert"]})')

        # 5f: non-JSON answer → network sentence
        route, _ = click_send_and_hold()
        route.fulfill(status=502, content_type='text/html', body='<html>Bad Gateway</html>')
        s = settle()
        check(s['alert'] is not None and s['alert']['text'] == MSG_NETWORK, f'non-JSON answer: network sentence ({s["alert"]})')

        # 5g: network failure → network sentence
        route, _ = click_send_and_hold()
        route.abort()
        s = settle()
        check(s['alert'] is not None and s['alert']['text'] == MSG_NETWORK, f'aborted request: network sentence ({s["alert"]})')

        # 5h: Notiz sends `content` (not transcript) and tags as a list
        page.click('#notion-target-group button[data-target=notes]')
        page.wait_for_timeout(300)
        route, _ = click_send_and_hold()
        body = json.loads(route.request.post_data)
        print('[5h body keys] ' + json.dumps({'target': body['target'], 'fields': sorted(body['fields'])}))
        check(body['target'] == 'notes' and body['fields'].get('content') == AUDIO_CONTENT and 'transcript' not in body['fields'],
              'Notiz: the row content travels as "content", not "transcript"')
        check(body['fields'].get('tags') == [TAG_NAME], f'Notiz: tags is a list ({body["fields"].get("tags")})')
        route.fulfill(status=200, content_type='application/json', body='{"success": true}')
        settle()

        # 5i: Inbox sends name/source + content
        page.click('#notion-target-group button[data-target=inbox]')
        page.wait_for_timeout(300)
        route, _ = click_send_and_hold()
        body = json.loads(route.request.post_data)
        print('[5i body keys] ' + json.dumps({'target': body['target'], 'fields': sorted(body['fields'])}))
        check(body['target'] == 'inbox' and body['fields'].get('name') == AUDIO_TITLE
              and body['fields'].get('source') == 'CONVERTER' and body['fields'].get('content') == AUDIO_CONTENT,
              'Inbox: name, source and content travel')
        route.fulfill(status=200, content_type='application/json', body='{"success": true}')
        settle()

        print('=== 6. a markdown_input row opens on "Notiz" ===')
        open_detail(page, ids['doc'])
        page.click('#notion-toggle-btn')
        page.wait_for_selector('#nf-title')
        page.wait_for_function("() => document.querySelectorAll('#notion-fields datalist').length > 0", timeout=30000)
        s = state(page)
        show('doc row', s)
        v = values(s)
        check(s['primary'] == ['notes'] and keys(s) == ['title', 'project', 'type', 'tags', 'people', 'summary'],
              f'default target for a non-audio row is "Notiz" ({s["primary"]})')
        check(v['title'] == DOC_TITLE and v['tags'] == '', 'title from the row, no tags')

        print('=== guards ===')
        # The handler above never calls continue_(): a held route is either
        # fulfilled or aborted by the script, so the count IS the proof.
        check(len(send_requests_seen) == len(held_sends) == 9,
              f'every send request was answered by the script, none reached the server '
              f'(seen={len(send_requests_seen)}, held={len(held_sends)})')
        check(page_errors == [], f'no uncaught JS error on the page ({page_errors})')

        browser.close()
finally:
    removed, left, tags_left = remove_rows(list(ids.values()), tag_id, user_id)
    print(f'[cleanup] removed {removed} rows of user_id={user_id}, left={left}, tags left={tags_left}')
    check(left == 0 and tags_left == 0, 'test rows and the test tag are gone')

print()
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('ALL PASSED')
