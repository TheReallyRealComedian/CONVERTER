#!/usr/bin/env python3
"""Browser smoke for the "An Notion senden" dialog (NOTION-MEETING-LINK).

pytest renders no template and runs no JS — this is the REAL-browser check of
the Notion panel in ``templates/library_detail.html`` + its JS. It runs INSIDE
the web container (the playwright base image ships Chromium) against the
deployed app, as a throwaway user whose three documents are written through the
ORM, and MEASURES what the dialog does instead of looking at it.

Part A = the form as it stood BEFORE the Notion code moved out of
``library_detail.js`` (criteria read off the running prod state in Phase 0,
measured twice). Since Phase 1 the audio row opens on "Bestehendes Meeting";
part A first switches to "Neues Meeting anlegen" and then runs unchanged:

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
   page WOULD have sent is read off the intercepted request.
6. A ``markdown_input`` row opens on target "Notiz".

Part B = "Bestehendes Meeting", the page's mechanics against SCRIPTED answers
(the candidate list and every send are answered by the script — deterministic,
no Notion involved; the server side of both is pinned by
``tests/test_notion_meeting_link.py``):

7. The audio row opens on "Bestehendes Meeting": way switch, reference line,
   hint, the list as delivered (the page does not re-sort), markers, the
   preselected entry, the send button disabled without a selection.
8. A meeting title is drawn as a text node (markup in a title stays text).
9. Day navigation asks for exactly the days the server named.
10. Send: the request is exactly ``{target, page_id, day, replace_transcript}``;
    409 → native confirm with the server's sentence; "Nein" sends nothing
    more; "Ja" repeats with ``replace_transcript: true``; success → toast, the
    link line, the list reloaded for the same day.
11. 413 / 5xx / network abort → the alert.
12. Way and target switches; a ``markdown_input`` row keeps the form as its
    default way.
13. A row with a remembered link shows "Verknüpft mit …" without opening the
    panel; a non-https url never becomes an anchor.

In parts A and B nothing is written to Notion: a send request that is not
answered by the script fails the run. ``/api/notion/suggestions`` is the one
real (read-only) call; only COUNTS of its lists are printed, never the names.

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

SMOKE_OUT=<prefix> additionally writes ``<prefix>_light.png`` and
``<prefix>_dark.png`` of the panel with the scripted list (and checks that
nothing reaches past the card).

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
OUT = os.environ.get('SMOKE_OUT')      # e.g. /tmp/smoke_notion → _light.png / _dark.png of the panel
CANNED_SUGGESTIONS = json.dumps({'people': ['Anna Beispiel', 'Bert Probe'], 'projects': ['Smoke-Projekt'],
                                 'meeting_types': ['Jour fixe'], 'note_types': ['Idee', 'Protokoll']})

AUDIO_TITLE = 'Smoke NOTION Audio'
AUDIO_CONTENT = '**Sprecher 1:** Smoke-Transkript, erste Zeile.\n\n**Sprecher 2:** Zweite Zeile mit <spitzen> & Klammern.'
DOC_TITLE = 'Smoke NOTION Dokument'
DOC_CONTENT = '# Smoke\n\nEin Absatz.'
TAG_NAME = 'zz-smoke-notion'

# Part B: scripted meetings. The ids only need the SHAPE of a Notion page id.
P_NEAR = '11111111-1111-4111-8111-111111111111'
P_TAKEN = '22222222-2222-4222-8222-222222222222'
P_MARKUP = '33333333-3333-4333-8333-333333333333'
P_ALLDAY = '44444444-4444-4444-8444-444444444444'
P_LATER = '55555555-5555-4555-8555-555555555555'
MARKUP_TITLE = '<img src=x onerror="window.__xss=1"> & <b>fett</b>'
LINK_TITLE = 'Smoke-Meeting <b>verknüpft</b>'
LINK_URL = 'https://app.notion.com/p/' + P_NEAR.replace('-', '')

SUBMIT_IDLE = 'An Notion senden'
SUBMIT_BUSY = 'Sende …'
MSG_GENERIC = 'Senden an Notion fehlgeschlagen. Erneut versuchen oder Server-Konfiguration prüfen.'
MSG_NETWORK = 'Verbindung zu Notion fehlgeschlagen. Netzwerk und Notion-MCP-Server-Status prüfen.'

failures = []


def _entry(page_id, title, day, weekday, date_text, time_text, length=None, **kw):
    return {'page_id': page_id, 'title': title, 'type': kw.get('type', 'Meeting'), 'url': None,
            'day': day, 'weekday': weekday, 'date_text': date_text, 'time_text': time_text,
            'all_day': time_text == 'ganztägig', 'length_minutes': length,
            'length_text': f'{length} min' if length else None,
            'has_transcript': kw.get('has_transcript', False), 'linked': kw.get('linked', False),
            'linked_here': kw.get('linked_here', False), 'notnotion': False}


def scripted_candidates(day, sent):
    """The answer of ``…/notion-meetings`` in part B. ``prev_day``/``next_day``
    are deliberately NOT the calendar neighbours: a page that computed the
    neighbour itself would ask for the wrong day and be caught."""
    reference = {'source': 'recorded_at', 'time_known': True, 'day': '2026-10-02',
                 'text': 'Fr, 02.10.2026, 14:40', 'hint': None,
                 'duration_seconds': 1500, 'duration_text': '25 min'}
    if day in (None, '2026-10-02'):
        near = _entry(P_NEAR, 'Jour fixe Smoke', '2026-10-02', 'Fr', '02.10.2026', '14:30–15:00', 30,
                      type='Jour Fixe', has_transcript=sent, linked=sent, linked_here=sent)
        return {'reference': reference, 'day': '2026-10-02', 'day_text': 'Fr, 02.10.2026',
                'prev_day': '2026-09-28', 'next_day': '2026-10-07', 'order': 'distance',
                'meetings': [
                    near,
                    _entry(P_TAKEN, 'Schon vergeben', '2026-10-02', 'Fr', '02.10.2026', '16:00–17:00', 60,
                           has_transcript=True, linked=True),
                    _entry(P_MARKUP, MARKUP_TITLE, '2026-10-03', 'Sa', '03.10.2026', '08:00', None, type=''),
                    _entry(P_ALLDAY, 'Messe', '2026-10-02', 'Fr', '02.10.2026', 'ganztägig', None),
                ],
                'preselected_page_id': P_NEAR, 'preselect_reason': 'linked' if sent else 'time',
                'truncated': False, 'link': None}
    if day == '2026-09-28':
        return {'reference': dict(reference, time_known=False, text='Fr, 02.10.2026',
                                  hint='Uhrzeit der Aufnahme unbekannt.'),
                'day': day, 'day_text': 'Mo, 28.09.2026', 'prev_day': '2026-09-21', 'next_day': '2026-10-02',
                'order': 'chronological', 'meetings': [], 'preselected_page_id': None,
                'preselect_reason': None, 'truncated': False, 'link': None}
    return {'reference': reference, 'day': day, 'day_text': 'Mi, 07.10.2026',
            'prev_day': '2026-10-02', 'next_day': '2026-10-12', 'order': 'chronological',
            'meetings': [_entry(P_LATER, 'Später', day, 'Mi', '07.10.2026', '10:00–11:00', 60)],
            'preselected_page_id': None, 'preselect_reason': None, 'truncated': False, 'link': None}


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
        linked = Conversion(user_id=user_id, conversion_type='audio_transcription',
                            title='Smoke NOTION verknüpft', content=AUDIO_CONTENT,
                            metadata_json=json.dumps({'notion_link': {
                                'page_id': P_NEAR, 'url': LINK_URL, 'meeting_title': LINK_TITLE,
                                'meeting_start': '2026-10-02T14:30:00.000+02:00',
                                'linked_at': '2026-10-02T13:00:00+00:00'}}))
        db.session.add_all([tag, audio, doc, linked])
        db.session.commit()
        return {'audio': audio.id, 'doc': doc.id, 'linked': linked.id}, tag.id


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
  const vis = el => !!el && el.offsetParent !== null;
  const linkBox = q('#notion-link-status');
  const linkA = linkBox.querySelector('a');
  const meetings = [...document.querySelectorAll('#notion-meeting-list .notion-meeting')].map(l => ({
    page: l.querySelector('input').value,
    checked: l.querySelector('input').checked,
    otherDay: l.classList.contains('notion-meeting--other-day'),
    when: l.querySelector('.notion-meeting__when').textContent,
    title: l.querySelector('.notion-meeting__title').textContent,
    titleChildren: l.querySelector('.notion-meeting__title').children.length,
    meta: [...l.querySelectorAll('.notion-meeting__meta > span')].map(x => x.textContent),
  }));
  return {
    way: { visible: vis(q('#notion-way-group')),
           pressed: [...document.querySelectorAll('#notion-way-group button')]
             .filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.dataset.way) },
    existingVisible: vis(q('#notion-existing')),
    formVisible: vis(q('#notion-new')),
    reference: q('#notion-reference').textContent,
    hint: q('#notion-reference-hint').hidden ? null : q('#notion-reference-hint').textContent,
    dayInput: q('#notion-day-input').value,
    meetings,
    notes: [...document.querySelectorAll('#notion-meeting-list .notion-meeting-list__note')].map(x => x.textContent),
    link: linkBox.hidden ? null : { text: linkBox.textContent, href: linkA ? linkA.getAttribute('href') : null,
                                    target: linkA ? linkA.target : null, rel: linkA ? linkA.rel : null,
                                    elements: linkBox.querySelectorAll('*').length },
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

        # Parts A and B answer the candidate list themselves: 'empty' while the
        # form is characterised, 'scenario' for the mechanics of the list.
        candidates = {'mode': 'empty', 'sent': False, 'requests': []}

        def on_candidates(route):
            url = route.request.url
            day = url.split('?day=')[1] if '?day=' in url else None
            candidates['requests'].append(day)
            body = scripted_candidates(day, candidates['sent'])
            if candidates['mode'] == 'empty':
                body = dict(scripted_candidates('2026-09-28', False), day='2026-10-02')
            route.fulfill(status=200, content_type='application/json', body=json.dumps(body))

        ctx.route('**/api/conversions/*/notion-meetings*', on_candidates)

        # Native confirm(): recorded, answered from a plan the step sets.
        dialogs = []
        dialog_plan = []

        def on_dialog(dialog):
            dialogs.append(dialog.message)
            if dialog_plan and dialog_plan.pop(0):
                dialog.accept()
            else:
                dialog.dismiss()

        page.on('dialog', on_dialog)

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
        check(s['link'] is None and len(candidates['requests']) == 0,
              'no link line for an unlinked row, no candidates request before the open')

        print('=== 2. first open (audio row): Meeting fields, typed values survive the suggestions re-render ===')
        held_suggestions = []
        page.route('**/api/notion/suggestions', lambda route: held_suggestions.append(route))
        page.click('#notion-toggle-btn')
        page.wait_for_selector('#notion-way-group')
        page.wait_for_timeout(400)
        s = state(page)
        check(s['way'] == {'visible': True, 'pressed': ['existing']} and s['existingVisible'] and not s['formVisible'],
              f'an audio row opens on "Bestehendes Meeting" ({s["way"]})')
        # Part A is about the FORM: switch to "Neues Meeting anlegen" and go on as before.
        page.click('#notion-way-group button[data-way=new]')
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

        # ---------------------------------------------------------------- part B
        candidates['mode'] = 'scenario'
        sends_part_a = len(held_sends)

        print('=== 7. audio row: "Bestehendes Meeting" — list as delivered, markers, preselection ===')
        open_detail(page, ids['audio'])
        n_cand = len(candidates['requests'])
        page.click('#notion-toggle-btn')
        page.wait_for_selector('#notion-meeting-list .notion-meeting')
        page.wait_for_timeout(300)
        s = state(page)
        show('existing way', s)
        check(candidates['requests'][n_cand:] == [None], f'one candidates request, without a day ({candidates["requests"][n_cand:]})')
        check(s['way'] == {'visible': True, 'pressed': ['existing']} and s['existingVisible'] and not s['formVisible'],
              'way switch visible, "Bestehendes Meeting" pressed, the form hidden')
        check(s['reference'] == 'Aufnahme: Fr, 02.10.2026, 14:40 · Dauer 25 min' and s['hint'] is None,
              f'reference line with the recording length, no hint ({s["reference"]!r})')
        check(s['dayInput'] == '2026-10-02', f'date field shows the delivered day ({s["dayInput"]})')
        check([m['page'] for m in s['meetings']] == [P_NEAR, P_TAKEN, P_MARKUP, P_ALLDAY],
              'the list keeps the order the server delivered (no re-sort in the page)')
        near, taken, markup, allday = s['meetings']
        check(near['when'] == 'Fr, 02.10.2026 · 14:30–15:00 · 30 min' and near['title'] == 'Jour fixe Smoke'
              and near['meta'] == ['Jour Fixe'],
              f'entry: weekday, date, time, length, title, type ({near})')
        check(taken['meta'] == ['Meeting', 'hat schon Transkript', 'schon mit CONVERTER verknüpft'],
              f'markers "hat schon Transkript" and "schon mit CONVERTER verknüpft" ({taken["meta"]})')
        check(allday['when'] == 'Fr, 02.10.2026 · ganztägig', f'all-day entry carries no length ({allday["when"]!r})')
        check(markup['otherDay'] and not near['otherDay'] and markup['when'] == 'Sa, 03.10.2026 · 08:00',
              'a neighbour day\'s entry is marked as such; an entry without end has no length')
        check([m['checked'] for m in s['meetings']] == [True, False, False, False],
              'the preselected entry is the checked one')
        check(s['submit'] == {'disabled': False, 'text': SUBMIT_IDLE}, 'with a selection the send button is enabled')

        print('=== 8. a meeting title is a text node ===')
        xss = page.evaluate("() => ({flag: window.__xss === undefined, imgs: document.querySelectorAll('#notion-meeting-list img, #notion-meeting-list b').length})")
        check(markup['title'] == MARKUP_TITLE and markup['titleChildren'] == 0 and xss == {'flag': True, 'imgs': 0},
              f'markup in a title stays text: no element, no handler ran ({xss})')

        print('=== 9. day navigation asks for the days the server named ===')
        n_cand = len(candidates['requests'])
        page.click('#notion-day-prev')
        page.wait_for_function("() => document.getElementById('notion-day-input').value === '2026-09-28'")
        s = state(page)
        show('prev day', s)
        check(candidates['requests'][n_cand:] == ['2026-09-28'],
              f'"zurück" asks for the server\'s prev_day, not the calendar neighbour ({candidates["requests"][n_cand:]})')
        check(s['meetings'] == [] and s['notes'] == ['Keine Meetings an diesem Tag und den Nachbartagen.'],
              f'empty day: the empty-state sentence ({s["notes"]})')
        check(s['hint'] == 'Uhrzeit der Aufnahme unbekannt.' and s['reference'] == 'Aufnahme: Fr, 02.10.2026 · Dauer 25 min',
              f'hint line shown when the server sends one ({s["hint"]!r})')
        check(s['submit']['disabled'] is True, 'no selection → the send button is disabled')
        page.click('#notion-day-next')            # server says: next of 09-28 is 10-02
        page.wait_for_function("() => document.getElementById('notion-day-input').value === '2026-10-02'")
        page.fill('#notion-day-input', '2026-10-07')
        page.wait_for_function("() => document.querySelectorAll('#notion-meeting-list .notion-meeting').length === 1")
        s = state(page)
        check(candidates['requests'][n_cand:] == ['2026-09-28', '2026-10-02', '2026-10-07'],
              f'"vor" and the date field ask for exactly those days ({candidates["requests"][n_cand:]})')
        check(s['meetings'][0]['checked'] is False and s['submit']['disabled'] is True,
              'a day without preselection: nothing checked, send disabled')
        page.click('#notion-meeting-list .notion-meeting input')
        s = state(page)
        check(s['meetings'][0]['checked'] and s['submit']['disabled'] is False, 'choosing an entry enables the send button')
        page.fill('#notion-day-input', '2026-10-02')
        page.wait_for_function("() => document.querySelectorAll('#notion-meeting-list .notion-meeting').length === 4")

        print('=== 10. send: exact request, 409 → confirm, Nein / Ja, success ===')
        route, busy = click_send_and_hold()
        check(busy['submit'] == {'disabled': True, 'text': SUBMIT_BUSY}, 'in flight: disabled, "Sende …"')
        body = json.loads(route.request.post_data)
        print('[10 body] ' + json.dumps(body))
        check(body == {'target': 'meetings', 'page_id': P_NEAR, 'day': '2026-10-02', 'replace_transcript': False},
              'the request is exactly {target, page_id, day, replace_transcript: false} — no transcript, no fields')
        question = '‚Jour fixe Smoke‘ (Fr, 02.10.2026, 14:30) hat schon ein Transkript. Überschreiben?'
        conflict = json.dumps({'code': 'transcript_exists', 'error': question.replace(' Überschreiben?', ''),
                               'confirm': question})
        dialog_plan[:] = [False]                  # "Nein"
        n_sends = len(held_sends)
        route.fulfill(status=409, content_type='application/json', body=conflict)
        s = settle()
        check(dialogs == [question], f'409 → native confirm with the server\'s sentence ({dialogs})')
        check(len(held_sends) == n_sends and s['alert'] is None and s['submit'] == {'disabled': False, 'text': SUBMIT_IDLE},
              '"Nein": no second request, no alert, button back')
        check(s['link'] is None, '"Nein": no link line')

        dialog_plan[:] = [True]                   # "Ja"
        n_cand = len(candidates['requests'])
        route, _ = click_send_and_hold()
        n_sends = len(held_sends)
        route.fulfill(status=409, content_type='application/json', body=conflict)
        for _ in range(50):
            if len(held_sends) > n_sends:
                break
            page.wait_for_timeout(100)
        check(len(held_sends) == n_sends + 1, '"Ja": the page sends again')
        route = held_sends[-1]
        body = json.loads(route.request.post_data)
        check(body == {'target': 'meetings', 'page_id': P_NEAR, 'day': '2026-10-02', 'replace_transcript': True},
              f'… with replace_transcript: true and nothing else changed ({body})')
        candidates['sent'] = True
        route.fulfill(status=200, content_type='application/json', body=json.dumps({
            'success': True, 'created': False, 'url': LINK_URL,
            'link': {'page_id': P_NEAR, 'url': LINK_URL, 'title': 'Jour fixe Smoke',
                     'date_text': 'Fr, 02.10.2026, 14:30', 'day': '2026-10-02'}}))
        s = settle()
        page.wait_for_function("() => document.querySelectorAll('#notion-meeting-list .notion-badge--here').length === 1")
        s = state(page)
        show('after send', s)
        check(s['toast'] == 'An Notion gesendet' and s['alert'] is None, f'success: toast, no alert ({s["toast"]!r})')
        check(s['opens'] == [], 'no window.open on this way (the link line carries the link)')
        check(s['link'] == {'text': 'Verknüpft mit Jour fixe Smoke, Fr, 02.10.2026, 14:30', 'href': LINK_URL,
                            'target': '_blank', 'rel': 'noopener noreferrer', 'elements': 1},
              f'link line "Verknüpft mit <Titel>, <Datum>" with the Notion link ({s["link"]})')
        check(candidates['requests'][n_cand:] == ['2026-10-02'], 'the list reloaded for the same day')
        check(s['meetings'][0]['meta'] == ['Jour Fixe', 'hat schon Transkript', 'mit diesem Dokument verknüpft']
              and s['meetings'][0]['checked'],
              f'the list shows the new state and keeps the meeting selected ({s["meetings"][0]["meta"]})')

        print('=== 11. 413, 5xx, network abort → the alert ===')
        too_long = 'Das Transkript ist zu lang für Notion: 212.345 Zeichen, erlaubt sind 200.000.'
        route, _ = click_send_and_hold()
        route.fulfill(status=413, content_type='application/json',
                      body=json.dumps({'code': 'transcript_too_long', 'error': too_long, 'length': 212345, 'max': 200000}))
        s = settle()
        check(s['alert'] is not None and s['alert']['text'] == too_long and 'c-alert--danger' in s['alert']['cls'],
              f'413: the sentence with length and limit ({s["alert"]})')
        route, busy = click_send_and_hold()
        check(busy['alert'] is None, 'a new send clears the alert')
        route.fulfill(status=502, content_type='application/json',
                      body=json.dumps({'error': 'Notion-Server nicht erreichbar. Später erneut versuchen.'}))
        s = settle()
        check(s['alert'] is not None and s['alert']['text'] == 'Notion-Server nicht erreichbar. Später erneut versuchen.',
              f'502: the server\'s German sentence ({s["alert"]})')
        route, _ = click_send_and_hold()
        route.abort()
        s = settle()
        check(s['alert'] is not None and s['alert']['text'] == 'Verbindung fehlgeschlagen. Netzwerk prüfen und erneut versuchen.',
              f'aborted request: network sentence ({s["alert"]})')
        check(len(dialogs) == 2, f'no further confirm dialogs ({len(dialogs)})')

        print('=== 12. way and target switches ===')
        n_cand = len(candidates['requests'])
        page.click('#notion-way-group button[data-way=new]')
        page.wait_for_selector('#nf-title')
        s = state(page)
        check(s['way']['pressed'] == ['new'] and s['formVisible'] and not s['existingVisible']
              and keys(s) == ['title', 'datum', 'project', 'people', 'type', 'summary'] and s['alert'] is None,
              '"Neues Meeting anlegen": the six-field form, the list hidden, the alert cleared')
        check(s['submit']['disabled'] is False, 'the form way never disables the send button')
        page.click('#notion-way-group button[data-way=existing]')
        page.wait_for_timeout(300)
        s = state(page)
        check(s['existingVisible'] and len(s['meetings']) == 4 and len(candidates['requests']) == n_cand,
              'back to "Bestehendes Meeting": the list is still there, no new request')
        page.click('#notion-target-group button[data-target=notes]')
        page.wait_for_timeout(300)
        s = state(page)
        check(not s['way']['visible'] and s['formVisible'] and not s['existingVisible'] and s['primary'] == ['notes']
              and s['submit']['disabled'] is False,
              'target "Notiz": no way switch, the note form, send enabled')
        page.click('#notion-target-group button[data-target=meetings]')
        page.wait_for_timeout(300)
        s = state(page)
        check(s['way'] == {'visible': True, 'pressed': ['existing']} and s['existingVisible'],
              'back on "Meeting": the chosen way is kept')

        open_detail(page, ids['doc'])
        page.click('#notion-toggle-btn')
        page.wait_for_selector('#nf-title')
        page.click('#notion-target-group button[data-target=meetings]')
        page.wait_for_selector('#nf-datum')
        s = state(page)
        check(s['way'] == {'visible': True, 'pressed': ['new']} and s['formVisible'] and not s['existingVisible'],
              f'a markdown row keeps "Neues Meeting anlegen" as its default way ({s["way"]})')

        print('=== 13. the remembered link is visible without opening the panel ===')
        n_cand = len(candidates['requests'])
        open_detail(page, ids['linked'])
        s = state(page)
        show('linked row, panel closed', s)
        check(s['panelHidden'] and s['link'] == {
            'text': f'Verknüpft mit {LINK_TITLE}, Fr, 02.10.2026, 14:30', 'href': LINK_URL,
            'target': '_blank', 'rel': 'noopener noreferrer', 'elements': 1},
            f'"Verknüpft mit <Titel>, <Datum>" with the link, title as text ({s["link"]})')
        check(len(candidates['requests']) == n_cand, 'showing the link asks for no candidates')
        belt = page.evaluate("""() => {
            const out = {};
            for (const url of ['javascript:alert(1)', 'http://app.notion.com/p/x', null, 'kein url']) {
                renderNotionLink({url, title: 'T', date_text: null});
                const box = document.getElementById('notion-link-status');
                out[String(url)] = {anchors: box.querySelectorAll('a').length, text: box.textContent};
            }
            renderNotionLink(null);
            out.cleared = document.getElementById('notion-link-status').hidden;
            return out;
        }""")
        print('[13 belt] ' + json.dumps(belt))
        check(all(v == {'anchors': 0, 'text': 'Verknüpft mit T'} for k, v in belt.items() if k != 'cleared') and belt['cleared'],
              'a url that is not https never becomes an anchor; a null link hides the line')

        if OUT:
            print('=== screenshots (SMOKE_OUT set): the panel in light and dark ===')
            for theme in ('light', 'dark'):
                page.evaluate("t => localStorage.setItem('globalTheme', t)", theme)
                open_detail(page, ids['audio'])
                page.click('#notion-toggle-btn')
                page.wait_for_selector('#notion-meeting-list .notion-meeting')
                page.wait_for_timeout(500)
                card = page.locator('#notion-panel').locator('xpath=..')
                card.screenshot(path=f'{OUT}_{theme}.png')
                box = card.bounding_box()
                overflow = page.evaluate("""() => { const c = document.getElementById('notion-panel').parentElement;
                    return [...c.querySelectorAll('*')].filter(e => e.offsetParent !== null
                        && e.getBoundingClientRect().right > c.getBoundingClientRect().right + 0.5).length; }""")
                print(f'[{theme}] card {round(box["width"])}x{round(box["height"])} px → {OUT}_{theme}.png, '
                      f'elements past the card edge: {overflow}')
                check(overflow == 0, f'{theme}: nothing in the panel reaches past the card')
            page.evaluate("() => localStorage.setItem('globalTheme', 'light')")

        print('=== guards ===')
        # The handler above never calls continue_(): a held route is either
        # fulfilled or aborted by the script, so the count IS the proof.
        check(sends_part_a == 9 and len(send_requests_seen) == len(held_sends) == 15,
              f'every send request was answered by the script, none reached the server '
              f'(part A {sends_part_a}, total seen={len(send_requests_seen)}, held={len(held_sends)})')
        check(all(day is None or re.fullmatch(r'\d{4}-\d{2}-\d{2}', day) for day in candidates['requests']),
              'every candidates request was answered by the script')
        check(page_errors == [], f'no uncaught JS error on the page ({page_errors})')

        browser.close()
finally:
    removed, left, tags_left = remove_rows(list(ids.values()), tag_id, user_id)
    print(f'[cleanup] removed {removed} rows of user_id={user_id}, left={left}, tags left={tags_left}')
    check(removed == 3 and left == 0 and tags_left == 0, 'test rows and the test tag are gone')

print()
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('ALL PASSED')
