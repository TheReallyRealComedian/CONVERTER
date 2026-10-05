#!/usr/bin/env python3
"""Browser smoke for LERN-TEXT — Lerntexte zu Karten und Sammlungen.

The pytest suite renders no JS — this is the one REAL-browser check of the
reader jump in ``static/js/library_detail.js`` (Teil A) and, from Phase 3
on, of the "Lerntext" button, the Nacharbeiten list and the launcher line in
``templates/review.html`` + ``static/js/review.js`` (Teil B). It runs with
Playwright INSIDE the app image (the playwright base image ships Chromium)
as a throwaway user with its OWN document, collection, highlight and cards
written through the ORM, and MEASURES the locked decisions:

Teil A — the reader opens ``/library/<id>#h=<encodeURIComponent(heading)>``
1. Each of three headings (one with umlaut and spaces) lands at the top of
   the viewport (BoundingClientRect), the cue class ``reader-section-target``
   is set and gone again after its ~2 s, no notice is shown — and the
   position holds after the highlights have been applied (one highlight sits
   in the document; the re-jump after ``loadHighlights`` keeps the heading
   where it is).
2. A heading that does not exist, and a malformed hash, show the notice
   "Abschnitt nicht gefunden. Der Text wurde seit der Verknüpfung geändert."
   ABOVE the text — outside ``.reader-view`` (the highlight anchors' text) —
   the text stays readable, the page stays at the top, nothing is cued.
3. Hash beats resume: with a stored reading progress of 50 % the plain URL
   resumes mid-document; the URL with ``#h=<last heading>`` opens on that
   heading instead — and the jump is NOT persisted as progress (the stored
   value is still 50 % afterwards), while a real scroll afterwards still is.
4. An in-page hash change (same tab) jumps again.

Teil B — the review (``/review``), five own cards in creation order
1. Launcher line "Lesen": hidden with nothing checked; ticking the own
   collection's pill shows it with the collection's Lerntext as a link (new
   tab, ``/library/<id>``); unticking hides it again — no extra request.
2. Card A: before the reveal there is NO ``Lerntext`` link in the DOM; after
   the reveal it is there with the heading url (``#h=…``), ``target=_blank``,
   ``rel=noopener`` and a title "<Dokument> › <Überschrift>"; rated Nochmal.
3. Card B (same heading): link gone again before its reveal, back after;
   Nochmal. Card C (no place): no link after the reveal; Nochmal. Card D:
   Vertiefen (one POST …/annotate), then Gut. Card E: Gut.
4. Done panel "Nacharbeiten": one document group whose title links to the
   text, below it the heading links "Säuren und Basen" (ONCE, though two
   cards) and "Redox" (via Vertiefen) — NOT "Einleitung" (Gut) — and between
   the last rating click and the panel exactly the two requests that existed
   before this sprint (POST …/review, GET /api/collections resync): the list
   itself costs 0 requests. "Neu laden" clears the block.

Test rows are written through the ORM under the throwaway user's own
``user_id`` — never through ``POST /api/cards``, which always writes to the
INGEST_USER / first user, i.e. Oli's account. The script REFUSES to run for
user id 1 or the INGEST_USER. Everything it wrote is removed at the end,
strictly by ``user_id`` + the ids it wrote; the user itself is created and
removed by the operator.

How to run — (a) WITHOUT a deploy, on a throwaway instance from the deployed
image with the Mac working tree streamed in (own SQLite under /tmp/w, no
volume, no network; ~2 min). ``zz_first`` is a placeholder so the smoke
user is not user id 1 — the id-1 guard below is the same on every instance:

    cd ~/CODE/CONVERTER && git ls-files -co --exclude-standard | grep -v -E '^(corpus|docs)/' > /tmp/lt_files \
    && COPYFILE_DISABLE=1 tar --no-xattrs -cf - -T /tmp/lt_files | ssh mintbox 'docker run -i --rm --network none --entrypoint sh converter-app:latest -c "
        mkdir /tmp/w && tar -xf - -C /tmp/w && cd /tmp/w \
        && export DATABASE_URL=sqlite:////tmp/w/smoke.db SECRET_KEY=smoke-only REDIS_URL=redis://127.0.0.1:9/0 \
        && flask --app app create-user zz_first --password placeholder-1234 >/dev/null \
        && flask --app app create-user zz_lerntext --password smoke-pass-1234 >/dev/null \
        && (gunicorn --bind 127.0.0.1:5000 --workers 1 --worker-class uvicorn.workers.UvicornWorker app:asgi_app >/tmp/w/gunicorn.log 2>&1 &) \
        && sleep 5 \
        && SMOKE_USER=zz_lerntext SMOKE_PASSWORD=smoke-pass-1234 BASE_URL=http://127.0.0.1:5000 SMOKE_OUT=/tmp/w/smoke python scripts/smoke_lern_text.py"'

(b) against the DEPLOYED web container (Phase 3 / acceptance):

    docker exec markdown-converter-web flask --app app create-user zz_lerntext --password '<random>'
    docker exec -i markdown-converter-web sh -c 'cat > /tmp/smoke_lern_text.py' < scripts/smoke_lern_text.py
    docker exec -e SMOKE_USER=zz_lerntext -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_lern_text.py
    # clean up STRICTLY by user_id (api_token carries Oli's iOS tokens): the
    # script removed its rows; delete the user's remaining Card/Collection/
    # Conversion/ApiToken rows and the User row via the ORM by that user_id,
    # then rm /tmp/smoke_lern_text*.

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SMOKE_OUT (/tmp/smoke_lern_text — screenshot prefix), SMOKE_APP_ROOT
(default: cwd — where ``app.py`` lives). Exit 0 = every check passed; every
measured value is printed so a failure is diagnosable from the output alone.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
OUT = os.environ.get('SMOKE_OUT', '/tmp/smoke_lern_text')

HEADINGS = ['Einleitung', 'Säuren und Basen', 'Redox']
# (context heading | None, label) in creation order — see create_rows.
CARD_PLAN = ((HEADINGS[1], 'A'), (HEADINGS[1], 'B'), (None, 'C'),
             (HEADINGS[2], 'D'), (HEADINGS[0], 'E'))
MISSING_TEXT = 'Abschnitt nicht gefunden. Der Text wurde seit der Verknüpfung geändert.'
PARAS_PER_SECTION = 40          # long enough to scroll at 900 px
HIGHLIGHT_EXACT = 'LERN-TEXT Markierungsanker Absatz 3'

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


def encode(heading):
    # Exactly models.library_url: encodeURIComponent's alphabet.
    return quote(heading, safe="!*'()")


# --- test data through the ORM (same container, same DB) -------------------
sys.path.insert(0, os.environ.get('SMOKE_APP_ROOT', os.getcwd()))
from app import app  # noqa: E402
from models import (Card, Collection, CollectionDocument, Conversion, Highlight,  # noqa: E402
                    Review, User, db)


def resolve_user():
    with app.app_context():
        user = User.query.filter_by(username=USER).first()
        if user is None:
            sys.exit(f'user {USER!r} does not exist — create it with flask create-user')
        if user.id == 1 or USER == os.environ.get('INGEST_USER'):
            sys.exit(f'refusing to run as user id {user.id} ({USER!r}): that is the '
                     'INGEST/first user = Oli\'s account')
        return user.id


def build_document():
    lines = []
    for i, heading in enumerate(HEADINGS):
        lines.append(('# ' if i == 0 else '## ') + heading)
        lines.append('')
        for p in range(1, PARAS_PER_SECTION + 1):
            if i == 0 and p == 3:
                lines.append(f'{HIGHLIGHT_EXACT} — dieser Satz trägt die Markierung des Smokes.')
            else:
                lines.append(f'Absatz {p} unter {heading}: Lerntext-Prosa des Smokes, '
                             f'lang genug, dass die Seite scrollt.')
            lines.append('')
    return '\n'.join(lines) + '\n'


def create_rows(user_id):
    """One document (three headings, one highlight), one collection listing
    it, two cards with a place in it and one without — all the throwaway
    user's own."""
    ids = {}
    with app.app_context():
        doc = Conversion(user_id=user_id, conversion_type='markdown_input',
                         title='LERN-TEXT Smoke-Dokument', content=build_document(),
                         lifecycle_status='archive')
        db.session.add(doc)
        db.session.flush()
        hl = Highlight(conversion_id=doc.id, exact=HIGHLIGHT_EXACT, prefix='', suffix='')
        db.session.add(hl)
        col = Collection(user_id=user_id, name='LERN-TEXT Smoke-Sammlung')
        db.session.add(col)
        db.session.flush()
        db.session.add(CollectionDocument(collection_id=col.id, conversion_id=doc.id,
                                          position=0))
        base = datetime.now(timezone.utc) - timedelta(minutes=5)
        cards = []
        # Teil B walks them in creation order (smart: new cards by created_at):
        # A, B → same heading, both Nochmal (heading listed ONCE); C → no
        # place, Nochmal (not listed); D → Vertiefen + Gut (listed via
        # Vertiefen); E → Gut (not listed).
        for i, (heading, label) in enumerate(CARD_PLAN):
            card = Card(user_id=user_id, type='atomic', front=f'LERN-TEXT Karte {label}',
                        back=f'Antwort {label}', created_by='smoke',
                        created_at=base + timedelta(seconds=i),
                        context_conversion_id=doc.id if heading else None,
                        context_heading=heading)
            db.session.add(card)
            db.session.flush()
            card.review = Review(due=datetime.now(timezone.utc) - timedelta(minutes=1),
                                 reps=0, lapses=0)
            card.collections.append(col)
            cards.append(card.id)
        db.session.commit()
        ids.update(doc=doc.id, highlight=hl.id, collection=col.id, cards=cards)
    return ids


def set_progress(doc_id, user_id, percent):
    with app.app_context():
        doc = Conversion.query.filter_by(id=doc_id, user_id=user_id).first()
        doc.last_read_percent = percent
        db.session.commit()


def get_progress(doc_id, user_id):
    with app.app_context():
        doc = Conversion.query.filter_by(id=doc_id, user_id=user_id).first()
        return doc.last_read_percent


def remove_rows(ids, user_id):
    with app.app_context():
        cards = Card.query.filter(Card.user_id == user_id, Card.id.in_(ids['cards'])).all()
        for c in cards:
            db.session.delete(c)
        col = Collection.query.filter_by(id=ids['collection'], user_id=user_id).first()
        if col is not None:
            db.session.delete(col)
        doc = Conversion.query.filter_by(id=ids['doc'], user_id=user_id).first()
        if doc is not None:
            db.session.delete(doc)   # highlight cascades, junction + card context ORM-side
        db.session.commit()
        left = (Card.query.filter(Card.user_id == user_id, Card.id.in_(ids['cards'])).count()
                + Collection.query.filter_by(id=ids['collection'], user_id=user_id).count()
                + Conversion.query.filter_by(id=ids['doc'], user_id=user_id).count())
        return len(cards) + (col is not None) + (doc is not None), left


# --- page helpers ----------------------------------------------------------
READER_STATE = """(target) => {
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const reader = document.querySelector('.reader-view');
  const headings = Array.from(reader.querySelectorAll('h1,h2,h3,h4,h5,h6'));
  const h = headings.find(el => norm(el.textContent) === norm(target)) || null;
  const rect = h ? h.getBoundingClientRect() : null;
  const notice = document.getElementById('reader-section-notice');
  const scroller = document.scrollingElement;
  return {
    found: !!h,
    top: rect ? Math.round(rect.top) : null,
    cued: h ? h.classList.contains('reader-section-target') : null,
    cuedAnywhere: document.querySelectorAll('.reader-section-target').length,
    noticeText: notice ? notice.textContent : null,
    noticeVisible: !!(notice && notice.offsetParent !== null),
    noticeInsideReader: !!(notice && reader.contains(notice)),
    readerTextLength: reader.innerText.length,
    highlightsApplied: document.querySelectorAll('.reader-view span.highlight').length,
    scrollTop: Math.round(scroller.scrollTop),
    scrollMax: Math.round(scroller.scrollHeight - scroller.clientHeight),
    progressWidth: (document.getElementById('reading-progress-fill') || {style: {}}).style.width || '',
    hash: location.hash,
  };
}"""


def reader_state(page, target):
    return page.evaluate(READER_STATE, target)


def open_doc(page, doc_id, hash_part=''):
    # A goto to the same path with another fragment is a SAME-DOCUMENT
    # navigation in Chromium (hashchange, no reload — measured: the page kept
    # its scroll position and the "fresh load" claims measured the wrong
    # path). about:blank in between forces a real load every time.
    page.goto('about:blank')
    page.goto(f'{BASE}/library/{doc_id}{hash_part}')
    page.wait_for_selector('.reader-view')
    page.wait_for_timeout(150)


def login(page):
    page.goto(f'{BASE}/login')
    page.fill('#username', USER)
    page.fill('#password', PASSWORD)
    page.click('button[type=submit]')
    page.wait_for_url(lambda url: '/login' not in url)
    print('logged in as', USER)


def part_a(page, ids, user_id):
    doc_id = ids['doc']
    print('=== A1. three headings land at the top, cue set then gone, no notice ===')
    for heading in HEADINGS:
        open_doc(page, doc_id, '#h=' + encode(heading))
        try:
            page.wait_for_selector('.reader-section-target', timeout=3000)
        except Exception:
            pass
        first = reader_state(page, heading)
        page.wait_for_timeout(1500)          # highlights are in by now → re-jump happened
        settled = reader_state(page, heading)
        page.wait_for_timeout(1500)          # > SECTION_TARGET_MS since the last (re)jump
        faded = reader_state(page, heading)
        page.screenshot(path=f'{OUT}_a1_{HEADINGS.index(heading)}.png')
        print(f'[{heading}] first={json.dumps(first, ensure_ascii=False)}')
        print(f'[{heading}] settled={json.dumps(settled, ensure_ascii=False)} faded.cued={faded["cued"]} faded.cuedAnywhere={faded["cuedAnywhere"]}')
        check(first['found'] and first['cued'], f'{heading!r}: heading found and cued right after the open')
        check(-2 <= first['top'] <= 40, f'{heading!r}: heading at the top of the viewport (top={first["top"]} px)')
        check(settled['highlightsApplied'] >= 1, f'{heading!r}: the highlight was applied ({settled["highlightsApplied"]} span)')
        check(-2 <= settled['top'] <= 40, f'{heading!r}: still at the top after the highlights (top={settled["top"]} px)')
        check(not faded['cued'] and faded['cuedAnywhere'] == 0, f'{heading!r}: the cue class is gone after its decay')
        check(first['noticeText'] is None and faded['noticeText'] is None, f'{heading!r}: no notice')
        if heading != HEADINGS[0]:
            check(first['scrollTop'] > 0, f'{heading!r}: the page actually scrolled (scrollTop={first["scrollTop"]})')

    print('=== A2. missing heading and malformed hash → notice above the text, nothing cued, no resume ===')
    set_progress(doc_id, user_id, 50.0)       # a #h= beats the resume even on a miss
    for label, hash_part in (('missing', '#h=' + encode('Gibt es nicht')),
                             ('malformed', '#h=%E0%A4%A')):
        open_doc(page, doc_id, hash_part)
        page.wait_for_timeout(1200)
        s = reader_state(page, 'Einleitung')
        page.screenshot(path=f'{OUT}_a2_{label}.png')
        print(f'[{label}] {json.dumps(s, ensure_ascii=False)}')
        check(s['noticeVisible'] and s['noticeText'] == MISSING_TEXT, f'{label}: the notice is visible with the exact sentence')
        check(not s['noticeInsideReader'], f'{label}: the notice sits OUTSIDE .reader-view')
        check(s['readerTextLength'] > 2000, f'{label}: the text stays readable ({s["readerTextLength"]} chars)')
        check(s['scrollTop'] == 0, f'{label}: the page stays at the top — no resume to 50 % (scrollTop={s["scrollTop"]})')
        check(s['cuedAnywhere'] == 0, f'{label}: nothing is cued')
    set_progress(doc_id, user_id, None)

    print('=== A3. hash beats resume; the jump is not persisted, a real scroll is ===')
    set_progress(doc_id, user_id, 50.0)
    open_doc(page, doc_id)
    page.wait_for_timeout(1200)
    plain = reader_state(page, HEADINGS[-1])
    print(f'[plain url] {json.dumps(plain, ensure_ascii=False)}')
    check(plain['scrollTop'] > 0 and abs(plain['scrollTop'] / max(plain['scrollMax'], 1) - 0.5) < 0.08,
          f'plain URL resumes mid-document (scrollTop={plain["scrollTop"]} of {plain["scrollMax"]})')
    check(plain['top'] > 100, f'plain URL: the last heading is below the fold (top={plain["top"]})')
    set_progress(doc_id, user_id, 50.0)       # the resume/open may have nudged it; pin it again
    open_doc(page, doc_id, '#h=' + encode(HEADINGS[-1]))
    page.wait_for_timeout(1500)
    jumped = reader_state(page, HEADINGS[-1])
    page.wait_for_timeout(3000)               # longer than the persist throttle (2 s)
    stored_after_jump = get_progress(doc_id, user_id)
    print(f'[hash url] {json.dumps(jumped, ensure_ascii=False)} stored_after_jump={stored_after_jump}')
    check(-2 <= jumped['top'] <= 40, f'hash URL opens on the last heading, not at 50 % (top={jumped["top"]})')
    check(jumped['scrollTop'] > plain['scrollTop'], f'the jump landed further down than the resume ({jumped["scrollTop"]} > {plain["scrollTop"]})')
    check(stored_after_jump == 50.0, f'the jump did NOT persist progress (stored={stored_after_jump})')
    page.mouse.wheel(0, 600)
    page.wait_for_timeout(3000)
    stored_after_scroll = get_progress(doc_id, user_id)
    after_scroll = reader_state(page, HEADINGS[-1])
    print(f'[after wheel] scrollTop={after_scroll["scrollTop"]} stored_after_scroll={stored_after_scroll}')
    check(stored_after_scroll is not None and stored_after_scroll > 50.0,
          f'a real scroll afterwards persists again (stored={stored_after_scroll})')

    print('=== A4. in-page hash change jumps again ===')
    open_doc(page, doc_id, '#h=' + encode(HEADINGS[0]))
    page.wait_for_timeout(800)
    page.evaluate("h => { location.hash = '#h=' + h; }", encode(HEADINGS[1]))
    page.wait_for_timeout(800)
    s = reader_state(page, HEADINGS[1])
    print(f'[hashchange] {json.dumps(s, ensure_ascii=False)}')
    check(s['found'] and -2 <= s['top'] <= 40 and s['cued'], f'hashchange to {HEADINGS[1]!r} lands on it and cues it (top={s["top"]})')


REVIEW_STATE = """() => {
  const q = s => document.querySelector(s);
  const text = s => (q(s) ? q(s).innerText.trim() : null);
  const vis = el => !!el && !el.classList.contains('hidden') && el.offsetParent !== null;
  const link = q('#review-lerntext-link');
  const readLine = q('#review-read-line');
  const rework = q('#review-rework');
  const groups = Array.from(document.querySelectorAll('#review-rework-list .review-rework__group')).map(g => ({
    doc: g.querySelector('.review-rework__doc').textContent,
    docHref: g.querySelector('.review-rework__doc').getAttribute('href'),
    headings: Array.from(g.querySelectorAll('.review-rework__list a')).map(a => ({text: a.textContent, href: a.getAttribute('href'), target: a.target, rel: a.rel})),
  }));
  return {
    question: text('#review-question'),
    revealed: vis(q('#review-rating')),
    linkInDom: !!link,
    linkHref: link ? link.getAttribute('href') : null,
    linkTarget: link ? link.target : null,
    linkRel: link ? link.rel : null,
    linkTitle: link ? link.title : null,
    linkText: link ? link.textContent : null,
    linkFirstInRow: link ? link.parentElement.firstElementChild === link : null,
    readLineVisible: vis(readLine),
    readLinks: Array.from(document.querySelectorAll('#review-read-list a')).map(a => ({text: a.textContent, href: a.getAttribute('href'), target: a.target, rel: a.rel})),
    doneVisible: vis(q('#review-done')),
    reworkVisible: vis(rework),
    reworkGroups: groups,
    groupChildren: q('#review-rework-list') ? q('#review-rework-list').children.length : null,
    emptyVisible: vis(q('#review-empty')),
  };
}"""


def review_state(page):
    return page.evaluate(REVIEW_STATE)


def open_review(page):
    page.goto('about:blank')
    page.goto(f'{BASE}/review')
    page.wait_for_selector('#review-card:not(.hidden), #review-empty:not(.hidden), #review-done:not(.hidden)')
    page.wait_for_function("() => document.querySelectorAll('.review-scope-pill[data-col]').length > 0")
    page.wait_for_timeout(400)


def label_of(question):
    return (question or '').replace('LERN-TEXT Karte ', '')


def blur(page):
    page.evaluate("() => document.activeElement && document.activeElement.blur()")


def part_b(page, ids, user_id):
    doc_id = ids['doc']
    doc_url = f'/library/{doc_id}'
    requests = []
    page.route('**/*', lambda route: (requests.append(f'{route.request.method} {route.request.url}'),
                                      route.continue_()))

    print('=== B1. launcher line "Lesen" follows the checked pills, no extra request ===')
    open_review(page)
    s = review_state(page)
    print(f'[launcher start] readLineVisible={s["readLineVisible"]} readLinks={json.dumps(s["readLinks"], ensure_ascii=False)}')
    check(not s['readLineVisible'] and s['readLinks'] == [], 'nothing checked → no "Lesen" line')
    n0 = len(requests)
    page.click(f'.review-scope-pill[data-col="{ids["collection"]}"] input')
    page.wait_for_timeout(1200)
    s = review_state(page)
    tick_requests = requests[n0:]
    print(f'[pill on] readLineVisible={s["readLineVisible"]} readLinks={json.dumps(s["readLinks"], ensure_ascii=False)} requests={tick_requests}')
    check(s['readLineVisible'], 'own collection checked → the "Lesen" line is visible')
    check(s['readLinks'] == [{'text': 'LERN-TEXT Smoke-Dokument', 'href': doc_url, 'target': '_blank', 'rel': 'noopener'}],
          f'the line links the collection\'s Lerntext (new tab, {doc_url})')
    check(all('/api/review-state' in r for r in tick_requests),
          f'ticking the pill fetched only the queue — no document request ({tick_requests})')
    page.click(f'.review-scope-pill[data-col="{ids["collection"]}"] input')
    page.wait_for_timeout(1200)
    s = review_state(page)
    check(not s['readLineVisible'] and s['readLinks'] == [], 'pill unchecked → the line is gone')

    print('=== B2./B3. Lerntext link only after the reveal, per card ===')
    expected = {'A': f'{doc_url}#h=' + encode(HEADINGS[1]), 'B': f'{doc_url}#h=' + encode(HEADINGS[1]),
                'C': None, 'D': f'{doc_url}#h=' + encode(HEADINGS[2]), 'E': f'{doc_url}#h=' + encode(HEADINGS[0])}
    ratings = {'A': '1', 'B': '1', 'C': '1', 'D': '3', 'E': '3'}
    last_rate_index = None
    for label in ('A', 'B', 'C', 'D', 'E'):
        s = review_state(page)
        check(label_of(s['question']) == label, f'card {label} is up ({s["question"]!r})')
        check(not s['linkInDom'], f'{label}: before the reveal there is NO Lerntext link in the DOM')
        blur(page)
        page.keyboard.press('Space')
        page.wait_for_timeout(300)
        s = review_state(page)
        print(f'[{label} revealed] ' + json.dumps({k: s[k] for k in ('linkInDom', 'linkHref', 'linkTarget', 'linkRel', 'linkTitle', 'linkText', 'linkFirstInRow')}, ensure_ascii=False))
        check(s['revealed'], f'{label}: revealed')
        if expected[label] is None:
            check(not s['linkInDom'], f'{label}: no place → no link after the reveal')
        else:
            heading = dict((lbl, h) for h, lbl in CARD_PLAN)[label]
            check(s['linkInDom'] and s['linkHref'] == expected[label], f'{label}: link with href {expected[label]}')
            check(s['linkTarget'] == '_blank' and s['linkRel'] == 'noopener', f'{label}: target=_blank, rel=noopener')
            check(s['linkText'] == 'Lerntext' and s['linkTitle'] == f'LERN-TEXT Smoke-Dokument › {heading}', f'{label}: text "Lerntext", title "<Dokument> › <Überschrift>" ({s["linkTitle"]!r})')
            check(s['linkFirstInRow'], f'{label}: the link is the first element of the footer row')
        if label == 'D':
            n0 = len(requests)
            page.click('#review-deepen-btn')
            page.wait_for_function("() => document.getElementById('review-deepen-btn').classList.contains('is-active')")
            page.wait_for_timeout(300)
            deepen_requests = [r for r in requests[n0:] if '/api/' in r]
            check(deepen_requests == [f'POST {BASE}/api/cards/{ids["cards"][3]}/annotate'],
                  f'D: Vertiefen sent exactly one POST …/annotate ({deepen_requests})')
        blur(page)
        last_rate_index = len(requests)
        page.keyboard.press(ratings[label])
        # The next card hides the rating row; the LAST card hides the whole
        # card and shows the done panel (the row keeps its state) — wait for
        # either, like smoke_review_skip waits for the done panel at the end.
        page.wait_for_function(
            "() => document.getElementById('review-rating').classList.contains('hidden')"
            " || !document.getElementById('review-done').classList.contains('hidden')")
        page.wait_for_timeout(600)

    print('=== B4. done panel: Nacharbeiten from session state, 0 extra requests ===')
    page.wait_for_selector('#review-done:not(.hidden)')
    page.wait_for_timeout(800)
    s = review_state(page)
    after_last_rate = [r for r in requests[last_rate_index:] if '/api/' in r or '/library' in r]
    print(f'[done] reworkVisible={s["reworkVisible"]} groups={json.dumps(s["reworkGroups"], ensure_ascii=False)} requests_after_last_rate={after_last_rate}')
    check(s['doneVisible'] and s['reworkVisible'], 'done panel with the Nacharbeiten block')
    check(len(s['reworkGroups']) == 1 and s['reworkGroups'][0]['doc'] == 'LERN-TEXT Smoke-Dokument'
          and s['reworkGroups'][0]['docHref'] == doc_url, 'one document group, its title links to the text')
    heads = s['reworkGroups'][0]['headings'] if s['reworkGroups'] else []
    check([h['text'] for h in heads] == [HEADINGS[1], HEADINGS[2]],
          f'headings: "{HEADINGS[1]}" once (two cards) + "{HEADINGS[2]}" (Vertiefen), not "{HEADINGS[0]}" (Gut) — got {[h["text"] for h in heads]}')
    check(all(h['target'] == '_blank' and h['rel'] == 'noopener' for h in heads), 'every heading link opens a new tab (noopener)')
    check([h['href'] for h in heads] == [expected['A'], expected['D']], 'heading links carry the #h= urls')
    check(after_last_rate == [f'POST {BASE}/api/cards/{ids["cards"][4]}/review', f'GET {BASE}/api/collections'],
          f'between the last rating and the panel only the two pre-existing requests — the list cost none ({after_last_rate})')
    page.click('#review-reload')
    page.wait_for_timeout(1200)
    s = review_state(page)
    check(not s['reworkVisible'] and s['groupChildren'] == 0, f'"Neu laden" clears the block (visible={s["reworkVisible"]}, children={s["groupChildren"]})')
    page.unroute('**/*')


user_id = resolve_user()
ids = create_rows(user_id)
print(f'user_id={user_id} rows={json.dumps(ids)}')

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={'width': 1200, 'height': 900})
        page = ctx.new_page()
        login(page)
        part_a(page, ids, user_id)
        part_b(page, ids, user_id)
        browser.close()
finally:
    removed, left = remove_rows(ids, user_id)
    print(f'cleanup: removed {removed} rows of user_id={user_id} (left: {left})')

print()
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('ALL PASSED')
