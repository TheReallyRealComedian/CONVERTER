#!/usr/bin/env python3
"""Browser smoke for "Überspringen" in the review (LEARN-SKIP).

The pytest suite renders no templates and runs no JS — this is the one
REAL-browser check of the skip button in ``templates/review.html`` +
``static/js/review.js``. It runs INSIDE the web container (the playwright
base image ships Chromium) against the deployed app, as a throwaway user
with FOUR cards of its own, and MEASURES the locked decisions instead of
looking at them:

1. Layout — the unrevealed card in light and dark, 375 px and desktop:
   "Aufdecken" and "Überspringen" share one row, both sit inside the card,
   the label spans stay on one line, the card does not overflow sideways
   (screenshots land as SMOKE_OUT_<theme>_<width>.png).
2. Skip on A of A B C D — the request counter (route interception over
   EVERY request) stays at 0, the progress line "Karte 1 von 4", the cap
   line and the collection/orphan badges are unchanged, and A's
   ``rating_history`` is byte-equal before/after (read from the DB).
3. A reload restores the server order: A is first again (the reorder lives
   only in the page's module state).
4. The order after the skip is B C D A — read by rating B, C, D through and
   seeing A return as "Karte 4 von 4".
5. The key "0" does nothing in the note textarea; after the reveal the
   button is hidden and the key is inert; on the last remaining card the
   button is disabled, the key is inert, no request leaves, and no done
   panel appears ("kein Endlos-Kreisel, kein falsches Done-Panel").
6. The returned card A is ratable: exactly ONE request (POST …/review), one
   ``rating_history`` entry, and the done panel says "Alle 4 fälligen
   Karten wiederholt."

Test cards are written through the ORM under the throwaway user's own
``user_id`` — never through ``POST /api/cards``, which always writes to the
INGEST_USER / first user, i.e. Oli's account. The script REFUSES to run for
user id 1 or the INGEST_USER. It removes the cards it created (their Review
rows cascade) at the end, strictly by ``user_id`` + the ids it wrote; the
user itself is created and removed by the operator (recipe below).

How to run (Mintbox, ~1 min):

    # 1. throwaway user — NEVER Oli's account
    docker exec markdown-converter-web flask --app app create-user zz_skip --password '<random>'
    docker cp scripts/smoke_review_skip.py markdown-converter-web:/tmp/smoke_skip.py
    # 2. run — screenshots land in the container as /tmp/smoke_skip_*.png
    docker exec -e SMOKE_USER=zz_skip -e SMOKE_PASSWORD='<random>' markdown-converter-web python /tmp/smoke_skip.py
    # 3. clean up STRICTLY by user_id (the api_token table carries Oli's iOS
    #    tokens): the script already removed its cards; delete the user's
    #    remaining Card/Collection/Tag/Conversion/ApiToken rows and the User
    #    row via the ORM filtered by that user_id, then rm /tmp/smoke_skip*.

Env: BASE_URL (default http://localhost:5000), SMOKE_USER, SMOKE_PASSWORD,
SMOKE_OUT (/tmp/smoke_skip). Exit 0 = every check passed; every measured
value is printed so a failure is diagnosable from the output alone.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from playwright.sync_api import sync_playwright

BASE = os.environ.get('BASE_URL', 'http://localhost:5000')
USER = os.environ.get('SMOKE_USER') or sys.exit('SMOKE_USER missing')
PASSWORD = os.environ.get('SMOKE_PASSWORD') or sys.exit('SMOKE_PASSWORD missing')
OUT = os.environ.get('SMOKE_OUT', '/tmp/smoke_skip')

LABELS = ['A', 'B', 'C', 'D']
QUESTION = 'LEARN-SKIP Karte {}'

failures = []


def check(ok, what):
    print(('  PASS ' if ok else '  FAIL ') + what)
    if not ok:
        failures.append(what)


# --- test data through the ORM (same container, same DB) -------------------
from app import app  # noqa: E402  (bootstrap shim; the CLI imports it the same way)
from models import Card, Review, User, db  # noqa: E402


def resolve_user():
    with app.app_context():
        user = User.query.filter_by(username=USER).first()
        if user is None:
            sys.exit(f'user {USER!r} does not exist — create it with flask create-user')
        if user.id == 1 or USER == os.environ.get('INGEST_USER'):
            sys.exit(f'refusing to run as user id {user.id} ({USER!r}): that is the '
                     'INGEST/first user = Oli\'s account')
        return user.id


def create_cards(user_id):
    """Four brand-new atomic cards A B C D in a total creation order
    (created_at 1 s apart AND ascending ids) — `smart` shows new cards in
    creation order (LEARN-QUEUE), so the page's initial queue is known."""
    base = datetime.now(timezone.utc) - timedelta(minutes=5)
    ids = []
    with app.app_context():
        for i, label in enumerate(LABELS):
            card = Card(user_id=user_id, type='atomic',
                        front=QUESTION.format(label), back=f'Antwort {label}',
                        created_by='smoke', created_at=base + timedelta(seconds=i))
            db.session.add(card)
            db.session.flush()
            card.review = Review(due=datetime.now(timezone.utc) - timedelta(minutes=1),
                                 reps=0, lapses=0)
            ids.append(card.id)
        db.session.commit()
    return ids


def rating_history(card_id, user_id):
    """Raw column text (byte-equality is the claim), fresh from the DB."""
    with app.app_context():
        card = Card.query.filter_by(id=card_id, user_id=user_id).first()
        return card.review.rating_history if card and card.review else None


def remove_cards(ids, user_id):
    with app.app_context():
        cards = Card.query.filter(Card.user_id == user_id, Card.id.in_(ids)).all()
        for c in cards:
            db.session.delete(c)  # Review cascades ORM-side
        db.session.commit()
        left = Card.query.filter(Card.user_id == user_id, Card.id.in_(ids)).count()
        return len(cards), left


# --- page helpers ----------------------------------------------------------
STATE = """() => {
  const q = s => document.querySelector(s);
  const text = s => (q(s) ? q(s).innerText.trim() : null);
  const skip = q('#review-skip-btn');
  const reveal = q('#review-reveal-btn');
  const vis = el => el && !el.classList.contains('hidden') && el.offsetParent !== null;
  return {
    question: text('#review-question'),
    progress: text('#review-progress'),
    cap: text('#review-cap-info'),
    badges: Array.from(document.querySelectorAll('.review-scope-pill__badge')).map(b => b.innerText.trim()),
    revealVisible: vis(reveal),
    skipVisible: vis(skip),
    skipDisabled: skip ? skip.disabled : null,
    skipKeyShown: skip ? skip.innerText.includes('0') : null,
    ratingVisible: vis(q('#review-rating')),
    doneVisible: vis(q('#review-done')),
    cardVisible: vis(q('#review-card')),
    toast: text('.toast-notification'),
  };
}"""

LAYOUT = """() => {
  const r = el => { const b = el.getBoundingClientRect(); return {top: Math.round(b.top), bottom: Math.round(b.bottom), left: Math.round(b.left), right: Math.round(b.right), width: Math.round(b.width), height: Math.round(b.height)}; };
  const card = document.getElementById('review-card');
  const reveal = document.getElementById('review-reveal-btn');
  const skip = document.getElementById('review-skip-btn');
  const label = skip.querySelector('.review-rate-btn__label');
  const meaning = skip.querySelector('.review-rate-btn__meaning');
  return {
    theme: document.documentElement.getAttribute('data-global-theme') || 'light',
    viewport: window.innerWidth,
    pageScrollWidth: document.documentElement.scrollWidth,
    cardOverflow: card.scrollWidth - card.clientWidth,
    card: r(card), reveal: r(reveal), skip: r(skip),
    labelLines: label.getClientRects().length,
    meaningLines: meaning.getClientRects().length,
    skipBg: getComputedStyle(skip).backgroundColor,
    revealBg: getComputedStyle(reveal).backgroundImage.slice(0, 40),
  };
}"""


def open_review(page, theme):
    page.goto(f'{BASE}/login')
    page.evaluate("t => localStorage.setItem('globalTheme', t)", theme)
    page.goto(f'{BASE}/review')
    page.wait_for_selector('#review-card:not(.hidden)')
    page.wait_for_function("() => document.getElementById('review-cap-info').innerText.length > 0")
    page.wait_for_timeout(400)


def state(page):
    return page.evaluate(STATE)


def blur(page):
    """Drop focus before keyboard steps — Space on a focused footer button
    would also activate it (the note toggle refocuses its textarea)."""
    page.evaluate("() => document.activeElement && document.activeElement.blur()")


def label_of(question):
    return question.replace(QUESTION.format(''), '') if question else None


user_id = resolve_user()
card_ids = create_cards(user_id)
print(f'user_id={user_id} cards={dict(zip(LABELS, card_ids))}')
card_a = card_ids[0]

try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={'width': 1200, 'height': 900})
        page = ctx.new_page()
        page.goto(f'{BASE}/login')
        page.fill('#username', USER)
        page.fill('#password', PASSWORD)
        page.click('button[type=submit]')
        page.wait_for_url(lambda url: '/login' not in url)
        print('logged in as', USER)

        print('=== 1. layout: light/dark × 375/desktop, unrevealed card ===')
        for theme in ('light', 'dark'):
            for width in (375, 1200):
                page.set_viewport_size({'width': width, 'height': 800})
                open_review(page, theme)
                m = page.evaluate(LAYOUT)
                page.screenshot(path=f'{OUT}_{theme}_{width}.png')
                print(f'[{theme} {width}] ' + json.dumps(m))
                same_row = abs(m['reveal']['top'] - m['skip']['top']) <= 2 and m['reveal']['right'] <= m['skip']['left']
                check(same_row, f'{theme} {width}: Aufdecken and Überspringen share one row, Aufdecken first')
                inside = (m['skip']['right'] <= m['card']['right'] and m['reveal']['left'] >= m['card']['left']
                          and m['skip']['bottom'] <= m['card']['bottom'])
                check(inside, f'{theme} {width}: both buttons lie inside the card')
                check(m['reveal']['width'] > m['skip']['width'],
                      f'{theme} {width}: Aufdecken is the wider (primary) button '
                      f'({m["reveal"]["width"]} vs {m["skip"]["width"]} px)')
                check(m['labelLines'] == 1 and m['meaningLines'] == 1,
                      f'{theme} {width}: label and meaning each stay on one line')
                # The page SHELL scrolls sideways at 375 px in headless Chromium
                # (main-content/header 394 px) with and without the button —
                # pre-existing, printed as pageScrollWidth; the claim here is
                # that the CARD does not overflow.
                check(m['cardOverflow'] <= 0, f'{theme} {width}: the card has no horizontal overflow ({m["cardOverflow"]} px)')
                check(m['theme'] == theme, f'{theme} {width}: page renders the {theme} theme')
        page.set_viewport_size({'width': 1200, 'height': 900})

        print('=== 2. skip on A: zero requests, counters unchanged, rating_history byte-equal ===')
        open_review(page, 'light')
        requests = []
        page.route('**/*', lambda route: (requests.append(route.request.url), route.continue_()))
        before = state(page)
        hist_before = rating_history(card_a, user_id)
        print('[before] ' + json.dumps(before) + f' rating_history={hist_before!r}')
        check(label_of(before['question']) == 'A', 'first card is A (creation order)')
        check(before['progress'] == 'Karte 1 von 4', f'progress reads "Karte 1 von 4" ({before["progress"]})')
        check(before['skipVisible'] and not before['skipDisabled'] and before['skipKeyShown'],
              'Überspringen is visible, enabled and shows its key "0"')
        n0 = len(requests)
        page.click('#review-skip-btn')
        page.wait_for_timeout(1200)
        after = state(page)
        skip_requests = requests[n0:]
        hist_after = rating_history(card_a, user_id)
        print('[after skip] ' + json.dumps(after) + f' requests={len(skip_requests)} rating_history={hist_after!r}')
        check(label_of(after['question']) == 'B', 'B is shown after skipping A')
        check(len(skip_requests) == 0, f'no request left the page during the skip ({len(skip_requests)}: {skip_requests})')
        check(hist_after == hist_before, 'A.rating_history is byte-equal before/after the skip')
        check(after['progress'] == before['progress'], f'progress unchanged ({after["progress"]})')
        check(after['cap'] == before['cap'], f'cap line unchanged ({after["cap"]!r})')
        check(after['badges'] == before['badges'], f'pill badges unchanged ({after["badges"]})')
        check(not after['ratingVisible'] and after['revealVisible'], 'B starts unrevealed')
        check(after['toast'] is not None, f'a toast confirms the skip ({after["toast"]!r})')

        print('=== 3. a reload restores the server order ===')
        page.unroute('**/*')
        open_review(page, 'light')
        s = state(page)
        check(label_of(s['question']) == 'A', f'after reload the first card is A again ({s["question"]})')
        page.route('**/*', lambda route: (requests.append(route.request.url), route.continue_()))

        print('=== 4./5. order B C D A, key in the note field, after reveal, last card ===')
        page.click('#review-skip-btn')          # A → end
        page.wait_for_timeout(600)
        s = state(page)
        check(label_of(s['question']) == 'B', 'B after skipping A')
        # key "0" typed into the note textarea must not skip
        page.click('#review-note-toggle')
        page.fill('#review-note-input', '')
        page.focus('#review-note-input')
        n0 = len(requests)
        page.keyboard.press('0')
        page.wait_for_timeout(400)
        s = state(page)
        note_val = page.input_value('#review-note-input')
        check(label_of(s['question']) == 'B' and note_val == '0' and len(requests) == n0,
              f'"0" in the note field types a 0 and skips nothing (card {label_of(s["question"])}, note {note_val!r})')
        page.click('#review-note-toggle')       # close the drawer
        blur(page)
        # rate B and C "Gut" via key 3 after revealing with Space
        seen = []
        for expected in ('B', 'C'):
            s = state(page)
            seen.append(label_of(s['question']))
            page.keyboard.press('Space')
            page.wait_for_timeout(200)
            page.keyboard.press('3')
            page.wait_for_function("() => document.getElementById('review-rating').classList.contains('hidden')")
            page.wait_for_timeout(500)
        s = state(page)
        seen.append(label_of(s['question']))
        print(f'[order] seen so far {seen} progress={s["progress"]!r}')
        check(seen == ['B', 'C', 'D'], f'after the skip the order runs B C D ({seen})')
        check(s['progress'] == 'Karte 3 von 4', f'D is "Karte 3 von 4" ({s["progress"]})')
        check(s['skipVisible'] and not s['skipDisabled'], 'on D (A still behind it) Überspringen is enabled')
        # after the reveal: button gone, key inert
        blur(page)
        page.keyboard.press('Space')
        page.wait_for_timeout(300)
        n0 = len(requests)
        s = state(page)
        check(s['ratingVisible'] and not s['skipVisible'] and not s['revealVisible'],
              'after the reveal the skip button is hidden')
        page.keyboard.press('0')
        page.wait_for_timeout(400)
        s2 = state(page)
        check(label_of(s2['question']) == 'D' and s2['ratingVisible'] and len(requests) == n0,
              'key "0" after the reveal changes nothing (still D, still revealed, no request)')
        page.evaluate("() => document.getElementById('review-skip-btn').click()")
        page.wait_for_timeout(300)
        s3 = state(page)
        check(label_of(s3['question']) == 'D' and s3['ratingVisible'], 'a programmatic click on the hidden button changes nothing')
        page.keyboard.press('3')                # rate D → A returns as the last card
        page.wait_for_timeout(800)
        s = state(page)
        print('[last card] ' + json.dumps(s))
        check(label_of(s['question']) == 'A', f'A returns after D ({s["question"]})')
        check(s['progress'] == 'Karte 4 von 4', f'A is "Karte 4 von 4" ({s["progress"]})')
        check(s['skipVisible'] and s['skipDisabled'], 'on the last remaining card Überspringen is visible but disabled')
        n0 = len(requests)
        blur(page)
        page.keyboard.press('0')
        page.wait_for_timeout(400)
        page.evaluate("() => document.getElementById('review-skip-btn').click()")
        page.wait_for_timeout(400)
        s = state(page)
        check(label_of(s['question']) == 'A' and s['cardVisible'] and not s['doneVisible'] and not s['ratingVisible'],
              'last card: key and click change nothing — still A, unrevealed, no done panel')
        check(len(requests) == n0, f'last card: no request left the page ({len(requests) - n0})')

        print('=== 6. the returned card is ratable — the ONE write of this smoke ===')
        hist_before = rating_history(card_a, user_id)
        n0 = len(requests)
        blur(page)
        page.keyboard.press('Space')
        page.wait_for_timeout(200)
        page.keyboard.press('3')
        page.wait_for_selector('#review-done:not(.hidden)')
        page.wait_for_timeout(600)
        rate_requests = [u for u in requests[n0:] if '/api/cards/' in u]
        hist_after = rating_history(card_a, user_id)
        done_text = page.inner_text('#review-done-text').strip()
        print(f'[rate A] card requests={rate_requests} rating_history before={hist_before!r} after={hist_after!r} done={done_text!r}')
        check(rate_requests == [f'{BASE}/api/cards/{card_a}/review'],
              'rating A sent exactly one POST …/review')
        parsed = json.loads(hist_after) if hist_after else []
        check(hist_before in (None, '[]', '') and len(parsed) == 1 and parsed[0]['rating'] == 'good',
              f'A.rating_history now holds exactly one "good" entry ({hist_after!r})')
        check(done_text == 'Alle 4 fälligen Karten wiederholt.', f'done panel text ({done_text!r})')

        browser.close()
finally:
    removed, left = remove_cards(card_ids, user_id)
    print(f'cleanup: removed {removed} cards of user_id={user_id} (left: {left})')

print()
if failures:
    print(f'{len(failures)} FAILED:')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('ALL PASSED')
