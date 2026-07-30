"""LEARN-UP — learn settings (one JSON blob), review-queue ordering (P1) and
daily caps (P3).

Pure-logic tests drive ``order_due_cards`` with fake cards and a seeded RNG
(deterministic shuffles); HTTP tests cover the settings roundtrip and the
review-state ordering/capping through the public boundary. The date-order
regression tests are the point of P1: equal-R cards must NOT come back in
creation order, and brand-new cards must be interleaved, not front-loaded.
P3 locks: caps trim AFTER ordering (shakiest/random N), new cards respect the
review cap, and the day is the BERLIN-local day counted against what was
already studied today. LEARN-QUEUE narrows P1 for the ``fresh`` pool only:
new cards come back in creation order (stable across fetches, oldest N under
the cap) — the anti-creation-order assertions keep holding for REVIEWS.
"""
import json
import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app_pkg.learn import (LEARN_SETTINGS_DEFAULTS, capped_session_counts,
                           count_done_today, forecast_buckets,
                           get_user_settings, local_day_bounds, local_day_end,
                           maturity_counts, order_due_cards, true_retention)
from models import Card, Review, User, db
from services.scheduler import (DEFAULT_MAXIMUM_INTERVAL, FSRSScheduler,
                                SM2Scheduler, get_scheduler)
from services.scheduler.fsrs_scheduler import simulate_workload

SETTINGS_URL = '/api/learn/settings'


# --- helpers -----------------------------------------------------------------

def _fake_card(cid, stability=None, days_ago=None, due=None, now=None,
               created_at=None):
    now = now or datetime.now(timezone.utc)
    review = SimpleNamespace(
        due=due or now,
        stability=stability,
        difficulty=5.0 if stability is not None else None,
        last_reviewed=(now - timedelta(days=days_ago)) if days_ago is not None else None,
    )
    return SimpleNamespace(id=cid, review=review, created_at=created_at)


def _ids(cards):
    return [c.id for c in cards]


def _make_card_with_review(app, user_id, stability=None, days_ago=None,
                           due_hours=-1):
    """A due card; ``stability=None`` = brand-new, else previously reviewed."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        card = Card(user_id=user_id, type='atomic', front='Q', back='A')
        card.review = Review(
            due=now + timedelta(hours=due_hours),
            stability=stability,
            difficulty=5.0 if stability is not None else None,
            last_reviewed=(now - timedelta(days=days_ago)) if days_ago is not None else None,
            reps=1 if stability is not None else 0,
        )
        db.session.add(card)
        db.session.commit()
        return card.id


# --- order_due_cards: smart mode ---------------------------------------------

def test_smart_orders_by_retrievability_ascending():
    now = datetime.now(timezone.utc)
    # Same elapsed (10 d), distinct stability → distinct R; lower S = lower R.
    shaky = _fake_card(1, stability=1.0, days_ago=10, now=now)
    mid = _fake_card(2, stability=10.0, days_ago=10, now=now)
    solid = _fake_card(3, stability=100.0, days_ago=10, now=now)
    out = order_due_cards([solid, shaky, mid], 'smart', FSRSScheduler(),
                          rng=random.Random(7), now=now)
    assert _ids(out) == [1, 2, 3]  # shakiest first


def test_smart_random_tiebreak_breaks_creation_order():
    now = datetime.now(timezone.utc)
    # Six cards with IDENTICAL R (same stability, same elapsed day) — input
    # order stands in for creation order. Without the tiebreak the stable
    # sort would hand back exactly the input order.
    cards = [_fake_card(i, stability=10.0, days_ago=5, now=now) for i in range(1, 7)]
    out_a = order_due_cards(cards, 'smart', FSRSScheduler(),
                            rng=random.Random(1), now=now)
    out_b = order_due_cards(cards, 'smart', FSRSScheduler(),
                            rng=random.Random(2), now=now)
    assert sorted(_ids(out_a)) == [1, 2, 3, 4, 5, 6]
    assert _ids(out_a) != [1, 2, 3, 4, 5, 6]      # not creation order
    assert _ids(out_a) != _ids(out_b)              # seed-dependent, truly random


def test_smart_interleaves_new_cards_evenly_not_frontloaded():
    now = datetime.now(timezone.utc)
    reviewed = [_fake_card(i, stability=10.0, days_ago=5, now=now) for i in range(1, 9)]
    fresh = [_fake_card(i, stability=None, now=now) for i in (100, 101)]
    out = order_due_cards(reviewed + fresh, 'smart', FSRSScheduler(),
                          rng=random.Random(3), now=now)
    fresh_pos = [i for i, c in enumerate(out) if c.id >= 100]
    # Fractional-position merge: 2 new among 8 reviews land at slots 2 and 7 —
    # spread through the stream, never at the front.
    assert fresh_pos == [2, 7]
    assert sorted(_ids(out)) == [1, 2, 3, 4, 5, 6, 7, 8, 100, 101]


def test_smart_sm2_engine_degrades_to_due_ascending():
    # SM-2 has no R (base-class None) → reviewed cards fall back to due-asc.
    now = datetime.now(timezone.utc)
    oldest = _fake_card(1, stability=6.0, days_ago=9, due=now - timedelta(days=3), now=now)
    middle = _fake_card(2, stability=6.0, days_ago=9, due=now - timedelta(days=2), now=now)
    newest = _fake_card(3, stability=6.0, days_ago=9, due=now - timedelta(days=1), now=now)
    out = order_due_cards([newest, oldest, middle], 'smart', SM2Scheduler(),
                          rng=random.Random(4), now=now)
    assert _ids(out) == [1, 2, 3]


# --- LEARN-QUEUE: new cards in creation order (smart mode) -------------------

def _fresh_batch(now, specs):
    """``specs`` = (id, minutes_ago_created) pairs → brand-new fake cards."""
    return [_fake_card(cid, stability=None, now=now,
                       created_at=now - timedelta(minutes=mins))
            for cid, mins in specs]


def test_new_cards_are_stable_across_fetches():
    # THE core property: two fetches on unchanged data hand back the same new
    # cards in the same order — the queue must not re-roll every 15 minutes.
    now = datetime.now(timezone.utc)
    cards = _fresh_batch(now, [(1, 50), (2, 40), (3, 30), (4, 20), (5, 10)])
    out_a = order_due_cards(cards, 'smart', FSRSScheduler(),
                            rng=random.Random(1), now=now)
    out_b = order_due_cards(cards, 'smart', FSRSScheduler(),
                            rng=random.Random(999), now=now)
    assert _ids(out_a) == _ids(out_b) == [1, 2, 3, 4, 5]


def test_new_cards_follow_created_at_then_id():
    now = datetime.now(timezone.utc)
    # Input order is deliberately scrambled and the ids run counter to the
    # timestamps, so only created_at can produce the expected sequence.
    cards = _fresh_batch(now, [(60, 10), (10, 60), (40, 30),
                               (20, 50), (50, 20), (30, 40)])
    out = order_due_cards(cards, 'smart', FSRSScheduler(),
                          rng=random.Random(2), now=now)
    assert _ids(out) == [10, 20, 30, 40, 50, 60]
    # Batch write → identical created_at; id is the tiebreak that makes the
    # order total (and therefore reproducible).
    same = now - timedelta(minutes=5)
    tied = [_fake_card(cid, stability=None, now=now, created_at=same)
            for cid in (13, 3, 8, 21, 5, 1)]
    out = order_due_cards(tied, 'smart', FSRSScheduler(),
                          rng=random.Random(3), now=now)
    assert _ids(out) == [1, 3, 5, 8, 13, 21]


def test_new_cap_takes_the_oldest_n():
    # Explicitly wanted consequence: the daily cap cuts chapter 5, not a
    # random slice of chapters 4 and 5.
    now = datetime.now(timezone.utc)
    cards = _fresh_batch(now, [(1, 50), (2, 40), (3, 30), (4, 20), (5, 10)])
    out = order_due_cards(cards, 'smart', FSRSScheduler(), rng=random.Random(4),
                          now=now, new_budget=2)
    assert _ids(out) == [1, 2]


def test_new_cards_stable_while_reviews_keep_random_tiebreak():
    # The two pools are governed differently in the SAME call: equal-R reviews
    # stay seed-dependent, the new cards do not move.
    now = datetime.now(timezone.utc)
    reviewed = [_fake_card(i, stability=10.0, days_ago=5, now=now) for i in range(1, 7)]
    fresh = _fresh_batch(now, [(100, 30), (101, 20), (102, 10)])
    out_a = order_due_cards(reviewed + fresh, 'smart', FSRSScheduler(),
                            rng=random.Random(1), now=now)
    out_b = order_due_cards(reviewed + fresh, 'smart', FSRSScheduler(),
                            rng=random.Random(2), now=now)
    assert [c.id for c in out_a if c.id >= 100] == [100, 101, 102]
    assert [c.id for c in out_b if c.id >= 100] == [100, 101, 102]
    assert [c.id for c in out_a if c.id < 100] != [c.id for c in out_b if c.id < 100]


# --- order_due_cards: random mode --------------------------------------------

def test_random_all_new_is_shuffled():
    # Moved here from smart mode by LEARN-QUEUE: 'random' means random, so the
    # assertion still holds — just not where new cards now keep their order.
    now = datetime.now(timezone.utc)
    cards = [_fake_card(i, stability=None, now=now) for i in range(1, 8)]
    out = order_due_cards(cards, 'random', FSRSScheduler(),
                          rng=random.Random(5), now=now)
    assert sorted(_ids(out)) == list(range(1, 8))
    assert _ids(out) != list(range(1, 8))


def test_random_mode_is_a_full_shuffle():
    now = datetime.now(timezone.utc)
    cards = ([_fake_card(i, stability=10.0, days_ago=5, now=now) for i in range(1, 6)]
             + [_fake_card(i, stability=None, now=now) for i in (100, 101)])
    out = order_due_cards(cards, 'random', FSRSScheduler(),
                          rng=random.Random(11), now=now)
    assert sorted(_ids(out)) == [1, 2, 3, 4, 5, 100, 101]
    assert _ids(out) != _ids(cards)


# --- settings: helper + API --------------------------------------------------

def test_get_user_settings_defaults_and_corrupt_blob():
    assert get_user_settings(SimpleNamespace(settings_json=None)) == LEARN_SETTINGS_DEFAULTS
    assert get_user_settings(SimpleNamespace(settings_json='{broken')) == LEARN_SETTINGS_DEFAULTS
    assert get_user_settings(SimpleNamespace(settings_json='"smart"')) == LEARN_SETTINGS_DEFAULTS
    assert get_user_settings(
        SimpleNamespace(settings_json='{"ordering_mode": "bogus", "junk": 1}')
    ) == LEARN_SETTINGS_DEFAULTS
    assert get_user_settings(
        SimpleNamespace(settings_json='{"ordering_mode": "random"}')
    )['ordering_mode'] == 'random'


def test_settings_get_defaults(authenticated_client):
    body = authenticated_client.get(SETTINGS_URL).get_json()
    assert body == {'ordering_mode': 'smart',
                    'daily_new_limit': 10, 'daily_review_limit': 200}


def test_settings_put_roundtrip_persists(app, authenticated_client, test_user):
    resp = authenticated_client.put(SETTINGS_URL, json={'ordering_mode': 'random'})
    assert resp.status_code == 200
    assert resp.get_json()['ordering_mode'] == 'random'
    body = authenticated_client.get(SETTINGS_URL).get_json()
    assert body['ordering_mode'] == 'random'
    with app.app_context():
        stored = db.session.get(User, test_user['id']).settings_json
        assert json.loads(stored)['ordering_mode'] == 'random'


def test_settings_put_rejects_invalid_value(authenticated_client):
    resp = authenticated_client.put(SETTINGS_URL, json={'ordering_mode': 'chaos'})
    assert resp.status_code == 400
    # Nothing written — GET still hands back the default.
    assert authenticated_client.get(SETTINGS_URL).get_json()['ordering_mode'] == 'smart'


def test_settings_put_rejects_unknown_key_and_bad_body(authenticated_client):
    assert authenticated_client.put(
        SETTINGS_URL, json={'daily_beer_limit': 3}).status_code == 400
    assert authenticated_client.put(
        SETTINGS_URL, json=['ordering_mode']).status_code == 400
    assert authenticated_client.put(
        SETTINGS_URL, data='no json', content_type='text/plain').status_code == 400


def test_settings_require_login(client):
    assert client.get(SETTINGS_URL).status_code in (302, 401)
    assert client.put(SETTINGS_URL, json={'ordering_mode': 'random'}).status_code in (302, 401)


# --- P3: local_day_bounds (Berlin, DST-fest) ---------------------------------

def test_local_day_bounds_berlin_summer_and_winter():
    # Sommer (CEST +02:00): 21:30 UTC = 23:30 Berlin → Tag begann 22:00Z Vortag.
    now = datetime(2026, 7, 18, 21, 30, tzinfo=timezone.utc)
    start, end = local_day_bounds(now)
    assert start == datetime(2026, 7, 17, 22, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 7, 18, 22, 0, tzinfo=timezone.utc)
    assert start <= now < end
    # Winter (CET +01:00): 23:30 UTC = 00:30 Berlin am FOLGETAG — ein UTC-Tag
    # würde hier mitten am Abend resetten (genau der verbotene Bug).
    now = datetime(2026, 1, 15, 23, 30, tzinfo=timezone.utc)
    start, end = local_day_bounds(now)
    assert start == datetime(2026, 1, 15, 23, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 1, 16, 23, 0, tzinfo=timezone.utc)
    assert start <= now < end


# --- P3: caps in order_due_cards (nach Ordering) -----------------------------

def test_review_cap_keeps_shakiest_n_in_smart_mode():
    now = datetime.now(timezone.utc)
    cards = [_fake_card(i, stability=float(s), days_ago=10, now=now)
             for i, s in enumerate((1, 2, 5, 20, 100), start=1)]
    out = order_due_cards(cards, 'smart', FSRSScheduler(), rng=random.Random(1),
                          now=now, review_budget=2, new_budget=10)
    assert _ids(out) == [1, 2]  # die 2 wackligsten, nicht die 2 ältesten


def test_new_cap_fills_review_headroom():
    now = datetime.now(timezone.utc)
    reviewed = [_fake_card(i, stability=10.0, days_ago=5, now=now) for i in (1, 2)]
    fresh = [_fake_card(i, stability=None, now=now) for i in range(100, 108)]
    out = order_due_cards(reviewed + fresh, 'smart', FSRSScheduler(),
                          rng=random.Random(2), now=now,
                          review_budget=6, new_budget=10)
    fresh_shown = [c for c in out if c.id >= 100]
    assert len(fresh_shown) == 4          # headroom = 6 − 2 Reviews
    assert len(out) == 6                  # total = review_budget


def test_new_respects_exhausted_review_cap():
    now = datetime.now(timezone.utc)
    reviewed = [_fake_card(i, stability=10.0, days_ago=5, now=now) for i in range(1, 6)]
    fresh = [_fake_card(i, stability=None, now=now) for i in (100, 101, 102)]
    out = order_due_cards(reviewed + fresh, 'smart', FSRSScheduler(),
                          rng=random.Random(3), now=now,
                          review_budget=3, new_budget=3)
    assert len(out) == 3
    assert all(c.review.stability is not None for c in out)  # 0 neue


def test_cap_takes_random_subset_on_equal_r():
    # 6 Karten identisches R, Budget 3: "erst shuffeln, dann cappen" =
    # zufällige 3 — verschiedene Seeds ziehen verschiedene Teilmengen.
    now = datetime.now(timezone.utc)
    cards = [_fake_card(i, stability=10.0, days_ago=5, now=now) for i in range(1, 7)]
    out_a = order_due_cards(cards, 'smart', FSRSScheduler(), rng=random.Random(1),
                            now=now, review_budget=3, new_budget=0)
    out_b = order_due_cards(cards, 'smart', FSRSScheduler(), rng=random.Random(2),
                            now=now, review_budget=3, new_budget=0)
    assert len(out_a) == len(out_b) == 3
    assert set(_ids(out_a)) != set(_ids(out_b))


def test_random_mode_caps_pools_too():
    now = datetime.now(timezone.utc)
    reviewed = [_fake_card(i, stability=10.0, days_ago=5, now=now) for i in range(1, 6)]
    fresh = [_fake_card(i, stability=None, now=now) for i in (100, 101)]
    out = order_due_cards(reviewed + fresh, 'random', FSRSScheduler(),
                          rng=random.Random(4), now=now,
                          review_budget=2, new_budget=5)
    assert len(out) == 2
    assert all(c.review.stability is not None for c in out)
    # Null-Budgets → leere Session.
    assert order_due_cards(reviewed + fresh, 'random', FSRSScheduler(),
                           rng=random.Random(5), now=now,
                           review_budget=0, new_budget=0) == []


# --- P3: Tages-Zählung (count_done_today) ------------------------------------

def _make_done_today_card(app, user_id, introduced_today, now=None):
    """Karte, die heute (Berlin) schon bewertet wurde — due in der Zukunft."""
    with app.app_context():
        now = now or datetime.now(timezone.utc)
        first_at = now if introduced_today else now - timedelta(days=30)
        history = [{'rating': 'good', 'reviewed_at': first_at.isoformat()}]
        if not introduced_today:
            history.append({'rating': 'good', 'reviewed_at': now.isoformat()})
        card = Card(user_id=user_id, type='atomic', front='Q', back='A')
        card.review = Review(
            due=now + timedelta(days=3), stability=5.0, difficulty=5.0,
            last_reviewed=now.replace(tzinfo=None),
            reps=1 if introduced_today else 5,
            rating_history=json.dumps(history),
        )
        db.session.add(card)
        db.session.commit()
        return card.id


def test_count_done_today_classifies_new_vs_review(app, test_user):
    uid = test_user['id']
    _make_done_today_card(app, uid, introduced_today=True)
    _make_done_today_card(app, uid, introduced_today=False)
    with app.app_context():
        # Gestern bewertete Karte zählt nicht gegen heutige Budgets.
        start, _ = local_day_bounds()
        card = Card(user_id=uid, type='atomic', front='Q', back='A')
        card.review = Review(
            due=start + timedelta(days=2), stability=3.0,
            last_reviewed=(start - timedelta(hours=2)).replace(tzinfo=None), reps=4,
            rating_history=json.dumps([{'rating': 'good',
                                        'reviewed_at': (start - timedelta(days=9)).isoformat()}]))
        db.session.add(card)
        db.session.commit()
        assert count_done_today(uid) == (1, 1)  # (reviews_done, new_done)


def test_count_done_today_fallback_without_history(app, test_user):
    # Kaputte/fehlende History → reps==1 klassifiziert als heute-neu.
    uid = test_user['id']
    with app.app_context():
        now = datetime.now(timezone.utc)
        for reps, history in ((1, None), (7, '{broken')):
            card = Card(user_id=uid, type='atomic', front='Q', back='A')
            card.review = Review(due=now + timedelta(days=1), stability=2.0,
                                 last_reviewed=now.replace(tzinfo=None),
                                 reps=reps, rating_history=history)
            db.session.add(card)
        db.session.commit()
        assert count_done_today(uid) == (1, 1)


# --- P3: Caps + Tages-Zählung über die HTTP-Grenze ---------------------------

def test_review_state_review_cap_http(app, authenticated_client, test_user):
    uid = test_user['id']
    for _ in range(5):
        _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    authenticated_client.put(SETTINGS_URL, json={'daily_review_limit': 2})
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['due_count'] == 2
    assert body['review_count'] == 2
    assert body['new_count'] == 0


def test_review_state_new_cap_http(app, authenticated_client, test_user):
    uid = test_user['id']
    for _ in range(3):
        _make_card_with_review(app, uid, stability=None)
    authenticated_client.put(SETTINGS_URL, json={'daily_new_limit': 1})
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['due_count'] == 1
    assert body['new_count'] == 1


def test_review_state_new_respects_review_cap_http(app, authenticated_client, test_user):
    uid = test_user['id']
    for _ in range(2):
        _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    for _ in range(3):
        _make_card_with_review(app, uid, stability=None)
    authenticated_client.put(SETTINGS_URL, json={'daily_review_limit': 2})
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['due_count'] == 2      # Review-Cap voll → 0 neue eingestreut
    assert body['new_count'] == 0


def test_review_state_daily_counting_new(app, authenticated_client, test_user):
    # Heute schon 1 neue Karte gelernt → bei Limit 2 kommt nur noch 1 nach.
    uid = test_user['id']
    _make_done_today_card(app, uid, introduced_today=True)
    for _ in range(5):
        _make_card_with_review(app, uid, stability=None)
    authenticated_client.put(SETTINGS_URL, json={'daily_new_limit': 2})
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['new_count'] == 1


def test_review_state_daily_counting_reviews(app, authenticated_client, test_user):
    # Heute schon 2 Reviews gemacht → bei Limit 3 bleibt Budget 1.
    uid = test_user['id']
    _make_done_today_card(app, uid, introduced_today=False)
    _make_done_today_card(app, uid, introduced_today=False)
    for _ in range(5):
        _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    authenticated_client.put(SETTINGS_URL, json={'daily_review_limit': 3})
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['due_count'] == 1
    assert body['review_count'] == 1


def test_settings_put_limit_validation(authenticated_client):
    for bad in (-1, 'zehn', True, 3.5, 10001, None):
        resp = authenticated_client.put(SETTINGS_URL, json={'daily_new_limit': bad})
        assert resp.status_code == 400, f'accepted {bad!r}'
    assert authenticated_client.put(
        SETTINGS_URL, json={'daily_review_limit': 0}).status_code == 200  # 0 = heute nichts


# --- review-state: ordering through the HTTP boundary ------------------------

def test_review_state_smart_orders_shakiest_first(app, authenticated_client, test_user):
    uid = test_user['id']
    # Created in scrambled order; equal elapsed → R ranks by stability.
    c_mid = _make_card_with_review(app, uid, stability=10.0, days_ago=10)
    c_solid = _make_card_with_review(app, uid, stability=100.0, days_ago=10)
    c_shaky = _make_card_with_review(app, uid, stability=1.0, days_ago=10)
    body = authenticated_client.get('/api/review-state').get_json()
    assert [c['id'] for c in body['due_cards']] == [c_shaky, c_mid, c_solid]


def test_review_state_new_card_interleaved_not_first(app, authenticated_client, test_user):
    uid = test_user['id']
    reviewed = [_make_card_with_review(app, uid, stability=10.0, days_ago=5)
                for _ in range(4)]
    fresh = _make_card_with_review(app, uid, stability=None)
    body = authenticated_client.get('/api/review-state').get_json()
    got = [c['id'] for c in body['due_cards']]
    assert sorted(got) == sorted(reviewed + [fresh])
    # 1 new among 4 reviews lands mid-stream (slot 2) — never front-loaded.
    assert got[2] == fresh
    assert body['due_cards'][2]['review']['stability'] is None


def test_review_state_random_mode_returns_same_set(app, authenticated_client, test_user):
    uid = test_user['id']
    ids = [_make_card_with_review(app, uid, stability=float(s), days_ago=10)
           for s in (1, 10, 100)]
    authenticated_client.put(SETTINGS_URL, json={'ordering_mode': 'random'})
    body = authenticated_client.get('/api/review-state').get_json()
    assert sorted(c['id'] for c in body['due_cards']) == sorted(ids)
    assert body['due_count'] == 3


# --- P4: Forecast-Bucketing (Berlin-Tage, Rückstand eigener Bucket) ----------

def _make_review_row(app, user_id, due, stability=5.0, last_reviewed=None,
                     rating_history=None, reps=1):
    with app.app_context():
        card = Card(user_id=user_id, type='atomic', front='Q', back='A')
        card.review = Review(
            due=due.replace(tzinfo=None), stability=stability,
            difficulty=5.0 if stability is not None else None,
            last_reviewed=(last_reviewed.replace(tzinfo=None)
                           if last_reviewed else None),
            reps=reps, rating_history=rating_history)
        db.session.add(card)
        db.session.commit()
        return card.id


def test_forecast_buckets_days_and_backlog(app, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    start, end = local_day_bounds(now)
    _make_review_row(app, uid, due=now - timedelta(days=2))       # Rückstand
    _make_review_row(app, uid, due=now - timedelta(minutes=1))    # Rückstand (due<=now)
    _make_review_row(app, uid, due=end - timedelta(seconds=30))   # Rest heute → Tag 0
    _make_review_row(app, uid, due=end + timedelta(hours=3))      # morgen → Tag 1
    _make_review_row(app, uid, due=start + timedelta(days=5, hours=2))  # Tag 5
    _make_review_row(app, uid, due=start + timedelta(days=60))    # außerhalb Fenster
    with app.app_context():
        fc = forecast_buckets(uid, now=now)
    assert fc['overdue'] == 2
    counts = [d['count'] for d in fc['days']]
    assert counts[0] == 1 and counts[1] == 1 and counts[5] == 1
    assert sum(counts) == 3                      # 60-Tage-Karte nicht im Fenster
    assert len(fc['days']) == 28
    assert fc['days'][0]['date'] == start.astimezone(
        __import__('app_pkg.learn', fromlist=['LOCAL_TZ']).LOCAL_TZ).date().isoformat()


# --- P4: Reifegrad-Klassifikation (21-Tage-Grenze) ---------------------------

def test_maturity_counts_boundaries(app, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    _make_review_row(app, uid, due=now, stability=None, reps=0)   # neu
    _make_review_row(app, uid, due=now + timedelta(days=20),      # jung (<21d)
                     last_reviewed=now)
    _make_review_row(app, uid, due=now + timedelta(days=21),      # reif (==21d)
                     last_reviewed=now)
    _make_review_row(app, uid, due=now + timedelta(days=100),     # reif
                     last_reviewed=now)
    _make_review_row(app, uid, due=now + timedelta(days=50),      # Fallback: stability
                     stability=10.0, last_reviewed=None)          #  → 10d = jung
    with app.app_context():
        assert maturity_counts(uid) == {'neu': 1, 'jung': 2, 'reif': 2}


# --- P4: True-Retention aus rating_history -----------------------------------

def _hist(*entries):
    return json.dumps([{'rating': r, 'reviewed_at': at.isoformat()}
                       for r, at in entries])


def test_true_retention_first_per_day_intro_excluded(app, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    # Karte A: Intro vor 10d (zählt NICHT), vor 5d gewusst, vor 3d erst
    # vergessen + später am selben Tag gewusst (erste Bewertung zählt = fail).
    _make_review_row(app, uid, due=now + timedelta(days=9), last_reviewed=now,
                     reps=4, rating_history=_hist(
                         ('good', now - timedelta(days=10)),
                         ('good', now - timedelta(days=5)),
                         ('again', now - timedelta(days=3)),
                         ('good', now - timedelta(days=3, hours=-2))))
    # Karte B: Review vor 40d liegt außerhalb des 30-Tage-Fensters.
    _make_review_row(app, uid, due=now + timedelta(days=9), last_reviewed=now,
                     reps=2, rating_history=_hist(
                         ('good', now - timedelta(days=60)),
                         ('good', now - timedelta(days=40))))
    # Karte C: nur Intro → trägt nichts bei.
    _make_review_row(app, uid, due=now + timedelta(days=2), last_reviewed=now,
                     reps=1, rating_history=_hist(('good', now - timedelta(days=1))))
    with app.app_context():
        result = true_retention(uid, now=now)
    assert result == {'pass': 1, 'fail': 1, 'rate': 0.5}


def test_true_retention_empty(app, test_user):
    with app.app_context():
        assert true_retention(test_user['id']) == {'pass': 0, 'fail': 0, 'rate': None}


# --- P4: Stats-Endpoint — "Heute" == Launcher (der Konsistenz-Wächter) -------

def test_stats_today_equals_review_state_counts(app, authenticated_client, test_user):
    uid = test_user['id']
    # Gemischte Lage: 5 fällige Reviews, 4 fällige Neue, heute schon 1 Review
    # + 1 Neue gelernt, enge Limits → die Cap-Arithmetik muss identisch greifen.
    for _ in range(5):
        _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    for _ in range(4):
        _make_card_with_review(app, uid, stability=None)
    _make_done_today_card(app, uid, introduced_today=True)
    _make_done_today_card(app, uid, introduced_today=False)
    authenticated_client.put(SETTINGS_URL, json={'daily_review_limit': 8,
                                                 'daily_new_limit': 3})
    stats = authenticated_client.get('/api/learn/stats').get_json()
    state = authenticated_client.get('/api/review-state').get_json()
    assert stats['today']['reviews_due'] == state['review_count']
    assert stats['today']['new_available'] == state['new_count']
    assert stats['today']['reviews_done'] == 1
    assert stats['today']['new_done'] == 1
    # Und die Zahlen selbst: Budget Reviews 8-1=7 → 5 gezeigt; Headroom 7-5=2
    # → Neue min(4, min(3-1, 2)) = 2.
    assert stats['today']['reviews_due'] == 5
    assert stats['today']['new_available'] == 2


def test_stats_shape(app, authenticated_client, test_user):
    body = authenticated_client.get('/api/learn/stats').get_json()
    assert set(body) == {'today', 'forecast', 'maturity', 'retention'}
    assert body['retention']['desired'] == 0.9      # env-Default, read-only
    assert body['retention']['window_days'] == 30


def test_stats_and_simulate_require_login(client):
    assert client.get('/api/learn/stats').status_code in (302, 401)
    assert client.get('/api/learn/simulate?retention=0.9').status_code in (302, 401)


# --- P4: Workload-Simulator --------------------------------------------------

def test_simulate_workload_monotonic_and_deterministic():
    lo = simulate_workload(0.85, 10)
    mid = simulate_workload(0.90, 10)
    hi = simulate_workload(0.95, 10)
    assert 0 < lo < mid < hi          # höhere Retention → kürzere Intervalle → mehr Reviews
    assert simulate_workload(0.90, 10) == mid   # deterministisch, kein RNG
    assert simulate_workload(0.90, 0) == 0.0
    assert simulate_workload(0.90, 20) == 2 * mid  # linear in new_per_day


# --- LEARN-TUNE: die Projektion deckelt mit demselben Wert wie der Scheduler --

def test_simulate_workload_honours_maximum_interval():
    uncapped = simulate_workload(0.90, 10, maximum_interval=DEFAULT_MAXIMUM_INTERVAL)
    capped = simulate_workload(0.90, 10, maximum_interval=60)
    # Richtung: der Deckel HEBT die Last — kürzere Intervalle = mehr Reviews
    # pro Karte und Jahr. Das ist der Preis des Deckels, kein Fehler.
    assert capped > uncapped > 0
    assert simulate_workload(0.90, 10, maximum_interval=21) > capped


def test_simulate_workload_reads_the_same_env_key_as_the_scheduler(monkeypatch):
    monkeypatch.delenv('FSRS_MAXIMUM_INTERVAL', raising=False)
    assert (simulate_workload(0.90, 10)
            == simulate_workload(0.90, 10, maximum_interval=DEFAULT_MAXIMUM_INTERVAL))
    monkeypatch.setenv('FSRS_MAXIMUM_INTERVAL', '60')
    # Invariante: Simulation und echter Scheduler deckeln bei demselben Wert.
    assert get_scheduler()._engine.maximum_interval == 60
    assert (simulate_workload(0.90, 10)
            == simulate_workload(0.90, 10, maximum_interval=60))
    monkeypatch.setenv('FSRS_MAXIMUM_INTERVAL', 'abc')          # Müll → Default
    assert (simulate_workload(0.90, 10)
            == simulate_workload(0.90, 10, maximum_interval=DEFAULT_MAXIMUM_INTERVAL))


def test_simulate_endpoint_and_validation(app, authenticated_client, test_user):
    body = authenticated_client.get(
        '/api/learn/simulate?retention=0.90&new_per_day=10').get_json()
    assert body['estimate'] is True
    assert body['reviews_per_day'] > 0
    # Default new_per_day = daily_new_limit aus den Settings.
    authenticated_client.put(SETTINGS_URL, json={'daily_new_limit': 0})
    assert authenticated_client.get(
        '/api/learn/simulate?retention=0.90').get_json()['reviews_per_day'] == 0.0
    for url in ('/api/learn/simulate',                          # retention fehlt
                '/api/learn/simulate?retention=abc',
                '/api/learn/simulate?retention=0.3',            # out of range
                '/api/learn/simulate?retention=0.9&new_per_day=-1',
                '/api/learn/simulate?retention=0.9&new_per_day=abc'):
        assert authenticated_client.get(url).status_code == 400, url


# --- LEARN-MORE: local_day_end + uncapped/ahead + die drei neuen Felder ------

def test_local_day_end_berlin_summer_winter_and_dst_edge():
    # Sommer (CEST +02:00): Tagesende 22:00Z; days_ahead schiebt Berliner Tage.
    now = datetime(2026, 7, 18, 21, 30, tzinfo=timezone.utc)
    assert local_day_end(now) == datetime(2026, 7, 18, 22, 0, tzinfo=timezone.utc)
    assert local_day_end(now, 1) == datetime(2026, 7, 19, 22, 0, tzinfo=timezone.utc)
    assert local_day_end(now) == local_day_bounds(now)[1]   # 0 = heutiges Ende
    # Winter (CET +01:00): 23:30Z liegt schon im Berliner FOLGETAG.
    now = datetime(2026, 1, 15, 23, 30, tzinfo=timezone.utc)
    assert local_day_end(now) == datetime(2026, 1, 16, 23, 0, tzinfo=timezone.utc)
    # DST-Kante: Sommerzeit endet 2026-10-25 → das Ende des 25.10. liegt in
    # CET; der Schritt über die Kante ist 25 Wanduhr-Stunden, nie naive +24h.
    now = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    assert local_day_end(now) == datetime(2026, 10, 24, 22, 0, tzinfo=timezone.utc)
    assert local_day_end(now, 1) == datetime(2026, 10, 25, 23, 0, tzinfo=timezone.utc)


def test_review_state_capped_reports_remaining_today(app, authenticated_client, test_user):
    uid = test_user['id']
    for _ in range(5):
        _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    authenticated_client.put(SETTINGS_URL, json={'daily_review_limit': 2})
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['due_count'] == 2
    assert body['remaining_today'] == 3
    assert body['next_ahead'] is None       # nichts Künftiges angelegt


def test_review_state_uncapped_lifts_caps(app, authenticated_client, test_user):
    uid = test_user['id']
    for _ in range(5):
        _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    authenticated_client.put(SETTINGS_URL, json={'daily_review_limit': 2})
    body = authenticated_client.get('/api/review-state?uncapped=1').get_json()
    assert body['due_count'] == 5
    assert body['remaining_today'] == 0


def test_review_state_uncapped_strict_read(app, authenticated_client, test_user):
    # Nur der explizite Wert '1' schaltet — kein Truthiness-Zufall.
    uid = test_user['id']
    for _ in range(3):
        _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    authenticated_client.put(SETTINGS_URL, json={'daily_review_limit': 1})
    for arg in ('0', 'true', 'yes', ''):
        body = authenticated_client.get(f'/api/review-state?uncapped={arg}').get_json()
        assert body['due_count'] == 1, arg


def test_review_state_uncapped_never_resurfaces_done_today(app, authenticated_client, test_user):
    # DIE Kern-Eigenschaft hinter "mehr lernen": heute Erledigtes trägt ein
    # zukünftiges due und kommt auch ohne Cap nie zurück — Uncapping ist per
    # Konstruktion wiederholungsfrei.
    uid = test_user['id']
    done = _make_done_today_card(app, uid, introduced_today=False)   # due now+3d
    open_id = _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    body = authenticated_client.get('/api/review-state?uncapped=1').get_json()
    ids = [c['id'] for c in body['due_cards']]
    assert open_id in ids
    assert done not in ids


def test_review_state_ahead_pulls_tomorrow(app, authenticated_client, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    today_id = _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    tomorrow_id = _make_review_row(
        app, uid, due=local_day_end(now, 1) - timedelta(hours=1),
        last_reviewed=now - timedelta(days=5))
    base = authenticated_client.get('/api/review-state').get_json()
    assert tomorrow_id not in [c['id'] for c in base['due_cards']]
    assert base['next_ahead'] == {'days': 1, 'count': 1}
    body = authenticated_client.get('/api/review-state?ahead=1').get_json()
    ids = [c['id'] for c in body['due_cards']]
    assert today_id in ids and tomorrow_id in ids


def test_review_state_ahead_borrows_day_by_day(app, authenticated_client, test_user):
    # Stufe 2 borgt tageweise: ahead=1 holt morgen, erst ahead=2 auch übermorgen.
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    tomorrow_id = _make_review_row(
        app, uid, due=local_day_end(now, 1) - timedelta(hours=1),
        last_reviewed=now - timedelta(days=5))
    day2_id = _make_review_row(
        app, uid, due=local_day_end(now, 2) - timedelta(hours=1),
        last_reviewed=now - timedelta(days=5))
    one = authenticated_client.get('/api/review-state?ahead=1').get_json()
    one_ids = [c['id'] for c in one['due_cards']]
    assert tomorrow_id in one_ids and day2_id not in one_ids
    assert one['next_ahead'] == {'days': 2, 'count': 1}   # der nächste Schritt
    two = authenticated_client.get('/api/review-state?ahead=2').get_json()
    two_ids = [c['id'] for c in two['due_cards']]
    assert tomorrow_id in two_ids and day2_id in two_ids
    assert two['next_ahead'] is None        # dahinter liegt nichts mehr


def test_review_state_ahead_implies_uncapped(app, authenticated_client, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    for _ in range(4):
        _make_card_with_review(app, uid, stability=10.0, days_ago=5)
    _make_review_row(app, uid, due=local_day_end(now, 1) - timedelta(hours=1),
                     last_reviewed=now - timedelta(days=5))
    authenticated_client.put(SETTINGS_URL, json={'daily_review_limit': 2})
    body = authenticated_client.get('/api/review-state?ahead=1').get_json()
    assert body['due_count'] == 5           # 4 heute + 1 morgen, Cap aufgehoben
    assert body['remaining_today'] == 0


def test_review_state_ahead_validation(authenticated_client):
    for bad in ('0', '8', 'abc', '1.5', '-1'):
        resp = authenticated_client.get(f'/api/review-state?ahead={bad}')
        assert resp.status_code == 400, bad
        assert 'ahead' in resp.get_json()['error']


def test_review_state_next_ahead_counts_only_through_target_day(app, authenticated_client, test_user):
    # count reicht nur bis ans Ende des Zieltags — die Tag-3-Karte zählt nicht.
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    _make_review_row(app, uid, due=local_day_end(now, 1) - timedelta(hours=1),
                     last_reviewed=now - timedelta(days=5))
    _make_review_row(app, uid, due=local_day_end(now, 3) - timedelta(hours=1),
                     last_reviewed=now - timedelta(days=5))
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['next_ahead'] == {'days': 1, 'count': 1}


def test_review_state_next_ahead_skips_empty_days(app, authenticated_client, test_user):
    # Lückentag: morgen leer, übermorgen voll → Stufe 2 springt direkt auf 2
    # (mit der fixen Ein-Tag-Definition verschwände sie hier komplett).
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    _make_review_row(app, uid, due=local_day_end(now, 2) - timedelta(hours=1),
                     last_reviewed=now - timedelta(days=5))
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['next_ahead'] == {'days': 2, 'count': 1}


def test_review_state_next_ahead_null_beyond_reach(app, authenticated_client, test_user):
    # Jenseits von Tag MAX (7) liegt etwas — aber außer Reichweite → null.
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    _make_review_row(app, uid, due=local_day_end(now, 7) + timedelta(hours=5),
                     last_reviewed=now - timedelta(days=5))
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['next_ahead'] is None


def test_review_state_next_ahead_null_at_max_step(app, authenticated_client, test_user):
    # Am Maximum gibt es keinen weiteren Schritt (die API würde ahead=8
    # 400en) — selbst wenn Tag 8 Karten hätte: null.
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    _make_review_row(app, uid, due=local_day_end(now, 7) + timedelta(hours=5),
                     last_reviewed=now - timedelta(days=5))
    body = authenticated_client.get('/api/review-state?ahead=7').get_json()
    assert body['next_ahead'] is None


def test_review_state_next_ahead_rest_of_today_is_step_one(app, authenticated_client, test_user):
    # Eine Karte, die HEUTE noch fällig wird (nach jetzt), holt der kleinste
    # existierende Schritt (ahead=1) — days ist nie 0.
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    _make_review_row(app, uid, due=now + timedelta(minutes=1),
                     last_reviewed=now - timedelta(days=5))
    body = authenticated_client.get('/api/review-state').get_json()
    assert body['next_ahead'] == {'days': 1, 'count': 1}


def test_review_state_day_end_is_aware_and_ahead_of_now(app, authenticated_client, test_user):
    body = authenticated_client.get('/api/review-state').get_json()
    day_end = datetime.fromisoformat(body['day_end'])
    assert day_end.tzinfo is not None
    assert day_end > datetime.now(timezone.utc)
    # Es IST das Berliner Tagesende, nicht bloß irgendein Zukunftswert.
    assert day_end == local_day_bounds()[1]
