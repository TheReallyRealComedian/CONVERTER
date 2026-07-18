"""LEARN-UP Phase 1 — learn settings (one JSON blob) + review-queue ordering.

Pure-logic tests drive ``order_due_cards`` with fake cards and a seeded RNG
(deterministic shuffles); HTTP tests cover the settings roundtrip and the
review-state ordering through the public boundary. The date-order regression
tests are the point of the phase: equal-R cards must NOT come back in
creation order, and brand-new cards must be interleaved, not front-loaded.
"""
import json
import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app_pkg.learn import LEARN_SETTINGS_DEFAULTS, get_user_settings, order_due_cards
from models import Card, Review, User, db
from services.scheduler import FSRSScheduler, SM2Scheduler

SETTINGS_URL = '/api/learn/settings'


# --- helpers -----------------------------------------------------------------

def _fake_card(cid, stability=None, days_ago=None, due=None, now=None):
    now = now or datetime.now(timezone.utc)
    review = SimpleNamespace(
        due=due or now,
        stability=stability,
        difficulty=5.0 if stability is not None else None,
        last_reviewed=(now - timedelta(days=days_ago)) if days_ago is not None else None,
    )
    return SimpleNamespace(id=cid, review=review)


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


def test_smart_all_new_is_shuffled():
    now = datetime.now(timezone.utc)
    cards = [_fake_card(i, stability=None, now=now) for i in range(1, 8)]
    out = order_due_cards(cards, 'smart', FSRSScheduler(),
                          rng=random.Random(5), now=now)
    assert sorted(_ids(out)) == list(range(1, 8))
    assert _ids(out) != list(range(1, 8))


def test_smart_sm2_engine_degrades_to_due_ascending():
    # SM-2 has no R (base-class None) → reviewed cards fall back to due-asc.
    now = datetime.now(timezone.utc)
    oldest = _fake_card(1, stability=6.0, days_ago=9, due=now - timedelta(days=3), now=now)
    middle = _fake_card(2, stability=6.0, days_ago=9, due=now - timedelta(days=2), now=now)
    newest = _fake_card(3, stability=6.0, days_ago=9, due=now - timedelta(days=1), now=now)
    out = order_due_cards([newest, oldest, middle], 'smart', SM2Scheduler(),
                          rng=random.Random(4), now=now)
    assert _ids(out) == [1, 2, 3]


# --- order_due_cards: random mode --------------------------------------------

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


def test_settings_get_default_smart(authenticated_client):
    body = authenticated_client.get(SETTINGS_URL).get_json()
    assert body == {'ordering_mode': 'smart'}


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
