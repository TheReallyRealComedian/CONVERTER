"""LERN-GROUP-AW Phase 2 — Token-Endpoint: der Agent baut den Tag-Baum.

`POST /api/tags/parent` (token, by-name) setzt/löst `parent_id` zwischen zwei
Tags (beide get_or_create → lowercased shared vocabulary), mit Zyklus-Guard
gespiegelt aus der Session-Logik. Distinct vom Session-`PATCH /api/tags/<id>`.
Der gesetzte Baum wirkt im `?tag=`-Review-Filter (Teilbaum).
"""
from datetime import datetime, timedelta, timezone

from models import Card, Review, Tag, User, db


CARD_TOKEN = 'r4-test-card-token-9b2e'
PARENT_URL = '/api/tags/parent'


def _card_auth(token=CARD_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _get_tag(app, uid, name):
    with app.app_context():
        return Tag.query.filter_by(user_id=uid, name=name).first()


# --- set / unparent by-name --------------------------------------------------

def test_set_parent_by_name_creates_both_tags(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    resp = client.post(PARENT_URL, headers=_card_auth(),
                       json={'tag': 'Transformer', 'parent': 'KI'})
    assert resp.status_code == 200
    body = resp.get_json()
    # both tags get_or_create'd + lowercased (shared vocabulary)
    assert body['name'] == 'transformer'
    with app.app_context():
        parent = Tag.query.filter_by(user_id=uid, name='ki').first()
        child = Tag.query.filter_by(user_id=uid, name='transformer').first()
        assert parent is not None and child is not None
        assert child.parent_id == parent.id
        assert body['parent_id'] == parent.id


def test_set_parent_reuses_existing_tags(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        Tag.get_or_create(uid, 'ki')
        Tag.get_or_create(uid, 'transformer')
        db.session.commit()
    client.post(PARENT_URL, headers=_card_auth(),
                json={'tag': 'Transformer', 'parent': 'KI'})
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='ki').count() == 1
        assert Tag.query.filter_by(user_id=uid, name='transformer').count() == 1


def test_unparent_with_null(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    client.post(PARENT_URL, headers=_card_auth(),
                json={'tag': 'transformer', 'parent': 'ki'})
    resp = client.post(PARENT_URL, headers=_card_auth(),
                       json={'tag': 'transformer', 'parent': None})
    assert resp.status_code == 200
    assert resp.get_json()['parent_id'] is None
    with app.app_context():
        child = Tag.query.filter_by(user_id=uid, name='transformer').first()
        assert child.parent_id is None


# --- cycle guard -------------------------------------------------------------

def test_cycle_guard_rejects_parent_in_subtree(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    # a → b (b child of a). Now try a's parent = b → b is in a's subtree → 400.
    client.post(PARENT_URL, headers=_card_auth(), json={'tag': 'b', 'parent': 'a'})
    resp = client.post(PARENT_URL, headers=_card_auth(), json={'tag': 'a', 'parent': 'b'})
    assert resp.status_code == 400


def test_self_reference_rejected(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    # tag == parent after normalisation → same row → self in own subtree → 400.
    resp = client.post(PARENT_URL, headers=_card_auth(), json={'tag': 'KI', 'parent': 'ki'})
    assert resp.status_code == 400


# --- bad body ----------------------------------------------------------------

def test_blank_tag_name_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(PARENT_URL, headers=_card_auth(),
                       json={'tag': '   ', 'parent': 'ki'}).status_code == 400
    assert client.post(PARENT_URL, headers=_card_auth(),
                       json={'tag': 'ok', 'parent': '   '}).status_code == 400


# --- auth + owner scope ------------------------------------------------------

def test_requires_token(app, client, test_user, monkeypatch):
    monkeypatch.delenv('CARD_TOKEN', raising=False)
    assert client.post(PARENT_URL, headers=_card_auth(),
                       json={'tag': 'a', 'parent': 'b'}).status_code == 503
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(PARENT_URL, headers=_card_auth('wrong'),
                       json={'tag': 'a', 'parent': 'b'}).status_code == 401


def test_tags_owned_by_target_user(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        other = User(username='mallory')
        other.set_password('password1234')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    client.post(PARENT_URL, headers=_card_auth(), json={'tag': 'kind', 'parent': 'eltern'})
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid).count() == 2
        assert Tag.query.filter_by(user_id=other_id).count() == 0


def test_csrf_exempt_under_enforced_csrf(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)
    # Bearer-only POST has no CSRF token → would be 400 if not exempt.
    assert client.post(PARENT_URL, headers=_card_auth(),
                       json={'tag': 'a', 'parent': 'b'}).status_code == 200


# --- the built tree drives the ?tag= review subtree filter -------------------

def test_built_tree_drives_review_subtree_filter(app, client, authenticated_client,
                                                  test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    # Agent builds: 'transformer' under 'ki'.
    client.post(PARENT_URL, headers=_card_auth(),
                json={'tag': 'transformer', 'parent': 'ki'})
    now = datetime.now(timezone.utc)
    with app.app_context():
        child = Tag.query.filter_by(user_id=uid, name='transformer').first()
        parent = Tag.query.filter_by(user_id=uid, name='ki').first()
        card = Card(user_id=uid, type='atomic', front='Q', back='A')
        card.tags.append(child)
        card.review = Review(due=now - timedelta(days=1))
        db.session.add(card)
        db.session.commit()
        parent_id = parent.id
    # Filtering by the PARENT tag must surface the child-tagged card (subtree).
    body = authenticated_client.get(f'/api/review-state?tag={parent_id}').get_json()
    assert {c['id'] for c in body['due_cards']}
    assert body['total_count'] == 1
