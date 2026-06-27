"""LERN-GROUP-AW Phase 1 — Sammlungen über die Card-Write-API.

Der Agent legt Karten in Sammlungen (Achse B) über `collections: [namen]` bei
create_card / update_card an: get_or_create **by-name, case-erhaltend** (anders
als Tags, die lowercasen), owner-scoped, Voll-Ersetzung. Token-Gate +
CSRF-Exempt wie die übrigen Card-Writes.
"""
from models import Card, Collection, User, db


CARD_TOKEN = 'r4-test-card-token-9b2e'
CARDS_URL = '/api/cards'


def _card_auth(token=CARD_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _atomic_payload(**ov):
    p = {'type': 'atomic', 'front': 'Was ist X?', 'back': 'X ist Y.'}
    p.update(ov)
    return p


# --- normalize_name / get_or_create model unit ------------------------------

def test_collection_normalize_preserves_case_collapses_whitespace(app):
    assert Collection.normalize_name('  Boehringer   Pipeline  ') == 'Boehringer Pipeline'
    assert Collection.normalize_name('Transformer') == 'Transformer'  # case kept
    assert Collection.normalize_name('   ') == ''
    assert Collection.normalize_name(None) == ''


def test_collection_get_or_create_reuses_and_rejects(app, test_user):
    uid = test_user['id']
    with app.app_context():
        a = Collection.get_or_create(uid, ' Boehringer-Pipeline ')
        db.session.commit()
        b = Collection.get_or_create(uid, 'Boehringer-Pipeline')
        assert a.id == b.id                      # same row, whitespace-trimmed
        assert a.name == 'Boehringer-Pipeline'   # case preserved
        # case difference IS a distinct collection (proper names, sprawl ok)
        c = Collection.get_or_create(uid, 'boehringer-pipeline')
        assert c.id != a.id
        # rejects
        assert Collection.get_or_create(uid, '   ') is None
        assert Collection.get_or_create(uid, 123) is None
        assert Collection.get_or_create(uid, 'x' * 121) is None


# --- create_card with collections -------------------------------------------

def test_create_card_with_collections_get_or_create(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    resp = client.post(CARDS_URL, headers=_card_auth(),
                       json=_atomic_payload(collections=['Horizont A', 'Kurs B']))
    assert resp.status_code == 201
    body = resp.get_json()
    assert sorted(c['name'] for c in body['collections']) == ['Horizont A', 'Kurs B']
    with app.app_context():
        card = Card.query.filter_by(user_id=uid).one()
        assert sorted(c.name for c in card.collections) == ['Horizont A', 'Kurs B']
        assert Collection.query.filter_by(user_id=uid).count() == 2


def test_create_card_reuses_existing_collection(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        existing = Collection.get_or_create(uid, 'Horizont A')
        db.session.commit()
        existing_id = existing.id
    resp = client.post(CARDS_URL, headers=_card_auth(),
                       json=_atomic_payload(collections=[' Horizont A ', 'Neu']))
    assert resp.status_code == 201
    with app.app_context():
        assert Collection.query.filter_by(user_id=uid, name='Horizont A').count() == 1
        card = Card.query.filter_by(user_id=uid).one()
        ha = next(c for c in card.collections if c.name == 'Horizont A')
        assert ha.id == existing_id


def test_create_card_without_collections_is_empty(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    resp = client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload())
    assert resp.status_code == 201
    assert resp.get_json()['collections'] == []


# --- update_card with collections -------------------------------------------

def test_patch_card_replaces_collections(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = client.post(CARDS_URL, headers=_card_auth(),
                      json=_atomic_payload(collections=['Alt'])).get_json()['id']
    resp = client.patch(f'/api/cards/{cid}', headers=_card_auth(),
                        json={'collections': ['Neu1', 'Neu2']})
    assert resp.status_code == 200
    assert sorted(c['name'] for c in resp.get_json()['collections']) == ['Neu1', 'Neu2']


def test_patch_card_empty_list_clears_collections(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = client.post(CARDS_URL, headers=_card_auth(),
                      json=_atomic_payload(collections=['Alt'])).get_json()['id']
    resp = client.patch(f'/api/cards/{cid}', headers=_card_auth(), json={'collections': []})
    assert resp.status_code == 200
    assert resp.get_json()['collections'] == []


def test_patch_card_non_list_collections_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload()).get_json()['id']
    resp = client.patch(f'/api/cards/{cid}', headers=_card_auth(),
                        json={'collections': 'Horizont A'})
    assert resp.status_code == 400


def test_patch_card_omitting_collections_keeps_them(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = client.post(CARDS_URL, headers=_card_auth(),
                      json=_atomic_payload(collections=['Bleibt'])).get_json()['id']
    # PATCH without a 'collections' key must not touch the set.
    resp = client.patch(f'/api/cards/{cid}', headers=_card_auth(), json={'state': 'wackelt'})
    assert [c['name'] for c in resp.get_json()['collections']] == ['Bleibt']


# --- owner scope + auth ------------------------------------------------------

def test_collections_created_at_target_user(app, client, test_user, monkeypatch):
    # The collection is owned by the token-target user (INGEST_USER hook), not
    # leaked across users.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        other = User(username='mallory')
        other.set_password('password1234')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    client.post(CARDS_URL, headers=_card_auth(),
                json=_atomic_payload(collections=['Mein Horizont']))
    with app.app_context():
        assert Collection.query.filter_by(user_id=uid, name='Mein Horizont').count() == 1
        assert Collection.query.filter_by(user_id=other_id).count() == 0


def test_create_card_collections_requires_token(app, client, test_user, monkeypatch):
    monkeypatch.delenv('CARD_TOKEN', raising=False)
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json=_atomic_payload(collections=['X'])).status_code == 503
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(CARDS_URL, headers=_card_auth('wrong'),
                       json=_atomic_payload(collections=['X'])).status_code == 401
