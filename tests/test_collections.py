"""LERN-GROUP Phase 2 — Sammlungen-Backend (Achse B).

Lockt fest: Collection-CRUD (owner-scoped, name non-blank + Duplikat-409),
Karte add/remove (idempotent, owner-404), die Lösch-Mechanik in BEIDE
Richtungen (Card-Delete und Collection-Delete sweepen card_collections — ohne
PRAGMA foreign_keys ORM-getrieben) und der ``?collection=``-Filter inkl.
``?tag=``+``?collection=``-AND auf ``/api/review-state``.
"""
from datetime import datetime, timedelta, timezone

from models import Card, Collection, Review, Tag, User, card_collections, db


# --- helpers -----------------------------------------------------------------

def _make_other_user(app, username='mallory'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def _make_card(app, user_id, tag_ids=(), due=None):
    with app.app_context():
        card = Card(user_id=user_id, type='atomic', front='Q', back='A')
        card.review = Review(due=due or datetime.now(timezone.utc))
        for tid in tag_ids:
            card.tags.append(Tag.query.get(tid))
        db.session.add(card)
        db.session.commit()
        return card.id


def _make_tag(app, user_id, name, parent_id=None):
    with app.app_context():
        t = Tag(user_id=user_id, name=name, parent_id=parent_id)
        db.session.add(t)
        db.session.commit()
        return t.id


def _make_collection(app, user_id, name='Kurs'):
    with app.app_context():
        col = Collection(user_id=user_id, name=name)
        db.session.add(col)
        db.session.commit()
        return col.id


# --- CRUD --------------------------------------------------------------------

def test_list_collections_empty(app, authenticated_client, test_user):
    r = authenticated_client.get('/api/collections')
    assert r.status_code == 200
    assert r.get_json() == []


def test_create_collection(app, authenticated_client, test_user):
    r = authenticated_client.post('/api/collections', json={'name': '  Transformer  '})
    assert r.status_code == 201
    body = r.get_json()
    assert body['name'] == 'Transformer'  # trimmed
    assert body['card_count'] == 0
    with app.app_context():
        assert Collection.query.filter_by(name='Transformer').count() == 1


def test_create_collection_blank_400(app, authenticated_client, test_user):
    assert authenticated_client.post('/api/collections', json={'name': ''}).status_code == 400
    assert authenticated_client.post('/api/collections', json={'name': '   '}).status_code == 400
    assert authenticated_client.post('/api/collections', json={'name': None}).status_code == 400
    assert authenticated_client.post('/api/collections', json={}).status_code == 400


def test_create_collection_duplicate_409(app, authenticated_client, test_user):
    assert authenticated_client.post('/api/collections', json={'name': 'Kurs'}).status_code == 201
    r = authenticated_client.post('/api/collections', json={'name': 'Kurs'})
    assert r.status_code == 409
    # trim macht 'Kurs ' zum Duplikat
    assert authenticated_client.post('/api/collections', json={'name': ' Kurs '}).status_code == 409


def test_list_collections_with_card_count(app, authenticated_client, test_user):
    uid = test_user['id']
    colid = _make_collection(app, uid, 'Mit Karten')
    c1 = _make_card(app, uid)
    c2 = _make_card(app, uid)
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': c1})
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': c2})
    _make_collection(app, uid, 'Leer')

    data = authenticated_client.get('/api/collections').get_json()
    by_name = {c['name']: c for c in data}
    assert by_name['Mit Karten']['card_count'] == 2
    assert by_name['Leer']['card_count'] == 0
    # alphabetisch sortiert
    assert [c['name'] for c in data] == ['Leer', 'Mit Karten']


def test_rename_collection(app, authenticated_client, test_user):
    uid = test_user['id']
    colid = _make_collection(app, uid, 'Alt')
    r = authenticated_client.patch(f'/api/collections/{colid}', json={'name': 'Neu'})
    assert r.status_code == 200
    assert r.get_json()['name'] == 'Neu'
    with app.app_context():
        assert Collection.query.get(colid).name == 'Neu'


def test_rename_collection_duplicate_409(app, authenticated_client, test_user):
    uid = test_user['id']
    _make_collection(app, uid, 'A')
    colid_b = _make_collection(app, uid, 'B')
    r = authenticated_client.patch(f'/api/collections/{colid_b}', json={'name': 'A'})
    assert r.status_code == 409


def test_collection_crud_owner_scoped_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app)
    foreign = _make_collection(app, other_id, 'Foreign')
    assert authenticated_client.patch(f'/api/collections/{foreign}', json={'name': 'X'}).status_code == 404
    assert authenticated_client.delete(f'/api/collections/{foreign}').status_code == 404
    assert authenticated_client.post(f'/api/collections/{foreign}/cards', json={'card_id': 1}).status_code == 404
    with app.app_context():
        assert Collection.query.get(foreign) is not None


def test_collections_require_login(app, client, test_user):
    assert client.get('/api/collections').status_code in (302, 401)
    assert client.post('/api/collections', json={'name': 'X'}).status_code in (302, 401)


# --- Karte add/remove --------------------------------------------------------

def test_add_card_idempotent(app, authenticated_client, test_user):
    uid = test_user['id']
    colid = _make_collection(app, uid)
    cid = _make_card(app, uid)
    r1 = authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': cid})
    assert r1.status_code == 200
    r2 = authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': cid})
    assert r2.status_code == 200
    with app.app_context():
        rows = db.session.execute(
            card_collections.select().where(card_collections.c.collection_id == colid)
        ).fetchall()
        assert len(rows) == 1


def test_add_foreign_card_404(app, authenticated_client, test_user):
    uid = test_user['id']
    colid = _make_collection(app, uid)
    other_id = _make_other_user(app)
    foreign_card = _make_card(app, other_id)
    r = authenticated_client.post(f'/api/collections/{colid}/cards',
                                  json={'card_id': foreign_card})
    assert r.status_code == 404


def test_add_card_bad_body_400(app, authenticated_client, test_user):
    uid = test_user['id']
    colid = _make_collection(app, uid)
    assert authenticated_client.post(f'/api/collections/{colid}/cards', json={}).status_code == 400
    assert authenticated_client.post(f'/api/collections/{colid}/cards',
                                     json={'card_id': 'abc'}).status_code == 400


def test_remove_card(app, authenticated_client, test_user):
    uid = test_user['id']
    colid = _make_collection(app, uid)
    cid = _make_card(app, uid)
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': cid})
    r = authenticated_client.delete(f'/api/collections/{colid}/cards/{cid}')
    assert r.status_code == 200
    with app.app_context():
        rows = db.session.execute(
            card_collections.select().where(card_collections.c.collection_id == colid)
        ).fetchall()
        assert rows == []
    # idempotent: erneutes Entfernen ist no-op 200
    assert authenticated_client.delete(f'/api/collections/{colid}/cards/{cid}').status_code == 200


def test_remove_foreign_card_404(app, authenticated_client, test_user):
    uid = test_user['id']
    colid = _make_collection(app, uid)
    other_id = _make_other_user(app)
    foreign_card = _make_card(app, other_id)
    r = authenticated_client.delete(f'/api/collections/{colid}/cards/{foreign_card}')
    assert r.status_code == 404


# --- Lösch-Mechanik (beide Richtungen) ---------------------------------------

def test_card_delete_sweeps_card_collections(app, authenticated_client, test_user):
    # Owning-side Delete: Card weg → card_collections-Zeile weg, Collection bleibt.
    uid = test_user['id']
    colid = _make_collection(app, uid)
    cid = _make_card(app, uid)
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': cid})
    with app.app_context():
        assert db.session.execute(
            card_collections.select().where(card_collections.c.card_id == cid)
        ).fetchall()  # sanity: row exists

    assert authenticated_client.delete(f'/api/cards/{cid}').status_code == 200
    with app.app_context():
        assert db.session.get(Card, cid) is None
        assert db.session.execute(
            card_collections.select().where(card_collections.c.card_id == cid)
        ).fetchall() == []
        assert db.session.get(Collection, colid) is not None  # Collection survives


def test_collection_delete_sweeps_card_collections(app, authenticated_client, test_user):
    # Backref-side Delete: Collection weg → card_collections-Zeile weg, Card bleibt.
    uid = test_user['id']
    colid = _make_collection(app, uid)
    cid = _make_card(app, uid)
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': cid})

    assert authenticated_client.delete(f'/api/collections/{colid}').status_code == 200
    with app.app_context():
        assert db.session.get(Collection, colid) is None
        assert db.session.execute(
            card_collections.select().where(card_collections.c.collection_id == colid)
        ).fetchall() == []
        assert db.session.get(Card, cid) is not None  # Card survives


# --- /api/review-state?collection= -------------------------------------------

def test_review_state_collection_filter(app, authenticated_client, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    colid = _make_collection(app, uid)
    c_in_due = _make_card(app, uid, due=now - timedelta(days=1))
    c_in_future = _make_card(app, uid, due=now + timedelta(days=1))
    c_out = _make_card(app, uid, due=now - timedelta(days=1))
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': c_in_due})
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': c_in_future})

    body = authenticated_client.get(f'/api/review-state?collection={colid}').get_json()
    due_ids = {c['id'] for c in body['due_cards']}
    assert due_ids == {c_in_due}
    assert c_out not in due_ids
    assert body['total_count'] == 2  # scope = beide Sammlungs-Karten
    assert body['due_count'] == 1


def test_review_state_tag_and_collection_and(app, authenticated_client, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    tag = _make_tag(app, uid, 'topic')
    colid = _make_collection(app, uid)
    # nur diese Karte ist BEIDES: im Topic UND in der Sammlung
    c_both = _make_card(app, uid, tag_ids=[tag], due=now - timedelta(days=1))
    c_tag_only = _make_card(app, uid, tag_ids=[tag], due=now - timedelta(days=1))
    c_col_only = _make_card(app, uid, due=now - timedelta(days=1))
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': c_both})
    authenticated_client.post(f'/api/collections/{colid}/cards', json={'card_id': c_col_only})

    body = authenticated_client.get(
        f'/api/review-state?tag={tag}&collection={colid}').get_json()
    assert {c['id'] for c in body['due_cards']} == {c_both}
    assert body['total_count'] == 1  # AND-Scope


def test_review_state_foreign_collection_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app)
    foreign = _make_collection(app, other_id, 'Foreign')
    assert authenticated_client.get(
        f'/api/review-state?collection={foreign}').status_code == 404


def test_review_state_garbage_collection_404(app, authenticated_client, test_user):
    assert authenticated_client.get('/api/review-state?collection=abc').status_code == 404
    assert authenticated_client.get('/api/review-state?collection=999999').status_code == 404
