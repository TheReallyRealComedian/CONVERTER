"""LERN-GROUP Phase 1 — hierarchische Tags (Achse A).

Lockt fest: ``Tag.parent_id`` + Self-Relationship, der Subtree-BFS-Helper,
``PATCH /api/tags/<id>`` (parent setzen/lösen, Zyklus-Guard, fremdes Parent),
das Reparenten der Kinder beim Tag-Delete, ``card_count`` in ``/api/tags`` und
der ``?tag=``-Teilbaum-Filter auf ``/api/review-state``.
"""
from datetime import datetime, timedelta, timezone

from models import Card, Review, Tag, User, db


# --- helpers -----------------------------------------------------------------

def _make_other_user(app, username='mallory'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def _make_tag(app, user_id, name, parent_id=None):
    with app.app_context():
        t = Tag(user_id=user_id, name=name, parent_id=parent_id)
        db.session.add(t)
        db.session.commit()
        return t.id


def _make_card(app, user_id, tag_ids=(), due=None):
    with app.app_context():
        card = Card(user_id=user_id, type='atomic', front='Q', back='A')
        card.review = Review(due=due or datetime.now(timezone.utc))
        for tid in tag_ids:
            card.tags.append(Tag.query.get(tid))
        db.session.add(card)
        db.session.commit()
        return card.id


# --- Subtree-BFS-Helper (Modell-Ebene) ---------------------------------------

def test_subtree_ids_multilevel(app, test_user):
    uid = test_user['id']
    root = _make_tag(app, uid, 'root')
    mid = _make_tag(app, uid, 'mid', parent_id=root)
    leaf = _make_tag(app, uid, 'leaf', parent_id=mid)
    sibling = _make_tag(app, uid, 'sibling', parent_id=root)
    other = _make_tag(app, uid, 'unrelated')
    with app.app_context():
        assert Tag.subtree_ids(root, uid) == {root, mid, leaf, sibling}
        assert Tag.subtree_ids(mid, uid) == {mid, leaf}
        assert Tag.subtree_ids(leaf, uid) == {leaf}
        assert other not in Tag.subtree_ids(root, uid)


def test_subtree_ids_scoped_per_user(app, test_user):
    uid = test_user['id']
    other_id = _make_other_user(app, 'mallory')
    mine = _make_tag(app, uid, 'mine')
    # Ein fremdes Tag, das (hypothetisch) auf mein Tag zeigt, darf nie in
    # meinem Teilbaum auftauchen — der BFS lädt nur die User-Tags.
    _make_tag(app, other_id, 'theirs', parent_id=mine)
    with app.app_context():
        assert Tag.subtree_ids(mine, uid) == {mine}


# --- PATCH /api/tags/<id> -----------------------------------------------------

def test_patch_sets_and_clears_parent(app, authenticated_client, test_user):
    uid = test_user['id']
    parent = _make_tag(app, uid, 'parent')
    child = _make_tag(app, uid, 'child')

    r = authenticated_client.patch(f'/api/tags/{child}', json={'parent_id': parent})
    assert r.status_code == 200
    assert r.get_json()['parent_id'] == parent

    r2 = authenticated_client.patch(f'/api/tags/{child}', json={'parent_id': None})
    assert r2.status_code == 200
    assert r2.get_json()['parent_id'] is None
    with app.app_context():
        assert Tag.query.get(child).parent_id is None


def test_patch_cycle_rejected(app, authenticated_client, test_user):
    uid = test_user['id']
    root = _make_tag(app, uid, 'root')
    child = _make_tag(app, uid, 'child', parent_id=root)
    # root unter sein eigenes Kind hängen → Zyklus.
    r = authenticated_client.patch(f'/api/tags/{root}', json={'parent_id': child})
    assert r.status_code == 400
    # Selbst-Referenz ebenfalls.
    r2 = authenticated_client.patch(f'/api/tags/{root}', json={'parent_id': root})
    assert r2.status_code == 400
    with app.app_context():
        assert Tag.query.get(root).parent_id is None


def test_patch_foreign_parent_rejected(app, authenticated_client, test_user):
    uid = test_user['id']
    mine = _make_tag(app, uid, 'mine')
    other_id = _make_other_user(app, 'mallory')
    foreign = _make_tag(app, other_id, 'foreign')
    r = authenticated_client.patch(f'/api/tags/{mine}', json={'parent_id': foreign})
    assert r.status_code == 400
    with app.app_context():
        assert Tag.query.get(mine).parent_id is None


def test_patch_foreign_tag_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'mallory')
    foreign = _make_tag(app, other_id, 'foreign')
    r = authenticated_client.patch(f'/api/tags/{foreign}', json={'parent_id': None})
    assert r.status_code == 404


def test_patch_missing_parent_id_field_400(app, authenticated_client, test_user):
    uid = test_user['id']
    t = _make_tag(app, uid, 'solo')
    r = authenticated_client.patch(f'/api/tags/{t}', json={})
    assert r.status_code == 400


def test_patch_unknown_parent_400(app, authenticated_client, test_user):
    uid = test_user['id']
    t = _make_tag(app, uid, 'solo')
    r = authenticated_client.patch(f'/api/tags/{t}', json={'parent_id': 999999})
    assert r.status_code == 400


def test_patch_requires_login(app, client, test_user):
    uid = test_user['id']
    t = _make_tag(app, uid, 'solo')
    r = client.patch(f'/api/tags/{t}', json={'parent_id': None})
    assert r.status_code in (302, 401)


# --- Tag-Delete reparentet Kinder --------------------------------------------

def test_delete_tag_reparents_children_to_root(app, authenticated_client, test_user):
    uid = test_user['id']
    root = _make_tag(app, uid, 'root')
    mid = _make_tag(app, uid, 'mid', parent_id=root)
    leaf = _make_tag(app, uid, 'leaf', parent_id=mid)

    r = authenticated_client.delete(f'/api/tags/{mid}')
    assert r.status_code == 200
    with app.app_context():
        assert Tag.query.get(mid) is None
        # leaf war Kind von mid → jetzt Wurzel (parent_id NULL), kein totes FK.
        assert Tag.query.get(leaf).parent_id is None
        # root unberührt.
        assert Tag.query.get(root) is not None


def test_delete_tag_drains_card_junction(app, authenticated_client, test_user):
    uid = test_user['id']
    tid = _make_tag(app, uid, 'gone')
    cid = _make_card(app, uid, tag_ids=[tid])
    r = authenticated_client.delete(f'/api/tags/{tid}')
    assert r.status_code == 200
    with app.app_context():
        card = Card.query.get(cid)
        assert all(t.id != tid for t in card.tags)


# --- GET /api/tags card_count -------------------------------------------------

def test_list_tags_includes_card_count_and_parent_id(app, authenticated_client, test_user):
    uid = test_user['id']
    parent = _make_tag(app, uid, 'aaa-parent')
    child = _make_tag(app, uid, 'bbb-child', parent_id=parent)
    _make_card(app, uid, tag_ids=[child])
    _make_card(app, uid, tag_ids=[child])
    _make_card(app, uid, tag_ids=[parent])

    data = authenticated_client.get('/api/tags').get_json()
    by_name = {t['name']: t for t in data}
    assert by_name['bbb-child']['card_count'] == 2
    assert by_name['aaa-parent']['card_count'] == 1
    assert by_name['bbb-child']['parent_id'] == parent
    assert by_name['aaa-parent']['parent_id'] is None


# --- /api/review-state?tag= (Teilbaum-Filter) --------------------------------

def test_review_state_tag_filter_subtree(app, authenticated_client, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    root = _make_tag(app, uid, 'root')
    child = _make_tag(app, uid, 'child', parent_id=root)
    unrelated = _make_tag(app, uid, 'unrelated')

    c_root = _make_card(app, uid, tag_ids=[root], due=now - timedelta(days=1))
    c_child = _make_card(app, uid, tag_ids=[child], due=now - timedelta(days=1))
    c_child_future = _make_card(app, uid, tag_ids=[child], due=now + timedelta(days=1))
    c_other = _make_card(app, uid, tag_ids=[unrelated], due=now - timedelta(days=1))

    body = authenticated_client.get(f'/api/review-state?tag={root}').get_json()
    due_ids = {c['id'] for c in body['due_cards']}
    assert due_ids == {c_root, c_child}
    assert c_other not in due_ids
    # total_count reflektiert den Scope (Teilbaum-Karten, auch nicht-fällige).
    assert body['total_count'] == 3  # c_root, c_child, c_child_future
    assert body['due_count'] == 2


def test_review_state_tag_filter_leaf_only(app, authenticated_client, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    root = _make_tag(app, uid, 'root')
    child = _make_tag(app, uid, 'child', parent_id=root)
    _make_card(app, uid, tag_ids=[root], due=now - timedelta(days=1))
    c_child = _make_card(app, uid, tag_ids=[child], due=now - timedelta(days=1))

    body = authenticated_client.get(f'/api/review-state?tag={child}').get_json()
    assert {c['id'] for c in body['due_cards']} == {c_child}
    assert body['total_count'] == 1


def test_review_state_no_filter_unchanged(app, authenticated_client, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    tid = _make_tag(app, uid, 'tag')
    _make_card(app, uid, tag_ids=[tid], due=now - timedelta(days=1))
    _make_card(app, uid, due=now - timedelta(days=1))  # untagged

    body = authenticated_client.get('/api/review-state').get_json()
    assert body['total_count'] == 2
    assert body['due_count'] == 2


def test_review_state_foreign_tag_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'mallory')
    foreign = _make_tag(app, other_id, 'foreign')
    r = authenticated_client.get(f'/api/review-state?tag={foreign}')
    assert r.status_code == 404


def test_review_state_unknown_tag_404(app, authenticated_client, test_user):
    r = authenticated_client.get('/api/review-state?tag=999999')
    assert r.status_code == 404


def test_review_state_garbage_tag_404(app, authenticated_client, test_user):
    r = authenticated_client.get('/api/review-state?tag=abc')
    assert r.status_code == 404
