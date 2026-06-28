"""TAG-CLEANUP Phase 2 — `POST /api/tags/delete` (token, by-name Lookup-only).

Destruktives Löschen mit force-Guard-Rail:
 • mit reassign_to → Refs umhängen (geteilter _reassign_tag_refs), Kinder →
   reassign_to, tag löschen.
 • ohne reassign_to + Objekte + force=false → GUARD-RAIL (dry-run: requires_force
   ohne 409; echter Lauf: 409, nichts geschrieben).
 • ohne reassign_to (keine Objekte ODER force=true) → detach-all + Kinder → NULL.
dry_run=true = DEFAULT (same-path-rollback). Nur synthetische Wegwerf-Tags.
"""
from models import (
    Card, Conversion, Highlight, Tag, User,
    card_tags, conversion_tags, highlight_tags, db,
)


CARD_TOKEN = 'r4-test-card-token-9b2e'
DELETE_URL = '/api/tags/delete'


def _card_auth(token=CARD_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _mk_tag(uid, name, parent_id=None):
    t = Tag(user_id=uid, name=name, parent_id=parent_id)
    db.session.add(t)
    db.session.flush()
    return t


def _mk_card(uid, tag):
    c = Card(user_id=uid, type='atomic', front='Q', back='A')
    c.tags.append(tag)
    db.session.add(c)
    db.session.flush()
    return c


def _mk_conversion(uid, tag):
    cv = Conversion(user_id=uid, conversion_type='document', title='T', content='x')
    cv.tag_refs.append(tag)
    db.session.add(cv)
    db.session.flush()
    return cv


def _link_count(junction, tag_id):
    return db.session.execute(
        db.select(db.func.count()).select_from(junction)
        .where(junction.c.tag_id == tag_id)
    ).scalar()


# --- reassign_to path --------------------------------------------------------

def test_delete_with_reassign_moves_and_deletes(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        tag = _mk_tag(uid, 'junk')
        keep = _mk_tag(uid, 'keep')
        kid = _mk_tag(uid, 'kid', parent_id=tag.id)
        _mk_card(uid, tag)
        _mk_conversion(uid, tag)
        db.session.commit()
        keep_id, kid_id = keep.id, kid.id

    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'reassign_to': 'keep', 'dry_run': False})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['tag_deleted'] is True
    assert body['reassign_to']['name'] == 'keep'
    assert body['reassigned']['cards']['moved'] == 1
    assert body['reassigned']['conversions']['moved'] == 1
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='junk').first() is None
        assert _link_count(card_tags, keep_id) == 1
        assert _link_count(conversion_tags, keep_id) == 1
        assert Tag.query.get(kid_id).parent_id == keep_id


def test_delete_reassign_dedups(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        tag = _mk_tag(uid, 'junk')
        keep = _mk_tag(uid, 'keep')
        c = _mk_card(uid, tag)
        c.tags.append(keep)  # card carries BOTH
        db.session.commit()
        keep_id = keep.id
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'reassign_to': 'keep', 'dry_run': False})
    assert resp.get_json()['reassigned']['cards']['deduped'] == 1
    with app.app_context():
        assert _link_count(card_tags, keep_id) == 1


# --- guard-rail: objects + no reassign + force=false -------------------------

def test_delete_guardrail_409_real_run(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        tag = _mk_tag(uid, 'junk')
        _mk_card(uid, tag)
        db.session.commit()
        tag_id = tag.id

    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'dry_run': False})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['requires_force'] is True
    assert body['affected']['cards'] == 1
    assert body['tag_deleted'] is False
    with app.app_context():
        # nothing changed: tag + link intact.
        assert Tag.query.get(tag_id) is not None
        assert _link_count(card_tags, tag_id) == 1


def test_delete_guardrail_dry_run_no_409(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        tag = _mk_tag(uid, 'junk')
        _mk_card(uid, tag)
        db.session.commit()
        tag_id = tag.id

    # default dry_run=true → 200 preview with requires_force, NO 409.
    resp = client.post(DELETE_URL, headers=_card_auth(), json={'tag': 'junk'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['requires_force'] is True
    assert body['tag_deleted'] is False
    assert body['affected']['cards'] == 1
    with app.app_context():
        assert Tag.query.get(tag_id) is not None


# --- force=true detaches and deletes -----------------------------------------

def test_delete_force_detaches_all_and_deletes(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        tag = _mk_tag(uid, 'junk')
        kid = _mk_tag(uid, 'kid', parent_id=tag.id)
        _mk_card(uid, tag)
        db.session.commit()
        tag_id, kid_id = tag.id, kid.id

    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'force': True, 'dry_run': False})
    assert resp.status_code == 200
    assert resp.get_json()['tag_deleted'] is True
    with app.app_context():
        assert Tag.query.get(tag_id) is None
        assert _link_count(card_tags, tag_id) == 0
        # child reparented to NULL (root), like the session-delete semantics.
        assert Tag.query.get(kid_id).parent_id is None


# --- no objects → direct delete (force irrelevant) ---------------------------

def test_delete_no_objects_direct(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'lonely')
        db.session.commit()
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'lonely', 'dry_run': False})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['tag_deleted'] is True
    assert body['requires_force'] is False
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='lonely').first() is None


# --- dry-run writes nothing in the detach path -------------------------------

def test_delete_dry_run_force_writes_nothing(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        tag = _mk_tag(uid, 'junk')
        _mk_card(uid, tag)
        db.session.commit()
        tag_id = tag.id
    # force=true but dry_run default true → preview only, nothing written.
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'force': True})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['applied'] is False and body['tag_deleted'] is False
    with app.app_context():
        assert Tag.query.get(tag_id) is not None
        assert _link_count(card_tags, tag_id) == 1


# --- reassign cycle guard / reassign==tag / missing --------------------------

def test_delete_reassign_cycle_guard(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        tag = _mk_tag(uid, 'parent')
        _mk_tag(uid, 'child', parent_id=tag.id)  # reassign target is descendant
        db.session.commit()
        tag_id = tag.id
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'parent', 'reassign_to': 'child', 'dry_run': False})
    assert resp.status_code == 400
    with app.app_context():
        assert Tag.query.get(tag_id) is not None


def test_delete_reassign_equals_tag_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'same')
        db.session.commit()
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'SAME', 'reassign_to': 'same', 'dry_run': False})
    assert resp.status_code == 400


def test_delete_reassign_missing_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'junk')
        db.session.commit()
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'reassign_to': 'ghost', 'dry_run': False})
    assert resp.status_code == 404


def test_delete_tag_missing_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'ghost', 'dry_run': False})
    assert resp.status_code == 404


# --- idempotent + input robustness -------------------------------------------

def test_delete_idempotent_second_run_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'lonely')
        db.session.commit()
    first = client.post(DELETE_URL, headers=_card_auth(),
                        json={'tag': 'lonely', 'dry_run': False})
    assert first.status_code == 200
    second = client.post(DELETE_URL, headers=_card_auth(),
                         json={'tag': 'lonely', 'dry_run': False})
    assert second.status_code == 404


def test_delete_non_string_name_400_not_500(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 123}).status_code == 400
    # truthy non-string reassign_to → 400, not 500.
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'junk')
        db.session.commit()
    assert client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'reassign_to': 99}).status_code == 400


def test_delete_strict_force_falsy_nonbool_stays_guarded(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        tag = _mk_tag(uid, 'junk')
        _mk_card(uid, tag)
        db.session.commit()
        tag_id = tag.id
    # force=1 (truthy non-bool) must NOT lift the guard-rail — only real True does.
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'force': 1, 'dry_run': False})
    assert resp.status_code == 409
    with app.app_context():
        assert Tag.query.get(tag_id) is not None


# --- auth + owner scope ------------------------------------------------------

def test_delete_requires_token(app, client, test_user, monkeypatch):
    monkeypatch.delenv('CARD_TOKEN', raising=False)
    assert client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'a'}).status_code == 503
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(DELETE_URL, headers=_card_auth('wrong'),
                       json={'tag': 'a'}).status_code == 401


def test_delete_owner_scoped(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        other = User(username='mallory')
        other.set_password('password1234')
        db.session.add(other)
        db.session.flush()
        other_id = other.id
        _mk_tag(other_id, 'junk')  # owned by mallory, not the token target
        db.session.commit()
    resp = client.post(DELETE_URL, headers=_card_auth(),
                       json={'tag': 'junk', 'dry_run': False})
    assert resp.status_code == 404
    with app.app_context():
        assert Tag.query.filter_by(user_id=other_id, name='junk').first() is not None


def test_delete_csrf_exempt_under_enforced_csrf(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'lonely')
        db.session.commit()
    resp = client.post(DELETE_URL, headers=_card_auth(), json={'tag': 'lonely'})
    assert resp.status_code == 200
