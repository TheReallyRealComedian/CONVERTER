"""TAG-CLEANUP Phase 1 — `POST /api/tags/merge` (token, by-name Lookup-only).

Destruktiver Merge: hängt Cards+Highlights+Conversions von source auf target um
(dedup-dann-repoint über alle drei Junctions), reparentet source-Kinder → target,
löscht source. dry_run=true = DEFAULT (same-path-rollback → apply-treue Vorschau,
nichts geschrieben). Zyklus-Guard via subtree_ids. Nur synthetische Wegwerf-Tags.
"""
from models import (
    Card, Conversion, Highlight, Tag, User,
    card_tags, conversion_tags, highlight_tags, db,
)


CARD_TOKEN = 'r4-test-card-token-9b2e'
MERGE_URL = '/api/tags/merge'


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


def _mk_highlight(uid, tag):
    cv = Conversion(user_id=uid, conversion_type='document', title='T', content='x')
    db.session.add(cv)
    db.session.flush()
    h = Highlight(conversion_id=cv.id, exact='quote')
    h.tags.append(tag)
    db.session.add(h)
    db.session.flush()
    return h


def _link_count(junction, tag_id):
    return db.session.execute(
        db.select(db.func.count()).select_from(junction)
        .where(junction.c.tag_id == tag_id)
    ).scalar()


# --- happy path: full reassign + delete --------------------------------------

def test_merge_moves_all_refs_and_deletes_source(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        src = _mk_tag(uid, 'src')
        tgt = _mk_tag(uid, 'tgt')
        _mk_card(uid, src)
        _mk_highlight(uid, src)
        _mk_conversion(uid, src)
        db.session.commit()
        src_id, tgt_id = src.id, tgt.id

    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'tgt', 'dry_run': False})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['applied'] is True and body['source_deleted'] is True
    assert body['reassigned']['cards']['moved'] == 1
    assert body['reassigned']['highlights']['moved'] == 1
    assert body['reassigned']['conversions']['moved'] == 1

    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='src').first() is None
        assert _link_count(card_tags, tgt_id) == 1
        assert _link_count(highlight_tags, tgt_id) == 1
        assert _link_count(conversion_tags, tgt_id) == 1
        assert _link_count(card_tags, src_id) == 0


# --- dedup: object carrying BOTH tags ----------------------------------------

def test_merge_dedups_object_on_both_tags(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        src = _mk_tag(uid, 'src')
        tgt = _mk_tag(uid, 'tgt')
        # one card carries BOTH src and tgt → must collapse to ONE tgt-link.
        c = _mk_card(uid, src)
        c.tags.append(tgt)
        db.session.commit()
        tgt_id = tgt.id

    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'tgt', 'dry_run': False})
    body = resp.get_json()
    assert body['reassigned']['cards']['deduped'] == 1
    assert body['reassigned']['cards']['moved'] == 0
    with app.app_context():
        assert _link_count(card_tags, tgt_id) == 1  # no duplicate row


# --- children reparented to target -------------------------------------------

def test_merge_reparents_children_to_target(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        src = _mk_tag(uid, 'src')
        tgt = _mk_tag(uid, 'tgt')
        kid = _mk_tag(uid, 'kid', parent_id=src.id)
        db.session.commit()
        tgt_id, kid_id = tgt.id, kid.id

    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'tgt', 'dry_run': False})
    body = resp.get_json()
    assert {c['name'] for c in body['children_reparented']} == {'kid'}
    with app.app_context():
        assert Tag.query.get(kid_id).parent_id == tgt_id


# --- cycle guard: target is a descendant of source --------------------------

def test_merge_cycle_guard_rejects(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        src = _mk_tag(uid, 'src')
        # tgt is a child of src → merging src→tgt would build a loop.
        tgt = _mk_tag(uid, 'tgt', parent_id=src.id)
        db.session.commit()
        src_id, tgt_id = src.id, tgt.id

    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'tgt', 'dry_run': False})
    assert resp.status_code == 400
    with app.app_context():
        # nothing changed.
        assert Tag.query.get(src_id) is not None
        assert Tag.query.get(tgt_id).parent_id == src_id


# --- source == target → no-op ------------------------------------------------

def test_merge_same_name_is_noop(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'dup')
        db.session.commit()
    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'DUP', 'target': 'dup', 'dry_run': False})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['source_deleted'] is False
    assert body['reassigned']['cards']['moved'] == 0
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='dup').first() is not None


# --- dry-run (default) writes nothing ----------------------------------------

def test_merge_dry_run_default_writes_nothing(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        src = _mk_tag(uid, 'src')
        tgt = _mk_tag(uid, 'tgt')
        _mk_card(uid, src)
        db.session.commit()
        src_id = src.id

    # no dry_run field → defaults to true.
    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'tgt'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['dry_run'] is True and body['applied'] is False
    # counts are still apply-accurate.
    assert body['reassigned']['cards']['moved'] == 1
    with app.app_context():
        # source still there, link intact.
        assert Tag.query.filter_by(user_id=uid, name='src').first() is not None
        assert _link_count(card_tags, src_id) == 1


# --- idempotent: second real run → 404 ---------------------------------------

def test_merge_idempotent_second_run_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'src')
        _mk_tag(uid, 'tgt')
        db.session.commit()
    first = client.post(MERGE_URL, headers=_card_auth(),
                        json={'source': 'src', 'target': 'tgt', 'dry_run': False})
    assert first.status_code == 200
    second = client.post(MERGE_URL, headers=_card_auth(),
                         json={'source': 'src', 'target': 'tgt', 'dry_run': False})
    assert second.status_code == 404


# --- missing source / target → 404 -------------------------------------------

def test_merge_missing_source_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'tgt')
        db.session.commit()
    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'ghost', 'target': 'tgt', 'dry_run': False})
    assert resp.status_code == 404


def test_merge_missing_target_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'src')
        db.session.commit()
    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'ghost', 'dry_run': False})
    assert resp.status_code == 404


# --- blank names → 400 -------------------------------------------------------

def test_merge_blank_name_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': '   ', 'target': 'tgt'}).status_code == 400
    assert client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': ''}).status_code == 400


# --- auth + owner scope ------------------------------------------------------

def test_merge_requires_token(app, client, test_user, monkeypatch):
    monkeypatch.delenv('CARD_TOKEN', raising=False)
    assert client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'a', 'target': 'b'}).status_code == 503
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(MERGE_URL, headers=_card_auth('wrong'),
                       json={'source': 'a', 'target': 'b'}).status_code == 401


def test_merge_owner_scoped(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        other = User(username='mallory')
        other.set_password('password1234')
        db.session.add(other)
        db.session.flush()
        other_id = other.id
        # mallory owns a 'src'; the token's target user (alice) does not.
        _mk_tag(other_id, 'src')
        _mk_tag(uid, 'tgt')
        db.session.commit()
    # alice (token target) has no 'src' → 404, mallory's tag untouched.
    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'tgt', 'dry_run': False})
    assert resp.status_code == 404
    with app.app_context():
        assert Tag.query.filter_by(user_id=other_id, name='src').first() is not None


# --- input-robustness hardenings (P2 retrofit) -------------------------------

def test_merge_non_string_name_400_not_500(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    # truthy non-string would .replace on an int → 500 without the isinstance guard.
    assert client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 123, 'target': 'tgt'}).status_code == 400
    assert client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': ['x']}).status_code == 400


def test_merge_strict_dry_run_falsy_nonbool_stays_dry(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'src')
        _mk_tag(uid, 'tgt')
        db.session.commit()
    # dry_run=0 (falsy non-bool) must NOT apply destructively — only real False does.
    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'tgt', 'dry_run': 0})
    assert resp.status_code == 200
    assert resp.get_json()['applied'] is False
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='src').first() is not None
    # the "false" string likewise stays a safe dry-run.
    resp2 = client.post(MERGE_URL, headers=_card_auth(),
                        json={'source': 'src', 'target': 'tgt', 'dry_run': 'false'})
    assert resp2.get_json()['applied'] is False
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='src').first() is not None


def test_merge_csrf_exempt_under_enforced_csrf(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)
    uid = test_user['id']
    with app.app_context():
        _mk_tag(uid, 'src')
        _mk_tag(uid, 'tgt')
        db.session.commit()
    # Bearer-only POST has no CSRF token → would be 400 if not exempt.
    resp = client.post(MERGE_URL, headers=_card_auth(),
                       json={'source': 'src', 'target': 'tgt'})
    assert resp.status_code == 200
