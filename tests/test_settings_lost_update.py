"""LOST-UPDATE — the shared settings blob merges atomically in the database.

P1 measured on ``User.settings_json`` (learn keys flat + the ``document_api``
namespace): a learn PUT and a document PUT at the same instant lost a write
in 125 of 200 rounds, and on a never-written blob a whole namespace VANISHED
30 of 30 times — ``write_settings_keys`` merged in Python over the row the
request had loaded, last writer wins. Now it is ONE UPDATE with SQLite's
``json_patch`` (RFC 7396): the merge runs under the database write lock — no
read, no version, no retry.

Two semantic differences to the former in-memory ``dict.update`` are pinned
here on purpose (documented, not hidden — see the helper's docstring):
a ``None`` value DELETES its key, and objects merge RECURSIVELY.
"""
import json

from sqlalchemy import text

from app_pkg.learn import write_settings_keys
from models import User, db


def _raw_blob(app, uid):
    with app.app_context():
        raw = db.session.execute(
            text('SELECT settings_json FROM "user" WHERE id = :uid'), {'uid': uid}).scalar()
    return json.loads(raw) if raw else None


def _set_raw_blob(app, uid, raw):
    with app.app_context():
        db.session.execute(text('UPDATE "user" SET settings_json = :raw WHERE id = :uid'),
                           {'raw': raw, 'uid': uid})
        db.session.commit()


def _other_writer_merges(uid, updates):
    """A concurrent writer on a SECOND pooled connection, committed."""
    with db.engine.begin() as conn:
        conn.execute(text(
            'UPDATE "user" SET settings_json = json_patch(coalesce(settings_json, \'{}\'), :u) '
            'WHERE id = :uid'), {'u': json.dumps(updates), 'uid': uid})


def test_json_patch_is_available_sentinel(app):
    # The helper depends on SQLite's JSON1 json_patch (container: 3.37.2).
    with app.app_context():
        # JSON as bind parameters — text() would read ':1' inside a literal
        # as a bind name (the production helper passes JSON the same way).
        merged = db.session.execute(
            text('SELECT json_patch(:a, :b)'), {'a': '{"a":1}', 'b': '{"b":2}'}).scalar()
    assert json.loads(merged) == {'a': 1, 'b': 2}


# --- the P1 failure, deterministically ----------------------------------------

def test_stale_object_cannot_revert_the_other_namespace(app, test_user):
    uid = test_user['id']
    _set_raw_blob(app, uid, json.dumps({'daily_new_limit': 5, 'document_api': {'default_mode': 'lokal'}}))
    with app.app_context():
        user = db.session.get(User, uid)                # loaded: document_api = lokal
        _other_writer_merges(uid, {'document_api': {'default_mode': 'cloud'}})  # lands in between
        write_settings_keys(user, {'daily_new_limit': 9, 'ordering_mode': 'smart'})
        db.session.commit()
    blob = _raw_blob(app, uid)
    assert blob['daily_new_limit'] == 9 and blob['ordering_mode'] == 'smart'   # ours
    assert blob['document_api'] == {'default_mode': 'cloud'}                   # theirs survives


def test_first_write_on_a_null_blob_cannot_vanish_the_other_namespace(app, test_user):
    # The 30-of-30 case: both namespaces' FIRST writes race on an empty blob.
    uid = test_user['id']
    _set_raw_blob(app, uid, None)
    with app.app_context():
        user = db.session.get(User, uid)                # loaded: NULL
        _other_writer_merges(uid, {'document_api': {'default_mode': 'cloud'}})
        write_settings_keys(user, {'daily_new_limit': 7})
        db.session.commit()
    assert _raw_blob(app, uid) == {'document_api': {'default_mode': 'cloud'}, 'daily_new_limit': 7}


def test_user_object_reads_the_merged_blob_after_the_write(app, test_user):
    uid = test_user['id']
    _set_raw_blob(app, uid, json.dumps({'document_api': {'default_mode': 'cloud'}}))
    with app.app_context():
        user = db.session.get(User, uid)
        write_settings_keys(user, {'daily_new_limit': 3})
        db.session.commit()
        assert json.loads(user.settings_json) == {
            'document_api': {'default_mode': 'cloud'}, 'daily_new_limit': 3}


def test_helper_does_not_commit(app, test_user):
    uid = test_user['id']
    _set_raw_blob(app, uid, json.dumps({'daily_new_limit': 1}))
    with app.app_context():
        user = db.session.get(User, uid)
        write_settings_keys(user, {'daily_new_limit': 2})
        db.session.rollback()                            # the caller owns the transaction
    assert _raw_blob(app, uid) == {'daily_new_limit': 1}


# --- documented semantics (differences to dict.update) ------------------------

def test_none_deletes_the_key_rfc7396(app, test_user):
    uid = test_user['id']
    _set_raw_blob(app, uid, json.dumps({'daily_new_limit': 4, 'ordering_mode': 'random'}))
    with app.app_context():
        write_settings_keys(db.session.get(User, uid), {'ordering_mode': None})
        db.session.commit()
    assert _raw_blob(app, uid) == {'daily_new_limit': 4}    # deleted, not stored as null


def test_objects_merge_recursively(app, test_user):
    # dict.update would have REPLACED the namespace and dropped 'extra'.
    uid = test_user['id']
    _set_raw_blob(app, uid, json.dumps({'document_api': {'default_mode': 'cloud', 'extra': 1}}))
    with app.app_context():
        write_settings_keys(db.session.get(User, uid), {'document_api': {'default_mode': 'lokal'}})
        db.session.commit()
    assert _raw_blob(app, uid) == {'document_api': {'default_mode': 'lokal', 'extra': 1}}


def test_corrupt_or_non_object_blob_starts_fresh(app, test_user):
    uid = test_user['id']
    for broken in ('{broken', '[1, 2]', '"smart"'):
        _set_raw_blob(app, uid, broken)
        with app.app_context():
            write_settings_keys(db.session.get(User, uid), {'daily_new_limit': 8})
            db.session.commit()
        assert _raw_blob(app, uid) == {'daily_new_limit': 8}
