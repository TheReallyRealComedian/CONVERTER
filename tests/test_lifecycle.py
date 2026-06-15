"""Lifecycle-status (R2-C) characterization tests.

Locks in:
- The inline ALTER-TABLE migration adds conversion.lifecycle_status and runs a
  one-shot differentiated backfill (ai_newsletter -> inbox, everything else ->
  archive). A second pass is a no-op because the column already exists, so a
  user-triaged row is never re-clobbered.
- PUT /api/conversions/<id> accepts lifecycle_status (validated against
  LIFECYCLE_STATUSES) analogous to is_favorite, and persists it. The column
  stays the source of truth on the read-API + the to_dict payload.

R2-H retired the GET /library?status location filter — the four-place views
(Inbox/Lese-Liste/Bibliothek/Archiv) replace it; that coverage now lives in
test_library_ia.py. The migration / PUT / to_dict behaviour is unchanged.
"""
from sqlalchemy import inspect, text

from app_pkg import _run_pending_migrations
from models import Conversion, db


def _make_conversion(app, user_id, **overrides):
    payload = dict(
        user_id=user_id,
        conversion_type='markdown_input',
        title='Sample Title',
        content='Sample content body.',
    )
    payload.update(overrides)
    with app.app_context():
        c = Conversion(**payload)
        db.session.add(c)
        db.session.commit()
        return c.id


def _status_of(app, cid):
    with app.app_context():
        return db.session.execute(
            text('SELECT lifecycle_status FROM conversion WHERE id = :id'),
            {'id': cid},
        ).scalar()


def _set_status(app, cid, status):
    with app.app_context():
        c = Conversion.query.get(cid)
        c.lifecycle_status = status
        db.session.commit()


# --- Migration: column-add + differentiated backfill ---

def test_migration_adds_column_and_differentiated_backfill(app, test_user):
    # Seed a newsletter + a non-newsletter while the column still exists.
    nl_id = _make_conversion(app, test_user['id'], conversion_type='ai_newsletter',
                             title='Newsletter')
    old_id = _make_conversion(app, test_user['id'], conversion_type='markdown_input',
                              title='Old tool output')
    with app.app_context():
        # Simulate a pre-R2-C schema: drop the index then the column (SQLite
        # refuses DROP COLUMN while an index references it).
        db.session.execute(text('DROP INDEX IF EXISTS ix_conversion_lifecycle_status'))
        db.session.execute(text('ALTER TABLE conversion DROP COLUMN lifecycle_status'))
        db.session.commit()
        assert 'lifecycle_status' not in {
            c['name'] for c in inspect(db.engine).get_columns('conversion')
        }
        # Run migrations -> re-add column + one-shot differentiated backfill.
        _run_pending_migrations(app)
        cols = {c['name'] for c in inspect(db.engine).get_columns('conversion')}
        assert 'lifecycle_status' in cols
    # Newsletter stays in the inbox triage; the old tool output goes to archive.
    assert _status_of(app, nl_id) == 'inbox'
    assert _status_of(app, old_id) == 'archive'


def test_migration_second_run_does_not_reclobber(app, test_user):
    # The column already exists (db.create_all) -> the migration block is
    # skipped, so a user-triaged non-newsletter is never reset to archive.
    cid = _make_conversion(app, test_user['id'], conversion_type='markdown_input')
    _set_status(app, cid, 'inbox')  # user pulled it back into the inbox
    with app.app_context():
        _run_pending_migrations(app)
    assert _status_of(app, cid) == 'inbox'


# --- PUT /api/conversions/<id> lifecycle_status ---

def test_put_sets_lifecycle_status_and_persists(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.put(f'/api/conversions/{cid}',
                                    json={'lifecycle_status': 'later'})
    assert resp.status_code == 200
    assert resp.get_json()['lifecycle_status'] == 'later'
    assert _status_of(app, cid) == 'later'


def test_put_rejects_invalid_lifecycle_status(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.put(f'/api/conversions/{cid}',
                                    json={'lifecycle_status': 'bogus'})
    assert resp.status_code == 400
    assert 'Lifecycle-Status' in resp.get_json()['error']
    # The rejected write must not have touched the row (still the default).
    assert _status_of(app, cid) == 'inbox'


# --- to_dict ---

def test_conversion_to_dict_includes_lifecycle_status(app, test_user):
    cid = _make_conversion(app, test_user['id'])
    with app.app_context():
        d = Conversion.query.get(cid).to_dict()
        assert d['lifecycle_status'] == 'inbox'
