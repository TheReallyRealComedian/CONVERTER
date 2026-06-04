"""Reading-list / queue (R2-D) characterization tests.

Locks in:
- The inline ALTER-TABLE migration adds conversion.queue_position with NO
  backfill — every row stays NULL (off-list) after the column is (re-)added.
- POST /api/conversions/<id>/queue with action add/remove/up/down: add appends
  at max+1.0 (idempotent), remove nulls it, up/down swap with the direct
  neighbour (one commit boundary) and are no-ops at the edges. Ownership is
  scoped (404 for another user's row) and an unknown action is a 400.
- GET /library?view=queue surfaces only queued + non-archived rows, ordered by
  position. GET /library?view=reading surfaces only in-progress rows
  (0 < last_read_percent < 95), most-recently-touched first.
- to_dict exposes queue_position.
"""
from datetime import datetime, timezone

from sqlalchemy import inspect, text

from app_pkg import _run_pending_migrations
from models import Conversion, User, db


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


def _queue_pos_of(app, cid):
    with app.app_context():
        return db.session.execute(
            text('SELECT queue_position FROM conversion WHERE id = :id'),
            {'id': cid},
        ).scalar()


def _set_status(app, cid, status):
    with app.app_context():
        c = Conversion.query.get(cid)
        c.lifecycle_status = status
        db.session.commit()


def _add_to_queue(client, cid):
    return client.post(f'/api/conversions/{cid}/queue', json={'action': 'add'})


# --- Migration: column-add, no backfill ---

def test_migration_adds_queue_position_without_backfill(app, test_user):
    a = _make_conversion(app, test_user['id'], conversion_type='ai_newsletter')
    b = _make_conversion(app, test_user['id'], conversion_type='markdown_input')
    with app.app_context():
        # Simulate a pre-R2-D schema: drop the index then the column (SQLite
        # refuses DROP COLUMN while an index references it).
        db.session.execute(text('DROP INDEX IF EXISTS ix_conversion_queue_position'))
        db.session.execute(text('ALTER TABLE conversion DROP COLUMN queue_position'))
        db.session.commit()
        assert 'queue_position' not in {
            c['name'] for c in inspect(db.engine).get_columns('conversion')
        }
        _run_pending_migrations(app)
        cols = {c['name'] for c in inspect(db.engine).get_columns('conversion')}
        assert 'queue_position' in cols
    # No backfill — every row starts NULL (empty reading list).
    assert _queue_pos_of(app, a) is None
    assert _queue_pos_of(app, b) is None


def test_migration_second_run_keeps_positions(app, test_user):
    # The column already exists (db.create_all) -> the migration block is
    # skipped, so a queued item's position is never reset.
    cid = _make_conversion(app, test_user['id'])
    with app.app_context():
        c = Conversion.query.get(cid)
        c.queue_position = 3.0
        db.session.commit()
        _run_pending_migrations(app)
    assert _queue_pos_of(app, cid) == 3.0


# --- POST /api/conversions/<id>/queue : add / remove ---

def test_queue_add_appends_at_end(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    b = _make_conversion(app, test_user['id'])
    r1 = _add_to_queue(authenticated_client, a)
    assert r1.status_code == 200
    assert r1.get_json()['queue_position'] == 1.0
    r2 = _add_to_queue(authenticated_client, b)
    assert r2.get_json()['queue_position'] == 2.0


def test_queue_add_is_idempotent(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    _add_to_queue(authenticated_client, a)
    # Re-adding does not bump the position.
    r = _add_to_queue(authenticated_client, a)
    assert r.status_code == 200
    assert r.get_json()['queue_position'] == 1.0


def test_queue_remove_sets_null(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    _add_to_queue(authenticated_client, a)
    r = authenticated_client.post(f'/api/conversions/{a}/queue', json={'action': 'remove'})
    assert r.status_code == 200
    assert r.get_json()['queue_position'] is None
    assert _queue_pos_of(app, a) is None


# --- up / down reorder ---

def test_queue_up_swaps_with_neighbour(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], title='A')
    b = _make_conversion(app, test_user['id'], title='B')
    _add_to_queue(authenticated_client, a)  # 1.0
    _add_to_queue(authenticated_client, b)  # 2.0
    r = authenticated_client.post(f'/api/conversions/{b}/queue', json={'action': 'up'})
    assert r.status_code == 200
    assert _queue_pos_of(app, b) == 1.0
    assert _queue_pos_of(app, a) == 2.0


def test_queue_down_swaps_with_neighbour(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], title='A')
    b = _make_conversion(app, test_user['id'], title='B')
    _add_to_queue(authenticated_client, a)  # 1.0
    _add_to_queue(authenticated_client, b)  # 2.0
    r = authenticated_client.post(f'/api/conversions/{a}/queue', json={'action': 'down'})
    assert r.status_code == 200
    assert _queue_pos_of(app, a) == 2.0
    assert _queue_pos_of(app, b) == 1.0


def test_queue_up_at_top_is_noop(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    b = _make_conversion(app, test_user['id'])
    _add_to_queue(authenticated_client, a)  # 1.0
    _add_to_queue(authenticated_client, b)  # 2.0
    r = authenticated_client.post(f'/api/conversions/{a}/queue', json={'action': 'up'})
    assert r.status_code == 200
    assert _queue_pos_of(app, a) == 1.0
    assert _queue_pos_of(app, b) == 2.0


def test_queue_down_at_bottom_is_noop(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    b = _make_conversion(app, test_user['id'])
    _add_to_queue(authenticated_client, a)  # 1.0
    _add_to_queue(authenticated_client, b)  # 2.0
    r = authenticated_client.post(f'/api/conversions/{b}/queue', json={'action': 'down'})
    assert r.status_code == 200
    assert _queue_pos_of(app, a) == 1.0
    assert _queue_pos_of(app, b) == 2.0


# --- guards: ownership + invalid action ---

def test_queue_other_users_conversion_404(app, authenticated_client, test_user):
    with app.app_context():
        bob = User(username='bob')
        bob.set_password('hunter2hunter2')
        db.session.add(bob)
        db.session.commit()
        bob_id = bob.id
    cid = _make_conversion(app, bob_id)
    r = authenticated_client.post(f'/api/conversions/{cid}/queue', json={'action': 'add'})
    assert r.status_code == 404
    assert _queue_pos_of(app, cid) is None


def test_queue_invalid_action_400(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    r = authenticated_client.post(f'/api/conversions/{a}/queue', json={'action': 'bogus'})
    assert r.status_code == 400
    # The rejected action must not have queued the row.
    assert _queue_pos_of(app, a) is None


def test_queue_missing_action_400(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    r = authenticated_client.post(f'/api/conversions/{a}/queue', json={})
    assert r.status_code == 400


# --- GET /library?view=queue ---

def test_view_queue_orders_by_position_and_excludes_archive(app, authenticated_client, test_user):
    alpha = _make_conversion(app, test_user['id'], title='AlphaDoc')
    beta = _make_conversion(app, test_user['id'], title='BetaDoc')
    archived = _make_conversion(app, test_user['id'], title='ArchivedDoc')
    _make_conversion(app, test_user['id'], title='PlainDoc')  # never queued
    _add_to_queue(authenticated_client, alpha)     # 1.0
    _add_to_queue(authenticated_client, beta)      # 2.0
    _add_to_queue(authenticated_client, archived)  # 3.0
    _set_status(app, archived, 'archive')

    resp = authenticated_client.get('/library?view=queue')
    assert resp.status_code == 200
    html = resp.data.decode()
    # Archived-but-queued is filtered out; never-queued is absent.
    assert 'ArchivedDoc' not in html
    assert 'PlainDoc' not in html
    # Ordered by position asc.
    assert 'AlphaDoc' in html and 'BetaDoc' in html
    assert html.index('AlphaDoc') < html.index('BetaDoc')


def test_view_queue_reflects_reorder(app, authenticated_client, test_user):
    alpha = _make_conversion(app, test_user['id'], title='AlphaDoc')
    beta = _make_conversion(app, test_user['id'], title='BetaDoc')
    _add_to_queue(authenticated_client, alpha)  # 1.0
    _add_to_queue(authenticated_client, beta)   # 2.0
    authenticated_client.post(f'/api/conversions/{beta}/queue', json={'action': 'up'})

    resp = authenticated_client.get('/library?view=queue')
    html = resp.data.decode()
    # After moving Beta up it now precedes Alpha.
    assert html.index('BetaDoc') < html.index('AlphaDoc')


# --- GET /library?view=reading ---

def test_view_reading_filters_in_progress_only(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='StartedDoc', last_read_percent=40.0)
    _make_conversion(app, test_user['id'], title='FreshDoc')  # NULL
    _make_conversion(app, test_user['id'], title='ZeroDoc', last_read_percent=0.0)
    _make_conversion(app, test_user['id'], title='DoneDoc', last_read_percent=99.0)

    resp = authenticated_client.get('/library?view=reading')
    assert resp.status_code == 200
    html = resp.data.decode()
    assert 'StartedDoc' in html
    assert 'FreshDoc' not in html
    assert 'ZeroDoc' not in html
    assert 'DoneDoc' not in html


def test_view_reading_sorts_most_recent_first(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='OlderDoc', last_read_percent=20.0,
                     updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    _make_conversion(app, test_user['id'], title='NewerDoc', last_read_percent=20.0,
                     updated_at=datetime(2026, 3, 1, tzinfo=timezone.utc))

    resp = authenticated_client.get('/library?view=reading')
    html = resp.data.decode()
    assert html.index('NewerDoc') < html.index('OlderDoc')


# --- to_dict ---

def test_conversion_to_dict_includes_queue_position(app, test_user):
    cid = _make_conversion(app, test_user['id'])
    with app.app_context():
        d = Conversion.query.get(cid).to_dict()
        assert 'queue_position' in d
        assert d['queue_position'] is None
