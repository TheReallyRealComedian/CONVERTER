"""Reading-list / queue (R2-D) characterization tests.

Locks in the queue *mechanics* that survive R2-H:
- The inline ALTER-TABLE migration adds conversion.queue_position with NO
  backfill — every row stays NULL (off-list) after the column is (re-)added.
- POST /api/conversions/<id>/queue is now the reorder-only endpoint: action
  up/down swap with the direct neighbour over the *visible* (non-archived,
  queued) reading-list (one commit boundary), no-op at the edges. Ownership is
  scoped (404) and any other action (incl. the retired add/remove, now on
  /place) is a 400.
- to_dict exposes queue_position.

R2-H moved the add/remove half of the queue onto POST /place (a move between
the four places) and retired the "Weiterlesen" section + the view=queue tab;
that coverage now lives in test_library_ia.py (POST /place + the Lese-Liste
view). The Lese-Liste *place* is still derived from queue_position here.
"""
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


def _place_leseliste(client, cid):
    """R2-H: put a row on the reading list (append at the end) via the move
    endpoint — the replacement for the retired /queue {action:'add'}."""
    return client.post(f'/api/conversions/{cid}/place', json={'place': 'leseliste'})


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


# --- POST /api/conversions/<id>/queue : up / down reorder ---

def test_queue_up_swaps_with_neighbour(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], title='A')
    b = _make_conversion(app, test_user['id'], title='B')
    _place_leseliste(authenticated_client, a)  # 1.0
    _place_leseliste(authenticated_client, b)  # 2.0
    r = authenticated_client.post(f'/api/conversions/{b}/queue', json={'action': 'up'})
    assert r.status_code == 200
    assert _queue_pos_of(app, b) == 1.0
    assert _queue_pos_of(app, a) == 2.0


def test_queue_down_swaps_with_neighbour(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], title='A')
    b = _make_conversion(app, test_user['id'], title='B')
    _place_leseliste(authenticated_client, a)  # 1.0
    _place_leseliste(authenticated_client, b)  # 2.0
    r = authenticated_client.post(f'/api/conversions/{a}/queue', json={'action': 'down'})
    assert r.status_code == 200
    assert _queue_pos_of(app, a) == 2.0
    assert _queue_pos_of(app, b) == 1.0


def test_queue_up_at_top_is_noop(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    b = _make_conversion(app, test_user['id'])
    _place_leseliste(authenticated_client, a)  # 1.0
    _place_leseliste(authenticated_client, b)  # 2.0
    r = authenticated_client.post(f'/api/conversions/{a}/queue', json={'action': 'up'})
    assert r.status_code == 200
    assert _queue_pos_of(app, a) == 1.0
    assert _queue_pos_of(app, b) == 2.0


def test_queue_down_at_bottom_is_noop(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    b = _make_conversion(app, test_user['id'])
    _place_leseliste(authenticated_client, a)  # 1.0
    _place_leseliste(authenticated_client, b)  # 2.0
    r = authenticated_client.post(f'/api/conversions/{b}/queue', json={'action': 'down'})
    assert r.status_code == 200
    assert _queue_pos_of(app, a) == 1.0
    assert _queue_pos_of(app, b) == 2.0


def test_queue_swap_runs_over_visible_set_not_archived(app, authenticated_client, test_user):
    # The up/down swap must operate over the *visible* (non-archived) Lese-Liste
    # the view renders — otherwise "up" swaps with an invisible archived
    # neighbour and appears to do nothing (R2-D-swap-fix). R2-H archives via
    # /place (which nulls the queue), so we simulate a directly-set archive row
    # that still carries a position to prove the guard still holds.
    a = _make_conversion(app, test_user['id'], title='AlphaDoc')
    b = _make_conversion(app, test_user['id'], title='BetaDoc')
    c = _make_conversion(app, test_user['id'], title='GammaDoc')
    _place_leseliste(authenticated_client, a)  # 1.0
    _place_leseliste(authenticated_client, b)  # 2.0
    _place_leseliste(authenticated_client, c)  # 3.0
    _set_status(app, b, 'archive')             # middle row archived, keeps 2.0
    # Visible queue is [A, C]; move the bottom (C) up -> it must become first.
    r = authenticated_client.post(f'/api/conversions/{c}/queue', json={'action': 'up'})
    assert r.status_code == 200
    # C swapped with A (the visible neighbour), not with the archived B.
    assert _queue_pos_of(app, c) < _queue_pos_of(app, a)
    assert _queue_pos_of(app, b) == 2.0  # archived row untouched
    # The view reflects it: C now precedes A, B stays hidden.
    html = authenticated_client.get('/library?view=leseliste').data.decode()
    assert 'BetaDoc' not in html
    assert html.index('GammaDoc') < html.index('AlphaDoc')


def test_view_leseliste_reflects_reorder(app, authenticated_client, test_user):
    alpha = _make_conversion(app, test_user['id'], title='AlphaDoc')
    beta = _make_conversion(app, test_user['id'], title='BetaDoc')
    _place_leseliste(authenticated_client, alpha)  # 1.0
    _place_leseliste(authenticated_client, beta)   # 2.0
    authenticated_client.post(f'/api/conversions/{beta}/queue', json={'action': 'up'})

    html = authenticated_client.get('/library?view=leseliste').data.decode()
    # After moving Beta up it now precedes Alpha.
    assert html.index('BetaDoc') < html.index('AlphaDoc')


# --- guards: ownership + invalid action ---

def test_queue_other_users_conversion_404(app, authenticated_client, test_user):
    with app.app_context():
        bob = User(username='bob')
        bob.set_password('hunter2hunter2')
        db.session.add(bob)
        db.session.commit()
        bob_id = bob.id
    cid = _make_conversion(app, bob_id)
    r = authenticated_client.post(f'/api/conversions/{cid}/queue', json={'action': 'up'})
    assert r.status_code == 404
    assert _queue_pos_of(app, cid) is None


def test_queue_invalid_action_400(app, authenticated_client, test_user):
    # add/remove moved to /place — they are no longer valid queue actions, the
    # endpoint only reorders now.
    a = _make_conversion(app, test_user['id'])
    for action in ('bogus', 'add', 'remove'):
        r = authenticated_client.post(f'/api/conversions/{a}/queue', json={'action': action})
        assert r.status_code == 400, action
    assert _queue_pos_of(app, a) is None


def test_queue_missing_action_400(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'])
    r = authenticated_client.post(f'/api/conversions/{a}/queue', json={})
    assert r.status_code == 400


# --- to_dict ---

def test_conversion_to_dict_includes_queue_position(app, test_user):
    cid = _make_conversion(app, test_user['id'])
    with app.app_context():
        d = Conversion.query.get(cid).to_dict()
        assert 'queue_position' in d
        assert d['queue_position'] is None
