"""Reading-progress API + migration characterization tests (R2-B).

Locks in:
- PATCH /api/conversions/<id>/progress persists the furthest-read percent,
  clamps out-of-range values instead of 400ing (fire-and-forget scroll signal),
  and rejects bool / non-numeric / missing-key / non-dict bodies as 400 and a
  foreign conversion as 404 — the same atomic-endpoint contract as the R1-B-B
  note-PATCH.
- R2-F forward-clamp: the persisted value only ever moves forward
  (max(stored, incoming)); a smaller incoming percent is a no-op and the
  response echoes the retained value, so no client regression can lower the
  mark. Range-clamp still applies first (out-of-range-low keeps the stored
  value rather than resetting to 0).
- Conversion.to_dict surfaces last_read_percent.
- _run_pending_migrations adds the last_read_percent column idempotently via
  inline ALTER TABLE (a second pass is a no-op).
"""
import json

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from app_pkg import _run_pending_migrations
from models import Conversion, User, db


def _make_conversion(app, user_id, **overrides):
    payload = dict(
        user_id=user_id,
        conversion_type='markdown_input',
        title='Sample Title',
        content='Sample content body.',
        tags='',
        metadata_json=json.dumps({'src': 'test'}),
    )
    payload.update(overrides)
    with app.app_context():
        c = Conversion(**payload)
        db.session.add(c)
        db.session.commit()
        return c.id


def _make_other_user(app, username='bob'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


# --- PATCH /api/conversions/<id>/progress ---

def test_api_progress_persists_valid_percent(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 42.5})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['success'] is True
    assert body['last_read_percent'] == 42.5
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 42.5


def test_api_progress_clamps_above_100(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 100.0001})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 100.0
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 100.0


def test_api_progress_clamps_below_0(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': -7})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 0.0
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 0.0


# --- R2-F forward-clamp (furthest-read only moves forward) ---

def test_api_progress_forward_clamp_ignores_lower_value(app, authenticated_client, test_user):
    # A smaller incoming percent must not lower the stored mark — the response
    # echoes the retained (higher) value, the DB is unchanged.
    cid = _make_conversion(app, test_user['id'], last_read_percent=80.0)
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 30})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 80.0
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 80.0


def test_api_progress_forward_clamp_accepts_higher_value(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], last_read_percent=30.0)
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 80})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 80.0
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 80.0


def test_api_progress_forward_clamp_below_zero_keeps_stored(app, authenticated_client, test_user):
    # Range-clamp pins -7 to 0.0; forward-clamp then keeps the stored 50 — an
    # out-of-range-low fire-and-forget signal must not reset progress.
    cid = _make_conversion(app, test_user['id'], last_read_percent=50.0)
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': -7})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 50.0
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 50.0


def test_api_progress_forward_clamp_above_100_takes_100(app, authenticated_client, test_user):
    # Range-clamp pins 100.0001 to 100.0; forward-clamp takes it (>= stored 50).
    cid = _make_conversion(app, test_user['id'], last_read_percent=50.0)
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 100.0001})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 100.0
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 100.0


# --- R2-G reset flag (bypasses the forward-clamp, NULL = "never read") ---

def test_api_progress_reset_clears_to_null_despite_stored(app, authenticated_client, test_user):
    # reset:true must set NULL even with a stored value — proves it bypasses the
    # forward-clamp, which would otherwise keep max(80, 0) = 80.
    cid = _make_conversion(app, test_user['id'], last_read_percent=80.0)
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'reset': True})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['success'] is True
    assert body['last_read_percent'] is None
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent is None


def test_api_progress_reset_wins_over_percent(app, authenticated_client, test_user):
    # reset:true ignores any accompanying percent — reset wins, result is NULL.
    cid = _make_conversion(app, test_user['id'], last_read_percent=80.0)
    resp = authenticated_client.patch(
        f'/api/conversions/{cid}/progress', json={'reset': True, 'percent': 50})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] is None
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent is None


def test_api_progress_reset_false_uses_normal_clamp(app, authenticated_client, test_user):
    # reset:false (falsy) falls through to the normal forward-clamp path: a
    # smaller percent is a no-op against the stored value.
    cid = _make_conversion(app, test_user['id'], last_read_percent=80.0)
    resp = authenticated_client.patch(
        f'/api/conversions/{cid}/progress', json={'reset': False, 'percent': 30})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 80.0
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 80.0


def test_api_progress_reset_rejects_non_bool(app, authenticated_client, test_user):
    # Truthy garbage (1, "true", …) must NOT trigger a reset — same explicit-type
    # stance as the percent bool-check: a non-bool reset is a 400, and the stored
    # value is left untouched.
    cid = _make_conversion(app, test_user['id'], last_read_percent=80.0)
    for bad in (1, 'true', 'yes', []):
        resp = authenticated_client.patch(
            f'/api/conversions/{cid}/progress', json={'reset': bad})
        assert resp.status_code == 400, f'reset={bad!r} should be 400'
    with app.app_context():
        assert Conversion.query.get(cid).last_read_percent == 80.0


def test_api_progress_accepts_integer_percent(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 50})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 50.0


def test_api_progress_rejects_non_numeric(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 'x'})
    assert resp.status_code == 400


def test_api_progress_rejects_bool(app, authenticated_client, test_user):
    # isinstance(True, int) is True — the endpoint must reject bools explicitly
    # so True/False can't pose as a 1/0 percent.
    cid = _make_conversion(app, test_user['id'])
    assert authenticated_client.patch(
        f'/api/conversions/{cid}/progress', json={'percent': True}).status_code == 400
    assert authenticated_client.patch(
        f'/api/conversions/{cid}/progress', json={'percent': False}).status_code == 400


def test_api_progress_rejects_missing_key(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'other_field': 1})
    assert resp.status_code == 400


def test_api_progress_rejects_null_percent(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': None})
    assert resp.status_code == 400


def test_api_progress_rejects_non_dict_body(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json=['not', 'a', 'dict'])
    assert resp.status_code == 400


def test_api_progress_on_foreign_conversion_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'bob')
    cid = _make_conversion(app, other_id, title="Bob's doc")
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 20})
    assert resp.status_code == 404


# --- Conversion.to_dict ---

def test_conversion_to_dict_includes_last_read_percent(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    with app.app_context():
        assert Conversion.query.get(cid).to_dict()['last_read_percent'] is None
    authenticated_client.patch(f'/api/conversions/{cid}/progress', json={'percent': 33})
    with app.app_context():
        assert Conversion.query.get(cid).to_dict()['last_read_percent'] == 33.0


# --- Inline ALTER-TABLE migration idempotency ---

def test_migration_adds_last_read_percent_column_idempotently(app):
    with app.app_context():
        engine = db.engine
        # Simulate a pre-R2-B schema: drop the column so the migration must
        # re-add it. SQLite < 3.35 has no DROP COLUMN — skip there. The
        # try/finally restores the column so a mid-test failure can't poison
        # the session-scoped engine for the rest of the suite.
        try:
            db.session.execute(text('ALTER TABLE conversion DROP COLUMN last_read_percent'))
            db.session.commit()
        except OperationalError:
            db.session.rollback()
            pytest.skip('SQLite build without DROP COLUMN support')

        try:
            assert 'last_read_percent' not in {
                c['name'] for c in inspect(engine).get_columns('conversion')
            }

            _run_pending_migrations(app)
            assert 'last_read_percent' in {
                c['name'] for c in inspect(engine).get_columns('conversion')
            }

            # Second pass is a no-op — no error, column present exactly once.
            _run_pending_migrations(app)
            cols = [c['name'] for c in inspect(engine).get_columns('conversion')]
            assert cols.count('last_read_percent') == 1
        finally:
            cols = {c['name'] for c in inspect(engine).get_columns('conversion')}
            if 'last_read_percent' not in cols:
                db.session.execute(
                    text('ALTER TABLE conversion ADD COLUMN last_read_percent FLOAT'))
                db.session.commit()
