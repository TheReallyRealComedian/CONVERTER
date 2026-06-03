"""Lifecycle-status (R2-C) characterization tests.

Locks in:
- The inline ALTER-TABLE migration adds conversion.lifecycle_status and runs a
  one-shot differentiated backfill (ai_newsletter -> inbox, everything else ->
  archive). A second pass is a no-op because the column already exists, so a
  user-triaged row is never re-clobbered.
- PUT /api/conversions/<id> accepts lifecycle_status (validated against
  LIFECYCLE_STATUSES) analogous to is_favorite, and persists it.
- GET /library?status filters on the column, contributes to has_active_filter,
  combines with ?tag, and is preserved across pagination. An unknown status is
  not treated as a filter.
"""
import re

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


# --- GET /library?status filter ---

def test_library_status_filter_surfaces_only_matching(app, authenticated_client, test_user):
    later_id = _make_conversion(app, test_user['id'], title='Later-Doc')
    archive_id = _make_conversion(app, test_user['id'], title='Archive-Doc')
    _set_status(app, later_id, 'later')
    _set_status(app, archive_id, 'archive')

    resp = authenticated_client.get('/library?status=later')
    assert resp.status_code == 200
    assert b'Later-Doc' in resp.data
    assert b'Archive-Doc' not in resp.data


def test_library_status_filter_unknown_value_shows_all(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], title='Alpha-Doc')
    b = _make_conversion(app, test_user['id'], title='Beta-Doc')
    _set_status(app, a, 'inbox')
    _set_status(app, b, 'archive')
    # ?status=bogus is not a known status -> no filtering, both rows show.
    resp = authenticated_client.get('/library?status=bogus')
    assert resp.status_code == 200
    assert b'Alpha-Doc' in resp.data
    assert b'Beta-Doc' in resp.data


def test_library_status_filter_sets_active_filter_empty_state(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], title='Inbox-Only')
    _set_status(app, cid, 'inbox')
    resp = authenticated_client.get('/library?status=archive')
    assert resp.status_code == 200
    assert b'Inbox-Only' not in resp.data
    # status is in has_active_filter -> the 0-hit *filtered* empty-state renders.
    assert 'Keine Treffer mit aktuellen Filtern'.encode() in resp.data


def test_library_status_filter_combines_with_tag(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], title='Later-KI')
    authenticated_client.post(f'/api/conversions/{a}/tags', json={'name': 'ki'})
    _set_status(app, a, 'later')
    b = _make_conversion(app, test_user['id'], title='Archive-KI')
    authenticated_client.post(f'/api/conversions/{b}/tags', json={'name': 'ki'})
    _set_status(app, b, 'archive')

    resp = authenticated_client.get('/library?status=later&tag=ki')
    assert resp.status_code == 200
    assert b'Later-KI' in resp.data
    assert b'Archive-KI' not in resp.data


def test_library_status_filter_preserved_across_pagination(app, authenticated_client, test_user):
    # 21 later-rows at default per_page=20 -> 2 pages. A page-2 pagination link
    # must keep ?status=later so the next page stays filtered.
    for i in range(21):
        cid = _make_conversion(app, test_user['id'], title=f'Later-{i:02d}')
        _set_status(app, cid, 'later')
    resp = authenticated_client.get('/library?status=later')
    assert resp.status_code == 200
    html = resp.data.decode()
    page2_links = re.findall(r'href="[^"]*page=2[^"]*"', html)
    assert page2_links, 'expected a page-2 pagination link'
    assert any('status=later' in link for link in page2_links)


# --- to_dict ---

def test_conversion_to_dict_includes_lifecycle_status(app, test_user):
    cid = _make_conversion(app, test_user['id'])
    with app.app_context():
        d = Conversion.query.get(cid).to_dict()
        assert d['lifecycle_status'] == 'inbox'
