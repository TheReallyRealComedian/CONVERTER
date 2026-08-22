"""LOST-UPDATE — the docwrite section replace is guarded by a content-bound version.

P1 measured: 8 agents replacing their OWN sections of one document at the same
instant lost 660 of 800 writes at 800 × HTTP 200 — ``replace_section`` splices
into the text it READ, and the whole content went back last-writer-wins. Now
``Conversion.content_version`` is bumped by EVERY content writer
(``Conversion.set_content``) and the section UPDATE is conditional on the
version that was read; a miss re-reads and re-splices into the other writer's
text, bounded by ``CONTENT_WRITE_ATTEMPTS``, then an honest 409.

Deliberately NOT a row-wide ``version_id_col`` on Conversion: progress,
/place, tags and the job reconciles write OTHER columns of the same row and
must not collide with a content edit — pinned below (a progress write does
not bump, the mapper carries no version column).

The concurrent writer is real, not mocked: a raw UPDATE over a SECOND engine
connection in its own committed transaction, injected between the request's
read and its write.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

import app_pkg.docwrite as docwrite_module
from app_pkg import _run_pending_migrations
from app_pkg.docwrite import CONTENT_WRITE_ATTEMPTS
from models import Conversion, db

CARD_TOKEN = 'lost-update-test-card-token-7f1c'
DOC = '# A\na\n# B\nb\n# C\nc\n'


def _auth():
    return {'Authorization': f'Bearer {CARD_TOKEN}'}


def _make_conversion(app, user_id, content=DOC):
    with app.app_context():
        conv = Conversion(user_id=user_id, conversion_type='markdown_input',
                          title='Doc', content=content)
        db.session.add(conv)
        db.session.commit()
        return conv.id


def _row(app, cid):
    """The row as SQLite holds it — raw, outside any ORM identity map."""
    with app.app_context():
        content, version, updated_at = db.session.execute(text(
            'SELECT content, content_version, updated_at FROM conversion WHERE id = :cid'),
            {'cid': cid}).one()
    return {'content': content, 'version': version, 'updated_at': updated_at}


def _other_writer_sets(cid, content):
    """A concurrent content write on a SECOND pooled connection, committed —
    what another agent (process/thread) does, seen from this request."""
    with db.engine.begin() as conn:
        conn.execute(text(
            'UPDATE conversion SET content = :c, content_version = content_version + 1 '
            'WHERE id = :cid'), {'c': content, 'cid': cid})


class _Interposer:
    """Wraps ``replace_section``; the first ``conflicts`` calls let another
    writer change the document first (between this request's read and its
    write) and every call records the text it was handed."""

    def __init__(self, cid, conflicts, other_text):
        self.cid, self.conflicts, self.other_text = cid, conflicts, other_text
        self.inputs = []

    def __call__(self, markdown_text, heading, new_section):
        self.inputs.append(markdown_text)
        if len(self.inputs) <= self.conflicts:
            _other_writer_sets(self.cid, self.other_text.format(n=len(self.inputs)))
        return _REAL_REPLACE(markdown_text, heading, new_section)


_REAL_REPLACE = docwrite_module.replace_section  # captured before any monkeypatch


# --- sentinel + writers -------------------------------------------------------

def test_content_version_is_content_bound_not_a_row_version():
    col = Conversion.__table__.c.content_version
    assert col.nullable is False
    assert str(col.server_default.arg) == '1'
    # No mapper-wide version: other columns' writers must not collide.
    assert Conversion.__mapper__.version_id_col is None


def test_set_content_bumps_in_the_database(app, test_user):
    cid = _make_conversion(app, test_user['id'])
    assert _row(app, cid)['version'] == 1
    with app.app_context():
        conv = db.session.get(Conversion, cid)
        conv.set_content('# new\n')
        db.session.commit()
    row = _row(app, cid)
    assert row['content'] == '# new\n'
    assert row['version'] == 2


def test_only_content_writers_bump_the_version(app, authenticated_client, test_user,
                                               monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'])
    assert _row(app, cid)['version'] == 1

    assert authenticated_client.patch(f'/api/conversions/{cid}/content', headers=_auth(),
                                      json={'content': DOC}).status_code == 200
    assert _row(app, cid)['version'] == 2                      # docwrite full replace
    assert authenticated_client.put(f'/api/conversions/{cid}',
                                    json={'content': DOC}).status_code == 200
    assert _row(app, cid)['version'] == 3                      # editor PUT with content
    assert authenticated_client.put(f'/api/conversions/{cid}',
                                    json={'title': 'renamed'}).status_code == 200
    assert _row(app, cid)['version'] == 3                      # title-only: no bump
    assert authenticated_client.patch(f'/api/conversions/{cid}/progress',
                                      json={'percent': 40}).status_code == 200
    assert _row(app, cid)['version'] == 3                      # other column: no bump
    assert authenticated_client.patch(f'/api/conversions/{cid}/section', headers=_auth(),
                                      json={'heading': 'B', 'content': '# B\nx'}).status_code == 200
    assert _row(app, cid)['version'] == 4                      # section replace


def test_section_replace_still_bumps_updated_at(app, client, test_user, monkeypatch):
    # The conditional write is a Core UPDATE — the column onupdate must still fire.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'])
    old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=3)
    with app.app_context():
        db.session.execute(text('UPDATE conversion SET updated_at = :t WHERE id = :cid'),
                           {'t': old, 'cid': cid})
        db.session.commit()
    assert client.patch(f'/api/conversions/{cid}/section', headers=_auth(),
                        json={'heading': 'B', 'content': '# B\nx'}).status_code == 200
    assert datetime.fromisoformat(str(_row(app, cid)['updated_at'])) > old


# --- migration ----------------------------------------------------------------

def test_migration_adds_content_version_and_backfills_legacy_rows(app, test_user):
    cid = _make_conversion(app, test_user['id'])
    with app.app_context():
        engine = db.engine
        try:
            db.session.execute(text('ALTER TABLE conversion DROP COLUMN content_version'))
            db.session.commit()
        except OperationalError:
            db.session.rollback()
            pytest.skip('SQLite build without DROP COLUMN support')
        try:
            assert 'content_version' not in {
                c['name'] for c in inspect(engine).get_columns('conversion')}
            _run_pending_migrations(app)
            cols = [c['name'] for c in inspect(engine).get_columns('conversion')]
            assert cols.count('content_version') == 1
            assert db.session.execute(
                text('SELECT content_version FROM conversion WHERE id = :cid'),
                {'cid': cid}).scalar() == 1                    # legacy row starts at 1
            _run_pending_migrations(app)                       # idempotent
            cols = [c['name'] for c in inspect(engine).get_columns('conversion')]
            assert cols.count('content_version') == 1
        finally:
            cols = {c['name'] for c in inspect(engine).get_columns('conversion')}
            if 'content_version' not in cols:
                db.session.execute(text(
                    'ALTER TABLE conversion ADD COLUMN content_version INTEGER NOT NULL DEFAULT 1'))
                db.session.commit()


# --- the conflict path --------------------------------------------------------

def test_section_conflict_retries_into_the_other_writers_text(app, client, test_user,
                                                               monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'])
    other = '# A\nOTHER a {n}\n# B\nb\n# C\nc\n'
    hook = _Interposer(cid, conflicts=1, other_text=other)
    monkeypatch.setattr(docwrite_module, 'replace_section', hook)

    resp = client.patch(f'/api/conversions/{cid}/section', headers=_auth(),
                        json={'heading': 'B', 'content': '# B\nNEW b'})
    assert resp.status_code == 200

    # Spliced twice: first into the text as read, then — after the conflict —
    # into the OTHER writer's text. Both edits survive.
    assert hook.inputs == [DOC, other.format(n=1)]
    expected = '# A\nOTHER a 1\n# B\nNEW b\n# C\nc\n'
    row = _row(app, cid)
    assert row['content'] == expected
    assert row['version'] == 3                                 # theirs 1→2, ours 2→3
    assert resp.get_json()['content'] == expected


def test_section_conflict_exhausted_409_and_nothing_of_ours(app, client, test_user,
                                                             monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'])
    other = '# A\nOTHER a {n}\n# B\nb\n# C\nc\n'
    hook = _Interposer(cid, conflicts=CONTENT_WRITE_ATTEMPTS, other_text=other)
    monkeypatch.setattr(docwrite_module, 'replace_section', hook)

    resp = client.patch(f'/api/conversions/{cid}/section', headers=_auth(),
                        json={'heading': 'B', 'content': '# B\nNEW b'})
    assert resp.status_code == 409
    assert 'gleichzeitig geändert' in resp.get_json()['error']
    assert len(hook.inputs) == CONTENT_WRITE_ATTEMPTS          # bounded, not endless

    # The other writer's last text stands untouched; nothing of ours, nothing lost.
    row = _row(app, cid)
    assert row['content'] == other.format(n=CONTENT_WRITE_ATTEMPTS)
    assert row['version'] == 1 + CONTENT_WRITE_ATTEMPTS


def test_full_replace_under_a_concurrent_section_edit_wins_by_intent(app, client, test_user,
                                                                      monkeypatch):
    # PATCH /content is an unconditional overwrite — but it bumps the version,
    # so a section replace that read the OLD text cannot resurrect it.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'])
    replaced = '# Z\nfresh document\n# B\nb\n# C\nc\n'
    hook = _Interposer(cid, conflicts=1, other_text=replaced)
    monkeypatch.setattr(docwrite_module, 'replace_section', hook)

    resp = client.patch(f'/api/conversions/{cid}/section', headers=_auth(),
                        json={'heading': 'B', 'content': '# B\nNEW b'})
    assert resp.status_code == 200
    # spliced into the NEW document, the old one is not resurrected
    assert _row(app, cid)['content'] == '# Z\nfresh document\n# B\nNEW b\n# C\nc\n'
