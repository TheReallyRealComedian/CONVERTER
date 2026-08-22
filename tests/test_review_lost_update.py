"""LOST-UPDATE — the Review row is guarded by optimistic locking.

SYNC-FREEZE P2 / LOST-UPDATE P1 measured ``api_review_card`` losing 12 % of
concurrent same-card ratings at 3,200 × HTTP 200 (an unguarded
read-modify-write over 2 processes × 8 threads). These tests pin the fix:

* ``Review.version`` IS the mapper's ``version_id_col`` (sentinel — a dropped
  mapper arg would bring the silent loss back with every other test green),
* the inline migration adds the column to a legacy ``review`` table and
  backfills existing rows with 1 (the value SQLAlchemy assigns on INSERT),
  idempotently,
* a version conflict at commit makes the endpoint re-read and apply the
  rating to the OTHER writer's result (reps/history carry BOTH ratings),
* after ``REVIEW_WRITE_ATTEMPTS`` conflicts the endpoint answers 409 with a
  German message and has written NOTHING of its own — never a silent loss,
* the card DELETE (cascade onto the versioned row) survives a rating that
  lands in between.

The concurrent writer is real, not mocked: a raw UPDATE through a SECOND
engine connection in its own committed transaction, injected between the
request's read and its write — so the conditional UPDATE genuinely misses
its row, exactly as another process or thread would make it miss.
"""
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import event, inspect, text
from sqlalchemy.exc import OperationalError

import app_pkg.cards as cards_module
from app_pkg import _run_pending_migrations
from app_pkg.cards import REVIEW_WRITE_ATTEMPTS
from models import Card, Review, db


def _make_card(app, user_id, ctype='atomic'):
    with app.app_context():
        card = Card(user_id=user_id, type=ctype, front='Q', back='A')
        card.review = Review(due=datetime.now(timezone.utc))
        db.session.add(card)
        db.session.commit()
        return card.id


def _review_row(app, card_id):
    """The row as SQLite holds it — raw, outside any ORM identity map."""
    with app.app_context():
        row = db.session.execute(text(
            'SELECT version, reps, rating_history FROM review WHERE card_id = :cid'),
            {'cid': card_id}).one()
    history = json.loads(row[2]) if row[2] else []
    return {'version': row[0], 'reps': row[1], 'history': history}


def _other_writer_rates(card_id, rating='hard'):
    """A concurrent rating of the same card: raw UPDATE on a SECOND pooled
    connection in its own committed transaction — what another process or
    thread does, seen from the running request's point of view."""
    with db.engine.begin() as conn:
        current = conn.execute(text(
            'SELECT rating_history FROM review WHERE card_id = :cid'),
            {'cid': card_id}).scalar()
        history = json.loads(current) if current else []
        history.append({'rating': rating,
                        'reviewed_at': datetime.now(timezone.utc).isoformat()})
        conn.execute(text(
            'UPDATE review SET version = version + 1, reps = reps + 1, '
            'rating_history = :h WHERE card_id = :cid'),
            {'h': json.dumps(history), 'cid': card_id})


class _ConflictingScheduler:
    """Wraps the real scheduler. The first ``conflicts`` calls of
    ``apply_rating`` let another writer rate the card first — i.e. between
    this request's read and its write — and every call records the state it
    was handed, so a test can see WHICH state the retry applied to."""

    def __init__(self, inner, card_id, conflicts):
        self.inner, self.card_id, self.conflicts = inner, card_id, conflicts
        self.states_seen = []

    def new_card_state(self):
        return self.inner.new_card_state()

    def apply_rating(self, state, rating):
        self.states_seen.append(dict(state))
        if len(self.states_seen) <= self.conflicts:
            _other_writer_rates(self.card_id)
        return self.inner.apply_rating(state, rating)


# --- sentinel + plain path ----------------------------------------------------

def test_review_version_is_the_mapper_version_id_col():
    col = Review.__table__.c.version
    assert Review.__mapper__.version_id_col is col
    assert col.nullable is False
    assert str(col.server_default.arg) == '1'


def test_each_rating_bumps_the_version(app, authenticated_client, test_user):
    cid = _make_card(app, test_user['id'])
    assert _review_row(app, cid)['version'] == 1
    for _ in range(2):
        assert authenticated_client.post(
            f'/api/cards/{cid}/review', json={'rating': 'good'}).status_code == 200
    row = _review_row(app, cid)
    assert row['version'] == 3
    assert row['reps'] == 2
    assert len(row['history']) == 2


# --- migration ----------------------------------------------------------------

def test_migration_adds_review_version_and_backfills_legacy_rows(app, test_user):
    cid = _make_card(app, test_user['id'])
    with app.app_context():
        engine = db.engine
        # Simulate a pre-LOST-UPDATE schema: drop the column so the migration
        # must re-add it. SQLite < 3.35 has no DROP COLUMN — skip there.
        try:
            db.session.execute(text('ALTER TABLE review DROP COLUMN version'))
            db.session.commit()
        except OperationalError:
            db.session.rollback()
            pytest.skip('SQLite build without DROP COLUMN support')

        try:
            assert 'version' not in {c['name'] for c in inspect(engine).get_columns('review')}

            _run_pending_migrations(app)
            cols = [c['name'] for c in inspect(engine).get_columns('review')]
            assert cols.count('version') == 1
            # The legacy row starts at 1 — the value a fresh INSERT gets, so the
            # first conditional UPDATE against it matches.
            assert db.session.execute(
                text('SELECT version FROM review WHERE card_id = :cid'),
                {'cid': cid}).scalar() == 1

            _run_pending_migrations(app)  # second pass is a no-op
            cols = [c['name'] for c in inspect(engine).get_columns('review')]
            assert cols.count('version') == 1
        finally:
            cols = {c['name'] for c in inspect(engine).get_columns('review')}
            if 'version' not in cols:
                db.session.execute(text(
                    'ALTER TABLE review ADD COLUMN version INTEGER NOT NULL DEFAULT 1'))
                db.session.commit()


# --- the conflict path --------------------------------------------------------

def test_conflict_retries_on_the_other_writers_result(app, authenticated_client, test_user,
                                                      monkeypatch):
    cid = _make_card(app, test_user['id'])
    hooked = _ConflictingScheduler(cards_module.get_scheduler(), cid, conflicts=1)
    monkeypatch.setattr(cards_module, 'get_scheduler', lambda: hooked)

    resp = authenticated_client.post(f'/api/cards/{cid}/review', json={'rating': 'good'})
    assert resp.status_code == 200

    # Two applications: the first on the state as loaded (reps 0), the second
    # — after the conflict — on the OTHER writer's result (reps 1). That is the
    # sequential outcome; the retry is the answer, not a workaround.
    assert [s['reps'] for s in hooked.states_seen] == [0, 1]

    row = _review_row(app, cid)
    assert row['reps'] == 2                                             # both ratings counted
    assert [h['rating'] for h in row['history']] == ['hard', 'good']   # theirs first, ours appended
    assert row['version'] == 3                                          # 1→2 (theirs), 2→3 (ours)
    assert resp.get_json()['review']['reps'] == 2


def test_exhausted_retries_answer_409_and_write_nothing(app, authenticated_client, test_user,
                                                         monkeypatch):
    cid = _make_card(app, test_user['id'])
    hooked = _ConflictingScheduler(cards_module.get_scheduler(), cid,
                                   conflicts=REVIEW_WRITE_ATTEMPTS)
    monkeypatch.setattr(cards_module, 'get_scheduler', lambda: hooked)

    resp = authenticated_client.post(f'/api/cards/{cid}/review', json={'rating': 'good'})
    assert resp.status_code == 409
    assert 'gleichzeitig bewertet' in resp.get_json()['error']
    assert len(hooked.states_seen) == REVIEW_WRITE_ATTEMPTS   # bounded, not endless

    # Only the other writer's ratings are on the row: nothing of ours (the
    # user was told), nothing of theirs lost.
    row = _review_row(app, cid)
    assert row['reps'] == REVIEW_WRITE_ATTEMPTS
    assert [h['rating'] for h in row['history']] == ['hard'] * REVIEW_WRITE_ATTEMPTS
    assert row['version'] == 1 + REVIEW_WRITE_ATTEMPTS


def test_delete_survives_a_rating_that_lands_in_between(app, authenticated_client, test_user):
    cid = _make_card(app, test_user['id'])
    fired = []

    def other_writer_before_flush(session, flush_context, instances):
        # Fires inside the DELETE request's flush, after its load and before
        # its DELETE statements — the other writer's rating lands exactly there.
        if not fired:
            fired.append(True)
            _other_writer_rates(cid)

    event.listen(db.session, 'before_flush', other_writer_before_flush)
    try:
        resp = authenticated_client.delete(f'/api/cards/{cid}')
    finally:
        event.remove(db.session, 'before_flush', other_writer_before_flush)

    assert resp.status_code == 200
    assert fired  # the conflict really happened
    with app.app_context():
        assert db.session.get(Card, cid) is None
        assert db.session.execute(
            text('SELECT count(*) FROM review WHERE card_id = :cid'),
            {'cid': cid}).scalar() == 0
