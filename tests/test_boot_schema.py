"""ARCH-FACTORY P0 — the gate before the cut: a boot test from a frozen legacy DDL.

``create_app()`` is the only place the schema bootstrap lives: ``create_all``
(new tables), the inline ALTERs (new columns on old tables), the CSV→junction
data migration, all under the startup lock. The suite exercised the pieces
(``_run_pending_migrations`` on a table with a dropped column) but never the
chain — the architecture audit (W-7, FACTORY-8) showed three mutations of
``create_app`` surviving 1341 tests: migration call gone, CSV call gone,
startup lock gone. Moving the factory out of the package init without a test
for that chain would be a move without a net.

The net: ``tests/fixtures/legacy_schema_5b33f75.sql`` is the schema of the
last ``models.py`` WITHOUT a migration (commit ``5b33f75^``, 2026-05-25, three
tables, five indexes), generated once and frozen. The tests here load it into
a fresh file, point ``DATABASE_URL`` at it, run the real ``create_app()`` and
compare the result with what ``create_all()`` declares today.

Coverage of this basis, measured: of the 12 migrated columns, SIX travel the
ALTER path (``highlight.note``, the four ``conversion`` columns,
``user.settings_json`` — their tables exist in the fixture); the other six
(``tag.parent_id``, four ``card`` columns, ``review.version``) arrive with
their tables through ``create_all`` and the ALTER guards find them present.
Of the three indexes the Prod file lacks, two are reproduced from this basis
(``ix_conversion_lifecycle_status``, ``ix_conversion_queue_position``);
``ix_tag_parent_id`` is Prod history (``tag`` created at R2-A, column added
at LERN-GROUP) — the generic index check below catches any missing index
regardless of how a file got there.

Why in-process and not a subprocess: the lock-order test needs to see the
fake lock and the real ``create_all`` in one call; a second Flask app on the
same ``db`` extension is what Flask-SQLAlchemy 3 supports (engines are
per-app, sessions per app context), and each booted app disposes its engine.
"""
import logging
import re
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

import flask_sqlalchemy
from sqlalchemy import Index, create_engine, inspect, text

from app_pkg import _run_pending_migrations, create_app
from models import Conversion, db

LEGACY_DDL = Path(__file__).parent / 'fixtures' / 'legacy_schema_5b33f75.sql'

# The twelve columns the inline migrations add, (table, column) — one line per
# ALTER in _run_pending_migrations, in its order.
MIGRATED_COLUMNS = [
    ('highlight', 'note'),
    ('conversion', 'last_read_percent'),
    ('conversion', 'lifecycle_status'),
    ('conversion', 'queue_position'),
    ('conversion', 'content_version'),
    ('user', 'settings_json'),
    ('tag', 'parent_id'),
    ('card', 'front_svg'),
    ('card', 'back_svg'),
    ('card', 'context_conversion_id'),
    ('card', 'context_heading'),
    ('review', 'version'),
]

# Log lines the bootstrap writes when it CHANGES something. A second boot on
# the same file must write none of them.
_CHANGE_LOG_RE = re.compile(
    r'added via ALTER TABLE|added \+ backfilled|index \S+ created|R2-A: migrated')


def _load_legacy(db_file):
    """A legacy-schema file with legacy data: one user, one conversion that
    still carries its tags as CSV (the R2-A migration's input), one without."""
    con = sqlite3.connect(db_file)
    try:
        con.executescript(LEGACY_DDL.read_text())
        con.execute("INSERT INTO user (id, username, password_hash, created_at) "
                    "VALUES (1, 'legacy', 'not-a-real-hash', '2026-05-25 16:30:40.000000')")
        con.execute("INSERT INTO conversion (id, user_id, conversion_type, title, content, "
                    "tags, is_favorite, created_at, updated_at) VALUES "
                    "(1, 1, 'markdown_input', 'Tagged', 'body', 'Alpha, beta ,alpha', 0, "
                    "'2026-05-25 16:31:00.000000', '2026-05-25 16:31:00.000000')")
        con.execute("INSERT INTO conversion (id, user_id, conversion_type, title, content, "
                    "tags, is_favorite, created_at, updated_at) VALUES "
                    "(2, 1, 'ai_newsletter', 'Untagged', 'body', '', 0, "
                    "'2026-05-25 16:32:00.000000', '2026-05-25 16:32:00.000000')")
        con.execute("INSERT INTO highlight (id, conversion_id, exact, prefix, suffix, "
                    "created_at, updated_at) VALUES (1, 1, 'body', '', '', "
                    "'2026-05-25 16:33:00.000000', '2026-05-25 16:33:00.000000')")
        con.commit()
    finally:
        con.close()


def _schema_snapshot(db_file):
    """Every object in sqlite_master with its DDL — the file's schema as text."""
    con = sqlite3.connect(db_file)
    try:
        return set(con.execute(
            'SELECT type, name, tbl_name, sql FROM sqlite_master').fetchall())
    finally:
        con.close()


def _declared_indexes():
    """What create_all() yields on an EMPTY database: {table: [index names]}.
    The reference is measured, not derived from Table.indexes, so it is
    exactly the set a fresh container file would carry."""
    engine = create_engine('sqlite://')
    try:
        db.metadata.create_all(engine)
        insp = inspect(engine)
        return {name: sorted(ix['name'] for ix in insp.get_indexes(name))
                for name in db.metadata.tables}
    finally:
        engine.dispose()


def _live_indexes(insp):
    return {name: sorted(ix['name'] for ix in insp.get_indexes(name))
            for name in db.metadata.tables if name in insp.get_table_names()}


@contextmanager
def _booted(monkeypatch, db_file):
    """The real factory on ``db_file``; the engine is disposed afterwards so
    the WAL file closes and nothing leaks into the session-scoped app."""
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_file}')
    boot_app = create_app()
    try:
        yield boot_app
    finally:
        with boot_app.app_context():
            db.session.remove()
            db.engine.dispose()


# --- the chain: legacy file → create_app() → declared schema ------------------

def test_legacy_db_boots_to_the_declared_schema(tmp_path, monkeypatch):
    db_file = tmp_path / 'legacy.db'
    _load_legacy(db_file)

    with _booted(monkeypatch, db_file) as boot_app, boot_app.app_context():
        insp = inspect(db.engine)

        # Every table the model declares exists (create_all's part).
        assert set(db.metadata.tables) <= set(insp.get_table_names())

        # Every migrated column exists (the ALTERs' part).
        missing = [(t, c) for t, c in MIGRATED_COLUMNS
                   if c not in {col['name'] for col in insp.get_columns(t)}]
        assert missing == [], f'columns missing after boot: {missing}'

        # The index set is EXACTLY the one create_all yields on an empty file —
        # per table, by name. Before the index reconcile this failed on
        # ix_conversion_lifecycle_status and ix_conversion_queue_position.
        assert _live_indexes(insp) == _declared_indexes()

        # The migrated lifecycle_status carries the model's default. The model
        # has only a Python-side default ('inbox', no server_default), so a
        # fresh create_all file reflects None here and the migrated file
        # DEFAULT 'inbox' — the VALUE is what has to agree.
        reflected = next(c for c in insp.get_columns('conversion')
                         if c['name'] == 'lifecycle_status')['default']
        assert reflected.strip("'") == Conversion.__table__.c.lifecycle_status.default.arg

        # The R2-A data migration ran: the CSV row is in the junction, drained.
        junction = db.session.execute(text(
            'SELECT t.name FROM conversion_tags ct JOIN tag t ON t.id = ct.tag_id '
            'WHERE ct.conversion_id = 1 ORDER BY t.name')).scalars().all()
        assert junction == ['alpha', 'beta']
        assert db.session.execute(text(
            'SELECT tags FROM conversion WHERE id = 1')).scalar() == ''
        # … and the lifecycle backfill differentiated (R2-C): newsletter → inbox.
        statuses = dict(db.session.execute(text(
            'SELECT id, lifecycle_status FROM conversion')).fetchall())
        assert statuses == {1: 'archive', 2: 'inbox'}


def test_second_boot_on_the_same_file_changes_nothing(tmp_path, monkeypatch, caplog):
    db_file = tmp_path / 'legacy.db'
    _load_legacy(db_file)
    with _booted(monkeypatch, db_file):
        pass
    before = _schema_snapshot(db_file)
    assert any(name == 'ix_conversion_queue_position' for _, name, _, _ in before)

    caplog.set_level(logging.INFO)
    caplog.clear()
    with _booted(monkeypatch, db_file) as boot_app, boot_app.app_context():
        rows = db.session.execute(text(
            'SELECT (SELECT count(*) FROM conversion_tags), '
            '(SELECT count(*) FROM tag), (SELECT tags FROM conversion WHERE id = 1)')).one()
    assert tuple(rows) == (2, 2, '')
    assert _schema_snapshot(db_file) == before
    changes = [m for m in caplog.messages if _CHANGE_LOG_RE.search(m)]
    assert changes == [], changes


# --- the lock: entered with the DB URI, before create_all, released after -----

def test_create_app_enters_the_startup_lock_around_the_bootstrap(tmp_path, monkeypatch):
    """The lock changes no schema, so the chain test cannot see it. This one
    fakes it where the factory looks it up and records the order of events;
    without the ``with _startup_lock(...)`` the list has no lock entries."""
    factory = sys.modules[create_app.__module__]
    events = []

    @contextmanager
    def fake_lock(uri):
        events.append(('lock', uri))
        yield
        events.append(('unlock', uri))

    real_create_all = flask_sqlalchemy.SQLAlchemy.create_all
    real_migrate = factory._run_pending_migrations

    def create_all(self, *args, **kwargs):
        events.append('create_all')
        return real_create_all(self, *args, **kwargs)

    def migrate(app):
        events.append('migrations')
        return real_migrate(app)

    monkeypatch.setattr(factory, '_startup_lock', fake_lock)
    monkeypatch.setattr(factory, '_run_pending_migrations', migrate)
    monkeypatch.setattr(flask_sqlalchemy.SQLAlchemy, 'create_all', create_all)

    db_file = tmp_path / 'fresh.db'
    with _booted(monkeypatch, db_file):
        pass

    uri = f'sqlite:///{db_file}'
    assert events == [('lock', uri), 'create_all', 'migrations', ('unlock', uri)]


# --- the index reconcile on its own ------------------------------------------

def test_missing_model_index_is_recreated_by_the_migrations(app, caplog):
    caplog.set_level(logging.INFO)
    with app.app_context():
        db.session.execute(text('DROP INDEX IF EXISTS ix_conversion_queue_position'))
        db.session.commit()
        assert not any(ix['name'] == 'ix_conversion_queue_position'
                       for ix in inspect(db.engine).get_indexes('conversion'))
        _run_pending_migrations(app)
        live = {ix['name']: ix['column_names']
                for ix in inspect(db.engine).get_indexes('conversion')}
    assert live['ix_conversion_queue_position'] == ['queue_position']
    assert any('ix_conversion_queue_position' in m and 'created' in m
               for m in caplog.messages), caplog.messages


def test_missing_unique_index_is_refused_and_logged(app, caplog):
    """A UNIQUE index that an existing table lacks is a correctness question
    over its rows, not a migration step: the reconcile logs it and leaves it.
    The model's one real UNIQUE index (``ix_review_card_id``) was created with
    its table and is present — the positive control that the refusal is about
    MISSING unique indexes only."""
    caplog.set_level(logging.INFO)
    table = Conversion.__table__
    probe = Index('ix_probe_unique_conversion_title', table.c.title, unique=True)
    try:
        assert probe in table.indexes  # attached by construction
        with app.app_context():
            _run_pending_migrations(app)
            insp = inspect(db.engine)
            names = {ix['name'] for ix in insp.get_indexes('conversion')}
            assert 'ix_probe_unique_conversion_title' not in names
            review = {ix['name']: ix for ix in insp.get_indexes('review')}
        assert review['ix_review_card_id']['unique']
    finally:
        table.indexes.discard(probe)
    refusals = [r for r in caplog.records
                if r.levelno == logging.WARNING
                and 'ix_probe_unique_conversion_title' in r.getMessage()
                and 'UNIQUE' in r.getMessage()]
    assert len(refusals) == 1, caplog.messages
