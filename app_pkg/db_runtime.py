"""SQLite runtime of the web app: WAL + ``busy_timeout`` on every connection,
the startup lock around the schema bootstrap, and the data directory — the
last two from ONE derivation of the DB URI (``_sqlite_file``).

Moved out of ``app_pkg/__init__.py`` in ARCH-FACTORY P1 (function bodies
byte-equal, sha256 table in the sprint report), with two additions of that
sprint: ``_sqlite_file`` is the shared URI→path derivation (the lock path
used to carry it alone), and ``_ensure_sqlite_dir`` replaces the factory's
hard-coded ``os.makedirs('/app/data')`` — the directory is created where the
URI says the file lives, and nowhere else. Pinned by tests/test_db_runtime.py.
"""
import fcntl
import os
from contextlib import contextmanager

from sqlalchemy import event
from sqlalchemy.engine import make_url

from app_pkg.config import SQLITE_BUSY_TIMEOUT_SECONDS
from models import db


def _register_sqlite_pragmas(app):
    """SYNC-FREEZE: WAL + an explicit ``busy_timeout`` on every SQLite connection.

    The app runs as several gunicorn processes on ONE SQLite file. In the
    rollback-journal mode the file was in (``PRAGMA journal_mode`` = ``delete``,
    never configured — ``SQLALCHEMY_ENGINE_OPTIONS`` was empty) a single
    writer locks the whole database against every reader; with one process
    that never showed, with N it would have turned a freeze into
    ``database is locked``. Hence WAL *before* workers (locked decision 1).

    Connection-level pragmas, so the SQLAlchemy ``connect`` event is the one
    place that reaches every pooled connection of every process. Values and
    reasoning: ``app_pkg.config.SQLITE_BUSY_TIMEOUT_SECONDS``.
    """
    with app.app_context():
        engine = db.engine
    if engine.dialect.name != 'sqlite':
        return

    @event.listens_for(engine, 'connect')
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            # busy_timeout FIRST: should the WAL switch itself have to wait
            # for a lock (first boot on a rollback-journal file while another
            # connection reads), it waits instead of failing.
            cursor.execute(
                f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_SECONDS * 1000}')
            cursor.execute('PRAGMA journal_mode=WAL')
        finally:
            cursor.close()


def _sqlite_file(uri):
    """Path of a file-based SQLite database, ``None`` for everything else
    (in-memory SQLite, other backends, unparsable URIs) — the ONE derivation
    behind the startup lock and the data directory."""
    try:
        url = make_url(uri)
    except Exception:
        return None
    if not url.drivername.startswith('sqlite'):
        return None
    database = url.database or ''
    if database in ('', ':memory:') or url.query.get('mode') == 'memory':
        return None
    return database


def _startup_lock_path(uri):
    """Lock file beside a file-based SQLite database, ``None`` otherwise."""
    database = _sqlite_file(uri)
    if database is None:
        return None
    return f'{database}.startup.lock'


def _ensure_sqlite_dir(uri):
    """Create the directory of a file-based SQLite database if it is missing
    (``os.makedirs(exist_ok=True)``) and return it; ``None`` when there is
    nothing to create — no file (in-memory, other backend) or a bare file
    name in the working directory.

    ARCH-FACTORY: ``create_app`` used to ``os.makedirs('/app/data')`` whatever
    the URI said — a container path hard-coded in the factory, which every
    dev box had to patch away (conftest.py turned ``os.makedirs`` into a
    no-op for ``/app/*``). The directory now follows the URI: in the
    container that is still ``/app/data`` (the ``app_data`` volume), in a
    test whatever ``DATABASE_URL`` points at.
    """
    database = _sqlite_file(uri)
    if database is None:
        return None
    directory = os.path.dirname(database)
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    return directory


@contextmanager
def _startup_lock(uri):
    """SYNC-FREEZE: serialise the schema bootstrap across worker processes.

    Every gunicorn worker runs ``create_app()`` — and with it
    ``db.create_all()`` and ``_run_pending_migrations`` — on its own. No
    ``--preload``: the per-process import is the model the app has always
    run under. (The gRPC channel in ``GoogleTTSService`` that made a fork
    after import unsafe left the web process with ARCH-NARR5; ``--preload``
    has not been re-examined since — the Deepgram client and the Redis
    connection are still built at import.)
    Both bootstrap steps are idempotent on a settled schema, but on the first
    boot after a schema change N processes would race check-then-ALTER: the
    loser dies on ``duplicate column`` / ``table already exists``, and
    gunicorn treats a worker that fails to boot as fatal for the whole server
    (``Arbiter.reap_workers`` → ``HaltServer``). An ``flock`` beside the
    database file makes the bootstrap strictly sequential — the first process
    migrates, the others find the schema complete. Databases without a file
    (in-memory) need no lock.
    """
    path = _startup_lock_path(uri)
    if path is None:
        yield
        return
    with open(path, 'a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
