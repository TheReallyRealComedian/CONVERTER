"""SYNC-FREEZE: SQLite runtime settings for several gunicorn processes.

``app_pkg/__init__.py`` switches every SQLite connection to WAL with an
explicit ``busy_timeout`` and serialises the schema bootstrap with a file
lock; the Dockerfile runs N worker processes. These tests pin the three
decisions so a later "cleanup" cannot silently fall back to one process on a
rollback journal — or to N processes without WAL.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import text

from app_pkg import _startup_lock, _startup_lock_path
from app_pkg.config import SQLITE_BUSY_TIMEOUT_SECONDS
from models import db

REPO = Path(__file__).resolve().parents[1]


def test_sqlite_connections_run_in_wal_mode(app):
    # The test database is a file (conftest), so the pragma is real here —
    # an in-memory database would answer 'memory' and prove nothing.
    with app.app_context():
        assert db.session.execute(text('PRAGMA journal_mode')).scalar() == 'wal'


def test_sqlite_busy_timeout_is_explicit(app):
    with app.app_context():
        assert (db.session.execute(text('PRAGMA busy_timeout')).scalar()
                == SQLITE_BUSY_TIMEOUT_SECONDS * 1000)
    # pysqlite's own default is 5 s; the point of the constant is that the
    # value is a decision, not an inheritance.
    assert SQLITE_BUSY_TIMEOUT_SECONDS != 5


def test_startup_lock_path_only_for_file_databases():
    assert (_startup_lock_path('sqlite:////app/data/converter.db')
            == '/app/data/converter.db.startup.lock')
    assert _startup_lock_path('sqlite:///:memory:') is None
    assert _startup_lock_path('sqlite://') is None
    assert _startup_lock_path('sqlite:///file:mem?mode=memory&uri=true') is None
    assert _startup_lock_path('postgresql://u:p@h/db') is None
    assert _startup_lock_path('not a url') is None


def test_startup_lock_waits_for_another_process(tmp_path):
    """Two processes, one lock: the second ``create_app`` bootstrap must wait
    until the first has finished migrating, never race it."""
    lock_db = tmp_path / 'other.db'
    lock_file = f'{lock_db}.startup.lock'
    holder = subprocess.Popen([sys.executable, '-c', (
        'import fcntl, time\n'
        f'fh = open({lock_file!r}, "a+")\n'
        'fcntl.flock(fh, fcntl.LOCK_EX)\n'
        'print("locked", flush=True)\n'
        'time.sleep(0.8)\n'
    )], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'locked'
        t0 = time.monotonic()
        with _startup_lock(f'sqlite:///{lock_db}'):
            waited = time.monotonic() - t0
    finally:
        holder.wait(timeout=10)
    assert waited >= 0.4, f'lock did not wait for the holder ({waited:.2f} s)'


def test_startup_lock_is_a_no_op_without_a_file():
    with _startup_lock('sqlite:///:memory:'):
        pass  # must neither fail nor create anything


def test_dockerfile_runs_several_worker_processes_without_preload():
    """The process count is the fix for the single-thread serialisation; a
    --preload would share non-fork-safe SDK clients across the processes."""
    dockerfile = REPO / 'Dockerfile'
    if not dockerfile.exists():
        pytest.skip('Dockerfile not shipped alongside the tests')
    cmd = [line for line in dockerfile.read_text().splitlines()
           if line.startswith('CMD ')][-1]
    match = re.search(r'"--workers",\s*"(\d+)"', cmd)
    assert match, cmd
    assert int(match.group(1)) >= 2, cmd
    assert '--preload' not in cmd
    assert 'uvicorn.workers.UvicornWorker' in cmd
