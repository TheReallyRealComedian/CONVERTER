"""SYNC-FREEZE: SQLite runtime settings for several gunicorn processes.

``app_pkg/__init__.py`` switches every SQLite connection to WAL with an
explicit ``busy_timeout`` and serialises the schema bootstrap with a file
lock; the Dockerfile runs N worker processes. These tests pin the three
decisions so a later "cleanup" cannot silently fall back to one process on a
rollback journal — or to N processes without WAL.
"""
import ast
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import text

from app_pkg import _startup_lock, _startup_lock_path, create_app
from app_pkg.config import SQLITE_BUSY_TIMEOUT_SECONDS
from app_pkg.db_runtime import _ensure_sqlite_dir
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


# --- ARCH-FACTORY: the data directory follows the DB URI, not a literal -----

def test_sqlite_directory_is_derived_from_the_uri(tmp_path, monkeypatch):
    """``create_app`` used to ``os.makedirs('/app/data')`` whatever the URI
    said — a container path in the factory that conftest.py had to no-op on
    every dev box. Now the directory of the SQLite file comes from the same
    derivation as the startup lock, and the factory creates exactly that."""
    nested = tmp_path / 'a' / 'b' / 'converter.db'
    assert _ensure_sqlite_dir(f'sqlite:///{nested}') == str(nested.parent)
    assert nested.parent.is_dir()
    assert _ensure_sqlite_dir(f'sqlite:///{nested}') == str(nested.parent)  # idempotent
    for uri in ('sqlite:///:memory:', 'sqlite://',
                'sqlite:///file:mem?mode=memory&uri=true',
                'postgresql://u:p@h/db', 'not a url',
                'sqlite:///converter.db'):  # bare name: the cwd, nothing to create
        assert _ensure_sqlite_dir(uri) is None, uri

    # The factory on a directory that does not exist yet: it is created from
    # the URI and the bootstrap runs in it.
    db_file = tmp_path / 'fresh' / 'data' / 'converter.db'
    assert not db_file.parent.exists()
    monkeypatch.setenv('DATABASE_URL', f'sqlite:///{db_file}')
    boot_app = create_app()
    try:
        assert db_file.parent.is_dir() and db_file.is_file()
    finally:
        with boot_app.app_context():
            db.session.remove()
            db.engine.dispose()


def _code_strings_and_makedirs(source):
    """String constants in CODE (docstrings excluded — comments are not in
    the AST at all) and the lines of ``os.makedirs(...)`` calls."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    strings = [(n.lineno, n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and id(n) not in docstrings]
    makedirs = [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'makedirs'
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'os']
    return strings, makedirs


def test_no_app_data_literal_outside_the_default_uri():
    """Sentinel: ``/app/data`` appears in the CODE under app_pkg/ only inside
    the default DB URI (the one place that says where the file lives), never
    as a path of its own — and ``os.makedirs`` is called in db_runtime only.
    Docstrings and comments may name the path (they explain the history)."""
    package = REPO / 'app_pkg'
    if not package.is_dir():
        pytest.skip('app_pkg not shipped alongside the tests')
    literals, makedirs = [], []
    for path in sorted(package.rglob('*.py')):
        strings, calls = _code_strings_and_makedirs(path.read_text())
        rel = path.relative_to(REPO)
        literals += [f'{rel}:{line}: {value!r}' for line, value in strings
                     if '/app/data' in value and not value.startswith('sqlite:////app/data/')]
        makedirs += [f'{rel}:{line}' for line in calls]
    assert literals == []
    assert [m.split(':')[0] for m in makedirs] == ['app_pkg/db_runtime.py'], makedirs


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
