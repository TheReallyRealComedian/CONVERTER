"""Test fixtures and stubs for the Flask test client.

The tests in this directory exercise the application through the public HTTP
boundary (``app.test_client()``).  External SDK clients (Deepgram, Google
Cloud TTS, the genai client of the Cloud-PDF path) are mocked at the place
they are *instantiated* — never inside the service implementation — so the
mocks survive internal refactors.

Two pieces of test-only setup happen at import time, *before* ``app`` is
imported, because both happen during ``app.py`` module load:

1. ``unstructured.partition.auto`` and ``playwright.async_api`` are stubbed
   in ``sys.modules``.  These are heavy production dependencies; the
   characterization tests mock them on the stub module / at the
   ``app.async_playwright`` boundary anyway (since DOC-WEB the router
   lazy-imports ``partition`` at call time — there is no ``app.partition``
   singleton anymore), so a lightweight stub is sufficient and keeps the
   dev-machine install footprint small.
2. ``os.makedirs`` is wrapped to no-op for ``/app/*`` paths so the
   container-internal ``os.makedirs('/app/data', exist_ok=True)`` line
   does not fail on macOS / Linux dev boxes.
"""
import atexit
import json
import os
import re
import subprocess
import sys
import threading
import types
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# --- Stubs for heavy production deps (must run before `import app`) ---

def _install_module_stubs():
    if 'unstructured.partition.auto' not in sys.modules:
        unstructured = types.ModuleType('unstructured')
        partition_pkg = types.ModuleType('unstructured.partition')
        partition_auto = types.ModuleType('unstructured.partition.auto')
        partition_auto.partition = lambda **_kwargs: []
        sys.modules['unstructured'] = unstructured
        sys.modules['unstructured.partition'] = partition_pkg
        sys.modules['unstructured.partition.auto'] = partition_auto

    if 'playwright.async_api' not in sys.modules:
        playwright_pkg = types.ModuleType('playwright')
        playwright_async = types.ModuleType('playwright.async_api')
        playwright_async.async_playwright = MagicMock()
        sys.modules['playwright'] = playwright_pkg
        sys.modules['playwright.async_api'] = playwright_async


_install_module_stubs()


# --- Env required by `import app` ---

# ARCH-BUILD: one database file PER PROCESS. The fixed name
# '<tempdir>/converter-test.db' made two concurrent pytest runs on one
# machine destroy each other's fixtures (measured 2026-10-05: 279 passed /
# 1359 errors in parallel, 1639 + 1 alone). The pid in the name keeps the
# runs apart; leftovers of a crashed run are swept only once their process
# is gone, the own file is removed at exit.
_TEST_DB_DIR = Path(tempfile.gettempdir())
_TEST_DB_FILE = _TEST_DB_DIR / f'converter-test-{os.getpid()}.db'


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, someone else's
    return True


def _remove_test_db(path):
    # SYNC-FREEZE: the app switches SQLite to WAL, which keeps '-wal'/'-shm'
    # side files next to the database — drop them with it so a previous run's
    # journal can never be replayed into a fresh test database. The schema
    # bootstrap's flock file ('.startup.lock', app_pkg/__init__.py) sits
    # there too.
    for stale in (Path(path), Path(f'{path}-wal'), Path(f'{path}-shm'),
                  Path(f'{path}.startup.lock')):
        if stale.exists():
            stale.unlink()


_LEFTOVER_RE = re.compile(r'converter-test-(\d+)\.db(?:$|[.-])')
for _leftover in _TEST_DB_DIR.glob('converter-test-*.db*'):
    _match = _LEFTOVER_RE.match(_leftover.name)
    if _match and int(_match.group(1)) != os.getpid() and not _pid_alive(int(_match.group(1))):
        _leftover.unlink()
_remove_test_db(_TEST_DB_DIR / 'converter-test.db')  # the pre-ARCH-BUILD fixed name
_remove_test_db(_TEST_DB_FILE)
atexit.register(_remove_test_db, _TEST_DB_FILE)

os.environ.setdefault('SECRET_KEY', 'test-secret-key')
os.environ.setdefault('DATABASE_URL', f'sqlite:///{_TEST_DB_FILE}')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/0')
os.environ.setdefault('NOTION_MCP_URL', 'http://notion-mcp.test')
os.environ.setdefault('MCP_AUTH_TOKEN', 'test-mcp-token')
os.environ.setdefault('NOTION_TOKEN', '')
# SEC-SOCKET: the worker asks the mineru launcher over HTTP. No test may
# resolve the compose hostname ``mineru-launcher`` (a DNS wait, not a clean
# failure) — an unmocked local run hits a closed loopback port instead and
# fails fast with "connection refused". ``fake_launcher`` overrides it.
os.environ.setdefault('MINERU_LAUNCHER_URL', 'http://127.0.0.1:9')

# Production code does `os.makedirs('/app/data', exist_ok=True)` at module
# load — silently no-op on the dev box where /app is not writable.
_real_makedirs = os.makedirs


def _safe_makedirs(path, *args, **kwargs):
    if str(path).startswith('/app'):
        return
    return _real_makedirs(path, *args, **kwargs)


os.makedirs = _safe_makedirs


# --- Now safe to import the app module ---

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as app_module  # noqa: E402
from models import db, User, Conversion  # noqa: E402


# --- Fixtures ---

@pytest.fixture(scope='session')
def app():
    """Configure the Flask app once for the test session."""
    flask_app = app_module.app
    flask_app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
        SQLALCHEMY_DATABASE_URI=os.environ['DATABASE_URL'],
        SERVER_NAME='localhost.test',
    )
    with flask_app.app_context():
        db.create_all()
    yield flask_app
    with flask_app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture(autouse=True)
def _reset_db(app):
    """Wipe DB tables between tests so each test sees a fresh state."""
    with app.app_context():
        db.session.remove()
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()
    yield


@pytest.fixture
def client(app):
    """Anonymous Flask test client."""
    return app.test_client()


@pytest.fixture
def test_user(app):
    """A pre-created user (username='alice', password='hunter2hunter2')."""
    with app.app_context():
        user = User(username='alice')
        user.set_password('hunter2hunter2')
        db.session.add(user)
        db.session.commit()
        return {'id': user.id, 'username': 'alice', 'password': 'hunter2hunter2'}


@pytest.fixture
def authenticated_client(client, test_user):
    """Test client with a logged-in session for ``test_user``."""
    resp = client.post('/login', data={
        'username': test_user['username'],
        'password': test_user['password'],
    }, follow_redirects=False)
    assert resp.status_code == 302, f'login failed: {resp.status_code} {resp.data!r}'
    return client


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / 'fixtures'


@pytest.fixture
def captured_templates(app):
    """Record (template, context) pairs rendered during a request.

    Lets tests assert on context values the template does not (yet) render —
    e.g. R2-E ships reading_items/inbox_count from the backend one phase
    before the frontend displays them.
    """
    from flask import template_rendered

    recorded = []

    def record(sender, template, context, **extra):
        recorded.append((template, context))

    template_rendered.connect(record, app)
    yield recorded
    template_rendered.disconnect(record, app)


# --- External-service mocks ---

@pytest.fixture
def mock_deepgram(app):
    """Replace the module-level ``deepgram_service`` singleton with a MagicMock.

    Tests can configure ``mock.transcribe_file.return_value = '...'`` etc.
    """
    mock_svc = MagicMock()
    original = app_module.deepgram_service
    app_module.deepgram_service = mock_svc
    yield mock_svc
    app_module.deepgram_service = original


@pytest.fixture
def mock_redis_queue(app):
    """Replace the module-level ``task_queue`` and patch ``Job.fetch``.

    The fixture yields a dict with three handles: ``queue`` (the mock RQ
    queue), ``job`` (a default MagicMock job that ``enqueue`` returns), and
    ``fetch`` (the ``Job.fetch`` mock, reconfigurable mid-test).

    Since JOB-ID-REUSE the web side creates the job id itself and passes it
    as ``enqueue(..., job_id=<mark>)``; the id of the mock job the queue
    returns (``test-job-123``) reaches nothing anymore. Tests read the mark
    from the response / ``enqueue.call_args.kwargs['job_id']``.
    """
    mock_queue = MagicMock()
    mock_job = MagicMock()
    mock_job.id = 'test-job-123'
    mock_queue.enqueue.return_value = mock_job

    original_queue = app_module.task_queue
    app_module.task_queue = mock_queue

    fetch_patcher = patch.object(app_module.Job, 'fetch')
    mock_fetch = fetch_patcher.start()
    # JOB-ID-REUSE: the reconciles read ``job.is_finished`` (finished without
    # a result file → failed). A bare MagicMock attribute is truthy — the
    # default fetched job is one that is still queued/running.
    mock_fetch.return_value.is_finished = False

    handles = {
        'queue': mock_queue,
        'job': mock_job,
        'fetch': mock_fetch,
    }
    yield handles

    fetch_patcher.stop()
    app_module.task_queue = original_queue


# --- SEC-SOCKET: the mineru launcher, real HTTP, faked docker CLI ---

def _docker_call_kind(argv):
    if argv[:3] == ['docker', 'image', 'inspect']:
        return 'image_inspect'
    if argv[:3] == ['docker', 'rm', '-f']:
        return 'kill'
    if argv[:3] == ['docker', 'volume', 'rm']:
        return 'volume_rm'
    if 'dd' in argv:
        return 'copy_in'
    if 'cp' in argv:
        return 'copy_out'
    return 'run'


def _mounts(argv):
    """``-v src:dst[:opts]`` of a docker run argv → [(src, dst)]."""
    return [tuple(argv[i + 1].split(':')[:2])
            for i, arg in enumerate(argv[:-1]) if arg == '-v']


@pytest.fixture
def fake_launcher(monkeypatch, tmp_path):
    """The REAL launcher (``services.mineru_launcher``) on an ephemeral
    loopback port with only ITS process boundary faked: ``subprocess.run``
    stands in for the docker CLI. The worker side (``services.pdf_local``)
    talks real HTTP to it, so request validation, argv building from the
    launcher's env, the run lock, the helper steps and the kill path all run
    for real — a patched HTTP client would test the fake, not the transport.

    The fake is a small daemon: ``-v`` sources resolve to host dirs (a
    volume name to ``tmp/docker_volumes/<name>``), ``dd`` copies the input
    into the in-volume (and snapshots it — the worker removes its job dir
    afterwards), the mineru run writes a content_list into the out-volume
    like a real run (nested output tree), ``cp -r`` copies it back into the
    job dir, ``volume rm`` deletes the volume dirs. Every call is recorded
    with its ``kind`` (copy_in / run / copy_out / kill / volume_rm). Bare
    metal: worker view == host view, one tmp dir. Knobs: ``content_list``,
    ``rc``, ``stderr``, ``raise_timeout``, ``write_output``.
    """
    import shutil

    from services import mineru_launcher

    volumes = tmp_path / 'docker_volumes'
    state = {'calls': [], 'content_list': [], 'rc': 0, 'raise_timeout': False,
             'write_output': True, 'stderr': '', 'input_pdfs': []}

    def resolve(argv, container_path):
        for src, dst in sorted(_mounts(argv), key=lambda m: -len(m[1])):
            if container_path == dst or container_path.startswith(dst + '/'):
                base = Path(src) if src.startswith('/') else volumes / src
                return base / container_path[len(dst):].lstrip('/')
        return None

    def fake_run(argv, capture_output=True, text=None, timeout=None, **kwargs):
        argv = list(argv)
        kind = _docker_call_kind(argv)
        state['calls'].append({'kind': kind, 'cmd': argv, 'timeout': timeout})
        ok = SimpleNamespace(returncode=0, stdout='', stderr='')
        if kind == 'image_inspect':  # ARCH-BUILD: the launcher's start-up identity line
            return SimpleNamespace(returncode=0, stderr='',
                                   stdout='sha256:fake 2026-01-01T00:00:00.000000000Z\n')
        if kind == 'kill':
            return ok
        if kind == 'volume_rm':
            for name in argv[3:]:
                shutil.rmtree(volumes / name, ignore_errors=True)
            return ok
        if kind == 'copy_in':
            args = dict(a.split('=', 1) for a in argv if a[:3] in ('if=', 'of='))
            source, dest = resolve(argv, args['if']), resolve(argv, args['of'])
            if source is None or not source.is_file():
                return SimpleNamespace(returncode=1, stdout='',
                                       stderr=f"dd: can't open '{args['if']}'")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            state['input_pdfs'].append(dest.read_bytes())
            return ok
        if kind == 'copy_out':
            source, dest = resolve(argv, '/o'), resolve(argv, argv[-1])
            if not dest.parent.is_dir():
                return SimpleNamespace(returncode=1, stdout='',
                                       stderr=f"cp: can't create '{argv[-1]}'")
            shutil.copytree(source, dest, dirs_exist_ok=True)
            return ok
        if state['raise_timeout']:
            # TimeoutExpired carries bytes even under text=True.
            raise subprocess.TimeoutExpired(argv, timeout, output=b'',
                                            stderr=b'mineru lief noch')
        out = resolve(argv, '/out')
        out.mkdir(parents=True, exist_ok=True)
        if state['rc'] == 0 and state['write_output']:
            nested = out / 'doc' / 'vlm'
            nested.mkdir(parents=True, exist_ok=True)
            (nested / 'doc_content_list.json').write_text(
                json.dumps(state['content_list']), encoding='utf-8')
        return SimpleNamespace(returncode=state['rc'], stdout='',
                               stderr=state['stderr'])

    monkeypatch.setattr(mineru_launcher.subprocess, 'run', fake_run)
    exchange = tmp_path / 'exchange'
    exchange.mkdir()
    for name in ('MINERU_MODELS_DIR', 'MINERU_IMAGE', 'MINERU_TIMEOUT_BASE_SECONDS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('DOC_LOCAL_EXCHANGE_DIR', str(exchange))
    monkeypatch.setenv('DOC_LOCAL_EXCHANGE_HOST_DIR', str(exchange))
    monkeypatch.setenv('EXCHANGE_OWNER', '0:0')

    server = mineru_launcher.make_server('127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    monkeypatch.setenv('MINERU_LAUNCHER_URL', url)
    state.update(exchange=exchange, volumes=volumes, url=url,
                 runs=lambda: [c for c in state['calls'] if c['kind'] == 'run'],
                 kinds=lambda: [c['kind'] for c in state['calls']])
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    # A failed assertion must not leak launcher state into the next test.
    if mineru_launcher._RUN_LOCK.locked():
        mineru_launcher._RUN_LOCK.release()
    mineru_launcher._STOPPING.clear()
