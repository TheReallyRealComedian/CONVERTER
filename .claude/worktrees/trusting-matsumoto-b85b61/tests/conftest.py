"""Test fixtures and stubs for the Flask test client.

The tests in this directory exercise the application through the public HTTP
boundary (``app.test_client()``).  External SDK clients (Gemini, Deepgram,
Google Cloud TTS) are mocked at the place they are *instantiated* — never
inside the service implementation — so the mocks survive future internal
refactors (Stage 2 blueprint split, Stage 3 gemini_service decomposition).

Two pieces of test-only setup happen at import time, *before* ``app`` is
imported, because both happen during ``app.py`` module load:

1. ``unstructured.partition.auto`` and ``playwright.async_api`` are stubbed
   in ``sys.modules``.  These are heavy production dependencies; the
   characterization tests mock them at the ``app.partition`` /
   ``app.async_playwright`` boundary anyway, so a lightweight stub is
   sufficient and keeps the dev-machine install footprint small.
2. ``os.makedirs`` is wrapped to no-op for ``/app/*`` paths so the
   container-internal ``os.makedirs('/app/data', exist_ok=True)`` line
   does not fail on macOS / Linux dev boxes.
"""
import os
import sys
import types
import tempfile
from pathlib import Path
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

_TEST_DB_FILE = Path(tempfile.gettempdir()) / 'converter-test.db'
if _TEST_DB_FILE.exists():
    _TEST_DB_FILE.unlink()

os.environ.setdefault('SECRET_KEY', 'test-secret-key')
os.environ.setdefault('DATABASE_URL', f'sqlite:///{_TEST_DB_FILE}')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/0')
os.environ.setdefault('NOTION_MCP_URL', 'http://notion-mcp.test')
os.environ.setdefault('MCP_AUTH_TOKEN', 'test-mcp-token')
os.environ.setdefault('NOTION_TOKEN', '')

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
def mock_gemini(app):
    """Replace the module-level ``gemini_service`` singleton with a MagicMock."""
    mock_svc = MagicMock()
    original = app_module.gemini_service
    app_module.gemini_service = mock_svc
    yield mock_svc
    app_module.gemini_service = original


@pytest.fixture
def mock_google_tts(app):
    """Replace the module-level ``google_tts_service`` singleton with a MagicMock."""
    mock_svc = MagicMock()
    original = app_module.google_tts_service
    app_module.google_tts_service = mock_svc
    yield mock_svc
    app_module.google_tts_service = original


@pytest.fixture
def mock_redis_queue(app):
    """Replace the module-level ``task_queue`` and patch ``Job.fetch``.

    The fixture yields a dict with three handles: ``queue`` (the mock RQ
    queue), ``job`` (a default MagicMock job that ``enqueue`` returns), and
    ``set_fetch`` (a callable to reconfigure ``Job.fetch`` mid-test).
    """
    mock_queue = MagicMock()
    mock_job = MagicMock()
    mock_job.get_id.return_value = 'test-job-123'
    mock_queue.enqueue.return_value = mock_job

    original_queue = app_module.task_queue
    app_module.task_queue = mock_queue

    fetch_patcher = patch.object(app_module.Job, 'fetch')
    mock_fetch = fetch_patcher.start()

    handles = {
        'queue': mock_queue,
        'job': mock_job,
        'fetch': mock_fetch,
    }
    yield handles

    fetch_patcher.stop()
    app_module.task_queue = original_queue


@pytest.fixture
def gemini_api_key_set(app):
    """Force ``app.GEMINI_API_KEY`` to a truthy value for routes that gate on it."""
    original = app_module.GEMINI_API_KEY
    app_module.GEMINI_API_KEY = 'test-gemini-key'
    yield 'test-gemini-key'
    app_module.GEMINI_API_KEY = original
