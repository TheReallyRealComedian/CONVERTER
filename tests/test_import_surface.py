"""ARCH-NARR5 — what an import drags into the process, and what ``app`` holds.

NARR-5 retired the alt-podcast flow (the *users*) and left the *providers*
standing: a Cloud-TTS client built at import in every web process — the only
reason the GCP key file was mounted at the internet-facing container — and
the WAV helpers of the living renderer inside a package whose ``__init__``
imported ``google.genai``. ARCH-NARR5 removed the singletons, moved the
helpers to ``services/wav_concat.py`` and deleted ``services/gemini``; these
sentinels hold that.

Each import is measured in its **own subprocess** (model:
``test_launcher_import_surface_is_minimal``): inside the pytest process
``sys.modules`` already carries whatever an earlier test imported, so an
in-process check could neither pass nor fail honestly.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import app as app_module
from app_pkg.decorators import require_service

REPO = Path(__file__).resolve().parents[1]

# ``tasks`` runs ``os.makedirs('/app/output_podcasts')`` at import — a no-op
# on a dev box where /app is not writable (the same ONE path as conftest.py;
# since ARCH-FACTORY the factory derives its directory from the DB URI, so
# nothing else may need this).
_PRELUDE = '''
import json, os, sys
_real_makedirs = os.makedirs
def _safe_makedirs(path, *args, **kwargs):
    if str(path) == '/app/output_podcasts':
        return
    return _real_makedirs(path, *args, **kwargs)
os.makedirs = _safe_makedirs
'''

_REPORT = '''
print(json.dumps(sorted(sys.modules)))
'''


def _modules_after(body, env=None):
    result = subprocess.run([sys.executable, '-c', _PRELUDE + body + _REPORT],
                            cwd=REPO, env=env, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def _loaded(modules, name):
    return name in modules or any(m.startswith(name + '.') for m in modules)


# --- the renderer and the worker's task module pay for no genai SDK ---------

@pytest.mark.parametrize('module', [
    'services.wav_concat',
    'services.narration_render',
    'services.google_tts_service',
    'tasks',
])
def test_import_does_not_load_the_genai_sdk(module):
    modules = _modules_after(f'import {module}\n')
    assert module in modules  # positive control: the import happened
    assert not _loaded(modules, 'google.genai')
    assert not _loaded(modules, 'services.gemini')


def test_the_genai_check_can_fire():
    # Positive control for the matcher: the Cloud-PDF path builds its own
    # genai client, so the SDK is installed and shows up when imported.
    modules = _modules_after('from google import genai\n')
    assert _loaded(modules, 'google.genai')


def test_wav_concat_is_stdlib_only():
    """The helpers the renderer needs load no SDK and nothing of the app."""
    modules = _modules_after('import services.wav_concat\n')
    project = {m for m in modules if m.split('.')[0] in ('services', 'app_pkg')}
    assert project == {'services', 'services.wav_concat'}
    for forbidden in ('google', 'grpc', 'flask', 'deepgram', 'pydub'):
        assert not _loaded(modules, forbidden), forbidden


# --- the web shim holds no Cloud-TTS and no Gemini client --------------------

def test_app_holds_no_retired_singleton():
    for name in ('google_tts_service', 'GoogleTTSService',
                 'GOOGLE_CREDENTIALS_PATH',
                 'gemini_service', 'GeminiService', 'GEMINI_API_KEY'):
        assert not hasattr(app_module, name), name
    assert hasattr(app_module, 'deepgram_service')  # positive control


@pytest.mark.parametrize('retired', ['google_tts', 'gemini'])
def test_require_service_knows_only_deepgram(retired):
    with pytest.raises(ValueError):
        require_service(retired)
    require_service('deepgram')  # positive control: the living key resolves


def test_the_gemini_package_is_gone():
    import services
    assert sorted(services._LAZY) == ['DeepgramService', 'GoogleTTSService']
    with pytest.raises(ImportError):
        from services import GeminiService  # noqa: F401
    # Not even as a namespace package (a leftover directory would be one).
    assert not (REPO / 'services' / 'gemini').exists()


# ``import app`` the way the web container runs it after ARCH-NARR5: both
# key names are in the env (``env_file`` still delivers them), but there is
# no key file at the path. The pre-sprint shim built a Cloud-TTS client at
# import and died right here with ``DefaultCredentialsError``; the heavy
# deps the suite never loads are stubbed exactly as conftest.py does.
_IMPORT_APP = '''
import types
from unittest.mock import MagicMock
unstructured = types.ModuleType('unstructured')
partition_pkg = types.ModuleType('unstructured.partition')
partition_auto = types.ModuleType('unstructured.partition.auto')
partition_auto.partition = lambda **_kwargs: []
sys.modules['unstructured'] = unstructured
sys.modules['unstructured.partition'] = partition_pkg
sys.modules['unstructured.partition.auto'] = partition_auto
playwright_pkg = types.ModuleType('playwright')
playwright_async = types.ModuleType('playwright.async_api')
playwright_async.async_playwright = MagicMock()
sys.modules['playwright'] = playwright_pkg
sys.modules['playwright.async_api'] = playwright_async
import app
assert app.deepgram_service is not None
'''


# --- ARCH-FACTORY: whoever wants config gets config, not the web app ---------
#
# ``app_pkg/__init__.py`` was the factory: every ``from app_pkg.config import``
# in the worker, the launcher's neighbours and six service modules ran it and
# loaded Flask, Flask-WTF, SQLAlchemy, click and ``models`` into a process
# that never builds a web app (measured at the pin before P2: ``import
# worker`` 622 modules, flask + sqlalchemy + models loaded). Since P2 the init
# is a PEP-562 loader for six names and imports nothing itself.

WEB_APP = ('flask', 'sqlalchemy', 'models')

SIX_NAMES = {
    'create_app': '.factory',
    '_run_pending_migrations': '.migrations',
    '_migrate_conversion_tags_csv_to_junction': '.migrations',
    '_startup_lock': '.db_runtime',
    '_startup_lock_path': '.db_runtime',
    'HttpsOnlySecureSessionInterface': '.security',
}


@pytest.mark.parametrize('module', [
    'app_pkg.config',
    'worker',
    'services.narration_library',
    'tasks',  # heavy by design (renderers, SDKs) — but no web app in it
])
def test_import_does_not_load_the_web_app(module):
    modules = _modules_after(f'import {module}\n')
    assert module in modules  # positive control: the import happened
    loaded = [name for name in WEB_APP if _loaded(modules, name)]
    assert loaded == [], f'{module} loads {loaded}'


def test_the_web_app_check_can_fire():
    # Positive control for the matcher: the factory IS the web app — importing
    # it alone loads all three.
    modules = _modules_after('import app_pkg.factory\n')
    assert [name for name in WEB_APP if _loaded(modules, name)] == list(WEB_APP)


def test_package_init_is_a_lazy_loader_for_exactly_six_names():
    import app_pkg
    from importlib import import_module

    assert app_pkg._LAZY == SIX_NAMES
    assert app_pkg.__all__ == list(SIX_NAMES)
    for name, module in SIX_NAMES.items():
        # each name is the object its home module defines, not a copy
        assert getattr(app_pkg, name) is getattr(import_module(module, 'app_pkg'), name)
    with pytest.raises(AttributeError):
        app_pkg.no_such_name  # noqa: B018 — the loader must raise, not import
    # ``from app_pkg import <submodule>`` keeps working through that AttributeError
    from app_pkg import config  # noqa: F401
    # … and the init itself imports nothing of the project: a fresh process
    # that imports only the package holds the package and nothing else of ours.
    modules = _modules_after('import app_pkg\n')
    project = {m for m in modules if m.split('.')[0] in ('app_pkg', 'services', 'models')}
    assert project == {'app_pkg'}
    assert not any(_loaded(modules, name) for name in WEB_APP)


def test_import_app_needs_no_key_file_and_loads_no_genai(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if k not in ('GOOGLE_APPLICATION_CREDENTIALS', 'GEMINI_API_KEY',
                        'DEEPGRAM_API_KEY', 'DATABASE_URL')}
    env.update(
        SECRET_KEY='test-secret-key',
        DATABASE_URL=f"sqlite:///{tmp_path / 'import-surface.db'}",
        REDIS_URL='redis://localhost:6379/0',
        GEMINI_API_KEY='test-gemini-key',
        DEEPGRAM_API_KEY='test-deepgram-key',
        GOOGLE_APPLICATION_CREDENTIALS=str(tmp_path / 'no-such-key.json'),
    )
    modules = _modules_after(_IMPORT_APP, env=env)
    assert 'app' in modules and 'tasks' in modules  # positive control
    assert not _loaded(modules, 'google.genai')
    assert not _loaded(modules, 'services.gemini')
