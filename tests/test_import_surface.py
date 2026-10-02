"""ARCH-NARR5 — what an import drags into the process, and what ``app`` holds.

NARR-5 retired the alt-podcast flow (the *users*) and left the *providers*
standing: a Cloud-TTS client built at import in every web process — the only
reason the GCP key file was mounted at the internet-facing container — and
the WAV helpers of the living renderer inside a package whose ``__init__``
imported ``google.genai``. These sentinels hold what ARCH-NARR5 removed.

Each import is measured in its **own subprocess** (model:
``test_launcher_import_surface_is_minimal``): inside the pytest process
``sys.modules`` already carries whatever an earlier test imported, so an
in-process check could neither pass nor fail honestly.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import app as app_module
from app_pkg.decorators import require_service

REPO = Path(__file__).resolve().parents[1]

# ``tasks`` runs ``os.makedirs('/app/output_podcasts')`` at import — a no-op
# on a dev box where /app is not writable (same wrap as conftest.py).
_PRELUDE = '''
import json, os, sys
_real_makedirs = os.makedirs
def _safe_makedirs(path, *args, **kwargs):
    if str(path).startswith('/app'):
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


# --- the web shim holds no Cloud-TTS client ---------------------------------

def test_app_holds_no_cloud_tts_singleton():
    for name in ('google_tts_service', 'GoogleTTSService',
                 'GOOGLE_CREDENTIALS_PATH'):
        assert not hasattr(app_module, name), name


def test_require_service_knows_no_google_tts():
    with pytest.raises(ValueError):
        require_service('google_tts')
    require_service('deepgram')  # positive control: the living key resolves
