"""Service package — the SDK-backed classes load LAZILY (SEC-SOCKET).

``from services import DeepgramService`` works exactly as before. What
changed: importing a pure sibling module no longer drags the Deepgram and
Cloud-TTS SDKs into the process. That matters for
``python -m services.mineru_launcher`` — the one holder of the host's docker
socket must stay minimal (stdlib + ``services.mineru_invocation``) — and for
``app_pkg.config``, which reads the mineru deadline from
``services.mineru_invocation`` and promises to stay free of service SDKs.
PEP 562 module ``__getattr__``; a resolved class is cached in the module
namespace, so later lookups never come back here.
"""
from importlib import import_module

_LAZY = {
    'DeepgramService': '.deepgram_service',
    'GoogleTTSService': '.google_tts_service',
}

__all__ = list(_LAZY)


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        # AttributeError (not ImportError) keeps ``from services import
        # <submodule>`` working: Python falls back to the submodule import.
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
