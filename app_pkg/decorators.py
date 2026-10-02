"""Route decorators for the CONVERTER blueprints.

Currently exposes ``require_service`` (F-011): a uniform 503 + DE-JSON gate
in front of the endpoints that need a configured service. Since ARCH-NARR5
the one key is ``deepgram`` (the alt-podcast endpoints that used
``google_tts`` and ``gemini`` went with NARR-5). The decorator looks the
service singleton up on the top-level ``app`` module at request time,
mirroring the existing late-import pattern the route modules use so test
patches at ``app.<name>`` continue to apply.
"""
from functools import wraps

from flask import jsonify


_SERVICE_LABELS = {
    'deepgram': 'Audio-Transkriptions-Service',
}

_SERVICE_ATTRS = {
    'deepgram': 'deepgram_service',
}


def require_service(service_name):
    """Return a decorator that 503s if the named service singleton is falsy.

    ``service_name`` must be one of the keys in ``_SERVICE_ATTRS``. Resolves
    the actual singleton on the top-level ``app`` module at call time so
    tests that patch ``app.deepgram_service = None`` (or similar) still take
    effect.
    """
    if service_name not in _SERVICE_ATTRS:
        raise ValueError(f"Unknown service: {service_name}")

    label = _SERVICE_LABELS[service_name]
    attr = _SERVICE_ATTRS[service_name]

    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            import app as _app_module
            if not getattr(_app_module, attr, None):
                return jsonify({
                    'error': f'{label} ist nicht konfiguriert. '
                             'API-Key fehlt in der Server-Konfiguration.'
                }), 503
            return view_func(*args, **kwargs)
        return wrapper
    return decorator
