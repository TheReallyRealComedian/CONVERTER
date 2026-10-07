"""``app_pkg`` — the web app's package: the route modules (one ``register(app)``
each, no blueprints), the factory and its parts, and ``config``.

ARCH-FACTORY P2: this init is a PEP-562 lazy loader, modelled on
``services/__init__.py``, and imports NOTHING of the project eagerly. Why:
``app_pkg.config`` is read by the worker and six service modules, and every
``from app_pkg.config import …`` runs this file first — until P2 it WAS the
factory and pulled Flask, Flask-WTF, SQLAlchemy, click and ``models`` into
processes that never build a web app (measured at the pin: ``import worker``
622 modules with flask, sqlalchemy and models loaded; ``import app_pkg.config``
621). Now ``import app_pkg.config`` costs ``config``, and the factory loads
when someone asks for it.

Exactly six names are served, each from the module that defines it;
anything else is an ``AttributeError`` — so ``from app_pkg import config``
(or ``learn``, ``library``, a route module) falls through to the submodule
import, as Python intends. A resolved name is cached in this namespace, so a
second lookup never comes back here. Sentinels: tests/test_import_surface.py.
"""
from importlib import import_module

_LAZY = {
    'create_app': '.factory',
    '_run_pending_migrations': '.migrations',
    '_migrate_conversion_tags_csv_to_junction': '.migrations',
    '_startup_lock': '.db_runtime',
    '_startup_lock_path': '.db_runtime',
    'HttpsOnlySecureSessionInterface': '.security',
}

__all__ = list(_LAZY)


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        # AttributeError (not ImportError) keeps ``from app_pkg import
        # <submodule>`` working: Python falls back to the submodule import.
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
