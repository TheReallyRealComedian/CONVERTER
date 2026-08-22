"""ASGI adapter for the Flask app: WSGI calls on a thread pool (SYNC-FREEZE P2).

asgiref's stock ``WsgiToAsgi`` runs every WSGI call through a bare
``@sync_to_async`` — ``thread_sensitive=True`` — which means ONE thread per
process (``SyncToAsync.single_thread_executor``, ``ThreadPoolExecutor(1)``).
Measured consequences (DOC-WEB-ASYNC P1, SYNC-FREEZE P1): a long request
parks every other request of its process until it ends; more processes only
dilute that (a worker keeps *accepting* connections while its single thread
is busy — the event loop is free — and parks ≈ busy/N of them); and a
context leak from that single thread into the next request of a keep-alive
connection kills fast request sequences (``RuntimeError: Single thread
executor already being used, would deadlock`` — or, the same leak seen on
the Mintbox, a request that is accepted and never answered). An asgiref bump
does not fix the leak (3.12.1 raises ``CurrentThreadExecutor already quit``
in the same spot).

This adapter keeps asgiref's WSGI translation byte-for-byte (same
``run_wsgi_app`` body — see the sentinel test) and changes one thing: the
call runs with ``thread_sensitive=False`` on a dedicated, bounded
``ThreadPoolExecutor``. Flask is built for that (``wsgi.multithread`` was
already ``True``; every request has its own app context, Flask-SQLAlchemy
scopes the session to it, SQLAlchemy's pysqlite dialect sets
``check_same_thread=False`` for file databases); the deadlock/leak path is
simply never entered, because the thread-sensitive branch is what sets it
up. Async views (``/convert-markdown`` → Playwright) keep working the way
they always did: Flask's ``async_to_sync`` finds the uvicorn loop through
asgiref's thread-local (set per call in ``thread_handler``) and runs the
coroutine there while the pool thread idles.

A DEDICATED executor rather than the loop's default one: asyncio uses the
default executor for its own blocking helpers (``getaddrinfo`` …); WSGI
calls must not be able to starve those, and the bound is a named decision
(``app_pkg.config.WEB_SYNC_THREADS``) instead of ``min(32, cpus + 4)``.
"""
from concurrent.futures import ThreadPoolExecutor

from asgiref.sync import sync_to_async
from asgiref.wsgi import WsgiToAsgi, WsgiToAsgiInstance

from app_pkg.config import WEB_SYNC_THREADS

# One pool per process, shared by every request of that process.
wsgi_executor = ThreadPoolExecutor(max_workers=WEB_SYNC_THREADS,
                                   thread_name_prefix='wsgi')

# The original, undecorated ``run_wsgi_app`` — asgiref stores the function
# on the SyncToAsync wrapper as ``.func`` (functools.update_wrapper also
# exposes it as ``__wrapped__``). Reaching through the class ``__dict__``
# avoids the descriptor, which would hand back a bound partial.
_upstream_run_wsgi_app = WsgiToAsgiInstance.__dict__['run_wsgi_app'].func


class ThreadPoolWsgiToAsgiInstance(WsgiToAsgiInstance):
    """Per-request instance: upstream body, pool thread instead of THE thread."""

    run_wsgi_app = sync_to_async(_upstream_run_wsgi_app, thread_sensitive=False,
                                 executor=wsgi_executor)


class ThreadPoolWsgiToAsgi(WsgiToAsgi):
    """Drop-in for ``asgiref.wsgi.WsgiToAsgi`` — same constructor, same scope
    handling, WSGI calls on :data:`wsgi_executor`."""

    async def __call__(self, scope, receive, send):
        await ThreadPoolWsgiToAsgiInstance(self.wsgi_application)(scope, receive, send)
