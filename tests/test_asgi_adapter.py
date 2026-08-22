"""SYNC-FREEZE P2: the thread-pool ASGI adapter (``app_pkg/asgi.py``).

The stock ``asgiref.wsgi.WsgiToAsgi`` serialises every WSGI call of a
process on ONE thread and leaks its context into the next request of a
keep-alive connection. These tests pin (1) that the app is served through
the pool adapter, (2) the asgiref internals the adapter reaches into — so an
asgiref bump that changes them fails loudly instead of silently falling back
to the single thread, (3) the behaviour: concurrent sync views overlap, and
the keep-alive leak (next request started from inside the previous request's
``send``, exactly what uvicorn's ``on_response_complete`` does) no longer
raises. Both behaviours are shown against the stock adapter as the contrast.
"""
import asyncio
import re
import threading
import time

import pytest
from asgiref.sync import SyncToAsync
from asgiref.wsgi import WsgiToAsgi, WsgiToAsgiInstance
from flask import Flask

import app as app_module
from app_pkg.asgi import (ThreadPoolWsgiToAsgi, ThreadPoolWsgiToAsgiInstance,
                          wsgi_executor)
from app_pkg.config import WEB_SYNC_THREADS


def _tiny_app(delay):
    flask_app = Flask('tiny')

    @flask_app.route('/slow')
    def slow():
        time.sleep(delay)
        return 'ok'

    return flask_app


def _scope(path='/slow'):
    return {'type': 'http', 'http_version': '1.1', 'method': 'GET', 'path': path,
            'query_string': b'', 'headers': [(b'host', b'test')],
            'server': ('test', 80), 'client': ('127.0.0.1', 1)}


async def _request(adapter, on_complete=None):
    """Drive one request through the ASGI interface; collect the messages.
    ``on_complete`` (if given) is called from INSIDE the final ``send`` —
    the uvicorn ``on_response_complete`` position."""
    sent = []

    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message):
        sent.append(message)
        if on_complete and message['type'] == 'http.response.body' \
                and not message.get('more_body', False):
            on_complete()

    await adapter(_scope(), receive, send)
    return sent


def _status(sent):
    return [m for m in sent if m['type'] == 'http.response.start'][0]['status']


def test_app_is_served_through_the_thread_pool_adapter():
    assert isinstance(app_module.asgi_app, ThreadPoolWsgiToAsgi)


def test_adapter_pins_the_asgiref_internals_it_relies_on():
    # Same WSGI translation as upstream: the wrapped function IS asgiref's.
    upstream = WsgiToAsgiInstance.__dict__['run_wsgi_app']
    ours = ThreadPoolWsgiToAsgiInstance.__dict__['run_wsgi_app']
    assert isinstance(upstream, SyncToAsync) and isinstance(ours, SyncToAsync)
    assert ours.func is upstream.func
    # ... but not on the single thread.
    assert upstream._thread_sensitive is True
    assert ours._thread_sensitive is False
    assert ours._executor is wsgi_executor
    assert wsgi_executor._max_workers == WEB_SYNC_THREADS


def test_concurrent_sync_views_overlap_on_the_pool():
    adapter = ThreadPoolWsgiToAsgi(_tiny_app(0.3))

    async def three():
        return await asyncio.gather(*(_request(adapter) for _ in range(3)))

    t0 = time.monotonic()
    results = asyncio.run(three())
    wall = time.monotonic() - t0
    assert [_status(r) for r in results] == [200, 200, 200]
    assert wall < 0.6, f'three 0.3 s views took {wall:.2f} s — serialised'


def test_stock_adapter_serialises_the_same_views():
    """The contrast that makes the test above meaningful."""
    adapter = WsgiToAsgi(_tiny_app(0.3))

    async def three():
        return await asyncio.gather(*(_request(adapter) for _ in range(3)))

    t0 = time.monotonic()
    results = asyncio.run(three())
    wall = time.monotonic() - t0
    assert [_status(r) for r in results] == [200, 200, 200]
    assert wall >= 0.85, f'stock adapter overlapped ({wall:.2f} s)?'


# TEST-HANG-KEEPALIVE (KLEINKRAM, 2026-08-22): the stock adapter's leak has
# TWO manifestations — the 500 ('would deadlock' / 'CurrentThreadExecutor …')
# and a follow-up request that is never answered (SYNC-FREEZE measured both
# on the live instance). The sentinel below used to know only the first: in
# the container it hit the hang in 3 of 4 runs and blocked the whole suite,
# on Python 3.10 and 3.12 alike. A hanging test is worse than a failing one,
# so the pair runs under a deadline and its expiry COUNTS as the leak.
KEEP_ALIVE_DEADLINE_SECONDS = 5.0


class KeepAliveHang(Exception):
    """The keep-alive pair did not return within the deadline — the silent
    manifestation of the leak."""


def _run_keep_alive_pair(adapter, deadline=None):
    """Request 2 is created as a task from inside request 1's final send —
    the position uvicorn's ``on_response_complete`` starts the next cycle
    of a keep-alive connection from (pipelined / fast-follow request).

    With ``deadline`` the pair runs on a daemon thread and is abandoned
    there if it does not return in time (``KeepAliveHang``). A thread join
    is the only timeout that covers both shapes of a hang: an in-loop
    ``asyncio.wait_for`` never fires when the loop itself is the thing
    that is stuck."""
    async def pair():
        second = []

        def start_second():
            second.append(asyncio.ensure_future(_request(adapter)))

        first = await _request(adapter, on_complete=start_second)
        return _status(first), await second[0]

    if deadline is None:
        return asyncio.run(pair())

    outcome = {}

    def run():
        try:
            outcome['result'] = asyncio.run(pair())
        except BaseException as exc:  # re-raised on the caller's thread below
            outcome['error'] = exc

    worker = threading.Thread(target=run, name='keep-alive-pair', daemon=True)
    worker.start()
    worker.join(deadline)
    if worker.is_alive():
        raise KeepAliveHang(f'keep-alive pair still running after {deadline:g} s')
    if 'error' in outcome:
        raise outcome['error']
    return outcome['result']


def test_keep_alive_follow_up_request_survives_on_the_pool():
    adapter = ThreadPoolWsgiToAsgi(_tiny_app(0.0))
    first_status, second = _run_keep_alive_pair(adapter)
    assert first_status == 200
    assert _status(second) == 200


def test_stock_adapter_leaks_its_context_into_the_follow_up_request():
    """The defect, reproduced without a server, in BOTH of its shapes: asgiref
    3.8.1 either raises 'Single thread executor already being used, would
    deadlock' (3.12.1: 'CurrentThreadExecutor already quit or is broken') or
    never answers the follow-up request at all — which shape you get is a
    race (the Mac always raises, the container mostly hangs). A clean pass
    is the failure here: it would mean the leak is gone and the sentinel —
    and the pool adapter's reason to exist — must be re-examined."""
    adapter = WsgiToAsgi(_tiny_app(0.0))
    try:
        _run_keep_alive_pair(adapter, deadline=KEEP_ALIVE_DEADLINE_SECONDS)
    except RuntimeError as exc:
        assert re.search('deadlock|CurrentThreadExecutor', str(exc)), exc
        print('leak manifestation: RuntimeError (the 500)')   # visible via -rP
    except KeepAliveHang as exc:
        print(f'leak manifestation: hang ({exc})')             # visible via -rP
    else:
        pytest.fail('stock adapter served the keep-alive follow-up cleanly — '
                    'leak gone? re-measure before touching the pool adapter')
