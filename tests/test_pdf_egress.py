"""SEC-SSRF: tests for the default-deny route handler ``app_pkg.pdf_egress``.

The handler fulfills from a patched ``fetch_public_https`` or aborts; it must
never pass a request through. A source-level sentinel guards that. The budget
caps (count, bytes, wall-clock) are exercised with the constants monkeypatched
low so the tests stay fast and deterministic.
"""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app_pkg.pdf_egress as pdf_egress
from app_pkg.pdf_egress import _Budget, _make_handler
from services.egress import EgressBlocked, EgressResponse


def _run(coro):
    return asyncio.run(coro)


def _make_route(url='https://example.org/x.png', headers=None):
    route = MagicMock()
    request = MagicMock()
    request.url = url
    request.all_headers = AsyncMock(
        return_value=headers or {'accept': 'image/*', 'user-agent': 'UA'})
    route.request = request
    route.abort = AsyncMock()
    route.fulfill = AsyncMock()
    return route


def _resp(body=b'PNG', status=200, content_type='image/png'):
    return EgressResponse(status=status, content_type=content_type,
                          body=body, final_url='https://example.org/x.png')


def test_public_resource_is_fulfilled():
    handler = _make_handler(_Budget())
    route = _make_route()
    with patch.object(pdf_egress, 'fetch_public_https', return_value=_resp()):
        _run(handler(route))
    route.fulfill.assert_awaited_once()
    kwargs = route.fulfill.await_args.kwargs
    assert kwargs['status'] == 200
    assert kwargs['headers'] == {'content-type': 'image/png'}
    assert kwargs['body'] == b'PNG'
    route.abort.assert_not_awaited()


def test_forwards_exactly_accept_and_user_agent():
    handler = _make_handler(_Budget())
    route = _make_route(headers={'accept': 'image/avif', 'user-agent': 'Chrome',
                                 'cookie': 'secret=1', 'referer': 'x'})
    captured = {}

    def fake_fetch(url, *, accept, user_agent):
        captured['accept'] = accept
        captured['user_agent'] = user_agent
        return _resp()

    with patch.object(pdf_egress, 'fetch_public_https', side_effect=fake_fetch):
        _run(handler(route))
    assert captured == {'accept': 'image/avif', 'user_agent': 'Chrome'}


def test_blocked_resource_is_aborted():
    handler = _make_handler(_Budget())
    route = _make_route(url='https://10.0.0.1/x.png')
    with patch.object(pdf_egress, 'fetch_public_https',
                      side_effect=EgressBlocked('ip_literal', '10.0.0.1')):
        _run(handler(route))
    route.abort.assert_awaited_once_with('blockedbyclient')
    route.fulfill.assert_not_awaited()


def test_unexpected_fetch_error_is_aborted_not_raised():
    handler = _make_handler(_Budget())
    route = _make_route()
    with patch.object(pdf_egress, 'fetch_public_https',
                      side_effect=RuntimeError('boom')):
        _run(handler(route))  # must not raise
    route.abort.assert_awaited_once_with('blockedbyclient')
    route.fulfill.assert_not_awaited()


def test_budget_request_count_cap(monkeypatch):
    monkeypatch.setattr(pdf_egress, '_MAX_REQUESTS', 2)
    handler = _make_handler(_Budget())
    with patch.object(pdf_egress, 'fetch_public_https', return_value=_resp()):
        r1, r2, r3 = _make_route(), _make_route(), _make_route()
        _run(handler(r1))
        _run(handler(r2))
        _run(handler(r3))
    r1.fulfill.assert_awaited_once()
    r2.fulfill.assert_awaited_once()
    r3.fulfill.assert_not_awaited()
    r3.abort.assert_awaited_once_with('blockedbyclient')


def test_budget_total_bytes_cap(monkeypatch):
    monkeypatch.setattr(pdf_egress, '_MAX_TOTAL_BYTES', 10)
    handler = _make_handler(_Budget())
    with patch.object(pdf_egress, 'fetch_public_https',
                      return_value=_resp(body=b'x' * 20)):
        r1, r2 = _make_route(), _make_route()
        _run(handler(r1))   # records 20 bytes
        _run(handler(r2))   # total already over cap → abort
    r1.fulfill.assert_awaited_once()
    r2.fulfill.assert_not_awaited()
    r2.abort.assert_awaited_once_with('blockedbyclient')


def test_budget_wall_clock_cap(monkeypatch):
    clock = {'t': 1000.0}
    monkeypatch.setattr(pdf_egress.time, 'monotonic', lambda: clock['t'])
    budget = _Budget()            # _start captured at t=1000
    clock['t'] = 1000.0 + pdf_egress._MAX_WALL_SECONDS + 1
    handler = _make_handler(budget)
    route = _make_route()
    with patch.object(pdf_egress, 'fetch_public_https', return_value=_resp()):
        _run(handler(route))
    route.abort.assert_awaited_once_with('blockedbyclient')
    route.fulfill.assert_not_awaited()


def test_handler_never_continues_a_route():
    """The abort/fulfill contract: the handler module must not call
    ``route.continue_`` anywhere — default-deny has no pass-through branch."""
    src = Path(pdf_egress.__file__).read_text()
    assert 'continue_' not in src


def test_belt_args_are_dead_proxy_and_loopback_not_bypassed():
    assert pdf_egress.PDF_BROWSER_ARGS == [
        '--proxy-server=http://127.0.0.1:9',
        '--proxy-bypass-list=<-loopback>',
    ]
