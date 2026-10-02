"""SEC-DG-TOKEN (2026-10-02) — the browser gets a short-lived grant, never the key.

``GET /api/get-deepgram-token`` used to hand the browser the Deepgram API key
itself (``create_temporary_key`` returned ``self.api_key`` and never read its
``ttl_seconds``). The key carries ``account:write`` and does not expire; the
docstring justified it with "LAN-only", a premise that fell on 2026-08-21.

What is pinned here:
  * the view answers only with a token minted by Deepgram's grant, with the
    lifetime and the deadline the SERVER sets (``app_pkg/config.py``);
  * every failure is a 502 with a German sentence — fail-closed, no branch
    falls back to the key, and neither key nor token reaches body or log;
  * the service has no method (and no attribute) left that could hand the key
    out;
  * the real SDK at the installed version speaks the measured wire form
    (``POST /v1/auth/grant``, ``{"ttl_seconds": …}``, the per-call deadline on
    the request, one attempt) — against an httpx mock transport, no network.

The mocks sit at the SDK boundary: a real ``DeepgramService`` with a known
fake key, its SDK client replaced.
"""
import inspect
import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from deepgram import DeepgramClient

import app as app_module
from app_pkg import config
from services.deepgram_service import DeepgramService

ROUTE = '/api/get-deepgram-token'
FAKE_KEY = 'dg-fake-key-0123456789abcdef0123456789abcdef'
FAKE_JWT = 'eyJhbGciOiJIUzI1NiJ9.eyJzY29wZSI6InVzYWdlOndyaXRlIn0.c2lnbmF0dXJl'
ERROR_SENTENCE = ('Transkriptions-Token konnte nicht erstellt werden. '
                  'Bitte erneut versuchen.')


def _granted(access_token=FAKE_JWT, expires_in=30.0):
    """What ``client.auth.v1.tokens.grant`` returns, reduced to the two fields."""
    return SimpleNamespace(access_token=access_token, expires_in=expires_in)


def _grant(service):
    return service.client.auth.v1.tokens.grant


@pytest.fixture
def live_service(app):
    """A real service with a known fake key and a mocked SDK client, installed
    as the app singleton the view looks up."""
    service = DeepgramService(api_key=FAKE_KEY)
    service.client = MagicMock()
    original = app_module.deepgram_service
    app_module.deepgram_service = service
    yield service
    app_module.deepgram_service = original


# What a broken grant can look like: nothing, a wrong shape, an empty token,
# an expiry that is none.
FORMLESS_RESULTS = [
    pytest.param(None, id='none'),
    pytest.param(SimpleNamespace(), id='no-fields'),
    pytest.param({'access_token': FAKE_JWT, 'expires_in': 30}, id='dict-not-model'),
    pytest.param(_granted(access_token=None), id='token-none'),
    pytest.param(_granted(access_token=''), id='token-empty'),
    pytest.param(_granted(access_token='   '), id='token-blank'),
    pytest.param(_granted(access_token=12345), id='token-not-a-string'),
    pytest.param(_granted(expires_in=None), id='expiry-none'),
    pytest.param(_granted(expires_in=0), id='expiry-zero'),
    pytest.param(_granted(expires_in=-30), id='expiry-negative'),
    pytest.param(_granted(expires_in='30'), id='expiry-string'),
    pytest.param(_granted(expires_in=True), id='expiry-bool'),
    pytest.param(_granted(expires_in=float('nan')), id='expiry-nan'),
]


# --------------------------------------------------------------------------
# The server sets lifetime and deadline
# --------------------------------------------------------------------------

def test_lifetime_and_deadline_are_server_constants():
    assert config.DEEPGRAM_LIVE_TOKEN_TTL_SECONDS == 30
    assert config.TIMEOUT_DEEPGRAM_GRANT_SECONDS == 10


# --------------------------------------------------------------------------
# View
# --------------------------------------------------------------------------

def test_view_returns_the_granted_token_and_its_expiry(authenticated_client, live_service):
    _grant(live_service).return_value = _granted()

    resp = authenticated_client.get(ROUTE)

    assert resp.status_code == 200
    assert resp.get_json() == {'deepgram_token': FAKE_JWT, 'expires_in': 30}
    _grant(live_service).assert_called_once_with(
        ttl_seconds=config.DEEPGRAM_LIVE_TOKEN_TTL_SECONDS,
        request_options={
            'timeout_in_seconds': config.TIMEOUT_DEEPGRAM_GRANT_SECONDS,
            'max_retries': 0,
        },
    )
    # A credential response is never stored: a cached answer would also be an
    # expired token on the next recording.
    assert resp.headers['Cache-Control'] == 'no-store'


def test_view_ignores_a_lifetime_the_client_asks_for(authenticated_client, live_service):
    _grant(live_service).return_value = _granted()

    resp = authenticated_client.get(ROUTE + '?ttl_seconds=3600&ttl=3600')

    assert resp.status_code == 200
    assert _grant(live_service).call_args.kwargs['ttl_seconds'] == 30


def test_view_grant_failure_is_502_without_key_or_token(authenticated_client, live_service,
                                                        caplog):
    _grant(live_service).side_effect = RuntimeError(
        f'upstream said no: Authorization: Token {FAKE_KEY} / {FAKE_JWT}')

    with caplog.at_level(logging.DEBUG):
        resp = authenticated_client.get(ROUTE)

    assert resp.status_code == 502
    assert resp.get_json() == {'error': ERROR_SENTENCE}
    body = resp.get_data(as_text=True)
    assert FAKE_KEY not in body and FAKE_JWT not in body
    assert FAKE_KEY not in caplog.text and FAKE_JWT not in caplog.text
    assert 'RuntimeError' in caplog.text  # the log still says what kind of failure


@pytest.mark.parametrize('result', FORMLESS_RESULTS)
def test_view_formless_grant_is_502_never_an_empty_token(authenticated_client, live_service,
                                                         result):
    _grant(live_service).return_value = result

    resp = authenticated_client.get(ROUTE)

    assert resp.status_code == 502
    assert resp.get_json() == {'error': ERROR_SENTENCE}


def test_view_requires_login(client, live_service):
    resp = client.get(ROUTE, follow_redirects=False)

    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']
    _grant(live_service).assert_not_called()


def test_view_503_when_deepgram_is_not_configured(authenticated_client):
    original = app_module.deepgram_service
    app_module.deepgram_service = None
    try:
        resp = authenticated_client.get(ROUTE)
    finally:
        app_module.deepgram_service = original

    assert resp.status_code == 503
    assert 'nicht konfiguriert' in resp.get_json()['error']


# --------------------------------------------------------------------------
# Sentinel: the key leaves through no branch
# --------------------------------------------------------------------------

def test_no_response_branch_carries_the_configured_key(authenticated_client, live_service):
    grant = _grant(live_service)
    responses = []

    grant.return_value = _granted()
    responses.append(authenticated_client.get(ROUTE))           # success

    grant.side_effect = RuntimeError('grant down')
    responses.append(authenticated_client.get(ROUTE))           # SDK error

    grant.side_effect = TimeoutError('deadline')
    responses.append(authenticated_client.get(ROUTE))           # timeout

    grant.side_effect = None
    grant.return_value = None
    responses.append(authenticated_client.get(ROUTE))           # formless result

    assert [r.status_code for r in responses] == [200, 502, 502, 502]
    for resp in responses:
        assert FAKE_KEY not in resp.get_data(as_text=True)
        assert FAKE_KEY not in str(resp.headers)


def test_service_cannot_hand_out_the_key():
    service = DeepgramService(api_key=FAKE_KEY)

    assert not hasattr(DeepgramService, 'create_temporary_key')
    # The service does not hold the key itself — only the SDK client does.
    assert FAKE_KEY not in [v for v in vars(service).values() if isinstance(v, str)]
    assert 'self.api_key' not in inspect.getsource(DeepgramService)


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------

def test_service_passes_lifetime_and_deadline_and_returns_token_and_expiry():
    service = DeepgramService(api_key=FAKE_KEY)
    service.client = MagicMock()
    _grant(service).return_value = _granted(expires_in=45.0)

    result = service.grant_live_token(ttl_seconds=45, timeout_seconds=7)

    assert result == (FAKE_JWT, 45.0)
    _grant(service).assert_called_once_with(
        ttl_seconds=45,
        request_options={'timeout_in_seconds': 7, 'max_retries': 0},
    )


def test_service_has_no_default_lifetime_or_deadline():
    service = DeepgramService(api_key=FAKE_KEY)
    service.client = MagicMock()

    with pytest.raises(TypeError):
        service.grant_live_token()
    with pytest.raises(TypeError):
        service.grant_live_token(30, 10)  # keyword-only
    _grant(service).assert_not_called()


@pytest.mark.parametrize('result', FORMLESS_RESULTS)
def test_service_rejects_an_empty_or_formless_grant(result):
    service = DeepgramService(api_key=FAKE_KEY)
    service.client = MagicMock()
    _grant(service).return_value = result

    with pytest.raises(RuntimeError) as excinfo:
        service.grant_live_token(ttl_seconds=30, timeout_seconds=10)

    assert FAKE_JWT not in str(excinfo.value)


def test_service_lets_an_sdk_error_through():
    service = DeepgramService(api_key=FAKE_KEY)
    service.client = MagicMock()
    _grant(service).side_effect = TimeoutError('deadline')

    with pytest.raises(TimeoutError):
        service.grant_live_token(ttl_seconds=30, timeout_seconds=10)


# --------------------------------------------------------------------------
# The real SDK at the installed version, without network
# --------------------------------------------------------------------------

def _service_on_transport(handler):
    service = DeepgramService(api_key=FAKE_KEY)
    service.client = DeepgramClient(
        api_key=FAKE_KEY,
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return service


def test_real_sdk_grant_speaks_the_measured_wire_form():
    """Signature, parameter names and return shape of the pinned SDK: a bump
    that renames one of them fails here, not in a browser."""
    seen = {}

    def handler(request):
        seen['method'] = request.method
        seen['host'] = request.url.host
        seen['path'] = request.url.path
        seen['body'] = json.loads(request.content)
        seen['authorization_is_the_key'] = (
            request.headers.get('authorization') == f'Token {FAKE_KEY}')
        seen['timeout'] = request.extensions.get('timeout')
        return httpx.Response(200, json={'access_token': FAKE_JWT, 'expires_in': 30})

    service = _service_on_transport(handler)

    token, expires_in = service.grant_live_token(ttl_seconds=30, timeout_seconds=10)

    assert (token, expires_in) == (FAKE_JWT, 30)
    assert seen == {
        'method': 'POST',
        'host': 'api.deepgram.com',
        'path': '/v1/auth/grant',
        'body': {'ttl_seconds': 30},
        'authorization_is_the_key': True,
        'timeout': {'connect': 10, 'read': 10, 'write': 10, 'pool': 10},
    }


@pytest.mark.parametrize('status', [403, 429, 500, 503])
def test_real_sdk_grant_is_one_attempt_and_raises(status):
    """The SDK's default is two retries with backoff (and a Retry-After of up
    to 60 s) on 429/5xx — in a web thread that would stretch the deadline
    several times over. One attempt, then the error."""
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(status, json={'err_code': 'X', 'err_msg': 'no'})

    service = _service_on_transport(handler)

    with pytest.raises(Exception) as excinfo:
        service.grant_live_token(ttl_seconds=30, timeout_seconds=10)

    assert calls == ['/v1/auth/grant']
    assert getattr(excinfo.value, 'status_code', None) == status


def test_real_sdk_grant_without_a_token_in_the_answer_raises():
    def handler(request):
        return httpx.Response(200, json={'expires_in': 30})

    service = _service_on_transport(handler)

    with pytest.raises(Exception):
        service.grant_live_token(ttl_seconds=30, timeout_seconds=10)
