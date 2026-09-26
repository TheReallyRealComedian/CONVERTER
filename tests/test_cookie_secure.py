"""SEC-AUDIT P2 — cookie Secure flags: Secure wherever TLS is.

Measured before (edge, 2026-09-25): ``set-cookie: session=…; HttpOnly;
Path=/; SameSite=Lax`` — no ``Secure``. Decision (Oli, P2): the session
cookie is Secure exactly when the request arrived over HTTPS (every browser
request behind nginx, via ProxyFix), the remember cookie always.

Why not a blanket ``SESSION_COOKIE_SECURE``: the converter-mcp container logs
in with a cookie session over plain http inside the Docker network, and httpx
never sends a Secure cookie over http — its login POST would lose the session
and die at CSRF. The Werkzeug test client ignores ``Secure`` when replaying
cookies, so it cannot show that; the MCP tests below use a real
``httpx.Client`` over ``httpx.WSGITransport`` instead.
"""
import re

import pytest

from app_pkg import HttpsOnlySecureSessionInterface

httpx = pytest.importorskip('httpx')

NGINX_HEADERS = {
    'X-Forwarded-For': '203.0.113.5',
    'X-Forwarded-Proto': 'https',
    'X-Forwarded-Host': 'converter.smallpieces.de',
    'X-Forwarded-Port': '443',
}
MCP_BASE_URL = 'http://markdown-converter-web:5000'  # converter-mcp's CONVERTER_BASE_URL


@pytest.fixture
def csrf_enabled(app):
    app.config['WTF_CSRF_ENABLED'] = True
    yield app
    app.config['WTF_CSRF_ENABLED'] = False


def _set_cookies(resp):
    """{cookie name: its Set-Cookie attribute string}."""
    out = {}
    for header in resp.headers.getlist('Set-Cookie'):
        name = header.split('=', 1)[0]
        out[name] = header
    return out


def _form_login(client, test_user, headers=None):
    resp = client.post('/login', data={'username': test_user['username'],
                                       'password': test_user['password']},
                       headers=headers or {})
    assert resp.status_code == 302
    return _set_cookies(resp)


# --- the flags ------------------------------------------------------------------


def test_login_behind_nginx_sets_secure_session_and_remember_cookie(client, test_user):
    cookies = _form_login(client, test_user, NGINX_HEADERS)
    for name in ('session', 'remember_token'):
        assert '; Secure' in cookies[name], name
        assert '; HttpOnly' in cookies[name], name
        assert 'SameSite=Lax' in cookies[name], name


def test_login_over_plain_http_session_not_secure_remember_still_secure(client, test_user):
    # Internal http callers (converter-mcp, in-container smokes, Mac dev)
    # need the session cookie back over http; the remember cookie they never use.
    cookies = _form_login(client, test_user)
    assert '; Secure' not in cookies['session']
    assert '; Secure' in cookies['remember_token']


def test_production_defaults_are_pinned(app):
    assert isinstance(app.session_interface, HttpsOnlySecureSessionInterface)
    assert app.config['REMEMBER_COOKIE_SECURE'] is True
    assert app.config['SESSION_COOKIE_HTTPONLY'] is True
    assert app.config['SESSION_COOKIE_SAMESITE'] == 'Lax'


# --- the converter-mcp login, with a real RFC-following client --------------------


def _mcp_login_then_read(app, test_user):
    """converter-mcp's ConverterClient._login + _get_json, verbatim in shape:
    GET /login, scrape csrf_token, POST the form, then a session GET."""
    with httpx.Client(transport=httpx.WSGITransport(app=app), base_url=MCP_BASE_URL,
                      follow_redirects=False) as mcp:
        page = mcp.get('/login')
        token = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', page.text).group(1)
        login = mcp.post('/login', data={'username': test_user['username'],
                                         'password': test_user['password'],
                                         'csrf_token': token})
        read = mcp.get('/api/conversions') if login.status_code == 302 else None
        return login, read


def test_mcp_cookie_session_over_plain_http_keeps_working(app, test_user, csrf_enabled):
    login, read = _mcp_login_then_read(app, test_user)
    assert login.status_code == 302
    assert read.status_code == 200
    assert 'items' in read.json()


def test_a_blanket_secure_session_cookie_would_break_the_mcp(app, test_user,
                                                             csrf_enabled, monkeypatch):
    # Why HttpsOnlySecureSessionInterface exists: with Secure on every
    # session cookie, httpx keeps the cookie from GET /login to itself over
    # http, the POST arrives without a session, the CSRF token has nothing to
    # match — the MCP's login dies before any read tool runs.
    class BlanketSecure(HttpsOnlySecureSessionInterface):
        def get_cookie_secure(self, app):
            return True

    monkeypatch.setattr(app, 'session_interface', BlanketSecure())
    login, read = _mcp_login_then_read(app, test_user)
    assert login.status_code == 400
    assert read is None
