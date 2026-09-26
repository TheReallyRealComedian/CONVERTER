"""SEC-AUDIT P2 — ProxyFix: one trusted hop (host nginx).

Before: every request behind nginx carried the Docker gateway as
``remote_addr`` (measured 172.21.0.1) and ``request.is_secure`` was False
behind TLS. ``ProxyFix(x_for=1, x_proto=1, x_host=1, x_port=1)`` restores
both — and with ``is_secure`` true, Flask-WTF's SSL-strict branch now applies
to cookie-session writes that come through nginx: a same-origin Referer is
required. These tests pin all of it:

  1. X-Forwarded-For becomes remote_addr; only the LAST hop (the address
     nginx appends) is trusted, a client-supplied prefix never wins.
  2. Without forwarded headers nothing changes (direct/internal callers).
  3. nginx's exact header set + same-origin Referer → the session write
     passes (X-Forwarded-Port 443 must not break the origin compare);
     no Referer / a foreign Referer → CSRF 400.
  4. Unchanged for the non-browser callers: a Bearer write behind nginx
     needs no Referer (inversion), and a plain-http form login without a
     Referer still works — the converter-mcp container logs in exactly so
     over the Docker network.

Fixture-order caveat (see test_csrf_inversion): ``authenticated_client``
before ``csrf_enabled``.
"""
import logging
import re
from pathlib import Path

import pytest

from models import Conversion, db

REPO = Path(__file__).resolve().parent.parent

# What host nginx sends upstream (proxy_set_header lines of the site config).
NGINX_HEADERS = {
    'X-Forwarded-For': '203.0.113.5',
    'X-Forwarded-Proto': 'https',
    'X-Forwarded-Host': 'converter.smallpieces.de',
    'X-Forwarded-Port': '443',
}
SAME_ORIGIN_REFERER = 'https://converter.smallpieces.de/library/1'


@pytest.fixture
def csrf_enabled(app):
    app.config['WTF_CSRF_ENABLED'] = True
    yield app
    app.config['WTF_CSRF_ENABLED'] = False


def _make_conversion(app, user_id):
    with app.app_context():
        c = Conversion(user_id=user_id, conversion_type='markdown_input',
                       title='doc', content='# body')
        db.session.add(c)
        db.session.commit()
        return c.id


def _failed_mobile_login_addr(client, caplog, headers=None):
    """remote_addr as the one auth-failure log line reports it."""
    with caplog.at_level(logging.WARNING, logger='app_pkg.mobile_auth'):
        resp = client.post('/api/auth/login', json={'username': 'nobody', 'password': 'x'},
                           headers=headers or {})
    assert resp.status_code == 401
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith('Mobile login failed from ')]
    assert lines, 'no auth-failure log line'
    return lines[-1].removeprefix('Mobile login failed from ')


# --- 1 + 2: remote_addr --------------------------------------------------------


def test_forwarded_for_becomes_remote_addr(client, caplog):
    assert _failed_mobile_login_addr(
        client, caplog, {'X-Forwarded-For': '203.0.113.5'}) == '203.0.113.5'


def test_only_the_hop_nginx_appends_is_trusted(client, caplog):
    # nginx's $proxy_add_x_forwarded_for APPENDS the real peer to whatever the
    # client sent — x_for=1 reads the last entry, so the forged prefix loses.
    addr = _failed_mobile_login_addr(
        client, caplog, {'X-Forwarded-For': '198.51.100.7, 203.0.113.5'})
    assert addr == '203.0.113.5'


def test_without_forwarded_headers_remote_addr_unchanged(client, caplog):
    assert _failed_mobile_login_addr(client, caplog) == '127.0.0.1'


# --- 3: is_secure behind nginx → Flask-WTF SSL-strict ------------------------


def _session_write(app, test_user, client, headers):
    cid = _make_conversion(app, test_user['id'])
    token = client.get('/api/csrf-token').get_json()['csrf_token']
    return client.patch(f'/api/conversions/{cid}/progress', json={'percent': 40},
                        headers={'X-CSRFToken': token, **headers})


def test_session_write_behind_nginx_with_same_origin_referer_passes(
        app, test_user, authenticated_client, csrf_enabled):
    resp = _session_write(app, test_user, authenticated_client,
                          {**NGINX_HEADERS, 'Referer': SAME_ORIGIN_REFERER})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 40


def test_session_write_behind_nginx_without_referer_is_csrf_400(
        app, test_user, authenticated_client, csrf_enabled):
    resp = _session_write(app, test_user, authenticated_client, NGINX_HEADERS)
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'csrf_expired'
    assert 'referrer' in resp.get_json()['message'].lower()


def test_session_write_behind_nginx_with_foreign_referer_is_csrf_400(
        app, test_user, authenticated_client, csrf_enabled):
    resp = _session_write(app, test_user, authenticated_client,
                          {**NGINX_HEADERS, 'Referer': 'https://evil.example/x'})
    assert resp.status_code == 400
    assert resp.get_json()['error'] == 'csrf_expired'


# --- 4: the non-browser callers are unaffected --------------------------------


def test_bearer_write_behind_nginx_needs_no_referer(app, test_user, client, csrf_enabled):
    login = client.post('/api/auth/login', json={'username': test_user['username'],
                                                 'password': test_user['password']})
    token = login.get_json()['token']
    resp = client.post('/api/auth/logout',
                       headers={**NGINX_HEADERS, 'Authorization': f'Bearer {token}'})
    assert resp.status_code == 200
    assert resp.get_json()['revoked'] is True


def test_plain_http_form_login_without_referer_still_works(app, test_user, client,
                                                           csrf_enabled):
    # The converter-mcp login, verbatim: GET /login, scrape csrf_token, POST
    # the form over plain http inside the Docker network — no Referer, no
    # forwarded headers, so is_secure stays False and SSL-strict never runs.
    page = client.get('/login')
    token = re.search(r'name="csrf_token"[^>]*value="([^"]+)"',
                      page.get_data(as_text=True)).group(1)
    resp = client.post('/login', data={'username': test_user['username'],
                                       'password': test_user['password'],
                                       'csrf_token': token})
    assert resp.status_code == 302


# --- the premise of trusting one hop ------------------------------------------


def test_web_port_is_bound_to_loopback_only():
    """ProxyFix trusts the X-Forwarded-* headers of whoever connects to :5656.
    That is sound only while host nginx is the sole outside client — so the
    published port must stay on 127.0.0.1 (0.0.0.0 served the app to the LAN
    without TLS, past nginx). Redis publishes nothing."""
    compose_file = REPO / 'docker-compose.yml'
    if not compose_file.exists():
        pytest.skip('docker-compose.yml not shipped alongside the tests')
    yaml = pytest.importorskip('yaml')
    services = yaml.safe_load(compose_file.read_text())['services']
    assert services['markdown-converter']['ports'] == ['127.0.0.1:5656:5000']
    assert 'ports' not in services['redis']
    assert 'ports' not in services['worker']
