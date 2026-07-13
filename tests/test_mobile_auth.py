"""MOBILE-AUTH P1 — POST /api/auth/login + bearer identity via request_loader.

Security-bearing characterization: hashed-only token storage (plaintext never
in DB or logs), generic 401 with an identical body for unknown-user and
wrong-password (anti-enumeration), bearer reads without any cookie, fail-closed
handling of broken/revoked/expired tokens, and — crucially — the web session
path staying byte-identical: unauthenticated page loads still 302 to /login,
and /api/* requests WITHOUT an Authorization header keep today's redirect
(the _utils.js session-expiry UX depends on it). The 401 is scoped to
requests that cannot come from the cookie web UI (Bearer header present, or
the app-only /api/auth/* paths).
"""
import hashlib
import logging
from datetime import datetime

from models import ApiToken, Conversion, User, db

LOGIN_URL = '/api/auth/login'
LIST_URL = '/api/conversions'


def _make_user(app, username='alice', password='hunter2hunter2'):
    with app.app_context():
        u = User(username=username)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        return u.id


def _make_conversion(app, user_id, title):
    with app.app_context():
        c = Conversion(user_id=user_id, conversion_type='markdown',
                       title=title, content='# body')
        db.session.add(c)
        db.session.commit()
        return c.id


def _login(client, username='alice', password='hunter2hunter2'):
    return client.post(LOGIN_URL, json={'username': username, 'password': password})


def _bearer(token):
    return {'Authorization': f'Bearer {token}'}


# --- Token issuance -------------------------------------------------------


def test_login_success_returns_token_and_user(app, client):
    user_id = _make_user(app)
    resp = _login(client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['user'] == {'id': user_id, 'username': 'alice'}
    token = body['token']
    assert isinstance(token, str) and len(token) >= 32
    # No session cookie — the mobile login issues a token, not a session.
    assert 'Set-Cookie' not in resp.headers


def test_token_stored_hashed_only(app, client):
    _make_user(app)
    token = _login(client).get_json()['token']
    with app.app_context():
        rows = ApiToken.query.all()
        assert len(rows) == 1
        row = rows[0]
        assert row.token_hash == hashlib.sha256(token.encode()).hexdigest()
        assert row.token_hash != token  # plaintext is not what's stored
        assert row.label == 'ios-app'
        assert row.expires_at is None
        # to_dict never exposes the hash
        assert 'token_hash' not in row.to_dict()


def test_login_failure_is_generic_for_unknown_user_and_wrong_password(app, client):
    _make_user(app)
    wrong_pw = client.post(LOGIN_URL, json={'username': 'alice', 'password': 'nope'})
    unknown = client.post(LOGIN_URL, json={'username': 'mallory', 'password': 'nope'})
    assert wrong_pw.status_code == unknown.status_code == 401
    # Identical body — no user-enumeration signal in the response.
    assert wrong_pw.get_json() == unknown.get_json()
    with app.app_context():
        assert ApiToken.query.count() == 0


def test_login_malformed_bodies_fail_generically(app, client):
    _make_user(app)
    no_json = client.post(LOGIN_URL, data='not json', content_type='text/plain')
    non_dict = client.post(LOGIN_URL, json=['a', 'list'])
    non_str = client.post(LOGIN_URL, json={'username': ['alice'], 'password': 42})
    for resp in (no_json, non_dict, non_str):
        assert resp.status_code == 401
        assert resp.get_json() == {'error': 'Nicht autorisiert.'}


def test_token_never_in_logs(app, client, caplog):
    _make_user(app)
    with caplog.at_level(logging.INFO):
        token = _login(client).get_json()['token']
    assert token not in caplog.text


# --- Bearer identity on @login_required reads -----------------------------


def test_bearer_read_without_cookie_is_user_scoped(app, client):
    alice_id = _make_user(app, 'alice')
    bob_id = _make_user(app, 'bob', 'other-password-99')
    _make_conversion(app, alice_id, 'alice doc')
    _make_conversion(app, bob_id, 'bob doc')

    token = _login(client).get_json()['token']
    resp = client.get(LIST_URL, headers=_bearer(token))
    assert resp.status_code == 200
    body = resp.get_json()
    assert [c['title'] for c in body['items']] == ['alice doc']
    assert body['total'] == 1


def test_broken_token_gets_generic_401(app, client):
    _make_user(app)
    resp = client.get(LIST_URL, headers=_bearer('definitely-not-a-token'))
    assert resp.status_code == 401
    assert resp.get_json() == {'error': 'Nicht autorisiert.'}


def test_revoked_token_gets_401(app, client):
    _make_user(app)
    token = _login(client).get_json()['token']
    with app.app_context():
        ApiToken.query.delete()
        db.session.commit()
    assert client.get(LIST_URL, headers=_bearer(token)).status_code == 401


def test_expired_token_gets_401(app, client):
    _make_user(app)
    token = _login(client).get_json()['token']
    with app.app_context():
        row = ApiToken.query.first()
        row.expires_at = datetime(2020, 1, 1)  # naive, like SQLite hands back
        db.session.commit()
    assert client.get(LIST_URL, headers=_bearer(token)).status_code == 401


def test_last_used_at_bumped_on_use(app, client):
    _make_user(app)
    token = _login(client).get_json()['token']
    with app.app_context():
        assert ApiToken.query.first().last_used_at is None
    client.get(LIST_URL, headers=_bearer(token))
    with app.app_context():
        assert ApiToken.query.first().last_used_at is not None


# --- me / logout (P2) ------------------------------------------------------


def test_me_with_token_returns_identity(app, client):
    user_id = _make_user(app)
    token = _login(client).get_json()['token']
    resp = client.get('/api/auth/me', headers=_bearer(token))
    assert resp.status_code == 200
    assert resp.get_json() == {'id': user_id, 'username': 'alice'}


def test_me_without_token_401(client):
    resp = client.get('/api/auth/me')
    assert resp.status_code == 401
    assert resp.get_json() == {'error': 'Nicht autorisiert.'}


def test_me_with_garbage_token_401(app, client):
    _make_user(app)
    assert client.get('/api/auth/me', headers=_bearer('garbage')).status_code == 401


def test_logout_revokes_presented_token(app, client):
    _make_user(app)
    token = _login(client).get_json()['token']
    resp = client.post('/api/auth/logout', headers=_bearer(token))
    assert resp.status_code == 200
    assert resp.get_json()['revoked'] is True
    with app.app_context():
        assert ApiToken.query.count() == 0
    # The presented token is dead: the next call bounces at the door.
    assert client.get('/api/auth/me', headers=_bearer(token)).status_code == 401
    assert client.post('/api/auth/logout', headers=_bearer(token)).status_code == 401


def test_logout_only_revokes_the_presented_token(app, client):
    _make_user(app)
    token_a = _login(client).get_json()['token']
    token_b = _login(client).get_json()['token']
    assert client.post('/api/auth/logout', headers=_bearer(token_a)).status_code == 200
    # The sibling token keeps working — revocation is per-row, not per-user.
    assert client.get('/api/auth/me', headers=_bearer(token_b)).status_code == 200
    with app.app_context():
        assert ApiToken.query.count() == 1


# --- Web session path stays byte-identical --------------------------------


def test_unauthenticated_page_still_redirects_to_login(client):
    resp = client.get('/library')
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_api_without_any_auth_keeps_redirect_for_web_ui(client):
    # Deliberate scoping: no Authorization header + not /api/auth/* → the
    # cookie-web session-expiry UX (302 → _utils.js "Session expired") stays.
    resp = client.get(LIST_URL)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_api_auth_path_returns_401_even_without_header(client):
    # /api/auth/* is app-only: no header at all must still be a clean 401
    # (the iOS app validates a missing/cleared token via these endpoints).
    resp = client.post(LOGIN_URL, json={})
    assert resp.status_code == 401


def test_session_login_still_works_alongside(authenticated_client):
    resp = authenticated_client.get(LIST_URL)
    assert resp.status_code == 200


# --- Structural: CSRF exemption scoped to the login view ------------------


def test_login_view_is_csrf_exempt_and_scoped(app):
    csrf = app.extensions['csrf']
    assert 'app_pkg.mobile_auth.api_auth_login' in csrf._exempt_views
    # The session-backed create route stays protected.
    assert 'app_pkg.library.api_create_conversion' not in csrf._exempt_views
