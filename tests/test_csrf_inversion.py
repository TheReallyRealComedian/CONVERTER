"""MOBILE-AUTH P2 — CSRF inversion: bearer writes skip CSRF, session writes don't.

conftest disables CSRF globally (``WTF_CSRF_ENABLED=False``), so every test
here flips it back ON via the ``csrf_enabled`` fixture — the inversion's
manual before_request honours that flag exactly like the automatic handler
did (that guard replication is itself part of what these tests pin down).

The security-bearing directions (P2 sign-off):
  1. session write WITH a valid X-CSRFToken    → 200  (web round-trip intact)
  2. the SAME session write WITHOUT the token  → 400  (web protection intact)
  3. the SAME write with Bearer, no CSRF       → 200  (inversion works)
plus the legacy-exempt sentinel: a header-less POST to a CARD_TOKEN view must
die at its own gate (503 "nicht konfiguriert"), NOT at CSRF (400) — that
catches a broken ``_exempt_views`` replication immediately.

Fixture-order caveat: ``authenticated_client`` must be requested BEFORE
``csrf_enabled`` so the form login happens while CSRF is still off.
"""
import pytest

from models import ApiToken, Conversion, User, db

LOGIN_URL = '/api/auth/login'


@pytest.fixture
def csrf_enabled(app):
    app.config['WTF_CSRF_ENABLED'] = True
    yield app
    app.config['WTF_CSRF_ENABLED'] = False


def _make_user(app, username='alice', password='hunter2hunter2'):
    with app.app_context():
        u = User(username=username)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        return u.id


def _make_conversion(app, user_id, title='doc'):
    with app.app_context():
        c = Conversion(user_id=user_id, conversion_type='markdown_input',
                       title=title, content='# body')
        db.session.add(c)
        db.session.commit()
        return c.id


def _bearer_login(client, username='alice', password='hunter2hunter2'):
    resp = client.post(LOGIN_URL, json={'username': username, 'password': password})
    assert resp.status_code == 200
    return resp.get_json()['token']


# --- Direction 1+2: web UI posture byte-identical -------------------------


def test_session_write_with_valid_csrf_token_succeeds(app, test_user,
                                                      authenticated_client,
                                                      csrf_enabled):
    cid = _make_conversion(app, test_user['id'])
    token = authenticated_client.get('/api/csrf-token').get_json()['csrf_token']
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress',
                                      json={'percent': 40},
                                      headers={'X-CSRFToken': token})
    assert resp.status_code == 200
    assert resp.get_json()['last_read_percent'] == 40


def test_session_write_without_csrf_token_is_400(app, test_user,
                                                 authenticated_client,
                                                 csrf_enabled):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.patch(f'/api/conversions/{cid}/progress',
                                      json={'percent': 40})
    assert resp.status_code == 400
    # The JSON-aware CSRFError handler answers on /api/* — proof the failure
    # is the CSRF gate, not some other 400.
    assert resp.get_json()['error'] == 'csrf_expired'
    with app.app_context():
        assert db.session.get(Conversion, cid).last_read_percent is None


def test_session_get_stays_unaffected(authenticated_client, csrf_enabled):
    assert authenticated_client.get('/api/conversions').status_code == 200


# --- Direction 3: bearer writes skip CSRF ----------------------------------


def test_bearer_login_works_with_csrf_enabled(app, client, csrf_enabled):
    # The login POST itself (no cookie, no bearer) passes only through the
    # replicated exempt-views check — this is the inversion's login proof.
    _make_user(app)
    assert _bearer_login(client)


def test_bearer_write_without_csrf_succeeds(app, client, csrf_enabled):
    alice = _make_user(app)
    cid = _make_conversion(app, alice)
    token = _bearer_login(client)
    resp = client.patch(f'/api/conversions/{cid}/progress',
                        json={'percent': 55},
                        headers={'Authorization': f'Bearer {token}'})
    assert resp.status_code == 200
    with app.app_context():
        assert db.session.get(Conversion, cid).last_read_percent == 55


def test_bearer_create_lands_on_token_owner(app, client, csrf_enabled):
    alice = _make_user(app, 'alice')
    _make_user(app, 'bob', 'other-password-99')
    token = _bearer_login(client)
    resp = client.post('/api/conversions',
                       json={'conversion_type': 'markdown_input',
                             'title': 'from the app',
                             'content': '# mobile write'},
                       headers={'Authorization': f'Bearer {token}'})
    assert resp.status_code == 201
    with app.app_context():
        row = Conversion.query.filter_by(title='from the app').one()
        assert row.user_id == alice


def test_invalid_bearer_write_dies_at_auth_not_csrf(app, client, csrf_enabled):
    # Fail-closed order: a bogus bearer skips CSRF but must then die at the
    # door with the generic 401 — never reach the view, never 400-CSRF.
    alice = _make_user(app)
    cid = _make_conversion(app, alice)
    resp = client.patch(f'/api/conversions/{cid}/progress',
                        json={'percent': 55},
                        headers={'Authorization': 'Bearer bogus'})
    assert resp.status_code == 401
    assert resp.get_json() == {'error': 'Nicht autorisiert.'}


def test_bearer_logout_without_csrf_succeeds(app, client, csrf_enabled):
    _make_user(app)
    token = _bearer_login(client)
    resp = client.post('/api/auth/logout',
                       headers={'Authorization': f'Bearer {token}'})
    assert resp.status_code == 200
    with app.app_context():
        assert ApiToken.query.count() == 0


# --- Legacy-exempt sentinel (Schärfung 2) ----------------------------------


def test_legacy_exempt_view_dies_at_own_gate_not_csrf(client, csrf_enabled,
                                                      monkeypatch):
    # Header-less POST to the CARD_TOKEN-gated create route: the view's own
    # fail-closed gate must answer (503 without CARD_TOKEN configured) — a
    # 400 here would mean the _exempt_views replication is broken and CSRF
    # fired before the view.
    monkeypatch.delenv('CARD_TOKEN', raising=False)
    resp = client.post('/api/cards', json={})
    assert resp.status_code == 503
    assert resp.get_json() == {'error': 'Card-API nicht konfiguriert.'}


def test_legacy_ingest_view_own_gate_intact(client, csrf_enabled, monkeypatch):
    # Same sentinel through the ingest module (different exempt registration
    # site): fail-closed 503 without INGEST_TOKEN, not 400-CSRF.
    monkeypatch.delenv('INGEST_TOKEN', raising=False)
    resp = client.post('/api/ingest/conversion', json={})
    assert resp.status_code == 503
