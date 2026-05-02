"""Authentication characterization tests.

Locks in: GET /login renders, POST /login with valid creds redirects, POST
/login with invalid creds re-renders, /logout clears the session, and the
production-default config has CSRF protection enabled.
"""
import importlib
import os
import sys

from models import db, User


def test_login_get_renders(client):
    resp = client.get('/login')
    assert resp.status_code == 200
    assert b'<form' in resp.data.lower()
    assert b'username' in resp.data.lower()


def test_login_post_valid_credentials_redirects(client, test_user):
    resp = client.post('/login', data={
        'username': test_user['username'],
        'password': test_user['password'],
    }, follow_redirects=False)
    assert resp.status_code == 302
    # Successful login redirects to the markdown converter (root).
    assert resp.headers['Location'].endswith('/')


def test_login_post_invalid_credentials_rerender_with_flash(client, test_user):
    resp = client.post('/login', data={
        'username': test_user['username'],
        'password': 'wrong-password',
    }, follow_redirects=False)
    assert resp.status_code == 200
    assert b'Invalid username or password' in resp.data


def test_login_already_authenticated_redirects(authenticated_client):
    resp = authenticated_client.get('/login', follow_redirects=False)
    assert resp.status_code == 302


def test_logout_clears_session(authenticated_client):
    resp = authenticated_client.get('/logout', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']
    # After logout, protected routes redirect back to /login.
    follow = authenticated_client.get('/', follow_redirects=False)
    assert follow.status_code == 302
    assert '/login' in follow.headers['Location']


def test_unauthenticated_protected_route_redirects_to_login(client):
    resp = client.get('/', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_csrf_enabled_in_production_default():
    """The TEST harness flips ``WTF_CSRF_ENABLED`` off; production must keep it on.

    Re-import the live app module from a clean import so its config dict reflects
    the production default (no test fixture has overridden it yet).
    """
    # Spawn a subprocess so the in-memory app config (already mutated by other
    # fixtures) doesn't affect this check.
    import subprocess
    code = (
        "import os, sys, types\n"
        "for n in ['unstructured','unstructured.partition','unstructured.partition.auto',\n"
        "          'playwright','playwright.async_api']:\n"
        "    m = types.ModuleType(n)\n"
        "    if n.endswith('.auto'): m.partition = lambda **kw: []\n"
        "    if n.endswith('.async_api'):\n"
        "        from unittest.mock import MagicMock; m.async_playwright = MagicMock()\n"
        "    sys.modules[n] = m\n"
        "os.environ['SECRET_KEY']='x'\n"
        "os.environ['DATABASE_URL']='sqlite:///:memory:'\n"
        "_om = os.makedirs\n"
        "def _safe(p, *a, **kw):\n"
        "    if str(p).startswith('/app'): return\n"
        "    return _om(p, *a, **kw)\n"
        "os.makedirs = _safe\n"
        "sys.path.insert(0, '.')\n"
        "import app as a\n"
        "# Flask-WTF defaults WTF_CSRF_ENABLED to True when CSRFProtect is registered.\n"
        "assert a.csrf is not None, 'CSRFProtect not initialised'\n"
        "assert a.app.config.get('WTF_CSRF_ENABLED', True) is True, 'CSRF disabled in prod default!'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f'subproc failed: {result.stderr}'
    assert 'OK' in result.stdout
