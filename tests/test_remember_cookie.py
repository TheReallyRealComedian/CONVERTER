"""SEC-AUDIT P2 — the remember cookie lives 30 days, not Flask-Login's 365.

The web login always sets ``remember=True``; before, the cookie's Expires sat
a year out. Named limit (see app_pkg/__init__.py): the value carries no
timestamp, so this bounds the honest browser, not a stolen cookie value —
SECRET_KEY rotation is the revocation lever.
"""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime


def test_remember_duration_is_pinned(app):
    assert app.config['REMEMBER_COOKIE_DURATION'] == timedelta(days=30)


def test_login_sets_a_30_day_remember_cookie(client, test_user):
    resp = client.post('/login', data={'username': test_user['username'],
                                       'password': test_user['password']})
    assert resp.status_code == 302
    header = next(h for h in resp.headers.getlist('Set-Cookie')
                  if h.startswith('remember_token='))
    expires = next(part.split('=', 1)[1] for part in header.split('; ')
                   if part.lower().startswith('expires='))
    lifetime = parsedate_to_datetime(expires) - datetime.now(timezone.utc)
    assert timedelta(days=29, hours=23) < lifetime <= timedelta(days=30)
