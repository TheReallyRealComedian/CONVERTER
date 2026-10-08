"""CSRF-REFRESH — the CSRF token lives with the session (Teil a), the error
handler answers JSON to the fetch wrapper, and the wrapper's refresh-and-retry
stands on a pinned server mechanism (Teil b).

conftest disables CSRF globally (``WTF_CSRF_ENABLED=False``); every test here
flips it back ON via the ``csrf_enabled`` fixture, like
``tests/test_csrf_inversion.py`` (which stays untouched). Fixture-order
caveat from there: ``authenticated_client`` must be requested BEFORE
``csrf_enabled`` so the form login happens while CSRF is still off.

Teil (a) — ``WTF_CSRF_TIME_LIMIT = None`` in ``create_app``:
  (i)   sentinel: the factory's config carries the key with the value None
        (Flask-WTF 1.2.1 ``_get_config`` reads ``config.get(name, 3600)`` — a
        SET None comes through as None and becomes ``max_age=None``, which
        itsdangerous 2.2.0 reads as "no age check");
  (ii)  a token stamped two hours ago is ACCEPTED — ``TimestampSigner
        .get_timestamp`` is patched to ``now - 7200`` ONLY while the token is
        handed out, the write runs on the real clock;
  (iii) positive control: the same flow with the hour limit restored in the
        test is rejected as expired — proof the patch really ages the stamp.
(i) and (ii) are red against HEAD, (iii) green.

Teil (b) — the server side the fetch wrapper (templates/base.html) stands on:
  (iv)  the CSRFError handler answers JSON ``csrf_expired`` as soon as the
        token came as the ``X-CSRFToken`` HEADER — even outside ``/api/``
        and without ``Accept: application/json`` (``POST /transform-document``
        is the one mutating target outside ``/api/``; the wrapper is the only
        sender of that header, forms send the field). Red against HEAD: there
        the answer is the HTML "Session expired" page;
  (v)   belt: without the header and without Accept it STAYS that HTML page
        with the meta refresh onto the Referer (the two forms' path);
  (vi)  belt for the retry: a session minted from the remember cookie has no
        ``csrf_token`` — the page's old token dies with "The CSRF session
        token is missing.", ``GET /api/csrf-token`` answers 200 AND sets the
        ``session`` cookie, the new token writes. The wrapper's refresh is
        this sequence; it is pinned here, not built.
"""
import io
import time
from unittest.mock import patch

import pytest
from itsdangerous.timed import TimestampSigner
from itsdangerous.url_safe import URLSafeTimedSerializer

from models import Conversion, db

TWO_HOURS = 7200
_UNSET = object()


@pytest.fixture
def csrf_enabled(app):
    app.config['WTF_CSRF_ENABLED'] = True
    yield app
    app.config['WTF_CSRF_ENABLED'] = False


def _make_conversion(app, user_id, title='doc'):
    with app.app_context():
        c = Conversion(user_id=user_id, conversion_type='markdown_input',
                       title=title, content='# body')
        db.session.add(c)
        db.session.commit()
        return c.id


def _token_stamped(client, seconds_ago):
    """A CSRF token from ``GET /api/csrf-token`` whose signature timestamp
    lies ``seconds_ago`` in the past. The patch covers ONLY the hand-out —
    the validating request later runs on the real clock, so the age it
    computes is real."""
    with patch.object(TimestampSigner, 'get_timestamp',
                      lambda self: int(time.time()) - seconds_ago):
        resp = client.get('/api/csrf-token')
    assert resp.status_code == 200
    return resp.get_json()['csrf_token']


def _stamp_age(app, token):
    """Seconds since the token's signature timestamp, read with the same
    serializer Flask-WTF uses — the test's own proof that the stamp is old."""
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'], salt='wtf-csrf-token')
    _, stamped_at = serializer.loads(token, return_timestamp=True)
    return int(time.time() - stamped_at.timestamp())


# --- (i) the sentinel on the factory's config -------------------------------


def test_csrf_time_limit_is_none_in_the_factory_config(app):
    # ``app`` IS ``create_app()``'s return (app.py: ``app = create_app()``);
    # conftest touches TESTING / WTF_CSRF_ENABLED / the DB URI / SERVER_NAME
    # and nothing else. The key must EXIST with the value None — an absent
    # key means Flask-WTF's default of 3600 seconds.
    assert app.config.get('WTF_CSRF_TIME_LIMIT', _UNSET) is None


# --- (ii) a two-hour-old token is accepted ---------------------------------


def test_two_hour_old_token_is_accepted(app, test_user, authenticated_client,
                                        csrf_enabled):
    cid = _make_conversion(app, test_user['id'])
    token = _token_stamped(authenticated_client, TWO_HOURS)
    assert _stamp_age(app, token) >= TWO_HOURS - 5

    resp = authenticated_client.put(f'/api/conversions/{cid}',
                                    json={'title': 'nach zwei Stunden gespeichert'},
                                    headers={'X-CSRFToken': token})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    with app.app_context():
        assert db.session.get(Conversion, cid).title == 'nach zwei Stunden gespeichert'


# --- (iii) positive control: with the hour limit the same token is expired --


def test_with_the_hour_limit_the_same_old_token_is_rejected_as_expired(
        app, test_user, authenticated_client, csrf_enabled):
    cid = _make_conversion(app, test_user['id'])
    token = _token_stamped(authenticated_client, TWO_HOURS)
    assert _stamp_age(app, token) >= TWO_HOURS - 5

    previous = app.config.get('WTF_CSRF_TIME_LIMIT', _UNSET)
    app.config['WTF_CSRF_TIME_LIMIT'] = 3600
    try:
        resp = authenticated_client.put(f'/api/conversions/{cid}',
                                        json={'title': 'darf nicht landen'},
                                        headers={'X-CSRFToken': token})
    finally:
        if previous is _UNSET:
            app.config.pop('WTF_CSRF_TIME_LIMIT', None)
        else:
            app.config['WTF_CSRF_TIME_LIMIT'] = previous
    assert resp.status_code == 400
    body = resp.get_json()
    assert body['error'] == 'csrf_expired'
    assert 'expired' in body['message']
    with app.app_context():
        assert db.session.get(Conversion, cid).title == 'doc'


# --- Teil (b), server: the handler answers JSON to the wrapper -------------

def _multipart_probe():
    # What the document converter sends on /transform-document (non-PDF
    # path); the CSRF check runs in before_request, the view never sees it.
    return {'document_file': (io.BytesIO(b'CSRF-REFRESH Probe, kein Dokument'), 'probe.txt')}


def test_csrf_error_with_token_header_is_json_even_outside_api(app, test_user,
                                                               authenticated_client,
                                                               csrf_enabled):
    # Like the browser: the page's session holds a csrf_token (the form login
    # above ran with CSRF off, so mint one) and the header carries a stale value.
    assert authenticated_client.get('/api/csrf-token').status_code == 200
    resp = authenticated_client.post('/transform-document', data=_multipart_probe(),
                                     headers={'X-CSRFToken': 'stale'})
    assert resp.status_code == 400
    assert resp.content_type.startswith('application/json'), resp.content_type
    body = resp.get_json()
    assert body['error'] == 'csrf_expired'
    assert body['message'] == 'The CSRF token is invalid.'


def test_csrf_error_without_token_header_stays_the_html_page(app, test_user,
                                                             authenticated_client,
                                                             csrf_enabled):
    referer = 'http://localhost.test/document-converter'
    resp = authenticated_client.post('/transform-document', data=_multipart_probe(),
                                     headers={'Referer': referer})
    assert resp.status_code == 400
    assert resp.content_type.startswith('text/html'), resp.content_type
    text = resp.get_data(as_text=True)
    assert 'Session expired' in text
    assert f'<meta http-equiv="refresh" content="2;url={referer}">' in text


# --- Teil (b), belt for the retry: a session minted from the remember cookie --

def test_remember_minted_session_refreshes_its_token_through_the_endpoint(
        app, test_user, authenticated_client, csrf_enabled):
    domain = app.config['SERVER_NAME']          # the test client's cookie domain
    cid = _make_conversion(app, test_user['id'])
    old = authenticated_client.get('/api/csrf-token').get_json()['csrf_token']

    # The form login set remember_token (login_user(..., remember=True)).
    # Drop ONLY the session cookie — the browser's "session ended" case.
    assert authenticated_client.get_cookie('remember_token', domain=domain) is not None
    assert authenticated_client.get_cookie('session', domain=domain) is not None
    authenticated_client.delete_cookie('session', domain=domain)
    assert authenticated_client.get_cookie('session', domain=domain) is None

    stale = authenticated_client.put(f'/api/conversions/{cid}', json={'title': 'alt'},
                                     headers={'X-CSRFToken': old})
    assert stale.status_code == 400
    assert stale.get_json() == {'error': 'csrf_expired',
                                'message': 'The CSRF session token is missing.'}

    fresh = authenticated_client.get('/api/csrf-token')
    assert fresh.status_code == 200
    assert any(h.startswith('session=') for h in fresh.headers.getlist('Set-Cookie')), \
        fresh.headers.getlist('Set-Cookie')
    assert authenticated_client.get_cookie('session', domain=domain) is not None
    new = fresh.get_json()['csrf_token']
    assert new and new != old

    done = authenticated_client.put(f'/api/conversions/{cid}', json={'title': 'neu'},
                                    headers={'X-CSRFToken': new})
    assert done.status_code == 200
    with app.app_context():
        assert db.session.get(Conversion, cid).title == 'neu'
