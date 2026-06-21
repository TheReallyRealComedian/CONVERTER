"""Sprint KINDLE Phase 2 — the Send-to-Kindle endpoint + mail service.

Mock boundary = the SMTP transport (``smtplib.SMTP_SSL`` / ``SMTP``): no real
network. Tests lock in the status mapping (404 foreign / 503 unconfigured /
400 empty / 502 SMTP-error / 302 anonymous), that a successful send builds the
right ``EmailMessage`` (To/From, exactly one ``.epub`` / ``application/epub+zip``
attachment), and the anti-relay guarantee — the recipient is the server-fixed
``KINDLE_TO_EMAIL`` and a ``to``/``recipient`` field in the request body is
ignored. The SMTP password must never reach the response body.
"""
import smtplib

import pytest

from models import db, Conversion, User


_KINDLE_ENV = {
    'KINDLE_SMTP_HOST': 'smtp.example.com',
    'KINDLE_SMTP_PORT': '465',
    'KINDLE_SMTP_USERNAME': 'sender@example.com',
    'KINDLE_SMTP_PASSWORD': 'super-secret-pw',
    'KINDLE_FROM_EMAIL': 'sender@example.com',
    'KINDLE_TO_EMAIL': 'olis-device@kindle.com',
}


def _configure_kindle(monkeypatch, **overrides):
    env = {**_KINDLE_ENV, **overrides}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


def _unconfigure_kindle(monkeypatch):
    # Defensive: clear any real KINDLE_* the dev box might carry.
    for key in _KINDLE_ENV:
        monkeypatch.delenv(key, raising=False)


def _make_conversion(app, user_id, **overrides):
    payload = dict(
        user_id=user_id,
        conversion_type='markdown_input',
        title='Mein Dokument',
        content='# Titel\n\nInhalt mit äöü.',
    )
    payload.update(overrides)
    with app.app_context():
        c = Conversion(**payload)
        db.session.add(c)
        db.session.commit()
        return c.id


def _make_other_user(app, username='bob'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


class _FakeSMTP:
    """Records connection params, login args, and the sent EmailMessage."""

    instances = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.login_args = None
        self.started_tls = False
        self.sent = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, username, password):
        self.login_args = (username, password)

    def starttls(self):
        self.started_tls = True

    def send_message(self, msg):
        self.sent.append(msg)


class _RaisingSMTP(_FakeSMTP):
    def send_message(self, msg):
        raise smtplib.SMTPException('simulated SMTP failure')


@pytest.fixture
def fake_smtp(monkeypatch):
    """Patch both SMTP transports with a recording fake (no real network)."""
    _FakeSMTP.instances = []
    monkeypatch.setattr(smtplib, 'SMTP_SSL', _FakeSMTP)
    monkeypatch.setattr(smtplib, 'SMTP', _FakeSMTP)
    return _FakeSMTP


def _url(cid):
    return f'/api/conversions/{cid}/send-to-kindle'


# --- status mapping ----------------------------------------------------------

def test_unconfigured_returns_503(app, authenticated_client, test_user, monkeypatch, fake_smtp):
    _unconfigure_kindle(monkeypatch)
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.post(_url(cid))
    assert resp.status_code == 503
    assert 'konfiguriert' in resp.get_json()['error'].lower()
    assert fake_smtp.instances == []  # SMTP never touched when unconfigured


def test_foreign_conversion_returns_404(app, authenticated_client, test_user, monkeypatch, fake_smtp):
    _configure_kindle(monkeypatch)
    other_id = _make_other_user(app)
    cid = _make_conversion(app, other_id)
    resp = authenticated_client.post(_url(cid))
    assert resp.status_code == 404
    assert fake_smtp.instances == []


def test_empty_content_returns_400(app, authenticated_client, test_user, monkeypatch, fake_smtp):
    _configure_kindle(monkeypatch)
    cid = _make_conversion(app, test_user['id'], content='')
    resp = authenticated_client.post(_url(cid))
    assert resp.status_code == 400
    assert 'inhalt' in resp.get_json()['error'].lower()
    assert fake_smtp.instances == []


def test_anonymous_is_redirected_to_login(client, app, test_user, monkeypatch):
    _configure_kindle(monkeypatch)
    cid = _make_conversion(app, test_user['id'])
    resp = client.post(_url(cid))
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


# --- success path ------------------------------------------------------------

def test_success_returns_200_and_builds_correct_message(
    app, authenticated_client, test_user, monkeypatch, fake_smtp
):
    env = _configure_kindle(monkeypatch)
    cid = _make_conversion(app, test_user['id'])

    resp = authenticated_client.post(_url(cid))
    assert resp.status_code == 200
    assert resp.get_json() == {'success': True}

    assert len(fake_smtp.instances) == 1
    smtp = fake_smtp.instances[0]
    assert smtp.login_args == (env['KINDLE_SMTP_USERNAME'], env['KINDLE_SMTP_PASSWORD'])
    assert smtp.timeout is not None  # connection timeout set, won't hang forever

    assert len(smtp.sent) == 1
    msg = smtp.sent[0]
    assert msg['To'] == env['KINDLE_TO_EMAIL']
    assert msg['From'] == env['KINDLE_FROM_EMAIL']

    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    att = attachments[0]
    assert att.get_content_type() == 'application/epub+zip'
    assert att.get_filename().endswith('.epub')
    assert att.get_content()[:2] == b'PK'  # real EPUB (ZIP) bytes attached


# --- SMTP failure ------------------------------------------------------------

def test_smtp_failure_returns_502_without_leaking_password(
    app, authenticated_client, test_user, monkeypatch
):
    env = _configure_kindle(monkeypatch)
    monkeypatch.setattr(smtplib, 'SMTP_SSL', _RaisingSMTP)
    cid = _make_conversion(app, test_user['id'])

    resp = authenticated_client.post(_url(cid))
    assert resp.status_code == 502
    assert 'fehlgeschlagen' in resp.get_json()['error'].lower()
    # No 500 leak, and the SMTP password never appears in the response body.
    assert env['KINDLE_SMTP_PASSWORD'] not in resp.get_data(as_text=True)


# --- anti-relay --------------------------------------------------------------

def test_recipient_is_server_fixed_ignoring_request_body(
    app, authenticated_client, test_user, monkeypatch, fake_smtp
):
    env = _configure_kindle(monkeypatch)
    cid = _make_conversion(app, test_user['id'])

    resp = authenticated_client.post(
        _url(cid),
        json={
            'to': 'attacker@evil.com',
            'recipient': 'evil@evil.com',
            'KINDLE_TO_EMAIL': 'evil@evil.com',
        },
    )
    assert resp.status_code == 200
    msg = fake_smtp.instances[0].sent[0]
    assert msg['To'] == env['KINDLE_TO_EMAIL']
    assert 'evil.com' not in (msg['To'] or '')
