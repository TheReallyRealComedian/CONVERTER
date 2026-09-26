"""SEC-AUDIT P2 — baseline security headers on every kind of response.

Measured before (edge, 2026-09-25): none of X-Content-Type-Options,
X-Frame-Options, Referrer-Policy, Permissions-Policy. The after_request hook
must reach an HTML page, a JSON route, a send_file download, error answers
(401 JSON, 302 redirect) and static files alike. HSTS is deliberately absent
here — it belongs to the TLS terminator (nginx).
"""
import json
import os

import pytest

from models import Conversion, db

EXPECTED = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'camera=(), geolocation=(), payment=(), usb=()',
}


def _assert_headers(resp):
    for name, value in EXPECTED.items():
        assert resp.headers.get(name) == value, f'{name} on {resp.status_code}'


def test_html_page_carries_headers(client):
    resp = client.get('/login')
    assert resp.status_code == 200
    assert resp.mimetype == 'text/html'
    _assert_headers(resp)


def test_json_route_carries_headers(authenticated_client):
    resp = authenticated_client.get('/api/conversions')
    assert resp.status_code == 200
    assert resp.mimetype == 'application/json'
    _assert_headers(resp)


@pytest.fixture
def narration_output_dir(tmp_path, monkeypatch):
    d = tmp_path / 'narrations'
    d.mkdir()
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(d))
    monkeypatch.setattr('app_pkg.narration.OUTPUT_DIR', str(d))
    return str(d)


def test_send_file_download_carries_headers(app, test_user, authenticated_client,
                                            narration_output_dir):
    with app.app_context():
        conv = Conversion(user_id=test_user['id'], conversion_type='audio_narration',
                          title='t', content='**A:** x',
                          metadata_json=json.dumps({'narration_status': 'ready'}))
        db.session.add(conv)
        db.session.commit()
        cid = conv.id
    with open(os.path.join(narration_output_dir, f'narration_{cid}.wav'), 'wb') as f:
        f.write(b'RIFF\x24\x00\x00\x00WAVEfmt dummy')

    resp = authenticated_client.get(f'/api/narrations/{cid}/audio')
    assert resp.status_code == 200
    assert resp.mimetype == 'audio/wav'
    _assert_headers(resp)


def test_error_answers_carry_headers(client):
    unauthorized = client.get('/api/auth/me')           # 401 JSON (unauthorized_handler)
    assert unauthorized.status_code == 401
    _assert_headers(unauthorized)
    redirect = client.get('/library')                   # 302 to /login
    assert redirect.status_code == 302
    _assert_headers(redirect)


def test_static_file_carries_headers(client):
    resp = client.get('/static/js/_utils.js')
    assert resp.status_code == 200
    _assert_headers(resp)


def test_permissions_policy_leaves_the_microphone_alone(client):
    # The audio converter's live transcription needs getUserMedia.
    assert 'microphone' not in client.get('/login').headers['Permissions-Policy']
