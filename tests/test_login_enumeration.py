"""SEC-AUDIT P2 — anti-enumeration: one password-hash check per login attempt.

Measured before (local test client, median of 20): the WEB login answered an
unknown user in 0.4 ms and a known user with a wrong password in 47.6 ms —
the response time named valid usernames. The mobile login already burned a
dummy hash. Now both go through ``User.authenticate``: exactly ONE
``check_password_hash`` call per attempt, against the dummy hash when the
user does not exist. Mock-based on purpose — no timing assertion in the suite
(the timing measurement lives in the audit report).
"""
from unittest.mock import patch

import pytest

import models
from models import DUMMY_PASSWORD_HASH


@pytest.fixture
def hash_calls():
    """Record every check_password_hash call while keeping the real result."""
    with patch('models.check_password_hash', wraps=models.check_password_hash) as spy:
        yield spy


def _hashes_checked(spy):
    return [call.args[0] for call in spy.call_args_list]


def test_web_login_unknown_user_burns_the_dummy_hash(client, hash_calls):
    resp = client.post('/login', data={'username': 'nobody', 'password': 'x'})
    assert resp.status_code == 200  # form re-rendered with the generic flash
    assert _hashes_checked(hash_calls) == [DUMMY_PASSWORD_HASH]


def test_web_login_empty_username_burns_the_dummy_hash(client, hash_calls):
    client.post('/login', data={'username': '   ', 'password': 'x'})
    assert _hashes_checked(hash_calls) == [DUMMY_PASSWORD_HASH]


def test_web_login_known_user_wrong_password_checks_the_real_hash_once(
        app, client, test_user, hash_calls):
    with app.app_context():
        real_hash = models.User.query.filter_by(username=test_user['username']).one().password_hash
    resp = client.post('/login', data={'username': test_user['username'], 'password': 'wrong'})
    assert resp.status_code == 200
    assert _hashes_checked(hash_calls) == [real_hash]


def test_web_login_success_still_logs_in(client, test_user):
    resp = client.post('/login', data={'username': test_user['username'],
                                       'password': test_user['password']})
    assert resp.status_code == 302


def test_mobile_login_unknown_user_burns_the_same_dummy_hash(client, hash_calls):
    resp = client.post('/api/auth/login', json={'username': 'nobody', 'password': 'x'})
    assert resp.status_code == 401
    assert _hashes_checked(hash_calls) == [DUMMY_PASSWORD_HASH]


def test_one_definition_for_both_logins():
    # The mobile login's private copy is gone — a second dummy hash would be
    # a second place for the two logins to drift apart again.
    import app_pkg.mobile_auth as mobile_auth
    assert not hasattr(mobile_auth, '_DUMMY_PASSWORD_HASH')
