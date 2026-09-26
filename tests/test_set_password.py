"""SEC-SET-PASSWORD — ``flask set-password <user> [--revoke-tokens]``.

Der einzige Weg, ein Passwort zu wechseln (Befund 1 der MCP-Rueckmeldung zu
SEC-AUDIT: es gab keinen). Operator-Werkzeug im Web-Container, kein Endpoint,
keine UI.

Belegt wird der Vertrag aus dem Sprint-Prompt: alter Hash faellt, neuer greift
(``User.authenticate`` — der einzige Login-Pfad) · unbekannter Benutzer, zu
kurzes Passwort und ungleiche Wiederholung enden mit Exit ≠ 0 und lassen den
Hash stehen · ohne Flag bleiben die Tokens (Zahl + Liste in der Ausgabe) · mit
``--revoke-tokens`` verschwinden NUR die Tokens dieses Benutzers · die
Passwortregel ist EINE Stelle, die auch ``create-user`` faehrt.
"""
import re
from datetime import datetime

from app_pkg.mobile_auth import issue_token
from models import ApiToken, User, db

OLD = 'hunter2hunter2'
NEW = 'correct-horse-battery'


# --- helpers -----------------------------------------------------------------

def _make_user(app, username='alice', password=OLD):
    with app.app_context():
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        return user.id


def _make_token(app, user_id, label, expires_at=None):
    with app.app_context():
        user = db.session.get(User, user_id)
        issue_token(user, label=label)
        row = user.api_tokens.order_by(ApiToken.id.desc()).first()
        if expires_at is not None:
            row.expires_at = expires_at
            db.session.commit()
        return row.id


def _hash(app, user_id):
    with app.app_context():
        return db.session.get(User, user_id).password_hash


def _authenticates(app, username, password):
    with app.app_context():
        return User.authenticate(username, password) is not None


def _invoke(app, *args, password=NEW, confirm=None):
    """Drive the hidden prompt: password + repetition, each on its own line."""
    confirm = password if confirm is None else confirm
    return app.test_cli_runner().invoke(
        args=['set-password', *args], input=f'{password}\n{confirm}\n')


# --- the change itself -------------------------------------------------------

def test_change_makes_old_password_fail_and_new_one_work(app):
    _make_user(app)
    before = _hash(app, 1)

    result = _invoke(app, 'alice')

    assert result.exit_code == 0, result.output
    assert 'Passwort fuer "alice" (id 1) geaendert.' in result.output
    assert _hash(app, 1) != before
    assert not _authenticates(app, 'alice', OLD)
    assert _authenticates(app, 'alice', NEW)


def test_output_says_what_it_does_not_do(app):
    _make_user(app)
    result = _invoke(app, 'alice')
    assert result.exit_code == 0, result.output
    assert 'Browser-Sessions bleiben gueltig' in result.output
    assert 'SECRET_KEY' in result.output
    assert 'API-Tokens dieses Benutzers: keine.' in result.output


def test_password_never_appears_in_output(app):
    _make_user(app)
    result = _invoke(app, 'alice')
    assert NEW not in result.output


# --- failure paths: exit != 0, hash untouched ---------------------------------

def test_unknown_user_fails_before_the_prompt(app):
    user_id = _make_user(app)
    before = _hash(app, user_id)

    result = _invoke(app, 'nobody')

    assert result.exit_code != 0
    assert 'Benutzer "nobody" nicht gefunden.' in result.output
    assert 'Neues Passwort' not in result.output  # no prompt for a typo'd name
    assert _hash(app, user_id) == before


def test_too_short_password_fails(app):
    user_id = _make_user(app)
    before = _hash(app, user_id)

    result = _invoke(app, 'alice', password='short7!')

    assert result.exit_code != 0
    assert 'mindestens 8 Zeichen' in result.output
    assert _hash(app, user_id) == before
    assert _authenticates(app, 'alice', OLD)


def test_mismatched_confirmation_writes_nothing(app):
    user_id = _make_user(app)
    before = _hash(app, user_id)

    # click re-prompts on a mismatch; with the input exhausted it aborts.
    result = _invoke(app, 'alice', password=NEW, confirm='something-else-99')

    assert result.exit_code != 0
    assert _hash(app, user_id) == before
    assert _authenticates(app, 'alice', OLD)
    assert not _authenticates(app, 'alice', NEW)


def test_too_short_password_does_not_touch_tokens_even_with_flag(app):
    user_id = _make_user(app)
    _make_token(app, user_id, 'ios-app')

    result = _invoke(app, 'alice', '--revoke-tokens', password='short7!')

    assert result.exit_code != 0
    with app.app_context():
        assert ApiToken.query.filter_by(user_id=user_id).count() == 1


# --- tokens ------------------------------------------------------------------

def test_without_flag_tokens_stay_and_are_listed(app):
    user_id = _make_user(app)
    far = datetime(2099, 1, 1, 12, 0)
    t_ios = _make_token(app, user_id, 'ios-app')
    t_mcp = _make_token(app, user_id, 'converter-mcp', expires_at=far)
    t_old = _make_token(app, user_id, 'alt', expires_at=datetime(2020, 1, 1))

    result = _invoke(app, 'alice')

    assert result.exit_code == 0, result.output
    assert 'API-Tokens dieses Benutzers: 3 — bleiben gueltig.' in result.output
    assert re.search(rf'id {t_ios}\s+ios-app\s+erstellt', result.output)
    assert 'laeuft nie ab' in result.output
    assert re.search(rf'id {t_mcp}\s+converter-mcp\s+erstellt', result.output)
    assert 'laeuft ab 2099-01-01 12:00 UTC' in result.output
    assert re.search(rf'id {t_old}\s+alt\s+erstellt', result.output)
    assert '(abgelaufen)' in result.output
    assert '--revoke-tokens' in result.output
    with app.app_context():
        assert ApiToken.query.filter_by(user_id=user_id).count() == 3
    assert _authenticates(app, 'alice', NEW)


def test_with_flag_only_this_users_tokens_are_deleted(app):
    alice = _make_user(app, 'alice')
    bob = _make_user(app, 'bob')
    _make_token(app, alice, 'ios-app')
    _make_token(app, alice, 'ios-app')
    _make_token(app, alice, 'converter-mcp')
    bob_token = _make_token(app, bob, 'ios-app')

    result = _invoke(app, 'alice', '--revoke-tokens')

    assert result.exit_code == 0, result.output
    assert '3 API-Tokens widerrufen (converter-mcp x1, ios-app x2)' in result.output
    assert 'neue Tokens' in result.output
    with app.app_context():
        assert ApiToken.query.filter_by(user_id=alice).count() == 0
        assert [t.id for t in ApiToken.query.filter_by(user_id=bob)] == [bob_token]
    assert _authenticates(app, 'alice', NEW)
    assert _authenticates(app, 'bob', OLD)


def test_with_flag_and_no_tokens_says_so(app):
    _make_user(app)
    result = _invoke(app, 'alice', '--revoke-tokens')
    assert result.exit_code == 0, result.output
    assert 'nichts zu widerrufen' in result.output


# --- one rule, two commands --------------------------------------------------

def test_create_user_shares_the_password_rule(app):
    result = app.test_cli_runner().invoke(
        args=['create-user', 'carol'], input='short7!\nshort7!\n')
    assert result.exit_code != 0
    assert 'mindestens 8 Zeichen' in result.output
    with app.app_context():
        assert User.query.filter_by(username='carol').first() is None
