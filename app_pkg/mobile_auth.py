"""Per-user bearer auth for the native iOS app — ``POST /api/auth/login``.

Sprint MOBILE-AUTH: the app exchanges username/password (JSON) once for an
opaque per-user bearer token and sends ``Authorization: Bearer <token>`` on
every call. The ``login_manager.request_loader`` wired in
``app_pkg/__init__.py`` resolves that token into a ``User``, so ALL
``@login_required`` views accept it without per-view changes.

Security posture (this is a new public credential surface — keep it tight,
mirrors ``ingest.py``):

* **Hashed storage.** Only ``sha256(token)`` is stored (``ApiToken.token_hash``);
  the plaintext leaves the server exactly once, in the login response body.
  A DB leak exposes no live tokens.
* **Generic 401 on ANY login failure.** Unknown username burns a dummy
  ``check_password_hash`` so response timing does not separate "no such
  user" from "wrong password" (anti-enumeration).
* **Tokens never logged.** Auth failures log remote_addr and a coarse
  reason only; the success log carries the user id, never the token.
* **Fail-closed expiry.** ``expires_at`` in the past == invalid token;
  revocation is a row delete, effective on the next request.
* **CSRF-exempt is scoped to this one view** (a session-less JSON caller has
  no CSRF cookie), same mechanism as the other token endpoints. The bearer
  bypass for *writes* on shared views is Phase 2 (CSRF inversion).
"""
import hashlib
import logging
import secrets
from datetime import datetime, timezone

from flask import jsonify, request
from flask_login import current_user, login_required
from werkzeug.security import check_password_hash, generate_password_hash

from models import ApiToken, User, db

logger = logging.getLogger(__name__)

TOKEN_LABEL_MAX = 80

# Burned on login attempts for unknown usernames so both failure paths do
# the same password-hash work — no timing split (anti-enumeration). Random
# input: this hash must never accidentally match a real password.
_DUMMY_PASSWORD_HASH = generate_password_hash(secrets.token_hex(16))


def _hash_token(plaintext):
    return hashlib.sha256(plaintext.encode('utf-8')).hexdigest()


def issue_token(user, label=None):
    """Create an ``ApiToken`` row for ``user`` and return the PLAINTEXT token.

    The plaintext is neither stored nor logged — the caller hands it to the
    client exactly once.
    """
    plaintext = secrets.token_urlsafe(32)
    db.session.add(ApiToken(user_id=user.id,
                            token_hash=_hash_token(plaintext),
                            label=label or None))
    db.session.commit()
    return plaintext


def resolve_token(plaintext):
    """Map a presented bearer token onto its ``User``, or ``None``.

    sha256 → indexed lookup → fail-closed expiry check → ``last_used_at``
    bump. Hashing before lookup removes attacker control over the compared
    bytes, and a B-tree lookup over uniform-random digests is no timing
    oracle — the constant-time property lives in the hash step.
    """
    if not plaintext:
        return None
    row = ApiToken.query.filter_by(token_hash=_hash_token(plaintext)).first()
    if row is None:
        return None
    now = datetime.now(timezone.utc)
    expires = row.expires_at
    if expires is not None:
        if expires.tzinfo is None:
            # SQLite hands back naive datetimes; stored values are UTC.
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= now:
            return None
    row.last_used_at = now
    db.session.commit()
    return row.user


def register(app):
    @app.route('/api/auth/login', methods=['POST'])
    def api_auth_login():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = {}
        username = data.get('username')
        password = data.get('password')
        # Mirror the web login's .strip() on username (auth.py) so the same
        # credentials work on both paths; non-str values fail generically.
        username = username.strip() if isinstance(username, str) else ''
        password = password if isinstance(password, str) else ''

        user = User.query.filter_by(username=username).first() if username else None
        if user is not None:
            authenticated = user.check_password(password)
        else:
            check_password_hash(_DUMMY_PASSWORD_HASH, password)
            authenticated = False

        if not authenticated:
            logger.warning('Mobile login failed from %s', request.remote_addr)
            return jsonify({'error': 'Nicht autorisiert.'}), 401

        label = data.get('label')
        if not isinstance(label, str) or not label.strip():
            label = 'ios-app'
        token = issue_token(user, label=label.strip()[:TOKEN_LABEL_MAX])
        logger.info('Mobile login: token issued for user id %s', user.id)
        return jsonify({'token': token,
                        'user': {'id': user.id, 'username': user.username}}), 200

    @app.route('/api/auth/me', methods=['GET'])
    @login_required
    def api_auth_me():
        # The app validates a stored token at launch. Identity comes from the
        # request_loader (bearer) — or a session, which is harmless: the
        # response only mirrors the caller's own identity.
        return jsonify({'id': current_user.id, 'username': current_user.username})

    @app.route('/api/auth/logout', methods=['POST'])
    @login_required
    def api_auth_logout():
        # Revokes the PRESENTED token (row delete). Idempotent at row level:
        # a double-tap whose second request still authenticated before the
        # first one deleted the row simply deletes 0 rows and stays 200.
        # (After the delete, the token no longer authenticates — a later
        # call is a 401 at the door, which the P2 tests pin down.)
        # A session caller presents no bearer: nothing to revoke, still 200.
        header = request.headers.get('Authorization', '')
        revoked = 0
        if header.startswith('Bearer '):
            token = header[len('Bearer '):].strip()
            if token:
                revoked = (ApiToken.query
                           .filter_by(token_hash=_hash_token(token))
                           .delete())
                db.session.commit()
        return jsonify({'ok': True, 'revoked': bool(revoked)}), 200

    # Session-less JSON caller has no CSRF cookie — exempt exactly this view,
    # same mechanism as the other token endpoints (see ingest.py). me is a
    # GET (never CSRF-checked); logout stays unexempted — bearer callers pass
    # via the P2 inversion skip, and a cookie-session POST to it remains a
    # protected session mutation like any other.
    app.extensions['csrf'].exempt(api_auth_login)
