"""Application factory for the Flask CONVERTER app.

The factory wires up extensions (SQLAlchemy, Flask-Login, Flask-WTF) and
registers what every request needs. Its parts live in sibling modules since
ARCH-FACTORY P1: ``security`` (cookie interface, CSRF inversion, headers,
CSRF endpoint, error handlers), ``db_runtime`` (SQLite pragmas, startup lock,
data directory from the DB URI), ``migrations`` (inline ALTERs, index
reconcile, CSV→junction) and ``cli`` (operator commands). Routes are
registered separately in ``app.py`` — one ``register(app)`` per feature
module, no blueprints, so endpoint names stay flat.

Service singletons (``deepgram_service``, ``task_queue`` etc.) live in
``app.py`` so the existing test suite, which patches them at
``app.<name>``, continues to work without changes.

Until P2 turns this file into a lazy loader it imports the sibling modules
eagerly and re-exports the six names ``app.py`` and the tests take from the
package: ``create_app``, ``_run_pending_migrations``, ``_startup_lock``,
``_startup_lock_path``, ``_migrate_conversion_tags_csv_to_junction``,
``HttpsOnlySecureSessionInterface``.
"""
import logging
import os
import re
import sys
from datetime import timedelta

from flask import Flask, flash, jsonify, redirect, request
from flask_login import LoginManager, login_url
from flask_wtf.csrf import CSRFProtect
from markupsafe import Markup
from werkzeug.middleware.proxy_fix import ProxyFix

from app_pkg.cli import _register_cli_commands
from app_pkg.db_runtime import (  # noqa: F401 — _startup_lock_path is re-exported
    _ensure_sqlite_dir, _register_sqlite_pragmas, _startup_lock, _startup_lock_path)
from app_pkg.migrations import (  # noqa: F401 — the CSV migration is re-exported
    _migrate_conversion_tags_csv_to_junction, _run_pending_migrations)
from app_pkg.security import (
    HttpsOnlySecureSessionInterface, _register_csrf_endpoint, _register_csrf_inversion,
    _register_error_handlers, _register_security_headers)
from models import User, db


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_app(import_name='app'):
    """Build and return a Flask app with extensions wired up."""
    _configure_logging()

    app = Flask(import_name)

    # SEC-AUDIT: the app sits behind exactly ONE trusted proxy hop (host nginx,
    # which sets X-Forwarded-For/Proto/Host/Port). Without this, every request
    # looked like it came from the Docker gateway (measured: remote_addr
    # 172.21.0.1 for every client — logs blind, any rate-limit key on it one
    # bucket for everybody) and request.is_secure was False behind TLS.
    # Trusting one hop is only sound while nginx is the only way in from
    # outside — hence the 127.0.0.1 port bind in docker-compose.yml; the
    # containers on the Docker networks can still reach :5000 directly and
    # are trusted peers. Consequence, deliberate: behind nginx is_secure is
    # now True, so Flask-WTF's SSL-strict check applies to every cookie-
    # session write — it needs a same-origin Referer (browsers send one for
    # same-origin requests). X-Forwarded-Port 443 makes the Host "…:443";
    # Werkzeug's get_host drops the default port for https, so the browser
    # Referer still matches.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    secret_key = os.environ.get('SECRET_KEY')
    if not secret_key:
        raise RuntimeError("SECRET_KEY environment variable must be set")
    app.config['SECRET_KEY'] = secret_key
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB (large audio files)
    app.config['REMEMBER_COOKIE_HTTPONLY'] = True
    app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'
    # SEC-AUDIT: the remember cookie is always Secure; the session cookie
    # follows the request's scheme (HttpsOnlySecureSessionInterface).
    app.config['REMEMBER_COOKIE_SECURE'] = True
    # SEC-AUDIT: 30 days instead of Flask-Login's 365 (the web login always
    # sets remember=True) — Oli signs in in the browser about once a month.
    # ⚠️ Limit: the cookie value is ``user_id|digest``, no timestamp; this
    # sets only the browser's Expires. A STOLEN value stays valid until
    # SECRET_KEY rotates — rotation is the revocation lever, not this knob.
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.session_interface = HttpsOnlySecureSessionInterface()
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL', 'sqlite:////app/data/converter.db'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    csrf = CSRFProtect(app)
    _register_csrf_inversion(app, csrf)
    db.init_app(app)
    _register_sqlite_pragmas(app)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except (ValueError, TypeError):
            return None

    @login_manager.request_loader
    def load_user_from_bearer(req):
        # MOBILE-AUTH: per-user bearer token for the iOS app. Flask-Login
        # consults the request_loader only when neither session nor
        # remember-cookie yields a user (login_manager.py:_load_user), so the
        # cookie-based web path never reaches this code.
        header = req.headers.get('Authorization', '')
        if not header.startswith('Bearer '):
            return None
        token = header[len('Bearer '):].strip()
        if not token:
            return None
        from app_pkg.mobile_auth import resolve_token
        return resolve_token(token)

    @login_manager.unauthorized_handler
    def unauthorized():
        # MOBILE-AUTH: bearer clients need a real 401 — the stock behaviour
        # (302 to /login) would hand the app a login page. But the web UI's
        # session-expiry UX *depends* on that 302 (_utils.js safeJSON derives
        # its "Session expired" message from response.redirected, and raw
        # fetch call-sites check r.status themselves), so the 401 is scoped
        # strictly to requests that cannot come from the cookie web UI: a
        # Bearer header present, or the app-only /api/auth/* endpoints.
        # Every cookie-web request keeps today's redirect byte-identically.
        if (request.headers.get('Authorization', '').startswith('Bearer ')
                or request.path.startswith('/api/auth/')):
            return jsonify({'error': 'Nicht autorisiert.'}), 401
        # Reproduce flask_login's default unauthorized() for everything else
        # (flash + redirect-to-login with next=), see LoginManager.unauthorized.
        if login_manager.login_message:
            flash(login_manager.login_message,
                  category=login_manager.login_message_category)
        return redirect(login_url(login_manager.login_view, request.url))

    _register_error_handlers(app)
    _register_security_headers(app)
    _register_csrf_endpoint(app)
    _register_cli_commands(app)
    _register_template_filters(app)

    with app.app_context():
        _ensure_sqlite_dir(app.config['SQLALCHEMY_DATABASE_URI'])
        with _startup_lock(app.config['SQLALCHEMY_DATABASE_URI']):
            db.create_all()
            _run_pending_migrations(app)

    return app


DE_MONTH_ABBR = (
    'Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun',
    'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez',
)


def _register_template_filters(app):
    @app.template_filter('file_size')
    def file_size(bytes_value):
        # Mirror of static/js/_utils.js formatFileSize. Sub-MB rendered as KB
        # instead of "0.0 MB" — DE comma decimal.
        n = float(bytes_value or 0)
        if n < 1024:
            return f"{int(n)} B"
        if n < 1024 * 1024:
            return f"{n / 1024:.1f}".replace('.', ',') + ' KB'
        return f"{n / (1024 * 1024):.1f}".replace('.', ',') + ' MB'

    @app.template_filter('format_card_datetime')
    def format_card_datetime(dt):
        # Container-locale-agnostic DE month abbreviation. Mirrors the
        # %d %b %Y, %H:%M shape used in library cards.
        if dt is None:
            return ''
        return f"{dt.day:02d} {DE_MONTH_ABBR[dt.month - 1]} {dt.year}, {dt.hour:02d}:{dt.minute:02d}"

    _SCRIPT_END_RE = re.compile(r'</(script)', re.IGNORECASE)

    @app.template_filter('script_safe')
    def script_safe(value):
        # Inside a <script type="text/markdown"> block (used by library_detail
        # as the raw-source side-channel for Copy/Download/Notion-send), the
        # HTML parser only terminates at </script (case-insensitive). The
        # element is a *raw text element*: `<` and `&` are NOT decoded, so
        # Jinja2's auto-escape would turn `<div>` into `&lt;div&gt;` that
        # textContent then hands back to JS verbatim — breaking byte-equality
        # with the DB content. Mark the result as safe and patch only the
        # </script token so the rest of the Markdown stays byte-identical.
        if value is None:
            return Markup('')
        return Markup(_SCRIPT_END_RE.sub(r'<\\/\1', str(value)))
