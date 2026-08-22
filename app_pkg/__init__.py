"""Application factory for the Flask CONVERTER app.

The factory wires up extensions (SQLAlchemy, Flask-Login, Flask-WTF),
registers global error handlers, the CSRF-token endpoint, and CLI commands.
Routes are registered separately in ``app.py``; later steps of Stage 2 move
them into per-feature blueprints under this package.

Service singletons (``deepgram_service``, ``gemini_service`` etc.) live in
``app.py`` so the existing test suite, which patches them at
``app.<name>``, continues to work without changes.
"""
import fcntl
import logging
import os
import re
import sys
from contextlib import contextmanager

import click
from flask import Flask, flash, jsonify, redirect, request, url_for
from flask_login import LoginManager, login_required, login_url
from flask_wtf.csrf import CSRFError, CSRFProtect, generate_csrf
from markupsafe import Markup
from sqlalchemy import event, inspect, text
from sqlalchemy.engine import make_url

from app_pkg.config import SQLITE_BUSY_TIMEOUT_SECONDS
from models import Card, Collection, Review, User, db
from services.scheduler.base import initial_review_state


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

    secret_key = os.environ.get('SECRET_KEY')
    if not secret_key:
        raise RuntimeError("SECRET_KEY environment variable must be set")
    app.config['SECRET_KEY'] = secret_key
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB (large audio files)
    app.config['REMEMBER_COOKIE_HTTPONLY'] = True
    app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
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
    _register_csrf_endpoint(app)
    _register_cli_commands(app)
    _register_template_filters(app)

    with app.app_context():
        os.makedirs('/app/data', exist_ok=True)
        with _startup_lock(app.config['SQLALCHEMY_DATABASE_URI']):
            db.create_all()
            _run_pending_migrations(app)

    return app


def _register_sqlite_pragmas(app):
    """SYNC-FREEZE: WAL + an explicit ``busy_timeout`` on every SQLite connection.

    The app runs as several gunicorn processes on ONE SQLite file. In the
    rollback-journal mode the file was in (``PRAGMA journal_mode`` = ``delete``,
    never configured — ``SQLALCHEMY_ENGINE_OPTIONS`` was empty) a single
    writer locks the whole database against every reader; with one process
    that never showed, with N it would have turned a freeze into
    ``database is locked``. Hence WAL *before* workers (locked decision 1).

    Connection-level pragmas, so the SQLAlchemy ``connect`` event is the one
    place that reaches every pooled connection of every process. Values and
    reasoning: ``app_pkg.config.SQLITE_BUSY_TIMEOUT_SECONDS``.
    """
    with app.app_context():
        engine = db.engine
    if engine.dialect.name != 'sqlite':
        return

    @event.listens_for(engine, 'connect')
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            # busy_timeout FIRST: should the WAL switch itself have to wait
            # for a lock (first boot on a rollback-journal file while another
            # connection reads), it waits instead of failing.
            cursor.execute(
                f'PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_SECONDS * 1000}')
            cursor.execute('PRAGMA journal_mode=WAL')
        finally:
            cursor.close()


def _startup_lock_path(uri):
    """Lock file beside a file-based SQLite database, ``None`` otherwise."""
    try:
        url = make_url(uri)
    except Exception:
        return None
    if not url.drivername.startswith('sqlite'):
        return None
    database = url.database or ''
    if database in ('', ':memory:') or url.query.get('mode') == 'memory':
        return None
    return f'{database}.startup.lock'


@contextmanager
def _startup_lock(uri):
    """SYNC-FREEZE: serialise the schema bootstrap across worker processes.

    Every gunicorn worker runs ``create_app()`` — and with it
    ``db.create_all()`` and ``_run_pending_migrations`` — on its own. No
    ``--preload``: the SDK clients built at import time (a gRPC channel in
    ``GoogleTTSService``, the Deepgram/genai HTTP clients) are not fork-safe,
    and the per-process import is the model the app has always run under.
    Both bootstrap steps are idempotent on a settled schema, but on the first
    boot after a schema change N processes would race check-then-ALTER: the
    loser dies on ``duplicate column`` / ``table already exists``, and
    gunicorn treats a worker that fails to boot as fatal for the whole server
    (``Arbiter.reap_workers`` → ``HaltServer``). An ``flock`` beside the
    database file makes the bootstrap strictly sequential — the first process
    migrates, the others find the schema complete. Databases without a file
    (in-memory) need no lock.
    """
    path = _startup_lock_path(uri)
    if path is None:
        yield
        return
    with open(path, 'a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _register_csrf_inversion(app, csrf):
    """MOBILE-AUTH P2 — CSRF inversion for bearer writes.

    Flask-WTF's automatic per-request protection is turned off
    (``WTF_CSRF_CHECK_DEFAULT = False``); the ``before_request`` below
    re-applies it explicitly to every cookie-session mutation and skips it
    for bearer requests, so the iOS app can write without the cookie/CSRF
    dance while the web UI's CSRF posture stays byte-identical.

    The handler replicates the guard chain of Flask-WTF==1.2.1's automatic
    ``csrf_protect`` (flask_wtf/csrf.py::CSRFProtect.init_app: ENABLED →
    CHECK_DEFAULT → method → endpoint → exempt-blueprints → exempt-views →
    protect()), because ``csrf.protect()`` itself checks NONE of those
    guards — only the method. ``_exempt_views`` / ``_exempt_blueprints``
    are private Flask-WTF attributes: re-verify this replication against
    upstream on any Flask-WTF version bump.
    """
    app.config['WTF_CSRF_CHECK_DEFAULT'] = False

    @app.before_request
    def csrf_protect_session_writes():
        if not app.config['WTF_CSRF_ENABLED']:
            return
        if request.method not in app.config['WTF_CSRF_METHODS']:
            return
        # A cross-site browser request cannot carry an Authorization header
        # (custom headers need a CORS preflight, which fails without server
        # opt-in), so header *presence* is a CSRF-safe skip signal — the
        # same trust class as X-CSRFToken. Validity is deliberately NOT
        # checked here: an invalid bearer skips CSRF but dies at auth (401),
        # fail-closed. There is no cookie authority to ride on these
        # requests from a cross-site context.
        if request.headers.get('Authorization', '').startswith('Bearer '):
            return
        if not request.endpoint:
            return
        if app.blueprints.get(request.blueprint) in csrf._exempt_blueprints:
            return
        view = app.view_functions.get(request.endpoint)
        if view is not None:
            dest = f'{view.__module__}.{view.__name__}'
            if dest in csrf._exempt_views:
                return
        csrf.protect()


def _run_pending_migrations(app):
    # No Alembic/Flask-Migrate in this project, and db.create_all() does not
    # patch columns onto pre-existing tables. Each entry is idempotent —
    # it inspects the live schema first and only ALTERs when needed, so
    # repeated container starts are safe.
    inspector = inspect(db.engine)
    if 'highlight' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('highlight')}
        if 'note' not in cols:
            db.session.execute(text('ALTER TABLE highlight ADD COLUMN note TEXT'))
            db.session.commit()
            app.logger.info("R1-B-B: highlight.note column added via ALTER TABLE")
    if 'conversion' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('conversion')}
        if 'last_read_percent' not in cols:
            db.session.execute(text('ALTER TABLE conversion ADD COLUMN last_read_percent FLOAT'))
            db.session.commit()
            app.logger.info("R2-B: conversion.last_read_percent column added via ALTER TABLE")
        if 'lifecycle_status' not in cols:
            db.session.execute(text("ALTER TABLE conversion ADD COLUMN lifecycle_status VARCHAR(20) DEFAULT 'inbox'"))
            # Einmaliger differenzierter Backfill (läuft nur beim Spalten-Add → idempotent):
            # Newsletter bleiben im Inbox-Triage, alte Tool-Outputs ins Archive.
            db.session.execute(text("UPDATE conversion SET lifecycle_status='archive' WHERE conversion_type != 'ai_newsletter'"))
            db.session.commit()
            app.logger.info("R2-C: conversion.lifecycle_status added + backfilled (ai_newsletter→inbox, rest→archive)")
        if 'queue_position' not in cols:
            db.session.execute(text('ALTER TABLE conversion ADD COLUMN queue_position FLOAT'))
            # No backfill — NULL means "not on the reading list", so everyone
            # starts with an empty list. Idempotent via the column guard above.
            db.session.commit()
            app.logger.info("R2-D: conversion.queue_position added via ALTER TABLE")
    if 'user' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('user')}
        if 'settings_json' not in cols:
            db.session.execute(text('ALTER TABLE "user" ADD COLUMN settings_json TEXT'))
            # No backfill — NULL means "all defaults" (app_pkg/learn.py merges
            # stored values over the defaults). Idempotent via the column guard.
            db.session.commit()
            app.logger.info("LEARN-UP: user.settings_json column added via ALTER TABLE")
    if 'tag' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('tag')}
        if 'parent_id' not in cols:
            db.session.execute(text('ALTER TABLE tag ADD COLUMN parent_id INTEGER'))
            # No backfill — NULL means "root", so every existing tag starts at the
            # top of the forest. Idempotent via the column guard above.
            db.session.commit()
            app.logger.info("LERN-GROUP: tag.parent_id column added via ALTER TABLE")
    if 'card' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('card')}
        if 'front_svg' not in cols:
            db.session.execute(text('ALTER TABLE card ADD COLUMN front_svg TEXT'))
            db.session.commit()
            app.logger.info("CARD-SVG: card.front_svg column added via ALTER TABLE")
        if 'back_svg' not in cols:
            db.session.execute(text('ALTER TABLE card ADD COLUMN back_svg TEXT'))
            # No backfill — NULL means "no figure". Idempotent via the column
            # guards above.
            db.session.commit()
            app.logger.info("CARD-SVG: card.back_svg column added via ALTER TABLE")
    _migrate_conversion_tags_csv_to_junction(app)


def _migrate_conversion_tags_csv_to_junction(app):
    # R2-A: drain the legacy Conversion.tags CSV column into the new
    # conversion_tags junction. Idempotent via the empty-CSV marker —
    # once a row is migrated we set tags='' so the next container start
    # skips it. Defensive against User-Detach-then-Restart races: the CSV
    # is *not* re-read after the first run, so a deleted junction row will
    # not be resurrected from the dead column.
    from models import Conversion, Tag
    candidates = Conversion.query.filter(
        Conversion.tags.isnot(None),
        Conversion.tags != '',
    ).all()
    if not candidates:
        return
    migrated = 0
    for conv in candidates:
        names = [n.strip() for n in (conv.tags or '').split(',') if n.strip()]
        for name in names:
            tag = Tag.get_or_create(conv.user_id, name)
            if tag and tag not in conv.tag_refs:
                conv.tag_refs.append(tag)
        conv.tags = ''
        migrated += 1
    db.session.commit()
    app.logger.info(
        f"R2-A: migrated {migrated} conversions from CSV to conversion_tags junction"
    )


def _register_error_handlers(app):
    @app.errorhandler(413)
    def request_entity_too_large(error):
        if request.content_type and 'multipart/form-data' in request.content_type:
            return jsonify({'error': 'File too large. Maximum upload size is 500 MB.'}), 413
        return jsonify({'error': 'Request too large.'}), 413

    @app.errorhandler(CSRFError)
    def handle_csrf_error(error):
        if request.accept_mimetypes.best == 'application/json' or request.path.startswith('/api/'):
            return jsonify({'error': 'csrf_expired', 'message': str(error.description)}), 400
        reload_url = request.referrer or url_for('markdown_converter')
        html = (
            '<!DOCTYPE html><html><head><meta charset="UTF-8">'
            '<title>Session expired</title>'
            f'<meta http-equiv="refresh" content="2;url={reload_url}">'
            '<style>body{font-family:system-ui,sans-serif;max-width:520px;margin:4rem auto;'
            'padding:2rem;color:#333;text-align:center;}h1{font-size:1.2rem;margin-bottom:1rem;}'
            'p{color:#666;line-height:1.5;}</style></head><body>'
            '<h1>Session expired</h1>'
            '<p>Your security token expired. Reloading the page automatically&hellip;</p>'
            f'<p><a href="{reload_url}">Click here if nothing happens.</a></p>'
            '</body></html>'
        )
        return html, 400


def _register_csrf_endpoint(app):
    @app.route('/api/csrf-token', methods=['GET'])
    @login_required
    def get_csrf_token():
        return jsonify({'csrf_token': generate_csrf()})


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


def _register_cli_commands(app):
    @app.cli.command('create-user')
    @click.argument('username')
    @click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True)
    def create_user_cmd(username, password):
        """Create a new user account."""
        if len(password) < 8:
            click.echo('Error: Password must be at least 8 characters.')
            return
        if User.query.filter_by(username=username).first():
            click.echo(f'Error: User "{username}" already exists.')
            return
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        click.echo(f'User "{username}" created successfully.')

    @app.cli.command('reset-collection')
    @click.argument('collection')
    @click.option('--apply', 'apply_changes', is_flag=True,
                  help='Schreibt wirklich. Ohne den Flag wird nur berichtet.')
    def reset_collection_cmd(collection, apply_changes):
        """Setzt die bewerteten Karten EINER Sammlung auf "neu" zurueck.

        COLLECTION ist der Name ODER die id der Sammlung.

        LEARN-BACK: einmalige Korrektur vergifteter Scheduling-Daten, KEIN
        wiederkehrendes Werkzeug (deshalb CLI und kein Endpoint/kein Knopf).
        Alle Bewertungen vor LEARN-RATE trugen eine andere Semantik — "Schwer"
        hiess "kaum gewusst", der Scheduler bekam ``hard`` wo ``again``
        gehoerte. Die daraus gewachsene Stabilitaet ist nicht bloss zu hoch,
        sie ist erfunden; umdatieren wuerde das Phantom-Modell mitschleppen.

        Zwei gesperrte Entscheidungen:

        * Die Zeile wird feldweise auf ``initial_review_state()`` gesetzt —
          die EINE Definition von "neu" (beide Engines geben sie aus
          ``new_card_state`` zurueck). Eine nachgebaute Feldliste liefe bei
          der naechsten Scheduler-Aenderung still auseinander.
        * ``rating_history`` wird geleert: die Eintraege tragen die ungueltige
          Semantik, ``count_done_today`` klassifiziert neu-vs-Review am ERSTEN
          Eintrag (stehengelassen zaehlte die Karte gegen das Review- statt
          das Neu-Budget), und ``true_retention`` wuerde sie noch ~30 Tage
          weiterzaehlen. Preis ist der Audit-Trail; sein Wert ist durch die
          Semantik ohnehin zerstoert.

        Ausgewaehlt werden nur Karten mit ``stability IS NOT NULL`` (bereits
        bewertet) — damit ist der Lauf per Konstruktion idempotent: der zweite
        findet 0 Zeilen. Karteninhalte, Tags und Sammlungen bleiben unberuehrt.
        """
        target, error = _resolve_collection(collection)
        if error:
            raise click.ClickException(error)

        cards = (target.cards
                 .join(Card.review)
                 .filter(Review.stability.isnot(None))
                 .order_by(Card.id)
                 .all())

        mode = 'APPLY' if apply_changes else 'DRY-RUN'
        click.echo(f'Sammlung "{target.name}" (id {target.id}, user {target.user_id}) '
                   f'— {mode}')
        click.echo(f'Bewertete Karten: {len(cards)} von {target.cards.count()} '
                   f'in der Sammlung')

        # A card can sit in several collections; it is reset when the TARGET is
        # among them (the only sensible semantics), but name the overlap so a
        # surprise shows up before the --apply.
        overlaps = {}
        for card in cards:
            for other in card.collections:
                if other.id != target.id:
                    overlaps[other.name] = overlaps.get(other.name, 0) + 1
        if overlaps:
            detail = ', '.join(f'"{name}" ({count})'
                               for name, count in sorted(overlaps.items()))
            click.echo(f'Davon auch in anderen Sammlungen: {detail} '
                       f'— sie werden mit zurueckgesetzt.')

        if not cards:
            click.echo('Nichts zu tun.')
            return
        if not apply_changes:
            click.echo('Nichts geschrieben (Dry-run). Mit --apply ausfuehren.')
            return

        # `initial_review_state()` hands back AWARE UTC, the column convention
        # is naive UTC. SQLite drops the tzinfo silently and stores the wall
        # clock, which today happens to BE the naive UTC — right for the wrong
        # reason. Route `due` through the same `_naive_utc` every other write
        # path uses, so a future zone change in the scheduler cannot make this
        # one path write Berlin wall-clock as UTC. (`last_reviewed` is None;
        # `_naive_utc(None)` returns None, so the loop stays as it is. Local
        # import: app_pkg.cards imports app_pkg.learn → top-level is circular.)
        from app_pkg.cards import _naive_utc
        fresh = initial_review_state()
        fresh['due'] = _naive_utc(fresh['due'])
        for card in cards:
            for field, value in fresh.items():
                setattr(card.review, field, value)
            card.review.rating_history = None
        db.session.commit()
        click.echo(f'{len(cards)} Karten zurueckgesetzt — neu und sofort faellig.')


def _resolve_collection(raw):
    """Resolve a collection by id OR name → ``(collection, error_message)``.

    The id is unambiguous, the name is what a human types at the prompt, so
    both are accepted. Names are per-user unique but not globally, so a name
    hitting several users is reported as ambiguous rather than guessed.
    """
    if raw.isdigit():
        found = Collection.query.filter_by(id=int(raw)).first()
        if found is not None:
            return found, None
    name = Collection.normalize_name(raw)
    matches = Collection.query.filter_by(name=name).all() if name else []
    if not matches:
        return None, f'Sammlung "{raw}" nicht gefunden (weder als id noch als Name).'
    if len(matches) > 1:
        owners = ', '.join(f'id {c.id} (user {c.user_id})' for c in matches)
        return None, (f'Sammlung "{name}" ist mehrdeutig: {owners}. '
                      f'Bitte die id angeben.')
    return matches[0], None
