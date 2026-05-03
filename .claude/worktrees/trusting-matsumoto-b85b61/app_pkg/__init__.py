"""Application factory for the Flask CONVERTER app.

The factory wires up extensions (SQLAlchemy, Flask-Login, Flask-WTF),
registers global error handlers, the CSRF-token endpoint, and CLI commands.
Routes are registered separately in ``app.py``; later steps of Stage 2 move
them into per-feature blueprints under this package.

Service singletons (``deepgram_service``, ``gemini_service`` etc.) live in
``app.py`` so the existing test suite, which patches them at
``app.<name>``, continues to work without changes.
"""
import logging
import os
import sys

import click
from flask import Flask, jsonify, request, url_for
from flask_login import LoginManager, login_required
from flask_wtf.csrf import CSRFError, CSRFProtect, generate_csrf

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

    CSRFProtect(app)
    db.init_app(app)

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

    _register_error_handlers(app)
    _register_csrf_endpoint(app)
    _register_cli_commands(app)

    with app.app_context():
        os.makedirs('/app/data', exist_ok=True)
        db.create_all()

    return app


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
