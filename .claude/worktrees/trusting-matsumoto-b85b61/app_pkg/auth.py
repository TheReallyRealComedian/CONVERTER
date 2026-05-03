"""Auth routes: login, logout.

Routes are registered via ``register(app)`` rather than a Flask
``Blueprint`` so endpoint names stay flat (``login`` instead of
``auth.login``). That keeps existing ``url_for(...)`` calls in the Jinja
templates working — touching templates is reserved for Stage 5.
"""
from urllib.parse import urlparse

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from models import User


def register(app):
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('markdown_converter'))
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user, remember=True)
                next_page = request.args.get('next')
                if next_page:
                    parsed = urlparse(next_page)
                    if parsed.netloc or parsed.scheme:
                        next_page = None
                return redirect(next_page or url_for('markdown_converter'))
            flash('Invalid username or password.', 'danger')
        return render_template('login.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('login'))
