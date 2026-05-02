"""Mermaid converter route."""
from flask import render_template
from flask_login import login_required


def register(app):
    @app.route('/mermaid-converter')
    @login_required
    def mermaid_converter():
        return render_template('mermaid_converter.html')
