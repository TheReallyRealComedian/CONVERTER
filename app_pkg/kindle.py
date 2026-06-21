"""Send a library element to the user's Kindle (KINDLE sprint).

``POST /api/conversions/<id>/send-to-kindle`` renders the conversion's Markdown
through the shared ``render_markdown_to_html``, wraps it into an EPUB, and mails
it to the server-fixed Kindle address. Session-write: ``@login_required``,
owner-scoped (foreign → 404), CSRF via the global ``base.html`` fetch wrapper
(not exempt). Fail-closed when Kindle isn't configured.
"""
from flask import jsonify
from flask_login import login_required
from werkzeug.utils import secure_filename

from services import kindle_service
from services.epub_service import build_epub

from .library import get_owned_conversion
from .markdown_render import render_markdown_to_html


def register(app):
    @app.route('/api/conversions/<int:conversion_id>/send-to-kindle', methods=['POST'])
    @login_required
    def send_conversion_to_kindle(conversion_id):
        conversion = get_owned_conversion(conversion_id)  # foreign/missing → 404

        if not kindle_service.is_configured():
            return jsonify({'error': 'Kindle nicht konfiguriert.'}), 503

        if not conversion.content:
            return jsonify({'error': 'Kein Inhalt zum Senden.'}), 400

        title = conversion.title or 'Dokument'
        html = render_markdown_to_html(conversion.content)
        epub = build_epub(title, html)
        safe_title = secure_filename(title) or 'dokument'

        try:
            kindle_service.send_to_kindle(f'{safe_title}.epub', epub, subject=title)
        except Exception:
            # Log the traceback (never the SMTP password, which isn't in it) and
            # shield the client from a 500 leak.
            app.logger.exception('Send-to-Kindle failed for conversion %s', conversion_id)
            return jsonify({'error': 'Versand an Kindle fehlgeschlagen.'}), 502

        return jsonify({'success': True})
