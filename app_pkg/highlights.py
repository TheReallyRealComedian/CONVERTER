"""Highlight CRUD API for reader-view text annotations (R1-B-A).

Anchors follow the W3C Web Annotation Data Model (Text-Quote-Selector):
``exact`` is the marked text, ``prefix``/``suffix`` provide disambiguation
context for the client-side re-apply walker. See
``docs/reader_architecture.md`` for the design rationale.
"""
from flask import jsonify, request
from flask_login import current_user, login_required

from models import Highlight, db

from .library import get_owned_conversion


MAX_EXACT_LEN = 5000
MAX_CONTEXT_LEN = 200


def register(app):
    @app.route('/api/conversions/<int:conversion_id>/highlights', methods=['POST'])
    @login_required
    def api_create_highlight(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        exact = data.get('exact')
        if not isinstance(exact, str) or not exact.strip():
            return jsonify({'error': 'Markierungstext fehlt.'}), 400
        if len(exact) > MAX_EXACT_LEN:
            return jsonify({'error': f'Markierungstext zu lang (max {MAX_EXACT_LEN} Zeichen).'}), 400

        prefix = data.get('prefix', '') or ''
        suffix = data.get('suffix', '') or ''
        if not isinstance(prefix, str) or not isinstance(suffix, str):
            return jsonify({'error': 'Kontextfelder müssen Zeichenketten sein.'}), 400
        if len(prefix) > MAX_CONTEXT_LEN or len(suffix) > MAX_CONTEXT_LEN:
            return jsonify({'error': f'Kontext zu lang (max {MAX_CONTEXT_LEN} Zeichen pro Feld).'}), 400

        highlight = Highlight(
            conversion_id=conversion.id,
            exact=exact,
            prefix=prefix[:MAX_CONTEXT_LEN],
            suffix=suffix[:MAX_CONTEXT_LEN],
        )
        db.session.add(highlight)
        db.session.commit()
        return jsonify(highlight.to_dict()), 201

    @app.route('/api/conversions/<int:conversion_id>/highlights', methods=['GET'])
    @login_required
    def api_list_highlights(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        rows = (Highlight.query
                .filter_by(conversion_id=conversion.id)
                .order_by(Highlight.created_at.asc())
                .all())
        return jsonify([h.to_dict() for h in rows])

    @app.route('/api/highlights/<int:highlight_id>', methods=['DELETE'])
    @login_required
    def api_delete_highlight(highlight_id):
        highlight = Highlight.query.get_or_404(highlight_id)
        # 404 (not 403) so we don't leak existence of foreign rows.
        if highlight.conversion.user_id != current_user.id:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        db.session.delete(highlight)
        db.session.commit()
        return jsonify({'success': True})
