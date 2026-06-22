"""Document-content write API — the agent's token-authed edits to a Conversion's
Markdown body (MCP-DOCWRITE).

Two PATCH endpoints, both session-less and **token-authed** with the SAME gate
the card writes use (``CARD_TOKEN`` — generic agent-write token despite the
card-y name; this is its third surface after card writes and highlight-annotate):

* ``PATCH /api/conversions/<id>/content`` — full replacement of ``content``
  (tool ``update_document``).
* ``PATCH /api/conversions/<id>/section`` — replace one heading-addressed
  section (tool ``replace_section``); the parsing lives in the pure
  ``services.markdown_sections`` module.

Auth posture mirrors the card writes exactly: fail-closed 503 without
``CARD_TOKEN``, constant-time 401, target user resolved server-side via
``INGEST_USER``/first() — never from the request. A foreign/missing conversion
is 404 (never leak existence, never 403/400). Both views are CSRF-exempt
(session-less Bearer writes carry no CSRF token). The session/CSRF-protected
``PUT /api/conversions/<id>`` (the UI/editor path) is deliberately untouched —
these are separate sub-paths so there's no method+path clash.
"""
from flask import jsonify, request

from models import Conversion, db
from services.markdown_sections import (
    replace_section,
    SectionNotFound,
    SectionAmbiguous,
)

# Reuse the generic agent-write gate (CARD_TOKEN, fail-closed, constant-time,
# INGEST_USER target) and the non-blank-string check — single source of truth,
# no churn on cards.py. The alias reads neutrally at the use site.
from .cards import _authorize_card_write as _authorize_agent_write, _nonblank


def register(app):
    @app.route('/api/conversions/<int:conversion_id>/content', methods=['PATCH'])
    def api_update_document(conversion_id):
        target, err = _authorize_agent_write()
        if err:
            return err

        conv = Conversion.query.filter_by(id=conversion_id, user_id=target.id).first()
        if conv is None:
            # 404 (not 403/400) — never leak another user's conversion.
            return jsonify({'error': 'Nicht gefunden.'}), 404

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        content = data.get('content')
        # Non-blank string required — guards against an agent bug wiping the doc
        # with an empty/missing/non-str content.
        if not _nonblank(content):
            return jsonify({'error': 'Feld content (nicht-leerer Text) erwartet.'}), 400

        conv.content = content
        db.session.commit()  # updated_at bumps via the column onupdate
        return jsonify(conv.to_dict())

    @app.route('/api/conversions/<int:conversion_id>/section', methods=['PATCH'])
    def api_replace_section(conversion_id):
        target, err = _authorize_agent_write()
        if err:
            return err

        conv = Conversion.query.filter_by(id=conversion_id, user_id=target.id).first()
        if conv is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        heading = data.get('heading')
        content = data.get('content')  # the new section (incl. its own heading)
        if not _nonblank(heading) or not _nonblank(content):
            return jsonify({'error': 'Felder heading und content (nicht-leerer Text) erwartet.'}), 400

        try:
            new_text = replace_section(conv.content, heading, content)
        except SectionNotFound:
            return jsonify({'error': 'Abschnitt nicht gefunden.'}), 404
        except SectionAmbiguous:
            return jsonify({'error': 'Abschnitt mehrdeutig (mehrere Headings gleichen Texts).'}), 409

        conv.content = new_text
        db.session.commit()
        return jsonify(conv.to_dict())

    # Token-authed, session-less writes carry no CSRF cookie → waive CSRF for
    # these two views only (the session PUT stays under the global CSRFProtect).
    app.extensions['csrf'].exempt(api_update_document)
    app.extensions['csrf'].exempt(api_replace_section)
