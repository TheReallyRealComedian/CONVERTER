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

LOST-UPDATE: the section replace is the one read-modify-write over
``content`` (the new section is spliced into the text that was READ). It
writes through a conditional UPDATE on ``Conversion.content_version`` — the
counter every content writer bumps via ``Conversion.set_content`` — and on a
miss re-reads and re-splices into the other writer's text (bounded, then an
honest 409). Measured in P1 without it: 8 agents on their own sections of
one document lost 660 of 800 writes at 800 × HTTP 200.
"""
import logging

from flask import jsonify, request
from sqlalchemy import update

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

logger = logging.getLogger(__name__)

# LOST-UPDATE: attempts the section replace makes before it reports a content
# conflict as 409 — same rationale and size as cards.REVIEW_WRITE_ATTEMPTS:
# one agent edits one document at a time, the bound is for the pathological
# burst (N writers on ONE document at the same instant need up to N attempts
# for the last one). Measured, P3 (scripts/measure_lost_updates.py --section,
# 100 rounds × 8 writers on one document): see the sprint report.
CONTENT_WRITE_ATTEMPTS = 8


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

        # A full replacement overwrites by intent — no condition, but it bumps
        # the content version so a section replace that read the OLD text
        # cannot splice into it and resurrect it.
        conv.set_content(content)
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

        # LOST-UPDATE: splice into the text we READ, write only if nobody
        # changed the content in between (content_version unchanged since the
        # read); otherwise re-read and splice into the OTHER writer's text.
        # A Core UPDATE with the condition in its WHERE — not a mapper-wide
        # version on Conversion, which would make progress/place/reconcile
        # writes of the same row collide with content edits.
        tbl = Conversion.__table__
        for attempt in range(1, CONTENT_WRITE_ATTEMPTS + 1):
            if attempt > 1:
                db.session.rollback()  # forget the stale row; the re-query reads fresh
                conv = Conversion.query.filter_by(id=conversion_id, user_id=target.id).first()
                if conv is None:
                    return jsonify({'error': 'Nicht gefunden.'}), 404
            try:
                new_text = replace_section(conv.content, heading, content)
            except SectionNotFound:
                return jsonify({'error': 'Abschnitt nicht gefunden.'}), 404
            except SectionAmbiguous:
                return jsonify({'error': 'Abschnitt mehrdeutig (mehrere Headings gleichen Texts).'}), 409

            loaded = conv.content_version
            matched = db.session.execute(
                update(tbl)
                .where(tbl.c.id == conv.id, tbl.c.content_version == loaded)
                .values(content=new_text, content_version=loaded + 1)
            ).rowcount
            if matched == 1:
                db.session.commit()  # expires conv → to_dict reloads the written row
                return jsonify(conv.to_dict())
            logger.info('Content write conflict on conversion %s (attempt %d of %d)',
                        conversion_id, attempt, CONTENT_WRITE_ATTEMPTS)
        db.session.rollback()
        logger.warning('Content write on conversion %s gave up after %d conflicts',
                       conversion_id, CONTENT_WRITE_ATTEMPTS)
        return jsonify({'error': 'Das Dokument wurde gerade gleichzeitig geändert. '
                                 'Bitte noch einmal schreiben.'}), 409

    # Token-authed, session-less writes carry no CSRF cookie → waive CSRF for
    # these two views only (the session PUT stays under the global CSRFProtect).
    app.extensions['csrf'].exempt(api_update_document)
    app.extensions['csrf'].exempt(api_replace_section)
