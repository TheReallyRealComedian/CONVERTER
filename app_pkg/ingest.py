"""Server-to-server ingestion endpoint — ``POST /api/ingest/conversion``.

This is the project's **first non-session endpoint**: no Flask-Login cookie,
authenticated instead by a shared-secret Bearer token. It exists so an external
app (``email-automation``, Sprint NL2) can *push* converted content — initially
AI newsletters — into the Library without a browser session.

Security posture (NL1 — keep this tight, it is a new auth surface):

* **CSRF-exempt, and ONLY this view.** A session-less, token-authenticated
  caller has no CSRF cookie to present, so the global ``CSRFProtect`` is waived
  for this one endpoint via ``app.extensions['csrf'].exempt(...)``. No other
  route is exempted.
* **Token compare is constant-time** (``hmac.compare_digest``) so a wrong token
  cannot be recovered byte-by-byte via timing.
* **Fail-closed.** With ``INGEST_TOKEN`` unset/empty the endpoint accepts
  nothing (503) — it is never open without a configured secret.
* **The token value is never logged.** Auth failures are logged (so probing is
  visible) with the remote address and a coarse reason only.

The endpoint is generic: it accepts any ``conversion_type`` in
``ALLOWED_CONVERSION_TYPES`` (the token is the trust boundary). Idempotency is
keyed on an optional ``source_id`` stored in ``metadata_json`` — no schema
touch. ``topics[]`` map onto the R2-A conversion-tags junction via
``Tag.get_or_create``; ``report_date`` (ISO) maps onto ``created_at``.

Body contract (see ``docs/ingest_contract.md``)::

    POST /api/ingest/conversion
    Authorization: Bearer <INGEST_TOKEN>
    Content-Type: application/json
    {
      "conversion_type": "ai_newsletter",   # must be in ALLOWED_CONVERSION_TYPES
      "title":           "...",             # default "Untitled", clipped to 255
      "content":         "# ... markdown",  # required
      "topics":          ["...", "..."],    # optional list -> tags (non-list ignored)
      "report_date":     "2026-05-30",       # optional ISO date -> created_at
      "source_id":       "<stable id>"      # optional -> dedup key
    }
"""
import hmac
import json
import logging
import os
import re
from datetime import datetime

from flask import jsonify, request

from models import Conversion, Tag, User, db

from .library import ALLOWED_CONVERSION_TYPES

logger = logging.getLogger(__name__)


def _bearer_token():
    """Extract the token from an ``Authorization: Bearer <token>`` header, or
    None if the header is missing or not a well-formed Bearer header."""
    header = request.headers.get('Authorization', '')
    prefix = 'Bearer '
    if not header.startswith(prefix):
        return None
    token = header[len(prefix):].strip()
    return token or None


def _resolve_target_user():
    """Pick the user that ingested rows are attributed to. If ``INGEST_USER``
    is set we resolve that exact username (and intentionally do NOT fall back —
    a set-but-missing username is a misconfiguration that should 503, not
    silently land on the wrong account). Otherwise the single-user fallback."""
    username = os.environ.get('INGEST_USER')
    if username:
        return User.query.filter_by(username=username).first()
    return User.query.first()


def _parse_report_date(value):
    """Parse an optional ISO date/datetime string for ``created_at``. Returns
    None on anything unparsable so the caller can fall back to the column
    default (now). ``datetime.fromisoformat`` accepts both "2026-05-30" and a
    full ISO datetime."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def _find_by_source_id(user_id, source_id):
    """Idempotency lookup. Volume is tiny (≈ weekly newsletters), so we narrow
    with a LIKE prefilter on the metadata_json text and then confirm with an
    exact parse — a substring collision can never dedup the wrong row, and the
    LIKE can only over-match (so it never misses a real hit). No schema touch:
    ``source_id`` lives inside ``metadata_json``."""
    escaped = re.sub(r'([%_\\])', r'\\\1', source_id)
    candidates = (Conversion.query
                  .filter_by(user_id=user_id)
                  .filter(Conversion.metadata_json.like(f'%{escaped}%'))
                  .all())
    for c in candidates:
        try:
            meta = json.loads(c.metadata_json) if c.metadata_json else {}
        except (ValueError, TypeError):
            continue
        if isinstance(meta, dict) and meta.get('source_id') == source_id:
            return c
    return None


def register(app):
    @app.route('/api/ingest/conversion', methods=['POST'])
    def api_ingest_conversion():
        # --- Auth (fail-closed, constant-time, token never logged) ---
        expected = os.environ.get('INGEST_TOKEN')
        if not expected:
            logger.warning('Ingestion rejected: INGEST_TOKEN not configured')
            return jsonify({'error': 'Ingestion nicht konfiguriert.'}), 503

        provided = _bearer_token()
        if provided is None or not hmac.compare_digest(provided.encode('utf-8'),
                                                        expected.encode('utf-8')):
            reason = 'missing bearer' if provided is None else 'token mismatch'
            logger.warning('Ingestion auth failed (%s) from %s', reason, request.remote_addr)
            return jsonify({'error': 'Nicht autorisiert.'}), 401

        # --- Target user (no session) ---
        target = _resolve_target_user()
        if target is None:
            logger.error('Ingestion rejected: no target user (INGEST_USER=%r)',
                         os.environ.get('INGEST_USER'))
            return jsonify({'error': 'Kein Ziel-Benutzer vorhanden.'}), 503

        # --- Body validation (mirrors the /api/conversions create route) ---
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        conversion_type = data.get('conversion_type', 'unknown')
        if conversion_type not in ALLOWED_CONVERSION_TYPES:
            return jsonify({'error': f'Invalid conversion type: {conversion_type}'}), 400

        content = data.get('content')
        if not content:
            return jsonify({'error': 'Content is required'}), 400

        title = (data.get('title') or 'Untitled')[:255]
        source_id = data.get('source_id')
        if source_id is not None and not isinstance(source_id, str):
            source_id = None

        # --- Dedup (idempotent over source_id) ---
        if source_id:
            existing = _find_by_source_id(target.id, source_id)
            if existing is not None:
                payload = existing.to_dict()
                payload['deduped'] = True
                return jsonify(payload), 200

        # --- Create ---
        metadata = {'ingested': True}
        if source_id:
            metadata['source_id'] = source_id

        kwargs = dict(
            user_id=target.id,
            conversion_type=conversion_type,
            title=title,
            content=content,
            metadata_json=json.dumps(metadata),
        )
        report_date = _parse_report_date(data.get('report_date'))
        if report_date is not None:
            kwargs['created_at'] = report_date

        conversion = Conversion(**kwargs)
        db.session.add(conversion)

        # topics[] -> tags via the R2-A junction. A non-list ``topics`` is
        # ignored (lenient — a malformed topics field never blocks an otherwise
        # valid newsletter); Tag.get_or_create skips non-str/blank/oversize.
        topics = data.get('topics')
        if isinstance(topics, list):
            for topic in topics:
                tag = Tag.get_or_create(target.id, topic)
                if tag is not None and tag not in conversion.tag_refs:
                    conversion.tag_refs.append(tag)

        db.session.commit()
        return jsonify(conversion.to_dict()), 201

    # First (and only) non-session endpoint: a token-authenticated server-to-
    # server caller has no CSRF cookie, so waive CSRF for THIS view only.
    app.extensions['csrf'].exempt(api_ingest_conversion)
