"""Recall-layer endpoints — global highlight reader, card API, review (R4-LEARN).

This module owns the spaced-repetition surface that sits over the existing
Highlights:

* **Phase 1** — ``GET /api/highlights/recent``: the global reader (every doc the
  user owns) the agent polls to discover what to turn into cards. The per-doc
  highlight API (``app_pkg/highlights.py``) is strict pro-Conversion; this is
  the missing global view.
* **Phase 2/3** (added later) — the token-authed card write API and the
  session-authed card/review reads + rate endpoint.

Auth split (locked): reads are ``@login_required`` (Session, consistent with the
GET-API the MCP already consumes); card writes use the Ingest token pattern.
"""
from datetime import datetime, timezone

from flask import jsonify, request
from flask_login import current_user, login_required

from models import Conversion, Highlight


RECENT_DEFAULT_LIMIT = 100
RECENT_MAX_LIMIT = 500


def _parse_since(value):
    """Parse the optional ``?since=`` ISO timestamp into a UTC-naive datetime
    (the storage shape — SQLite drops tzinfo on write). Returns None on anything
    unparsable so the caller simply skips the filter. An aware input is
    converted to UTC then stripped, so a 'Z'/offset timestamp and a naive,
    round-tripped ``created_at`` both compare correctly against the column."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _clamp_limit(raw, default, cap):
    """Parse ``?limit=`` → an int in [1, cap], falling back to ``default`` on a
    missing/garbage/non-positive value."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return default
    if n < 1:
        return default
    return min(n, cap)


def register(app):
    @app.route('/api/highlights/recent', methods=['GET'])
    @login_required
    def api_highlights_recent():
        # Global reader over every doc the user owns — the agent's entry point
        # for "what got highlighted since <t>". Owner-scoped via the join.
        query = (Highlight.query
                 .join(Conversion, Highlight.conversion_id == Conversion.id)
                 .filter(Conversion.user_id == current_user.id))

        since = _parse_since(request.args.get('since'))
        if since is not None:
            query = query.filter(Highlight.created_at >= since)

        limit = _clamp_limit(request.args.get('limit'),
                             RECENT_DEFAULT_LIMIT, RECENT_MAX_LIMIT)
        rows = (query.order_by(Highlight.created_at.desc())
                .limit(limit)
                .all())

        return jsonify([{
            'id': h.id,
            'exact': h.exact,
            'note': h.note,
            'tags': [{'id': t.id, 'name': t.name} for t in h.tags],
            'created_at': h.created_at.isoformat() if h.created_at else None,
            'conversion_id': h.conversion_id,
            'title': h.conversion.title,
        } for h in rows])
