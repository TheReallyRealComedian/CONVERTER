"""Recall-layer endpoints — global highlight reader, card write API, card/review
reads (R4-LEARN).

This module owns the spaced-repetition surface that sits over the existing
Highlights:

* ``GET /api/highlights/recent`` — the global reader (every doc the user owns)
  the agent polls to discover what to turn into cards. The per-doc highlight API
  (``app_pkg/highlights.py``) is strict pro-Conversion; this is the missing
  global view.
* ``POST``/``PATCH /api/cards`` — the agent's **token-authed** card writes,
  reusing the Ingest posture (see below).
* ``GET /api/cards``, ``GET /api/cards/<id>``, ``GET /api/review-state`` — the
  **session-authed**, owner-scoped reads (consistent with the GET-API the MCP
  already consumes).

Auth split (locked, R4-LEARN): writes use the Ingest token pattern with a
*separate* ``CARD_TOKEN`` (independent rotation); reads + the rate endpoint
(Phase 3) are ``@login_required``. The token compare is constant-time, the
endpoint is fail-closed without a configured secret, CSRF is waived for the two
write views only, and the token is never logged. The target user (writes have no
session) is resolved by the SAME ``INGEST_USER``/first() resolver Ingest uses,
so agent-authored cards land on the same account as ingested conversions.
"""
import hmac
import json
import logging
import os
from datetime import datetime, timezone

from flask import jsonify, render_template, request
from flask_login import current_user, login_required
from sqlalchemy.orm import contains_eager, joinedload

from models import Card, Collection, Conversion, Highlight, Review, Tag, db
from services.scheduler import RATINGS, get_scheduler

# Reuse the Ingest auth primitives so card writes resolve the SAME target user
# and parse the Bearer header identically — a single source of truth for "who
# does a session-less write belong to" (Memory reference_token_auth_ingest_endpoint).
from .ingest import _bearer_token, _resolve_target_user


logger = logging.getLogger(__name__)

RECENT_DEFAULT_LIMIT = 100
RECENT_MAX_LIMIT = 500
CARDS_DEFAULT_LIMIT = 100
CARDS_MAX_LIMIT = 500

CARD_TYPES = ('atomic', 'generative')
CARD_STATES = ('ok', 'wackelt')
MAX_CARD_NOTE_LEN = 2000


# --- shared parsing helpers --------------------------------------------------

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


def _parse_offset(raw):
    """Parse ``?offset=`` → a non-negative int, 0 on missing/garbage."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def _nonblank(value):
    """True iff ``value`` is a non-empty (post-strip) string."""
    return isinstance(value, str) and bool(value.strip())


def _naive_utc(dt):
    """Strip an aware datetime to naive UTC for storage (the column convention —
    SQLite holds naive UTC wall-clock). The scheduler returns aware-UTC."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None)


def _parse_owned(raw, model):
    """Resolve a numeric query value to the current user's row of ``model``, or
    None when it is non-numeric, unknown, or foreign (LERN-GROUP review scope).
    Callers map None → 404 so a foreign id leaks nothing. Shared by ``?tag=``
    and ``?collection=``."""
    try:
        obj_id = int(raw)
    except (TypeError, ValueError):
        return None
    obj = model.query.get(obj_id)
    if obj is None or obj.user_id != current_user.id:
        return None
    return obj


# --- card write helpers ------------------------------------------------------

def _authorize_card_write():
    """Shared token-auth gate for the card write endpoints (POST + PATCH).

    Returns ``(user, None)`` on success or ``(None, (response, status))`` on
    failure. Mirrors the Ingest posture exactly: fail-closed (503) without
    CARD_TOKEN, constant-time Bearer compare (401 on missing/wrong), token never
    logged. The target user is the Ingest resolver (INGEST_USER/first())."""
    expected = os.environ.get('CARD_TOKEN')
    if not expected:
        logger.warning('Card write rejected: CARD_TOKEN not configured')
        return None, (jsonify({'error': 'Card-API nicht konfiguriert.'}), 503)

    provided = _bearer_token()
    if provided is None or not hmac.compare_digest(provided.encode('utf-8'),
                                                   expected.encode('utf-8')):
        reason = 'missing bearer' if provided is None else 'token mismatch'
        logger.warning('Card write auth failed (%s) from %s', reason, request.remote_addr)
        return None, (jsonify({'error': 'Nicht autorisiert.'}), 401)

    target = _resolve_target_user()
    if target is None:
        logger.error('Card write rejected: no target user (INGEST_USER=%r)',
                     os.environ.get('INGEST_USER'))
        return None, (jsonify({'error': 'Kein Ziel-Benutzer vorhanden.'}), 503)

    return target, None


def _validate_card_type_payload(card_type, front, back, cloze_text, prompt):
    """Per-type content validation. Returns an error string (→ 400) or None.

    ``atomic`` needs (front AND back) OR cloze_text; ``generative`` needs prompt.
    """
    if card_type == 'atomic':
        has_front_back = _nonblank(front) and _nonblank(back)
        has_cloze = _nonblank(cloze_text)
        if not (has_front_back or has_cloze):
            return "Atomic-Karte braucht front und back oder cloze_text."
    elif card_type == 'generative':
        if not _nonblank(prompt):
            return "Generative Karte braucht prompt."
    return None


def _validate_highlight_ownership(highlight_id, user_id):
    """Validate an optional highlight_id reference. Returns an error string
    (→ 400) when set-but-invalid, else None. A bad number type, a missing
    highlight, or one owned by another user are all rejected — the provenance
    link must point at the target user's own highlight."""
    if highlight_id is None:
        return None
    # JSON numbers arrive as int; reject strings/floats/bools explicitly.
    if not isinstance(highlight_id, int) or isinstance(highlight_id, bool):
        return 'highlight_id muss eine Zahl sein.'
    hl = Highlight.query.filter_by(id=highlight_id).first()
    if hl is None or hl.conversion.user_id != user_id:
        return 'Ungültige highlight_id.'
    return None


def _replace_card_tags(card, names, user_id):
    """Replace a card's tags with the normalised get_or_create set."""
    card.tags = []
    if not isinstance(names, list):
        return
    for name in names:
        tag = Tag.get_or_create(user_id, name)
        if tag is not None and tag not in card.tags:
            card.tags.append(tag)


def _replace_highlight_tags(highlight, names, user_id):
    """Replace a highlight's tags with the normalised get_or_create set
    (shared vocabulary — identische Tag-Rows wie Card-/UI-Tags)."""
    highlight.tags = []
    if not isinstance(names, list):
        return
    for name in names:
        tag = Tag.get_or_create(user_id, name)
        if tag is not None and tag not in highlight.tags:
            highlight.tags.append(tag)


def _card_summary(card):
    """Slim per-row dict for the list endpoint — the question side + triage
    fields, no answer/snapshot bulk. (Full card is GET /api/cards/<id>.)"""
    review = card.review
    return {
        'id': card.id,
        'type': card.type,
        'state': card.state,
        'highlight_id': card.highlight_id,
        'front': card.front,
        'cloze_text': card.cloze_text,
        'prompt': card.prompt,
        'tags': [{'id': t.id, 'name': t.name} for t in card.tags],
        'due': review.due.isoformat() if review and review.due else None,
        'created_at': card.created_at.isoformat() if card.created_at else None,
    }


def register(app):
    @app.route('/review', methods=['GET'])
    @login_required
    def review_page():
        # The review UI shell. The due queue + counters are fetched client-side
        # from /api/review-state; the page itself is static chrome.
        return render_template('review.html')

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

    @app.route('/api/cards', methods=['POST'])
    def api_create_card():
        target, err = _authorize_card_write()
        if err:
            return err

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        card_type = data.get('type')
        if card_type not in CARD_TYPES:
            return jsonify({'error': "Feld 'type' muss 'atomic' oder 'generative' sein."}), 400

        front, back = data.get('front'), data.get('back')
        cloze_text, prompt = data.get('cloze_text'), data.get('prompt')
        type_error = _validate_card_type_payload(card_type, front, back, cloze_text, prompt)
        if type_error:
            return jsonify({'error': type_error}), 400

        highlight_id = data.get('highlight_id')
        hl_error = _validate_highlight_ownership(highlight_id, target.id)
        if hl_error:
            return jsonify({'error': hl_error}), 400

        card = Card(
            user_id=target.id,
            highlight_id=highlight_id,
            type=card_type,
            front=front,
            back=back,
            cloze_text=cloze_text,
            prompt=prompt,
            note=data.get('note'),
            source_snapshot=data.get('source_snapshot'),
            source_doc_title=data.get('source_doc_title'),
        )
        # Add before touching the tags collection: get_or_create's lookup
        # autoflushes, and the Tag.cards backref warns if the card isn't yet in
        # the session (the M2M row would be dropped from the tag side).
        db.session.add(card)
        _replace_card_tags(card, data.get('tags'), target.id)
        # Locked decision: create the Review row alongside the card in the
        # FSRS-"new" state — due now, reps/lapses 0, the rest NULL.
        card.review = Review(due=datetime.now(timezone.utc), reps=0, lapses=0)

        db.session.commit()
        return jsonify(card.to_dict()), 201

    @app.route('/api/cards/<int:card_id>', methods=['PATCH'])
    def api_patch_card(card_id):
        target, err = _authorize_card_write()
        if err:
            return err

        card = Card.query.filter_by(id=card_id, user_id=target.id).first()
        if card is None:
            # 404 (not 403) — never leak the existence of another user's card.
            return jsonify({'error': 'Karte nicht gefunden.'}), 404

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        if 'type' in data:
            if data['type'] not in CARD_TYPES:
                return jsonify({'error': "Feld 'type' muss 'atomic' oder 'generative' sein."}), 400
            card.type = data['type']
        if 'state' in data:
            if data['state'] not in CARD_STATES:
                return jsonify({'error': "Feld 'state' muss 'ok' oder 'wackelt' sein."}), 400
            card.state = data['state']
        if 'highlight_id' in data:
            hl_error = _validate_highlight_ownership(data['highlight_id'], target.id)
            if hl_error:
                return jsonify({'error': hl_error}), 400
            card.highlight_id = data['highlight_id']
        for field in ('front', 'back', 'cloze_text', 'prompt', 'note',
                      'source_snapshot', 'source_doc_title'):
            if field in data:
                setattr(card, field, data[field])
        if 'tags' in data:
            if not isinstance(data['tags'], list):
                return jsonify({'error': "Feld 'tags' muss eine Liste sein."}), 400
            _replace_card_tags(card, data['tags'], target.id)

        db.session.commit()  # updated_at bumps via the column onupdate
        return jsonify(card.to_dict())

    @app.route('/api/highlights/<int:highlight_id>/annotate', methods=['PATCH'])
    def api_annotate_highlight(highlight_id):
        # The agent's token-authed write-back onto an EXISTING highlight: set,
        # replace or clear its tags (persistent bucket-tagging) and/or its note.
        # Same posture as the card writes (token auth, CSRF-exempt), but here the
        # highlight_id is a PATH param = the addressed resource → a missing or
        # foreign highlight is 404, NOT the 400 the card writes give a bad *body*
        # highlight_id. The anchor keys (exact/prefix/suffix) are deliberately
        # ignored — the agent annotates, it never moves a marker. The session
        # note path (PATCH /api/highlights/<id>) stays the UI's.
        target, err = _authorize_card_write()
        if err:
            return err

        hl = Highlight.query.filter_by(id=highlight_id).first()
        if hl is None or hl.conversion.user_id != target.id:
            # 404 (not 403/400) — the id is the addressed resource; never leak
            # the existence of another user's highlight.
            return jsonify({'error': 'Nicht gefunden.'}), 404

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        if 'tags' not in data and 'note' not in data:
            return jsonify({'error': 'Nichts zu ändern (tags oder note erwartet).'}), 400

        if 'tags' in data:
            if not isinstance(data['tags'], list):
                return jsonify({'error': "Feld 'tags' muss eine Liste sein."}), 400
            _replace_highlight_tags(hl, data['tags'], target.id)
        if 'note' in data:
            note = data['note']
            if note is not None and not isinstance(note, str):
                return jsonify({'error': 'Notiz muss Text oder null sein.'}), 400
            if isinstance(note, str) and len(note) > MAX_CARD_NOTE_LEN:
                return jsonify({'error': f'Notiz zu lang (max {MAX_CARD_NOTE_LEN} Zeichen).'}), 400
            # Empty string is a delete-intent — store NULL, not "".
            hl.note = None if (isinstance(note, str) and note == '') else note

        db.session.commit()  # updated_at bumps via the column onupdate
        return jsonify(hl.to_dict())

    @app.route('/api/cards', methods=['GET'])
    @login_required
    def api_list_cards():
        query = (Card.query
                 .filter_by(user_id=current_user.id)
                 .options(joinedload(Card.review)))

        state = request.args.get('state')
        if state:
            query = query.filter(Card.state == state)

        highlight_id = request.args.get('highlight_id')
        if highlight_id:
            try:
                query = query.filter(Card.highlight_id == int(highlight_id))
            except ValueError:
                return jsonify({'error': 'highlight_id muss eine Zahl sein.'}), 400

        limit = _clamp_limit(request.args.get('limit'), CARDS_DEFAULT_LIMIT, CARDS_MAX_LIMIT)
        offset = _parse_offset(request.args.get('offset'))
        rows = (query.order_by(Card.created_at.desc())
                .limit(limit).offset(offset).all())
        return jsonify([_card_summary(c) for c in rows])

    @app.route('/api/cards/<int:card_id>', methods=['GET'])
    @login_required
    def api_get_card(card_id):
        card = Card.query.filter_by(id=card_id, user_id=current_user.id).first_or_404()
        return jsonify(card.to_dict())

    @app.route('/api/review-state', methods=['GET'])
    @login_required
    def api_review_state():
        # The due queue (due <= now) the Phase-4 review UI walks, newest-due
        # first, plus the counters. Full cards so the UI renders without an
        # extra fetch per card; contains_eager avoids the review N+1.
        #
        # LERN-GROUP scope filters (optional, combinable → AND):
        #   ?tag=<id>        cards whose tag is in the SUBTREE of <id> (Achse A)
        #   ?collection=<id> cards in collection <id> (Achse B)
        # Both chain onto the SAME `base`, so combining them ANDs and the
        # scope-correct total_count falls out for free. When scoped, total_count
        # reflects the SCOPE (cards in the scope, not just due ones); unscoped it
        # stays "all of the user's cards".
        now = datetime.now(timezone.utc)
        base = Card.query.filter_by(user_id=current_user.id)

        tag_arg = request.args.get('tag')
        if tag_arg is not None:
            tag = _parse_owned(tag_arg, Tag)
            if tag is None:
                return jsonify({'error': 'Tag nicht gefunden.'}), 404
            subtree = Tag.subtree_ids(tag.id, current_user.id)
            base = base.filter(Card.tags.any(Tag.id.in_(subtree)))

        collection_arg = request.args.get('collection')
        if collection_arg is not None:
            collection = _parse_owned(collection_arg, Collection)
            if collection is None:
                return jsonify({'error': 'Sammlung nicht gefunden.'}), 404
            base = base.filter(Card.collections.any(Collection.id == collection.id))

        due_cards = (base
                     .join(Card.review)
                     .filter(Review.due <= now)
                     .order_by(Review.due.asc())
                     .options(contains_eager(Card.review))
                     .all())
        total_count = base.count()
        return jsonify({
            'due_count': len(due_cards),
            'total_count': total_count,
            'due_cards': [c.to_dict() for c in due_cards],
        })

    @app.route('/api/cards/<int:card_id>/review', methods=['POST'])
    @login_required
    def api_review_card(card_id):
        # The USER rates in the review UI → SESSION auth (not the agent token).
        # No auto-grading: the rating always comes from the body.
        card = Card.query.filter_by(id=card_id, user_id=current_user.id).first_or_404()

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        rating = data.get('rating')
        if rating not in RATINGS:
            return jsonify({'error': "Feld 'rating' muss again|hard|good|easy sein."}), 400

        scheduler = get_scheduler()
        review = card.review
        if review is None:
            # Defensive: POST /api/cards always creates the row, but never assume.
            review = Review(card_id=card.id)
            db.session.add(review)
            current_state = scheduler.new_card_state()
        else:
            current_state = {
                'due': review.due,
                'stability': review.stability,
                'difficulty': review.difficulty,
                'last_reviewed': review.last_reviewed,
                'reps': review.reps or 0,
                'lapses': review.lapses or 0,
            }

        new_state = scheduler.apply_rating(current_state, rating)
        review.due = _naive_utc(new_state['due'])
        review.stability = new_state['stability']
        review.difficulty = new_state['difficulty']
        review.last_reviewed = _naive_utc(new_state['last_reviewed'])
        review.reps = new_state['reps']
        review.lapses = new_state['lapses']

        # Append to the rating history log (JSON list on the Review row).
        history = []
        if review.rating_history:
            try:
                parsed = json.loads(review.rating_history)
                if isinstance(parsed, list):
                    history = parsed
            except (ValueError, TypeError):
                history = []
        history.append({'rating': rating,
                        'reviewed_at': new_state['last_reviewed'].isoformat()})
        review.rating_history = json.dumps(history)

        # Generative card + weak rating optionally flags it shaky — the entry
        # point into the agent dialogue-recall ("Vertiefen", Phase 4).
        if card.type == 'generative' and rating in ('again', 'hard'):
            card.state = 'wackelt'

        db.session.commit()
        return jsonify(card.to_dict())

    @app.route('/api/cards/<int:card_id>/annotate', methods=['POST'])
    @login_required
    def api_annotate_card(card_id):
        # The session user's review-time card update: flag 'wackelt' (the
        # "Vertiefen" button → entry into agent dialogue-recall) and/or set the
        # inline note. The PATCH endpoint is agent-token-only, so the review UI
        # user (a session, no CARD_TOKEN) needs this separate session path.
        card = Card.query.filter_by(id=card_id, user_id=current_user.id).first_or_404()

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        if 'state' not in data and 'note' not in data:
            return jsonify({'error': 'Nichts zu ändern (state oder note erwartet).'}), 400

        if 'state' in data:
            if data['state'] not in CARD_STATES:
                return jsonify({'error': "Feld 'state' muss 'ok' oder 'wackelt' sein."}), 400
            card.state = data['state']
        if 'note' in data:
            note = data['note']
            if note is not None and not isinstance(note, str):
                return jsonify({'error': 'Notiz muss Text oder null sein.'}), 400
            if isinstance(note, str) and len(note) > MAX_CARD_NOTE_LEN:
                return jsonify({'error': f'Notiz zu lang (max {MAX_CARD_NOTE_LEN} Zeichen).'}), 400
            # Empty string is a delete-intent — store NULL, not "".
            card.note = None if (isinstance(note, str) and note == '') else note

        db.session.commit()
        return jsonify(card.to_dict())

    @app.route('/api/cards/<int:card_id>', methods=['DELETE'])
    @login_required
    def api_delete_card(card_id):
        # The session USER deletes THEIR OWN card from the review UI. Owner-scoped
        # via Card.user_id (foreign/missing id → 404, same first_or_404 posture as
        # api_annotate_card — never leak another user's card existence). Deleted
        # through the ORM, NOT raw SQL: card.review rides the 'all, delete-orphan'
        # cascade and the card_tags junction rows are swept via the secondary
        # relationship — a bare DELETE FROM card would orphan both (no
        # PRAGMA foreign_keys=ON). Session-write → stays under the global CSRF
        # protection (the base.html fetch wrapper sends X-CSRFToken); NOT exempt.
        card = Card.query.filter_by(id=card_id, user_id=current_user.id).first_or_404()
        db.session.delete(card)
        db.session.commit()
        return jsonify({'success': True})

    # Token-authed, session-less writes carry no CSRF cookie → waive CSRF for
    # THESE TWO views only (the reads stay protected by the global CSRFProtect).
    app.extensions['csrf'].exempt(api_create_card)
    app.extensions['csrf'].exempt(api_patch_card)
    app.extensions['csrf'].exempt(api_annotate_highlight)
