"""Tag API + Tag-Manager-Page (R1-B-C).

Tags are a per-user namespace (unique(user_id, name)). The ``highlight_tags``
junction wires a Highlight to N Tags; the ``conversion_tags`` junction is
deferred to R2-A. Tag names are normalised to lowercase + trimmed at the
API boundary so ``"KI"``, ``"ki"`` and ``" KI "`` collapse onto the same row
— this is a deliberate disambiguation choice, not a SQL constraint.
"""
from flask import jsonify, render_template, request
from flask_login import current_user, login_required
from sqlalchemy import func

from models import Highlight, Tag, db, highlight_tags


MAX_TAG_NAME_LEN = 80


def _normalize_tag_name(raw):
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().lower()
    if not cleaned:
        return None
    if len(cleaned) > MAX_TAG_NAME_LEN:
        return None
    return cleaned


def _get_owned_highlight(highlight_id):
    highlight = Highlight.query.get_or_404(highlight_id)
    # 404 (not 403) so existence of foreign rows doesn't leak.
    if highlight.conversion.user_id != current_user.id:
        return None
    return highlight


def register(app):
    @app.route('/tags')
    @login_required
    def tags_page():
        return render_template('tags.html')

    @app.route('/api/tags', methods=['GET'])
    @login_required
    def api_list_tags():
        counts_subq = (db.session.query(
            highlight_tags.c.tag_id,
            func.count(highlight_tags.c.highlight_id).label('cnt'),
        )
            .group_by(highlight_tags.c.tag_id)
            .subquery())
        rows = (db.session.query(Tag, counts_subq.c.cnt)
                .outerjoin(counts_subq, Tag.id == counts_subq.c.tag_id)
                .filter(Tag.user_id == current_user.id)
                .order_by(Tag.name.asc())
                .all())
        return jsonify([t.to_dict(highlight_count=int(cnt or 0)) for t, cnt in rows])

    @app.route('/api/highlights/<int:highlight_id>/tags', methods=['POST'])
    @login_required
    def api_attach_tag(highlight_id):
        highlight = _get_owned_highlight(highlight_id)
        if highlight is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        name = _normalize_tag_name(data.get('name'))
        if name is None:
            return jsonify({'error': f'Tag-Name fehlt oder ist zu lang (max {MAX_TAG_NAME_LEN} Zeichen).'}), 400

        tag = Tag.query.filter_by(user_id=current_user.id, name=name).first()
        if tag is None:
            tag = Tag(user_id=current_user.id, name=name)
            db.session.add(tag)
            db.session.flush()

        if tag in highlight.tags:
            # Idempotent attach — no-op 200 statt 409, sonst crashen schnelle Doppel-Clicks.
            return jsonify(tag.to_dict()), 200

        highlight.tags.append(tag)
        db.session.commit()
        return jsonify(tag.to_dict()), 201

    @app.route('/api/highlights/<int:highlight_id>/tags/<int:tag_id>', methods=['DELETE'])
    @login_required
    def api_detach_tag(highlight_id, tag_id):
        highlight = _get_owned_highlight(highlight_id)
        if highlight is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        tag = Tag.query.get_or_404(tag_id)
        if tag.user_id != current_user.id:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        if tag in highlight.tags:
            highlight.tags.remove(tag)
            db.session.commit()
        return jsonify({'success': True})

    @app.route('/api/tags/<int:tag_id>', methods=['DELETE'])
    @login_required
    def api_delete_tag(tag_id):
        tag = Tag.query.get_or_404(tag_id)
        if tag.user_id != current_user.id:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        # Drain die M:N-Junction manuell — SQLAlchemy cascade='all,delete-orphan' geht
        # bei secondary= nicht, also explizit. Direkter DELETE auf der highlight_tags-
        # Tabelle ist atomar und entgeht der vollen Highlights-Iteration.
        db.session.execute(
            highlight_tags.delete().where(highlight_tags.c.tag_id == tag.id)
        )
        db.session.delete(tag)
        db.session.commit()
        return jsonify({'success': True})
