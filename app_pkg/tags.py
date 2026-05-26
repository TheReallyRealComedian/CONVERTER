"""Tag API + Tag-Manager-Page (R1-B-C + R2-A).

Tags are a per-user namespace (unique(user_id, name)). Two junctions share
the same Tag rows: ``highlight_tags`` (R1-B-C) wires a Highlight to N Tags,
``conversion_tags`` (R2-A) wires a Conversion to N Tags. Name normalisation
(lowercase + trim) lives on ``Tag.get_or_create`` so all three call sites —
highlight POST, conversion POST, the CSV migration helper — collapse
``"KI"`` / ``"ki"`` / ``" KI "`` onto the same row.
"""
from flask import jsonify, render_template, request
from flask_login import current_user, login_required
from sqlalchemy import func

from models import Highlight, Tag, conversion_tags, db, highlight_tags


def _get_owned_highlight(highlight_id):
    highlight = Highlight.query.get_or_404(highlight_id)
    # 404 (not 403) so existence of foreign rows doesn't leak.
    if highlight.conversion.user_id != current_user.id:
        return None
    return highlight


def _tag_name_error():
    return jsonify({
        'error': f'Tag-Name fehlt oder ist zu lang (max {Tag.MAX_NAME_LEN} Zeichen).'
    }), 400


def register(app):
    @app.route('/tags')
    @login_required
    def tags_page():
        return render_template('tags.html')

    @app.route('/api/tags', methods=['GET'])
    @login_required
    def api_list_tags():
        highlight_counts = (db.session.query(
            highlight_tags.c.tag_id,
            func.count(highlight_tags.c.highlight_id).label('cnt'),
        )
            .group_by(highlight_tags.c.tag_id)
            .subquery())
        conversion_counts = (db.session.query(
            conversion_tags.c.tag_id,
            func.count(conversion_tags.c.conversion_id).label('cnt'),
        )
            .group_by(conversion_tags.c.tag_id)
            .subquery())
        rows = (db.session.query(Tag, highlight_counts.c.cnt, conversion_counts.c.cnt)
                .outerjoin(highlight_counts, Tag.id == highlight_counts.c.tag_id)
                .outerjoin(conversion_counts, Tag.id == conversion_counts.c.tag_id)
                .filter(Tag.user_id == current_user.id)
                .order_by(Tag.name.asc())
                .all())
        return jsonify([
            t.to_dict(highlight_count=int(hc or 0), conversion_count=int(cc or 0))
            for t, hc, cc in rows
        ])

    @app.route('/api/highlights/<int:highlight_id>/tags', methods=['POST'])
    @login_required
    def api_attach_tag(highlight_id):
        highlight = _get_owned_highlight(highlight_id)
        if highlight is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        tag = Tag.get_or_create(current_user.id, data.get('name'))
        if tag is None:
            return _tag_name_error()

        if tag in highlight.tags:
            # Idempotent attach — no-op 200 statt 409, sonst crashen schnelle Doppel-Clicks.
            db.session.commit()
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
        # Drain both M:N-Junctions manuell — SQLAlchemy cascade='all,delete-orphan'
        # greift bei secondary= nicht. Direkter DELETE auf den Junction-Tabellen ist
        # atomar und entgeht einer vollen Iteration über Highlights + Conversions.
        db.session.execute(
            highlight_tags.delete().where(highlight_tags.c.tag_id == tag.id)
        )
        db.session.execute(
            conversion_tags.delete().where(conversion_tags.c.tag_id == tag.id)
        )
        db.session.delete(tag)
        db.session.commit()
        return jsonify({'success': True})
