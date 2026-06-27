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

from models import Highlight, Tag, card_tags, conversion_tags, db, highlight_tags

# LERN-GROUP-AW: der Agent baut den Tag-Baum über den Token-Gate (CARD_TOKEN),
# nicht die Session — derselbe Gate wie Card-/Highlight-Annotate-/Doc-Write.
from .cards import _authorize_card_write as _authorize_agent_write


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
        card_counts = (db.session.query(
            card_tags.c.tag_id,
            func.count(card_tags.c.card_id).label('cnt'),
        )
            .group_by(card_tags.c.tag_id)
            .subquery())
        rows = (db.session.query(
            Tag, highlight_counts.c.cnt, conversion_counts.c.cnt, card_counts.c.cnt)
                .outerjoin(highlight_counts, Tag.id == highlight_counts.c.tag_id)
                .outerjoin(conversion_counts, Tag.id == conversion_counts.c.tag_id)
                .outerjoin(card_counts, Tag.id == card_counts.c.tag_id)
                .filter(Tag.user_id == current_user.id)
                .order_by(Tag.name.asc())
                .all())
        return jsonify([
            t.to_dict(highlight_count=int(hc or 0), conversion_count=int(cc or 0),
                      card_count=int(cardc or 0))
            for t, hc, cc, cardc in rows
        ])

    @app.route('/api/tags/<int:tag_id>', methods=['PATCH'])
    @login_required
    def api_update_tag(tag_id):
        # LERN-GROUP Achse A: ein Tag in den Wald einordnen (parent_id setzen
        # oder auf NULL = Wurzel lösen). Owner-scoped, Zyklus-Guard.
        tag = Tag.query.get_or_404(tag_id)
        if tag.user_id != current_user.id:
            return jsonify({'error': 'Nicht gefunden.'}), 404

        data = request.get_json(silent=True)
        if not isinstance(data, dict) or 'parent_id' not in data:
            return jsonify({'error': "Feld 'parent_id' fehlt (int oder null)."}), 400
        parent_id = data.get('parent_id')

        if parent_id is None:
            tag.parent_id = None
            db.session.commit()
            return jsonify(tag.to_dict())

        if not isinstance(parent_id, int) or isinstance(parent_id, bool):
            return jsonify({'error': "Feld 'parent_id' muss int oder null sein."}), 400

        parent = Tag.query.get(parent_id)
        if parent is None or parent.user_id != current_user.id:
            return jsonify({'error': 'Eltern-Tag nicht gefunden.'}), 400
        # Zyklus-Guard: das neue Eltern-Tag darf nicht das Tag selbst noch im
        # Teilbaum des Tags liegen — sonst hängt der Wald in einer Schleife.
        if parent_id in Tag.subtree_ids(tag_id, current_user.id):
            return jsonify({'error': 'Zyklus: Eltern-Tag liegt im Teilbaum.'}), 400

        tag.parent_id = parent_id
        db.session.commit()
        return jsonify(tag.to_dict())

    @app.route('/api/tags/parent', methods=['POST'])
    def api_set_tag_parent():
        # LERN-GROUP-AW Achse A, agent-write: der Agent baut den Tag-Baum
        # by-name über den Token-Gate. DISTINCT vom Session-PATCH
        # /api/tags/<id> (kein Path-Clash, by-name statt by-id, token statt
        # session). Body {tag: str, parent: str|null}. Beide Tags via
        # Tag.get_or_create (lowercased, shared vocabulary — "KI" → "ki",
        # konsistent mit den Karten-Tags). Spiegelt den Session-Zyklus-Guard.
        target, err = _authorize_agent_write()
        if err:
            return err

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        tag = Tag.get_or_create(target.id, data.get('tag'))
        if tag is None:
            return _tag_name_error()

        parent_name = data.get('parent')
        if parent_name is None:
            # Entwurzeln — Tag wird Wurzel.
            tag.parent_id = None
            db.session.commit()
            return jsonify(tag.to_dict())

        parent = Tag.get_or_create(target.id, parent_name)
        if parent is None:
            return _tag_name_error()
        # Zyklus-Guard (Spiegel der Session-Logik): das Eltern-Tag darf weder
        # das Tag selbst sein noch in dessen Teilbaum liegen — fängt
        # Selbst-Referenz (tag == parent nach Normalisierung) gleich mit.
        if parent.id in Tag.subtree_ids(tag.id, target.id):
            return jsonify({'error': 'Zyklus: Eltern-Tag liegt im Teilbaum.'}), 400

        tag.parent_id = parent.id
        db.session.commit()
        return jsonify(tag.to_dict())

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
        # LERN-GROUP: Kinder an die Wurzel reparenten, bevor das Tag verschwindet
        # — SQLite fährt ohne PRAGMA foreign_keys, das deklarierte ON DELETE ist
        # inert (Memory reference_sqlite_no_fk_pragma_orm_delete), sonst bliebe ein
        # totes parent_id zurück. Direkter UPDATE statt ORM-Iteration.
        db.session.execute(
            Tag.__table__.update()
            .where(Tag.parent_id == tag.id)
            .values(parent_id=None)
        )
        # Drain both M:N-Junctions manuell — SQLAlchemy cascade='all,delete-orphan'
        # greift bei secondary= nicht. Direkter DELETE auf den Junction-Tabellen ist
        # atomar und entgeht einer vollen Iteration über Highlights + Conversions.
        db.session.execute(
            highlight_tags.delete().where(highlight_tags.c.tag_id == tag.id)
        )
        db.session.execute(
            conversion_tags.delete().where(conversion_tags.c.tag_id == tag.id)
        )
        # card_tags (R4-LEARN) teilt dieselbe inerte ON-DELETE-Lage — auch hier
        # die Junction explizit drainen, sonst bleibt eine Karte mit totem Tag-Link.
        db.session.execute(
            card_tags.delete().where(card_tags.c.tag_id == tag.id)
        )
        db.session.delete(tag)
        db.session.commit()
        return jsonify({'success': True})

    # Token-authed, session-less write carries no CSRF cookie → waive CSRF for
    # THIS view only (the session reads/writes stay under the global CSRFProtect).
    app.extensions['csrf'].exempt(api_set_tag_parent)
