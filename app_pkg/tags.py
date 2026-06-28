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
from sqlalchemy import func, select

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


def _require_tag_name(raw):
    """Validate + normalise a by-name lookup field for the destructive token
    tools. Returns ``(normalized, None)`` or ``(None, error_response)``.

    A *truthy* non-string (``{"source": 123}``) must 400, not 500 — passing it
    straight to ``normalize_name`` would ``.replace`` on a non-str and raise an
    AttributeError. Mirrors the ``get_or_create`` isinstance guard."""
    if not isinstance(raw, str):
        return None, _tag_name_error()
    norm = Tag.normalize_name(raw)
    if not norm:
        return None, _tag_name_error()
    return norm, None


def _is_dry_run(data):
    """Strict dry-run read for the destructive tools: the write fires ONLY on a
    real JSON ``false``. Default true; any non-False value (a falsy non-bool like
    ``0``/``""`` OR a ``"false"`` string) stays a safe dry-run — a typo must never
    silently apply a destructive run."""
    return data.get('dry_run', True) is not False


def _is_force(data):
    """Strict force read: the guard-rail lifts ONLY on a real JSON ``true``."""
    return data.get('force', False) is True


# TAG-CLEANUP: die drei Junctions teilen alle die Tag-Rows. Jeder Reassign
# (merge_tags ODER delete_tag mit reassign_to) muss alle drei umhängen.
_TAG_JUNCTIONS = (
    ('cards', card_tags, card_tags.c.card_id),
    ('highlights', highlight_tags, highlight_tags.c.highlight_id),
    ('conversions', conversion_tags, conversion_tags.c.conversion_id),
)


def _reassign_tag_refs(source_id, target_id):
    """Dedup-then-repoint a source tag's junction rows onto a target tag,
    across all three junctions (TAG-CLEANUP). Shared by merge_tags and
    delete_tag(reassign_to=...).

    Per junction: first DELETE the source rows whose object ALSO carries the
    target tag — otherwise the repoint UPDATE would mint a duplicate / hit the
    composite-PK constraint. Then UPDATE the surviving source rows onto the
    target. SQLite runs without ``PRAGMA foreign_keys`` so the junctions are
    handled explicitly (Memory reference_sqlite_no_fk_pragma_orm_delete).

    Returns ``{'cards': {'moved': N, 'deduped': M}, 'highlights': {…},
    'conversions': {…}}`` from the live statement ``rowcount`` — accurate even
    in a dry-run that rolls back afterwards (counts are read before rollback)."""
    out = {}
    for key, junction, obj_col in _TAG_JUNCTIONS:
        deduped = db.session.execute(
            junction.delete()
            .where(junction.c.tag_id == source_id)
            .where(obj_col.in_(
                select(obj_col).where(junction.c.tag_id == target_id)
            ))
        )
        moved = db.session.execute(
            junction.update()
            .where(junction.c.tag_id == source_id)
            .values(tag_id=target_id)
        )
        out[key] = {'moved': moved.rowcount, 'deduped': deduped.rowcount}
    return out


def _tag_object_counts(tag_id):
    """The (cards, highlights, conversions) attach-count of a tag across the
    three junctions. Used by delete_tag to drive the force-guard-rail."""
    out = {}
    for key, junction, _obj_col in _TAG_JUNCTIONS:
        out[key] = db.session.execute(
            select(func.count()).select_from(junction)
            .where(junction.c.tag_id == tag_id)
        ).scalar() or 0
    return out


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

    @app.route('/api/tags/merge', methods=['POST'])
    def api_merge_tags():
        # TAG-CLEANUP: destruktiver Merge — alle Refs von source auf target
        # umhängen, source-Kinder → target reparenten, source löschen. Token-Gate
        # (CARD_TOKEN), by-name Lookup-only (KEIN get_or_create — ein Merge
        # konsolidiert auf ein BEKANNTES Kanon-Tag, kein Phantom-Ziel aus einem
        # Tippfehler; Lookup-only hält den dry-run garantiert schreibfrei).
        # Body {source, target, dry_run=true}. dry_run=true = DEFAULT (same-path-
        # rollback → apply-treue Vorschau).
        target_user, err = _authorize_agent_write()
        if err:
            return err

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        source_norm, err = _require_tag_name(data.get('source'))
        if err:
            return err
        target_norm, err = _require_tag_name(data.get('target'))
        if err:
            return err
        dry_run = _is_dry_run(data)

        # source == target nach Normalisierung → No-op (nichts zu mergen).
        if source_norm == target_norm:
            return jsonify({
                'dry_run': dry_run, 'applied': False,
                'source': {'id': None, 'name': source_norm},
                'target': {'id': None, 'name': target_norm},
                'reassigned': {k: {'moved': 0, 'deduped': 0}
                               for k in ('cards', 'highlights', 'conversions')},
                'children_reparented': [], 'source_deleted': False,
            }), 200

        source = Tag.query.filter_by(user_id=target_user.id, name=source_norm).first()
        if source is None:
            return jsonify({'error': 'Quell-Tag nicht gefunden.'}), 404
        target = Tag.query.filter_by(user_id=target_user.id, name=target_norm).first()
        if target is None:
            return jsonify({'error': 'Ziel-Tag nicht gefunden.'}), 404

        # Zyklus-Guard: Ziel darf nicht echter Nachfahre der Quelle sein — sonst
        # baute das Reparenten der Kinder (parent_id=target) eine Schleife.
        if target.id in Tag.subtree_ids(source.id, target_user.id) and target.id != source.id:
            return jsonify({
                'error': 'Merge würde einen Zyklus erzeugen (Ziel liegt im '
                         'Teilbaum der Quelle) — erst via set_tag_parent entwirren.'
            }), 400

        # IDs/Namen JETZT in Python-Werte ziehen — nach einem dry-run-Rollback
        # sind die ORM-Objekte expired (Memory: vor Rollback kopieren).
        source_id, source_name = source.id, source.name
        target_id, target_name = target.id, target.name

        counts = _reassign_tag_refs(source_id, target_id)

        # Kinder → target reparenten (Guard garantiert target ∉ source-Teilbaum).
        child_rows = db.session.execute(
            select(Tag.id, Tag.name).where(Tag.parent_id == source_id)
        ).all()
        children_reparented = [{'id': cid, 'name': cname} for cid, cname in child_rows]
        db.session.execute(
            Tag.__table__.update()
            .where(Tag.parent_id == source_id)
            .values(parent_id=target_id)
        )

        db.session.delete(source)

        if dry_run:
            db.session.rollback()
        else:
            db.session.commit()

        return jsonify({
            'dry_run': dry_run, 'applied': not dry_run,
            'source': {'id': source_id, 'name': source_name},
            'target': {'id': target_id, 'name': target_name},
            'reassigned': counts,
            'children_reparented': children_reparented,
            'source_deleted': not dry_run,
        }), 200

    @app.route('/api/tags/delete', methods=['POST'])
    def api_delete_tag_token():
        # TAG-CLEANUP: destruktives Löschen über den Token-Gate (CARD_TOKEN).
        # Name DISTINCT von der Session-api_delete_tag (by-id, bleibt unberührt).
        # Body {tag, reassign_to=null, dry_run=true, force=false}.
        #  • mit reassign_to → Refs auf reassign_to umhängen (geteilter Helper),
        #    Kinder → reassign_to, tag löschen.
        #  • ohne reassign_to, tag HAT Objekte, force=false → GUARD-RAIL: dry-run
        #    liefert requires_force ohne 409, echter Lauf → 409 (nichts geschrieben).
        #  • ohne reassign_to (keine Objekte ODER force=true) → detach-all über die
        #    drei Junctions + Kinder → NULL (Wurzel, wie Session-Delete) + löschen.
        # dry_run=true = DEFAULT (same-path-rollback, apply-treu).
        target_user, err = _authorize_agent_write()
        if err:
            return err

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        tag_norm, err = _require_tag_name(data.get('tag'))
        if err:
            return err
        dry_run = _is_dry_run(data)
        force = _is_force(data)

        tag = Tag.query.filter_by(user_id=target_user.id, name=tag_norm).first()
        if tag is None:
            return jsonify({'error': 'Tag nicht gefunden.'}), 404

        affected = _tag_object_counts(tag.id)
        has_objects = any(v > 0 for v in affected.values())

        tag_id, tag_name = tag.id, tag.name
        re_raw = data.get('reassign_to')
        reassign_to = None
        reassigned = None

        if re_raw is not None:
            # --- reassign-then-delete path ---
            re_norm, err = _require_tag_name(re_raw)
            if err:
                return err
            if re_norm == tag_norm:
                return jsonify({'error': 'reassign_to == tag.'}), 400
            reassign_to = Tag.query.filter_by(user_id=target_user.id, name=re_norm).first()
            if reassign_to is None:
                return jsonify({'error': 'Ziel-Tag (reassign_to) nicht gefunden.'}), 404
            # Zyklus-Guard (wie Merge): reassign_to darf nicht echter Nachfahre
            # von tag sein — sonst baut das Kinder-Reparenten eine Schleife.
            if reassign_to.id in Tag.subtree_ids(tag_id, target_user.id) and reassign_to.id != tag_id:
                return jsonify({
                    'error': 'reassign_to liegt im Teilbaum von tag — würde einen '
                             'Zyklus erzeugen; erst via set_tag_parent entwirren.'
                }), 400
            re_id, re_name = reassign_to.id, reassign_to.name
            reassigned = _reassign_tag_refs(tag_id, re_id)
            new_parent = re_id
        else:
            # --- no reassign_to ---
            if has_objects and not force:
                # GUARD-RAIL: löst Objekte NICHT ohne explizites force.
                if dry_run:
                    return jsonify({
                        'dry_run': True, 'applied': False,
                        'tag': {'id': tag_id, 'name': tag_name},
                        'reassign_to': None, 'reassigned': None,
                        'affected': affected, 'children_reparented': [],
                        'requires_force': True, 'tag_deleted': False,
                    }), 200
                return jsonify({
                    'error': (f'Tag hat {sum(affected.values())} angehängte Objekte und '
                              'kein reassign_to — reassign_to setzen oder force=true.'),
                    'affected': affected, 'requires_force': True, 'tag_deleted': False,
                }), 409
            # keine Objekte ODER force=true → von allen Objekten lösen.
            for _key, junction, _obj_col in _TAG_JUNCTIONS:
                db.session.execute(junction.delete().where(junction.c.tag_id == tag_id))
            new_parent = None

        # Kinder reparenten (→ reassign_to bzw. NULL) — vorher Namen/IDs greifen.
        child_rows = db.session.execute(
            select(Tag.id, Tag.name).where(Tag.parent_id == tag_id)
        ).all()
        children_reparented = [{'id': cid, 'name': cname} for cid, cname in child_rows]
        db.session.execute(
            Tag.__table__.update()
            .where(Tag.parent_id == tag_id)
            .values(parent_id=new_parent)
        )

        db.session.delete(tag)

        if dry_run:
            db.session.rollback()
        else:
            db.session.commit()

        return jsonify({
            'dry_run': dry_run, 'applied': not dry_run,
            'tag': {'id': tag_id, 'name': tag_name},
            'reassign_to': ({'id': re_id, 'name': re_name} if reassign_to is not None else None),
            'reassigned': reassigned,
            'affected': affected,
            'children_reparented': children_reparented,
            'requires_force': False,
            'tag_deleted': not dry_run,
        }), 200

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
    app.extensions['csrf'].exempt(api_merge_tags)
    app.extensions['csrf'].exempt(api_delete_tag_token)
