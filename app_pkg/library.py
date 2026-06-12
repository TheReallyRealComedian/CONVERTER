"""Library list/detail and the /api/conversions CRUD endpoints."""
import json

from flask import jsonify, render_template, request
from flask_login import current_user, login_required

from models import Conversion, Tag, conversion_tags, db

from .markdown_render import render_markdown_to_html


ALLOWED_CONVERSION_TYPES = {
    'document_to_markdown',
    'audio_transcription',
    'dialogue_formatting',
    'markdown_input',
    'ai_newsletter',
}

# R2-C: lifecycle triage locations. Internal keys; DE labels (Inbox/Später/
# Archiv) live in the templates. Used by the ?status filter + the PUT handler.
LIFECYCLE_STATUSES = {'inbox', 'later', 'archive'}

ALLOWED_PER_PAGE = (20, 50, 100)
DEFAULT_PER_PAGE = 20


def pagination_args(page, conversion_type, search, favorites, sort, per_page, tag='', status='', view=''):
    """Build the **kwargs for url_for('library', …) so favorites='', the
    default per_page, an empty tag, an empty status and an empty view drop out
    of the URL entirely (F-6 P9 + P12; R2-B adds the tag filter, R2-C the status
    filter, R2-D the view mode)."""
    args = {'page': page, 'type': conversion_type, 'search': search, 'sort': sort}
    if favorites:
        args['favorites'] = '1'
    if per_page != DEFAULT_PER_PAGE:
        args['per_page'] = per_page
    if tag:
        args['tag'] = tag
    if status:
        args['status'] = status
    if view:
        args['view'] = view
    return args


def get_owned_conversion(conversion_id):
    """Look up a Conversion that belongs to the current user, or 404.

    Centralises the owner-scoped lookup that was open-coded in four
    separate routes (F-010).
    """
    return Conversion.query.filter_by(
        id=conversion_id, user_id=current_user.id,
    ).first_or_404()


def register(app):
    @app.route('/library')
    @login_required
    def library():
        conversion_type = request.args.get('type', '')
        search = request.args.get('search', '').strip()
        favorites = request.args.get('favorites', '') == '1'
        # Tags are stored lowercase+trimmed (Tag.get_or_create) — normalise the
        # incoming value so ?tag=KI matches the stored "ki".
        tag = request.args.get('tag', '').strip().lower()
        # R2-C: lifecycle triage filter. Only a known status filters; anything
        # else is treated as "no status filter" (current_status stays '').
        status = request.args.get('status', '').strip()
        current_status = status if status in LIFECYCLE_STATUSES else ''
        # R2-E "Readwise-3er" IA: 'inbox' = the untriaged pile (status inbox
        # AND not queued — queueing an item triages it away without touching
        # its status; de-queueing an item that is still 'inbox' drops it back
        # into triage); 'queue' = the manual reading-list (queued + not
        # archived, ordered by position, R2-D). Anything else — including the
        # retired R2-D 'reading' — falls through to the default library list.
        # A mode, not a filter — kept out of has_active_filter so the "Filter
        # zurücksetzen" empty-state stays scoped to type/search/tag/status.
        view = request.args.get('view', '')
        current_view = view if view in ('inbox', 'queue') else ''
        sort = request.args.get('sort', 'newest')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', DEFAULT_PER_PAGE, type=int)
        if per_page not in ALLOWED_PER_PAGE:
            per_page = DEFAULT_PER_PAGE

        query = Conversion.query.filter_by(user_id=current_user.id)

        if conversion_type and conversion_type in ALLOWED_CONVERSION_TYPES:
            query = query.filter_by(conversion_type=conversion_type)
        if favorites:
            query = query.filter_by(is_favorite=True)
        if tag:
            # R2-B: exact-match tag filter over the conversion_tags junction —
            # the same Conversion.tag_refs.any(...) path the R2-A search branch
            # uses, with == instead of ilike (the name is already normalised).
            query = query.filter(Conversion.tag_refs.any(Tag.name == tag))
        if current_status:
            query = query.filter(Conversion.lifecycle_status == current_status)
        if search:
            # R2-A: tag-search joins against the conversion_tags junction. The
            # legacy Conversion.tags CSV branch was removed — after the
            # migration the column is dead and would never match.
            # P3-fix: .contains(autoescape=True) escapes %/_ and emits the
            # ESCAPE clause itself. The previous re.sub + ilike was a silent
            # no-op — ilike has no default ESCAPE, so the inserted backslashes
            # matched literally and broke substring search (Memory
            # reference_sqlalchemy_like_escape.md). LIKE is case-insensitive on
            # SQLite, so dropping ilike keeps the case behaviour.
            query = query.filter(
                db.or_(
                    Conversion.title.contains(search, autoescape=True),
                    Conversion.content.contains(search, autoescape=True),
                    Conversion.tag_refs.any(Tag.name.contains(search, autoescape=True)),
                )
            )

        # View-mode filters AND on top of the type/tag/status/search chips.
        if current_view == 'inbox':
            query = query.filter(
                Conversion.lifecycle_status == 'inbox',
                Conversion.queue_position.is_(None),
            )
        elif current_view == 'queue':
            query = query.filter(
                Conversion.queue_position.isnot(None),
                Conversion.lifecycle_status != 'archive',
            )

        # The queue view overrides the sort param (manual position); the inbox
        # view keeps the regular sort handling (newest first by default).
        if current_view == 'queue':
            query = query.order_by(Conversion.queue_position.asc())
        elif sort == 'oldest':
            query = query.order_by(Conversion.created_at.asc())
        elif sort == 'title':
            query = query.order_by(Conversion.title.asc())
        else:
            query = query.order_by(Conversion.created_at.desc())

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        # R2-E: "Weiterlesen" section above the queue — in-progress reads
        # (0 < last_read_percent < 95) that are neither queued (their card
        # progress bar already shows it) nor archived. Sort carried over from
        # the retired view=reading: most-recently-touched first. Unpaginated
        # and unfiltered by the chips; in practice rarely more than a handful.
        reading_items = []
        if current_view == 'queue':
            reading_items = Conversion.query.filter(
                Conversion.user_id == current_user.id,
                Conversion.last_read_percent.isnot(None),
                Conversion.last_read_percent > 0,
                Conversion.last_read_percent < 95,
                Conversion.queue_position.is_(None),
                Conversion.lifecycle_status != 'archive',
            ).order_by(Conversion.updated_at.desc()).all()

        # R2-E: badge count for the Inbox tab — untriaged = status inbox AND
        # not queued. Global per user (independent of the active filters) and
        # computed for every view because the tab bar is always visible.
        inbox_count = Conversion.query.filter(
            Conversion.user_id == current_user.id,
            Conversion.lifecycle_status == 'inbox',
            Conversion.queue_position.is_(None),
        ).count()

        has_active_filter = bool(
            (conversion_type and conversion_type in ALLOWED_CONVERSION_TYPES)
            or search
            or favorites
            or tag
            or current_status
        )

        # Tags of this user that hang on at least one conversion — feeds the
        # tag chip row. R2-E: each row carries its usage count over the
        # conversion_tags junction, ordered count-desc with alphabetical
        # tie-break so the template can slice a top-N. Rows expose .name and
        # .count (Row objects, not Tag instances).
        tag_usage = db.func.count(conversion_tags.c.conversion_id)
        available_tags = (
            db.session.query(Tag.name, tag_usage.label('count'))
            .join(conversion_tags, conversion_tags.c.tag_id == Tag.id)
            .join(Conversion, Conversion.id == conversion_tags.c.conversion_id)
            .filter(Conversion.user_id == current_user.id)
            .group_by(Tag.id, Tag.name)
            .order_by(tag_usage.desc(), Tag.name.asc())
            .all()
        )

        return render_template('library.html',
                               conversions=pagination.items,
                               pagination=pagination,
                               reading_items=reading_items,
                               inbox_count=inbox_count,
                               current_type=conversion_type,
                               current_search=search,
                               current_favorites=favorites,
                               current_sort=sort,
                               current_per_page=per_page,
                               current_tag=tag,
                               current_status=current_status,
                               current_view=current_view,
                               available_tags=available_tags,
                               allowed_per_page=ALLOWED_PER_PAGE,
                               has_active_filter=has_active_filter,
                               pagination_args=pagination_args)

    @app.route('/library/<int:conversion_id>')
    @login_required
    def library_detail(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        metadata = json.loads(conversion.metadata_json) if conversion.metadata_json else {}
        content_html = render_markdown_to_html(conversion.content)
        return render_template(
            'library_detail.html',
            conversion=conversion,
            metadata=metadata,
            content_html=content_html,
            conversion_tag_refs=[t.to_dict() for t in conversion.tag_refs],
        )

    @app.route('/api/conversions', methods=['POST'])
    @login_required
    def api_create_conversion():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        if not data.get('content'):
            return jsonify({'error': 'Content is required'}), 400

        conversion_type = data.get('conversion_type', 'unknown')
        if conversion_type not in ALLOWED_CONVERSION_TYPES:
            return jsonify({'error': f'Invalid conversion type: {conversion_type}'}), 400

        title = data.get('title', 'Untitled')[:255]

        conversion = Conversion(
            user_id=current_user.id,
            conversion_type=conversion_type,
            title=title,
            content=data['content'],
            source_filename=data.get('source_filename'),
            source_mimetype=data.get('source_mimetype'),
            source_size_bytes=data.get('source_size_bytes'),
            metadata_json=json.dumps(data.get('metadata', {})),
            tags=data.get('tags', ''),
        )
        db.session.add(conversion)
        db.session.commit()
        return jsonify(conversion.to_dict()), 201

    @app.route('/api/conversions/<int:conversion_id>', methods=['PUT'])
    @login_required
    def api_update_conversion(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        # R2-A: the legacy CSV `tags` path on PUT is gone. The frontend now
        # uses /api/conversions/<id>/tags POST + DELETE for attach/detach
        # against the conversion_tags junction; the old CSV column is a
        # dead column that the migration helper has already drained.
        if 'title' in data:
            conversion.title = str(data['title'])[:255]
        if 'content' in data:
            conversion.content = data['content']
        if 'is_favorite' in data:
            conversion.is_favorite = bool(data['is_favorite'])
        if 'lifecycle_status' in data:
            # R2-C: the status toggle (Card + Detail) PUTs here, like is_favorite.
            if data['lifecycle_status'] not in LIFECYCLE_STATUSES:
                return jsonify({'error': 'Ungültiger Lifecycle-Status.'}), 400
            conversion.lifecycle_status = data['lifecycle_status']

        db.session.commit()
        return jsonify(conversion.to_dict())

    @app.route('/api/conversions/<int:conversion_id>', methods=['DELETE'])
    @login_required
    def api_delete_conversion(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        db.session.delete(conversion)
        db.session.commit()
        return jsonify({'success': True})

    @app.route('/api/conversions/<int:conversion_id>/progress', methods=['PATCH'])
    @login_required
    def api_update_conversion_progress(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        percent = data.get('percent')
        # Bool is an int subclass — reject it explicitly so True/False can't
        # pose as a 1/0 percent. Then require a real number: missing / None /
        # non-numeric is a 400 (atomic endpoint, no hidden no-op like R1-B-B).
        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            return jsonify({'error': 'Feld "percent" fehlt oder ist ungültig.'}), 400

        # Out-of-range is a rounding artefact of a fire-and-forget scroll
        # signal (e.g. 100.0001), not a client error — clamp instead of 400.
        percent = max(0.0, min(100.0, float(percent)))
        conversion.last_read_percent = percent
        db.session.commit()
        return jsonify({'success': True, 'last_read_percent': percent}), 200

    @app.route('/api/conversions/<int:conversion_id>/queue', methods=['POST'])
    @login_required
    def api_update_conversion_queue(conversion_id):
        # R2-D reading-list queue: add/remove toggle + up/down reorder. Float
        # positions so a future drag-reorder can slot between neighbours.
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        action = data.get('action')
        if action not in {'add', 'remove', 'up', 'down'}:
            return jsonify({'error': 'Ungültige Queue-Aktion.'}), 400

        if action == 'add':
            # Already on the list → no-op; otherwise append at the end. max()
            # ignores NULLs in SQL, so this is the largest position in use.
            if conversion.queue_position is None:
                max_pos = db.session.query(db.func.max(Conversion.queue_position)).filter(
                    Conversion.user_id == current_user.id,
                ).scalar()
                conversion.queue_position = (max_pos + 1.0) if max_pos is not None else 1.0
        elif action == 'remove':
            conversion.queue_position = None
        else:
            # up/down: swap queue_position with the direct neighbour among the
            # user's queued items (sorted asc). Two-row update, one commit
            # boundary. No-op at the top/bottom edge or when not queued at all.
            # The set MUST match the queue-view's visible set (archive excluded,
            # Decision #5 keeps archived rows queued) — else "up" swaps with an
            # invisible archived neighbour and appears to do nothing.
            if conversion.queue_position is not None:
                queued = Conversion.query.filter(
                    Conversion.user_id == current_user.id,
                    Conversion.queue_position.isnot(None),
                    Conversion.lifecycle_status != 'archive',
                ).order_by(Conversion.queue_position.asc()).all()
                idx = next((i for i, c in enumerate(queued) if c.id == conversion.id), None)
                if idx is not None:
                    neighbour_idx = (idx - 1) if action == 'up' else (idx + 1)
                    if 0 <= neighbour_idx < len(queued):
                        neighbour = queued[neighbour_idx]
                        conversion.queue_position, neighbour.queue_position = (
                            neighbour.queue_position, conversion.queue_position,
                        )

        db.session.commit()
        return jsonify(conversion.to_dict())

    @app.route('/api/conversions/<int:conversion_id>/tags', methods=['POST'])
    @login_required
    def api_attach_conversion_tag(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        tag = Tag.get_or_create(current_user.id, data.get('name'))
        if tag is None:
            return jsonify({
                'error': f'Tag-Name fehlt oder ist zu lang (max {Tag.MAX_NAME_LEN} Zeichen).'
            }), 400

        if tag in conversion.tag_refs:
            # Idempotent attach — no-op 200 statt 409, mirrors highlight POST.
            db.session.commit()
            return jsonify(tag.to_dict()), 200

        conversion.tag_refs.append(tag)
        db.session.commit()
        return jsonify(tag.to_dict()), 201

    @app.route('/api/conversions/<int:conversion_id>/tags/<int:tag_id>', methods=['DELETE'])
    @login_required
    def api_detach_conversion_tag(conversion_id, tag_id):
        conversion = get_owned_conversion(conversion_id)
        tag = Tag.query.get_or_404(tag_id)
        if tag.user_id != current_user.id:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        if tag in conversion.tag_refs:
            conversion.tag_refs.remove(tag)
            db.session.commit()
        return jsonify({'success': True})
