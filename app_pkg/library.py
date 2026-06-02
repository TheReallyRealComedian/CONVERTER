"""Library list/detail and the /api/conversions CRUD endpoints."""
import json
import re

from flask import jsonify, render_template, request
from flask_login import current_user, login_required

from models import Conversion, Tag, db

from .markdown_render import render_markdown_to_html


ALLOWED_CONVERSION_TYPES = {
    'document_to_markdown',
    'audio_transcription',
    'dialogue_formatting',
    'markdown_input',
    'ai_newsletter',
}

ALLOWED_PER_PAGE = (20, 50, 100)
DEFAULT_PER_PAGE = 20


def pagination_args(page, conversion_type, search, favorites, sort, per_page, tag=''):
    """Build the **kwargs for url_for('library', …) so favorites='', the
    default per_page and an empty tag drop out of the URL entirely
    (F-6 P9 + P12; R2-B adds the tag filter)."""
    args = {'page': page, 'type': conversion_type, 'search': search, 'sort': sort}
    if favorites:
        args['favorites'] = '1'
    if per_page != DEFAULT_PER_PAGE:
        args['per_page'] = per_page
    if tag:
        args['tag'] = tag
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
        if search:
            escaped_search = re.sub(r'([%_\\])', r'\\\1', search)
            # R2-A: tag-search now joins against the conversion_tags junction
            # (Conversion.tag_refs.any(Tag.name.ilike(...))). The legacy
            # Conversion.tags CSV branch was removed — after the migration the
            # column is dead and would never match. Filtered Views (R2-B) will
            # build on this junction path.
            query = query.filter(
                db.or_(
                    Conversion.title.ilike(f'%{escaped_search}%'),
                    Conversion.content.ilike(f'%{escaped_search}%'),
                    Conversion.tag_refs.any(Tag.name.ilike(f'%{escaped_search}%')),
                )
            )

        if sort == 'oldest':
            query = query.order_by(Conversion.created_at.asc())
        elif sort == 'title':
            query = query.order_by(Conversion.title.asc())
        else:
            query = query.order_by(Conversion.created_at.desc())

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        has_active_filter = bool(
            (conversion_type and conversion_type in ALLOWED_CONVERSION_TYPES)
            or search
            or favorites
            or tag
        )

        # Tags of this user that hang on at least one conversion — feeds the
        # filter-chip row. Tag.conversions is the dynamic backref from the
        # conversion_tags junction (models.py).
        available_tags = Tag.query.filter(
            Tag.conversions.any(Conversion.user_id == current_user.id)
        ).order_by(Tag.name).all()

        return render_template('library.html',
                               conversions=pagination.items,
                               pagination=pagination,
                               current_type=conversion_type,
                               current_search=search,
                               current_favorites=favorites,
                               current_sort=sort,
                               current_per_page=per_page,
                               current_tag=tag,
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
