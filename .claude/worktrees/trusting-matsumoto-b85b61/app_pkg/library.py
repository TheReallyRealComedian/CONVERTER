"""Library list/detail and the /api/conversions CRUD endpoints."""
import json
import re

from flask import jsonify, render_template, request
from flask_login import current_user, login_required

from models import Conversion, db


ALLOWED_CONVERSION_TYPES = {
    'document_to_markdown',
    'audio_transcription',
    'dialogue_formatting',
    'markdown_input',
}


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
        sort = request.args.get('sort', 'newest')
        page = request.args.get('page', 1, type=int)
        per_page = 20

        query = Conversion.query.filter_by(user_id=current_user.id)

        if conversion_type:
            query = query.filter_by(conversion_type=conversion_type)
        if favorites:
            query = query.filter_by(is_favorite=True)
        if search:
            escaped_search = re.sub(r'([%_\\])', r'\\\1', search)
            query = query.filter(
                db.or_(
                    Conversion.title.ilike(f'%{escaped_search}%'),
                    Conversion.content.ilike(f'%{escaped_search}%'),
                    Conversion.tags.ilike(f'%{escaped_search}%')
                )
            )

        if sort == 'oldest':
            query = query.order_by(Conversion.created_at.asc())
        elif sort == 'title':
            query = query.order_by(Conversion.title.asc())
        else:
            query = query.order_by(Conversion.created_at.desc())

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        return render_template('library.html',
                               conversions=pagination.items,
                               pagination=pagination,
                               current_type=conversion_type,
                               current_search=search,
                               current_favorites=favorites,
                               current_sort=sort)

    @app.route('/library/<int:conversion_id>')
    @login_required
    def library_detail(conversion_id):
        conversion = get_owned_conversion(conversion_id)
        metadata = json.loads(conversion.metadata_json) if conversion.metadata_json else {}
        return render_template('library_detail.html', conversion=conversion, metadata=metadata)

    @app.route('/api/conversions', methods=['POST'])
    @login_required
    def api_create_conversion():
        data = request.get_json()
        if not data or not data.get('content'):
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
        data = request.get_json()

        if 'title' in data:
            conversion.title = str(data['title'])[:255]
        if 'tags' in data:
            conversion.tags = str(data['tags'])[:500]
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
