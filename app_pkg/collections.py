"""Collection API — curated, flat card bundles (LERN-GROUP Achse B).

A Collection is a named set of cards the user assembles for a purpose (a
horizon, a course, a topic pack — one entity, no kind distinction in v1). It is
cross-cutting: a card can sit in any number of collections. All endpoints are
``@login_required`` and owner-scoped (foreign id → 404, never leak existence),
mirroring the card read/delete posture in ``app_pkg/cards.py`` — collections are
a pure user-side surface, no agent token.

The M2M lives on ``Card.collections`` (owning side); deleting either a Card or a
Collection sweeps the ``card_collections`` rows through the ORM (SQLite runs
without ``PRAGMA foreign_keys=ON`` so the declared ``ON DELETE CASCADE`` is
inert — verified empirically that the backref side drains too).
"""
from flask import jsonify, request
from flask_login import current_user, login_required
from sqlalchemy import func

from models import Card, Collection, card_collections, db


def _get_owned_collection(collection_id):
    """The current user's Collection or None (caller maps None → 404)."""
    collection = Collection.query.get(collection_id)
    if collection is None or collection.user_id != current_user.id:
        return None
    return collection


def register(app):
    @app.route('/api/collections', methods=['GET'])
    @login_required
    def api_list_collections():
        card_counts = (db.session.query(
            card_collections.c.collection_id,
            func.count(card_collections.c.card_id).label('cnt'),
        )
            .group_by(card_collections.c.collection_id)
            .subquery())
        rows = (db.session.query(Collection, card_counts.c.cnt)
                .outerjoin(card_counts, Collection.id == card_counts.c.collection_id)
                .filter(Collection.user_id == current_user.id)
                .order_by(Collection.name.asc())
                .all())
        return jsonify([
            col.to_dict(card_count=int(cnt or 0)) for col, cnt in rows
        ])

    @app.route('/api/collections', methods=['POST'])
    @login_required
    def api_create_collection():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        name = data.get('name')
        if not isinstance(name, str) or not name.strip():
            return jsonify({'error': 'Name fehlt.'}), 400
        name = name.strip()
        if len(name) > Collection.MAX_NAME_LEN:
            return jsonify({
                'error': f'Name zu lang (max {Collection.MAX_NAME_LEN} Zeichen).'
            }), 400
        existing = Collection.query.filter_by(
            user_id=current_user.id, name=name).first()
        if existing is not None:
            return jsonify({'error': 'Sammlung existiert bereits.'}), 409

        description = data.get('description')
        if description is not None and not isinstance(description, str):
            return jsonify({'error': "Feld 'description' muss Text sein."}), 400
        collection = Collection(user_id=current_user.id, name=name,
                                description=(description or None))
        db.session.add(collection)
        db.session.commit()
        return jsonify(collection.to_dict(card_count=0)), 201

    @app.route('/api/collections/<int:collection_id>', methods=['PATCH'])
    @login_required
    def api_update_collection(collection_id):
        collection = _get_owned_collection(collection_id)
        if collection is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        if 'name' in data:
            name = data.get('name')
            if not isinstance(name, str) or not name.strip():
                return jsonify({'error': 'Name fehlt.'}), 400
            name = name.strip()
            if len(name) > Collection.MAX_NAME_LEN:
                return jsonify({
                    'error': f'Name zu lang (max {Collection.MAX_NAME_LEN} Zeichen).'
                }), 400
            clash = Collection.query.filter(
                Collection.user_id == current_user.id,
                Collection.name == name,
                Collection.id != collection.id,
            ).first()
            if clash is not None:
                return jsonify({'error': 'Sammlung existiert bereits.'}), 409
            collection.name = name

        if 'description' in data:
            description = data.get('description')
            if description is not None and not isinstance(description, str):
                return jsonify({'error': "Feld 'description' muss Text sein."}), 400
            collection.description = description or None

        db.session.commit()
        return jsonify(collection.to_dict())

    @app.route('/api/collections/<int:collection_id>', methods=['DELETE'])
    @login_required
    def api_delete_collection(collection_id):
        collection = _get_owned_collection(collection_id)
        if collection is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        # ORM delete sweeps the card_collections rows via the Card.collections
        # relationship (verified: the backref side drains too). The cards survive.
        db.session.delete(collection)
        db.session.commit()
        return jsonify({'success': True})

    @app.route('/api/collections/<int:collection_id>/cards', methods=['POST'])
    @login_required
    def api_add_card_to_collection(collection_id):
        collection = _get_owned_collection(collection_id)
        if collection is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        card_id = data.get('card_id')
        if not isinstance(card_id, int) or isinstance(card_id, bool):
            return jsonify({'error': "Feld 'card_id' muss int sein."}), 400
        card = Card.query.filter_by(id=card_id, user_id=current_user.id).first()
        if card is None:
            return jsonify({'error': 'Karte nicht gefunden.'}), 404
        if card not in collection.cards:
            collection.cards.append(card)
            db.session.commit()
        return jsonify({'success': True})

    @app.route('/api/collections/<int:collection_id>/cards/<int:card_id>',
               methods=['DELETE'])
    @login_required
    def api_remove_card_from_collection(collection_id, card_id):
        collection = _get_owned_collection(collection_id)
        if collection is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        card = Card.query.filter_by(id=card_id, user_id=current_user.id).first()
        if card is None:
            return jsonify({'error': 'Karte nicht gefunden.'}), 404
        if card in collection.cards:
            collection.cards.remove(card)
            db.session.commit()
        return jsonify({'success': True})
