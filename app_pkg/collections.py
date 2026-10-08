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
from datetime import datetime, timezone

from flask import jsonify, request
from flask_login import current_user, login_required
from sqlalchemy import func

from models import (Card, Collection, CollectionDocument, Conversion, Review,
                    card_collections, db)


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
        # LEARN-UP badges: due count per collection in ONE group-by (not N
        # queries). Same rawness as the review queue — Review.due <= now,
        # before any daily limits (those cap the session, not the badge).
        now = datetime.now(timezone.utc)
        due_counts = (db.session.query(
            card_collections.c.collection_id,
            func.count(card_collections.c.card_id).label('due_cnt'),
        )
            .join(Review, Review.card_id == card_collections.c.card_id)
            .filter(Review.due <= now)
            .group_by(card_collections.c.collection_id)
            .subquery())
        rows = (db.session.query(Collection, card_counts.c.cnt, due_counts.c.due_cnt)
                .outerjoin(card_counts, Collection.id == card_counts.c.collection_id)
                .outerjoin(due_counts, Collection.id == due_counts.c.collection_id)
                .filter(Collection.user_id == current_user.id)
                .order_by(Collection.name.asc())
                .all())
        # LERN-TEXT: the Lerntexte per collection in ONE query (not one per
        # row); additive per-entry field — the response stays a bare array.
        documents = Collection.documents_by_id([col.id for col, _cnt, _due in rows])
        return jsonify([
            col.to_dict(card_count=int(cnt or 0), due_count=int(due or 0),
                        documents=documents[col.id])
            for col, cnt, due in rows
        ])

    @app.route('/api/collections', methods=['POST'])
    @login_required
    def api_create_collection():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        # ARCH-LIBRARY-KLEIN: the ONE normalisation (trim + collapse inner
        # whitespace, case kept) — the same Collection.normalize_name the
        # agent path runs in get_or_create. A bare strip() here made "Chemie
        # Basics" with two spaces one collection through the UI and another
        # through the agent. '' for a non-string or a blank name.
        name = Collection.normalize_name(data.get('name'))
        if not name:
            return jsonify({'error': 'Name fehlt.'}), 400
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
        return jsonify(collection.to_dict(card_count=0, documents=[])), 201

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
            # Same normalisation as POST and the agent path (see above).
            name = Collection.normalize_name(data.get('name'))
            if not name:
                return jsonify({'error': 'Name fehlt.'}), 400
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
        # relationship (verified: the backref side drains too) and the
        # collection_documents rows via Collection.document_links (LERN-TEXT).
        # The cards and the documents survive.
        db.session.delete(collection)
        db.session.commit()
        return jsonify({'success': True})

    @app.route('/api/collections/<int:collection_id>/documents', methods=['PUT'])
    @login_required
    def api_set_collection_documents(collection_id):
        # LERN-TEXT: REPLACE the collection's Lerntext list. ``documents`` is
        # the ordered list of document ids (position = index; repeats keep
        # their first position); an empty list clears. Every id must be the
        # current user's own document, else 404 and NOTHING is written — the
        # list before the call stays as it was (checked before the first
        # write, one commit). Same session/owner posture as the siblings.
        collection = _get_owned_collection(collection_id)
        if collection is None:
            return jsonify({'error': 'Nicht gefunden.'}), 404
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        documents = data.get('documents')
        if (not isinstance(documents, list)
                or any(not isinstance(d, int) or isinstance(d, bool) for d in documents)):
            return jsonify({'error': "Feld 'documents' muss eine Liste von Zahlen sein."}), 400
        ordered = list(dict.fromkeys(documents))

        if ordered:
            owned = {row[0] for row in (db.session.query(Conversion.id)
                                        .filter(Conversion.user_id == current_user.id,
                                                Conversion.id.in_(ordered))
                                        .all())}
            if any(doc_id not in owned for doc_id in ordered):
                return jsonify({'error': 'Dokument nicht gefunden.'}), 404

        # Clear + flush BEFORE inserting: a kept id would otherwise collide on
        # the composite primary key (the unit of work inserts before it
        # deletes within one flush). Still one transaction — a failure
        # between the two leaves the old list in place.
        collection.document_links.clear()
        db.session.flush()
        collection.document_links.extend(
            CollectionDocument(conversion_id=doc_id, position=position)
            for position, doc_id in enumerate(ordered)
        )
        db.session.commit()
        return jsonify(collection.to_dict())

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
