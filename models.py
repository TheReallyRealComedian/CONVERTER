import json
import re
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy import event
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    conversions = db.relationship('Conversion', backref='user', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


conversion_tags = db.Table(
    'conversion_tags',
    db.Column('conversion_id', db.Integer, db.ForeignKey('conversion.id', ondelete='CASCADE'),
              primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id', ondelete='CASCADE'),
              primary_key=True),
)


class Conversion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    conversion_type = db.Column(db.String(30), nullable=False, index=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    source_filename = db.Column(db.String(255))
    source_mimetype = db.Column(db.String(100))
    source_size_bytes = db.Column(db.Integer)
    metadata_json = db.Column(db.Text)
    # Dead column after R2-A — the CSV-to-junction migration drains it and
    # the frontend writes nothing here anymore. Kept around because SQLite
    # DROP COLUMN is a table rebuild; a future cleanup sprint can remove it.
    tags = db.Column(db.String(500), default='')
    is_favorite = db.Column(db.Boolean, default=False)
    # R2-B: furthest-read scroll position as a percent (0–100), nullable.
    # Drives the list-view card progress bar and resume-on-open. Percent-based
    # so it survives content-length changes; added via inline ALTER TABLE.
    last_read_percent = db.Column(db.Float, nullable=True)
    # R2-C: lifecycle triage location — 'inbox' / 'later' / 'archive'.
    # Orthogonal to last_read_percent ("read" stays the progress, not a 4th
    # status). Indexed because the list-view filters on it. Added via inline
    # ALTER TABLE; the column-add backfills ai_newsletter→inbox, rest→archive.
    lifecycle_status = db.Column(db.String(20), default='inbox', index=True)
    # R2-D: manual reading-list priority. NULL = not on the list; a Float so a
    # future drag-reorder can slot an item *between* two neighbours without
    # renumbering the whole list. Orthogonal to lifecycle_status (location) and
    # last_read_percent (progress). No backfill — everyone starts off-list.
    queue_position = db.Column(db.Float, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    highlights = db.relationship('Highlight', backref='conversion',
                                 cascade='all, delete-orphan', lazy='dynamic')
    tag_refs = db.relationship('Tag', secondary=conversion_tags, lazy='joined',
                               backref=db.backref('conversions', lazy='dynamic'))

    def to_dict(self):
        return {
            'id': self.id,
            'conversion_type': self.conversion_type,
            'title': self.title,
            'content': self.content,
            'source_filename': self.source_filename,
            'source_mimetype': self.source_mimetype,
            'source_size_bytes': self.source_size_bytes,
            'metadata': json.loads(self.metadata_json) if self.metadata_json else {},
            'tags': self.tags,
            'tag_refs': [t.to_dict() for t in self.tag_refs],
            'is_favorite': self.is_favorite,
            'last_read_percent': self.last_read_percent,
            'lifecycle_status': self.lifecycle_status,
            'queue_position': self.queue_position,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


highlight_tags = db.Table(
    'highlight_tags',
    db.Column('highlight_id', db.Integer, db.ForeignKey('highlight.id', ondelete='CASCADE'),
              primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id', ondelete='CASCADE'),
              primary_key=True),
)


class Highlight(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversion_id = db.Column(db.Integer, db.ForeignKey('conversion.id'), nullable=False, index=True)
    exact = db.Column(db.Text, nullable=False)
    prefix = db.Column(db.Text, default='')
    suffix = db.Column(db.Text, default='')
    note = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    tags = db.relationship('Tag', secondary=highlight_tags, lazy='joined',
                           backref=db.backref('highlights', lazy='dynamic'))

    def to_dict(self):
        return {
            'id': self.id,
            'conversion_id': self.conversion_id,
            'exact': self.exact,
            'prefix': self.prefix,
            'suffix': self.suffix,
            'note': self.note,
            'tags': [t.to_dict() for t in self.tags],
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    name = db.Column(db.String(80), nullable=False)
    # LERN-GROUP Achse A: hierarchische Tags. NULL = Wurzel. Die Hierarchie
    # ordnet das geteilte Tag-Vokabular zu einem Wald — keine Karte/Highlight/
    # Conversion muss neu getaggt werden. Self-FK; ON DELETE ist ohne PRAGMA
    # foreign_keys inert (Memory reference_sqlite_no_fk_pragma_orm_delete), das
    # Reparenten der Kinder beim Tag-Delete erledigt die Delete-View explizit.
    parent_id = db.Column(db.Integer, db.ForeignKey('tag.id'), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    children = db.relationship(
        'Tag', backref=db.backref('parent', remote_side=[id]),
        lazy='select',
    )

    __table_args__ = (
        db.UniqueConstraint('user_id', 'name', name='uq_tag_user_name'),
    )

    MAX_NAME_LEN = 80

    @classmethod
    def normalize_name(cls, name):
        """Canonical tag-name normalisation — the single place every path
        (UI attach, ingest topics, scripts/cleanup_tags.py) goes through.

        R2-A: lowercase + trim. R2-E adds stripping of Markdown artefacts
        that LLM-generated newsletter topics drag in (``** [anthropic``):
        ``*`` and `` ` `` anywhere, ``[``/``]`` only at the edges, and
        runs of whitespace collapse to one space. Returns '' when nothing
        survives (caller decides skip/400)."""
        cleaned = name.replace('*', '').replace('`', '')
        cleaned = cleaned.strip().strip('[]')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip().lower()

    @classmethod
    def get_or_create(cls, user_id, name):
        # Single source of truth for "find or insert a tag, normalised via
        # normalize_name". Call sites: highlight-tag POST, conversion-tag
        # POST, ingest topics, and the CSV-to-junction migration helper.
        # Returns None on non-str/blank-after-normalisation/oversize so
        # callers can map to a 400 or skip without re-validating.
        if not isinstance(name, str):
            return None
        normalized = cls.normalize_name(name)
        if not normalized or len(normalized) > cls.MAX_NAME_LEN:
            return None
        tag = cls.query.filter_by(user_id=user_id, name=normalized).first()
        if tag is None:
            tag = cls(user_id=user_id, name=normalized)
            db.session.add(tag)
            db.session.flush()
        return tag

    @classmethod
    def subtree_ids(cls, root_id, user_id):
        """The set ``{root_id} ∪ all descendants`` within the user's tag forest
        (LERN-GROUP Achse A). Loads the user's (id, parent_id) pairs once and
        BFS-walks down — the tag count per user is small, so load-all + BFS is
        simpler and cheaper than a recursive CTE. Returns ``{root_id}`` even if
        the root has no children; an unknown/foreign root yields ``{root_id}``
        too (callers validate ownership before calling)."""
        rows = (db.session.query(cls.id, cls.parent_id)
                .filter(cls.user_id == user_id).all())
        children_map = {}
        for tid, pid in rows:
            children_map.setdefault(pid, []).append(tid)
        result = set()
        queue = [root_id]
        while queue:
            node = queue.pop()
            if node in result:
                continue
            result.add(node)
            queue.extend(children_map.get(node, []))
        return result

    def to_dict(self, highlight_count=None, conversion_count=None, card_count=None):
        out = {
            'id': self.id,
            'name': self.name,
            'parent_id': self.parent_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
        if highlight_count is not None:
            out['highlight_count'] = highlight_count
        if conversion_count is not None:
            out['conversion_count'] = conversion_count
        if card_count is not None:
            out['card_count'] = card_count
        return out


# --- R4-LEARN: the spaced-repetition / recall layer over the Highlights ---

card_tags = db.Table(
    'card_tags',
    db.Column('card_id', db.Integer, db.ForeignKey('card.id', ondelete='CASCADE'),
              primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id', ondelete='CASCADE'),
              primary_key=True),
)

# LERN-GROUP Achse B: kuratierte, flache Sammlungen (karten-only v1). M2M; eine
# Karte in beliebig vielen Sammlungen. ondelete='CASCADE' ist ohne PRAGMA
# foreign_keys inert (dokumentarisch) — die echte Lösch-Mechanik läuft ORM-seitig
# über die Card.collections-Relationship (Memory reference_sqlite_no_fk_pragma_orm_delete).
card_collections = db.Table(
    'card_collections',
    db.Column('card_id', db.Integer, db.ForeignKey('card.id', ondelete='CASCADE'),
              primary_key=True),
    db.Column('collection_id', db.Integer,
              db.ForeignKey('collection.id', ondelete='CASCADE'), primary_key=True),
)


class Card(db.Model):
    """A spaced-repetition card sitting over a Highlight (R4-LEARN).

    **Self-contained**: ``front``/``back``/``cloze_text``/``prompt`` live on the
    card; review never reads the Highlight live. ``highlight_id`` is best-effort
    provenance only — the ``before_delete`` event below nulls it when the
    Highlight goes away (the card + its Review survive). Cards are authored by
    the external agent via the token-auth write API (Phase 2), never generated
    inside CONVERTER.
    """
    id = db.Column(db.Integer, primary_key=True)
    # Owner scope. A card is a top-level entity that outlives its highlight, so
    # it carries its own user_id (like Conversion/Tag) rather than scoping
    # through the nullable highlight — the owner-scoped reads must still work
    # after the provenance link is broken.
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    # Provenance FK. Bare column, no Highlight<->Card relationship: the declared
    # ON DELETE SET NULL is INERT (no PRAGMA foreign_keys=ON) — the before_delete
    # event is the real mechanic. Declared anyway to mirror the junction tables.
    highlight_id = db.Column(db.Integer, db.ForeignKey('highlight.id', ondelete='SET NULL'),
                             nullable=True, index=True)
    # Authoring-time snapshot of the source, so a later edit/delete of the
    # Highlight or Conversion never changes what the card shows.
    source_snapshot = db.Column(db.Text, nullable=True)
    source_doc_title = db.Column(db.Text, nullable=True)
    type = db.Column(db.String(20), nullable=False)  # 'atomic' | 'generative'
    front = db.Column(db.Text, nullable=True)
    back = db.Column(db.Text, nullable=True)
    cloze_text = db.Column(db.Text, nullable=True)
    prompt = db.Column(db.Text, nullable=True)
    note = db.Column(db.Text, nullable=True)
    state = db.Column(db.String(20), default='ok', nullable=False)  # 'ok' | 'wackelt'
    created_by = db.Column(db.String(40), default='agent', nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    tags = db.relationship('Tag', secondary=card_tags, lazy='joined',
                           backref=db.backref('cards', lazy='dynamic'))
    # LERN-GROUP Achse B: kuratierte Sammlungen. Owning side hier → deleting a
    # card sweeps its card_collections rows via the ORM (no FK pragma).
    collections = db.relationship('Collection', secondary=card_collections,
                                  backref=db.backref('cards', lazy='dynamic'))
    # 1:1 FSRS state. ORM cascade (DB-level cascade is inert without the pragma);
    # POST /api/cards creates the row in the "new" state.
    review = db.relationship('Review', uselist=False, backref='card',
                             cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'highlight_id': self.highlight_id,
            'source_snapshot': self.source_snapshot,
            'source_doc_title': self.source_doc_title,
            'type': self.type,
            'front': self.front,
            'back': self.back,
            'cloze_text': self.cloze_text,
            'prompt': self.prompt,
            'note': self.note,
            'state': self.state,
            'created_by': self.created_by,
            'tags': [{'id': t.id, 'name': t.name} for t in self.tags],
            'review': self.review.to_dict() if self.review else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class Review(db.Model):
    """1:1 FSRS scheduling state for a Card (R4-LEARN).

    Own table (clean for a future review-history layer, not merged onto the
    card). Created alongside the card in the FSRS-"new" state (``due`` = now,
    ``reps`` = 0, ``lapses`` = 0, rest NULL); the rate endpoint (Phase 3)
    advances it through the swappable Scheduler.
    """
    id = db.Column(db.Integer, primary_key=True)
    card_id = db.Column(db.Integer, db.ForeignKey('card.id'), nullable=False,
                        unique=True, index=True)
    due = db.Column(db.DateTime, index=True)
    stability = db.Column(db.Float, nullable=True)
    difficulty = db.Column(db.Float, nullable=True)
    last_reviewed = db.Column(db.DateTime, nullable=True)
    reps = db.Column(db.Integer, default=0, nullable=False)
    lapses = db.Column(db.Integer, default=0, nullable=False)
    rating_history = db.Column(db.Text, nullable=True)  # JSON list, appended on rate

    def to_dict(self):
        return {
            'id': self.id,
            'card_id': self.card_id,
            'due': self.due.isoformat() if self.due else None,
            'stability': self.stability,
            'difficulty': self.difficulty,
            'last_reviewed': self.last_reviewed.isoformat() if self.last_reviewed else None,
            'reps': self.reps,
            'lapses': self.lapses,
            'rating_history': json.loads(self.rating_history) if self.rating_history else [],
        }


@event.listens_for(Highlight, 'before_delete')
def _null_card_provenance_on_highlight_delete(mapper, connection, target):
    """Break the card→highlight provenance link ORM-side — the real delete
    mechanic (R4-LEARN must-fix).

    CONVERTER runs SQLite WITHOUT ``PRAGMA foreign_keys=ON``, so the
    ``ON DELETE SET NULL`` declared on ``card.highlight_id`` never fires at the
    DB level. We do it here instead: whenever a Highlight is deleted — directly
    via the DELETE endpoint OR swept by the Conversion ``delete-orphan`` cascade
    (this event fires once per deleted Highlight either way) — null
    ``highlight_id`` on every Card that points at it. The Card and its Review
    survive; only the provenance link is lost.

    Runs on the flush ``connection`` with a Core UPDATE (never the Session — it
    is mid-flush). Cards already in the identity map pick up the change after
    the post-commit expire.
    """
    tbl = Card.__table__
    connection.execute(
        tbl.update().where(tbl.c.highlight_id == target.id).values(highlight_id=None)
    )


class Collection(db.Model):
    """A curated, flat bundle of cards (LERN-GROUP Achse B).

    One named card-set for a purpose the user picks (a horizon, a course, a
    topic pack — one entity, no kind distinction in v1). Cross-cutting: a card
    sits in any number of collections. Owner-scoped (own user_id) and per-user
    unique by name, mirroring Tag. The M2M side lives on ``Card.collections``;
    ``Collection.cards`` is the backref.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'name', name='uq_collection_user_name'),
    )

    MAX_NAME_LEN = 120

    def to_dict(self, card_count=None):
        out = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
        if card_count is not None:
            out['card_count'] = card_count
        return out
