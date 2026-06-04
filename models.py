import json
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
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
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'name', name='uq_tag_user_name'),
    )

    MAX_NAME_LEN = 80

    @classmethod
    def get_or_create(cls, user_id, name):
        # Single source of truth for "find or insert a tag, normalised to
        # lowercase+trim". Three call sites: highlight-tag POST, conversion-
        # tag POST, and the CSV-to-junction migration helper. Returns None on
        # blank/oversize so callers can map to a 400 without re-validating.
        if not isinstance(name, str):
            return None
        normalized = name.strip().lower()
        if not normalized or len(normalized) > cls.MAX_NAME_LEN:
            return None
        tag = cls.query.filter_by(user_id=user_id, name=normalized).first()
        if tag is None:
            tag = cls(user_id=user_id, name=normalized)
            db.session.add(tag)
            db.session.flush()
        return tag

    def to_dict(self, highlight_count=None, conversion_count=None):
        out = {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
        if highlight_count is not None:
            out['highlight_count'] = highlight_count
        if conversion_count is not None:
            out['conversion_count'] = conversion_count
        return out
