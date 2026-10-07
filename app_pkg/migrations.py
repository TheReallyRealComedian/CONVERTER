"""Inline schema migrations — no Alembic (the decision stands). ``db.create_all()``
creates missing TABLES at every start; the steps here patch what it never
touches on an existing table: new COLUMNS (one idempotent ALTER each, guarded
by a live-schema check), the INDEXES the model declares and an ALTERed file
lacks (generic reconcile, ARCH-FACTORY) and the one data migration (R2-A,
CSV tags → junction). They run once per process start, right after
``create_all``, under the startup lock (``app_pkg.db_runtime``).

Moved verbatim out of ``app_pkg/__init__.py`` in ARCH-FACTORY P1 (function
bodies byte-equal). The gate for the whole chain is tests/test_boot_schema.py:
a frozen legacy DDL → ``create_app()`` → the schema ``create_all`` declares.
"""
from sqlalchemy import inspect, text
from sqlalchemy.schema import CreateIndex

from models import db


def _run_pending_migrations(app):
    # No Alembic/Flask-Migrate in this project, and db.create_all() does not
    # patch columns onto pre-existing tables. Each entry is idempotent —
    # it inspects the live schema first and only ALTERs when needed, so
    # repeated container starts are safe.
    inspector = inspect(db.engine)
    if 'highlight' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('highlight')}
        if 'note' not in cols:
            db.session.execute(text('ALTER TABLE highlight ADD COLUMN note TEXT'))
            db.session.commit()
            app.logger.info("R1-B-B: highlight.note column added via ALTER TABLE")
    if 'conversion' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('conversion')}
        if 'last_read_percent' not in cols:
            db.session.execute(text('ALTER TABLE conversion ADD COLUMN last_read_percent FLOAT'))
            db.session.commit()
            app.logger.info("R2-B: conversion.last_read_percent column added via ALTER TABLE")
        if 'lifecycle_status' not in cols:
            db.session.execute(text("ALTER TABLE conversion ADD COLUMN lifecycle_status VARCHAR(20) DEFAULT 'inbox'"))
            # Einmaliger differenzierter Backfill (läuft nur beim Spalten-Add → idempotent):
            # Newsletter bleiben im Inbox-Triage, alte Tool-Outputs ins Archive.
            db.session.execute(text("UPDATE conversion SET lifecycle_status='archive' WHERE conversion_type != 'ai_newsletter'"))
            db.session.commit()
            app.logger.info("R2-C: conversion.lifecycle_status added + backfilled (ai_newsletter→inbox, rest→archive)")
        if 'queue_position' not in cols:
            db.session.execute(text('ALTER TABLE conversion ADD COLUMN queue_position FLOAT'))
            # No backfill — NULL means "not on the reading list", so everyone
            # starts with an empty list. Idempotent via the column guard above.
            db.session.commit()
            app.logger.info("R2-D: conversion.queue_position added via ALTER TABLE")
        if 'content_version' not in cols:
            # LOST-UPDATE: content-bound optimistic-locking counter
            # (models.Conversion.content_version). NOT NULL needs the DEFAULT
            # so SQLite backfills every legacy row with 1 — the INSERT value;
            # NULL would make the conditional section UPDATE miss forever.
            db.session.execute(text(
                'ALTER TABLE conversion ADD COLUMN content_version INTEGER NOT NULL DEFAULT 1'))
            db.session.commit()
            app.logger.info("LOST-UPDATE: conversion.content_version column added via ALTER TABLE")
    if 'user' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('user')}
        if 'settings_json' not in cols:
            db.session.execute(text('ALTER TABLE "user" ADD COLUMN settings_json TEXT'))
            # No backfill — NULL means "all defaults" (app_pkg/learn.py merges
            # stored values over the defaults). Idempotent via the column guard.
            db.session.commit()
            app.logger.info("LEARN-UP: user.settings_json column added via ALTER TABLE")
    if 'tag' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('tag')}
        if 'parent_id' not in cols:
            db.session.execute(text('ALTER TABLE tag ADD COLUMN parent_id INTEGER'))
            # No backfill — NULL means "root", so every existing tag starts at the
            # top of the forest. Idempotent via the column guard above.
            db.session.commit()
            app.logger.info("LERN-GROUP: tag.parent_id column added via ALTER TABLE")
    if 'card' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('card')}
        if 'front_svg' not in cols:
            db.session.execute(text('ALTER TABLE card ADD COLUMN front_svg TEXT'))
            db.session.commit()
            app.logger.info("CARD-SVG: card.front_svg column added via ALTER TABLE")
        if 'back_svg' not in cols:
            db.session.execute(text('ALTER TABLE card ADD COLUMN back_svg TEXT'))
            # No backfill — NULL means "no figure". Idempotent via the column
            # guards above.
            db.session.commit()
            app.logger.info("CARD-SVG: card.back_svg column added via ALTER TABLE")
        # LERN-TEXT: the card's Lerntext place (models.Card.context_*). No
        # backfill — NULL means "no place". The junction table
        # collection_documents needs no entry here: db.create_all() (which
        # runs right before this function at startup) creates missing
        # TABLES, it only never patches COLUMNS onto existing ones.
        if 'context_conversion_id' not in cols:
            db.session.execute(text('ALTER TABLE card ADD COLUMN context_conversion_id INTEGER'))
            db.session.commit()
            app.logger.info("LERN-TEXT: card.context_conversion_id column added via ALTER TABLE")
        if 'context_heading' not in cols:
            db.session.execute(text('ALTER TABLE card ADD COLUMN context_heading TEXT'))
            db.session.commit()
            app.logger.info("LERN-TEXT: card.context_heading column added via ALTER TABLE")
        # ix_card_context_conversion_id (index=True on the ALTERed column) is
        # made by _reconcile_model_indexes below, like every other model index
        # an ALTERed file lacks — ARCH-FACTORY replaced the one-off step here.
    if 'review' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('review')}
        if 'version' not in cols:
            # LOST-UPDATE: optimistic-locking counter (models.Review.version,
            # the mapper's version_id_col). NOT NULL needs the DEFAULT so
            # SQLite backfills every legacy row with 1 — the value SQLAlchemy
            # assigns on INSERT; a NULL here would make the conditional UPDATE
            # miss forever. Idempotent via the column guard above.
            db.session.execute(text(
                'ALTER TABLE review ADD COLUMN version INTEGER NOT NULL DEFAULT 1'))
            db.session.commit()
            app.logger.info("LOST-UPDATE: review.version column added via ALTER TABLE")
    _reconcile_model_indexes(app)
    _migrate_conversion_tags_csv_to_junction(app)


def _reconcile_model_indexes(app):
    """ARCH-FACTORY (audit W-7): create the model indexes an ALTERed file lacks.

    ``db.create_all()`` makes a table's indexes only together with the table;
    a column that reaches an EXISTING table through one of the ALTERs above
    arrives without the index its model declares (``index=True``). Measured
    on the Prod file 2026-10-07: 17 indexes against the 20 of a fresh
    create_all — ``ix_conversion_lifecycle_status``,
    ``ix_conversion_queue_position`` and ``ix_tag_parent_id`` missing (the
    LERN-TEXT one-off for ``ix_card_context_conversion_id`` was the only step
    that knew about this). So, after the ALTERs: every index ``db.metadata``
    declares and the live table lacks (by name) is created from the model's
    own ``Index`` object — same name, same columns, one DDL
    (``CREATE INDEX IF NOT EXISTS``), nothing to keep in step by hand.
    Nothing is dropped or renamed. A log line per created index.

    A missing UNIQUE index is refused, not built: over existing rows it is a
    correctness question (which duplicates lose?), not a migration step — it
    is logged as a WARNING and left to a deliberate migration. The model's
    one UNIQUE index today, ``ix_review_card_id``, was created with its table
    and is present on every file, so the refusal fires for nothing today.
    """
    inspector = inspect(db.engine)
    existing = set(inspector.get_table_names())
    for table in db.metadata.sorted_tables:
        if table.name not in existing:
            continue
        present = {ix['name'] for ix in inspector.get_indexes(table.name)}
        for index in sorted(table.indexes, key=lambda ix: ix.name or ''):
            if index.name in present:
                continue
            columns = ', '.join(c.name for c in index.columns)
            if index.unique:
                app.logger.warning(
                    f"ARCH-FACTORY: UNIQUE index {index.name} on {table.name} ({columns}) "
                    f"is missing and was NOT created — uniqueness over existing rows "
                    f"needs a deliberate migration")
                continue
            db.session.execute(CreateIndex(index, if_not_exists=True))
            db.session.commit()
            app.logger.info(
                f"ARCH-FACTORY: index {index.name} created on {table.name} ({columns})")


def _migrate_conversion_tags_csv_to_junction(app):
    # R2-A: drain the legacy Conversion.tags CSV column into the new
    # conversion_tags junction. Idempotent via the empty-CSV marker —
    # once a row is migrated we set tags='' so the next container start
    # skips it. Defensive against User-Detach-then-Restart races: the CSV
    # is *not* re-read after the first run, so a deleted junction row will
    # not be resurrected from the dead column.
    from models import Conversion, Tag
    candidates = Conversion.query.filter(
        Conversion.tags.isnot(None),
        Conversion.tags != '',
    ).all()
    if not candidates:
        return
    migrated = 0
    for conv in candidates:
        names = [n.strip() for n in (conv.tags or '').split(',') if n.strip()]
        for name in names:
            tag = Tag.get_or_create(conv.user_id, name)
            if tag and tag not in conv.tag_refs:
                conv.tag_refs.append(tag)
        conv.tags = ''
        migrated += 1
    db.session.commit()
    app.logger.info(
        f"R2-A: migrated {migrated} conversions from CSV to conversion_tags junction"
    )
