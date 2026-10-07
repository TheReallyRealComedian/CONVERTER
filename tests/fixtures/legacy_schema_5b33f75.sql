-- Legacy schema of CONVERTER BEFORE the first inline migration (ARCH-FACTORY P0 gate).
--
-- Basis:        12d2c6dc4004e6f9b4c486b8a0d0e0ba09971dc2 (5b33f75^)
--               2026-05-25 16:30:40 +0200  R1-B-A: highlight-core -- schema, api, selection-ux, save+reapply
-- First ALTER:  5b33f7511f2a36adab2177ff478e70535deb412e (5b33f75)
--               2026-05-25 18:46:55 +0200  R1-B-B: highlight-notes + sidebar -- schema-add, PATCH, sidebar-list, scroll-to-highlight  -> highlight.note
--
-- Generated once, 2026-10-07, from `git show 5b33f75^:models.py` loaded as a module
-- (its own flask_sqlalchemy.SQLAlchemy() instance), Flask app with
-- SQLALCHEMY_DATABASE_URI on an empty SQLite file, db.create_all(), then
-- `SELECT sql FROM sqlite_master` (tables in creation order, indexes by name).
-- Toolchain of that run: Flask 3.1.3, Flask-SQLAlchemy 3.1.1,
-- SQLAlchemy 2.0.46, SQLite 3.51.0, Python 3.12.2.
-- FROZEN: the basis does not move; this file is never regenerated. The boot
-- test (tests/test_boot_schema.py) loads it into a fresh file and proves that
-- create_app() brings it to the schema create_all() declares today.
--
-- The UNIQUE (username) constraint creates sqlite_autoindex_user_1 implicitly;
-- SQLite stores no SQL for it, so it does not appear below.

-- table user
CREATE TABLE user (
	id INTEGER NOT NULL, 
	username VARCHAR(80) NOT NULL, 
	password_hash VARCHAR(256) NOT NULL, 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (username)
);

-- table conversion
CREATE TABLE conversion (
	id INTEGER NOT NULL, 
	user_id INTEGER NOT NULL, 
	conversion_type VARCHAR(30) NOT NULL, 
	title VARCHAR(255) NOT NULL, 
	content TEXT NOT NULL, 
	source_filename VARCHAR(255), 
	source_mimetype VARCHAR(100), 
	source_size_bytes INTEGER, 
	metadata_json TEXT, 
	tags VARCHAR(500), 
	is_favorite BOOLEAN, 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES user (id)
);

-- table highlight
CREATE TABLE highlight (
	id INTEGER NOT NULL, 
	conversion_id INTEGER NOT NULL, 
	exact TEXT NOT NULL, 
	prefix TEXT, 
	suffix TEXT, 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(conversion_id) REFERENCES conversion (id)
);

-- index ix_conversion_conversion_type on conversion
CREATE INDEX ix_conversion_conversion_type ON conversion (conversion_type);

-- index ix_conversion_created_at on conversion
CREATE INDEX ix_conversion_created_at ON conversion (created_at);

-- index ix_conversion_user_id on conversion
CREATE INDEX ix_conversion_user_id ON conversion (user_id);

-- index ix_highlight_conversion_id on highlight
CREATE INDEX ix_highlight_conversion_id ON highlight (conversion_id);

-- index ix_highlight_created_at on highlight
CREATE INDEX ix_highlight_created_at ON highlight (created_at);
