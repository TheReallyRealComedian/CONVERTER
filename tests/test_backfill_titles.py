"""TITLE-FIX Phase 2 — the degenerate-title backfill script.

Exercises ``scripts.backfill_titles.run`` against the test DB: dry-run predicts
but writes nothing, ``--apply`` re-titles a degenerate row from its first
heading, a real title is never touched, and a second ``--apply`` is a no-op
(the row is no longer degenerate, so it is not even a candidate). The script
imports the runtime ``derive_title`` / ``_is_degenerate_title`` — no logic is
reimplemented here, so backfill and live endpoints can't drift.
"""
from models import Conversion, User, db
from scripts.backfill_titles import run


def _make_conversion(app, title, content, ctype='markdown_input'):
    with app.app_context():
        user = User.query.filter_by(username='solo').first()
        if user is None:
            user = User(username='solo')
            user.set_password('password1234')
            db.session.add(user)
            db.session.commit()
        c = Conversion(user_id=user.id, conversion_type=ctype,
                       title=title, content=content)
        db.session.add(c)
        db.session.commit()
        return c.id


def test_dry_run_predicts_but_writes_nothing(app):
    content = '<!-- Seite 1 -->\n\n# Real Heading\nbody'
    cid = _make_conversion(app, '<!-- Seite 1 -->', content)
    with app.app_context():
        predicted = run(apply_changes=False)
    assert predicted == 1  # the change is predicted …
    with app.app_context():
        # … but rolled back: the degenerate title is still on disk.
        assert db.session.get(Conversion, cid).title == '<!-- Seite 1 -->'


def test_apply_retitles_degenerate_row_and_leaves_content(app):
    content = '<!-- Seite 1 -->\n\n# Addiction to product expression\nbody'
    cid = _make_conversion(app, '<!-- Seite 1 -->', content)
    with app.app_context():
        changes = run(apply_changes=True)
    assert changes == 1
    with app.app_context():
        c = db.session.get(Conversion, cid)
        assert c.title == 'Addiction to product expression'
        assert c.content == content  # content is never mutated


def test_apply_retitles_blank_title_row(app):
    """The trim-aware prefilter catches a whitespace-only title too."""
    cid = _make_conversion(app, '   ', '# Heading From Blank\nbody')
    with app.app_context():
        assert run(apply_changes=True) == 1
    with app.app_context():
        assert db.session.get(Conversion, cid).title == 'Heading From Blank'


def test_real_titled_row_is_untouched(app):
    cid = _make_conversion(app, 'My Real Title', '# Some Other Heading\nbody')
    with app.app_context():
        changes = run(apply_changes=True)
    assert changes == 0
    with app.app_context():
        assert db.session.get(Conversion, cid).title == 'My Real Title'


def test_second_apply_is_noop(app):
    content = '<!-- Seite 1 -->\n\n# Real Heading\nbody'
    cid = _make_conversion(app, '<!-- Seite 1 -->', content)
    with app.app_context():
        assert run(apply_changes=True) == 1
    with app.app_context():
        # Title is now non-degenerate → no longer a candidate → nothing to do.
        assert run(apply_changes=True) == 0
    with app.app_context():
        assert db.session.get(Conversion, cid).title == 'Real Heading'


def test_skips_when_derivation_is_itself_degenerate(app):
    """A comment-only deck derives to 'Untitled' (degenerate) — keep the marker
    rather than write a placeholder over it (conservative)."""
    cid = _make_conversion(app, '<!-- Seite 1 -->', '<!-- Seite 1 -->\n<!-- Grafik -->')
    with app.app_context():
        changes = run(apply_changes=True)
    assert changes == 0
    with app.app_context():
        assert db.session.get(Conversion, cid).title == '<!-- Seite 1 -->'
