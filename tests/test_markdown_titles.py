"""TITLE-FIX — derive a real title from the first heading, not the first line.

Three layers:

* Pure unit tests for ``derive_title`` / ``_is_degenerate_title`` (the heart of
  the fix): HTML-comment- and fenced-code-aware first-heading extraction, with
  emphasis stripped, falling back to the first non-empty line then ``Untitled``.
* Session create route (``POST /api/conversions``) — derives only when the
  posted title is degenerate; a real client title is kept verbatim; content is
  never mutated.
* Token-authenticated ingest route (``POST /api/ingest/conversion``) — same
  smart-derive, same verbatim rule (Bearer scaffolding mirrors test_ingest).
"""
import pytest

from models import Conversion, User, db
from services.markdown_sections import derive_title, _is_degenerate_title


# --- derive_title: first heading wins, comment- and fence-aware ---

def test_heading_after_single_line_comment():
    md = '<!-- Seite 1 -->\n\n# Addiction to product expression\n\nbody'
    assert derive_title(md) == 'Addiction to product expression'


def test_heading_after_multiline_comment_and_hash_inside_comment_ignored():
    """A ``#`` *inside* a multi-line HTML comment must not become the title;
    the real heading after the comment does (``_iter_headings`` only knows
    code fences, so the comment strip is what guards this)."""
    md = '\n'.join([
        '<!--',
        'Seite 1',
        '# fake heading inside the comment',
        '-->',
        '# Real Heading',
        'body',
    ])
    assert derive_title(md) == 'Real Heading'


def test_heading_after_multiple_comments():
    md = '<!-- Seite 1 --><!-- Grafik: chart -->\n# Real\nbody'
    assert derive_title(md) == 'Real'


@pytest.mark.parametrize('md,expected', [
    ('# *Optimization matters*', 'Optimization matters'),
    ('# **Bold Title**', 'Bold Title'),
    ('#   _Underscored_  ', 'Underscored'),
    ('## Plain Heading', 'Plain Heading'),
])
def test_emphasis_markers_stripped(md, expected):
    assert derive_title(md) == expected


def test_no_heading_falls_back_to_first_non_comment_line():
    md = '<!-- Seite 1 -->\n\nGTM is the topic\nmore text'
    assert derive_title(md) == 'GTM is the topic'


def test_heading_wins_over_an_earlier_non_empty_line():
    """Two-pass: headings are considered before the first-line fallback, so a
    paragraph above the heading does not pre-empt it."""
    md = 'intro paragraph\n# The Heading\nbody'
    assert derive_title(md) == 'The Heading'


def test_hash_inside_code_fence_is_not_the_title():
    """Reuse proof: ``_iter_headings`` skips fenced ``#`` lines, so the heading
    after the fence wins, not the ``# comment`` inside it."""
    md = '\n'.join([
        '```python',
        '# this is a comment, not a heading',
        'x = 1',
        '```',
        '# Real Heading',
    ])
    assert derive_title(md) == 'Real Heading'


@pytest.mark.parametrize('md', [
    '',
    '   \n\n  ',
    '<!-- only a comment -->',
    '<!--\nmulti-line\ncomment only\n-->',
])
def test_empty_or_comment_only_content_is_untitled(md):
    assert derive_title(md) == 'Untitled'


def test_derive_title_is_not_truncated():
    """The helper never truncates — the caller clips to its own column limit."""
    long = 'A' * 400
    assert derive_title(f'# {long}') == long


# --- _is_degenerate_title ---

@pytest.mark.parametrize('title', [
    None, '', '   ', 'Untitled', 'untitled', 'UNTITLED',
    'Untitled Markdown', 'untitled markdown', '<!-- Seite 1 -->',
])
def test_degenerate_titles(title):
    assert _is_degenerate_title(title) is True


@pytest.mark.parametrize('title', [
    'Real Title', 'GTM', 'Addiction to product expression',
    'Untitled Heroes',  # only the exact placeholders are degenerate
])
def test_real_titles_are_not_degenerate(title):
    assert _is_degenerate_title(title) is False


# --- Session create route: derive on degenerate, verbatim otherwise ---

DEGENERATE = pytest.mark.parametrize('posted', ['<!-- Seite 1 -->', '', 'Untitled'])


@DEGENERATE
def test_create_derives_title_when_posted_is_degenerate(app, authenticated_client,
                                                        test_user, posted):
    content = '<!-- Seite 1 -->\n\n# Addiction to product expression\n\nbody'
    resp = authenticated_client.post('/api/conversions', json={
        'conversion_type': 'markdown_input',
        'title': posted,
        'content': content,
    })
    assert resp.status_code == 201
    assert resp.get_json()['title'] == 'Addiction to product expression'
    # Content is stored verbatim — the page marker stays (Reader/EPUB need it).
    with app.app_context():
        row = Conversion.query.filter_by(user_id=test_user['id']).first()
        assert row.content == content


def test_create_derives_title_when_title_absent(authenticated_client):
    resp = authenticated_client.post('/api/conversions', json={
        'conversion_type': 'markdown_input',
        'content': '# Derived From Heading\nbody',
    })
    assert resp.status_code == 201
    assert resp.get_json()['title'] == 'Derived From Heading'


def test_create_keeps_real_title_verbatim(authenticated_client):
    resp = authenticated_client.post('/api/conversions', json={
        'conversion_type': 'markdown_input',
        'title': 'My Real Title',
        'content': '# Some Other Heading\nbody',
    })
    assert resp.status_code == 201
    assert resp.get_json()['title'] == 'My Real Title'


# --- Ingest route: same derive rule (Bearer scaffolding from test_ingest) ---

INGEST_TOKEN = 'title-fix-test-token-3b9f'
INGEST_URL = '/api/ingest/conversion'


def _make_ingest_user(app, username='solo'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def _ingest_headers(token=INGEST_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def test_ingest_derives_title_when_degenerate(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', INGEST_TOKEN)
    _make_ingest_user(app)
    resp = client.post(INGEST_URL, headers=_ingest_headers(), json={
        'conversion_type': 'ai_newsletter',
        'title': '<!-- Seite 1 -->',
        'content': '<!-- Seite 1 -->\n\n# Weekly AI Roundup\n\nbody',
    })
    assert resp.status_code == 201
    assert resp.get_json()['title'] == 'Weekly AI Roundup'


def test_ingest_empty_title_derives_from_heading(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', INGEST_TOKEN)
    _make_ingest_user(app)
    resp = client.post(INGEST_URL, headers=_ingest_headers(), json={
        'conversion_type': 'ai_newsletter',
        'title': '',
        'content': '# Derived Newsletter Title\nbody',
    })
    assert resp.status_code == 201
    assert resp.get_json()['title'] == 'Derived Newsletter Title'


def test_ingest_keeps_real_title_verbatim(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', INGEST_TOKEN)
    _make_ingest_user(app)
    resp = client.post(INGEST_URL, headers=_ingest_headers(), json={
        'conversion_type': 'ai_newsletter',
        'title': '2026-05-30 - AI Newsletter Analyse',
        'content': '# Some Heading\nbody',
    })
    assert resp.status_code == 201
    assert resp.get_json()['title'] == '2026-05-30 - AI Newsletter Analyse'
