"""MCP-DOCWRITE Phase 2 — the agent's token-authed document-content writes.

Both endpoints (``PATCH .../content`` = update_document, ``PATCH .../section`` =
replace_section) reuse the card-write gate (CARD_TOKEN → 503 fail-closed / 401),
are owner-scoped (foreign/missing conversion → 404), validate non-blank fields
(→ 400), and are CSRF-exempt. The section endpoint maps the parser's exceptions
onto 404 (not found) / 409 (ambiguous). Token scaffolding mirrors test_cards.py.
"""
from models import Conversion, User, db


CARD_TOKEN = 'r4-test-card-token-9b2e'


def _auth(token=CARD_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _make_conversion(app, user_id, content='# body', title='Doc'):
    with app.app_context():
        conv = Conversion(user_id=user_id, conversion_type='markdown_input',
                          title=title, content=content)
        db.session.add(conv)
        db.session.commit()
        return conv.id


def _make_other_user(app, username='mallory'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def _content_url(cid):
    return f'/api/conversions/{cid}/content'


def _section_url(cid):
    return f'/api/conversions/{cid}/section'


def _content(app, cid):
    with app.app_context():
        return Conversion.query.get(cid).content


# --- auth: 503 fail-closed / 401 missing+wrong (both endpoints) ---------------

def test_both_endpoints_fail_closed_without_token(app, client, test_user, monkeypatch):
    monkeypatch.delenv('CARD_TOKEN', raising=False)
    cid = _make_conversion(app, test_user['id'])
    # Unset -> 503 even with a Bearer present (config check precedes auth).
    assert client.patch(_content_url(cid), headers=_auth(), json={'content': 'x'}).status_code == 503
    assert client.patch(_section_url(cid), headers=_auth(),
                        json={'heading': 'h', 'content': 'x'}).status_code == 503
    # Empty string is "unset" too.
    monkeypatch.setenv('CARD_TOKEN', '')
    assert client.patch(_content_url(cid), headers=_auth(), json={'content': 'x'}).status_code == 503
    assert client.patch(_section_url(cid), headers=_auth(),
                        json={'heading': 'h', 'content': 'x'}).status_code == 503


def test_both_endpoints_401_missing_and_wrong_token(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'], content='# keep')
    assert client.patch(_content_url(cid), json={'content': 'x'}).status_code == 401
    assert client.patch(_content_url(cid), headers=_auth('nope'),
                        json={'content': 'x'}).status_code == 401
    assert client.patch(_section_url(cid), json={'heading': 'h', 'content': 'x'}).status_code == 401
    assert client.patch(_section_url(cid), headers=_auth('nope'),
                        json={'heading': 'h', 'content': 'x'}).status_code == 401
    assert _content(app, cid) == '# keep'  # nothing written on a rejected write


# --- owner-scope: foreign / missing conversion → 404 (both) ------------------

def test_both_endpoints_foreign_and_missing_conversion_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    # Target resolves to alice (test_user = first()); this conversion is mallory's.
    foreign = _make_conversion(app, _make_other_user(app), content='# secret')
    assert client.patch(_content_url(foreign), headers=_auth(),
                        json={'content': 'x'}).status_code == 404
    assert client.patch(_section_url(foreign), headers=_auth(),
                        json={'heading': 'h', 'content': 'x'}).status_code == 404
    # nonexistent id → 404 (no existence leak)
    assert client.patch(_content_url(999999), headers=_auth(),
                        json={'content': 'x'}).status_code == 404
    assert client.patch(_section_url(999999), headers=_auth(),
                        json={'heading': 'h', 'content': 'x'}).status_code == 404
    assert _content(app, foreign) == '# secret'  # untouched


# --- update_document --------------------------------------------------------

def test_update_document_replaces_content(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'], content='# Old\nold body')
    resp = client.patch(_content_url(cid), headers=_auth(),
                        json={'content': '# New\nbrand new'})
    assert resp.status_code == 200
    assert resp.get_json()['content'] == '# New\nbrand new'  # to_dict carries it
    assert _content(app, cid) == '# New\nbrand new'          # DB probe


def test_update_document_rejects_blank_missing_or_nonstr_content(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'], content='# Keep\nkeep me')
    for bad in [{}, {'content': ''}, {'content': '   '}, {'content': None}, {'content': 123}]:
        assert client.patch(_content_url(cid), headers=_auth(), json=bad).status_code == 400
    # a non-dict body is also 400
    assert client.patch(_content_url(cid), headers=_auth(), json=[1, 2]).status_code == 400
    assert _content(app, cid) == '# Keep\nkeep me'  # never wiped


# --- replace_section --------------------------------------------------------

def test_replace_section_replaces_only_target(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'],
                           content='# A\nbody a\n# B\nbody b\n# C\nbody c\n')
    resp = client.patch(_section_url(cid), headers=_auth(),
                        json={'heading': 'B', 'content': '# B\nNEW b'})
    assert resp.status_code == 200
    expected = '# A\nbody a\n# B\nNEW b\n# C\nbody c\n'
    assert resp.get_json()['content'] == expected
    assert _content(app, cid) == expected  # A and C untouched, B replaced


def test_replace_section_missing_heading_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'], content='# A\na\n')
    resp = client.patch(_section_url(cid), headers=_auth(),
                        json={'heading': 'Nope', 'content': '# Nope\nx'})
    assert resp.status_code == 404
    # distinct message from the owner-404 — proves the SectionNotFound path
    assert resp.get_json()['error'] == 'Abschnitt nicht gefunden.'
    assert _content(app, cid) == '# A\na\n'


def test_replace_section_ambiguous_409(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'], content='# Dup\na\n# Dup\nb\n')
    resp = client.patch(_section_url(cid), headers=_auth(),
                        json={'heading': 'Dup', 'content': '# Dup\nx'})
    assert resp.status_code == 409
    assert _content(app, cid) == '# Dup\na\n# Dup\nb\n'  # untouched on ambiguity


def test_replace_section_missing_fields_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_conversion(app, test_user['id'], content='# A\na\n')
    for bad in [{}, {'heading': 'A'}, {'content': 'x'},
                {'heading': '', 'content': 'x'}, {'heading': 'A', 'content': ''},
                {'heading': 'A', 'content': None}, [1, 2]]:
        assert client.patch(_section_url(cid), headers=_auth(), json=bad).status_code == 400
    assert _content(app, cid) == '# A\na\n'  # untouched


# --- CSRF: exempt only the two new write views ------------------------------

def test_docwrite_csrf_exempt_under_enforced_csrf(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    # conftest disables CSRF globally; flip it ON. Bearer-only PATCH carries no
    # CSRF token and must still succeed because the views are exempt.
    monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)
    cid = _make_conversion(app, test_user['id'], content='# A\na\n')
    assert client.patch(_content_url(cid), headers=_auth(),
                        json={'content': '# A\nnew'}).status_code == 200
    assert client.patch(_section_url(cid), headers=_auth(),
                        json={'heading': 'A', 'content': '# A\nnew2'}).status_code == 200


def test_only_docwrite_views_are_csrf_exempt(app):
    csrf = app.extensions['csrf']
    assert 'app_pkg.docwrite.api_update_document' in csrf._exempt_views
    assert 'app_pkg.docwrite.api_replace_section' in csrf._exempt_views
