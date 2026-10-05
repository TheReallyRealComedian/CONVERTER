"""LERN-TEXT — Lerntexte zu Karten und Sammlungen (Phase 1: Schema + API).

Ein Lerntext ist ein gewöhnliches Library-Dokument; was ihn zum Lerntext
macht, ist die Verknüpfung. Zwei Richtungen:

* **Karte → genau eine Textstelle**: ``Card.context_conversion_id`` +
  ``Card.context_heading`` (NULL = der Text als Ganzes), geschrieben über
  ``context`` an ``POST``/``PATCH /api/cards``. Fail-closed beim Schreiben:
  fremdes Dokument → 404, Überschrift fehlt im Text → 400, mehrdeutig → 409
  — dieselbe Überschriften-Erkennung und Vergleichsregel wie
  ``services.markdown_sections.replace_section`` (fenced-code-aware,
  level-agnostisch, ``#``-Präfix gestrippt).
* **Sammlung → Texte**: Junction ``collection_documents`` mit Position,
  ersetzt über ``PUT /api/collections/<id>/documents``; ``GET
  /api/collections`` bleibt ein blankes Array (iOS dekodiert
  ``[LearnCollection]``) und trägt je Eintrag additiv ``documents``.

Löschen ist ORM-Sache (SQLite ohne FK-Pragma): Dokument weg → Karten-Kontext
NULL + Junction leer im selben Commit; Sammlung weg → Junction leer; Karte
weg → nichts weiter. Jeder Fall fährt über den bestehenden Endpunkt.

``to_dict`` ohne N+1: der Dokument-Titel reist mit dem Karten-SELECT, die Zahl
der SQL-Statements gegen ``conversion`` hängt nicht von der Kartenzahl ab.
"""
import re
from datetime import datetime, timezone

import pytest
from sqlalchemy import event, inspect, text

from app_pkg import _run_pending_migrations
from models import Card, Collection, Conversion, Review, User, db


CARD_TOKEN = 'lern-text-test-token-7f1c'
CARDS_URL = '/api/cards'

# Three headings (one with umlaut + spaces), a fence with a ``#`` line that
# must NOT count as a heading, and nothing ambiguous.
DOC = """# Einleitung

Ein kurzer Text.

## Säuren und Basen

Mehr Text.

```python
# Kein Heading
x = 1
```

## Redox

Schluss.
"""

# ``Intro`` twice at different levels → ambiguous (level-agnostic rule).
AMBIGUOUS_DOC = """# Intro

a

### Intro

b
"""


# --- helpers -----------------------------------------------------------------

def _auth(token=CARD_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _make_user(app, username='mallory'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def _make_doc(app, user_id, content=DOC, title='Chemie-Basics'):
    with app.app_context():
        conv = Conversion(user_id=user_id, conversion_type='markdown_input',
                          title=title, content=content)
        db.session.add(conv)
        db.session.commit()
        return conv.id


def _make_collection(app, user_id, name='Chemie'):
    with app.app_context():
        col = Collection(user_id=user_id, name=name)
        db.session.add(col)
        db.session.commit()
        return col.id


def _payload(**overrides):
    p = {'type': 'atomic', 'front': 'Was ist eine Säure?', 'back': 'Protonendonator.'}
    p.update(overrides)
    return p


def _post_card(client, **overrides):
    return client.post(CARDS_URL, headers=_auth(), json=_payload(**overrides))


def _context_columns(app, card_id):
    with app.app_context():
        return db.session.execute(
            text('SELECT context_conversion_id, context_heading FROM card WHERE id = :id'),
            {'id': card_id},
        ).fetchone()


def _junction_rows(app, collection_id):
    """``[(conversion_id, position), …]`` in position order — raw SQL, so the
    test reads the table and not a relationship's view of it."""
    with app.app_context():
        return db.session.execute(
            text('SELECT conversion_id, position FROM collection_documents '
                 'WHERE collection_id = :c ORDER BY position'),
            {'c': collection_id},
        ).fetchall()


def _put_documents(client, collection_id, documents):
    return client.put(f'/api/collections/{collection_id}/documents',
                      json={'documents': documents})


# --- A. find_heading: one recognition, one comparison rule --------------------

def test_find_heading_counts_matches_outside_fences():
    from services.markdown_sections import find_heading

    assert find_heading(DOC, 'Einleitung') == 1
    assert find_heading(DOC, 'Säuren und Basen') == 1
    assert find_heading(DOC, 'Redox') == 1
    assert find_heading(DOC, 'Nicht da') == 0
    # The ``# Kein Heading`` line sits inside a fence — not a heading.
    assert find_heading(DOC, 'Kein Heading') == 0
    # Same normalisation as replace_section: leading hashes + whitespace go.
    assert find_heading(DOC, '## Redox') == 1
    assert find_heading(DOC, '  Redox  ') == 1
    assert find_heading(AMBIGUOUS_DOC, 'Intro') == 2


@pytest.mark.parametrize('markdown, heading', [
    (DOC, 'Säuren und Basen'),        # unique → 1 / replace works
    (DOC, '## Säuren und Basen'),     # hash-prefixed target, same rule
    (DOC, 'Kein Heading'),            # only inside a fence → 0 / not found
    (DOC, 'Fehlt'),                   # absent → 0 / not found
    (AMBIGUOUS_DOC, 'Intro'),         # two levels → 2 / ambiguous
    (AMBIGUOUS_DOC, '# Intro'),
])
def test_find_heading_and_replace_section_agree(markdown, heading):
    # The context check and the section writer must never disagree on what
    # "the heading" is: feed both the same input and compare their verdicts.
    from services.markdown_sections import (SectionAmbiguous, SectionNotFound,
                                            find_heading, replace_section)

    count = find_heading(markdown, heading)
    try:
        replace_section(markdown, heading, '## X\n\nneu')
        verdict = 'ok'
    except SectionNotFound:
        verdict = 'not_found'
    except SectionAmbiguous:
        verdict = 'ambiguous'
    assert {0: 'not_found', 1: 'ok'}.get(count, 'ambiguous') == verdict


# --- B. Karte → Textstelle ----------------------------------------------------

def test_create_card_with_context_persists_and_serializes(app, authenticated_client,
                                                          test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])

    resp = _post_card(authenticated_client,
                      context={'document_id': doc_id, 'heading': 'Säuren und Basen'})
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()
    assert body['context'] == {
        'document_id': doc_id,
        'document_title': 'Chemie-Basics',
        'heading': 'Säuren und Basen',
        # encodeURIComponent form: umlaut as UTF-8 bytes, space as %20.
        'url': f'/library/{doc_id}#h=S%C3%A4uren%20und%20Basen',
    }
    assert _context_columns(app, body['id']) == (doc_id, 'Säuren und Basen')

    # The session read (what the review UI and the MCP get) carries the same.
    detail = authenticated_client.get(f"{CARDS_URL}/{body['id']}").get_json()
    assert detail['context'] == body['context']


@pytest.mark.parametrize('heading, encoded', [
    ('Säuren & Basen (Teil 1)', 'S%C3%A4uren%20%26%20Basen%20(Teil%201)'),
    ("Oli's Notizen!", "Oli's%20Notizen!"),
    ('C# und F?', 'C%23%20und%20F%3F'),
    ('100% Lösung', '100%25%20L%C3%B6sung'),
    ('a*b~c', 'a*b~c'),
])
def test_context_url_encodes_like_encodeuricomponent(app, authenticated_client, test_user,
                                                     monkeypatch, heading, encoded):
    # The reader decodes the hash with decodeURIComponent — the server must
    # produce exactly that alphabet: ``A-Za-z0-9 - _ . ! ~ * ' ( )`` unescaped,
    # everything else UTF-8 percent-encoded.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'], content=f'# Start\n\ntext\n\n## {heading}\n\nmehr\n')
    resp = _post_card(authenticated_client, context={'document_id': doc_id, 'heading': heading})
    assert resp.status_code == 201, resp.get_json()
    assert resp.get_json()['context']['url'] == f'/library/{doc_id}#h={encoded}'


def test_create_card_with_document_only_context(app, authenticated_client, test_user,
                                                monkeypatch):
    # No heading = "the text as a whole": heading NULL, url without a hash.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])

    for ctx in ({'document_id': doc_id}, {'document_id': doc_id, 'heading': None}):
        resp = _post_card(authenticated_client, context=ctx)
        assert resp.status_code == 201, resp.get_json()
        body = resp.get_json()
        assert body['context'] == {
            'document_id': doc_id,
            'document_title': 'Chemie-Basics',
            'heading': None,
            'url': f'/library/{doc_id}',
        }
        assert _context_columns(app, body['id']) == (doc_id, None)


def test_create_card_without_context_serializes_null(app, authenticated_client, test_user,
                                                     monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    for payload in (_payload(), _payload(context=None)):
        resp = authenticated_client.post(CARDS_URL, headers=_auth(), json=payload)
        assert resp.status_code == 201
        body = resp.get_json()
        assert 'context' in body and body['context'] is None
        assert _context_columns(app, body['id']) == (None, None)


def test_context_heading_normalises_like_replace_section(app, authenticated_client,
                                                         test_user, monkeypatch):
    # ``## Redox`` and ``  Redox  `` both address the heading "Redox"; the
    # stored value is the canonical text so the url/hash matches the reader.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    for raw in ('## Redox', '  Redox  '):
        resp = _post_card(authenticated_client, context={'document_id': doc_id, 'heading': raw})
        assert resp.status_code == 201, resp.get_json()
        assert resp.get_json()['context']['heading'] == 'Redox'
        assert resp.get_json()['context']['url'] == f'/library/{doc_id}#h=Redox'


def test_context_blank_heading_means_whole_document(app, authenticated_client, test_user,
                                                    monkeypatch):
    # House style (note, SVG fields): an empty string is a clear-intent →
    # NULL, i.e. the document as a whole. Never a 400, never stored as ''.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    resp = _post_card(authenticated_client, context={'document_id': doc_id, 'heading': '   '})
    assert resp.status_code == 201, resp.get_json()
    assert resp.get_json()['context']['heading'] is None
    assert _context_columns(app, resp.get_json()['id']) == (doc_id, None)


def test_patch_card_context_sets_changes_and_clears(app, authenticated_client, test_user,
                                                    monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    other_id = _make_doc(app, test_user['id'], title='Zweiter Text',
                         content='# Anfang\n\nx\n\n## Mitte\n\ny\n')
    cid = _post_card(authenticated_client).get_json()['id']
    url = f'{CARDS_URL}/{cid}'

    # set
    resp = authenticated_client.patch(url, headers=_auth(),
                                      json={'context': {'document_id': doc_id,
                                                        'heading': 'Einleitung'}})
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()['context']['heading'] == 'Einleitung'
    assert _context_columns(app, cid) == (doc_id, 'Einleitung')

    # change (another document, another heading)
    resp = authenticated_client.patch(url, headers=_auth(),
                                      json={'context': {'document_id': other_id,
                                                        'heading': 'Mitte'}})
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()['context'] == {
        'document_id': other_id, 'document_title': 'Zweiter Text',
        'heading': 'Mitte', 'url': f'/library/{other_id}#h=Mitte',
    }

    # a PATCH without the key leaves it alone
    resp = authenticated_client.patch(url, headers=_auth(), json={'note': 'n'})
    assert resp.status_code == 200
    assert resp.get_json()['context']['document_id'] == other_id

    # clear
    resp = authenticated_client.patch(url, headers=_auth(), json={'context': None})
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()['context'] is None
    assert _context_columns(app, cid) == (None, None)


def test_context_foreign_or_missing_document_404_writes_nothing(app, authenticated_client,
                                                                test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    mallory = _make_user(app)
    foreign_doc = _make_doc(app, mallory, title='Fremd')

    for doc_id in (foreign_doc, 999_999):
        resp = _post_card(authenticated_client,
                          context={'document_id': doc_id, 'heading': 'Einleitung'})
        assert resp.status_code == 404, resp.get_json()
        assert resp.get_json()['error'] == 'Dokument nicht gefunden.'
    with app.app_context():
        assert Card.query.count() == 0  # fail-closed: no card without its context

    # PATCH: the existing context survives a rejected write.
    own_doc = _make_doc(app, test_user['id'])
    cid = _post_card(authenticated_client,
                     context={'document_id': own_doc, 'heading': 'Redox'}).get_json()['id']
    resp = authenticated_client.patch(f'{CARDS_URL}/{cid}', headers=_auth(),
                                      json={'context': {'document_id': foreign_doc,
                                                        'heading': 'Einleitung'}})
    assert resp.status_code == 404
    assert _context_columns(app, cid) == (own_doc, 'Redox')


def test_context_heading_not_found_400_with_sentence(app, authenticated_client, test_user,
                                                     monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    resp = _post_card(authenticated_client,
                      context={'document_id': doc_id, 'heading': 'Nicht da'})
    assert resp.status_code == 400, resp.get_json()
    assert resp.get_json()['error'] == "Überschrift nicht gefunden: ‚Nicht da'."
    with app.app_context():
        assert Card.query.count() == 0


def test_context_heading_inside_code_fence_does_not_count(app, authenticated_client,
                                                          test_user, monkeypatch):
    # ``# Kein Heading`` is a Python comment inside a fence — fenced-code-aware
    # like replace_section, so it is NOT a valid target.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    resp = _post_card(authenticated_client,
                      context={'document_id': doc_id, 'heading': 'Kein Heading'})
    assert resp.status_code == 400
    assert resp.get_json()['error'] == "Überschrift nicht gefunden: ‚Kein Heading'."


def test_context_heading_ambiguous_409_names_the_count(app, authenticated_client, test_user,
                                                       monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    two = _make_doc(app, test_user['id'], content=AMBIGUOUS_DOC, title='Zwei')
    three = _make_doc(app, test_user['id'], content=AMBIGUOUS_DOC + '\n## Intro\n\nc\n',
                      title='Drei')

    resp = _post_card(authenticated_client, context={'document_id': two, 'heading': 'Intro'})
    assert resp.status_code == 409, resp.get_json()
    assert resp.get_json()['error'] == 'Überschrift kommt 2-mal vor.'

    resp = _post_card(authenticated_client, context={'document_id': three, 'heading': 'Intro'})
    assert resp.status_code == 409
    assert resp.get_json()['error'] == 'Überschrift kommt 3-mal vor.'
    with app.app_context():
        assert Card.query.count() == 0


@pytest.mark.parametrize('context', [
    {'heading': 'Einleitung'},            # heading without document_id
    {},                                   # nothing to point at
    {'document_id': None, 'heading': 'Einleitung'},
    {'document_id': 'abc'},
    {'document_id': True},
    {'document_id': 1.5},
    'nicht-ein-objekt',
    [1],
])
def test_context_malformed_400(app, authenticated_client, test_user, monkeypatch, context):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    _make_doc(app, test_user['id'])
    resp = _post_card(authenticated_client, context=context)
    assert resp.status_code == 400, resp.get_json()
    assert 'error' in resp.get_json()
    with app.app_context():
        assert Card.query.count() == 0


def test_context_heading_must_be_text_or_null(app, authenticated_client, test_user,
                                              monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    for bad in (12, True, ['Einleitung'], {'x': 1}):
        resp = _post_card(authenticated_client, context={'document_id': doc_id, 'heading': bad})
        assert resp.status_code == 400, resp.get_json()
    with app.app_context():
        assert Card.query.count() == 0


def test_patch_rejected_context_leaves_other_fields_untouched(app, authenticated_client,
                                                              test_user, monkeypatch):
    # Fail-closed as a unit: a PATCH that carries a good ``front`` AND a bad
    # context writes nothing — the card is exactly as before.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    cid = _post_card(authenticated_client, front='Alt').get_json()['id']
    resp = authenticated_client.patch(f'{CARDS_URL}/{cid}', headers=_auth(),
                                      json={'front': 'Neu',
                                            'context': {'document_id': doc_id,
                                                        'heading': 'Fehlt'}})
    assert resp.status_code == 400
    detail = authenticated_client.get(f'{CARDS_URL}/{cid}').get_json()
    assert detail['front'] == 'Alt'
    assert detail['context'] is None


def test_card_summary_stays_without_context(app, authenticated_client, test_user,
                                            monkeypatch):
    # Decision: the slim list form carries NO context (like the figures) —
    # the detail read and review-state do.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    cid = _post_card(authenticated_client,
                     context={'document_id': doc_id, 'heading': 'Redox'}).get_json()['id']
    rows = authenticated_client.get(CARDS_URL).get_json()
    assert len(rows) == 1 and rows[0]['id'] == cid
    assert 'context' not in rows[0]
    assert authenticated_client.get(f'{CARDS_URL}/{cid}').get_json()['context']['heading'] == 'Redox'


def test_review_state_cards_carry_context(app, authenticated_client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    doc_id = _make_doc(app, test_user['id'])
    with_ctx = _post_card(authenticated_client,
                          context={'document_id': doc_id, 'heading': 'Redox'}).get_json()['id']
    without = _post_card(authenticated_client).get_json()['id']
    state = authenticated_client.get('/api/review-state').get_json()
    by_id = {c['id']: c for c in state['due_cards']}
    assert by_id[with_ctx]['context']['url'] == f'/library/{doc_id}#h=Redox'
    assert by_id[with_ctx]['context']['document_title'] == 'Chemie-Basics'
    assert by_id[without]['context'] is None


# --- C. Sammlung → Texte -------------------------------------------------------

def test_put_collection_documents_replaces_orders_and_lists(app, authenticated_client,
                                                            test_user):
    uid = test_user['id']
    col = _make_collection(app, uid)
    a = _make_doc(app, uid, title='A')
    b = _make_doc(app, uid, title='B')
    c = _make_doc(app, uid, title='C')

    resp = _put_documents(authenticated_client, col, [c, a])
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()['documents'] == [
        {'id': c, 'title': 'C', 'url': f'/library/{c}'},
        {'id': a, 'title': 'A', 'url': f'/library/{a}'},
    ]
    assert _junction_rows(app, col) == [(c, 0), (a, 1)]

    # Replace, not merge: the old rows are gone, positions start at 0 again.
    resp = _put_documents(authenticated_client, col, [b, a])
    assert resp.status_code == 200
    assert [d['id'] for d in resp.get_json()['documents']] == [b, a]
    assert _junction_rows(app, col) == [(b, 0), (a, 1)]

    listing = authenticated_client.get('/api/collections').get_json()
    assert isinstance(listing, list)  # iOS decodes [LearnCollection] — never a wrapper
    entry = next(e for e in listing if e['id'] == col)
    assert entry['documents'] == [
        {'id': b, 'title': 'B', 'url': f'/library/{b}'},
        {'id': a, 'title': 'A', 'url': f'/library/{a}'},
    ]
    assert entry['card_count'] == 0 and entry['due_count'] == 0  # unchanged neighbours


def test_put_collection_documents_empty_list_clears(app, authenticated_client, test_user):
    uid = test_user['id']
    col = _make_collection(app, uid)
    a = _make_doc(app, uid, title='A')
    assert _put_documents(authenticated_client, col, [a]).status_code == 200
    resp = _put_documents(authenticated_client, col, [])
    assert resp.status_code == 200
    assert resp.get_json()['documents'] == []
    assert _junction_rows(app, col) == []
    with app.app_context():
        assert db.session.get(Conversion, a) is not None  # the document itself stays


def test_put_collection_documents_dedupes_repeated_ids(app, authenticated_client, test_user):
    uid = test_user['id']
    col = _make_collection(app, uid)
    a = _make_doc(app, uid, title='A')
    b = _make_doc(app, uid, title='B')
    resp = _put_documents(authenticated_client, col, [a, a, b, a])
    assert resp.status_code == 200, resp.get_json()
    assert [d['id'] for d in resp.get_json()['documents']] == [a, b]
    assert _junction_rows(app, col) == [(a, 0), (b, 1)]


def test_put_collection_documents_foreign_or_missing_404_writes_nothing(app,
                                                                        authenticated_client,
                                                                        test_user):
    uid = test_user['id']
    mallory = _make_user(app)
    col = _make_collection(app, uid)
    a = _make_doc(app, uid, title='A')
    b = _make_doc(app, uid, title='B')
    foreign = _make_doc(app, mallory, title='Fremd')
    assert _put_documents(authenticated_client, col, [a]).status_code == 200

    for docs in ([b, foreign], [foreign], [b, 999_999]):
        resp = _put_documents(authenticated_client, col, docs)
        assert resp.status_code == 404, resp.get_json()
        assert resp.get_json()['error'] == 'Dokument nicht gefunden.'
        # Atomic: the previous list is untouched, b was NOT written.
        assert _junction_rows(app, col) == [(a, 0)]


@pytest.mark.parametrize('body', [
    None,
    [],
    {},
    {'documents': None},
    {'documents': 'abc'},
    {'documents': [1, 'x']},
    {'documents': [True]},
    {'documents': [1.5]},
])
def test_put_collection_documents_malformed_400(app, authenticated_client, test_user, body):
    uid = test_user['id']
    col = _make_collection(app, uid)
    a = _make_doc(app, uid, title='A')
    assert _put_documents(authenticated_client, col, [a]).status_code == 200
    resp = authenticated_client.put(f'/api/collections/{col}/documents', json=body)
    assert resp.status_code == 400, resp.get_json()
    assert _junction_rows(app, col) == [(a, 0)]


def test_put_collection_documents_owner_404_and_login_required(app, client,
                                                               authenticated_client,
                                                               test_user):
    uid = test_user['id']
    mallory = _make_user(app)
    foreign_col = _make_collection(app, mallory, name='Fremde Sammlung')
    a = _make_doc(app, uid, title='A')
    resp = _put_documents(authenticated_client, foreign_col, [a])
    assert resp.status_code == 404
    assert _junction_rows(app, foreign_col) == []
    assert _put_documents(authenticated_client, 999_999, [a]).status_code == 404


def test_put_collection_documents_anonymous_is_redirected(app, client, test_user):
    uid = test_user['id']
    col = _make_collection(app, uid)
    resp = client.put(f'/api/collections/{col}/documents', json={'documents': []})
    assert resp.status_code == 302  # cookie-web posture: login redirect, like the siblings


def test_list_collections_always_carries_documents(app, authenticated_client, test_user):
    uid = test_user['id']
    empty = _make_collection(app, uid, name='Leer')
    full = _make_collection(app, uid, name='Voll')
    a = _make_doc(app, uid, title='A')
    assert _put_documents(authenticated_client, full, [a]).status_code == 200

    listing = authenticated_client.get('/api/collections').get_json()
    assert isinstance(listing, list)
    by_id = {e['id']: e for e in listing}
    assert by_id[empty]['documents'] == []
    assert by_id[full]['documents'] == [{'id': a, 'title': 'A', 'url': f'/library/{a}'}]

    # The create response carries the (empty) list too — one shape everywhere.
    created = authenticated_client.post('/api/collections', json={'name': 'Neu'}).get_json()
    assert created['documents'] == []


# --- D. Löschen über die Endpunkte -------------------------------------------

def test_delete_document_nulls_card_context_and_clears_junction(app, authenticated_client,
                                                                test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    doc_id = _make_doc(app, uid)
    keep_id = _make_doc(app, uid, title='Bleibt')
    col = _make_collection(app, uid)
    assert _put_documents(authenticated_client, col, [doc_id, keep_id]).status_code == 200
    cid = _post_card(authenticated_client,
                     context={'document_id': doc_id, 'heading': 'Redox'}).get_json()['id']
    other = _post_card(authenticated_client,
                       context={'document_id': keep_id, 'heading': 'Einleitung'}).get_json()['id']

    assert authenticated_client.delete(f'/api/conversions/{doc_id}').status_code == 200

    # Same commit: the card lost its context (both columns), the junction row
    # is gone, the card and the collection survive, the other links stay.
    assert _context_columns(app, cid) == (None, None)
    assert authenticated_client.get(f'{CARDS_URL}/{cid}').get_json()['context'] is None
    assert _context_columns(app, other) == (keep_id, 'Einleitung')
    assert _junction_rows(app, col) == [(keep_id, 1)]
    with app.app_context():
        assert db.session.get(Conversion, doc_id) is None
        assert db.session.get(Card, cid) is not None
        assert db.session.get(Collection, col) is not None


def test_delete_collection_clears_junction_documents_and_cards_survive(app,
                                                                       authenticated_client,
                                                                       test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    doc_id = _make_doc(app, uid)
    col = _make_collection(app, uid)
    assert _put_documents(authenticated_client, col, [doc_id]).status_code == 200
    cid = _post_card(authenticated_client,
                     context={'document_id': doc_id, 'heading': 'Redox'}).get_json()['id']

    assert authenticated_client.delete(f'/api/collections/{col}').status_code == 200

    with app.app_context():
        assert db.session.get(Collection, col) is None
        assert db.session.get(Conversion, doc_id) is not None
        assert db.session.execute(
            text('SELECT count(*) FROM collection_documents WHERE collection_id = :c'),
            {'c': col}).scalar() == 0
    assert _context_columns(app, cid) == (doc_id, 'Redox')  # the card's own link is untouched


def test_delete_card_leaves_document_and_junction(app, authenticated_client, test_user,
                                                  monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    doc_id = _make_doc(app, uid)
    col = _make_collection(app, uid)
    assert _put_documents(authenticated_client, col, [doc_id]).status_code == 200
    cid = _post_card(authenticated_client,
                     context={'document_id': doc_id, 'heading': 'Redox'}).get_json()['id']

    assert authenticated_client.delete(f'{CARDS_URL}/{cid}').status_code == 200

    with app.app_context():
        assert db.session.get(Card, cid) is None
        assert db.session.get(Conversion, doc_id) is not None
        assert db.session.get(Collection, col) is not None
    assert _junction_rows(app, col) == [(doc_id, 0)]


# --- E. Kein N+1 -------------------------------------------------------------

def _count_conversion_statements(app, client, path):
    """Number of SQL statements that touch the ``conversion`` table during
    one request — the thing this sprint adds to the card serialisation."""
    hits = []

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        if 'conversion' in statement.lower():
            hits.append(statement)

    with app.app_context():
        engine = db.engine
    event.listen(engine, 'before_cursor_execute', before_cursor_execute)
    try:
        resp = client.get(path)
        assert resp.status_code == 200, resp.get_json()
        return len(hits), resp.get_json()
    finally:
        event.remove(engine, 'before_cursor_execute', before_cursor_execute)


def test_review_state_document_titles_do_not_scale_with_card_count(app, authenticated_client,
                                                                   test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    docs = [_make_doc(app, uid, title=f'Text {i}') for i in range(3)]
    headings = ('Einleitung', 'Säuren und Basen', 'Redox')

    def add_cards(n):
        for i in range(n):
            resp = _post_card(authenticated_client,
                              context={'document_id': docs[i % 3], 'heading': headings[i % 3]})
            assert resp.status_code == 201, resp.get_json()

    # ?uncapped=1 — the daily new-card limit (10) would otherwise trim the
    # queue and the two requests would serialise the same number of cards.
    add_cards(5)
    small, body = _count_conversion_statements(app, authenticated_client,
                                               '/api/review-state?uncapped=1')
    assert len(body['due_cards']) == 5

    add_cards(45)
    large, body = _count_conversion_statements(app, authenticated_client,
                                               '/api/review-state?uncapped=1')
    assert len(body['due_cards']) == 50
    assert all(c['context']['document_title'].startswith('Text ') for c in body['due_cards'])

    # Independent of the card count: the titles ride with the card SELECT,
    # not one query per card.
    assert small == large, (small, large)


# --- F. Migration -------------------------------------------------------------

def test_migration_adds_context_columns_and_junction_idempotently(app):
    # Simulate the pre-LERN-TEXT schema: no context columns on card, no
    # junction table. The CARD-SVG pattern (DROP COLUMN) does not work here —
    # SQLite refuses DROP COLUMN on a column named in a FOREIGN KEY clause
    # (measured, 3.51: "unknown column … in foreign key definition"). So the
    # legacy ``card`` is rebuilt from the real table's OWN DDL minus the two
    # columns: the real table steps aside under another name
    # (legacy_alter_table=ON keeps the other tables' FK text pointing at
    # "card"), the legacy copy takes its name, the startup sequence
    # (create_all, then the ALTERs) runs against it, and the finally block
    # swaps the real table back — a mid-test failure cannot poison the
    # session-scoped engine.
    with app.app_context():
        engine = db.engine
        ddl = db.session.execute(text(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='card'")).scalar()
        kept = [ln for ln in ddl.split('\n')
                if 'context_conversion_id' not in ln and 'context_heading' not in ln]
        # The context FK was the last constraint line — mend the dangling comma.
        legacy_ddl = re.sub(r',\s*\n\)', '\n)', '\n'.join(kept))
        assert 'context_' not in legacy_ddl and 'highlight_id' in legacy_ddl
        legacy_mode = db.session.execute(text('PRAGMA legacy_alter_table')).scalar()
        db.session.execute(text('PRAGMA legacy_alter_table=ON'))
        db.session.execute(text('DROP INDEX IF EXISTS ix_card_context_conversion_id'))
        db.session.execute(text('ALTER TABLE card RENAME TO card_real'))
        db.session.execute(text(legacy_ddl))
        db.session.execute(text('DROP TABLE IF EXISTS collection_documents'))
        db.session.commit()
        try:
            insp = inspect(engine)
            cols = {c['name'] for c in insp.get_columns('card')}
            assert 'context_conversion_id' not in cols and 'context_heading' not in cols
            assert 'collection_documents' not in insp.get_table_names()

            # The startup sequence: create_all (new tables) then the ALTERs.
            db.create_all()
            _run_pending_migrations(app)
            insp = inspect(engine)
            cols = {c['name'] for c in insp.get_columns('card')}
            assert {'context_conversion_id', 'context_heading'} <= cols
            assert 'collection_documents' in insp.get_table_names()
            junction_cols = {c['name'] for c in insp.get_columns('collection_documents')}
            assert {'collection_id', 'conversion_id', 'position'} <= junction_cols
            assert any(ix['column_names'] == ['context_conversion_id']
                       for ix in insp.get_indexes('card'))

            # Second pass is a no-op — no error, each column present exactly once.
            db.create_all()
            _run_pending_migrations(app)
            cols = [c['name'] for c in inspect(engine).get_columns('card')]
            assert cols.count('context_conversion_id') == 1
            assert cols.count('context_heading') == 1
        finally:
            db.session.rollback()
            db.session.execute(text('PRAGMA legacy_alter_table=ON'))
            db.session.execute(text('DROP TABLE IF EXISTS card'))
            db.session.execute(text('ALTER TABLE card_real RENAME TO card'))
            db.session.execute(text(f'PRAGMA legacy_alter_table={int(legacy_mode or 0)}'))
            db.session.commit()
            db.create_all()               # collection_documents, if the test died early
            _run_pending_migrations(app)  # the context index went with the legacy table


def test_card_context_columns_exist_in_fresh_schema(app, test_user):
    # create_all alone (a fresh DB) yields the columns + the junction — the
    # ORM model declares them, the migration only patches old files.
    with app.app_context():
        insp = inspect(db.engine)
        cols = {c['name'] for c in insp.get_columns('card')}
        assert {'context_conversion_id', 'context_heading'} <= cols
        assert 'collection_documents' in insp.get_table_names()
        # ORM round-trip of the two columns.
        doc_id = _make_doc(app, test_user['id'])
        card = Card(user_id=test_user['id'], type='atomic', front='Q', back='A',
                    context_conversion_id=doc_id, context_heading='Redox')
        card.review = Review(due=datetime.now(timezone.utc))
        db.session.add(card)
        db.session.commit()
        assert Card.query.get(card.id).context_heading == 'Redox'
