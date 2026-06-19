"""R4-LEARN Phase 1 — recall-layer schema, the Highlight before_delete event,
and the global /api/highlights/recent reader.

The before_delete tests are the rot→grün pair for the locked must-fix: SQLite
runs without PRAGMA foreign_keys=ON, so the ON DELETE SET NULL declared on
card.highlight_id is inert — an ORM before_delete event is the real mechanic.
Without it, deleting a Highlight leaves a dangling highlight_id on the card.
"""
from datetime import datetime

from models import Card, Conversion, Highlight, Review, Tag, User, db


# --- helpers -----------------------------------------------------------------

def _make_conversion(app, user_id, title='Doc', ctype='markdown_input', created_at=None):
    with app.app_context():
        conv = Conversion(user_id=user_id, conversion_type=ctype, title=title,
                          content='# body')
        if created_at is not None:
            conv.created_at = created_at
        db.session.add(conv)
        db.session.commit()
        return conv.id


def _make_highlight(app, conversion_id, exact='foo', note=None, created_at=None,
                    tags=None, user_id=None):
    with app.app_context():
        hl = Highlight(conversion_id=conversion_id, exact=exact, note=note)
        if created_at is not None:
            hl.created_at = created_at
        if tags:
            for name in tags:
                hl.tags.append(Tag.get_or_create(user_id, name))
        db.session.add(hl)
        db.session.commit()
        return hl.id


def _make_card(app, user_id, highlight_id=None, ctype='atomic', with_review=True, due=None):
    with app.app_context():
        card = Card(user_id=user_id, highlight_id=highlight_id, type=ctype,
                    front='Q', back='A')
        if with_review:
            card.review = Review(due=due or datetime.utcnow())
        db.session.add(card)
        db.session.commit()
        return card.id


# --- schema: tables exist + persistence + tags + review ----------------------

def test_card_review_and_card_tags_persist(app, test_user):
    uid = test_user['id']
    with app.app_context():
        card = Card(user_id=uid, type='atomic', front='Q', back='A',
                    source_doc_title='Doc', source_snapshot='quote in context')
        card.review = Review(due=datetime(2026, 6, 19, 12, 0, 0))
        card.tags.append(Tag.get_or_create(uid, 'Biologie'))
        db.session.add(card)
        db.session.commit()
        cid = card.id

    with app.app_context():
        card = db.session.get(Card, cid)
        assert card is not None
        assert card.type == 'atomic'
        assert card.state == 'ok'           # column default
        assert card.created_by == 'agent'   # column default
        assert card.source_doc_title == 'Doc'
        # 1:1 review back-reference
        assert card.review is not None
        assert card.review.card_id == cid
        assert card.review.reps == 0        # column default
        assert card.review.lapses == 0
        # tags via card_tags, normalised lowercase
        assert [t.name for t in card.tags] == ['biologie']
        d = card.to_dict()
        assert d['review']['card_id'] == cid
        assert d['tags'][0]['name'] == 'biologie'
        assert set(d['tags'][0].keys()) == {'id', 'name'}


# --- before_delete: provenance break, card + review survive (rot→grün) -------

def test_before_delete_highlight_nulls_card_keeps_card_and_review(app, test_user):
    uid = test_user['id']
    conv_id = _make_conversion(app, uid)
    hl_id = _make_highlight(app, conv_id, exact='quote')
    card_id = _make_card(app, uid, highlight_id=hl_id)

    with app.app_context():
        db.session.delete(db.session.get(Highlight, hl_id))
        db.session.commit()

    with app.app_context():
        assert db.session.get(Highlight, hl_id) is None
        card = db.session.get(Card, card_id)
        assert card is not None                  # card survives the delete
        assert card.highlight_id is None         # provenance link broken
        assert card.review is not None           # review survives too


def test_before_delete_via_conversion_cascade_nulls_card(app, test_user):
    uid = test_user['id']
    conv_id = _make_conversion(app, uid)
    hl_id = _make_highlight(app, conv_id, exact='quote')
    card_id = _make_card(app, uid, highlight_id=hl_id)

    # Deleting the parent Conversion sweeps the Highlight via delete-orphan; the
    # before_delete event still fires per-highlight → card.highlight_id nulled.
    with app.app_context():
        db.session.delete(db.session.get(Conversion, conv_id))
        db.session.commit()

    with app.app_context():
        assert Highlight.query.count() == 0      # cascaded away
        card = db.session.get(Card, card_id)
        assert card is not None                  # card survives the cascade
        assert card.highlight_id is None
        assert card.review is not None


def test_before_delete_only_nulls_matching_cards(app, test_user):
    uid = test_user['id']
    conv_id = _make_conversion(app, uid)
    hl_a = _make_highlight(app, conv_id, exact='a')
    hl_b = _make_highlight(app, conv_id, exact='b')
    card_a = _make_card(app, uid, highlight_id=hl_a)
    card_b = _make_card(app, uid, highlight_id=hl_b)

    with app.app_context():
        db.session.delete(db.session.get(Highlight, hl_a))
        db.session.commit()

    with app.app_context():
        assert db.session.get(Card, card_a).highlight_id is None
        assert db.session.get(Card, card_b).highlight_id == hl_b   # untouched


# --- global recent reader ----------------------------------------------------

def test_recent_global_across_docs_sorted_desc(app, authenticated_client, test_user):
    uid = test_user['id']
    c1 = _make_conversion(app, uid, title='Doc One')
    c2 = _make_conversion(app, uid, title='Doc Two')
    _make_highlight(app, c1, exact='old', created_at=datetime(2026, 6, 1, 9, 0))
    _make_highlight(app, c2, exact='new', created_at=datetime(2026, 6, 10, 9, 0),
                    tags=['Wichtig'], user_id=uid)

    resp = authenticated_client.get('/api/highlights/recent')
    assert resp.status_code == 200
    body = resp.get_json()
    # newest first, across BOTH docs (the per-doc API can't do this)
    assert [h['exact'] for h in body] == ['new', 'old']
    first = body[0]
    assert first['conversion_id'] == c2
    assert first['title'] == 'Doc Two'
    assert first['tags'][0]['name'] == 'wichtig'
    assert set(first.keys()) == {'id', 'exact', 'note', 'tags', 'created_at',
                                 'conversion_id', 'title'}


def test_recent_since_filter(app, authenticated_client, test_user):
    uid = test_user['id']
    c = _make_conversion(app, uid)
    _make_highlight(app, c, exact='before', created_at=datetime(2026, 6, 1, 9, 0))
    _make_highlight(app, c, exact='after', created_at=datetime(2026, 6, 10, 9, 0))

    resp = authenticated_client.get('/api/highlights/recent?since=2026-06-05T00:00:00')
    assert resp.status_code == 200
    assert [h['exact'] for h in resp.get_json()] == ['after']


def test_recent_owner_scoped(app, authenticated_client, test_user):
    uid = test_user['id']
    with app.app_context():
        other = User(username='mallory')
        other.set_password('password1234')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    c_mine = _make_conversion(app, uid, title='Mine')
    c_other = _make_conversion(app, other_id, title='Theirs')
    _make_highlight(app, c_mine, exact='mine')
    _make_highlight(app, c_other, exact='theirs')

    resp = authenticated_client.get('/api/highlights/recent')
    assert [h['exact'] for h in resp.get_json()] == ['mine']


def test_recent_limit_applies_and_garbage_falls_back(app, authenticated_client, test_user):
    uid = test_user['id']
    c = _make_conversion(app, uid)
    for i in range(3):
        _make_highlight(app, c, exact=f'h{i}',
                        created_at=datetime(2026, 6, 1 + i, 9, 0))

    assert len(authenticated_client.get('/api/highlights/recent?limit=2').get_json()) == 2
    # garbage limit -> default (100), so all 3 come back
    assert len(authenticated_client.get('/api/highlights/recent?limit=abc').get_json()) == 3


def test_recent_requires_login(client):
    resp = client.get('/api/highlights/recent')
    assert resp.status_code in (302, 401)


# --- helper unit tests (cap without seeding 500 rows; tz normalisation) ------

def test_clamp_limit_helper():
    from app_pkg.cards import _clamp_limit, RECENT_DEFAULT_LIMIT, RECENT_MAX_LIMIT
    assert _clamp_limit('50', RECENT_DEFAULT_LIMIT, RECENT_MAX_LIMIT) == 50
    assert _clamp_limit('9999', RECENT_DEFAULT_LIMIT, RECENT_MAX_LIMIT) == RECENT_MAX_LIMIT
    assert _clamp_limit(None, RECENT_DEFAULT_LIMIT, RECENT_MAX_LIMIT) == RECENT_DEFAULT_LIMIT
    assert _clamp_limit('-3', RECENT_DEFAULT_LIMIT, RECENT_MAX_LIMIT) == RECENT_DEFAULT_LIMIT
    assert _clamp_limit('abc', RECENT_DEFAULT_LIMIT, RECENT_MAX_LIMIT) == RECENT_DEFAULT_LIMIT


def test_parse_since_helper():
    from app_pkg.cards import _parse_since
    assert _parse_since('2026-06-05T00:00:00') == datetime(2026, 6, 5, 0, 0, 0)
    # aware input is converted to UTC then stripped to naive
    assert _parse_since('2026-06-05T02:00:00+02:00') == datetime(2026, 6, 5, 0, 0, 0)
    assert _parse_since('garbage') is None
    assert _parse_since('') is None
    assert _parse_since(None) is None
