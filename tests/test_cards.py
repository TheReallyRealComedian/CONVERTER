"""R4-LEARN Phase 1 — recall-layer schema, the Highlight before_delete event,
and the global /api/highlights/recent reader.

The before_delete tests are the rot→grün pair for the locked must-fix: SQLite
runs without PRAGMA foreign_keys=ON, so the ON DELETE SET NULL declared on
card.highlight_id is inert — an ORM before_delete event is the real mechanic.
Without it, deleting a Highlight leaves a dangling highlight_id on the card.
"""
from datetime import datetime, timedelta, timezone

from models import Card, Conversion, Highlight, Review, Tag, User, card_tags, db


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


def _make_card(app, user_id, highlight_id=None, ctype='atomic', with_review=True,
               due=None, state='ok'):
    with app.app_context():
        card = Card(user_id=user_id, highlight_id=highlight_id, type=ctype,
                    front='Q', back='A', state=state)
        if with_review:
            card.review = Review(due=due or datetime.now(timezone.utc))
        db.session.add(card)
        db.session.commit()
        return card.id


def _make_other_user(app, username='mallory'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


CARD_TOKEN = 'r4-test-card-token-9b2e'
CARDS_URL = '/api/cards'


def _card_auth(token=CARD_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _atomic_payload(**ov):
    p = {'type': 'atomic', 'front': 'Was ist X?', 'back': 'X ist Y.'}
    p.update(ov)
    return p


def _generative_payload(**ov):
    p = {'type': 'generative', 'prompt': 'Erkläre X laut.'}
    p.update(ov)
    return p


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


# =============================================================================
# Phase 2 — card write API (token) + card/review reads (session)
# =============================================================================

# --- write auth: 503 fail-closed / 401 missing+wrong / 201 real -------------

def test_card_create_fail_closed_without_token(app, client, test_user, monkeypatch):
    monkeypatch.delenv('CARD_TOKEN', raising=False)
    # Unset -> 503 even with a Bearer present (config check precedes auth).
    assert client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload()).status_code == 503
    # Empty string is "unset" too.
    monkeypatch.setenv('CARD_TOKEN', '')
    assert client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload()).status_code == 503
    with app.app_context():
        assert Card.query.count() == 0


def test_card_create_401_missing_and_wrong_token(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(CARDS_URL, json=_atomic_payload()).status_code == 401
    assert client.post(CARDS_URL, headers=_card_auth('the-wrong-token'),
                       json=_atomic_payload()).status_code == 401
    with app.app_context():
        assert Card.query.count() == 0


def test_card_create_201_persists_card_and_review(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    resp = client.post(CARDS_URL, headers=_card_auth(),
                       json=_atomic_payload(tags=['Biologie', 'Wichtig'],
                                            note='merken', source_doc_title='Doc',
                                            source_snapshot='quote in context'))
    assert resp.status_code == 201
    body = resp.get_json()
    assert body['type'] == 'atomic'
    assert body['state'] == 'ok'              # default
    assert body['created_by'] == 'agent'      # default
    assert body['source_doc_title'] == 'Doc'
    assert sorted(t['name'] for t in body['tags']) == ['biologie', 'wichtig']
    # locked decision: the Review row is created in the FSRS-"new" state
    assert body['review'] is not None
    assert body['review']['due'] is not None
    assert body['review']['reps'] == 0
    assert body['review']['lapses'] == 0
    with app.app_context():
        card = Card.query.filter_by(user_id=uid).one()
        assert card.user_id == uid            # stamped from the resolver
        assert card.review is not None
        assert card.review.due is not None
        assert sorted(t.name for t in card.tags) == ['biologie', 'wichtig']  # via card_tags


def test_card_create_targets_ingest_user(app, client, test_user, monkeypatch):
    # The card write resolves the target user via the SAME INGEST_USER hook.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    alice = test_user['id']
    bob = _make_other_user(app, 'bob')
    monkeypatch.setenv('INGEST_USER', 'bob')
    resp = client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload())
    assert resp.status_code == 201
    with app.app_context():
        assert Card.query.filter_by(user_id=bob).count() == 1
        assert Card.query.filter_by(user_id=alice).count() == 0


# --- CSRF: exempt only the two write views (proven under enforced CSRF) ------

def test_card_create_csrf_exempt_under_enforced_csrf(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    # conftest disables CSRF globally; flip it back ON. A Bearer-only POST has no
    # CSRF token and must still succeed because the view is exempt.
    monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)
    assert client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload()).status_code == 201


def test_only_card_write_views_are_csrf_exempt(app):
    csrf = app.extensions['csrf']
    assert 'app_pkg.cards.api_create_card' in csrf._exempt_views
    assert 'app_pkg.cards.api_patch_card' in csrf._exempt_views
    # the token-authed highlight write joins the exempt set (session-less write)
    assert 'app_pkg.cards.api_annotate_highlight' in csrf._exempt_views
    # session-authed reads are NOT exempt
    assert 'app_pkg.cards.api_list_cards' not in csrf._exempt_views
    assert 'app_pkg.cards.api_review_state' not in csrf._exempt_views
    # the session-authed user DELETE stays under CSRF too (state-changing)
    assert 'app_pkg.cards.api_delete_card' not in csrf._exempt_views


# --- per-type validation: 400 ------------------------------------------------

def test_card_create_atomic_cloze_is_valid(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    resp = client.post(CARDS_URL, headers=_card_auth(),
                       json={'type': 'atomic', 'cloze_text': 'Die {{Mitochondrien}} sind ...'})
    assert resp.status_code == 201


def test_card_create_atomic_missing_fields_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    # neither (front AND back) nor cloze_text
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json={'type': 'atomic', 'front': 'only front'}).status_code == 400
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json={'type': 'atomic'}).status_code == 400


def test_card_create_generative_valid_and_invalid(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json=_generative_payload()).status_code == 201
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json={'type': 'generative'}).status_code == 400


def test_card_create_invalid_type_and_body_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json={'type': 'bogus', 'front': 'a', 'back': 'b'}).status_code == 400
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json=['not', 'a', 'dict']).status_code == 400


# --- highlight_id ownership --------------------------------------------------

def test_card_create_highlight_ownership(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    # own highlight -> 201
    conv = _make_conversion(app, uid)
    hl = _make_highlight(app, conv, exact='q')
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json=_atomic_payload(highlight_id=hl)).status_code == 201
    # foreign highlight -> 400 (target user is first()=alice)
    other_id = _make_other_user(app)
    conv_o = _make_conversion(app, other_id)
    hl_o = _make_highlight(app, conv_o, exact='foreign')
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json=_atomic_payload(highlight_id=hl_o)).status_code == 400
    # nonexistent highlight -> 400
    assert client.post(CARDS_URL, headers=_card_auth(),
                       json=_atomic_payload(highlight_id=999999)).status_code == 400


def test_card_create_reuses_existing_tag(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    with app.app_context():
        Tag.get_or_create(uid, 'ki')
        db.session.commit()
        existing = Tag.query.filter_by(user_id=uid, name='ki').first().id
    resp = client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload(tags=['KI', 'neu']))
    assert resp.status_code == 201
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='ki').count() == 1
        card = Card.query.filter_by(user_id=uid).one()
        ki = next(t for t in card.tags if t.name == 'ki')
        assert ki.id == existing


# --- PATCH -------------------------------------------------------------------

def test_card_patch_refines_and_toggles_state(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload()).get_json()['id']
    resp = client.patch(f'/api/cards/{cid}', headers=_card_auth(),
                        json={'back': 'Neue Antwort', 'state': 'wackelt', 'tags': ['neu']})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['back'] == 'Neue Antwort'
    assert body['state'] == 'wackelt'
    assert [t['name'] for t in body['tags']] == ['neu']
    # reset wackelt -> ok
    resp2 = client.patch(f'/api/cards/{cid}', headers=_card_auth(), json={'state': 'ok'})
    assert resp2.get_json()['state'] == 'ok'


def test_card_patch_requires_token_and_rejects_bad_state(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = client.post(CARDS_URL, headers=_card_auth(), json=_atomic_payload()).get_json()['id']
    # no token -> 401
    assert client.patch(f'/api/cards/{cid}', json={'state': 'ok'}).status_code == 401
    # bad state -> 400
    assert client.patch(f'/api/cards/{cid}', headers=_card_auth(),
                        json={'state': 'bogus'}).status_code == 400


def test_card_patch_foreign_card_404(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    other_id = _make_other_user(app)
    foreign = _make_card(app, other_id)   # owned by mallory; target=first()=alice
    assert client.patch(f'/api/cards/{foreign}', headers=_card_auth(),
                        json={'state': 'wackelt'}).status_code == 404


# --- reads: owner-scoped, filtered, paginated --------------------------------

def test_cards_list_owner_scoped_with_filters(app, authenticated_client, test_user):
    uid = test_user['id']
    conv = _make_conversion(app, uid)
    hl = _make_highlight(app, conv, exact='q')
    c1 = _make_card(app, uid, highlight_id=hl, state='ok')
    c2 = _make_card(app, uid, state='wackelt')
    _make_card(app, _make_other_user(app), state='ok')   # foreign — must not appear

    allc = authenticated_client.get(CARDS_URL).get_json()
    assert {c['id'] for c in allc} == {c1, c2}
    assert 'back' not in allc[0]                          # slim summary, no answer side
    # state filter
    assert [c['id'] for c in authenticated_client.get(CARDS_URL + '?state=wackelt').get_json()] == [c2]
    # highlight_id filter
    assert [c['id'] for c in authenticated_client.get(f'{CARDS_URL}?highlight_id={hl}').get_json()] == [c1]


def test_cards_list_limit_offset(app, authenticated_client, test_user):
    uid = test_user['id']
    for _ in range(3):
        _make_card(app, uid)
    assert len(authenticated_client.get(CARDS_URL + '?limit=2&offset=0').get_json()) == 2
    assert len(authenticated_client.get(CARDS_URL + '?limit=2&offset=2').get_json()) == 1


def test_card_get_full_and_foreign_404(app, authenticated_client, test_user):
    uid = test_user['id']
    cid = _make_card(app, uid)
    full = authenticated_client.get(f'/api/cards/{cid}').get_json()
    assert full['id'] == cid
    assert 'back' in full                 # full to_dict carries the answer side
    assert full['review'] is not None
    foreign = _make_card(app, _make_other_user(app))
    assert authenticated_client.get(f'/api/cards/{foreign}').status_code == 404


def test_review_state_due_now_vs_future(app, authenticated_client, test_user):
    uid = test_user['id']
    now = datetime.now(timezone.utc)
    c_due = _make_card(app, uid, due=now - timedelta(days=1))
    c_future = _make_card(app, uid, due=now + timedelta(days=1))
    body = authenticated_client.get('/api/review-state').get_json()
    due_ids = [c['id'] for c in body['due_cards']]
    assert c_due in due_ids
    assert c_future not in due_ids
    assert body['due_count'] == 1
    assert body['total_count'] == 2
    # due_cards are full cards (the Phase-4 review queue renders without refetch)
    assert 'back' in body['due_cards'][0]


def test_card_reads_require_login(client):
    assert client.get(CARDS_URL).status_code in (302, 401)
    assert client.get(CARDS_URL + '/1').status_code in (302, 401)
    assert client.get('/api/review-state').status_code in (302, 401)


# =============================================================================
# Phase 3 — rate endpoint (session, FSRS/SM-2)
# =============================================================================

def test_review_endpoint_updates_review_row(app, authenticated_client, test_user):
    uid = test_user['id']
    cid = _make_card(app, uid, ctype='atomic')
    resp = authenticated_client.post(f'/api/cards/{cid}/review', json={'rating': 'good'})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['id'] == cid
    review = body['review']
    assert review['reps'] == 1
    assert review['lapses'] == 0
    assert review['due'] is not None
    assert review['last_reviewed'] is not None
    assert review['rating_history'][-1]['rating'] == 'good'   # history appended
    with app.app_context():                                   # persisted
        r = db.session.get(Card, cid).review
        assert r.reps == 1
        assert r.last_reviewed is not None


def test_review_endpoint_again_is_a_lapse(app, authenticated_client, test_user):
    uid = test_user['id']
    cid = _make_card(app, uid, ctype='atomic')
    authenticated_client.post(f'/api/cards/{cid}/review', json={'rating': 'good'})
    resp = authenticated_client.post(f'/api/cards/{cid}/review', json={'rating': 'again'})
    review = resp.get_json()['review']
    assert review['reps'] == 2
    assert review['lapses'] == 1


def test_review_endpoint_generative_weak_sets_wackelt(app, authenticated_client, test_user):
    uid = test_user['id']
    cid = _make_card(app, uid, ctype='generative')
    resp = authenticated_client.post(f'/api/cards/{cid}/review', json={'rating': 'hard'})
    assert resp.get_json()['state'] == 'wackelt'


def test_review_endpoint_atomic_weak_stays_ok(app, authenticated_client, test_user):
    uid = test_user['id']
    cid = _make_card(app, uid, ctype='atomic')
    resp = authenticated_client.post(f'/api/cards/{cid}/review', json={'rating': 'again'})
    assert resp.get_json()['state'] == 'ok'   # only generative flips to wackelt


def test_review_endpoint_bad_rating_400(app, authenticated_client, test_user):
    uid = test_user['id']
    cid = _make_card(app, uid)
    assert authenticated_client.post(f'/api/cards/{cid}/review',
                                     json={'rating': 'bogus'}).status_code == 400
    assert authenticated_client.post(f'/api/cards/{cid}/review', json={}).status_code == 400


def test_review_endpoint_requires_login_not_token(app, client, test_user, monkeypatch):
    # The rate endpoint is @login_required — a CARD_TOKEN bearer (agent) must NOT
    # authorize it. Proves the auth-split: user rates, agent writes.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_card(app, test_user['id'])
    resp = client.post(f'/api/cards/{cid}/review', headers=_card_auth(), json={'rating': 'good'})
    assert resp.status_code in (302, 401)   # NOT 200


def test_review_endpoint_foreign_card_404(app, authenticated_client, test_user):
    foreign = _make_card(app, _make_other_user(app))
    assert authenticated_client.post(f'/api/cards/{foreign}/review',
                                     json={'rating': 'good'}).status_code == 404


def test_review_endpoint_engine_swap_to_sm2(app, authenticated_client, test_user, monkeypatch):
    # Engine swap via config — the rate endpoint produces a plausible due either way.
    monkeypatch.setenv('SCHEDULER_ENGINE', 'sm2')
    cid = _make_card(app, test_user['id'])
    resp = authenticated_client.post(f'/api/cards/{cid}/review', json={'rating': 'good'})
    assert resp.status_code == 200
    review = resp.get_json()['review']
    assert review['due'] is not None
    assert review['reps'] == 1


# =============================================================================
# Phase 4 — review page route + session annotate path (Vertiefen + note)
# =============================================================================

def test_review_page_requires_login(client):
    assert client.get('/review').status_code in (302, 401)


def test_review_page_renders_for_user(authenticated_client):
    resp = authenticated_client.get('/review')
    assert resp.status_code == 200
    assert b'review-card' in resp.data          # the shell rendered


def test_annotate_sets_wackelt(app, authenticated_client, test_user):
    cid = _make_card(app, test_user['id'], ctype='atomic')
    resp = authenticated_client.post(f'/api/cards/{cid}/annotate', json={'state': 'wackelt'})
    assert resp.status_code == 200
    assert resp.get_json()['state'] == 'wackelt'
    with app.app_context():
        assert db.session.get(Card, cid).state == 'wackelt'


def test_annotate_sets_and_clears_note(app, authenticated_client, test_user):
    cid = _make_card(app, test_user['id'])
    assert authenticated_client.post(f'/api/cards/{cid}/annotate',
                                     json={'note': 'merken'}).get_json()['note'] == 'merken'
    # empty string is a delete-intent → null
    assert authenticated_client.post(f'/api/cards/{cid}/annotate',
                                     json={'note': ''}).get_json()['note'] is None


def test_annotate_rejects_bad_state_and_empty_body(app, authenticated_client, test_user):
    cid = _make_card(app, test_user['id'])
    assert authenticated_client.post(f'/api/cards/{cid}/annotate',
                                     json={'state': 'bogus'}).status_code == 400
    assert authenticated_client.post(f'/api/cards/{cid}/annotate', json={}).status_code == 400


def test_annotate_foreign_card_404(app, authenticated_client, test_user):
    foreign = _make_card(app, _make_other_user(app))
    assert authenticated_client.post(f'/api/cards/{foreign}/annotate',
                                     json={'state': 'wackelt'}).status_code == 404


def test_annotate_requires_login_not_token(app, client, test_user, monkeypatch):
    # Session-only — a CARD_TOKEN bearer must NOT authorize the user's annotate path.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_card(app, test_user['id'])
    resp = client.post(f'/api/cards/{cid}/annotate', headers=_card_auth(), json={'state': 'wackelt'})
    assert resp.status_code in (302, 401)


# =============================================================================
# Phase 5 — DELETE /api/cards/<id> (session; the USER deletes their own card)
# =============================================================================

def test_card_delete_removes_card_review_and_junction(app, authenticated_client, test_user):
    # The cascade proof: a card carrying BOTH a Review row and a card_tags
    # junction — the two rows a raw DELETE FROM card would orphan (no FK pragma).
    # The ORM delete must take both, while the shared Tag itself survives.
    uid = test_user['id']
    with app.app_context():
        card = Card(user_id=uid, type='atomic', front='Q', back='A')
        card.review = Review(due=datetime.now(timezone.utc))
        card.tags.append(Tag.get_or_create(uid, 'biologie'))
        db.session.add(card)
        db.session.commit()
        cid, tag_id = card.id, card.tags[0].id
    # sanity: the junction row exists before the delete, so the assert means something
    with app.app_context():
        rows = db.session.execute(
            card_tags.select().where(card_tags.c.card_id == cid)).fetchall()
        assert len(rows) == 1

    resp = authenticated_client.delete(f'/api/cards/{cid}')
    assert resp.status_code == 200
    assert resp.get_json() == {'success': True}

    with app.app_context():
        assert db.session.get(Card, cid) is None                   # card gone
        assert Review.query.filter_by(card_id=cid).count() == 0    # review cascaded, no orphan
        rows = db.session.execute(
            card_tags.select().where(card_tags.c.card_id == cid)).fetchall()
        assert rows == []                                          # junction swept, no orphan
        assert db.session.get(Tag, tag_id) is not None             # shared Tag survives


def test_card_delete_foreign_card_404(app, authenticated_client, test_user):
    # Another user's card → 404 (never leak existence), and it stays put.
    foreign = _make_card(app, _make_other_user(app))
    assert authenticated_client.delete(f'/api/cards/{foreign}').status_code == 404
    with app.app_context():
        assert db.session.get(Card, foreign) is not None


def test_card_delete_nonexistent_404(app, authenticated_client, test_user):
    assert authenticated_client.delete('/api/cards/999999').status_code == 404


def test_card_delete_requires_login_not_token(app, client, test_user, monkeypatch):
    # Session-only — a CARD_TOKEN bearer (agent) must NOT authorize delete; the
    # agent writes, the user deletes in the UI. Proves the auth-split.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    cid = _make_card(app, test_user['id'])
    resp = client.delete(f'/api/cards/{cid}', headers=_card_auth())
    assert resp.status_code in (302, 401)
    with app.app_context():
        assert db.session.get(Card, cid) is not None   # not deleted


# =============================================================================
# Phase 6 — PATCH /api/highlights/<id>/annotate (token; the AGENT writes back
# tags/note onto an existing highlight for persistent bucket-tagging)
# =============================================================================

HL_ANNOTATE = '/api/highlights/{}/annotate'


def _make_owned_highlight(app, uid, exact='quote', note=None):
    """A highlight on the target user's own conversion. The token write target
    is first()=alice (unless INGEST_USER is set), so highlights must hang off
    alice's conversion to be 'owned'."""
    conv = _make_conversion(app, uid)
    return _make_highlight(app, conv, exact=exact, note=note)


# --- write auth: 503 fail-closed / 401 missing+wrong ------------------------

def test_highlight_annotate_fail_closed_without_token(app, client, test_user, monkeypatch):
    monkeypatch.delenv('CARD_TOKEN', raising=False)
    hl = _make_owned_highlight(app, test_user['id'])
    # Unset -> 503 even with a Bearer present (config check precedes auth).
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'tags': ['ki']}).status_code == 503
    # Empty string is "unset" too.
    monkeypatch.setenv('CARD_TOKEN', '')
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'tags': ['ki']}).status_code == 503


def test_highlight_annotate_401_missing_and_wrong_token(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    hl = _make_owned_highlight(app, test_user['id'])
    assert client.patch(HL_ANNOTATE.format(hl), json={'tags': ['ki']}).status_code == 401
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth('the-wrong-token'),
                        json={'tags': ['ki']}).status_code == 401


# --- ownership: foreign + missing are 404 (path resource, not body ref) ------

def test_highlight_annotate_foreign_and_missing_404(app, client, test_user, monkeypatch):
    # The id is the addressed RESOURCE: a foreign/missing highlight is 404 (not
    # the 400 the card writes give a bad body highlight_id, not 403). No leak.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    other_id = _make_other_user(app)            # mallory; target=first()=alice
    conv_o = _make_conversion(app, other_id)
    hl_o = _make_highlight(app, conv_o, exact='foreign')
    assert client.patch(HL_ANNOTATE.format(hl_o), headers=_card_auth(),
                        json={'tags': ['ki']}).status_code == 404
    assert client.patch(HL_ANNOTATE.format(999999), headers=_card_auth(),
                        json={'tags': ['ki']}).status_code == 404


# --- tags: set / replace / clear --------------------------------------------

def test_highlight_annotate_sets_replaces_clears_tags(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    hl = _make_owned_highlight(app, uid)
    # set
    resp = client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'tags': ['Biologie', 'Wichtig']})
    assert resp.status_code == 200
    with app.app_context():
        assert sorted(t.name for t in db.session.get(Highlight, hl).tags) == ['biologie', 'wichtig']
    # full replacement
    client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(), json={'tags': ['neu']})
    with app.app_context():
        assert [t.name for t in db.session.get(Highlight, hl).tags] == ['neu']
    # [] clears all tags
    client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(), json={'tags': []})
    with app.app_context():
        assert db.session.get(Highlight, hl).tags == []


def test_highlight_annotate_tags_must_be_list_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    hl = _make_owned_highlight(app, test_user['id'])
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'tags': 'ki'}).status_code == 400


# --- note: set / clear ('' -> NULL) / type-guard ----------------------------

def test_highlight_annotate_sets_and_clears_note(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    hl = _make_owned_highlight(app, test_user['id'])
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'note': 'merken'}).get_json()['note'] == 'merken'
    # empty string is a delete-intent → NULL
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'note': ''}).get_json()['note'] is None
    # non-str, non-null → 400
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'note': 123}).status_code == 400


# --- normalisation + shared vocabulary --------------------------------------

def test_highlight_annotate_normalises_tags(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    hl = _make_owned_highlight(app, uid)
    resp = client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'tags': ['KI', 'ki', ' KI ']})
    assert resp.status_code == 200
    with app.app_context():
        # exactly one Tag row (name='ki'), and exactly one tag on the highlight
        assert Tag.query.filter_by(user_id=uid, name='ki').count() == 1
        assert [t.name for t in db.session.get(Highlight, hl).tags] == ['ki']


def test_highlight_annotate_reuses_card_tag_row(app, client, test_user, monkeypatch):
    # Shared vocabulary, no parallel system: a tag first created on a CARD is the
    # SAME row when later set on a highlight (same id, count stays 1).
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    card = client.post(CARDS_URL, headers=_card_auth(),
                       json=_atomic_payload(tags=['ki'])).get_json()
    card_tag_id = card['tags'][0]['id']
    hl = _make_owned_highlight(app, uid)
    resp = client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(), json={'tags': ['KI']})
    assert resp.get_json()['tags'][0]['id'] == card_tag_id
    with app.app_context():
        assert Tag.query.filter_by(user_id=uid, name='ki').count() == 1


# --- response shape + anchor immutability + empty body ----------------------

def test_highlight_annotate_response_carries_resolved_tags(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    hl = _make_owned_highlight(app, test_user['id'])
    resp = client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(), json={'tags': ['ki']})
    assert resp.status_code == 200
    tags = resp.get_json()['tags']
    assert tags[0]['name'] == 'ki'
    assert 'id' in tags[0]


def test_highlight_annotate_ignores_anchor_keys(app, client, test_user, monkeypatch):
    # exact/prefix/suffix in the body are ignored — the agent annotates, it never
    # moves a marker. The PATCH still succeeds via the tag; the anchors stay put.
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    uid = test_user['id']
    conv = _make_conversion(app, uid)
    hl = _make_highlight(app, conv, exact='original')
    with app.app_context():
        o = db.session.get(Highlight, hl)
        before = (o.exact, o.prefix, o.suffix)
    resp = client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={'tags': ['ki'], 'exact': 'HACKED',
                              'prefix': 'P', 'suffix': 'S'})
    assert resp.status_code == 200
    with app.app_context():
        after = db.session.get(Highlight, hl)
        assert (after.exact, after.prefix, after.suffix) == before
        assert after.exact == 'original'                       # not overwritten


def test_highlight_annotate_empty_body_and_non_dict_400(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    hl = _make_owned_highlight(app, test_user['id'])
    # neither tags nor note → 400
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json={}).status_code == 400
    # non-dict body → 400
    assert client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(),
                        json=['nope']).status_code == 400


# --- CSRF: exempt under enforced CSRF (Bearer-only, no CSRF token) -----------

def test_highlight_annotate_csrf_exempt_under_enforced_csrf(app, client, test_user, monkeypatch):
    monkeypatch.setenv('CARD_TOKEN', CARD_TOKEN)
    # conftest disables CSRF globally; flip it back ON. A Bearer-only PATCH has no
    # CSRF token and must still succeed because the view is exempt.
    monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)
    hl = _make_owned_highlight(app, test_user['id'])
    resp = client.patch(HL_ANNOTATE.format(hl), headers=_card_auth(), json={'tags': ['ki']})
    assert resp.status_code == 200   # NOT 400/403
