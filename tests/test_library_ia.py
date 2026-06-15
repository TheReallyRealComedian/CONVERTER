"""Flat four-place IA (R2-H) characterization tests.

R2-H collapses the three overlapping axes (lifecycle_status / queue_position /
is_favorite) onto ONE flat axis of four mutually-exclusive places —
Inbox · Lese-Liste · Bibliothek · Archiv — *derived* from the existing columns
with no schema touch. This file locks in:

- The derivation precedence (archive > queued > inbox > neutral shelf) behind
  the four ?view tabs, including the precedence collisions (queued+inbox →
  Lese-Liste; archived+queued → Archiv) and the bibliothek default.
- POST /api/conversions/<id>/place is the one move-action: it sets the
  (lifecycle_status, queue_position) combo per place, holds exclusivity
  (Archiv/Inbox/Bibliothek null the queue, Lese-Liste appends), is idempotent
  for an already-listed item, owner-scoped (404), and 400s an unknown place.
- Search is the global finder: it spans every non-archive place and ignores the
  per-tab place filter; the Bibliothek tab *without* a query is the neutral
  shelf only.
- inbox_count (the tab badge) ships with every view and counts the Inbox place
  (status inbox AND not queued), independent of the active filters.
- available_tags rows carry a usage count over the conversion_tags junction,
  ordered count-desc with alphabetical tie-break (top-N-ready for the chip row).
"""
from models import Conversion, Tag, User, db


def _make_conversion(app, user_id, **overrides):
    payload = dict(
        user_id=user_id,
        conversion_type='markdown_input',
        title='Sample Title',
        content='Sample content body.',
    )
    payload.update(overrides)
    with app.app_context():
        c = Conversion(**payload)
        db.session.add(c)
        db.session.commit()
        return c.id


def _row(app, cid):
    """(lifecycle_status, queue_position) tuple straight from the row."""
    with app.app_context():
        c = Conversion.query.get(cid)
        return c.lifecycle_status, c.queue_position


def _set_place(client, cid, place):
    return client.post(f'/api/conversions/{cid}/place', json={'place': place})


# --- the four views (derivation precedence) ---

def test_view_inbox_shows_only_inbox_place(app, authenticated_client, test_user):
    inbox = _make_conversion(app, test_user['id'], title='InboxDoc',
                             lifecycle_status='inbox')
    _make_conversion(app, test_user['id'], title='ShelfDoc', lifecycle_status='later')
    _make_conversion(app, test_user['id'], title='ArchivedDoc', lifecycle_status='archive')
    _make_conversion(app, test_user['id'], title='QueuedInboxDoc',
                     lifecycle_status='inbox', queue_position=1.0)

    html = authenticated_client.get('/library?view=inbox').data.decode()
    assert 'InboxDoc' in html
    assert 'ShelfDoc' not in html          # later + not queued = Bibliothek
    assert 'ArchivedDoc' not in html
    assert 'QueuedInboxDoc' not in html    # queued = Lese-Liste (precedence)
    _ = inbox


def test_view_bibliothek_shows_neutral_shelf_only(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='ShelfDoc', lifecycle_status='later')
    _make_conversion(app, test_user['id'], title='InboxDoc', lifecycle_status='inbox')
    _make_conversion(app, test_user['id'], title='QueuedDoc',
                     lifecycle_status='later', queue_position=1.0)
    _make_conversion(app, test_user['id'], title='ArchivedDoc', lifecycle_status='archive')

    # No view param → the bare /library lands on the Bibliothek place.
    html = authenticated_client.get('/library').data.decode()
    assert 'ShelfDoc' in html
    assert 'InboxDoc' not in html
    assert 'QueuedDoc' not in html         # queued = Lese-Liste (precedence)
    assert 'ArchivedDoc' not in html


def test_view_leseliste_shows_queued_non_archive_ordered(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], title='AlphaDoc',
                         lifecycle_status='later', queue_position=2.0)
    b = _make_conversion(app, test_user['id'], title='BetaDoc',
                         lifecycle_status='later', queue_position=1.0)
    _make_conversion(app, test_user['id'], title='ShelfDoc', lifecycle_status='later')
    _make_conversion(app, test_user['id'], title='ArchivedQueuedDoc',
                     lifecycle_status='archive', queue_position=3.0)

    html = authenticated_client.get('/library?view=leseliste').data.decode()
    assert 'ShelfDoc' not in html              # not queued
    assert 'ArchivedQueuedDoc' not in html     # archive beats the queue
    # Ordered by queue_position asc: Beta (1.0) before Alpha (2.0).
    assert html.index('BetaDoc') < html.index('AlphaDoc')
    _ = (a, b)


def test_view_archiv_shows_only_archived(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='ArchivedDoc', lifecycle_status='archive')
    _make_conversion(app, test_user['id'], title='InboxDoc', lifecycle_status='inbox')
    _make_conversion(app, test_user['id'], title='ShelfDoc', lifecycle_status='later')

    html = authenticated_client.get('/library?view=archiv').data.decode()
    assert 'ArchivedDoc' in html
    assert 'InboxDoc' not in html
    assert 'ShelfDoc' not in html


def test_precedence_queued_inbox_resolves_to_leseliste(app, authenticated_client, test_user):
    # An inbox row that is also queued is Lese-Liste, never Inbox (queued beats
    # inbox in the precedence order).
    _make_conversion(app, test_user['id'], title='QueuedInboxDoc',
                     lifecycle_status='inbox', queue_position=1.0)
    assert 'QueuedInboxDoc' not in authenticated_client.get('/library?view=inbox').data.decode()
    assert 'QueuedInboxDoc' in authenticated_client.get('/library?view=leseliste').data.decode()


def test_precedence_archive_beats_queue(app, authenticated_client, test_user):
    # An archived row that still carries a queue_position is Archiv, never
    # Lese-Liste (archive is the top of the precedence order).
    _make_conversion(app, test_user['id'], title='ArchivedQueuedDoc',
                     lifecycle_status='archive', queue_position=1.0)
    assert 'ArchivedQueuedDoc' not in authenticated_client.get('/library?view=leseliste').data.decode()
    assert 'ArchivedQueuedDoc' in authenticated_client.get('/library?view=archiv').data.decode()


def test_unknown_view_defaults_to_bibliothek(app, authenticated_client, test_user):
    # An unknown ?view (incl. the retired R2-D 'reading') falls back to the
    # neutral shelf, same as the bare landing view.
    _make_conversion(app, test_user['id'], title='ShelfDoc', lifecycle_status='later')
    _make_conversion(app, test_user['id'], title='InboxDoc', lifecycle_status='inbox')

    html = authenticated_client.get('/library?view=reading').data.decode()
    assert 'ShelfDoc' in html
    assert 'InboxDoc' not in html


# --- POST /api/conversions/<id>/place (the one move-action) ---

def test_place_inbox_sets_inbox_and_nulls_queue(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], lifecycle_status='later', queue_position=5.0)
    r = _set_place(authenticated_client, cid, 'inbox')
    assert r.status_code == 200
    assert _row(app, cid) == ('inbox', None)


def test_place_bibliothek_sets_later_and_nulls_queue(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], lifecycle_status='inbox', queue_position=5.0)
    r = _set_place(authenticated_client, cid, 'bibliothek')
    assert r.status_code == 200
    assert _row(app, cid) == ('later', None)


def test_place_archiv_archives_and_dequeues(app, authenticated_client, test_user):
    # R2-H decision: archiving takes the item off the reading list (supersedes
    # R2-D's "archiving does not dequeue") — the exclusive model requires it.
    cid = _make_conversion(app, test_user['id'], lifecycle_status='later', queue_position=5.0)
    r = _set_place(authenticated_client, cid, 'archiv')
    assert r.status_code == 200
    assert _row(app, cid) == ('archive', None)


def test_place_leseliste_appends_at_end_and_sets_later(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], lifecycle_status='inbox')
    b = _make_conversion(app, test_user['id'], lifecycle_status='inbox')
    assert _set_place(authenticated_client, a, 'leseliste').status_code == 200
    assert _set_place(authenticated_client, b, 'leseliste').status_code == 200
    assert _row(app, a) == ('later', 1.0)
    assert _row(app, b) == ('later', 2.0)   # appended after a


def test_place_leseliste_idempotent_keeps_slot(app, authenticated_client, test_user):
    a = _make_conversion(app, test_user['id'], lifecycle_status='inbox')
    b = _make_conversion(app, test_user['id'], lifecycle_status='inbox')
    _set_place(authenticated_client, a, 'leseliste')   # 1.0
    _set_place(authenticated_client, b, 'leseliste')   # 2.0
    # Re-placing a row already on the list keeps its slot (no re-append/bump).
    r = _set_place(authenticated_client, a, 'leseliste')
    assert r.status_code == 200
    assert _row(app, a) == ('later', 1.0)
    assert _row(app, b) == ('later', 2.0)


def test_place_invalid_400(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], lifecycle_status='inbox')
    r = _set_place(authenticated_client, cid, 'spaeter')
    assert r.status_code == 400
    assert 'Ort' in r.get_json()['error']
    # The rejected move must not have touched the row.
    assert _row(app, cid) == ('inbox', None)


def test_place_missing_place_400(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], lifecycle_status='inbox')
    r = authenticated_client.post(f'/api/conversions/{cid}/place', json={})
    assert r.status_code == 400


def test_place_non_dict_body_400(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], lifecycle_status='inbox')
    r = authenticated_client.post(f'/api/conversions/{cid}/place', json=['nope'])
    assert r.status_code == 400
    assert 'JSON-Objekt' in r.get_json()['error']


def test_place_other_users_conversion_404(app, authenticated_client, test_user):
    with app.app_context():
        bob = User(username='bob')
        bob.set_password('hunter2hunter2')
        db.session.add(bob)
        db.session.commit()
        bob_id = bob.id
    cid = _make_conversion(app, bob_id, lifecycle_status='inbox')
    r = _set_place(authenticated_client, cid, 'archiv')
    assert r.status_code == 404
    assert _row(app, cid) == ('inbox', None)  # untouched


# --- search: the global finder (spans non-archive, ignores the place tab) ---

def test_search_spans_all_non_archive_places(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='NeedleInbox',
                     content='find the needle', lifecycle_status='inbox')
    _make_conversion(app, test_user['id'], title='NeedleLese', content='find the needle',
                     lifecycle_status='later', queue_position=1.0)
    _make_conversion(app, test_user['id'], title='NeedleShelf',
                     content='find the needle', lifecycle_status='later')
    _make_conversion(app, test_user['id'], title='NeedleArchive',
                     content='find the needle', lifecycle_status='archive')

    html = authenticated_client.get('/library?search=needle').data.decode()
    assert 'NeedleInbox' in html
    assert 'NeedleLese' in html
    assert 'NeedleShelf' in html
    assert 'NeedleArchive' not in html     # archive is out of the global finder


def test_search_on_bibliothek_tab_still_spans_non_archive(app, authenticated_client, test_user):
    # The Bibliothek tab WITHOUT a query is the neutral shelf only; WITH a query
    # it spans non-archive like every other tab (search overrides the place).
    _make_conversion(app, test_user['id'], title='ShelfNeedle',
                     content='needle', lifecycle_status='later')
    _make_conversion(app, test_user['id'], title='InboxNeedle',
                     content='needle', lifecycle_status='inbox')

    no_query = authenticated_client.get('/library?view=bibliothek').data.decode()
    assert 'ShelfNeedle' in no_query
    assert 'InboxNeedle' not in no_query   # neutral shelf only

    with_query = authenticated_client.get('/library?view=bibliothek&search=needle').data.decode()
    assert 'ShelfNeedle' in with_query
    assert 'InboxNeedle' in with_query     # search spans into the inbox


# --- inbox_count badge ---

def test_inbox_count_is_inbox_place_and_shipped_every_view(
        app, authenticated_client, test_user, captured_templates):
    _make_conversion(app, test_user['id'], title='InboxOne', lifecycle_status='inbox')
    _make_conversion(app, test_user['id'], title='InboxTwo', lifecycle_status='inbox')
    _make_conversion(app, test_user['id'], title='ShelfDoc', lifecycle_status='later')
    _make_conversion(app, test_user['id'], title='QueuedInbox',
                     lifecycle_status='inbox', queue_position=1.0)  # queued → not Inbox

    # The tab bar is always visible, so every view ships the count — and it
    # counts the Inbox place only (queued-inbox excluded), independent of view.
    for url in ('/library', '/library?view=inbox', '/library?view=leseliste',
                '/library?view=archiv'):
        captured_templates.clear()
        resp = authenticated_client.get(url)
        assert resp.status_code == 200
        _, ctx = captured_templates[-1]
        assert ctx['inbox_count'] == 2, url


# --- available_tags usage counts (unchanged from R2-E) ---

def test_available_tags_count_sorted_with_alpha_tiebreak(app, authenticated_client,
                                                         test_user, captured_templates):
    a = _make_conversion(app, test_user['id'], title='DocA')
    b = _make_conversion(app, test_user['id'], title='DocB')
    # alpha hangs on two conversions; beta + zeta on one each (tie -> name).
    for cid, names in ((a, ('alpha', 'zeta')), (b, ('alpha', 'beta'))):
        for name in names:
            r = authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': name})
            assert r.status_code in (200, 201)

    authenticated_client.get('/library')
    _, ctx = captured_templates[-1]
    assert [(t.name, t.count) for t in ctx['available_tags']] == [
        ('alpha', 2), ('beta', 1), ('zeta', 1),
    ]


# --- view preserved across pagination ---

def test_view_preserved_across_pagination(app, authenticated_client, test_user):
    # 21 inbox rows at default per_page=20 -> 2 pages. A page-2 pagination link
    # must keep the active view so the next page stays on the Inbox tab.
    for i in range(21):
        _make_conversion(app, test_user['id'], title=f'Inbox-{i:02d}',
                         lifecycle_status='inbox')
    html = authenticated_client.get('/library?view=inbox').data.decode()
    import re
    page2_links = re.findall(r'href="[^"]*page=2[^"]*"', html)
    assert page2_links, 'expected a page-2 pagination link'
    assert any('view=inbox' in link for link in page2_links)
