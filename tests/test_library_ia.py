"""Library-IA (R2-E "Readwise-3er") characterization tests.

Locks in:
- GET /library?view=inbox surfaces only untriaged rows: lifecycle_status =
  'inbox' AND queue_position IS NULL. Queueing an inbox item removes it from
  the inbox view without touching its status; de-queueing an item that is
  still 'inbox' drops it back into triage.
- inbox_count (the tab badge) is shipped with every view and counts the
  untriaged set globally, independent of the active filters.
- available_tags rows carry a usage count over the conversion_tags junction,
  ordered count-desc with alphabetical tie-break (top-N-ready for the chip
  row + typeahead).
"""
from models import Conversion, db


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


def _set_status(app, cid, status):
    with app.app_context():
        c = Conversion.query.get(cid)
        c.lifecycle_status = status
        db.session.commit()


def _add_to_queue(client, cid):
    return client.post(f'/api/conversions/{cid}/queue', json={'action': 'add'})


# --- GET /library?view=inbox ---

def test_view_inbox_shows_only_untriaged(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='UntriagedDoc')  # default: inbox
    later = _make_conversion(app, test_user['id'], title='LaterDoc')
    archived = _make_conversion(app, test_user['id'], title='ArchivedDoc')
    queued_inbox = _make_conversion(app, test_user['id'], title='QueuedInboxDoc')
    _set_status(app, later, 'later')
    _set_status(app, archived, 'archive')
    # Queueing triages: the item stays lifecycle_status='inbox' but leaves
    # the inbox view because it is on the reading list now.
    _add_to_queue(authenticated_client, queued_inbox)

    resp = authenticated_client.get('/library?view=inbox')
    assert resp.status_code == 200
    html = resp.data.decode()
    assert 'UntriagedDoc' in html
    assert 'LaterDoc' not in html
    assert 'ArchivedDoc' not in html
    assert 'QueuedInboxDoc' not in html


def test_view_inbox_dequeue_drops_item_back(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], title='TriageMeDoc')
    _add_to_queue(authenticated_client, cid)
    assert 'TriageMeDoc' not in authenticated_client.get('/library?view=inbox').data.decode()

    # De-queue while still status 'inbox' -> back into triage (by design:
    # un-queued + untriaged = inbox again).
    authenticated_client.post(f'/api/conversions/{cid}/queue', json={'action': 'remove'})
    assert 'TriageMeDoc' in authenticated_client.get('/library?view=inbox').data.decode()


def test_view_inbox_dequeued_later_item_stays_out(app, authenticated_client, test_user):
    # De-queueing only falls back into the inbox when the status still says
    # inbox — a triaged-to-later item stays out.
    cid = _make_conversion(app, test_user['id'], title='LaterQueuedDoc')
    _add_to_queue(authenticated_client, cid)
    _set_status(app, cid, 'later')
    authenticated_client.post(f'/api/conversions/{cid}/queue', json={'action': 'remove'})
    assert 'LaterQueuedDoc' not in authenticated_client.get('/library?view=inbox').data.decode()


# --- inbox_count (tab badge) ---

def test_inbox_count_shipped_with_every_view(app, authenticated_client, test_user,
                                             captured_templates):
    _make_conversion(app, test_user['id'], title='InboxOne')
    _make_conversion(app, test_user['id'], title='InboxTwo')
    later = _make_conversion(app, test_user['id'], title='LaterDoc')
    queued = _make_conversion(app, test_user['id'], title='QueuedDoc')
    _set_status(app, later, 'later')
    _add_to_queue(authenticated_client, queued)

    # The tab bar is always visible, so every view ships the count — and the
    # ?status=archive case proves it ignores the active filters.
    for url in ('/library', '/library?view=inbox', '/library?view=queue',
                '/library?status=archive'):
        captured_templates.clear()
        resp = authenticated_client.get(url)
        assert resp.status_code == 200
        _, ctx = captured_templates[-1]
        assert ctx['inbox_count'] == 2, url


# --- available_tags usage counts ---

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
