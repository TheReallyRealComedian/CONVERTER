"""Library + conversions API characterization tests.

Locks in: empty list view renders, list view shows existing conversions,
detail view loads a conversion that belongs to the user, the API endpoints
that create / update / delete conversions write to the DB, and a non-owner
cannot read another user's conversion (404).
"""
import json

from models import db, User, Conversion


def _make_conversion(app, user_id, **overrides):
    payload = dict(
        user_id=user_id,
        conversion_type='markdown_input',
        title='Sample Title',
        content='Sample content body.',
        tags='alpha,beta',
        metadata_json=json.dumps({'src': 'test'}),
    )
    payload.update(overrides)
    with app.app_context():
        c = Conversion(**payload)
        db.session.add(c)
        db.session.commit()
        return c.id


def test_library_empty_renders(authenticated_client):
    resp = authenticated_client.get('/library')
    assert resp.status_code == 200


def test_library_empty_state_filter_aware(authenticated_client):
    """F-6 P7: empty list with an active filter renders the filter-mismatch
    variant with a reset link, not the global "first conversion" hint.
    """
    global_empty = authenticated_client.get('/library')
    assert b'Noch keine gespeicherten Eintr' in global_empty.data

    filtered_empty = authenticated_client.get('/library?favorites=1')
    assert b'Keine Treffer mit aktuellen Filtern' in filtered_empty.data
    assert b'Filter zur' in filtered_empty.data


def test_library_lists_existing_conversion(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='My Library Entry')
    resp = authenticated_client.get('/library')
    assert resp.status_code == 200
    assert b'My Library Entry' in resp.data


def test_library_detail_renders_owned_conversion(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], title='Detail Page Entry')
    resp = authenticated_client.get(f'/library/{cid}')
    assert resp.status_code == 200
    assert b'Detail Page Entry' in resp.data


def test_library_detail_404_for_other_users_conversion(app, authenticated_client, test_user):
    with app.app_context():
        other = User(username='bob')
        other.set_password('password1234')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    cid = _make_conversion(app, other_id, title="Bob's secret")
    resp = authenticated_client.get(f'/library/{cid}')
    assert resp.status_code == 404


def test_api_create_conversion_persists(app, authenticated_client, test_user):
    resp = authenticated_client.post(
        '/api/conversions',
        json={
            'conversion_type': 'audio_transcription',
            'title': 'Recorded meeting',
            'content': 'Transcript body...',
            'tags': 'meeting',
        },
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data['title'] == 'Recorded meeting'
    assert data['conversion_type'] == 'audio_transcription'
    with app.app_context():
        rows = Conversion.query.filter_by(user_id=test_user['id']).all()
    assert len(rows) == 1


def test_api_create_conversion_accepts_ai_newsletter(app, authenticated_client, test_user):
    """NL1: the ai_newsletter type is accepted by the create-validation
    (ALLOWED_CONVERSION_TYPES) — foundation for the Library filter/badge and
    the ingestion endpoint that creates rows of this type.
    """
    resp = authenticated_client.post(
        '/api/conversions',
        json={
            'conversion_type': 'ai_newsletter',
            'title': '2026-05-30 - AI Newsletter Analyse',
            'content': '# Newsletter body',
        },
    )
    assert resp.status_code == 201
    assert resp.get_json()['conversion_type'] == 'ai_newsletter'
    # The list view renders the new type badge: the template elif emits the
    # "AI-Newsletter" label and the .type-ai_newsletter CSS hook for the row.
    lib = authenticated_client.get('/library')
    assert lib.status_code == 200
    assert b'type-ai_newsletter' in lib.data
    assert b'AI-Newsletter' in lib.data


def test_api_create_conversion_rejects_missing_content(authenticated_client):
    resp = authenticated_client.post('/api/conversions', json={'title': 'No content'})
    assert resp.status_code == 400


def test_api_create_conversion_rejects_non_dict_body(authenticated_client):
    """F-017: a JSON list / scalar body used to crash the route on
    ``data.get(...)`` with an AttributeError surfacing as 500. The shared
    inline guard now returns 400 + DE-microcopy instead.
    """
    resp = authenticated_client.post(
        '/api/conversions',
        json=['not', 'a', 'dict'],
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert 'JSON-Objekt' in body['error']


def test_api_update_conversion_changes_title(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], title='Original')
    resp = authenticated_client.put(
        f'/api/conversions/{cid}',
        json={'title': 'Renamed'},
    )
    assert resp.status_code == 200
    assert resp.get_json()['title'] == 'Renamed'


def test_api_update_conversion_rejects_non_dict_body(app, authenticated_client, test_user):
    """F-3 P1: PUT must return 400 for non-dict bodies so the JS auto-save
    handler can surface a Failure-Banner instead of guessing at success.
    """
    cid = _make_conversion(app, test_user['id'], title='Edge case')
    resp = authenticated_client.put(
        f'/api/conversions/{cid}',
        json=['not', 'a', 'dict'],
    )
    assert resp.status_code == 400
    assert 'JSON-Objekt' in resp.get_json()['error']


def test_api_delete_conversion_removes_row(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'], title='To delete')
    resp = authenticated_client.delete(f'/api/conversions/{cid}')
    assert resp.status_code == 200
    with app.app_context():
        assert Conversion.query.get(cid) is None


def test_file_size_filter_matches_js_helper(app):
    """F-3 P12: the server-side ``file_size`` Jinja filter mirrors the
    formatFileSize helper in static/js/_utils.js. Same anchors so the
    JS download view and the rendered detail view agree on the unit.
    """
    fmt = app.jinja_env.filters['file_size']
    assert fmt(222) == '222 B'
    assert fmt(4731) == '4,6 KB'
    assert fmt(1234567) == '1,2 MB'


def test_format_card_datetime_filter_renders_de_month_abbr(app):
    """F-6 P11: card date abbreviations are container-locale-agnostic and
    use the DE month list (Mär instead of Mar, Mai instead of May).
    """
    from datetime import datetime
    fmt = app.jinja_env.filters['format_card_datetime']
    assert fmt(datetime(2026, 5, 10, 14, 30)) == '10 Mai 2026, 14:30'
    assert fmt(datetime(2026, 3, 1, 9, 5)) == '01 Mär 2026, 09:05'
    assert fmt(datetime(2025, 12, 30, 23, 59)) == '30 Dez 2025, 23:59'
    assert fmt(None) == ''


def test_library_ignores_unknown_type_filter(app, authenticated_client, test_user):
    """F-6 P10: an unknown ``?type=...`` value falls back to the unfiltered
    list (no DB query against the unknown enum value).
    """
    _make_conversion(app, test_user['id'], title='Markdown entry',
                     conversion_type='markdown_input')
    resp = authenticated_client.get('/library?type=nonsense')
    assert resp.status_code == 200
    assert b'Markdown entry' in resp.data


def test_library_ignores_unknown_per_page(authenticated_client):
    """F-6 P12: per_page values outside the allowlist fall back to the
    default, so the URL contract can't be used to over-fetch.
    """
    resp = authenticated_client.get('/library?per_page=999')
    assert resp.status_code == 200
    resp = authenticated_client.get('/library?per_page=50')
    assert resp.status_code == 200


def test_api_delete_conversion_404_for_other_users_conversion(app, authenticated_client, test_user):
    """F-3 P3: DELETE on a non-owned row must 404 so the JS Delete handler
    can route the user back to the Library via the explicit race-404 branch.
    """
    with app.app_context():
        other = User(username='carol')
        other.set_password('password1234')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    cid = _make_conversion(app, other_id, title="Carol's secret")
    resp = authenticated_client.delete(f'/api/conversions/{cid}')
    assert resp.status_code == 404


def test_api_update_conversion_404_for_other_users_conversion(app, authenticated_client, test_user):
    """F-6 P3: PUT on a non-owned row must 404 so the list-view Favorite
    toggle handler can surface a failure-banner via safeJSON instead of
    silently flipping the glyph.
    """
    with app.app_context():
        other = User(username='dave')
        other.set_password('password1234')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    cid = _make_conversion(app, other_id, title="Dave's secret")
    resp = authenticated_client.put(
        f'/api/conversions/{cid}',
        json={'is_favorite': True},
    )
    assert resp.status_code == 404


# --- MCP1: JSON read-API (GET /api/conversions[/<id>]) ---


def test_api_list_conversions_requires_login(client):
    """MCP1: the JSON list endpoint is @login_required like the rest of the
    /api surface — an anonymous client is redirected to the login view (302),
    never served another user's rows."""
    resp = client.get('/api/conversions')
    assert resp.status_code == 302
    assert '/login' in resp.headers['Location']


def test_api_list_conversions_owner_scoped(app, authenticated_client, test_user):
    """A sees only A's conversions, never B's (owner-scoped on current_user)."""
    with app.app_context():
        other = User(username='erin')
        other.set_password('password1234')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    _make_conversion(app, test_user['id'], title='Mine')
    _make_conversion(app, other_id, title='Not mine')

    resp = authenticated_client.get('/api/conversions')
    assert resp.status_code == 200
    data = resp.get_json()
    assert [it['title'] for it in data['items']] == ['Mine']
    assert data['total'] == 1


def test_api_list_conversions_type_filter(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='Doc',
                     conversion_type='document_to_markdown')
    _make_conversion(app, test_user['id'], title='Audio',
                     conversion_type='audio_transcription')

    resp = authenticated_client.get('/api/conversions?type=audio_transcription')
    assert resp.status_code == 200
    data = resp.get_json()
    assert [it['title'] for it in data['items']] == ['Audio']
    assert data['total'] == 1


def test_api_list_conversions_invalid_type_400(authenticated_client):
    assert authenticated_client.get('/api/conversions?type=nonsense').status_code == 400


def test_api_list_conversions_status_filter(app, authenticated_client, test_user):
    _make_conversion(app, test_user['id'], title='Inbox one', lifecycle_status='inbox')
    _make_conversion(app, test_user['id'], title='Archived one', lifecycle_status='archive')

    resp = authenticated_client.get('/api/conversions?status=archive')
    assert resp.status_code == 200
    assert [it['title'] for it in resp.get_json()['items']] == ['Archived one']


def test_api_list_conversions_invalid_status_400(authenticated_client):
    assert authenticated_client.get('/api/conversions?status=bogus').status_code == 400


def test_api_list_conversions_exclude_status(app, authenticated_client, test_user):
    """exclude_status=archive yields the "unarchived" set the MCP wants."""
    _make_conversion(app, test_user['id'], title='Live one', lifecycle_status='inbox')
    _make_conversion(app, test_user['id'], title='Archived one', lifecycle_status='archive')

    resp = authenticated_client.get('/api/conversions?exclude_status=archive')
    assert resp.status_code == 200
    titles = [it['title'] for it in resp.get_json()['items']]
    assert 'Live one' in titles
    assert 'Archived one' not in titles


def test_api_list_conversions_invalid_exclude_status_400(authenticated_client):
    assert authenticated_client.get('/api/conversions?exclude_status=bogus').status_code == 400


def test_api_list_conversions_summary_omits_content(app, authenticated_client, test_user):
    """Summary carries content_length + content_preview + parsed metadata but
    NOT the full content body (the whole point of the slim list)."""
    body = 'X' * 1000
    _make_conversion(app, test_user['id'], title='Long', content=body,
                     metadata_json=json.dumps({'src': 'test', 'k': 1}))
    item = authenticated_client.get('/api/conversions').get_json()['items'][0]
    assert 'content' not in item
    assert item['content_length'] == 1000
    assert item['content_preview'] == 'X' * 300
    assert item['metadata'] == {'src': 'test', 'k': 1}
    assert item['tag_refs'] == []


def test_api_list_conversions_limit_offset_and_total(app, authenticated_client, test_user):
    from datetime import datetime
    for i in range(3):
        _make_conversion(app, test_user['id'], title=f'C{i}',
                         created_at=datetime(2026, 1, i + 1))
    data = authenticated_client.get('/api/conversions?limit=1&offset=0').get_json()
    assert len(data['items']) == 1
    assert data['total'] == 3  # total is the full count, before the limit window
    assert data['limit'] == 1
    assert data['offset'] == 0


def test_api_list_conversions_limit_caps_at_500(authenticated_client):
    resp = authenticated_client.get('/api/conversions?limit=999')
    assert resp.status_code == 200
    assert resp.get_json()['limit'] == 500


def test_api_list_conversions_invalid_limit_offset_400(authenticated_client):
    assert authenticated_client.get('/api/conversions?limit=abc').status_code == 400
    assert authenticated_client.get('/api/conversions?limit=0').status_code == 400
    assert authenticated_client.get('/api/conversions?offset=-1').status_code == 400
    assert authenticated_client.get('/api/conversions?offset=abc').status_code == 400


def test_api_list_conversions_sorted_created_at_desc(app, authenticated_client, test_user):
    from datetime import datetime
    _make_conversion(app, test_user['id'], title='Oldest', created_at=datetime(2026, 1, 1))
    _make_conversion(app, test_user['id'], title='Newest', created_at=datetime(2026, 6, 1))
    _make_conversion(app, test_user['id'], title='Middle', created_at=datetime(2026, 3, 1))
    titles = [it['title'] for it in
              authenticated_client.get('/api/conversions').get_json()['items']]
    assert titles == ['Newest', 'Middle', 'Oldest']


def test_api_get_conversion_full_to_dict(app, authenticated_client, test_user):
    body = 'Full body content here.'
    cid = _make_conversion(app, test_user['id'], title='One', content=body,
                           metadata_json=json.dumps({'recorded_at': '2026-06-12T10:00:00+02:00'}))
    resp = authenticated_client.get(f'/api/conversions/{cid}')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['content'] == body  # single reader returns the FULL content
    assert data['metadata'] == {'recorded_at': '2026-06-12T10:00:00+02:00'}
    assert data['id'] == cid


def test_api_get_conversion_404_missing(authenticated_client):
    assert authenticated_client.get('/api/conversions/999999').status_code == 404


def test_api_get_conversion_404_for_other_users_conversion(app, authenticated_client, test_user):
    with app.app_context():
        other = User(username='frank')
        other.set_password('password1234')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    cid = _make_conversion(app, other_id, title="Frank's secret")
    assert authenticated_client.get(f'/api/conversions/{cid}').status_code == 404
