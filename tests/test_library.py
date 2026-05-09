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
