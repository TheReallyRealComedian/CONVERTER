"""Highlight CRUD API characterization tests (R1-B-A).

Locks in: POST/GET/DELETE on the highlights endpoints respect ownership,
input validation rejects empty / oversized payloads, cascade delete wipes
highlights when their parent conversion is removed.
"""
import json

from models import db, Conversion, Highlight, User


def _make_conversion(app, user_id, **overrides):
    payload = dict(
        user_id=user_id,
        conversion_type='markdown_input',
        title='Sample Title',
        content='Sample content body.',
        tags='alpha',
        metadata_json=json.dumps({'src': 'test'}),
    )
    payload.update(overrides)
    with app.app_context():
        c = Conversion(**payload)
        db.session.add(c)
        db.session.commit()
        return c.id


def _make_other_user(app, username='bob'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def test_api_create_highlight_persists(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.post(
        f'/api/conversions/{cid}/highlights',
        json={'exact': 'Sample content', 'prefix': '', 'suffix': ' body.'},
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert data['exact'] == 'Sample content'
    assert data['suffix'] == ' body.'
    assert data['conversion_id'] == cid
    assert 'id' in data
    with app.app_context():
        rows = Highlight.query.filter_by(conversion_id=cid).all()
        assert len(rows) == 1


def test_api_create_highlight_rejects_empty_exact(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.post(
        f'/api/conversions/{cid}/highlights',
        json={'exact': '   ', 'prefix': '', 'suffix': ''},
    )
    assert resp.status_code == 400


def test_api_create_highlight_rejects_oversized_exact(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.post(
        f'/api/conversions/{cid}/highlights',
        json={'exact': 'x' * 5001, 'prefix': '', 'suffix': ''},
    )
    assert resp.status_code == 400


def test_api_create_highlight_rejects_non_dict_body(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    resp = authenticated_client.post(
        f'/api/conversions/{cid}/highlights',
        json=['not', 'a', 'dict'],
    )
    assert resp.status_code == 400


def test_api_create_highlight_on_foreign_conversion_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'bob')
    cid = _make_conversion(app, other_id, title="Bob's doc")
    resp = authenticated_client.post(
        f'/api/conversions/{cid}/highlights',
        json={'exact': 'Sample', 'prefix': '', 'suffix': ''},
    )
    assert resp.status_code == 404


def test_api_list_highlights_returns_own_rows_in_order(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    authenticated_client.post(f'/api/conversions/{cid}/highlights',
                              json={'exact': 'first', 'prefix': '', 'suffix': ''})
    authenticated_client.post(f'/api/conversions/{cid}/highlights',
                              json={'exact': 'second', 'prefix': '', 'suffix': ''})
    resp = authenticated_client.get(f'/api/conversions/{cid}/highlights')
    assert resp.status_code == 200
    data = resp.get_json()
    assert [h['exact'] for h in data] == ['first', 'second']


def test_api_list_highlights_on_foreign_conversion_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'carol')
    cid = _make_conversion(app, other_id, title="Carol's doc")
    resp = authenticated_client.get(f'/api/conversions/{cid}/highlights')
    assert resp.status_code == 404


def test_api_delete_highlight_removes_row(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    create = authenticated_client.post(
        f'/api/conversions/{cid}/highlights',
        json={'exact': 'gone', 'prefix': '', 'suffix': ''},
    )
    hid = create.get_json()['id']
    resp = authenticated_client.delete(f'/api/highlights/{hid}')
    assert resp.status_code == 200
    with app.app_context():
        assert Highlight.query.get(hid) is None
    follow = authenticated_client.get(f'/api/conversions/{cid}/highlights')
    assert follow.get_json() == []


def test_api_delete_highlight_on_foreign_row_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'dave')
    cid = _make_conversion(app, other_id, title="Dave's doc")
    with app.app_context():
        h = Highlight(conversion_id=cid, exact='secret', prefix='', suffix='')
        db.session.add(h)
        db.session.commit()
        hid = h.id
    resp = authenticated_client.delete(f'/api/highlights/{hid}')
    assert resp.status_code == 404
    with app.app_context():
        assert Highlight.query.get(hid) is not None


def test_conversion_delete_cascades_to_highlights(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    authenticated_client.post(f'/api/conversions/{cid}/highlights',
                              json={'exact': 'doomed', 'prefix': '', 'suffix': ''})
    with app.app_context():
        assert Highlight.query.filter_by(conversion_id=cid).count() == 1
    delete_resp = authenticated_client.delete(f'/api/conversions/{cid}')
    assert delete_resp.status_code == 200
    with app.app_context():
        assert Highlight.query.filter_by(conversion_id=cid).count() == 0


# --- R1-B-B: Highlight-Notes via PATCH ---

def _create_highlight(client, cid, exact='Sample content'):
    resp = client.post(
        f'/api/conversions/{cid}/highlights',
        json={'exact': exact, 'prefix': '', 'suffix': ''},
    )
    assert resp.status_code == 201
    return resp.get_json()['id']


def test_api_list_highlights_includes_note_field(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    _create_highlight(authenticated_client, cid, 'first')
    resp = authenticated_client.get(f'/api/conversions/{cid}/highlights')
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert 'note' in data[0]
    assert data[0]['note'] is None


def test_api_patch_highlight_sets_note(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _create_highlight(authenticated_client, cid)
    resp = authenticated_client.patch(
        f'/api/highlights/{hid}',
        json={'note': 'Wichtiges Detail.'},
    )
    assert resp.status_code == 200
    assert resp.get_json()['note'] == 'Wichtiges Detail.'
    with app.app_context():
        assert Highlight.query.get(hid).note == 'Wichtiges Detail.'


def test_api_patch_highlight_with_null_clears_note(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _create_highlight(authenticated_client, cid)
    authenticated_client.patch(f'/api/highlights/{hid}', json={'note': 'temp'})
    resp = authenticated_client.patch(f'/api/highlights/{hid}', json={'note': None})
    assert resp.status_code == 200
    assert resp.get_json()['note'] is None
    with app.app_context():
        assert Highlight.query.get(hid).note is None


def test_api_patch_highlight_with_empty_string_clears_note(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _create_highlight(authenticated_client, cid)
    authenticated_client.patch(f'/api/highlights/{hid}', json={'note': 'temp'})
    resp = authenticated_client.patch(f'/api/highlights/{hid}', json={'note': ''})
    assert resp.status_code == 200
    # Empty string normalisiert auf NULL — keine Unterscheidung zwischen "leer" und "nicht gesetzt".
    assert resp.get_json()['note'] is None
    with app.app_context():
        assert Highlight.query.get(hid).note is None


def test_api_patch_highlight_rejects_oversized_note(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _create_highlight(authenticated_client, cid)
    resp = authenticated_client.patch(
        f'/api/highlights/{hid}',
        json={'note': 'x' * 2001},
    )
    assert resp.status_code == 400


def test_api_patch_highlight_rejects_non_dict_body(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _create_highlight(authenticated_client, cid)
    resp = authenticated_client.patch(
        f'/api/highlights/{hid}',
        json=['not', 'a', 'dict'],
    )
    assert resp.status_code == 400


def test_api_patch_highlight_rejects_missing_note_key(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _create_highlight(authenticated_client, cid)
    resp = authenticated_client.patch(
        f'/api/highlights/{hid}',
        json={'other_field': 'value'},
    )
    assert resp.status_code == 400


def test_api_patch_highlight_rejects_non_string_non_null_note(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _create_highlight(authenticated_client, cid)
    resp = authenticated_client.patch(
        f'/api/highlights/{hid}',
        json={'note': 12345},
    )
    assert resp.status_code == 400


def test_api_patch_highlight_on_foreign_row_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'eve')
    cid = _make_conversion(app, other_id, title="Eve's doc")
    with app.app_context():
        h = Highlight(conversion_id=cid, exact='secret', prefix='', suffix='')
        db.session.add(h)
        db.session.commit()
        hid = h.id
    resp = authenticated_client.patch(
        f'/api/highlights/{hid}',
        json={'note': 'intruder'},
    )
    assert resp.status_code == 404
    with app.app_context():
        assert Highlight.query.get(hid).note is None
