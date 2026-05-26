"""Tag API characterization tests (R1-B-C).

Locks in: find-or-create attach, lowercase+trim normalisation, idempotent
attach, detach without deleting the tag, full tag delete cascading the
junction, per-user namespace isolation.
"""
import json

from models import Conversion, Highlight, Tag, User, db


def _make_conversion(app, user_id, **overrides):
    payload = dict(
        user_id=user_id,
        conversion_type='markdown_input',
        title='Sample Title',
        content='Sample content body.',
        tags='',
        metadata_json=json.dumps({'src': 'test'}),
    )
    payload.update(overrides)
    with app.app_context():
        c = Conversion(**payload)
        db.session.add(c)
        db.session.commit()
        return c.id


def _make_highlight(app, conversion_id, exact='Sample content'):
    with app.app_context():
        h = Highlight(conversion_id=conversion_id, exact=exact, prefix='', suffix='')
        db.session.add(h)
        db.session.commit()
        return h.id


def _make_other_user(app, username='bob'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def _login(client, username, password):
    resp = client.post('/login', data={'username': username, 'password': password})
    assert resp.status_code == 302, f'login failed: {resp.status_code}'


# --- GET /api/tags ---

def test_api_list_tags_empty(app, authenticated_client, test_user):
    resp = authenticated_client.get('/api/tags')
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_api_list_tags_returns_sorted_with_counts(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid_a = _make_highlight(app, cid, 'first')
    hid_b = _make_highlight(app, cid, 'second')
    authenticated_client.post(f'/api/highlights/{hid_a}/tags', json={'name': 'zeta'})
    authenticated_client.post(f'/api/highlights/{hid_a}/tags', json={'name': 'alpha'})
    authenticated_client.post(f'/api/highlights/{hid_b}/tags', json={'name': 'alpha'})

    resp = authenticated_client.get('/api/tags')
    assert resp.status_code == 200
    data = resp.get_json()
    assert [t['name'] for t in data] == ['alpha', 'zeta']
    by_name = {t['name']: t for t in data}
    assert by_name['alpha']['highlight_count'] == 2
    assert by_name['zeta']['highlight_count'] == 1
    # R2-A: conversion_count is part of the payload, defaults to 0 when no
    # conversion has attached the tag yet.
    assert by_name['alpha']['conversion_count'] == 0
    assert by_name['zeta']['conversion_count'] == 0


def test_api_list_tags_includes_conversion_count(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    authenticated_client.post(f'/api/conversions/{cid}/tags', json={'name': 'docs'})
    cid2 = _make_conversion(app, test_user['id'], title='Second')
    authenticated_client.post(f'/api/conversions/{cid2}/tags', json={'name': 'docs'})

    resp = authenticated_client.get('/api/tags')
    assert resp.status_code == 200
    data = resp.get_json()
    by_name = {t['name']: t for t in data}
    assert by_name['docs']['conversion_count'] == 2
    assert by_name['docs']['highlight_count'] == 0


# --- POST /api/highlights/<id>/tags ---

def test_api_attach_tag_creates_and_attaches(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    resp = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'ki'})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body['name'] == 'ki'
    with app.app_context():
        tag = Tag.query.filter_by(name='ki').first()
        assert tag is not None
        assert tag.user_id == test_user['id']
        h = Highlight.query.get(hid)
        assert tag in h.tags


def test_api_attach_tag_reuses_existing_tag(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid_a = _make_highlight(app, cid, 'first')
    hid_b = _make_highlight(app, cid, 'second')
    r1 = authenticated_client.post(f'/api/highlights/{hid_a}/tags', json={'name': 'shared'})
    assert r1.status_code == 201
    first_id = r1.get_json()['id']

    r2 = authenticated_client.post(f'/api/highlights/{hid_b}/tags', json={'name': 'shared'})
    assert r2.status_code == 201
    assert r2.get_json()['id'] == first_id

    with app.app_context():
        assert Tag.query.filter_by(name='shared').count() == 1


def test_api_attach_tag_idempotent_when_already_attached(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    r1 = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'dup'})
    assert r1.status_code == 201
    r2 = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'dup'})
    # Doppel-Attach ist no-op 200, kein 409, kein Crash.
    assert r2.status_code == 200
    assert r2.get_json()['name'] == 'dup'
    with app.app_context():
        h = Highlight.query.get(hid)
        assert len([t for t in h.tags if t.name == 'dup']) == 1


def test_api_attach_tag_normalizes_lowercase_and_trim(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    resp = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': '  KI  '})
    assert resp.status_code == 201
    assert resp.get_json()['name'] == 'ki'

    # Zweiter Call mit „KI" (case-different) muss denselben Tag treffen.
    resp2 = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'KI'})
    assert resp2.status_code == 200
    assert resp2.get_json()['id'] == resp.get_json()['id']


def test_api_attach_tag_rejects_empty_and_oversized(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    assert authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': ''}).status_code == 400
    assert authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': '   '}).status_code == 400
    assert authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'x' * 81}).status_code == 400
    assert authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': None}).status_code == 400


def test_api_attach_tag_to_foreign_highlight_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'bob')
    cid = _make_conversion(app, other_id, title="Bob's doc")
    hid = _make_highlight(app, cid)
    resp = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'intruder'})
    assert resp.status_code == 404


# --- DELETE /api/highlights/<id>/tags/<tag_id> ---

def test_api_detach_tag_removes_junction_keeps_tag(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    attach = authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'keepme'})
    tag_id = attach.get_json()['id']

    resp = authenticated_client.delete(f'/api/highlights/{hid}/tags/{tag_id}')
    assert resp.status_code == 200
    with app.app_context():
        h = Highlight.query.get(hid)
        assert all(t.id != tag_id for t in h.tags)
        assert Tag.query.get(tag_id) is not None


def test_api_detach_foreign_user_tag_returns_404(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    other_id = _make_other_user(app, 'carol')
    with app.app_context():
        foreign_tag = Tag(user_id=other_id, name='secret')
        db.session.add(foreign_tag)
        db.session.commit()
        foreign_tag_id = foreign_tag.id
    resp = authenticated_client.delete(f'/api/highlights/{hid}/tags/{foreign_tag_id}')
    assert resp.status_code == 404


# --- DELETE /api/tags/<id> ---

def test_api_delete_tag_cascades_to_junction(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid_a = _make_highlight(app, cid, 'first')
    hid_b = _make_highlight(app, cid, 'second')
    attach = authenticated_client.post(f'/api/highlights/{hid_a}/tags', json={'name': 'gone'})
    tag_id = attach.get_json()['id']
    authenticated_client.post(f'/api/highlights/{hid_b}/tags', json={'name': 'gone'})

    resp = authenticated_client.delete(f'/api/tags/{tag_id}')
    assert resp.status_code == 200
    with app.app_context():
        assert Tag.query.get(tag_id) is None
        for hid in (hid_a, hid_b):
            h = Highlight.query.get(hid)
            assert all(t.id != tag_id for t in h.tags)


def test_api_delete_foreign_tag_returns_404(app, authenticated_client, test_user):
    other_id = _make_other_user(app, 'dave')
    with app.app_context():
        t = Tag(user_id=other_id, name='private')
        db.session.add(t)
        db.session.commit()
        tag_id = t.id
    resp = authenticated_client.delete(f'/api/tags/{tag_id}')
    assert resp.status_code == 404
    with app.app_context():
        assert Tag.query.get(tag_id) is not None


# --- Per-user namespace ---

def test_api_list_tags_isolated_per_user(app, client, test_user):
    cid_alice = _make_conversion(app, test_user['id'])
    hid_alice = _make_highlight(app, cid_alice)
    _login(client, test_user['username'], test_user['password'])
    client.post(f'/api/highlights/{hid_alice}/tags', json={'name': 'alice-only'})
    client.get('/logout')

    bob_id = _make_other_user(app, 'bob')
    cid_bob = _make_conversion(app, bob_id)
    hid_bob = _make_highlight(app, cid_bob)
    _login(client, 'bob', 'password1234')
    client.post(f'/api/highlights/{hid_bob}/tags', json={'name': 'bob-only'})

    resp = client.get('/api/tags')
    assert resp.status_code == 200
    names = [t['name'] for t in resp.get_json()]
    assert names == ['bob-only']


# --- Highlight to_dict includes tags ---

def test_highlight_list_response_includes_tags(app, authenticated_client, test_user):
    cid = _make_conversion(app, test_user['id'])
    hid = _make_highlight(app, cid)
    authenticated_client.post(f'/api/highlights/{hid}/tags', json={'name': 'meta'})

    resp = authenticated_client.get(f'/api/conversions/{cid}/highlights')
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1
    assert 'tags' in data[0]
    assert len(data[0]['tags']) == 1
    assert data[0]['tags'][0]['name'] == 'meta'


# --- /tags page route ---

def test_tags_page_renders_for_authenticated_user(app, authenticated_client, test_user):
    resp = authenticated_client.get('/tags')
    assert resp.status_code == 200
    assert b'Tags' in resp.data


def test_tags_page_requires_login(app, client):
    resp = client.get('/tags', follow_redirects=False)
    assert resp.status_code == 302
    assert '/login' in resp.headers.get('Location', '')
