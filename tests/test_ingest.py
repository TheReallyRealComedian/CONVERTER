"""Ingestion endpoint characterization tests — POST /api/ingest/conversion.

This is the project's first non-session, token-authenticated endpoint, so the
security posture is locked in here: fail-closed without INGEST_TOKEN (503),
constant-time Bearer compare (401 on missing/wrong), CSRF-exempt for THIS view
only, and idempotency over source_id without a schema touch.

The test config sets ``WTF_CSRF_ENABLED=False`` globally (conftest), so a plain
"POST without CSRF token succeeds" would pass even without the exempt() call.
The CSRF test therefore flips protection back ON and proves the Bearer-only
POST still succeeds, plus a structural check that no other view is exempt.
"""
import json
from datetime import datetime, timezone

from models import Conversion, Tag, User, db


TOKEN = 'nl1-test-ingest-token-7f3a9c'
URL = '/api/ingest/conversion'


def _make_user(app, username='alice'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def _auth(token=TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _newsletter(**overrides):
    payload = {
        'conversion_type': 'ai_newsletter',
        'title': '2026-05-30 - AI Newsletter Analyse',
        'content': '# AI Newsletter\n\nHeute geht es um KI-Agenten.',
    }
    payload.update(overrides)
    return payload


# --- 201 create + persistence + mappings ---

def test_ingest_creates_conversion_and_persists(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    uid = _make_user(app)
    resp = client.post(URL, headers=_auth(), json=_newsletter(
        topics=['KI-Agenten', 'Code-Generierung'],
        report_date='2026-05-30',
        source_id='notion-page-abc',
    ))
    assert resp.status_code == 201
    body = resp.get_json()
    assert body['conversion_type'] == 'ai_newsletter'
    assert body['title'] == '2026-05-30 - AI Newsletter Analyse'

    with app.app_context():
        rows = Conversion.query.filter_by(user_id=uid).all()
        assert len(rows) == 1
        c = rows[0]
        assert c.conversion_type == 'ai_newsletter'
        assert c.content.startswith('# AI Newsletter')
        meta = json.loads(c.metadata_json)
        assert meta['source_id'] == 'notion-page-abc'
        assert meta['ingested'] is True
        # topics -> tags via the R2-A junction, normalised lowercase
        assert sorted(t.name for t in c.tag_refs) == ['code-generierung', 'ki-agenten']
        # report_date -> created_at
        assert (c.created_at.year, c.created_at.month, c.created_at.day) == (2026, 5, 30)


# --- Auth: 401 missing / wrong, 503 fail-closed ---

def test_ingest_missing_bearer_returns_401(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    _make_user(app)
    resp = client.post(URL, json=_newsletter())
    assert resp.status_code == 401
    with app.app_context():
        assert Conversion.query.count() == 0


def test_ingest_wrong_token_returns_401(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    _make_user(app)
    resp = client.post(URL, headers=_auth('the-wrong-token'), json=_newsletter())
    assert resp.status_code == 401
    with app.app_context():
        assert Conversion.query.count() == 0


def test_ingest_fail_closed_when_token_unset_or_empty(app, client, monkeypatch):
    # Unset -> 503, even with a Bearer present (auth-config check precedes auth).
    monkeypatch.delenv('INGEST_TOKEN', raising=False)
    _make_user(app)
    resp = client.post(URL, headers=_auth(), json=_newsletter())
    assert resp.status_code == 503

    # Empty string is "unset" too.
    monkeypatch.setenv('INGEST_TOKEN', '')
    resp_empty = client.post(URL, headers=_auth(), json=_newsletter())
    assert resp_empty.status_code == 503
    with app.app_context():
        assert Conversion.query.count() == 0


# --- Body validation: 400 ---

def test_ingest_rejects_non_dict_body(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    _make_user(app)
    resp = client.post(URL, headers=_auth(), json=['not', 'a', 'dict'])
    assert resp.status_code == 400
    assert 'JSON-Objekt' in resp.get_json()['error']


def test_ingest_rejects_missing_content(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    _make_user(app)
    resp = client.post(URL, headers=_auth(), json={'conversion_type': 'ai_newsletter',
                                                    'title': 'No body'})
    assert resp.status_code == 400


def test_ingest_rejects_invalid_conversion_type(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    _make_user(app)
    resp = client.post(URL, headers=_auth(), json={'conversion_type': 'bogus_type',
                                                    'content': 'body'})
    assert resp.status_code == 400
    # Missing conversion_type is equally rejected.
    resp2 = client.post(URL, headers=_auth(), json={'content': 'body'})
    assert resp2.status_code == 400


# --- Dedup: idempotent over source_id ---

def test_ingest_dedup_idempotent_same_source_id(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    uid = _make_user(app)
    payload = _newsletter(source_id='stable-id-1')

    r1 = client.post(URL, headers=_auth(), json=payload)
    assert r1.status_code == 201
    id1 = r1.get_json()['id']
    assert 'deduped' not in r1.get_json()

    r2 = client.post(URL, headers=_auth(), json=payload)
    assert r2.status_code == 200
    assert r2.get_json()['deduped'] is True
    assert r2.get_json()['id'] == id1

    with app.app_context():
        assert Conversion.query.filter_by(user_id=uid).count() == 1


def test_ingest_dedup_with_wildcard_chars_in_source_id(app, client, monkeypatch):
    """Regression: the dedup prefilter must neutralise SQL-LIKE wildcards in
    the source_id. A source_id containing '_' (a LIKE "any single char")
    must still dedup on re-POST — otherwise the lookup misses the stored row
    and silently creates a duplicate. Pre-fix (manual escape without an
    ESCAPE clause) this produced a 2nd row; the fix uses contains(autoescape).
    """
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    uid = _make_user(app)
    payload = _newsletter(source_id='2026_05_30_ai_newsletter')  # underscores = LIKE wildcards

    r1 = client.post(URL, headers=_auth(), json=payload)
    assert r1.status_code == 201
    id1 = r1.get_json()['id']

    r2 = client.post(URL, headers=_auth(), json=payload)
    assert r2.status_code == 200, f'expected dedup 200, got {r2.status_code}'
    assert r2.get_json()['deduped'] is True
    assert r2.get_json()['id'] == id1
    with app.app_context():
        assert Conversion.query.filter_by(user_id=uid).count() == 1


def test_ingest_without_source_id_does_not_dedup(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    uid = _make_user(app)
    # No source_id -> each POST is a fresh row (no idempotency key to match on).
    assert client.post(URL, headers=_auth(), json=_newsletter()).status_code == 201
    assert client.post(URL, headers=_auth(), json=_newsletter()).status_code == 201
    with app.app_context():
        assert Conversion.query.filter_by(user_id=uid).count() == 2


# --- topics[] -> tags ---

def test_ingest_topics_map_to_tags_and_reuse_existing(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    uid = _make_user(app)
    # Pre-create a 'ki' tag so the ingest must REUSE it, not duplicate.
    with app.app_context():
        Tag.get_or_create(uid, 'ki')
        db.session.commit()
        existing_id = Tag.query.filter_by(user_id=uid, name='ki').first().id

    resp = client.post(URL, headers=_auth(), json=_newsletter(topics=['KI', 'Neues Thema']))
    assert resp.status_code == 201
    with app.app_context():
        c = Conversion.query.filter_by(user_id=uid).first()
        assert sorted(t.name for t in c.tag_refs) == ['ki', 'neues thema']
        assert Tag.query.filter_by(user_id=uid, name='ki').count() == 1
        ki = next(t for t in c.tag_refs if t.name == 'ki')
        assert ki.id == existing_id


def test_ingest_non_list_topics_are_ignored(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    uid = _make_user(app)
    resp = client.post(URL, headers=_auth(), json=_newsletter(topics='not-a-list'))
    assert resp.status_code == 201
    with app.app_context():
        c = Conversion.query.filter_by(user_id=uid).first()
        assert list(c.tag_refs) == []


# --- report_date -> created_at ---

def test_ingest_missing_or_unparseable_report_date_defaults_to_now(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    uid = _make_user(app)
    # Missing report_date.
    client.post(URL, headers=_auth(), json=_newsletter(source_id='no-date'))
    # Unparseable report_date.
    client.post(URL, headers=_auth(), json=_newsletter(source_id='bad-date',
                                                        report_date='not-a-date'))
    now = datetime.utcnow()
    with app.app_context():
        for c in Conversion.query.filter_by(user_id=uid).all():
            assert c.created_at is not None
            assert abs((now - c.created_at).total_seconds()) < 300


# --- User resolution (no session) ---

def test_ingest_targets_user_named_by_ingest_user_env(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    alice = _make_user(app, 'alice')
    bob = _make_user(app, 'bob')
    monkeypatch.setenv('INGEST_USER', 'bob')

    resp = client.post(URL, headers=_auth(), json=_newsletter())
    assert resp.status_code == 201
    with app.app_context():
        assert Conversion.query.filter_by(user_id=bob).count() == 1
        assert Conversion.query.filter_by(user_id=alice).count() == 0


def test_ingest_single_user_fallback_when_ingest_user_unset(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    monkeypatch.delenv('INGEST_USER', raising=False)
    uid = _make_user(app, 'solo')

    resp = client.post(URL, headers=_auth(), json=_newsletter())
    assert resp.status_code == 201
    with app.app_context():
        assert Conversion.query.filter_by(user_id=uid).count() == 1


def test_ingest_ingest_user_set_but_missing_returns_503(app, client, monkeypatch):
    # A set-but-unresolvable INGEST_USER must 503, never silently fall back to
    # another account.
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    _make_user(app, 'alice')
    monkeypatch.setenv('INGEST_USER', 'ghost')

    resp = client.post(URL, headers=_auth(), json=_newsletter())
    assert resp.status_code == 503


def test_ingest_no_user_returns_503(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    monkeypatch.delenv('INGEST_USER', raising=False)
    # No users in the (wiped) DB.
    resp = client.post(URL, headers=_auth(), json=_newsletter())
    assert resp.status_code == 503


# --- CSRF exemption (proven under enforced CSRF) ---

def test_ingest_succeeds_with_csrf_enforced_and_no_csrf_token(app, client, monkeypatch):
    monkeypatch.setenv('INGEST_TOKEN', TOKEN)
    _make_user(app)
    # Re-enable CSRF (conftest disables it globally). A session-less, Bearer-only
    # POST carries NO CSRF token; it must still succeed because the view is
    # exempt. Without the exempt() call this would be a 400 CSRF error.
    monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)
    resp = client.post(URL, headers=_auth(), json=_newsletter())
    assert resp.status_code == 201


def test_only_the_ingest_view_is_csrf_exempt(app):
    csrf = app.extensions['csrf']
    assert 'app_pkg.ingest.api_ingest_conversion' in csrf._exempt_views
    # The session-backed create route is NOT exempt — the exemption is scoped
    # to the single non-session endpoint.
    assert 'app_pkg.library.api_create_conversion' not in csrf._exempt_views
