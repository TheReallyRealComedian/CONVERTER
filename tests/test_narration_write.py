"""NARR-3 Phase 2 — POST /api/narrations (token write + enqueue).

The agent's narration entry point: a ``NARRATION_TOKEN``-authed, CSRF-exempt POST
that validates the turn contract, creates the ``pending`` audio_narration
Conversion, and enqueues the DB-free render task. Auth posture mirrors the Card
write (separate secret, fail-closed 503, constant-time 401, never logged).

RQ is mocked at the ``task_queue`` / ``Job`` singletons (``mock_redis_queue``
fixture) — no real Redis, no real Cloud-TTS call. The enqueue is captured to
assert the task + args + job options without running a worker.
"""
import json

import pytest

from app_pkg.config import TIMEOUT_RQ_JOB_SECONDS
from models import Conversion, User, db
from services.narration_render import DEFAULT_NARRATION_MODEL
from tasks import generate_narration_task


NARR_URL = '/api/narrations'
NARRATION_TOKEN = 'narr-test-token-7c3f'

_TURNS = [
    {'speaker': 'Anna', 'text': 'Hallo zusammen.'},
    {'speaker': 'Ben', 'text': 'Schön, hier zu sein.'},
]
_VOICES = {'Anna': 'Kore', 'Ben': 'Puck'}


def _auth(token=NARRATION_TOKEN):
    return {'Authorization': f'Bearer {token}'}


def _payload(**ov):
    p = {
        'title': 'Mein Dialog',
        'mode': 'two_speaker',
        'voices': dict(_VOICES),
        'turns': [dict(t) for t in _TURNS],
        'style_prompt': 'ruhig',
    }
    p.update(ov)
    return p


def _make_other_user(app, username='bob'):
    with app.app_context():
        u = User(username=username)
        u.set_password('password1234')
        db.session.add(u)
        db.session.commit()
        return u.id


def _narration_count(app):
    with app.app_context():
        return Conversion.query.filter_by(conversion_type='audio_narration').count()


# --- auth matrix: 503 fail-closed / 401 missing+wrong / 202 ------------------

def test_create_fail_closed_without_token(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.delenv('NARRATION_TOKEN', raising=False)
    # Unset → 503 even with a Bearer present (config check precedes auth).
    assert client.post(NARR_URL, headers=_auth(), json=_payload()).status_code == 503
    monkeypatch.setenv('NARRATION_TOKEN', '')
    assert client.post(NARR_URL, headers=_auth(), json=_payload()).status_code == 503
    assert _narration_count(app) == 0


def test_create_401_missing_and_wrong_token(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    assert client.post(NARR_URL, json=_payload()).status_code == 401
    assert client.post(NARR_URL, headers=_auth('the-wrong-token'),
                       json=_payload()).status_code == 401
    assert _narration_count(app) == 0


def test_create_202_with_good_token(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    resp = client.post(NARR_URL, headers=_auth(), json=_payload())
    assert resp.status_code == 202
    body = resp.get_json()
    assert body['status'] == 'pending'
    assert body['job_id'] == 'test-job-123'        # from the mock job
    assert isinstance(body['narration_id'], int)


def test_create_targets_ingest_user(app, client, test_user, monkeypatch, mock_redis_queue):
    # The narration write resolves the target via the SAME INGEST_USER hook.
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    bob = _make_other_user(app, 'bob')
    monkeypatch.setenv('INGEST_USER', 'bob')
    resp = client.post(NARR_URL, headers=_auth(), json=_payload())
    assert resp.status_code == 202
    with app.app_context():
        assert Conversion.query.filter_by(
            conversion_type='audio_narration', user_id=bob).count() == 1
        assert Conversion.query.filter_by(
            conversion_type='audio_narration', user_id=test_user['id']).count() == 0


# --- body validation: 400 ----------------------------------------------------

def test_create_400_non_dict_body(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    resp = client.post(NARR_URL, headers=_auth(), json=['not', 'a', 'dict'])
    assert resp.status_code == 400
    assert _narration_count(app) == 0


def test_create_400_on_validate_turns_violations(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    # empty turns
    assert client.post(NARR_URL, headers=_auth(), json=_payload(turns=[])).status_code == 400
    # unknown mode
    assert client.post(NARR_URL, headers=_auth(), json=_payload(mode='solo')).status_code == 400
    # voices missing a declared speaker → message names the missing speaker
    r = client.post(NARR_URL, headers=_auth(), json=_payload(voices={'Anna': 'Kore'}))
    assert r.status_code == 400
    assert 'Ben' in r.get_json()['error']
    assert _narration_count(app) == 0


# --- pending Conversion + enqueue capture ------------------------------------

def test_create_persists_pending_conversion(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    resp = client.post(NARR_URL, headers=_auth(), json=_payload())
    assert resp.status_code == 202
    nid = resp.get_json()['narration_id']

    with app.app_context():
        conv = db.session.get(Conversion, nid)
        assert conv is not None
        assert conv.user_id == test_user['id']          # owner = resolved target
        assert conv.conversion_type == 'audio_narration'
        assert conv.lifecycle_status == 'inbox'
        assert conv.title == 'Mein Dialog'              # real title kept verbatim
        assert '**Anna:**' in conv.content              # speaker-labelled markdown
        meta = json.loads(conv.metadata_json)
        assert meta['narration_status'] == 'pending'
        assert meta['tts_model'] == DEFAULT_NARRATION_MODEL
        assert meta['speakers'] == _VOICES
        assert meta['transcript'] == _TURNS
        assert meta['job_id'] == 'test-job-123'         # written back post-enqueue
        assert meta['audio_filename'] == f'narration_{nid}.wav'


def test_create_enqueues_render_task_with_args(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    resp = client.post(NARR_URL, headers=_auth(), json=_payload())
    assert resp.status_code == 202
    nid = resp.get_json()['narration_id']

    enqueue = mock_redis_queue['queue'].enqueue
    enqueue.assert_called_once()
    call = enqueue.call_args
    # positional task args: (task, conversion_id, turns, voices, style_prompt,
    #                        mode, language_code, tts_model)
    assert call.args[0] is generate_narration_task
    assert call.args[1] == nid
    assert call.args[2] == _TURNS
    assert call.args[3] == _VOICES
    assert call.args[4] == 'ruhig'                       # style_prompt
    assert call.args[5] == 'two_speaker'                 # mode
    assert call.args[6] == 'de-DE'                       # language default
    assert call.args[7] == DEFAULT_NARRATION_MODEL       # tts_model default
    # RQ job options
    assert call.kwargs['meta'] == {'user_id': test_user['id'], 'conversion_id': nid}
    assert call.kwargs['job_timeout'] == TIMEOUT_RQ_JOB_SECONDS


def test_create_derives_title_when_blank(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    resp = client.post(NARR_URL, headers=_auth(), json=_payload(title=''))
    assert resp.status_code == 202
    nid = resp.get_json()['narration_id']
    with app.app_context():
        # blank title → derived from content, never persisted empty (NOT NULL).
        assert db.session.get(Conversion, nid).title.strip()


# --- CSRF: exempt only the write view (proven under enforced CSRF) -----------

def test_create_csrf_exempt_under_enforced_csrf(app, client, test_user, monkeypatch, mock_redis_queue):
    monkeypatch.setenv('NARRATION_TOKEN', NARRATION_TOKEN)
    # conftest disables CSRF globally; flip it back ON. A Bearer-only POST has no
    # CSRF token and must still succeed because the view is exempt.
    monkeypatch.setitem(app.config, 'WTF_CSRF_ENABLED', True)
    assert client.post(NARR_URL, headers=_auth(), json=_payload()).status_code == 202


def test_only_narration_write_is_csrf_exempt(app):
    csrf = app.extensions['csrf']
    assert 'app_pkg.narration.api_create_narration' in csrf._exempt_views
    # the session-authed serve route stays under CSRF (read, not a token write)
    assert 'app_pkg.narration.narration_audio' not in csrf._exempt_views
