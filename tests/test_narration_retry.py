"""NARR-5 — POST /api/narrations/<id>/retry (session re-enqueue of a failed render).

A session-authed, CSRF-protected endpoint Oli clicks in the library-detail UI to
re-run a failed narration from its **stored** render inputs (transcript /
speakers / mode / style_prompt / language_code / tts_model) — no new agent call,
no token. Only ``failed`` narrations are retryable (409 otherwise). RQ is mocked
at the ``task_queue`` / ``Job`` singletons (``mock_redis_queue``); the enqueue is
captured to assert the task + args without running a worker.
"""
import json

from models import Conversion, User, db
from services.narration_render import DEFAULT_NARRATION_MODEL
from tasks import generate_narration_task


RETRY_URL = '/api/narrations/{}/retry'

_TURNS = [
    {'speaker': 'Anna', 'text': 'Hallo zusammen.'},
    {'speaker': 'Ben', 'text': 'Schön, hier zu sein.'},
]
_VOICES = {'Anna': 'Kore', 'Ben': 'Puck'}


def _make_narration(app, user_id, *, status='failed', ctype='audio_narration',
                    metadata_extra=None):
    """Create a synthetic narration Conversion with full render-input metadata."""
    metadata = {
        'narration_status': status,
        'audio_filename': 'narration_PLACEHOLDER.wav',
        'audio_mimetype': 'audio/wav',
        'duration_seconds': None,
        'tts_model': 'gemini-2.5-flash-tts',
        'speakers': dict(_VOICES),
        'transcript': [dict(t) for t in _TURNS],
        'mode': 'two_speaker',
        'style_prompt': 'ruhig',
        'language_code': 'de-DE',
        'error': 'Vertonung fehlgeschlagen.',
        'job_id': 'old-job-1',
    }
    if metadata_extra is not None:
        metadata.update(metadata_extra)
    with app.app_context():
        conv = Conversion(
            user_id=user_id,
            conversion_type=ctype,
            title='Test-Vertonung',
            content='**Anna:** Hallo zusammen.\n\n**Ben:** Schön, hier zu sein.',
            metadata_json=json.dumps(metadata),
        )
        db.session.add(conv)
        db.session.commit()
        return conv.id


def _second_user(app, username='mallory'):
    with app.app_context():
        u = User(username=username)
        u.set_password('hunter2hunter2')
        db.session.add(u)
        db.session.commit()
        return u.id


def _stored_meta(app, cid):
    with app.app_context():
        return json.loads(db.session.get(Conversion, cid).metadata_json)


# --- happy path: re-enqueue from stored metadata -----------------------------

def test_retry_failed_reenqueues_from_metadata(
        authenticated_client, app, test_user, mock_redis_queue):
    cid = _make_narration(app, test_user['id'], status='failed')

    resp = authenticated_client.post(RETRY_URL.format(cid))
    assert resp.status_code == 202
    body = resp.get_json()
    assert body['status'] == 'pending'
    assert body['job_id'] == 'test-job-123'  # from the mock job

    enqueue = mock_redis_queue['queue'].enqueue
    enqueue.assert_called_once()
    call = enqueue.call_args
    # positional: (task, conversion_id, turns, voices, style_prompt, mode,
    #              language_code, tts_model) — exactly the worker's signature.
    assert call.args[0] is generate_narration_task
    assert call.args[1] == cid
    assert call.args[2] == _TURNS
    assert call.args[3] == _VOICES
    assert call.args[4] == 'ruhig'                 # style_prompt
    assert call.args[5] == 'two_speaker'           # mode
    assert call.args[6] == 'de-DE'                 # language_code
    assert call.args[7] == 'gemini-2.5-flash-tts'  # tts_model
    assert call.kwargs['meta'] == {'user_id': test_user['id'], 'conversion_id': cid}


def test_retry_resets_metadata_to_pending(
        authenticated_client, app, test_user, mock_redis_queue):
    cid = _make_narration(app, test_user['id'], status='failed')

    assert authenticated_client.post(RETRY_URL.format(cid)).status_code == 202

    meta = _stored_meta(app, cid)
    assert meta['narration_status'] == 'pending'
    assert meta['error'] is None
    assert meta['duration_seconds'] is None
    assert meta['job_id'] == 'test-job-123'        # new job, not the old one
    # render inputs untouched (faithful re-run)
    assert meta['transcript'] == _TURNS
    assert meta['speakers'] == _VOICES


# --- gate: only failed is retryable ------------------------------------------

def test_retry_pending_409(authenticated_client, app, test_user, mock_redis_queue):
    cid = _make_narration(app, test_user['id'], status='pending')
    resp = authenticated_client.post(RETRY_URL.format(cid))
    assert resp.status_code == 409
    mock_redis_queue['queue'].enqueue.assert_not_called()


def test_retry_ready_409(authenticated_client, app, test_user, mock_redis_queue):
    cid = _make_narration(app, test_user['id'], status='ready')
    assert authenticated_client.post(RETRY_URL.format(cid)).status_code == 409
    mock_redis_queue['queue'].enqueue.assert_not_called()


# --- owner / type isolation (no type leak) -----------------------------------

def test_retry_foreign_404(authenticated_client, app, mock_redis_queue):
    other = _second_user(app)
    cid = _make_narration(app, other, status='failed')
    assert authenticated_client.post(RETRY_URL.format(cid)).status_code == 404
    mock_redis_queue['queue'].enqueue.assert_not_called()


def test_retry_wrong_type_404(authenticated_client, app, test_user, mock_redis_queue):
    cid = _make_narration(app, test_user['id'], status='failed', ctype='markdown_input')
    assert authenticated_client.post(RETRY_URL.format(cid)).status_code == 404
    mock_redis_queue['queue'].enqueue.assert_not_called()


def test_retry_requires_login(client, app, test_user, mock_redis_queue):
    cid = _make_narration(app, test_user['id'], status='failed')
    # anonymous → login redirect (302), never enqueues
    assert client.post(RETRY_URL.format(cid)).status_code in (302, 401)
    mock_redis_queue['queue'].enqueue.assert_not_called()


# --- fallback for pre-NARR-5 rows (no stored mode/style/language) -------------

def test_retry_derives_mode_when_absent(
        authenticated_client, app, test_user, mock_redis_queue):
    # An older narration stored neither mode nor style nor language.
    cid = _make_narration(
        app, test_user['id'], status='failed',
        metadata_extra={'mode': None, 'style_prompt': None, 'language_code': None})

    assert authenticated_client.post(RETRY_URL.format(cid)).status_code == 202

    call = mock_redis_queue['queue'].enqueue.call_args
    assert call.args[4] is None              # style_prompt → None
    assert call.args[5] == 'two_speaker'     # mode derived from 2 voices
    assert call.args[6] == 'de-DE'           # language → default
    assert call.args[7] == DEFAULT_NARRATION_MODEL  # tts_model present, untouched


def test_retry_derives_single_speaker_mode(
        authenticated_client, app, test_user, mock_redis_queue):
    cid = _make_narration(
        app, test_user['id'], status='failed',
        metadata_extra={'mode': None, 'speakers': {'Anna': 'Kore'}})

    assert authenticated_client.post(RETRY_URL.format(cid)).status_code == 202
    assert mock_redis_queue['queue'].enqueue.call_args.args[5] == 'single_speaker'


# --- CSRF: retry is session-authed, NOT exempt -------------------------------

def test_retry_not_csrf_exempt(app):
    csrf = app.extensions['csrf']
    assert 'app_pkg.narration.api_narration_retry' not in csrf._exempt_views
