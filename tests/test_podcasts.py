"""Podcast generation + status-polling characterization tests.

Locks in:
- POST /generate-gemini-podcast enqueues a job and returns ``{"job_id": ...}``.
- GET /podcast-status/<id> reports the RQ job state machine
  (queued/started → processing, finished → completed+result, failed → failed+error).
- GET /podcast-status/<id> returns 404 when ``Job.fetch`` raises
  ``NoSuchJobError``.
- GET /podcast-status/<id> returns 500 when ``Job.fetch`` raises any other
  exception — this is the F-001 characterization (Stage 4 narrowed
  ``except Exception`` so transport/auth errors no longer masquerade as 404).

RQ is mocked via the ``mock_redis_queue`` fixture: ``app.task_queue`` becomes
a MagicMock, and ``rq.job.Job.fetch`` is patched on the class.
"""
import pytest
from unittest.mock import MagicMock

from rq.exceptions import NoSuchJobError


def _job_with(status, *, user_id, result=None, exc_info=None):
    """Build a MagicMock that mimics an ``rq.job.Job`` with ``status``."""
    job = MagicMock()
    job.get_status.return_value = status
    job.meta = {'user_id': user_id}
    job.result = result
    job.exc_info = exc_info
    job.is_finished = (status == 'finished')
    return job


def test_generate_gemini_podcast_enqueues_job(
    authenticated_client, mock_redis_queue, gemini_api_key_set
):
    resp = authenticated_client.post(
        '/generate-gemini-podcast',
        json={'dialogue': [{'speaker': 'A', 'text': 'hi'}], 'language': 'en'},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['job_id'] == 'test-job-123'
    assert body['status'] == 'queued'
    mock_redis_queue['queue'].enqueue.assert_called_once()


def test_generate_gemini_podcast_503_without_api_key(authenticated_client):
    import app as app_module
    original = app_module.GEMINI_API_KEY
    app_module.GEMINI_API_KEY = None
    try:
        resp = authenticated_client.post(
            '/generate-gemini-podcast',
            json={'dialogue': []},
        )
    finally:
        app_module.GEMINI_API_KEY = original
    assert resp.status_code == 503


def test_generate_gemini_podcast_503_de_microcopy(authenticated_client):
    """F-011: the shared ``require_service('gemini')`` decorator returns
    a DE-microcopy JSON body when GEMINI_API_KEY is unset."""
    import app as app_module
    original = app_module.GEMINI_API_KEY
    app_module.GEMINI_API_KEY = None
    try:
        resp = authenticated_client.post(
            '/generate-gemini-podcast',
            json={'dialogue': []},
        )
    finally:
        app_module.GEMINI_API_KEY = original
    assert resp.status_code == 503
    body = resp.get_json()
    assert 'Gemini-API-Key' in body['error']
    assert 'nicht konfiguriert' in body['error']


def test_generate_podcast_503_when_google_tts_not_configured(authenticated_client):
    """F-011: ``require_service('google_tts')`` returns 503 + DE-JSON when
    ``google_tts_service`` is None (no Google credentials)."""
    import app as app_module
    original = app_module.google_tts_service
    app_module.google_tts_service = None
    try:
        resp = authenticated_client.post(
            '/generate-podcast',
            json={'text': 'hello world'},
        )
    finally:
        app_module.google_tts_service = original
    assert resp.status_code == 503
    body = resp.get_json()
    assert 'Google Cloud TTS' in body['error']
    assert 'nicht konfiguriert' in body['error']


def test_podcast_status_queued_reports_processing(
    authenticated_client, mock_redis_queue, test_user
):
    mock_redis_queue['fetch'].return_value = _job_with('queued', user_id=test_user['id'])
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'processing'}


def test_podcast_status_started_reports_processing(
    authenticated_client, mock_redis_queue, test_user
):
    mock_redis_queue['fetch'].return_value = _job_with('started', user_id=test_user['id'])
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'processing'}


def test_podcast_status_finished_reports_completed_with_result(
    authenticated_client, mock_redis_queue, test_user
):
    mock_redis_queue['fetch'].return_value = _job_with(
        'finished', user_id=test_user['id'], result='/app/output_podcasts/x.wav'
    )
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['status'] == 'completed'
    assert body['result'] == '/app/output_podcasts/x.wav'


def test_podcast_status_failed_reports_failed_with_error(
    authenticated_client, mock_redis_queue, test_user
):
    mock_redis_queue['fetch'].return_value = _job_with(
        'failed', user_id=test_user['id'], exc_info='Traceback: boom'
    )
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['status'] == 'failed'
    assert 'boom' in body['error']


def test_podcast_status_other_user_job_returns_404(
    authenticated_client, mock_redis_queue, test_user
):
    """A job that exists but belongs to a different user is reported as 404."""
    mock_redis_queue['fetch'].return_value = _job_with('finished', user_id=test_user['id'] + 999)
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 404


def test_podcast_status_no_such_job_returns_404(
    authenticated_client, mock_redis_queue
):
    """``NoSuchJobError`` from RQ → 404 (the legitimate not-found case)."""
    mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 404
    assert resp.get_json()['error'] == 'Job not found'


def test_podcast_status_other_exception_returns_500(
    authenticated_client, mock_redis_queue
):
    """F-001 characterization: any non-NoSuchJobError exception (e.g. Redis
    connection refused, auth failure, payload deserialisation) must surface
    as 500 — *not* be silently masked as 404 like the original code did.
    """
    mock_redis_queue['fetch'].side_effect = ConnectionError('redis down')
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 500
    assert resp.get_json()['error'] == 'Job lookup failed'


def test_podcast_download_no_such_job_returns_404(
    authenticated_client, mock_redis_queue
):
    mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')
    resp = authenticated_client.get('/podcast-download/job-x')
    assert resp.status_code == 404


def test_podcast_download_other_exception_returns_500(
    authenticated_client, mock_redis_queue
):
    """F-001 characterization for the download route (sister of /podcast-status)."""
    mock_redis_queue['fetch'].side_effect = TimeoutError('redis timeout')
    resp = authenticated_client.get('/podcast-download/job-x')
    assert resp.status_code == 500


def test_format_dialogue_invalid_narration_style_returns_400(
    authenticated_client, mock_gemini, gemini_api_key_set
):
    """F-013: ``narration_style`` outside STYLE_DIRECTIVES gets a clean 400
    instead of silently falling back to the conversational default in the
    Gemini service. Same shape covers ``script_length`` and ``language``."""
    resp = authenticated_client.post(
        '/format-dialogue-with-llm',
        json={
            'raw_text': 'some content here',
            'narration_style': 'shakespearean',
            'script_length': 'medium',
            'language': 'en',
            'num_speakers': 2,
        },
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert 'Erzählstil' in body['error']
    mock_gemini.format_dialogue_with_llm.assert_not_called()


def test_generate_podcast_invalid_speaking_rate_returns_400(
    authenticated_client, mock_google_tts
):
    """F-013: ``speaking_rate`` out of Google-TTS range (0.25–4.0) gets a 400
    instead of letting the SDK raise + surface as a 500."""
    resp = authenticated_client.post(
        '/generate-podcast',
        json={
            'text': 'hello world',
            'speaking_rate': 99.0,
            'pitch': 0.0,
        },
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert 'Sprechgeschwindigkeit' in body['error']
    mock_google_tts.synthesize_speech.assert_not_called()


def test_podcast_status_stopped_reports_cancelled(
    authenticated_client, mock_redis_queue, test_user
):
    """rq 2.x send_stop_job_command surfaces as ``stopped`` once the worker
    has SIGKILLed the horse — frontend should see a clean ``cancelled``
    terminal so it can show the abort banner instead of a generic failure."""
    mock_redis_queue['fetch'].return_value = _job_with('stopped', user_id=test_user['id'])
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'cancelled'}


def test_podcast_status_failed_with_cancelled_meta_reports_cancelled(
    authenticated_client, mock_redis_queue, test_user
):
    """Belt-and-suspenders: even if rq surfaces ``failed`` (older rq path /
    edge case), the ``cancelled_by_user`` meta flag set by /podcast-cancel
    still collapses the terminal to ``cancelled`` for the frontend."""
    job = _job_with('failed', user_id=test_user['id'], exc_info='SIGKILL')
    job.meta = {'user_id': test_user['id'], 'cancelled_by_user': True}
    mock_redis_queue['fetch'].return_value = job
    resp = authenticated_client.get('/podcast-status/job-x')
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'cancelled'}


def test_podcast_cancel_started_sends_stop_command(
    authenticated_client, mock_redis_queue, test_user, monkeypatch
):
    """For a started job the cancel endpoint must invoke rq's
    send_stop_job_command (which SIGKILLs the work-horse via PubSub) — not
    Job.cancel(), which only flips status for queued jobs."""
    job = _job_with('started', user_id=test_user['id'])
    mock_redis_queue['fetch'].return_value = job

    stop_calls = []
    monkeypatch.setattr(
        'app_pkg.podcasts.send_stop_job_command',
        lambda conn, jid: stop_calls.append((conn, jid)),
    )

    resp = authenticated_client.post('/podcast-cancel/job-x')
    assert resp.status_code == 202
    assert resp.get_json()['status'] == 'cancelling'
    assert len(stop_calls) == 1
    assert stop_calls[0][1] == 'job-x'
    job.cancel.assert_not_called()
    assert job.meta.get('cancelled_by_user') is True
    job.save_meta.assert_called_once()


def test_podcast_cancel_queued_calls_job_cancel(
    authenticated_client, mock_redis_queue, test_user, monkeypatch
):
    """For a queued job, Job.cancel() removes it from the queue — no horse
    to SIGKILL yet, so send_stop_job_command must NOT be called."""
    job = _job_with('queued', user_id=test_user['id'])
    mock_redis_queue['fetch'].return_value = job

    stop_calls = []
    monkeypatch.setattr(
        'app_pkg.podcasts.send_stop_job_command',
        lambda conn, jid: stop_calls.append((conn, jid)),
    )

    resp = authenticated_client.post('/podcast-cancel/job-x')
    assert resp.status_code == 202
    job.cancel.assert_called_once()
    assert stop_calls == []


def test_podcast_cancel_other_user_returns_404(
    authenticated_client, mock_redis_queue, test_user
):
    """Ownership check must mirror /podcast-status: foreign job → 404."""
    mock_redis_queue['fetch'].return_value = _job_with('started', user_id=test_user['id'] + 999)
    resp = authenticated_client.post('/podcast-cancel/job-x')
    assert resp.status_code == 404


def test_podcast_cancel_finished_returns_already_finished(
    authenticated_client, mock_redis_queue, test_user, monkeypatch
):
    """If the job finished before the cancel landed, do not delete the
    output file — return a clean already_finished marker for the frontend."""
    job = _job_with('finished', user_id=test_user['id'], result='/app/output_podcasts/job-x.wav')
    mock_redis_queue['fetch'].return_value = job

    stop_calls = []
    monkeypatch.setattr(
        'app_pkg.podcasts.send_stop_job_command',
        lambda conn, jid: stop_calls.append((conn, jid)),
    )

    resp = authenticated_client.post('/podcast-cancel/job-x')
    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'already_finished'
    assert stop_calls == []
    job.cancel.assert_not_called()


def test_podcast_cancel_started_unlinks_orphan_wav(
    authenticated_client, mock_redis_queue, test_user, monkeypatch, tmp_path
):
    """Cancel must remove an orphan ``{job_id}.wav`` if the worker already
    moved a partial output into OUTPUT_DIR before SIGKILL landed."""
    monkeypatch.setattr('app_pkg.podcasts.OUTPUT_DIR', str(tmp_path))
    orphan = tmp_path / 'job-x.wav'
    orphan.write_bytes(b'partial wav bytes')

    job = _job_with('started', user_id=test_user['id'])
    mock_redis_queue['fetch'].return_value = job
    monkeypatch.setattr('app_pkg.podcasts.send_stop_job_command', lambda conn, jid: None)

    resp = authenticated_client.post('/podcast-cancel/job-x')
    assert resp.status_code == 202
    assert not orphan.exists()


def test_podcast_cancel_no_such_job_returns_404(
    authenticated_client, mock_redis_queue
):
    mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')
    resp = authenticated_client.post('/podcast-cancel/job-x')
    assert resp.status_code == 404


def test_podcast_cancel_other_exception_returns_500(
    authenticated_client, mock_redis_queue
):
    """F-001 narrow-except parity with /podcast-status: non-NoSuchJobError
    exceptions surface as 500, not silently as 404."""
    mock_redis_queue['fetch'].side_effect = ConnectionError('redis down')
    resp = authenticated_client.post('/podcast-cancel/job-x')
    assert resp.status_code == 500


def test_podcast_download_path_traversal_rejected(
    authenticated_client, mock_redis_queue, test_user, tmp_path, monkeypatch
):
    """F-005: a job.result outside OUTPUT_DIR must be rejected with 403,
    even when the path passes os.path.exists. Guards against prefix-collision
    that the old str.startswith check would have allowed
    (e.g. /app/output_podcasts2/x.wav matching /app/output_podcasts).
    """
    # Create a real file outside OUTPUT_DIR so os.path.exists passes.
    rogue_file = tmp_path / 'rogue.wav'
    rogue_file.write_bytes(b'fake audio')

    mock_redis_queue['fetch'].return_value = _job_with(
        'finished', user_id=test_user['id'], result=str(rogue_file)
    )

    resp = authenticated_client.get('/podcast-download/job-x')
    assert resp.status_code == 403
    assert 'außerhalb' in resp.get_json()['error']
