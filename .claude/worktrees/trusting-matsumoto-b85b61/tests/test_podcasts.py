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
