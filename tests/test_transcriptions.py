"""SYNC-FREEZE P3 — the file transcription as a job on the worker.

Replaces ``tests/test_audio.py`` (the synchronous ``POST /transcribe-audio-file``
is gone). Locks in: submit → pending row + source file on the volume (named
after the job mark, JOB-ID-REUSE) + RQ envelope from the audio duration; the validation 400s and the 503 gate;
idempotency (same file + language → the stored state, nothing enqueued);
the MCP1 ``recorded_at`` precedence at submit; every reconcile branch
(pending / ready from the result file / failed from RQ / unreadable result /
job gone / Redis blip); legacy rows answering as ready; the DB-free worker
task end-to-end against a mocked Deepgram service; and the envelope mirror
pinned to the service's chunking thresholds.

RQ is mocked at the ``task_queue`` / ``Job`` singletons (``mock_redis_queue``),
the Deepgram singleton only gates configured-ness (``mock_deepgram``), the
shared-volume dir is a monkeypatched tmp directory.
"""
import io
import json
import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rq.exceptions import NoSuchJobError

import app_pkg.audio as audio_module
import tasks
from app_pkg.config import (
    AUDIO_CHUNK_OVERLAP_SECONDS,
    AUDIO_CHUNK_SECONDS,
    AUDIO_SINGLE_REQUEST_MAX_SECONDS,
    TIMEOUT_AUDIO_JOB_BASE_SECONDS,
    TIMEOUT_AUDIO_JOB_PER_CHUNK_SECONDS,
    TIMEOUT_DEEPGRAM_SECONDS,
    RQ_SERIALIZER,
    TIMEOUT_RQ_JOB_HARD_CAP,
    audio_chunk_count,
    transcribe_job_timeout_for,
)
from models import Conversion, db
from services import transcription_jobs as tj
from services.deepgram_service import DeepgramService
from tests.test_rq_serializer import assert_rq_json_roundtrip

URL = '/api/transcriptions'


@pytest.fixture
def transcription_dir(tmp_path, monkeypatch):
    d = tmp_path / 'transcriptions'
    monkeypatch.setattr(tj, 'TRANSCRIPTION_DIR', str(d))
    # ffprobe is not guaranteed on the dev box — the duration is a test input.
    monkeypatch.setattr(audio_module, 'probe_duration_seconds', lambda path: 1901.1)
    return d


def _post(client, data=b'RIFF....fake audio', filename='260521_0176.wav',
          language='de', extra=None):
    fields = {'audio_file': (io.BytesIO(data), filename), 'language': language}
    if extra:
        fields.update(extra)
    return client.post(URL, data=fields, content_type='multipart/form-data')


def _row(conversion_id):
    return db.session.get(Conversion, conversion_id)


def _submit(client, **kwargs):
    """Submit and return ``(conversion_id, job_mark)`` — the mark names the
    job's files and is the RQ job id (JOB-ID-REUSE)."""
    body = _post(client, **kwargs).get_json()
    return body['id'], body['job_id']


def _job(is_failed=False, exc_info=None, is_finished=False):
    return SimpleNamespace(is_failed=is_failed, exc_info=exc_info,
                           is_finished=is_finished)


# --- submit -------------------------------------------------------------------

def test_submit_creates_pending_job(app, authenticated_client, test_user,
                                    mock_deepgram, mock_redis_queue, transcription_dir):
    resp = _post(authenticated_client)
    assert resp.status_code == 202
    body = resp.get_json()
    assert body['status'] == 'pending'
    mark = body['job_id']
    assert str(uuid.UUID(mark)) == mark   # web-made, not the queue's own id

    with app.app_context():
        row = _row(body['id'])
        assert row.conversion_type == 'audio_transcription'
        assert row.user_id == test_user['id']
        assert row.title == '260521_0176'
        assert row.content == ''
        assert row.lifecycle_status == 'archive'
        assert row.source_filename == '260521_0176.wav'
        meta = json.loads(row.metadata_json)
    assert meta['transcription_status'] == 'pending'
    assert meta['job_id'] == mark
    assert meta['language'] == 'de'
    assert meta['source_format'] == 'wav'
    assert meta['duration_seconds'] == 1901.1
    assert len(meta['source_sha256']) == 64
    # The source lies under the job mark, never under the row id.
    assert os.listdir(transcription_dir) == [f'source_{mark}.wav']
    assert os.path.exists(tj.transcription_source_path(mark, 'wav'))

    # The unchanged service is NOT called in the web process — the job is.
    mock_deepgram.transcribe_file.assert_not_called()
    call = mock_redis_queue['queue'].enqueue.call_args
    assert call.args[0] is tasks.transcribe_audio_task
    assert call.args[1:] == (mark, 'wav', 'de')
    assert call.kwargs['job_id'] == mark
    assert call.kwargs['job_timeout'] == transcribe_job_timeout_for(1901.1)
    assert call.kwargs['meta'] == {'user_id': test_user['id']}


def test_submit_enqueue_survives_rq_json(authenticated_client, mock_deepgram,
                                         mock_redis_queue, transcription_dir):
    """SEC-REDIS-AUTH: the transcription job args survive the JSON serializer."""
    assert _post(authenticated_client).status_code == 202
    assert_rq_json_roundtrip(mock_redis_queue['queue'].enqueue.call_args)


def test_submit_without_language_defaults_to_german(app, authenticated_client,
                                                    mock_deepgram, mock_redis_queue,
                                                    transcription_dir):
    """TRANS-DE-DEFAULT: a POST that carries no ``language`` field lands as
    ``de``. Oli dictates in German; the server default must agree with the
    module default in audio_converter.js and the ``lang-active`` button —
    this pins the server side, the browser side is a live-smoke check."""
    fields = {'audio_file': (io.BytesIO(b'RIFF....fake audio'), '260521_0176.wav')}
    resp = authenticated_client.post(URL, data=fields, content_type='multipart/form-data')
    assert resp.status_code == 202
    body = resp.get_json()
    with app.app_context():
        meta = json.loads(_row(body['id']).metadata_json)
    assert meta['language'] == 'de'
    call = mock_redis_queue['queue'].enqueue.call_args
    assert call.args[1:] == (body['job_id'], 'wav', 'de')


def test_submit_envelope_scales_with_the_duration(authenticated_client, mock_deepgram,
                                                  mock_redis_queue, transcription_dir,
                                                  monkeypatch):
    monkeypatch.setattr(audio_module, 'probe_duration_seconds', lambda path: 3 * 3600)
    assert _post(authenticated_client).status_code == 202
    call = mock_redis_queue['queue'].enqueue.call_args
    assert call.kwargs['job_timeout'] == transcribe_job_timeout_for(3 * 3600)
    assert call.kwargs['job_timeout'] > transcribe_job_timeout_for(1901.1)


def test_submit_validations(authenticated_client, mock_deepgram, mock_redis_queue,
                            transcription_dir):
    # no file part
    resp = authenticated_client.post(URL, data={'language': 'de'},
                                     content_type='multipart/form-data')
    assert resp.status_code == 400 and 'audio_file' in resp.get_json()['error']
    # unsupported extension — before anything touches the volume or the queue
    resp = _post(authenticated_client, filename='evil.xyz')
    assert resp.status_code == 400
    assert 'MP3, WAV, M4A, OGG, FLAC, WEBM' in resp.get_json()['error']
    # unsupported language (F-013)
    resp = _post(authenticated_client, language='xx-XX')
    assert resp.status_code == 400 and 'Sprache' in resp.get_json()['error']
    # empty file
    resp = _post(authenticated_client, data=b'')
    assert resp.status_code == 400 and resp.get_json()['error'] == 'Leere Datei.'
    mock_redis_queue['queue'].enqueue.assert_not_called()
    assert not os.path.exists(transcription_dir) or not any(
        name.startswith('source_') for name in os.listdir(transcription_dir))


def test_submit_503_when_deepgram_not_configured(authenticated_client, mock_redis_queue,
                                                 transcription_dir):
    """F-011: the shared ``require_service('deepgram')`` gate stays in front
    of the job — a missing key is a 503 with DE microcopy, nothing enqueued."""
    import app as app_module
    original = app_module.deepgram_service
    app_module.deepgram_service = None
    try:
        resp = _post(authenticated_client)
    finally:
        app_module.deepgram_service = original
    assert resp.status_code == 503
    assert 'nicht konfiguriert' in resp.get_json()['error']
    mock_redis_queue['queue'].enqueue.assert_not_called()


def test_submit_requires_login(client, transcription_dir):
    resp = _post(client)
    assert resp.status_code == 302 and '/login' in resp.headers['Location']


def test_sync_route_is_gone(authenticated_client, mock_deepgram):
    resp = authenticated_client.post('/transcribe-audio-file', data={'language': 'de'},
                                     content_type='multipart/form-data')
    assert resp.status_code == 404


# --- idempotency ------------------------------------------------------------------

def test_same_file_and_language_dedups(app, authenticated_client, mock_deepgram,
                                       mock_redis_queue, transcription_dir):
    mock_redis_queue['fetch'].return_value = _job()
    first = _post(authenticated_client).get_json()
    again = _post(authenticated_client)
    assert again.status_code == 200
    body = again.get_json()
    assert body['deduped'] is True and body['id'] == first['id']
    assert body['status'] == 'pending'
    assert mock_redis_queue['queue'].enqueue.call_count == 1

    # a different language is a different job
    other = _post(authenticated_client, language='en')
    assert other.status_code == 202 and other.get_json()['id'] != first['id']
    assert mock_redis_queue['queue'].enqueue.call_count == 2


def test_failed_job_does_not_dedup(app, authenticated_client, mock_deepgram,
                                   mock_redis_queue, transcription_dir):
    """Re-submitting the file IS the retry path."""
    first = _post(authenticated_client).get_json()
    mock_redis_queue['fetch'].return_value = _job(is_failed=True, exc_info='boom')
    polled = authenticated_client.get(f'{URL}/{first["id"]}').get_json()
    assert polled['status'] == 'failed'
    mock_redis_queue['fetch'].return_value = _job()
    again = _post(authenticated_client)
    assert again.status_code == 202 and again.get_json()['id'] != first['id']


# --- recorded_at (MCP1) at submit -------------------------------------------------

def test_recorded_at_filename_beats_client(app, authenticated_client, mock_deepgram,
                                           mock_redis_queue, transcription_dir):
    body = _post(authenticated_client, filename='260521_0176.wav',
                 extra={'recorded_at': '1716300000000'}).get_json()
    with app.app_context():
        meta = json.loads(_row(body['id']).metadata_json)
    assert meta['recorded_at'].startswith('2026-05-21')
    assert meta['recorded_at_source'] == 'filename'


def test_recorded_at_falls_back_to_client_epoch_ms(app, authenticated_client,
                                                   mock_deepgram, mock_redis_queue,
                                                   transcription_dir):
    body = _post(authenticated_client, data=b'ID3 other bytes', filename='Besprechung.mp3',
                 extra={'recorded_at': '1716300000000'}).get_json()
    with app.app_context():
        meta = json.loads(_row(body['id']).metadata_json)
    assert meta['recorded_at'].startswith('2024-05-21')
    assert meta['recorded_at_source'] == 'client'

    body = _post(authenticated_client, data=b'ID3 third bytes', filename='Notiz.mp3',
                 extra={'recorded_at': 'gestern'}).get_json()
    with app.app_context():
        meta = json.loads(_row(body['id']).metadata_json)
    assert 'recorded_at' not in meta


def test_browser_file_time_is_kept_next_to_the_winner(app, authenticated_client, mock_deepgram,
                                                      mock_redis_queue, transcription_dir):
    """NOTION-MEETING-LINK: the filename date beats the browser's
    ``lastModified`` — which was then thrown away. It is now stored as
    ``recorded_at_client`` (additive, no logic reads it) so it becomes
    measurable whether copying off the recorder preserves the recording
    time."""
    body = _post(authenticated_client, filename='260521_0176.wav',
                 extra={'recorded_at': '1716300000000'}).get_json()
    with app.app_context():
        meta = json.loads(_row(body['id']).metadata_json)
    assert meta['recorded_at_source'] == 'filename' and meta['recorded_at'].startswith('2026-05-21')
    assert meta['recorded_at_client'] == '2024-05-21T14:00:00+00:00'

    # the client value wins (no date in the name): both keys carry it
    body = _post(authenticated_client, data=b'ID3 other bytes', filename='Besprechung.mp3',
                 extra={'recorded_at': '1716300000000'}).get_json()
    with app.app_context():
        meta = json.loads(_row(body['id']).metadata_json)
    assert meta['recorded_at'] == meta['recorded_at_client'] == '2024-05-21T14:00:00+00:00'

    # nothing sent / unparseable: the key is absent, never null
    for data, extra in ((b'ID3 third bytes', None), (b'ID3 fourth bytes', {'recorded_at': 'gestern'})):
        body = _post(authenticated_client, data=data, filename='260522_0177.wav', extra=extra).get_json()
        with app.app_context():
            meta = json.loads(_row(body['id']).metadata_json)
        assert 'recorded_at_client' not in meta


# --- poll / reconcile -------------------------------------------------------------

def test_poll_pending_while_the_job_runs(authenticated_client, mock_deepgram,
                                         mock_redis_queue, transcription_dir):
    cid = _post(authenticated_client).get_json()['id']
    mock_redis_queue['fetch'].return_value = _job()
    body = authenticated_client.get(f'{URL}/{cid}').get_json()
    assert body['status'] == 'pending'
    assert body['transcript'] is None
    assert body['metadata']['language'] == 'de'
    assert body['source']['filename'] == '260521_0176.wav'
    assert body['lifecycle_status'] == 'archive'


def test_poll_ready_reads_the_result_file_and_discards_the_files(
        app, authenticated_client, mock_deepgram, mock_redis_queue, transcription_dir):
    cid, mark = _submit(authenticated_client)
    tj.write_result_file(mark, {'transcript': '**Sprecher 1:** Hallo.\n\n**Sprecher 2:** Moin.',
                                'transcript_length': 40, 'file_size_mb': 0.0,
                                'language': 'de'})
    body = authenticated_client.get(f'{URL}/{cid}').get_json()
    assert body['status'] == 'ready'
    assert body['transcript'].startswith('**Sprecher 1:**')
    assert body['metadata']['transcript_length'] == len(body['transcript'])
    assert body['error'] is None
    with app.app_context():
        row = _row(cid)
        assert row.content == body['transcript']
        assert json.loads(row.metadata_json)['transcription_status'] == 'ready'
    assert os.listdir(transcription_dir) == []   # result AND source discarded
    # terminal state is idempotent — no RQ lookup anymore
    mock_redis_queue['fetch'].reset_mock()
    assert authenticated_client.get(f'{URL}/{cid}').get_json()['status'] == 'ready'
    mock_redis_queue['fetch'].assert_not_called()


def test_poll_failed_job_surfaces_the_exc_info_tail(authenticated_client, mock_deepgram,
                                                    mock_redis_queue, transcription_dir):
    cid, mark = _submit(authenticated_client)
    exc_info = 'x' * 3000 + '\nRuntimeError: Transcription failed for chunk 2'
    mock_redis_queue['fetch'].return_value = _job(is_failed=True, exc_info=exc_info)
    body = authenticated_client.get(f'{URL}/{cid}').get_json()
    assert body['status'] == 'failed'
    assert body['error'].endswith('RuntimeError: Transcription failed for chunk 2')
    assert len(body['error']) <= 2000
    assert not os.path.exists(tj.transcription_source_path(mark, 'wav'))


def test_poll_unreadable_result_file_fails(authenticated_client, mock_deepgram,
                                           mock_redis_queue, transcription_dir):
    cid, mark = _submit(authenticated_client)
    with open(tj.transcription_result_path(mark), 'w') as f:
        f.write('{not json')
    body = authenticated_client.get(f'{URL}/{cid}').get_json()
    assert body['status'] == 'failed'
    assert body['error'] == 'Ergebnisdatei unlesbar.'
    assert os.path.exists(tj.transcription_result_path(mark))  # kept for diagnosis


def test_poll_job_gone_fails(authenticated_client, mock_deepgram, mock_redis_queue,
                             transcription_dir):
    cid = _post(authenticated_client).get_json()['id']
    mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')
    body = authenticated_client.get(f'{URL}/{cid}').get_json()
    assert body['status'] == 'failed' and body['error'] == 'Job nicht mehr auffindbar.'


def test_poll_reads_job_with_rq_serializer(authenticated_client, mock_deepgram,
                                           mock_redis_queue, transcription_dir):
    """SEC-REDIS-AUTH: the reconcile reads the job via app.fetch_job (JSON)."""
    cid, mark = _submit(authenticated_client)
    mock_redis_queue['fetch'].return_value.is_failed = False
    assert authenticated_client.get(f'{URL}/{cid}').get_json()['status'] == 'pending'
    fetch = mock_redis_queue['fetch']
    assert fetch.call_args.args == (mark,)
    assert fetch.call_args.kwargs['serializer'] is RQ_SERIALIZER


def test_poll_transient_redis_error_stays_pending(authenticated_client, mock_deepgram,
                                                  mock_redis_queue, transcription_dir):
    cid, mark = _submit(authenticated_client)
    mock_redis_queue['fetch'].side_effect = ConnectionError('redis down')
    body = authenticated_client.get(f'{URL}/{cid}').get_json()
    assert body['status'] == 'pending'
    assert os.path.exists(tj.transcription_source_path(mark, 'wav'))


def test_poll_foreign_missing_or_wrong_type_is_404(app, authenticated_client, test_user,
                                                   mock_deepgram, mock_redis_queue,
                                                   transcription_dir):
    with app.app_context():
        other = Conversion(user_id=test_user['id'], conversion_type='markdown_to_pdf',
                           title='x', content='y')
        db.session.add(other)
        db.session.commit()
        other_id = other.id
    assert authenticated_client.get(f'{URL}/{other_id}').status_code == 404
    assert authenticated_client.get(f'{URL}/999999').status_code == 404


def test_legacy_row_answers_as_ready(app, authenticated_client, test_user, mock_deepgram,
                                     mock_redis_queue, transcription_dir):
    """Rows saved by the pre-P3 synchronous flow carry no job keys."""
    with app.app_context():
        legacy = Conversion(user_id=test_user['id'], conversion_type='audio_transcription',
                            title='Diktat', content='Alter Text.',
                            metadata_json=json.dumps({'language': 'de',
                                                      'transcript_length': 10}))
        db.session.add(legacy)
        db.session.commit()
        legacy_id = legacy.id
    body = authenticated_client.get(f'{URL}/{legacy_id}').get_json()
    assert body['status'] == 'ready' and body['transcript'] == 'Alter Text.'
    mock_redis_queue['fetch'].assert_not_called()


# --- the worker task (DB-free) ----------------------------------------------------

class _FakeDeepgram:
    calls = []

    def __init__(self, api_key):
        _FakeDeepgram.calls.append(('init', api_key))

    def transcribe_file(self, audio_data, language):
        _FakeDeepgram.calls.append(('transcribe', len(audio_data), language))
        return 'Hallo Welt.'


def test_worker_task_writes_result_and_removes_the_source(transcription_dir, monkeypatch):
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'worker-key')
    monkeypatch.setattr(tasks, 'DeepgramService', _FakeDeepgram)
    _FakeDeepgram.calls = []
    tj.ensure_transcription_dir()
    with open(tj.transcription_source_path('job-7', 'mp3'), 'wb') as f:
        f.write(b'abc')

    path = tasks.transcribe_audio_task('job-7', 'mp3', 'de')

    assert path == tj.transcription_result_path('job-7')
    assert os.listdir(transcription_dir) == ['result_job-7.json']
    payload = tj.read_result_file('job-7')
    assert payload == {'transcript': 'Hallo Welt.', 'transcript_length': 11,
                       'file_size_mb': 0.0, 'language': 'de'}
    assert not os.path.exists(tj.transcription_source_path('job-7', 'mp3'))
    # one service PER JOB, built with the worker's key — never a shared one
    assert _FakeDeepgram.calls == [('init', 'worker-key'), ('transcribe', 3, 'de')]


def test_worker_task_failure_reraises_and_removes_the_source(transcription_dir, monkeypatch):
    monkeypatch.setenv('DEEPGRAM_API_KEY', 'worker-key')
    broken = MagicMock()
    broken.return_value.transcribe_file.side_effect = RuntimeError('chunk 2 failed')
    monkeypatch.setattr(tasks, 'DeepgramService', broken)
    tj.ensure_transcription_dir()
    with open(tj.transcription_source_path('job-8', 'wav'), 'wb') as f:
        f.write(b'abc')
    with pytest.raises(RuntimeError, match='chunk 2 failed'):
        tasks.transcribe_audio_task('job-8', 'wav', 'de')
    assert not os.path.exists(tj.transcription_result_path('job-8'))
    assert not os.path.exists(tj.transcription_source_path('job-8', 'wav'))


def test_worker_task_without_key_fails_loudly(transcription_dir, monkeypatch):
    monkeypatch.delenv('DEEPGRAM_API_KEY', raising=False)
    with pytest.raises(ValueError, match='DEEPGRAM_API_KEY'):
        tasks.transcribe_audio_task('job-9', 'wav', 'de')


# --- the RQ envelope --------------------------------------------------------------

def test_envelope_mirrors_the_service_chunking_thresholds():
    """config stays SDK-free, so it MIRRORS the service's thresholds — a
    change on either side must show up here, not as a false-killed job."""
    assert AUDIO_SINGLE_REQUEST_MAX_SECONDS == DeepgramService.MAX_AUDIO_DURATION_SECONDS
    assert AUDIO_CHUNK_SECONDS == DeepgramService.CHUNK_DURATION_SECONDS
    assert AUDIO_CHUNK_OVERLAP_SECONDS == DeepgramService.OVERLAP_SECONDS
    assert TIMEOUT_AUDIO_JOB_PER_CHUNK_SECONDS == (
        (DeepgramService.MAX_RETRIES + 1) * TIMEOUT_DEEPGRAM_SECONDS + 2 + 4)


def test_envelope_from_the_audio_duration():
    assert audio_chunk_count(None) == 1
    assert audio_chunk_count('junk') == 1
    assert audio_chunk_count(0) == 1
    assert audio_chunk_count(AUDIO_SINGLE_REQUEST_MAX_SECONDS) == 1
    # 2 h: ceil(7200 / 1795) = 5 — the AudioChunker estimate
    assert audio_chunk_count(7200) == 5
    single = TIMEOUT_AUDIO_JOB_BASE_SECONDS + TIMEOUT_AUDIO_JOB_PER_CHUNK_SECONDS
    assert transcribe_job_timeout_for(None) == single
    assert transcribe_job_timeout_for(1901.1) == single
    # Chunking starts at FOUR chunks (5401 s / 1795 s → 4): 300 + 4 × 3606 is
    # already above the shared 4-h cap, so every chunked file rides the cap.
    assert audio_chunk_count(5401) == 4
    assert single + TIMEOUT_AUDIO_JOB_PER_CHUNK_SECONDS * 3 > TIMEOUT_RQ_JOB_HARD_CAP
    assert transcribe_job_timeout_for(5401) == TIMEOUT_RQ_JOB_HARD_CAP
    assert transcribe_job_timeout_for(7200) == TIMEOUT_RQ_JOB_HARD_CAP
    # a per-request deadline of 1200 s is never tighter than one chunk's worst case
    assert TIMEOUT_AUDIO_JOB_PER_CHUNK_SECONDS >= 3 * TIMEOUT_DEEPGRAM_SECONDS
