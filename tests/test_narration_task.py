"""NARR-3 Phase 1 — DB-free worker task + web-side reconcile.

Two surfaces, both mocked at the established boundaries (no real Gemini /
Cloud-TTS call, no real Redis):

* ``generate_narration_task`` (tasks.py) — the worker renders via a mocked
  ``GoogleTTSService.synthesize_narration`` and moves the temp WAV onto the
  id-derived path ``narration_<id>.wav``. It is **DB-free** by design (Option B):
  it never touches the Conversion.
* ``reconcile_narration`` (app_pkg/narration.py) — the web side flips a pending
  narration to ready/failed from the file's existence + the RQ job's terminal
  state. The job is a MagicMock; ``Job.fetch`` is patched on app.py via the
  ``mock_redis_queue`` fixture.

``OUTPUT_DIR`` is monkeypatched in ``services.narration_library`` so the
id-derived ``narration_audio_path`` resolves into a tmp dir (no container path).
"""
import json
import wave
from unittest.mock import MagicMock

import pytest
from rq.exceptions import NoSuchJobError

import tasks
from app_pkg.narration import reconcile_narration
from models import Conversion, db
from services.narration_library import build_narration_metadata


_TURNS = [
    {'speaker': 'Anna', 'text': 'Hallo zusammen.'},
    {'speaker': 'Ben', 'text': 'Schön, hier zu sein.'},
]
_VOICES = {'Anna': 'Kore', 'Ben': 'Puck'}


def _write_real_wav(path, seconds=1, rate=24000):
    """A valid mono / 16-bit WAV of ``seconds`` length so _wav_duration reads it."""
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b'\x00\x00' * (rate * seconds))


# --- worker task: generate_narration_task (DB-free) --------------------------

def test_generate_narration_task_writes_id_derived_wav(tmp_path, monkeypatch):
    """Renderer returns a temp WAV → task moves it to ``narration_<id>.wav``."""
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', '/fake/creds.json')
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(tmp_path))

    temp_wav = tmp_path / 'render-tmp.wav'
    _write_real_wav(temp_wav)

    fake_svc = MagicMock()
    fake_svc.synthesize_narration.return_value = str(temp_wav)
    fake_cls = MagicMock(return_value=fake_svc)
    monkeypatch.setattr(tasks, 'GoogleTTSService', fake_cls)

    result = tasks.generate_narration_task(
        42, _TURNS, _VOICES, 'ruhig', 'two_speaker', 'de-DE', 'gemini-2.5-flash-tts')

    final = tmp_path / 'narration_42.wav'
    assert result == str(final)
    assert final.exists()
    assert not temp_wav.exists()  # moved, not copied
    # creds + render args threaded through verbatim
    fake_cls.assert_called_once_with('/fake/creds.json')
    fake_svc.synthesize_narration.assert_called_once_with(
        _TURNS, _VOICES, style_prompt='ruhig', mode='two_speaker',
        language_code='de-DE', model_name='gemini-2.5-flash-tts')


def test_generate_narration_task_missing_creds_raises(monkeypatch):
    """No GOOGLE_APPLICATION_CREDENTIALS → ValueError before any render."""
    monkeypatch.delenv('GOOGLE_APPLICATION_CREDENTIALS', raising=False)
    fake_cls = MagicMock()
    monkeypatch.setattr(tasks, 'GoogleTTSService', fake_cls)

    with pytest.raises(ValueError):
        tasks.generate_narration_task(
            1, _TURNS, _VOICES, None, 'two_speaker', 'de-DE', 'm')
    fake_cls.assert_not_called()  # bailed before instantiating the service


def test_generate_narration_task_renderer_raises_propagates(tmp_path, monkeypatch):
    """A renderer exception propagates (→ RQ marks the job failed); no file left."""
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', '/fake/creds.json')
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(tmp_path))

    fake_svc = MagicMock()
    fake_svc.synthesize_narration.side_effect = RuntimeError("TTS boom")
    monkeypatch.setattr(tasks, 'GoogleTTSService', MagicMock(return_value=fake_svc))

    with pytest.raises(RuntimeError, match="TTS boom"):
        tasks.generate_narration_task(
            7, _TURNS, _VOICES, None, 'two_speaker', 'de-DE', 'm')
    assert not (tmp_path / 'narration_7.wav').exists()  # nothing written on failure


# --- web-side reconcile: reconcile_narration ---------------------------------

def _make_pending(app, user_id, *, job_id='job-xyz'):
    """Create a synthetic ``pending`` audio_narration Conversion; return its id."""
    with app.app_context():
        conv = Conversion(
            user_id=user_id, conversion_type='audio_narration',
            title='Vertonung', content='**Anna:** Hallo.', metadata_json='{}')
        db.session.add(conv)
        db.session.flush()  # id needed for the id-derived audio_filename
        meta = build_narration_metadata(
            conv.id, status='pending', tts_model='gemini-2.5-flash-tts',
            speakers=_VOICES, transcript=_TURNS)
        if job_id is not None:
            meta['job_id'] = job_id
        conv.metadata_json = json.dumps(meta)
        db.session.commit()
        return conv.id


def _meta(app, cid):
    with app.app_context():
        return json.loads(db.session.get(Conversion, cid).metadata_json)


def test_reconcile_file_exists_flips_ready_with_duration(app, test_user, tmp_path, monkeypatch):
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(tmp_path))
    cid = _make_pending(app, test_user['id'])
    _write_real_wav(tmp_path / f'narration_{cid}.wav', seconds=2)

    with app.app_context():
        reconcile_narration(db.session.get(Conversion, cid))

    meta = _meta(app, cid)
    assert meta['narration_status'] == 'ready'
    assert meta['duration_seconds'] == 2   # frames / framerate
    assert meta['error'] is None


def test_reconcile_job_failed_flips_failed(app, test_user, tmp_path, monkeypatch, mock_redis_queue):
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(tmp_path))
    cid = _make_pending(app, test_user['id'], job_id='job-fail')  # no file on disk

    failed_job = MagicMock()
    failed_job.is_failed = True
    failed_job.exc_info = 'Traceback (most recent call last):\nValueError: render exploded'
    mock_redis_queue['fetch'].return_value = failed_job

    with app.app_context():
        reconcile_narration(db.session.get(Conversion, cid))

    meta = _meta(app, cid)
    assert meta['narration_status'] == 'failed'
    assert 'render exploded' in meta['error']


def test_reconcile_job_gone_flips_failed(app, test_user, tmp_path, monkeypatch, mock_redis_queue):
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(tmp_path))
    cid = _make_pending(app, test_user['id'], job_id='job-gone')
    mock_redis_queue['fetch'].side_effect = NoSuchJobError('gone')

    with app.app_context():
        reconcile_narration(db.session.get(Conversion, cid))

    meta = _meta(app, cid)
    assert meta['narration_status'] == 'failed'
    assert meta['error'] == 'Job nicht mehr auffindbar.'


def test_reconcile_no_job_id_flips_failed(app, test_user, tmp_path, monkeypatch):
    """A pending row with no file AND no recorded job_id is unrecoverable."""
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(tmp_path))
    cid = _make_pending(app, test_user['id'], job_id=None)

    with app.app_context():
        reconcile_narration(db.session.get(Conversion, cid))

    meta = _meta(app, cid)
    assert meta['narration_status'] == 'failed'
    assert meta['error'] == 'Job nicht mehr auffindbar.'


def test_reconcile_job_running_stays_pending(app, test_user, tmp_path, monkeypatch, mock_redis_queue):
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(tmp_path))
    cid = _make_pending(app, test_user['id'], job_id='job-run')

    running_job = MagicMock()
    running_job.is_failed = False
    mock_redis_queue['fetch'].return_value = running_job

    with app.app_context():
        reconcile_narration(db.session.get(Conversion, cid))

    assert _meta(app, cid)['narration_status'] == 'pending'  # unchanged


def test_reconcile_terminal_state_is_idempotent(app, test_user, tmp_path, monkeypatch, mock_redis_queue):
    """A ready narration is not re-reconciled — duration is NOT recomputed and
    the RQ job is never fetched, even with a file present."""
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(tmp_path))
    cid = _make_pending(app, test_user['id'])
    with app.app_context():
        conv = db.session.get(Conversion, cid)
        meta = json.loads(conv.metadata_json)
        meta['narration_status'] = 'ready'
        meta['duration_seconds'] = 99
        conv.metadata_json = json.dumps(meta)
        db.session.commit()
    _write_real_wav(tmp_path / f'narration_{cid}.wav', seconds=2)  # would yield 2

    with app.app_context():
        reconcile_narration(db.session.get(Conversion, cid))

    meta = _meta(app, cid)
    assert meta['narration_status'] == 'ready'
    assert meta['duration_seconds'] == 99   # untouched → no re-reconcile
    mock_redis_queue['fetch'].assert_not_called()
