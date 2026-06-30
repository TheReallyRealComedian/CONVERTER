"""NARR-2 serve + delete-cleanup + NARR-3 status/serve-reconcile tests.

Exercises ``GET /api/narrations/<id>`` (status poll + reconcile),
``GET /api/narrations/<id>/audio`` (serve + reconcile-before-gate), and the
audio-file cleanup hook in ``api_delete_conversion`` through the HTTP boundary,
against synthetic ``audio_narration`` Conversions + WAV files. ``OUTPUT_DIR`` is
monkeypatched to a tmp dir in both the serve module and the persistence helper so
no container path is touched.

The WAVs are written by hand: a headerless ``_DUMMY_WAV`` where only existence
matters, and a real ``_write_real_wav`` where ``reconcile`` must read a duration.
"""
import json
import os
import wave
from unittest.mock import MagicMock

import pytest

from models import Conversion, User, db


_DUMMY_WAV = b'RIFF\x24\x00\x00\x00WAVEfmt dummy-audio-bytes'


@pytest.fixture
def narration_output_dir(tmp_path, monkeypatch):
    """Point the narration audio path + traversal guard at a writable tmp dir."""
    d = tmp_path / 'narrations'
    d.mkdir()
    monkeypatch.setattr('services.narration_library.OUTPUT_DIR', str(d))
    monkeypatch.setattr('app_pkg.narration.OUTPUT_DIR', str(d))
    return str(d)


def _make_narration(app, user_id, *, status='ready', ctype='audio_narration', job_id=None):
    """Create a synthetic audio_narration Conversion; return its id."""
    metadata = {
        'narration_status': status,
        'audio_filename': 'narration_PLACEHOLDER.wav',
        'audio_mimetype': 'audio/wav',
    }
    if job_id is not None:
        metadata['job_id'] = job_id
    with app.app_context():
        conv = Conversion(
            user_id=user_id,
            conversion_type=ctype,
            title='Test-Vertonung',
            content='**Anna:** Hallo zusammen.\n\n**Ben:** Schön hier zu sein.',
            metadata_json=json.dumps(metadata),
        )
        db.session.add(conv)
        db.session.commit()
        return conv.id


def _write_wav(output_dir, conversion_id, data=_DUMMY_WAV):
    path = os.path.join(output_dir, f'narration_{conversion_id}.wav')
    with open(path, 'wb') as f:
        f.write(data)
    return path


def _write_real_wav(output_dir, conversion_id, seconds=2, rate=24000):
    """A valid mono/16-bit WAV so reconcile's _wav_duration reads a real length."""
    path = os.path.join(output_dir, f'narration_{conversion_id}.wav')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b'\x00\x00' * (rate * seconds))
    return path


def _stored_status(app, conversion_id):
    with app.app_context():
        return json.loads(db.session.get(Conversion, conversion_id).metadata_json)['narration_status']


def _second_user(app, username='mallory'):
    with app.app_context():
        u = User(username=username)
        u.set_password('hunter2hunter2')
        db.session.add(u)
        db.session.commit()
        return u.id


# --- Serve ---

def test_serve_ready_narration_returns_audio_and_keeps_file(
        authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='ready')
    path = _write_wav(narration_output_dir, cid)

    resp = authenticated_client.get(f'/api/narrations/{cid}/audio')

    assert resp.status_code == 200
    assert resp.mimetype == 'audio/wav'
    assert resp.data == _DUMMY_WAV
    # Persistent — the file is NOT deleted on serve (unlike podcast_download).
    assert os.path.exists(path)


def test_serve_foreign_narration_404(authenticated_client, app, narration_output_dir):
    other_id = _second_user(app)
    cid = _make_narration(app, other_id, status='ready')
    _write_wav(narration_output_dir, cid)  # file exists, but not the caller's

    resp = authenticated_client.get(f'/api/narrations/{cid}/audio')
    assert resp.status_code == 404


def test_serve_wrong_type_404(authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='ready', ctype='markdown_input')
    _write_wav(narration_output_dir, cid)

    resp = authenticated_client.get(f'/api/narrations/{cid}/audio')
    assert resp.status_code == 404


def test_serve_pending_no_file_404(authenticated_client, app, test_user, narration_output_dir):
    # Pending with NO audio file: reconcile can't make it ready (no file, no
    # job_id → it flips to failed), so the serve gate still blocks. NARR-3
    # changed only the pending-WITH-file case (see the reconcile test below).
    cid = _make_narration(app, test_user['id'], status='pending')

    resp = authenticated_client.get(f'/api/narrations/{cid}/audio')
    assert resp.status_code == 404


def test_serve_missing_file_404(authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='ready')
    # no file written

    resp = authenticated_client.get(f'/api/narrations/{cid}/audio')
    assert resp.status_code == 404


def test_serve_traversal_blocked_403(
        authenticated_client, app, test_user, narration_output_dir, tmp_path, monkeypatch):
    cid = _make_narration(app, test_user['id'], status='ready')
    # An existing file OUTSIDE OUTPUT_DIR (tmp_path is the parent of the
    # patched OUTPUT_DIR). Force the path resolver to return it.
    outside = tmp_path / 'outside.wav'
    outside.write_bytes(_DUMMY_WAV)
    monkeypatch.setattr('app_pkg.narration.narration_audio_path', lambda _id: str(outside))

    resp = authenticated_client.get(f'/api/narrations/{cid}/audio')
    assert resp.status_code == 403
    assert os.path.exists(outside)  # guard blocks serve; never touches the file


def test_serve_pending_with_file_reconciles_then_serves(
        authenticated_client, app, test_user, narration_output_dir):
    # NARR-3: a pending element whose WAV is already on disk gets reconciled to
    # ready by the serve route (before the ready-gate) and streams immediately.
    cid = _make_narration(app, test_user['id'], status='pending')
    _write_real_wav(narration_output_dir, cid)

    resp = authenticated_client.get(f'/api/narrations/{cid}/audio')
    assert resp.status_code == 200
    assert resp.mimetype == 'audio/wav'
    assert _stored_status(app, cid) == 'ready'  # reconcile persisted the flip


# --- NARR-3 status poll: GET /api/narrations/<id> + reconcile ---

def test_status_foreign_narration_404(authenticated_client, app, narration_output_dir):
    other_id = _second_user(app)
    cid = _make_narration(app, other_id, status='ready')
    assert authenticated_client.get(f'/api/narrations/{cid}').status_code == 404


def test_status_wrong_type_404(authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='ready', ctype='markdown_input')
    assert authenticated_client.get(f'/api/narrations/{cid}').status_code == 404


def test_status_pending_with_file_reconciles_to_ready(
        authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='pending')
    _write_real_wav(narration_output_dir, cid, seconds=2)

    resp = authenticated_client.get(f'/api/narrations/{cid}')
    assert resp.status_code == 200
    meta = resp.get_json()['metadata']
    assert meta['narration_status'] == 'ready'
    assert meta['duration_seconds'] == 2     # frames / framerate
    assert _stored_status(app, cid) == 'ready'  # persisted, not just in the response


def test_status_pending_job_failed_reconciles_to_failed(
        authenticated_client, app, test_user, narration_output_dir, mock_redis_queue):
    cid = _make_narration(app, test_user['id'], status='pending', job_id='job-x')
    # no file on disk → reconcile consults the (mocked) RQ job
    failed_job = MagicMock()
    failed_job.is_failed = True
    failed_job.exc_info = 'Traceback...\nValueError: kaputt'
    mock_redis_queue['fetch'].return_value = failed_job

    resp = authenticated_client.get(f'/api/narrations/{cid}')
    assert resp.status_code == 200
    meta = resp.get_json()['metadata']
    assert meta['narration_status'] == 'failed'
    assert 'kaputt' in meta['error']


def test_status_ready_is_idempotent(
        authenticated_client, app, test_user, narration_output_dir, mock_redis_queue):
    cid = _make_narration(app, test_user['id'], status='ready')

    resp = authenticated_client.get(f'/api/narrations/{cid}')
    assert resp.status_code == 200
    assert resp.get_json()['metadata']['narration_status'] == 'ready'
    mock_redis_queue['fetch'].assert_not_called()  # terminal → no job lookup


# --- Delete cleanup ---

def test_delete_narration_unlinks_audio(authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='ready')
    path = _write_wav(narration_output_dir, cid)
    assert os.path.exists(path)

    resp = authenticated_client.delete(f'/api/conversions/{cid}')
    assert resp.status_code == 200

    assert not os.path.exists(path)  # audio unlinked post-commit
    with app.app_context():
        assert db.session.get(Conversion, cid) is None  # row gone


def test_delete_non_narration_leaves_audio_untouched(
        authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='ready', ctype='markdown_input')
    # A stray file at this id's narration path — cleanup only fires for
    # audio_narration, so a non-narration delete must not touch it.
    path = _write_wav(narration_output_dir, cid)

    resp = authenticated_client.delete(f'/api/conversions/{cid}')
    assert resp.status_code == 200

    assert os.path.exists(path)  # untouched
    with app.app_context():
        assert db.session.get(Conversion, cid) is None


def test_delete_narration_missing_file_does_not_raise(
        authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='ready')
    # no audio file on disk — best-effort cleanup must not raise

    resp = authenticated_client.delete(f'/api/conversions/{cid}')
    assert resp.status_code == 200  # delete succeeds despite no file
    with app.app_context():
        assert db.session.get(Conversion, cid) is None
