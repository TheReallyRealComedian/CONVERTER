"""NARR-2 serve + delete-cleanup integration tests.

Exercises ``GET /api/narrations/<id>/audio`` and the audio-file cleanup hook in
``api_delete_conversion`` through the HTTP boundary, against synthetic
``audio_narration`` Conversions + dummy WAV files. ``OUTPUT_DIR`` is monkeypatched
to a tmp dir in both the serve module and the persistence helper so no container
path is touched.

This sprint builds only the persistence shell — nothing generates audio — so the
WAVs here are written by hand.
"""
import json
import os

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


def _make_narration(app, user_id, *, status='ready', ctype='audio_narration'):
    """Create a synthetic audio_narration Conversion; return its id."""
    metadata = {
        'narration_status': status,
        'audio_filename': 'narration_PLACEHOLDER.wav',
        'audio_mimetype': 'audio/wav',
    }
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


def test_serve_not_ready_404(authenticated_client, app, test_user, narration_output_dir):
    cid = _make_narration(app, test_user['id'], status='pending')
    _write_wav(narration_output_dir, cid)  # even with a file, pending is not servable

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
