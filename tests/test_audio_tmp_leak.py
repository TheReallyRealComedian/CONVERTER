"""AUDIO-TMP-LEAK (KLEINKRAM, 2026-08-22) — the probe copy must not outlive its use.

``AudioChunker.needs_splitting`` writes the upload to ``/tmp/tmp*.audio`` so
ffprobe can read it and used to RETAIN it unconditionally "for split_audio".
On the non-split path (≤ 90 min — every normal job) split_audio never runs,
so one 22–91 MB copy per transcription stayed behind until the next container
restart (measured: eight files after eight runs). The fix deletes the copy at
the place that created it whenever no split follows.

Both paths are pinned here, at the chunker AND through
``DeepgramService.transcribe_file``:
  * no split   → nothing left in the temp dir, no retained path;
  * split      → the copy is handed over to split_audio, which still finds it
                 (the chunked path must keep working) and deletes it afterwards.
``tempfile.tempdir`` is pointed at an isolated directory so the tests can list
exactly what a run leaves behind.
"""
import os
import tempfile
from unittest.mock import patch

import pytest

from services.audio_chunker import AudioChunker
from services.deepgram_service import DeepgramService


def _probe(duration_sec):
    """What ffprobe returns, reduced to the keys needs_splitting reads."""
    return {
        "format": {"duration": str(duration_sec)},
        "streams": [{"sample_rate": "16000", "channels": "1"}],
    }


@pytest.fixture
def scratch_tmp(tmp_path, monkeypatch):
    """Isolated temp dir: mkstemp() lands here, so leftovers are countable."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def _leftovers(scratch):
    return sorted(p.name for p in scratch.iterdir())


# --------------------------------------------------------------------------
# Chunker level
# --------------------------------------------------------------------------

def test_needs_splitting_deletes_probe_copy_when_no_split_follows(scratch_tmp):
    chunker = AudioChunker(max_duration_seconds=5400, max_file_size_mb=500)
    assert chunker._tmp_path is None  # explicit initial state, no hasattr dance

    with patch.object(AudioChunker, "_get_audio_metadata", return_value=_probe(1800)):
        needs_split, metadata = chunker.needs_splitting(b"short recording")

    assert needs_split is False
    assert metadata["duration_seconds"] == 1800
    assert _leftovers(scratch_tmp) == []          # the leak
    assert chunker._tmp_path is None
    assert chunker._tmp_owned is False


def test_needs_splitting_hands_probe_copy_over_and_split_audio_consumes_it(scratch_tmp):
    chunker = AudioChunker(max_duration_seconds=5400, chunk_duration_seconds=1800,
                           overlap_seconds=5, max_file_size_mb=500)

    with patch.object(AudioChunker, "_get_audio_metadata", return_value=_probe(7200)):
        needs_split, metadata = chunker.needs_splitting(b"long recording")

    assert needs_split is True
    assert metadata["estimated_chunks"] == 5
    retained = _leftovers(scratch_tmp)
    assert len(retained) == 1 and retained[0].endswith(".audio")
    assert chunker._tmp_path == os.path.join(str(scratch_tmp), retained[0])
    assert chunker._tmp_owned is True

    seen_inputs = []

    def fake_extract(self, input_path, start_sec, duration_sec):
        seen_inputs.append(input_path)
        return b"mp3-bytes"

    with patch.object(AudioChunker, "_extract_chunk_ffmpeg", fake_extract):
        chunks = chunker.split_audio(b"long recording")

    # The chunked path read the handed-over copy (not a fresh fallback file) …
    assert len(chunks) == 5
    assert set(seen_inputs) == {os.path.join(str(scratch_tmp), retained[0])}
    # … and cleaned it up afterwards, leaving no retained state behind.
    assert _leftovers(scratch_tmp) == []
    assert chunker._tmp_path is None
    assert chunker._tmp_owned is False


def test_needs_splitting_probe_failure_leaves_nothing_behind(scratch_tmp):
    chunker = AudioChunker()
    with patch.object(AudioChunker, "_get_audio_metadata",
                      side_effect=RuntimeError("ffprobe failed")):
        needs_split, metadata = chunker.needs_splitting(b"garbage")
    assert needs_split is False
    assert "error" in metadata
    assert _leftovers(scratch_tmp) == []


# --------------------------------------------------------------------------
# Service level — the path tasks.transcribe_audio_task actually runs
# --------------------------------------------------------------------------

def test_transcribe_file_single_request_leaves_no_tmp_file(scratch_tmp):
    service = DeepgramService(api_key="fake-key")
    with patch.object(AudioChunker, "_get_audio_metadata", return_value=_probe(1800)), \
            patch.object(DeepgramService, "_transcribe_single",
                         return_value="Ein kurzes Diktat.") as single:
        out = service.transcribe_file(b"short recording", language="de")

    assert out == "Ein kurzes Diktat."
    single.assert_called_once()
    assert _leftovers(scratch_tmp) == []
    assert service.chunker._tmp_path is None


def test_transcribe_file_chunked_path_still_works_and_leaves_no_tmp_file(scratch_tmp):
    service = DeepgramService(api_key="fake-key")
    service.INTER_CHUNK_DELAY = 0

    def fake_extract(self, input_path, start_sec, duration_sec):
        assert os.path.exists(input_path)   # the handed-over copy is still there
        return b"mp3-bytes"

    with patch.object(AudioChunker, "_get_audio_metadata", return_value=_probe(7200)), \
            patch.object(AudioChunker, "_extract_chunk_ffmpeg", fake_extract), \
            patch.object(DeepgramService, "_transcribe_single",
                         side_effect=["Teil eins.", "Teil zwei.", "Teil drei.",
                                      "Teil vier.", "Teil fünf."]) as single:
        out = service.transcribe_file(b"long recording", language="de")

    assert single.call_count == 5
    assert out.startswith("> Hinweis: Aufnahme über 90 Minuten")
    assert "Teil eins." in out and "Teil fünf." in out
    assert _leftovers(scratch_tmp) == []
    assert service.chunker._tmp_path is None
