"""Diarization (Sprecher-Erkennung) characterization tests — Sprint DIARIZE.

Deckt ab:
- ``format_diarized_transcript`` (pure): Multi-Speaker → ``**Sprecher N:**``-Blocks
  mit Konsolidierung aufeinanderfolgender gleicher Sprecher; Single-Speaker →
  byte-gleicher Plain-Text (Diktat-Regression-Guard); fehlende/leere utterances
  oder ``speaker=None`` → Plain-Fallback.
- ``_transcribe_single``: fordert ``diarize_model=v2`` via
  ``additional_query_parameters`` an (NIE ``diarize=True`` daneben), lässt alle
  bestehenden Optionen unangetastet und routet die utterances durch den Formatter.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from services.deepgram_service import (
    DeepgramService,
    MULTI_CHUNK_DIARIZATION_NOTICE,
    format_diarized_transcript,
)


def _utt(speaker, transcript):
    return SimpleNamespace(speaker=speaker, transcript=transcript)


# --------------------------------------------------------------------------
# format_diarized_transcript — pure function
# --------------------------------------------------------------------------

def test_multi_speaker_produces_labeled_blocks():
    utterances = [
        _utt(0, "Hallo zusammen."),
        _utt(1, "Guten Morgen."),
        _utt(0, "Fangen wir an."),
    ]
    out = format_diarized_transcript(utterances, plain_transcript="ignored plain")
    assert out == (
        "**Sprecher 1:** Hallo zusammen.\n\n"
        "**Sprecher 2:** Guten Morgen.\n\n"
        "**Sprecher 1:** Fangen wir an."
    )


def test_consecutive_same_speaker_consolidated_into_one_block():
    utterances = [
        _utt(0, "Erster Satz."),
        _utt(0, "Zweiter Satz."),
        _utt(1, "Antwort."),
    ]
    out = format_diarized_transcript(utterances, plain_transcript="ignored")
    assert out == (
        "**Sprecher 1:** Erster Satz. Zweiter Satz.\n\n"
        "**Sprecher 2:** Antwort."
    )


def test_single_speaker_returns_plain_transcript_byte_identical():
    plain = "Dies ist ein Einzel-Diktat ohne Labels."
    utterances = [
        _utt(0, "Dies ist ein Einzel-Diktat"),
        _utt(0, "ohne Labels."),
    ]
    out = format_diarized_transcript(utterances, plain_transcript=plain)
    assert out == plain


def test_empty_utterances_returns_plain():
    plain = "Plain-Fallback."
    assert format_diarized_transcript([], plain_transcript=plain) == plain
    assert format_diarized_transcript(None, plain_transcript=plain) == plain


def test_speaker_none_falls_back_to_plain():
    plain = "Kein Sprecher-Feld → Plain."
    utterances = [
        _utt(None, "Erster."),
        _utt(0, "Zweiter."),
    ]
    out = format_diarized_transcript(utterances, plain_transcript=plain)
    assert out == plain


# --------------------------------------------------------------------------
# _transcribe_single — SDK boundary
# --------------------------------------------------------------------------

def _make_service_with_mock_client(transcript, utterances):
    service = DeepgramService(api_key="fake-key")
    service.client = MagicMock()
    alt = SimpleNamespace(transcript=transcript)
    channel = SimpleNamespace(alternatives=[alt])
    results = SimpleNamespace(channels=[channel], utterances=utterances)
    response = SimpleNamespace(results=results)
    service.client.listen.v1.media.transcribe_file.return_value = response
    return service


def test_transcribe_single_requests_diarize_model_v2_without_diarize_true():
    service = _make_service_with_mock_client("plain text", None)
    service._transcribe_single(b"audio-bytes", "de")

    call = service.client.listen.v1.media.transcribe_file
    call.assert_called_once()
    kwargs = call.call_args.kwargs

    # v2 aktiviert via Query-Param-Passthrough
    aqp = kwargs["request_options"]["additional_query_parameters"]
    assert aqp == {"diarize_model": "v2"}
    # NIEMALS diarize=True daneben (Request würde sonst rejected)
    assert "diarize" not in kwargs
    # bestehende Optionen unverändert
    assert kwargs["model"] == "nova-3"
    assert kwargs["utterances"] is True
    assert kwargs["smart_format"] is True
    assert kwargs["paragraphs"] is True
    assert "keyterm" in kwargs


def test_transcribe_single_multi_speaker_returns_labeled_markdown():
    utterances = [_utt(0, "Frage?"), _utt(1, "Antwort.")]
    service = _make_service_with_mock_client("Frage? Antwort.", utterances)
    out = service._transcribe_single(b"audio", "de")
    assert out == "**Sprecher 1:** Frage?\n\n**Sprecher 2:** Antwort."


def test_transcribe_single_single_speaker_returns_plain():
    utterances = [_utt(0, "Nur ich rede.")]
    service = _make_service_with_mock_client("Nur ich rede hier.", utterances)
    out = service._transcribe_single(b"audio", "de")
    assert out == "Nur ich rede hier."


def test_transcribe_single_no_utterances_returns_plain():
    service = _make_service_with_mock_client("Kein Diarization-Output.", None)
    out = service._transcribe_single(b"audio", "de")
    assert out == "Kein Diarization-Output."


def test_transcribe_single_chunk_path_disables_diarization():
    """apply_diarization=False → Plain, selbst bei ≥2 erkannten Sprechern."""
    utterances = [_utt(0, "A"), _utt(1, "B")]
    service = _make_service_with_mock_client("A B plain.", utterances)
    out = service._transcribe_single(b"audio", "de", apply_diarization=False)
    assert out == "A B plain."
    assert "**Sprecher" not in out


# --------------------------------------------------------------------------
# transcribe_file — Schwellen-Routing (needs_splitting gemockt)
# --------------------------------------------------------------------------

def test_single_request_below_threshold_gets_diarization():
    """≤90 min (needs_splitting=False) → 1 Request mit Sprecher-Labels."""
    utterances = [_utt(0, "Frage?"), _utt(1, "Antwort.")]
    service = _make_service_with_mock_client("Frage? Antwort.", utterances)
    service.chunker.needs_splitting = MagicMock(return_value=(False, {"duration": 5000}))

    out = service.transcribe_file(b"audio", "de")
    assert out == "**Sprecher 1:** Frage?\n\n**Sprecher 2:** Antwort."


def test_over_threshold_chunks_plain_with_degradation_notice():
    """>90 min (needs_splitting=True) → Plain (keine Labels) + Degradations-Hinweis."""
    utterances = [_utt(0, "A"), _utt(1, "B")]
    service = _make_service_with_mock_client("Chunk plain text.", utterances)
    service.chunker.needs_splitting = MagicMock(return_value=(True, {"duration": 6000}))
    service.chunker.split_audio = MagicMock(return_value=[
        SimpleNamespace(index=0, audio_data=b"chunk0", is_last=True),
    ])

    out = service.transcribe_file(b"audio", "de")
    assert out.startswith(MULTI_CHUNK_DIARIZATION_NOTICE)
    assert "Chunk plain text." in out
    assert "**Sprecher" not in out
