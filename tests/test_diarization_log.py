"""ARCH-BUILD (W-16) — a silent diarization degradation becomes loud.

``format_diarized_transcript`` stays pure and byte-identical (its tests in
tests/test_diarization.py are the proof); the log lines live at the CALL
SITE ``_transcribe_single``, where ``apply_diarization`` is known:

* diarization requested, no speakers delivered (no utterances, or a
  ``speaker`` missing) → WARNING „Diarisierung angefordert, keine Sprecher
  geliefert."
* exactly one speaker → INFO „Diarisierung: ein Sprecher, Fließtext."
  (Oli's single-voice dictations are not a warning case)
* two or more speakers → no line (the labeled Markdown is the evidence)
* chunk path (``apply_diarization=False``) → no line at all

The pure helper ``diarization_outcome`` carries the distinction for both
places.
"""
import logging

import pytest

from services.deepgram_service import diarization_outcome, format_diarized_transcript
from tests.test_diarization import _make_service_with_mock_client, _utt

LOGGER = 'services.deepgram_service'
WARN = 'Diarisierung angefordert, keine Sprecher geliefert.'
INFO = 'Diarisierung: ein Sprecher, Fließtext.'


def _diarization_records(caplog):
    return [(r.levelno, r.getMessage()) for r in caplog.records
            if r.name == LOGGER and r.getMessage().startswith('Diarisierung')]


@pytest.mark.parametrize('utterances, outcome', [
    (None, 'none'),
    ([], 'none'),
    ([_utt(0, 'a'), _utt(None, 'b')], 'none'),
    ([_utt(0, 'a'), _utt(0, 'b')], 'single'),
    ([_utt(0, 'a'), _utt(1, 'b')], 'multi'),
])
def test_diarization_outcome(utterances, outcome):
    assert diarization_outcome(utterances) == outcome
    # the pure formatter agrees: anything but 'multi' is the plain text
    plain = 'plain'
    formatted = format_diarized_transcript(utterances, plain)
    assert (formatted == plain) == (outcome != 'multi')


@pytest.mark.parametrize('utterances', [None, [], [_utt(0, 'a'), _utt(None, 'b')]])
def test_no_speakers_is_a_warning_at_the_call_site(caplog, utterances):
    service = _make_service_with_mock_client('plain text', utterances)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = service._transcribe_single(b'audio', 'de')
    assert out == 'plain text'
    assert _diarization_records(caplog) == [(logging.WARNING, WARN)]


def test_one_speaker_is_an_info_line_not_a_warning(caplog):
    service = _make_service_with_mock_client('plain text', [_utt(0, 'a'), _utt(0, 'b')])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = service._transcribe_single(b'audio', 'de')
    assert out == 'plain text'
    assert _diarization_records(caplog) == [(logging.INFO, INFO)]


def test_two_speakers_log_nothing_about_diarization(caplog):
    service = _make_service_with_mock_client('plain text', [_utt(0, 'a'), _utt(1, 'b')])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = service._transcribe_single(b'audio', 'de')
    assert out.startswith('**Sprecher 1:**')
    assert _diarization_records(caplog) == []


def test_chunk_path_logs_nothing_about_diarization(caplog):
    # apply_diarization=False never asked for speakers — missing ones are no finding
    service = _make_service_with_mock_client('plain text', None)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = service._transcribe_single(b'audio', 'de', apply_diarization=False)
    assert out == 'plain text'
    assert _diarization_records(caplog) == []
