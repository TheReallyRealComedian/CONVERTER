"""NARR-1 faithful-synthesis-core characterization tests.

Exercises ``services.gemini.tts.synthesize_turns`` (the faithful narration
entry) plus the two ``generate_podcast`` adjustments it relies on:

- ``voices={label: voice_name}`` decouples the transcript speaker *label* from
  the Gemini *voice* (``None`` = label-as-voice, byte-identical old flow).
- ``filter_metadata=False`` skips the URL/caption/short-line drop so the
  faithful path never loses a legitimate turn (fidelity).

Mocks at the SDK boundary: ``client.models.generate_content`` returns a
minimal fake audio response and its ``contents``/``config`` kwargs are
captured for assertions. No real Gemini calls — mirrors the conftest
MagicMock style so it survives future service splits.

Note: single-speaker *synthesis* is intentionally only exercised at the
validation layer here, not end-to-end — see NARR-1 Phase-1 report re the
``generate_single_chunk`` single-speaker branch still keying voice off the
label rather than the ``voices`` map.
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.gemini.tts import generate_podcast, synthesize_turns


# 24 kHz / 16-bit mono → even byte length so wave.writeframes accepts it.
_AUDIO_BYTES = b'\x00\x01' * 1200


def _fake_audio_response():
    """A response shaped exactly as ``generate_single_chunk`` validates it."""
    inline = SimpleNamespace(data=_AUDIO_BYTES, mime_type='audio/L16;rate=24000')
    part = SimpleNamespace(inline_data=inline)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content, finish_reason='STOP', safety_ratings=[])
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


def _make_client():
    client = MagicMock()
    client.models.generate_content.return_value = _fake_audio_response()
    return client


def _capture(client):
    """Return ``(contents, config)`` from the single generate_content call."""
    client.models.generate_content.assert_called_once()
    kwargs = client.models.generate_content.call_args.kwargs
    return kwargs['contents'], kwargs['config']


def _voice_map(config):
    """``label -> voice_name`` from a captured multi-speaker config."""
    cfgs = config.speech_config.multi_speaker_voice_config.speaker_voice_configs
    return {c.speaker: c.voice_config.prebuilt_voice_config.voice_name for c in cfgs}


@pytest.fixture
def cleanup_wavs():
    """Collect temp-WAV paths returned by synth calls and unlink them."""
    paths = []
    yield paths
    for p in paths:
        if p and os.path.exists(p):
            os.unlink(p)


# --- Validation (pure; the SDK is never touched) ---

def test_empty_turns_raises():
    client = _make_client()
    with pytest.raises(ValueError):
        synthesize_turns(client, [], {'A': 'Kore'})
    client.models.generate_content.assert_not_called()


def test_turn_missing_text_raises():
    client = _make_client()
    with pytest.raises(ValueError):
        synthesize_turns(client, [{'speaker': 'Anna', 'text': '   '}], {'Anna': 'Kore'})
    client.models.generate_content.assert_not_called()


def test_turn_missing_speaker_raises():
    client = _make_client()
    with pytest.raises(ValueError):
        synthesize_turns(client, [{'speaker': '', 'text': 'Hallo'}], {'Anna': 'Kore'})
    client.models.generate_content.assert_not_called()


def test_two_speaker_with_three_speakers_raises():
    client = _make_client()
    turns = [
        {'speaker': 'A', 'text': 'eins'},
        {'speaker': 'B', 'text': 'zwei'},
        {'speaker': 'C', 'text': 'drei'},
    ]
    with pytest.raises(ValueError):
        synthesize_turns(client, turns, {'A': 'Kore', 'B': 'Puck', 'C': 'Charon'},
                         mode='two_speaker')
    client.models.generate_content.assert_not_called()


def test_single_speaker_with_two_speakers_raises():
    client = _make_client()
    turns = [
        {'speaker': 'A', 'text': 'eins'},
        {'speaker': 'B', 'text': 'zwei'},
    ]
    with pytest.raises(ValueError):
        synthesize_turns(client, turns, {'A': 'Kore', 'B': 'Puck'}, mode='single_speaker')
    client.models.generate_content.assert_not_called()


def test_speaker_without_voice_raises():
    client = _make_client()
    turns = [
        {'speaker': 'Anna', 'text': 'Hallo'},
        {'speaker': 'Ben', 'text': 'Hi'},
    ]
    with pytest.raises(ValueError):
        synthesize_turns(client, turns, {'Anna': 'Kore'})  # Ben uncovered
    client.models.generate_content.assert_not_called()


def test_voices_not_a_dict_raises():
    client = _make_client()
    turns = [{'speaker': 'Anna', 'text': 'Hallo'}]
    with pytest.raises(ValueError):
        synthesize_turns(client, turns, ['Kore'], mode='single_speaker')
    client.models.generate_content.assert_not_called()


def test_unknown_mode_raises():
    client = _make_client()
    turns = [{'speaker': 'Anna', 'text': 'Hallo'}]
    with pytest.raises(ValueError):
        synthesize_turns(client, turns, {'Anna': 'Kore'}, mode='trio')
    client.models.generate_content.assert_not_called()


# --- Label↔Voice decoupling ---

def test_label_voice_decoupling_multi_speaker(cleanup_wavs):
    client = _make_client()
    turns = [
        {'speaker': 'Anna', 'text': 'Hallo zusammen.'},
        {'speaker': 'Ben', 'text': 'Schön, dabei zu sein.'},
    ]
    cleanup_wavs.append(synthesize_turns(client, turns, {'Anna': 'Kore', 'Ben': 'Puck'}))

    contents, config = _capture(client)
    # The voice config carries the *mapped* voice per label.
    assert _voice_map(config) == {'Anna': 'Kore', 'Ben': 'Puck'}
    # The transcript keeps the human label, never the voice name.
    assert 'Anna:' in contents and 'Ben:' in contents
    assert 'Kore' not in contents and 'Puck' not in contents


# --- Fidelity: faithful path keeps everything (load-bearing) ---

def test_faithful_path_keeps_url_like_turn(cleanup_wavs):
    client = _make_client()
    turns = [
        {'speaker': 'Anna', 'text': 'Mehr dazu auf example.de'},
        {'speaker': 'Ben', 'text': 'Klar, gerne.'},
    ]
    cleanup_wavs.append(synthesize_turns(client, turns, {'Anna': 'Kore', 'Ben': 'Puck'}))

    contents, _ = _capture(client)
    # The .de line would be dropped by the metadata filter — faithful path keeps it.
    assert 'Mehr dazu auf example.de' in contents


def test_default_path_drops_url_like_line(cleanup_wavs):
    """Contrast: ``generate_podcast`` still filters by default (old flow intact)."""
    client = _make_client()
    dialogue = [
        {'speaker': 'Anna', 'style': '', 'text': 'Mehr dazu auf example.de'},
        {'speaker': 'Ben', 'style': '', 'text': 'Klar, gerne.'},
    ]
    cleanup_wavs.append(generate_podcast(client, dialogue, filter_metadata=True))

    contents, _ = _capture(client)
    assert 'example.de' not in contents
    assert 'Klar, gerne.' in contents


# --- Transcript verbatim (inline tags survive, no double bracketing) ---

def test_inline_tag_preserved_verbatim(cleanup_wavs):
    client = _make_client()
    turns = [
        {'speaker': 'Anna', 'text': '[ruhig] Hallo'},
        {'speaker': 'Ben', 'text': 'Hi.'},
    ]
    cleanup_wavs.append(synthesize_turns(client, turns, {'Anna': 'Kore', 'Ben': 'Puck'}))

    contents, _ = _capture(client)
    assert 'Anna: [ruhig] Hallo' in contents
    # style='' → no empty-bracket prefix from the "{speaker}: [{style}] …" branch.
    assert 'Anna: [] ' not in contents


# --- Old flow byte-identical (voices=None → label is voice) ---

def test_old_flow_label_is_voice_when_voices_none(cleanup_wavs):
    client = _make_client()
    dialogue = [
        {'speaker': 'A', 'style': '', 'text': 'hi'},
        {'speaker': 'B', 'style': '', 'text': 'yo'},
    ]
    cleanup_wavs.append(generate_podcast(client, dialogue))  # no voices, default filter

    _, config = _capture(client)
    assert _voice_map(config) == {'A': 'A', 'B': 'B'}


# --- Facade delegation ---

def test_facade_synthesize_turns_delegates(cleanup_wavs):
    from services.gemini import GeminiService

    svc = GeminiService.__new__(GeminiService)  # skip real genai.Client construction
    svc.client = _make_client()
    svc.pydub_available = True

    turns = [
        {'speaker': 'Anna', 'text': 'Hallo.'},
        {'speaker': 'Ben', 'text': 'Hi.'},
    ]
    cleanup_wavs.append(svc.synthesize_turns(turns, {'Anna': 'Kore', 'Ben': 'Puck'}))

    contents, config = _capture(svc.client)
    assert _voice_map(config) == {'Anna': 'Kore', 'Ben': 'Puck'}
    assert 'Anna: Hallo.' in contents
