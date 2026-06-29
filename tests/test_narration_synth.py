"""Alt-flow characterization: ``generate_podcast``'s NARR-1 decoupling params.

NARR-1B pivoted the faithful-narration render engine from the ``google.genai``
path to the Cloud-TTS path (``services.narration_render`` →
``GoogleTTSService.synthesize_narration``). The NARR-1 ``GeminiService.synthesize_turns``
facade (genai path) was therefore removed; its **contract validation** now lives
in ``narration_render.validate_turns`` and is tested in
``tests/test_narration_render.py``, alongside the Cloud-render behaviour
(label↔voice decoupling, verbatim transcript, prompt-as-separate-field).

What stays here: the two ``generate_podcast`` parameters NARR-1 added —
``voices={label: voice_name}`` and ``filter_metadata`` — remain on the old
podcast engine (inert with their defaults, so the historical flow is
byte-identical). These tests pin that they still behave when exercised, and that
the default path is unchanged.

Mocks at the SDK boundary: ``client.models.generate_content`` returns a minimal
fake audio response and its ``contents``/``config`` kwargs are captured for
assertions. No real Gemini calls.
"""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.gemini.tts import generate_podcast


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


# --- filter_metadata (NARR-1 param, kept inert on the old engine) ---

def test_default_path_drops_url_like_line(cleanup_wavs):
    """Default ``generate_podcast`` still filters metadata (old flow intact)."""
    client = _make_client()
    dialogue = [
        {'speaker': 'Anna', 'style': '', 'text': 'Mehr dazu auf example.de'},
        {'speaker': 'Ben', 'style': '', 'text': 'Klar, gerne.'},
    ]
    cleanup_wavs.append(generate_podcast(client, dialogue, filter_metadata=True))

    contents, _ = _capture(client)
    assert 'example.de' not in contents
    assert 'Klar, gerne.' in contents


def test_filter_metadata_off_keeps_url_like_turn(cleanup_wavs):
    """``filter_metadata=False`` keeps a .de line the default filter would drop."""
    client = _make_client()
    dialogue = [
        {'speaker': 'Anna', 'style': '', 'text': 'Mehr dazu auf example.de'},
        {'speaker': 'Ben', 'style': '', 'text': 'Klar, gerne.'},
    ]
    cleanup_wavs.append(generate_podcast(client, dialogue, filter_metadata=False))

    contents, _ = _capture(client)
    assert 'Mehr dazu auf example.de' in contents


# --- voices map (NARR-1 param; None = byte-identical old flow) ---

def test_old_flow_label_is_voice_when_voices_none(cleanup_wavs):
    client = _make_client()
    dialogue = [
        {'speaker': 'A', 'style': '', 'text': 'hi'},
        {'speaker': 'B', 'style': '', 'text': 'yo'},
    ]
    cleanup_wavs.append(generate_podcast(client, dialogue))  # no voices, default filter

    _, config = _capture(client)
    assert _voice_map(config) == {'A': 'A', 'B': 'B'}


def test_voices_map_decouples_label_from_voice(cleanup_wavs):
    """The ``voices`` map still keys voice off the map, not the label."""
    client = _make_client()
    dialogue = [
        {'speaker': 'Anna', 'style': '', 'text': 'Hallo.'},
        {'speaker': 'Ben', 'style': '', 'text': 'Hi.'},
    ]
    cleanup_wavs.append(generate_podcast(
        client, dialogue, voices={'Anna': 'Kore', 'Ben': 'Puck'}, filter_metadata=False
    ))

    contents, config = _capture(client)
    assert _voice_map(config) == {'Anna': 'Kore', 'Ben': 'Puck'}
    # The transcript keeps the human label, never the voice name.
    assert 'Anna:' in contents and 'Kore' not in contents
