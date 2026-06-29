"""NARR-1B Cloud-TTS faithful-narration renderer tests.

Exercises ``services.narration_render`` at the ``client.synthesize_speech``
boundary — the renderer takes the ``texttospeech`` client as an argument, so the
mock captures the real ``input`` / ``voice`` / ``audio_config`` proto-plus
objects and the tests assert on their fields directly (no real Cloud calls).

Covers the contract pulled from NARR-1 (``validate_turns``), byte-based chunking
(utf-8, never mid-turn, oversized-turn split), header-agnostic WAV wrapping, the
multi-speaker (``multi_speaker_markup``) vs single-speaker (``text=``) routing,
label↔voice decoupling (``speaker_alias`` ≠ ``speaker_id``), the ``prompt``
separate-field (no transcript leakage), and api_core error handling + retry.
"""
import io
import os
import wave
from types import SimpleNamespace

import pytest
from google.api_core import exceptions as gax
from google.cloud import texttospeech

from services import narration_render
from services.narration_render import (
    chunk_turns,
    pcm_to_wav_bytes,
    render_turns,
    validate_turns,
)


# 24 kHz / 16-bit mono → even byte length so wave.writeframes accepts it.
_FAKE_PCM = b'\x00\x01' * 1200


def _build_wav_bytes(pcm=_FAKE_PCM):
    """A real RIFF/WAV container around ``pcm`` (mono/16-bit/24 kHz)."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)
    return buf.getvalue()


def _make_client(audio_content=_FAKE_PCM, side_effect=None):
    """Mock client whose ``synthesize_speech`` returns fake audio (or raises)."""
    from unittest.mock import MagicMock
    client = MagicMock()
    if side_effect is not None:
        client.synthesize_speech.side_effect = side_effect
    else:
        client.synthesize_speech.return_value = SimpleNamespace(audio_content=audio_content)
    return client


def _call(client, idx=0):
    """Return ``(input, voice, audio_config)`` from the idx-th synth call."""
    kwargs = client.synthesize_speech.call_args_list[idx].kwargs
    return kwargs['input'], kwargs['voice'], kwargs['audio_config']


def _assert_valid_wav(path):
    with wave.open(path, 'rb') as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 24000
        assert w.getsampwidth() == 2
        assert w.getnframes() > 0


@pytest.fixture
def cleanup_wavs():
    paths = []
    yield paths
    for p in paths:
        if p and os.path.exists(p):
            os.unlink(p)


# --- validate_turns (pure; mirrors NARR-1, the Phase-2 repoint target) ---

def test_validate_empty_turns_raises():
    with pytest.raises(ValueError):
        validate_turns([], {'A': 'Kore'}, 'two_speaker')


def test_validate_turn_missing_text_raises():
    with pytest.raises(ValueError):
        validate_turns([{'speaker': 'Anna', 'text': '   '}], {'Anna': 'Kore'}, 'single_speaker')


def test_validate_turn_missing_speaker_raises():
    with pytest.raises(ValueError):
        validate_turns([{'speaker': '', 'text': 'Hallo'}], {'Anna': 'Kore'}, 'single_speaker')


def test_validate_two_speaker_with_three_raises():
    turns = [
        {'speaker': 'A', 'text': 'eins'},
        {'speaker': 'B', 'text': 'zwei'},
        {'speaker': 'C', 'text': 'drei'},
    ]
    with pytest.raises(ValueError):
        validate_turns(turns, {'A': 'Kore', 'B': 'Puck', 'C': 'Charon'}, 'two_speaker')


def test_validate_single_speaker_with_two_raises():
    turns = [{'speaker': 'A', 'text': 'eins'}, {'speaker': 'B', 'text': 'zwei'}]
    with pytest.raises(ValueError):
        validate_turns(turns, {'A': 'Kore', 'B': 'Puck'}, 'single_speaker')


def test_validate_speaker_without_voice_raises():
    turns = [{'speaker': 'Anna', 'text': 'Hallo'}, {'speaker': 'Ben', 'text': 'Hi'}]
    with pytest.raises(ValueError):
        validate_turns(turns, {'Anna': 'Kore'}, 'two_speaker')  # Ben uncovered


def test_validate_voices_not_a_dict_raises():
    with pytest.raises(ValueError):
        validate_turns([{'speaker': 'Anna', 'text': 'Hallo'}], ['Kore'], 'single_speaker')


def test_validate_unknown_mode_raises():
    with pytest.raises(ValueError):
        validate_turns([{'speaker': 'Anna', 'text': 'Hallo'}], {'Anna': 'Kore'}, 'trio')


def test_validate_returns_distinct_speakers_order_preserving():
    turns = [
        {'speaker': 'Ben', 'text': 'eins'},
        {'speaker': 'Anna', 'text': 'zwei'},
        {'speaker': 'Ben', 'text': 'drei'},
    ]
    assert validate_turns(turns, {'Ben': 'Puck', 'Anna': 'Kore'}, 'two_speaker') == ['Ben', 'Anna']


def test_render_does_not_call_synth_on_invalid_input():
    client = _make_client()
    with pytest.raises(ValueError):
        render_turns(client, [], {'A': 'Kore'})
    client.synthesize_speech.assert_not_called()


# --- chunk_turns (pure; byte-based, never mid-turn) ---

def test_chunk_groups_whole_turns_never_mid_turn():
    turns = [
        {'speaker': 'A', 'text': 'eins'},
        {'speaker': 'B', 'text': 'zwei'},
        {'speaker': 'A', 'text': 'drei'},
    ]
    chunks = chunk_turns(turns, max_bytes=8)  # 4 bytes each → [eins,zwei] | [drei]
    assert len(chunks) == 2
    # Flattening reproduces the original order verbatim — no turn was split.
    flat = [t['text'] for c in chunks for t in c]
    assert flat == ['eins', 'zwei', 'drei']


def test_chunk_byte_measurement_utf8_umlauts():
    # 'ä'/'ö' are 2 bytes each in utf-8: 50 chars == 100 bytes per turn.
    turns = [{'speaker': 'A', 'text': 'ä' * 50}, {'speaker': 'A', 'text': 'ö' * 50}]
    # By bytes: 100 + 100 = 200 > 150 → 2 chunks. By char-len: 50 + 50 = 100 → 1.
    chunks = chunk_turns(turns, max_bytes=150)
    assert len(chunks) == 2


def test_chunk_splits_oversized_turn_at_sentence_boundaries():
    big = {'speaker': 'Anna',
           'text': 'Satz eins ist hier. Satz zwei ist da. Satz drei kommt jetzt. Satz vier am Ende.'}
    chunks = chunk_turns([big], max_bytes=30)
    sub_turns = [t for c in chunks for t in c]
    assert len(sub_turns) >= 2                      # the single turn was split
    assert all(t['speaker'] == 'Anna' for t in sub_turns)  # same speaker throughout
    joined = ' '.join(t['text'] for t in sub_turns)
    assert 'Satz eins' in joined and 'Satz vier' in joined  # content preserved


def test_chunk_single_small_turn_is_one_chunk():
    chunks = chunk_turns([{'speaker': 'A', 'text': 'kurz'}])
    assert chunks == [[{'speaker': 'A', 'text': 'kurz'}]]


# --- pcm_to_wav_bytes (header-agnostic) ---

def test_pcm_to_wav_wraps_raw_pcm():
    out = pcm_to_wav_bytes(_FAKE_PCM)
    assert out[:4] == b'RIFF'
    with wave.open(io.BytesIO(out), 'rb') as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 24000
        assert w.getsampwidth() == 2
        assert w.readframes(w.getnframes()) == _FAKE_PCM


def test_pcm_to_wav_passthrough_when_already_riff():
    wav = _build_wav_bytes()
    out = pcm_to_wav_bytes(wav)
    assert out == wav                # already a WAV → returned unchanged
    with wave.open(io.BytesIO(out), 'rb') as w:
        assert w.getnframes() > 0    # still valid


def test_pcm_to_wav_empty_raises():
    with pytest.raises(ValueError):
        pcm_to_wav_bytes(b'')


# --- render_turns: multi-speaker (multi_speaker_markup) ---

def test_render_multi_speaker_builds_alias_voice_configs(cleanup_wavs):
    client = _make_client()
    turns = [
        {'speaker': 'Anna', 'text': 'Hallo zusammen.'},
        {'speaker': 'Ben', 'text': 'Schön, dabei zu sein.'},
    ]
    cleanup_wavs.append(render_turns(
        client, turns, {'Anna': 'Kore', 'Ben': 'Puck'}, style_prompt='ruhig und warm'
    ))

    si, voice, audio_config = _call(client)

    # speaker_alias (sanitized positional) ↔ speaker_id (mapped voice).
    cfgs = voice.multi_speaker_voice_config.speaker_voice_configs
    assert [(c.speaker_alias, c.speaker_id) for c in cfgs] == [('Speaker1', 'Kore'), ('Speaker2', 'Puck')]

    # Each Turn.speaker equals a declared alias (never the human label).
    turn_speakers = [t.speaker for t in si.multi_speaker_markup.turns]
    assert turn_speakers == ['Speaker1', 'Speaker2']
    transcript = ' '.join(t.text for t in si.multi_speaker_markup.turns)
    assert 'Anna' not in transcript and 'Ben' not in transcript

    # prompt rides on its own field, not inside the transcript.
    assert si.prompt == 'ruhig und warm'
    assert 'ruhig und warm' not in transcript

    # model_name set (activates Gemini-TTS); name NOT set on the multi path.
    assert voice.model_name == narration_render.DEFAULT_NARRATION_MODEL
    assert voice.name == ''
    assert voice.language_code == 'de-DE'

    # LINEAR16 @ 24 kHz.
    assert audio_config.audio_encoding == texttospeech.AudioEncoding.LINEAR16
    assert audio_config.sample_rate_hertz == 24000


def test_render_multi_speaker_transcript_verbatim_with_inline_tag(cleanup_wavs):
    client = _make_client()
    turns = [
        {'speaker': 'Anna', 'text': '[ruhig] Hallo'},
        {'speaker': 'Ben', 'text': 'Hi.'},
    ]
    cleanup_wavs.append(render_turns(client, turns, {'Anna': 'Kore', 'Ben': 'Puck'}))
    si, _, _ = _call(client)
    texts = [t.text for t in si.multi_speaker_markup.turns]
    assert texts == ['[ruhig] Hallo', 'Hi.']  # inline tag preserved verbatim


# --- render_turns: single-speaker (text=) — fixes the NARR-1 gap ---

def test_render_single_speaker_uses_text_path(cleanup_wavs):
    client = _make_client()
    turns = [
        {'speaker': 'Anna', 'text': 'Erster Satz.'},
        {'speaker': 'Anna', 'text': 'Zweiter Satz.'},
    ]
    cleanup_wavs.append(render_turns(
        client, turns, {'Anna': 'Charon'}, mode='single_speaker', style_prompt='nachdenklich'
    ))

    si, voice, _ = _call(client)
    # Single speaker → text= path, no multi_speaker_markup.
    assert si.text == 'Erster Satz.\nZweiter Satz.'
    assert len(si.multi_speaker_markup.turns) == 0
    assert si.prompt == 'nachdenklich'
    # Voice correctly mapped via name= (the NARR-1 single-speaker fix).
    assert voice.name == 'Charon'
    assert voice.model_name == narration_render.DEFAULT_NARRATION_MODEL
    assert len(voice.multi_speaker_voice_config.speaker_voice_configs) == 0


def test_render_two_speaker_mode_with_one_distinct_uses_text_path(cleanup_wavs):
    """Discrepancy #5: exactly 1 distinct speaker always routes via text=."""
    client = _make_client()
    turns = [{'speaker': 'Solo', 'text': 'Allein hier.'}]
    cleanup_wavs.append(render_turns(client, turns, {'Solo': 'Kore'}, mode='two_speaker'))
    si, voice, _ = _call(client)
    assert si.text == 'Allein hier.'
    assert voice.name == 'Kore'
    assert len(si.multi_speaker_markup.turns) == 0


# --- render_turns: chunking carries prompt + speaker_configs on every chunk ---

def test_render_multi_chunk_same_prompt_and_configs_each_chunk(cleanup_wavs):
    client = _make_client()
    # 4 turns ~1200 bytes each → 2 chunks under the 3500-byte cap.
    big = 'Hallo Welt. ' * 100  # 1200 bytes, no single turn over the cap
    turns = [
        {'speaker': 'Anna', 'text': big},
        {'speaker': 'Ben', 'text': big},
        {'speaker': 'Anna', 'text': big},
        {'speaker': 'Ben', 'text': big},
    ]
    cleanup_wavs.append(render_turns(
        client, turns, {'Anna': 'Kore', 'Ben': 'Puck'},
        style_prompt='lebhaft', pydub_available=False,
    ))

    assert client.synthesize_speech.call_count == 2
    for idx in range(2):
        si, voice, _ = _call(client, idx)
        assert si.prompt == 'lebhaft'  # same prompt on every chunk
        cfgs = voice.multi_speaker_voice_config.speaker_voice_configs
        assert [(c.speaker_alias, c.speaker_id) for c in cfgs] == [('Speaker1', 'Kore'), ('Speaker2', 'Puck')]


# --- render_turns: WAV header-agnostic end-to-end (both fake shapes) ---

def test_render_returns_valid_wav_from_raw_pcm(cleanup_wavs):
    client = _make_client(audio_content=_FAKE_PCM)  # raw PCM, no RIFF
    path = render_turns(client, [{'speaker': 'A', 'text': 'hi'}], {'A': 'Kore'}, mode='single_speaker')
    cleanup_wavs.append(path)
    _assert_valid_wav(path)


def test_render_returns_valid_wav_from_riff(cleanup_wavs):
    client = _make_client(audio_content=_build_wav_bytes())  # already RIFF/WAV
    path = render_turns(client, [{'speaker': 'A', 'text': 'hi'}], {'A': 'Kore'}, mode='single_speaker')
    cleanup_wavs.append(path)
    _assert_valid_wav(path)


# --- render_turns: api_core error handling + retry ---

def test_render_invalid_argument_propagates_without_retry():
    client = _make_client(side_effect=gax.InvalidArgument('bad turn'))
    with pytest.raises(gax.InvalidArgument):
        render_turns(client, [{'speaker': 'A', 'text': 'hi'}], {'A': 'Kore'}, mode='single_speaker')
    client.synthesize_speech.assert_called_once()  # 400 never retried


def test_render_resource_exhausted_retries_then_succeeds(cleanup_wavs, monkeypatch):
    monkeypatch.setattr(narration_render.time, 'sleep', lambda *a, **k: None)
    client = _make_client(side_effect=[
        gax.ResourceExhausted('429'),
        SimpleNamespace(audio_content=_FAKE_PCM),
    ])
    path = render_turns(client, [{'speaker': 'A', 'text': 'hi'}], {'A': 'Kore'}, mode='single_speaker')
    cleanup_wavs.append(path)
    assert client.synthesize_speech.call_count == 2  # retried once, then ok
    _assert_valid_wav(path)


def test_render_empty_audio_is_hard_error(tmp_path):
    client = _make_client(audio_content=b'')  # empty → hard chunk failure
    with pytest.raises(ValueError):
        render_turns(client, [{'speaker': 'A', 'text': 'hi'}], {'A': 'Kore'}, mode='single_speaker')


def test_render_empty_audio_leaves_no_temp_file(cleanup_wavs):
    """First chunk ok, second returns empty → partial temp WAV cleaned up."""
    client = _make_client(side_effect=[
        SimpleNamespace(audio_content=_FAKE_PCM),
        SimpleNamespace(audio_content=b''),
    ])
    big = 'Hallo Welt. ' * 100
    turns = [
        {'speaker': 'Anna', 'text': big},
        {'speaker': 'Ben', 'text': big},
        {'speaker': 'Anna', 'text': big},
        {'speaker': 'Ben', 'text': big},
    ]
    import tempfile
    before = set(os.listdir(tempfile.gettempdir()))
    with pytest.raises(ValueError):
        render_turns(client, turns, {'Anna': 'Kore', 'Ben': 'Puck'}, pydub_available=False)
    after = set(os.listdir(tempfile.gettempdir()))
    # No new .wav temp files leaked by the failed render.
    leaked = [f for f in (after - before) if f.endswith('.wav')]
    assert leaked == []
