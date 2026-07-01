"""Cloud-TTS faithful-narration renderer (NARR-1B).

Pure module — the ``texttospeech`` client is passed in as an argument so the
whole surface is mockable at the ``client.synthesize_speech`` boundary. This is
the v1 render engine for the faithful-narration path: it takes a structured turn
list plus a label→voice map and renders it to a WAV file via Google **Cloud**
TTS (``google-cloud-texttospeech``), *not* the ``google.genai`` API path.

Why Cloud TTS (NARR-1B pivot, 2026-06-29): it natively solves three things the
genai path could not —

1. **Label↔Voice separation** — the human label (``Turn.speaker``) is decoupled
   from the Gemini voice via ``speaker_alias`` (a generated ``Speaker1``/
   ``Speaker2`` join-key) vs ``speaker_id`` (the voice). Single-speaker is
   rendered through the ``text=`` path with an explicit ``name=`` voice, which
   fixes the NARR-1 single-speaker gap (voice was keyed off the label).
2. **Director's notes without leakage** — ``style_prompt`` rides on its own
   ``SynthesisInput.prompt`` field, structurally separate from the transcript;
   it is never concatenated into the spoken text.
3. **Documented multi-speaker markup** — ``MultiSpeakerMarkup`` turns.

Defensive by construction (resolves the source-discrepancy flags from the
sprint): the model name is configurable, WAV wrapping is header-agnostic, and
chunking is utf-8-byte-based with conservative headroom.
"""
import io
import logging
import os
import re
import tempfile
import time
import wave

from google.api_core import exceptions as gax
from google.cloud import texttospeech

from app_pkg.config import TIMEOUT_TTS_SYNTH_SECONDS
from services.gemini.audio import concatenate_with_pydub, concatenate_with_wave

logger = logging.getLogger(__name__)


# Cloud-TTS Gemini model name. Note: **no** ``-preview-`` infix, unlike the
# genai-path name (``gemini-2.5-flash-preview-tts``). Env-overridable so the
# Phase-3 live-verify finding can be pinned without a code change.
DEFAULT_NARRATION_MODEL = os.environ.get('NARRATION_TTS_MODEL') or 'gemini-2.5-flash-tts'

# Conservative utf-8-byte cap per chunk's transcript (headroom under the
# 4000-byte cap one source documents; German umlauts/ß are multi-byte, so this
# is measured in bytes, never ``len``). Whole turns only — never split mid-turn.
MAX_TRANSCRIPT_BYTES = 3500

# LINEAR16 PCM @ 24 kHz mono 16-bit — the Gemini-TTS output contract.
_SAMPLE_RATE_HZ = 24000
_CHANNELS = 1
_SAMPLE_WIDTH = 2

# Retryable api_core errors (transient): backoff + retry. 400/404 are caller
# errors and propagate immediately (never retried).
_RETRYABLE = (gax.ResourceExhausted, gax.ServiceUnavailable, gax.DeadlineExceeded)


def validate_turns(turns, voices, mode):
    """Validate the faithful-narration contract; return distinct speakers.

    Pulled verbatim from NARR-1's ``synthesize_turns`` so the contract lives in
    one pure place. Raises ``ValueError`` on any violation. Returns the
    order-preserving list of distinct speaker labels (reused by the renderer for
    positional alias assignment).
    """
    if mode not in ('single_speaker', 'two_speaker'):
        raise ValueError(f"Unknown mode: {mode!r} (expected 'single_speaker' or 'two_speaker')")

    if not isinstance(turns, list) or not turns:
        raise ValueError("turns must be a non-empty list")

    speakers = []
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {i} must be a dict with 'speaker' and 'text'")
        speaker = turn.get('speaker')
        text = turn.get('text')
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"Turn {i} has a missing or blank 'speaker'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Turn {i} has a missing or blank 'text'")
        speakers.append(speaker)

    # Distinct speakers, order-preserving.
    distinct_speakers = list(dict.fromkeys(speakers))

    if mode == 'single_speaker' and len(distinct_speakers) != 1:
        raise ValueError(
            f"single_speaker mode requires exactly 1 distinct speaker, got {len(distinct_speakers)}"
        )
    if mode == 'two_speaker' and not (1 <= len(distinct_speakers) <= 2):
        raise ValueError(
            f"two_speaker mode requires 1–2 distinct speakers, got {len(distinct_speakers)}"
        )

    if not isinstance(voices, dict):
        raise ValueError("voices must be a dict mapping speaker label -> voice name")
    missing = [s for s in distinct_speakers if s not in voices]
    if missing:
        raise ValueError(f"voices is missing an entry for speaker(s): {missing}")

    return distinct_speakers


def _split_oversized_turn(turn, max_bytes):
    """Split a single over-cap turn at sentence boundaries into same-speaker turns.

    Byte-based: a turn whose text exceeds ``max_bytes`` utf-8 is broken at
    sentence punctuation into several turns carrying the same speaker. A single
    sentence over the cap is left whole (best-effort) rather than cut mid-word.
    """
    text = turn['text']
    if len(text.encode('utf-8')) <= max_bytes:
        return [turn]

    # Capture the delimiters so sentence terminators stay attached.
    segments = re.split(r'([.!?…]+\s+)', text)

    chunks = []
    current = ""
    current_bytes = 0
    for seg in segments:
        if not seg:
            continue
        seg_bytes = len(seg.encode('utf-8'))
        if current_bytes > 0 and current_bytes + seg_bytes > max_bytes:
            if current.strip():
                chunks.append(current.strip())
            current = seg
            current_bytes = seg_bytes
        else:
            current += seg
            current_bytes += seg_bytes
    if current.strip():
        chunks.append(current.strip())

    if not chunks:
        return [turn]
    return [{'speaker': turn['speaker'], 'text': c} for c in chunks]


def chunk_turns(turns, max_bytes=MAX_TRANSCRIPT_BYTES):
    """Group whole turns into ~``max_bytes`` utf-8 transcript chunks.

    Two passes: (1) expand any single over-cap turn at sentence boundaries into
    same-speaker turns, then (2) accumulate whole turns until the byte budget
    would be exceeded. Never splits mid-turn. utf-8 byte measurement keeps
    German umlauts/ß from under-counting against the cap.
    """
    expanded = []
    for turn in turns:
        expanded.extend(_split_oversized_turn(turn, max_bytes))

    chunks = []
    current = []
    current_bytes = 0
    for turn in expanded:
        tb = len(turn['text'].encode('utf-8'))
        if current and current_bytes + tb > max_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(turn)
        current_bytes += tb
    if current:
        chunks.append(current)

    return chunks


def pcm_to_wav_bytes(audio_content, rate=_SAMPLE_RATE_HZ, ch=_CHANNELS, width=_SAMPLE_WIDTH):
    """Wrap synth output to WAV bytes, header-agnostically.

    The sources disagree on whether LINEAR16 ``audio_content`` already carries a
    RIFF/WAV header. So: if it starts with ``b'RIFF'`` it is already a WAV
    container and is returned unchanged; otherwise it is raw PCM and gets wrapped
    via the stdlib ``wave`` module (mono, 16-bit, 24 kHz by default). Either way
    the result is a valid WAV byte string. Empty input is a hard error.
    """
    if not audio_content:
        raise ValueError("empty audio_content — cannot wrap to WAV")

    if audio_content[:4] == b'RIFF':
        return audio_content

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(ch)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(audio_content)
    return buf.getvalue()


def _synthesize_with_retry(client, synthesis_input, voice, audio_config,
                           max_retries=2, base_delay=1.0, *, timeout=None):
    """Call ``client.synthesize_speech`` with backoff on transient errors.

    Retries (progressive backoff) on 429/503/DeadlineExceeded only. 400
    (``InvalidArgument``) / 404 (``NotFound``) propagate immediately. Empty
    ``audio_content`` is a hard chunk failure (never silently concatenated as
    silence) and propagates as ``ValueError`` without retry.

    ``timeout`` is the absolute per-call gRPC deadline (seconds) forwarded to the
    SDK; on expiry the C-core timer raises ``DeadlineExceeded`` (already in
    ``_RETRYABLE``), so a wedged call can no longer park the RQ work-horse.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(base_delay * attempt)
                logger.warning(f"Retry {attempt}/{max_retries} for synthesize_speech")
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config, timeout=timeout
            )
            audio_content = response.audio_content
            if not audio_content:
                raise ValueError("synthesize_speech returned empty audio_content (hard chunk failure)")
            return audio_content
        except _RETRYABLE as e:
            last_error = e
            logger.warning(f"Transient TTS error (attempt {attempt + 1}): {type(e).__name__}: {e}")
            continue
    raise last_error


def _build_chunk_request(chunk, alias_for, voices, speaker_configs, *,
                         style_prompt, language_code, model_name):
    """Build ``(SynthesisInput, VoiceSelectionParams)`` for one chunk.

    Routing is decided **per chunk** by the chunk's own distinct-speaker count,
    not the piece's global count: a chunk that uses a single speaker renders via
    the ``text=`` path (with an explicit ``name=`` voice), a chunk that uses two
    renders via ``multi_speaker_markup``. This keeps a single-speaker chunk
    inside a two-speaker piece — e.g. a long monolog that fills a whole chunk —
    from declaring an alias in its ``MultiSpeakerVoiceConfig`` that the chunk's
    markup never uses. Voice identity is preserved across both paths (the same
    ``voice_id`` per speaker), and the style ``prompt`` rides on every chunk.
    """
    chunk_speakers = list(dict.fromkeys(t['speaker'] for t in chunk))

    if len(chunk_speakers) == 1:
        single_label = chunk_speakers[0]
        # One speaker → join the chunk's turn texts into a single blob and
        # render via text= with an explicit voice name (the NARR-1 fix).
        text = "\n".join(t['text'] for t in chunk)
        synthesis_input = texttospeech.SynthesisInput(text=text, prompt=style_prompt or None)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voices[single_label],
            model_name=model_name,
        )
        return synthesis_input, voice

    markup = texttospeech.MultiSpeakerMarkup(turns=[
        texttospeech.MultiSpeakerMarkup.Turn(speaker=alias_for[t['speaker']], text=t['text'])
        for t in chunk
    ])
    synthesis_input = texttospeech.SynthesisInput(
        multi_speaker_markup=markup, prompt=style_prompt or None
    )
    # No name= here — name + multi_speaker_voice_config are mutually exclusive.
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        model_name=model_name,
        multi_speaker_voice_config=texttospeech.MultiSpeakerVoiceConfig(
            speaker_voice_configs=speaker_configs
        ),
    )
    return synthesis_input, voice


def render_turns(client, turns, voices, *, style_prompt=None, mode='two_speaker',
                 language_code='de-DE', model_name=DEFAULT_NARRATION_MODEL,
                 pydub_available=True, synth_timeout=TIMEOUT_TTS_SYNTH_SECONDS):
    """Render a structured turn list to a WAV file via Cloud Gemini TTS.

    Args:
        client: A ``texttospeech.TextToSpeechClient`` (or a mock exposing
            ``synthesize_speech``).
        turns: Non-empty list of ``{'speaker': label, 'text': str}``. Performance
            tags stay **inline** in ``text`` and are passed through verbatim.
        voices: ``{label: voice_id}`` covering every distinct speaker; the voice
            is the Gemini ``speaker_id`` (e.g. ``'Kore'``).
        style_prompt: Optional director's note → ``SynthesisInput.prompt`` (its
            own field; never concatenated into the transcript). Falsy → unset.
        mode: ``'single_speaker'`` (exactly 1 label) or ``'two_speaker'`` (1–2);
            validation only. **Synthesis routing is decided per chunk** by the
            chunk's own distinct-speaker count (see ``_build_chunk_request``): a
            chunk using one speaker renders via ``text=``, a chunk using two via
            ``multi_speaker_markup``. So a single-speaker chunk inside a
            two-speaker piece (e.g. a long monolog) never declares an unused
            alias, and the NARR-1 single-speaker voice gap stays fixed.
        language_code: BCP-47 code (default ``'de-DE'``).
        model_name: Gemini-TTS model. **Required** to activate Gemini TTS; set on
            ``VoiceSelectionParams`` for both paths.
        pydub_available: Use PyDub for multi-chunk concat (else ``wave`` fallback).
        synth_timeout: Absolute per-call gRPC deadline (seconds) forwarded to
            every ``synthesize_speech`` call; on expiry the SDK raises
            ``DeadlineExceeded`` (already retryable). Defaults to the shared
            ``TIMEOUT_TTS_SYNTH_SECONDS`` so the enforced deadline and the RQ
            job envelope stay derived from a single source.

    Returns:
        str: Path to a temporary WAV file (concatenated when chunked).

    Raises:
        ValueError: contract violation (see ``validate_turns``) or empty audio.
        google.api_core.exceptions.GoogleAPICallError: on a non-retryable or
            exhausted-retry API failure.
    """
    distinct_speakers = validate_turns(turns, voices, mode)

    # Verbatim text; ignore any incoming style field (tags live inline in text).
    norm_turns = [{'speaker': t['speaker'], 'text': t['text']} for t in turns]

    # Positional alias = collision-free + guaranteed alphanumeric (no whitespace),
    # as required for ``speaker_alias``. The human label never reaches the API as
    # an alias; it is only a join-key into ``alias_for`` / ``voices``.
    alias_for = {label: f"Speaker{i + 1}" for i, label in enumerate(distinct_speakers)}

    chunks = chunk_turns(norm_turns)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=_SAMPLE_RATE_HZ,
    )

    # Multi-speaker configs — declared once and identical on every multi-speaker
    # chunk (voice stability). A two-speaker piece has exactly these two distinct
    # speakers, so any chunk routed to the markup path uses both aliases; a chunk
    # that uses only one routes to text= instead (no unused-alias declaration).
    speaker_configs = [
        texttospeech.MultispeakerPrebuiltVoice(
            speaker_alias=alias_for[label], speaker_id=voices[label]
        )
        for label in distinct_speakers
    ]

    chunk_files = []
    try:
        for chunk in chunks:
            synthesis_input, voice = _build_chunk_request(
                chunk, alias_for, voices, speaker_configs,
                style_prompt=style_prompt, language_code=language_code, model_name=model_name,
            )
            audio_content = _synthesize_with_retry(
                client, synthesis_input, voice, audio_config, timeout=synth_timeout
            )
            wav_bytes = pcm_to_wav_bytes(audio_content)

            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.write(wav_bytes)
            tmp.close()
            chunk_files.append(tmp.name)
    except Exception:
        # Clean up partial chunk files; never leave temp WAVs behind on failure.
        for f in chunk_files:
            if os.path.exists(f):
                os.unlink(f)
        raise

    if len(chunk_files) == 1:
        return chunk_files[0]
    if pydub_available:
        return concatenate_with_pydub(chunk_files)
    return concatenate_with_wave(chunk_files)
