# services/gemini/tts.py
"""TTS pipeline entry: generate_podcast + chunking orchestration.

Caller-facing entry point ``generate_podcast`` validates input, builds
speaker voice configs, runs filter/split passes, then either calls into
``synthesis.generate_single_chunk`` directly or delegates to
``_generate_with_chunking`` which retries + concatenates via the
``audio`` module.
"""
import logging
import os
import time
from typing import Dict, List

from google.genai import types

from rq import get_current_job

from services.gemini.audio import concatenate_with_pydub, concatenate_with_wave
from services.gemini.dialogue import filter_metadata_lines, split_long_dialogue_turns
from services.gemini.synthesis import generate_single_chunk

logger = logging.getLogger(__name__)


def _set_stage(stage, **extras):
    """F-4.3 P3: write stage marker to job.meta. No-op outside RQ worker.

    Inlined here (not imported from tasks.py) to avoid the import cycle
    services.gemini.tts -> tasks -> services.GeminiService -> tts. Mirrors
    the public ``tasks.update_job_stage`` for the finalizing-stage call.
    """
    job = get_current_job()
    if job is None:
        return
    try:
        job.meta['stage'] = stage
        for key, value in extras.items():
            job.meta[key] = value
        job.save_meta()
    except Exception as e:
        logger.warning(f"_set_stage failed for stage={stage}: {e}")


# Available TTS models
TTS_MODELS = {
    'gemini-2.5-flash-preview-tts': 'Gemini 2.5 Flash TTS (newest)',
    'gemini-2.5-pro-preview-tts': 'Gemini 2.5 Pro TTS (higher quality)'
}
DEFAULT_TTS_MODEL = 'gemini-2.5-flash-preview-tts'

# Chunking-Konfiguration
MAX_LINES_PER_CHUNK = 80
MAX_CHARS_PER_CHUNK = 3000
CHUNK_OVERLAP_LINES = 2
INTER_CHUNK_DELAY = 1.0
RATE_LIMIT_DELAY = 0.4


def generate_podcast(client, dialogue, language='en', tts_model=None, pydub_available=True,
                     voices=None, filter_metadata=True):
    """
    Generate multi-speaker podcast audio using Gemini TTS with automatic chunking for long podcasts.

    Args:
        client: A configured ``google.genai.Client`` instance.
        dialogue: List of dicts with 'speaker', 'style', 'text'
        language: Language code
        tts_model: TTS model to use (default: gemini-2.5-flash-preview-tts)
        pydub_available: Whether to use PyDub for audio concat (False → wave fallback)
        voices: Optional ``{label: voice_name}`` map decoupling the transcript
            speaker label from the Gemini voice. ``None`` (default) keeps the
            historical behaviour where the label *is* the voice name
            (byte-identical for the old podcast flow).
        filter_metadata: When ``True`` (default) drop URL/caption/short-line
            metadata via ``filter_metadata_lines``. The faithful narration path
            passes ``False`` so legitimate turns are never dropped (fidelity).

    Returns:
        str: Path to temporary WAV file (concatenated if chunked)
    """
    # Validate TTS model
    if not tts_model or tts_model not in TTS_MODELS:
        tts_model = DEFAULT_TTS_MODEL

    logger.info(f"Using TTS model: {tts_model}")
    if not dialogue or len(dialogue) == 0:
        raise ValueError("No dialogue provided")

    # Build speaker voice configs and prepare dialogue
    speaker_voice_configs = []
    seen_speakers = set()
    dialogue_lines = []

    for turn in dialogue:
        speaker = turn.get('speaker', 'Kore')
        text = turn.get('text', '').strip()
        style = turn.get('style', '').strip()

        if not text:
            continue

        if speaker not in seen_speakers:
            voice_name = voices.get(speaker, speaker) if voices else speaker
            speaker_voice_configs.append(
                types.SpeakerVoiceConfig(
                    speaker=speaker,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                )
            )
            seen_speakers.add(speaker)

        dialogue_lines.append({
            'speaker': speaker,
            'style': style,
            'text': text
        })

    logger.info(f"Raw dialogue: {len(dialogue_lines)} turns")

    _set_stage('filtering')

    # Filter and process with statistics
    raw_count = len(dialogue_lines)
    if filter_metadata:
        dialogue_lines = filter_metadata_lines(dialogue_lines)
    filtered_count = raw_count - len(dialogue_lines)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} metadata lines")

    pre_split = len(dialogue_lines)
    dialogue_lines = split_long_dialogue_turns(dialogue_lines, max_words=50)
    if len(dialogue_lines) > pre_split:
        logger.info(f"Split long turns: {pre_split} -> {len(dialogue_lines)} lines")

    if not dialogue_lines:
        raise ValueError("No valid dialogue lines after filtering")

    logger.info(f"Final dialogue: {len(dialogue_lines)} turns")

    # Check if we need to chunk - with detailed logging
    total_chars = sum(len(line.get('text', '')) for line in dialogue_lines)
    logger.info(f"=== CHUNKING DECISION ===")
    logger.info(f"  Lines: {len(dialogue_lines)} / {MAX_LINES_PER_CHUNK} max")
    logger.info(f"  Chars: {total_chars} / {MAX_CHARS_PER_CHUNK} max")

    if len(dialogue_lines) <= MAX_LINES_PER_CHUNK:
        logger.info(f"  -> Single chunk (Zeilen unter Schwellwert)")
        _set_stage('tts_chunk', chunk_current=1, chunk_total=1)
        return generate_single_chunk(client, dialogue_lines, speaker_voice_configs, tts_model, DEFAULT_TTS_MODEL)
    else:
        logger.info(f"  -> Multi-chunk erforderlich!")
        _set_stage('chunking')
        return _generate_with_chunking(client, dialogue_lines, speaker_voice_configs, tts_model, pydub_available)


def synthesize_turns(client, turns, voices, *, mode='two_speaker', language='de',
                     tts_model=None, pydub_available=True):
    """Faithful narration entry: render a structured turn list to audio verbatim.

    Unlike ``generate_podcast``'s historical callers, this path bypasses the
    script-generation/tag machinery entirely — the agent (NARR-4) supplies the
    turns directly. It validates the contract, maps turns onto the existing
    synth shape, then delegates to ``generate_podcast`` with label→voice
    decoupling on and the metadata filter **off** (fidelity: no turn dropped).

    Args:
        client: A configured ``google.genai.Client`` instance.
        turns: Non-empty list of ``{'speaker': str, 'text': str}``. Performance
            tags stay **inline** in ``text`` (e.g. ``"[ruhig] …"``) and are
            passed through verbatim; there is no separate style field in v1.
        voices: ``{label: voice_name}`` map; must cover every distinct speaker.
        mode: ``'single_speaker'`` (exactly 1 distinct speaker) or
            ``'two_speaker'`` (1–2 distinct speakers). Validation only — the
            synth already branches on ``len(unique_speakers)``.
        language: Language code (default ``'de'``).
        tts_model: TTS model (``generate_podcast`` defaults it when falsy).
        pydub_available: Whether PyDub is available for chunk concat.

    Returns:
        str: Path to temporary WAV file (concatenated if chunked).

    Raises:
        ValueError: on any contract violation (empty turns, blank
            speaker/text, mode/speaker-count mismatch, ``voices`` not a dict
            or not covering all speakers).
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

    # Map onto the synth shape. Tags stay inline in text; style stays empty so
    # the transcript line is "{speaker}: {text}" with no extra bracketing.
    dialogue = [{'speaker': t['speaker'], 'style': '', 'text': t['text']} for t in turns]

    return generate_podcast(
        client, dialogue,
        language=language,
        tts_model=tts_model,
        pydub_available=pydub_available,
        voices=voices,
        filter_metadata=False,
    )


def _generate_with_chunking(client, dialogue_lines: List[Dict], speaker_voice_configs: List,
                             tts_model: str, pydub_available: bool):
    """Generate audio in chunks and concatenate them."""

    # Split into chunks
    chunks = _create_dialogue_chunks(dialogue_lines)
    total_chunks = len(chunks)

    logger.info(f"======================================")
    logger.info(f"=== MULTI-CHUNK PODCAST GENERATION ===")
    logger.info(f"======================================")
    logger.info(f"Total dialogue lines: {len(dialogue_lines)}")
    logger.info(f"Split into {total_chunks} chunks")
    logger.info(f"Chunk config: max {MAX_LINES_PER_CHUNK} lines, max {MAX_CHARS_PER_CHUNK} chars")

    # Estimate total time
    total_chars = sum(len(f"{l['speaker']}: {l['text']}") for l in dialogue_lines)
    estimated_audio_minutes = total_chars / 800  # rough estimate: 800 chars ~= 1 min audio
    logger.info(f"Total input: {total_chars} chars (~{estimated_audio_minutes:.1f} min audio estimated)")

    # Generate audio for each chunk
    chunk_files = []
    chunk_times = []
    total_start_time = time.time()

    max_retries = 2
    retry_delay = 5.0  # seconds to wait before retry

    for i, chunk in enumerate(chunks):
        chunk_chars = sum(len(f"{l['speaker']}: {l['text']}") for l in chunk)
        logger.info(f"")
        logger.info(f">>> CHUNK {i+1}/{total_chunks} <<<")
        logger.info(f"Lines: {len(chunk)}, Chars: {chunk_chars}")

        _set_stage('tts_chunk', chunk_current=i + 1, chunk_total=total_chunks)

        chunk_start_time = time.time()
        chunk_file = None
        last_error = None

        # Retry loop for this chunk
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.warning(f"Retry {attempt}/{max_retries} for chunk {i+1}")
                    time.sleep(retry_delay * attempt)  # Progressive backoff

                # Generate this chunk
                chunk_file = generate_single_chunk(client, chunk, speaker_voice_configs, tts_model, DEFAULT_TTS_MODEL)
                break  # Success, exit retry loop

            except Exception as e:
                last_error = e
                logger.warning(f"Chunk {i+1} attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:100]}")
                if attempt < max_retries:
                    logger.info(f"Will retry in {retry_delay * (attempt + 1)}s...")
                continue

        if chunk_file is None:
            # All retries failed
            chunk_elapsed = time.time() - chunk_start_time
            total_elapsed = time.time() - total_start_time
            logger.error(f"")
            logger.error(f"!!! CHUNK {i+1}/{total_chunks} FAILED AFTER {max_retries+1} ATTEMPTS !!!")
            logger.error(f"Failed after {chunk_elapsed:.1f}s (total: {total_elapsed:.1f}s)")
            logger.error(f"Error type: {type(last_error).__name__}")
            logger.error(f"Error: {str(last_error)}")
            logger.error(f"Completed chunks before failure: {len(chunk_files)}")
            # Clean up partial chunks
            for f in chunk_files:
                if os.path.exists(f):
                    os.unlink(f)
            raise last_error

        chunk_files.append(chunk_file)
        chunk_elapsed = time.time() - chunk_start_time
        chunk_times.append(chunk_elapsed)

        # Progress summary
        total_elapsed = time.time() - total_start_time
        avg_chunk_time = sum(chunk_times) / len(chunk_times)
        remaining_chunks = total_chunks - (i + 1)
        estimated_remaining = remaining_chunks * avg_chunk_time

        logger.info(f"Chunk {i+1} completed in {chunk_elapsed:.1f}s")
        logger.info(f"Progress: {i+1}/{total_chunks} chunks ({(i+1)/total_chunks*100:.0f}%)")
        logger.info(f"Total elapsed: {total_elapsed:.1f}s, Est. remaining: {estimated_remaining:.1f}s")

        # Rate limiting delay
        if i < len(chunks) - 1:
            logger.info(f"Waiting {INTER_CHUNK_DELAY}s before next chunk...")
            time.sleep(max(RATE_LIMIT_DELAY, INTER_CHUNK_DELAY))

    total_generation_time = time.time() - total_start_time
    logger.info(f"")
    logger.info(f"=== ALL CHUNKS GENERATED ===")
    logger.info(f"Total generation time: {total_generation_time:.1f}s ({total_generation_time/60:.1f} min)")
    logger.info(f"Average chunk time: {sum(chunk_times)/len(chunk_times):.1f}s")
    logger.info(f"Chunk times: {[f'{t:.1f}s' for t in chunk_times]}")

    _set_stage('concatenating')

    # Concatenate chunks
    if pydub_available:
        return concatenate_with_pydub(chunk_files)
    else:
        return concatenate_with_wave(chunk_files)


def _create_dialogue_chunks(dialogue_lines: List[Dict]) -> List[List[Dict]]:
    """Split dialogue into overlapping chunks for consistent voice generation."""

    chunks = []
    current_chunk = []
    current_char_count = 0

    for i, line in enumerate(dialogue_lines):
        line_chars = len(f"{line['speaker']}: {line['text']}")

        # Check if adding this line would exceed limits
        if (len(current_chunk) >= MAX_LINES_PER_CHUNK or
                current_char_count + line_chars > MAX_CHARS_PER_CHUNK) and current_chunk:

            # Save current chunk
            chunks.append(current_chunk.copy())

            # Start new chunk with overlap
            if CHUNK_OVERLAP_LINES > 0 and len(current_chunk) > CHUNK_OVERLAP_LINES:
                # Include last N lines from previous chunk for voice consistency
                current_chunk = current_chunk[-CHUNK_OVERLAP_LINES:]
                current_char_count = sum(
                    len(f"{l['speaker']}: {l['text']}") for l in current_chunk
                )
            else:
                current_chunk = []
                current_char_count = 0

        current_chunk.append(line)
        current_char_count += line_chars

    # Add remaining lines as final chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Log chunk statistics
    for i, chunk in enumerate(chunks):
        chars = sum(len(f"{l['speaker']}: {l['text']}") for l in chunk)
        logger.info(f"Chunk {i+1}: {len(chunk)} lines, {chars} characters")

    return chunks
