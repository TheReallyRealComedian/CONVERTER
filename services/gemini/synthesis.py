# services/gemini/synthesis.py
"""Single-chunk Gemini TTS synthesis.

Holds the workhorse function that performs one Gemini TTS API call,
validates the response, and writes the audio bytes to a temporary WAV
file. Split out from ``tts.py`` because it is the largest single block
in the package (~165 LOC) and has its own validation/logging surface.
"""
import logging
import tempfile
import time
import wave
from typing import Dict, List

from google.genai import types

logger = logging.getLogger(__name__)


# Imported lazily inside ``generate_single_chunk`` via the explicit argument
# so this module does not need to know about TTS_MODELS / DEFAULT_TTS_MODEL.


def generate_single_chunk(client, dialogue_lines: List[Dict], speaker_voice_configs: List,
                           tts_model: str, default_tts_model: str):
    """Generate audio for a single chunk of dialogue."""

    # Format for TTS
    formatted_lines = []
    unique_speakers = set()

    for turn in dialogue_lines:
        unique_speakers.add(turn['speaker'])
        if turn['style']:
            formatted_lines.append(f"{turn['speaker']}: [{turn['style']}] {turn['text']}")
        else:
            formatted_lines.append(f"{turn['speaker']}: {turn['text']}")

    full_dialogue = "\n".join(formatted_lines)
    unique_speakers = list(unique_speakers)

    # Calculate input size for logging
    input_chars = len(full_dialogue)
    input_words = len(full_dialogue.split())

    logger.info(f"=== TTS API CALL START ===")
    logger.info(f"Input: {len(formatted_lines)} lines, {len(unique_speakers)} speakers")
    logger.info(f"Input size: {input_chars} chars, {input_words} words")
    logger.info(f"Speakers: {unique_speakers}")

    # Start timing
    api_start_time = time.time()

    # Generate audio
    try:
        # Use the selected TTS model
        tts_model = tts_model or default_tts_model
        logger.info(f"TTS Model: {tts_model}")

        if len(unique_speakers) == 1:
            logger.info("Mode: single-speaker")
            response = client.models.generate_content(
                model=tts_model,
                contents=full_dialogue,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=unique_speakers[0]
                            )
                        )
                    )
                )
            )
        else:
            logger.info("Mode: multi-speaker")
            # Filter speaker_voice_configs to only include speakers in this chunk
            chunk_speaker_configs = [
                config for config in speaker_voice_configs
                if config.speaker in unique_speakers
            ]

            response = client.models.generate_content(
                model=tts_model,
                contents=full_dialogue,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=chunk_speaker_configs
                        )
                    )
                )
            )

        api_elapsed = time.time() - api_start_time
        logger.info(f"=== TTS API CALL SUCCESS ===")
        logger.info(f"API response time: {api_elapsed:.2f}s")

        # Log response metadata if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            um = response.usage_metadata
            logger.info(f"Usage metadata: prompt_tokens={getattr(um, 'prompt_token_count', 'N/A')}, "
                       f"candidates_tokens={getattr(um, 'candidates_token_count', 'N/A')}, "
                       f"total_tokens={getattr(um, 'total_token_count', 'N/A')}")

        # Log candidate info
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                logger.info(f"Finish reason: {candidate.finish_reason}")
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                logger.info(f"Safety ratings: {candidate.safety_ratings}")

    except Exception as e:
        api_elapsed = time.time() - api_start_time
        logger.error(f"=== TTS API CALL FAILED ===")
        logger.error(f"API failed after: {api_elapsed:.2f}s")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Response status: {getattr(e.response, 'status_code', 'N/A')}")
            logger.error(f"Response text: {getattr(e.response, 'text', 'N/A')[:500]}")
        raise

    # Detailed response validation with logging
    if not response:
        logger.error("Response is None")
        raise ValueError("No response from Gemini-TTS")

    if not response.candidates:
        logger.error("No candidates in response")
        # Log any available info about the response
        if hasattr(response, 'prompt_feedback'):
            logger.error(f"Prompt feedback: {response.prompt_feedback}")
        raise ValueError("No candidates in Gemini-TTS response")

    candidate = response.candidates[0]

    # Log candidate details for debugging
    logger.info(f"Candidate finish_reason: {getattr(candidate, 'finish_reason', 'N/A')}")
    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
        for rating in candidate.safety_ratings:
            logger.info(f"Safety: {rating.category} = {rating.probability}")

    if not candidate.content:
        logger.error("Candidate has no content!")
        logger.error(f"Full candidate object: {candidate}")
        if hasattr(candidate, 'finish_reason'):
            logger.error(f"Finish reason: {candidate.finish_reason}")
        raise ValueError(f"Gemini returned empty content. Finish reason: {getattr(candidate, 'finish_reason', 'unknown')}")

    if not candidate.content.parts:
        logger.error("Content has no parts!")
        logger.error(f"Content object: {candidate.content}")
        raise ValueError("Gemini content has no parts")

    if not hasattr(candidate.content.parts[0], 'inline_data') or not candidate.content.parts[0].inline_data:
        logger.error("No inline_data in parts!")
        logger.error(f"Parts[0]: {candidate.content.parts[0]}")
        raise ValueError("Gemini response has no audio data (inline_data missing)")

    audio_data = candidate.content.parts[0].inline_data.data
    mime_type = candidate.content.parts[0].inline_data.mime_type

    if not audio_data:
        logger.error("inline_data.data is empty")
        raise ValueError("No audio data in response")

    # Calculate audio duration estimate (24kHz, 16-bit mono = 48000 bytes/sec)
    audio_duration_sec = len(audio_data) / 48000
    logger.info(f"Audio received: {len(audio_data)} bytes ({len(audio_data)/1024:.1f} KB)")
    logger.info(f"Audio duration: ~{audio_duration_sec:.1f}s ({audio_duration_sec/60:.1f} min)")
    logger.info(f"Audio format: {mime_type}")
    logger.info(f"Processing ratio: {api_elapsed:.1f}s API time → {audio_duration_sec:.1f}s audio")

    # Convert to WAV
    temp_audio_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

    with wave.open(temp_audio_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_data)

    logger.info(f"Audio saved to: {temp_audio_path}")

    return temp_audio_path
