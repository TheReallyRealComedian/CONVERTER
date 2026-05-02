# services/gemini_service.py - Mit dynamischer Tag-Berechnung
import logging
import tempfile
import os
import wave
import time
from typing import List, Dict, Optional
from google.genai import types

from services.gemini.client import create_client, is_pydub_available
from services.gemini.dialogue import (
    filter_metadata_lines,
    split_long_dialogue_turns,
)
from services.gemini.script import format_dialogue_with_llm as _format_dialogue_with_llm

logger = logging.getLogger(__name__)


class GeminiService:
    # Chunking-Konfiguration
    MAX_LINES_PER_CHUNK = 80
    MAX_CHARS_PER_CHUNK = 3000
    CHUNK_OVERLAP_LINES = 2
    INTER_CHUNK_DELAY = 1.0
    RATE_LIMIT_DELAY = 0.4

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = create_client(api_key)
        self.pydub_available = is_pydub_available()

    def format_dialogue_with_llm(self, raw_text, num_speakers=2, speaker_descriptions=None,
                                  language='en', narration_style='conversational',
                                  script_length='medium', custom_prompt=None):
        return _format_dialogue_with_llm(
            self.client, raw_text,
            num_speakers=num_speakers,
            speaker_descriptions=speaker_descriptions,
            language=language,
            narration_style=narration_style,
            script_length=script_length,
            custom_prompt=custom_prompt,
        )

    # Available TTS models
    TTS_MODELS = {
        'gemini-2.5-flash-preview-tts': 'Gemini 2.5 Flash TTS (newest)',
        'gemini-2.5-pro-preview-tts': 'Gemini 2.5 Pro TTS (higher quality)'
    }
    DEFAULT_TTS_MODEL = 'gemini-2.5-flash-preview-tts'

    def generate_podcast(self, dialogue, language='en', tts_model=None):
        """
        Generate multi-speaker podcast audio using Gemini TTS with automatic chunking for long podcasts.

        Args:
            dialogue: List of dicts with 'speaker', 'style', 'text'
            language: Language code
            tts_model: TTS model to use (default: gemini-2.5-flash-preview-tts)

        Returns:
            str: Path to temporary WAV file (concatenated if chunked)
        """
        # Validate TTS model
        if not tts_model or tts_model not in self.TTS_MODELS:
            tts_model = self.DEFAULT_TTS_MODEL

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
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=speaker
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

        # Filter and process with statistics
        raw_count = len(dialogue_lines)
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
        logger.info(f"  Lines: {len(dialogue_lines)} / {self.MAX_LINES_PER_CHUNK} max")
        logger.info(f"  Chars: {total_chars} / {self.MAX_CHARS_PER_CHUNK} max")

        if len(dialogue_lines) <= self.MAX_LINES_PER_CHUNK:
            logger.info(f"  -> Single chunk (Zeilen unter Schwellwert)")
            return self._generate_single_chunk(dialogue_lines, speaker_voice_configs, tts_model)
        else:
            logger.info(f"  -> Multi-chunk erforderlich!")
            return self._generate_with_chunking(dialogue_lines, speaker_voice_configs, tts_model)

    def _generate_single_chunk(self, dialogue_lines: List[Dict], speaker_voice_configs: List, tts_model: str = None):
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
            tts_model = tts_model or self.DEFAULT_TTS_MODEL
            logger.info(f"TTS Model: {tts_model}")

            if len(unique_speakers) == 1:
                logger.info("Mode: single-speaker")
                response = self.client.models.generate_content(
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

                response = self.client.models.generate_content(
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

    def _generate_with_chunking(self, dialogue_lines: List[Dict], speaker_voice_configs: List, tts_model: str = None):
        """Generate audio in chunks and concatenate them."""

        # Split into chunks
        chunks = self._create_dialogue_chunks(dialogue_lines)
        total_chunks = len(chunks)

        logger.info(f"======================================")
        logger.info(f"=== MULTI-CHUNK PODCAST GENERATION ===")
        logger.info(f"======================================")
        logger.info(f"Total dialogue lines: {len(dialogue_lines)}")
        logger.info(f"Split into {total_chunks} chunks")
        logger.info(f"Chunk config: max {self.MAX_LINES_PER_CHUNK} lines, max {self.MAX_CHARS_PER_CHUNK} chars")

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
                    chunk_file = self._generate_single_chunk(chunk, speaker_voice_configs, tts_model)
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
                logger.info(f"Waiting {self.INTER_CHUNK_DELAY}s before next chunk...")
                time.sleep(max(self.RATE_LIMIT_DELAY, self.INTER_CHUNK_DELAY))

        total_generation_time = time.time() - total_start_time
        logger.info(f"")
        logger.info(f"=== ALL CHUNKS GENERATED ===")
        logger.info(f"Total generation time: {total_generation_time:.1f}s ({total_generation_time/60:.1f} min)")
        logger.info(f"Average chunk time: {sum(chunk_times)/len(chunk_times):.1f}s")
        logger.info(f"Chunk times: {[f'{t:.1f}s' for t in chunk_times]}")

        # Concatenate chunks
        if self.pydub_available:
            return self._concatenate_with_pydub(chunk_files)
        else:
            return self._concatenate_with_wave(chunk_files)

    def _create_dialogue_chunks(self, dialogue_lines: List[Dict]) -> List[List[Dict]]:
        """Split dialogue into overlapping chunks for consistent voice generation."""

        chunks = []
        current_chunk = []
        current_char_count = 0

        for i, line in enumerate(dialogue_lines):
            line_chars = len(f"{line['speaker']}: {line['text']}")

            # Check if adding this line would exceed limits
            if (len(current_chunk) >= self.MAX_LINES_PER_CHUNK or
                    current_char_count + line_chars > self.MAX_CHARS_PER_CHUNK) and current_chunk:

                # Save current chunk
                chunks.append(current_chunk.copy())

                # Start new chunk with overlap
                if self.CHUNK_OVERLAP_LINES > 0 and len(current_chunk) > self.CHUNK_OVERLAP_LINES:
                    # Include last N lines from previous chunk for voice consistency
                    current_chunk = current_chunk[-self.CHUNK_OVERLAP_LINES:]
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

    def _concatenate_with_pydub(self, audio_files: List[str]) -> str:
        """Concatenate audio files using PyDub with silence between chunks."""
        from pydub import AudioSegment

        logger.info(f"Concatenating {len(audio_files)} audio files with PyDub")

        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=1000)  # 1 second silence

        for i, file_path in enumerate(audio_files):
            audio = AudioSegment.from_wav(file_path)

            if i > 0:
                # Add silence between chunks
                combined += silence

            combined += audio

            # Clean up chunk file
            os.unlink(file_path)

        # Export combined audio
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        combined.export(output_path, format="wav")

        logger.info(f"Combined audio saved to: {output_path}")
        return output_path

    def _concatenate_with_wave(self, audio_files: List[str]) -> str:
        """Concatenate audio files using wave module (fallback if PyDub unavailable)."""

        logger.info(f"Concatenating {len(audio_files)} audio files with wave module")

        # Read all wave data
        frames = []
        params = None

        for file_path in audio_files:
            with wave.open(file_path, 'rb') as wav_file:
                if params is None:
                    params = wav_file.getparams()

                # Add audio frames
                frames.append(wav_file.readframes(wav_file.getnframes()))

                # Add 1 second of silence (24000 samples at 24kHz)
                if file_path != audio_files[-1]:
                    silence_frames = b'\x00\x00' * 24000
                    frames.append(silence_frames)

            # Clean up chunk file
            os.unlink(file_path)

        # Write combined audio
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        with wave.open(output_path, 'wb') as out_wav:
            out_wav.setparams(params)
            for frame_data in frames:
                out_wav.writeframes(frame_data)

        logger.info(f"Combined audio saved to: {output_path}")
        return output_path

