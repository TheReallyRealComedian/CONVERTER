# services/gemini/audio.py
"""WAV concatenation helpers for multi-chunk podcast outputs.

Both functions take a list of WAV file paths and return the path to a new
combined WAV file. They unlink the inputs along the way (legacy behaviour).
"""
import logging
import os
import tempfile
import wave
from typing import List

logger = logging.getLogger(__name__)


def concatenate_with_pydub(audio_files: List[str]) -> str:
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


def concatenate_with_wave(audio_files: List[str]) -> str:
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
