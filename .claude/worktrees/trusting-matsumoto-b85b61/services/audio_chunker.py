# services/audio_chunker.py
"""
Audio-Splitting und Transkript-Merging für lange Dateien.
Uses ffmpeg/ffprobe for splitting (streams data, no full RAM decode).
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Repräsentiert einen Audio-Chunk mit Metadaten."""
    index: int
    audio_data: bytes
    start_ms: int
    end_ms: int
    duration_ms: int
    is_last: bool


class AudioChunker:
    """
    Teilt Audio-Dateien in überlappende Chunks für Transkription.
    Uses ffprobe for metadata and ffmpeg for splitting (no full decode into RAM).
    """

    def __init__(
        self,
        max_duration_seconds: int = 600,
        chunk_duration_seconds: int = 300,
        overlap_seconds: int = 5,
        max_file_size_mb: int = 100
    ):
        self.max_duration_ms = max_duration_seconds * 1000
        self.chunk_duration_ms = chunk_duration_seconds * 1000
        self.overlap_ms = overlap_seconds * 1000
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _get_audio_metadata(self, file_path: str) -> dict:
        """Get audio metadata using ffprobe (no RAM overhead)."""
        result = subprocess.run(
            [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_entries', 'format=duration,size,bit_rate',
                '-show_entries', 'stream=sample_rate,channels',
                file_path
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        return json.loads(result.stdout)

    def needs_splitting(self, audio_data: bytes) -> Tuple[bool, dict]:
        """
        Prüft ob Audio gesplittet werden muss.
        Uses ffprobe instead of loading entire file into RAM.

        Returns:
            Tuple[bool, dict]: (needs_split, metadata)
        """
        file_size = len(audio_data)

        # Write to temp file for ffprobe (ffprobe needs a file path)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.audio')
        try:
            with os.fdopen(tmp_fd, 'wb') as f:
                f.write(audio_data)

            probe = self._get_audio_metadata(tmp_path)

            duration_sec = float(probe['format']['duration'])
            duration_ms = int(duration_sec * 1000)

            # Extract stream info
            sample_rate = None
            channels = None
            if probe.get('streams'):
                stream = probe['streams'][0]
                sample_rate = int(stream.get('sample_rate', 0)) or None
                channels = int(stream.get('channels', 0)) or None

            metadata = {
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "duration_seconds": round(duration_sec, 1),
                "duration_formatted": f"{int(duration_sec // 3600)}h{int((duration_sec % 3600) // 60):02d}m",
                "sample_rate": sample_rate,
                "channels": channels
            }

            needs_split = (
                duration_ms > self.max_duration_ms or
                file_size > self.max_file_size_bytes
            )

            if needs_split:
                step_ms = self.chunk_duration_ms - self.overlap_ms
                estimated_chunks = max(1, (duration_ms + step_ms - 1) // step_ms)
                metadata["estimated_chunks"] = estimated_chunks
                logger.info(f"Audio needs splitting: {metadata}")
            else:
                logger.info(f"Audio within limits, no splitting needed: {metadata}")

            # Store temp path and duration for split_audio to reuse
            self._tmp_path = tmp_path
            self._total_duration_ms = duration_ms
            self._tmp_owned = True  # We own the temp file

            return needs_split, metadata

        except Exception as e:
            # Clean up on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            logger.error(f"Could not analyze audio: {e}")
            return False, {"error": str(e)}

    def split_audio(self, audio_data: bytes) -> List[AudioChunk]:
        """
        Teilt Audio in überlappende Chunks using ffmpeg.
        Each chunk is extracted as MP3 via ffmpeg (streaming, no full decode).

        Returns:
            List[AudioChunk]: Liste der Audio-Chunks
        """
        # Use cached temp file from needs_splitting if available
        if hasattr(self, '_tmp_path') and self._tmp_path and os.path.exists(self._tmp_path):
            input_path = self._tmp_path
            total_duration_ms = self._total_duration_ms
            owns_file = getattr(self, '_tmp_owned', False)
        else:
            # Fallback: write to temp file
            tmp_fd, input_path = tempfile.mkstemp(suffix='.audio')
            with os.fdopen(tmp_fd, 'wb') as f:
                f.write(audio_data)
            probe = self._get_audio_metadata(input_path)
            total_duration_ms = int(float(probe['format']['duration']) * 1000)
            owns_file = True

        try:
            chunks = []
            start_ms = 0
            index = 0

            while start_ms < total_duration_ms:
                end_ms = min(start_ms + self.chunk_duration_ms, total_duration_ms)
                is_last = (end_ms >= total_duration_ms)
                duration_ms = end_ms - start_ms

                # Extract chunk with ffmpeg (streams, doesn't decode entire file)
                chunk_bytes = self._extract_chunk_ffmpeg(
                    input_path,
                    start_ms / 1000.0,
                    duration_ms / 1000.0
                )

                chunk = AudioChunk(
                    index=index,
                    audio_data=chunk_bytes,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration_ms,
                    is_last=is_last
                )
                chunks.append(chunk)

                logger.info(
                    f"Created chunk {index + 1}: "
                    f"{start_ms/1000:.1f}s - {end_ms/1000:.1f}s "
                    f"({duration_ms/1000:.1f}s, {len(chunk_bytes)/1024:.1f} KB)"
                )

                if is_last:
                    break
                start_ms = end_ms - self.overlap_ms
                index += 1

            logger.info(f"Split audio into {len(chunks)} chunks (total: {total_duration_ms/1000:.1f}s)")
            return chunks

        finally:
            # Clean up temp file
            if owns_file and os.path.exists(input_path):
                os.unlink(input_path)
            self._tmp_path = None
            self._tmp_owned = False

    def _extract_chunk_ffmpeg(self, input_path: str, start_sec: float, duration_sec: float) -> bytes:
        """Extract a chunk as MP3 using ffmpeg (pipes to stdout, no temp file needed)."""
        result = subprocess.run(
            [
                'ffmpeg', '-v', 'warning',
                '-ss', str(start_sec),
                '-t', str(duration_sec),
                '-i', input_path,
                '-acodec', 'libmp3lame',
                '-ab', '64k',       # 64kbps is fine for speech
                '-ac', '1',         # mono
                '-ar', '16000',     # 16kHz sample rate (optimal for Deepgram)
                '-f', 'mp3',
                'pipe:1'
            ],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg chunk extraction failed: {result.stderr.decode()[:500]}")
        return result.stdout


class TranscriptMerger:
    """
    Merged überlappende Transkripte zu einem zusammenhängenden Text.
    """

    def __init__(
        self,
        min_overlap_words: int = 3,
        fuzzy_threshold: float = 0.7,
        search_window_words: int = 50
    ):
        self.min_overlap_words = min_overlap_words
        self.fuzzy_threshold = fuzzy_threshold
        self.search_window_words = search_window_words

    def merge_transcripts(self, transcripts: List[str]) -> str:
        """
        Merged eine Liste von Transkripten.

        Args:
            transcripts: Liste der Teil-Transkripte in Reihenfolge

        Returns:
            str: Zusammengeführtes Transkript
        """
        if not transcripts:
            return ""

        if len(transcripts) == 1:
            return transcripts[0]

        logger.info(f"Merging {len(transcripts)} transcript chunks")

        merged = transcripts[0]

        for i, next_transcript in enumerate(transcripts[1:], start=1):
            logger.info(f"Merging chunk {i + 1} into result...")

            # Finde Überlappung und merge
            merged = self._merge_two_transcripts(merged, next_transcript, i)

        # Bereinigung
        merged = self._clean_merged_text(merged)

        logger.info(f"Merge complete: {len(merged)} characters")
        return merged

    def _merge_two_transcripts(self, first: str, second: str, chunk_index: int) -> str:
        """Merged zwei aufeinanderfolgende Transkripte."""

        # Normalisiere für Vergleich
        first_words = first.split()
        second_words = second.split()

        logger.debug(
            f"Chunk {chunk_index}: Merging first={len(first_words)} words + second={len(second_words)} words"
        )

        if not first_words or not second_words:
            return first + " " + second

        # Suche Ende von first in Anfang von second
        # Nimm die letzten N Wörter von first
        search_end = first_words[-self.search_window_words:]
        search_start = second_words[:self.search_window_words]

        best_match = self._find_best_overlap(search_end, search_start)

        if best_match:
            overlap_start, overlap_length, similarity = best_match

            # NUR Overlaps am Anfang von second akzeptieren (Position 0)
            # Bei anderen Positionen werden fälschlicherweise Wörter VOR dem Overlap entfernt
            if overlap_start == 0:
                logger.info(
                    f"Chunk {chunk_index}: Found overlap of {overlap_length} words "
                    f"(similarity: {similarity:.2f}) at position {overlap_start}"
                )
                # Schneide nur die Overlap-Wörter aus second heraus
                if overlap_length < len(second_words):
                    merged = first + " " + " ".join(second_words[overlap_length:])
                else:
                    # Overlap ist gesamter second-Text -> nur first behalten
                    merged = first
            else:
                # Overlap nicht am Anfang = False Positive
                logger.warning(
                    f"Chunk {chunk_index}: Overlap at position {overlap_start} (not at start), "
                    f"concatenating to avoid data loss"
                )
                merged = self._smart_concatenate(first, second)
        else:
            logger.warning(
                f"Chunk {chunk_index}: No clear overlap found, "
                f"concatenating with sentence boundary detection"
            )
            merged = self._smart_concatenate(first, second)

        logger.debug(f"Chunk {chunk_index}: Result = {len(merged.split())} words")
        return merged

    def _find_best_overlap(
        self,
        end_words: List[str],
        start_words: List[str]
    ) -> Optional[Tuple[int, int, float]]:
        """
        Findet die beste Überlappung zwischen Ende und Anfang.

        Returns:
            Tuple[position_in_start, overlap_length, similarity] oder None
        """
        best_match = None
        best_score = 0

        # Probiere verschiedene Overlap-Längen
        for overlap_len in range(self.min_overlap_words, len(end_words) + 1):
            # Letzten overlap_len Wörter von end
            end_phrase = " ".join(end_words[-overlap_len:]).lower()

            # Suche in start_words
            for start_pos in range(len(start_words) - overlap_len + 1):
                start_phrase = " ".join(start_words[start_pos:start_pos + overlap_len]).lower()

                # Berechne Ähnlichkeit
                similarity = SequenceMatcher(None, end_phrase, start_phrase).ratio()

                if similarity >= self.fuzzy_threshold:
                    # Gewichtung: längere Matches sind besser
                    score = similarity * (overlap_len ** 0.5)

                    if score > best_score:
                        best_score = score
                        best_match = (start_pos, overlap_len, similarity)

        return best_match

    def _smart_concatenate(self, first: str, second: str) -> str:
        """
        Fallback-Verkettung wenn keine Überlappung gefunden wurde.
        Fügt einfach zusammen ohne etwas abzuschneiden - Datenverlust vermeiden!
        Doppelte Leerzeichen werden später in _clean_merged_text() bereinigt.
        """
        return first + " " + second

    def _clean_merged_text(self, text: str) -> str:
        """Bereinigt das gemergte Transkript."""
        # Entferne doppelte Leerzeichen
        text = re.sub(r'\s+', ' ', text)

        # Entferne doppelte Satzzeichen
        text = re.sub(r'([.!?])\s*\1+', r'\1', text)

        # Korrigiere Groß-/Kleinschreibung nach Satzzeichen
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

        return text.strip()
