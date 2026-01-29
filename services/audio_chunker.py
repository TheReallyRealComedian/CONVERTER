# services/audio_chunker.py
"""
Audio-Splitting und Transkript-Merging für lange Dateien.
"""

import logging
import re
from io import BytesIO
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

from pydub import AudioSegment

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

    def needs_splitting(self, audio_data: bytes) -> Tuple[bool, dict]:
        """
        Prüft ob Audio gesplittet werden muss.

        Returns:
            Tuple[bool, dict]: (needs_split, metadata)
        """
        file_size = len(audio_data)

        try:
            audio = AudioSegment.from_file(BytesIO(audio_data))
            duration_ms = len(audio)
            duration_sec = duration_ms / 1000
        except Exception as e:
            logger.error(f"Could not analyze audio: {e}")
            # Im Fehlerfall: kein Split (Originalverhalten)
            return False, {"error": str(e)}

        metadata = {
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "duration_seconds": round(duration_sec, 1),
            "duration_formatted": f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}",
            "sample_rate": audio.frame_rate,
            "channels": audio.channels
        }

        needs_split = (
            duration_ms > self.max_duration_ms or
            file_size > self.max_file_size_bytes
        )

        if needs_split:
            estimated_chunks = max(1, int(duration_ms / (self.chunk_duration_ms - self.overlap_ms)) + 1)
            metadata["estimated_chunks"] = estimated_chunks
            logger.info(f"Audio needs splitting: {metadata}")
        else:
            logger.info(f"Audio within limits, no splitting needed: {metadata}")

        return needs_split, metadata

    def split_audio(self, audio_data: bytes) -> List[AudioChunk]:
        """
        Teilt Audio in überlappende Chunks.

        Returns:
            List[AudioChunk]: Liste der Audio-Chunks
        """
        audio = AudioSegment.from_file(BytesIO(audio_data))
        total_duration_ms = len(audio)

        chunks = []
        start_ms = 0
        index = 0

        while start_ms < total_duration_ms:
            # Berechne End-Position
            end_ms = min(start_ms + self.chunk_duration_ms, total_duration_ms)
            is_last = (end_ms >= total_duration_ms)

            # Extrahiere Chunk
            chunk_audio = audio[start_ms:end_ms]

            # Exportiere als WAV in Bytes
            buffer = BytesIO()
            chunk_audio.export(buffer, format="wav")
            chunk_bytes = buffer.getvalue()

            chunk = AudioChunk(
                index=index,
                audio_data=chunk_bytes,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                is_last=is_last
            )
            chunks.append(chunk)

            logger.info(
                f"Created chunk {index + 1}: "
                f"{start_ms/1000:.1f}s - {end_ms/1000:.1f}s "
                f"({(end_ms - start_ms)/1000:.1f}s, {len(chunk_bytes)/1024:.1f} KB)"
            )

            # Nächste Start-Position (mit Überlappung)
            if is_last:
                break
            start_ms = end_ms - self.overlap_ms
            index += 1

        logger.info(f"Split audio into {len(chunks)} chunks (total: {total_duration_ms/1000:.1f}s)")
        return chunks


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

        if not first_words or not second_words:
            return first + " " + second

        # Suche Ende von first in Anfang von second
        # Nimm die letzten N Wörter von first
        search_end = first_words[-self.search_window_words:]
        search_start = second_words[:self.search_window_words]

        best_match = self._find_best_overlap(search_end, search_start)

        if best_match:
            overlap_start, overlap_length, similarity = best_match
            logger.info(
                f"Chunk {chunk_index}: Found overlap of {overlap_length} words "
                f"(similarity: {similarity:.2f}) at position {overlap_start}"
            )

            # Schneide Überlappung aus second heraus
            merged = first + " " + " ".join(second_words[overlap_start + overlap_length:])
        else:
            logger.warning(
                f"Chunk {chunk_index}: No clear overlap found, "
                f"concatenating with sentence boundary detection"
            )
            merged = self._smart_concatenate(first, second)

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
