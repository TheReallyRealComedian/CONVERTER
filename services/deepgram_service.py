# services/deepgram_service.py
import logging
import time
from pathlib import Path
import json
from typing import Optional

from deepgram import DeepgramClient

from .audio_chunker import AudioChunker, TranscriptMerger, AudioChunk

logger = logging.getLogger(__name__)


class DeepgramService:
    # Chunking-Konfiguration
    MAX_AUDIO_DURATION_SECONDS = 600   # 10 Minuten
    CHUNK_DURATION_SECONDS = 300       # 5 Minuten pro Chunk
    OVERLAP_SECONDS = 5                # 5 Sekunden Überlappung
    MAX_FILE_SIZE_MB = 100             # 100 MB
    INTER_CHUNK_DELAY = 0.5            # Rate limiting zwischen Chunks
    MAX_RETRIES = 2                    # Retries pro Chunk

    def __init__(self, api_key):
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is required")
        self.api_key = api_key
        self.client = DeepgramClient(api_key=api_key)

        # Initialisiere Chunker und Merger
        self.chunker = AudioChunker(
            max_duration_seconds=self.MAX_AUDIO_DURATION_SECONDS,
            chunk_duration_seconds=self.CHUNK_DURATION_SECONDS,
            overlap_seconds=self.OVERLAP_SECONDS,
            max_file_size_mb=self.MAX_FILE_SIZE_MB
        )
        self.merger = TranscriptMerger()
    
    def load_keyterms(self, language='en'):
        """
        Load domain-specific keyterms for improved transcription accuracy.
        Returns a list of keyterms for the specified language.
        """
        try:
            keyterms_path = Path('/app/keyterms.json')
            if not keyterms_path.exists():
                logger.warning("keyterms.json not found, using empty keyterms list")
                return []
            
            with open(keyterms_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Combine language-specific and universal terms
            terms = data.get(language, []) + data.get('universal', [])
            
            # Limit to 100 terms as per Deepgram recommendation
            terms = terms[:100]
            
            logger.info(f"Loaded {len(terms)} keyterms for language '{language}'")
            return terms
        except Exception as e:
            logger.error(f"Error loading keyterms: {e}")
            return []
    
    def transcribe_file(self, audio_data: bytes, language: str = 'en') -> str:
        """
        Transcribe audio file mit automatischem Splitting für lange Dateien.

        Args:
            audio_data: Audio file data (bytes)
            language: Language code (e.g., 'en', 'de')

        Returns:
            str: Transcribed text
        """
        # Prüfe ob Splitting nötig
        needs_split, metadata = self.chunker.needs_splitting(audio_data)

        if not needs_split:
            # Originalverhalten für kurze Dateien
            logger.info("Audio within limits, using direct transcription")
            return self._transcribe_single(audio_data, language)

        # Splitting-Modus
        logger.info("=" * 50)
        logger.info("=== CHUNKED TRANSCRIPTION MODE ===")
        logger.info("=" * 50)
        logger.info(f"Audio metadata: {metadata}")

        # Teile Audio
        chunks = self.chunker.split_audio(audio_data)
        total_chunks = len(chunks)

        # Transkribiere jeden Chunk
        transcripts = []
        total_start = time.time()

        for chunk in chunks:
            logger.info(f"")
            logger.info(f">>> Processing chunk {chunk.index + 1}/{total_chunks} <<<")

            transcript = self._transcribe_chunk_with_retry(
                chunk, language, total_chunks
            )

            if transcript is not None:
                if transcript.strip():
                    transcripts.append(transcript)
                    logger.info(f"Chunk {chunk.index + 1} transcript: {len(transcript)} chars")
                else:
                    # Leere Chunks nicht zur Liste hinzufügen - vermeidet unnötige Merge-Operationen
                    logger.warning(f"Chunk {chunk.index + 1} returned empty transcript (silence/music?) - skipping")
            else:
                logger.error(f"Chunk {chunk.index + 1} failed!")
                raise RuntimeError(f"Transcription failed for chunk {chunk.index + 1}")

            # Rate limiting zwischen Chunks
            if not chunk.is_last:
                time.sleep(self.INTER_CHUNK_DELAY)

        total_elapsed = time.time() - total_start
        logger.info(f"")
        logger.info(f"=== ALL CHUNKS TRANSCRIBED ===")
        logger.info(f"Total time: {total_elapsed:.1f}s")
        logger.info(f"Chunks with content: {len(transcripts)}/{total_chunks}")

        # Merge Transkripte
        merged_transcript = self.merger.merge_transcripts(transcripts)

        logger.info(f"Final transcript: {len(merged_transcript)} chars")
        logger.info("=" * 50)

        return merged_transcript

    def _transcribe_single(self, audio_data: bytes, language: str) -> str:
        """Transkribiert eine einzelne Audio-Datei (Original-Methode)."""
        try:
            # Load keyterms for the selected language
            keyterms = self.load_keyterms(language)

            logger.info(f"Transcribing with Nova-3, language={language}, keyterms={len(keyterms)}")

            # SDK 5.1.0: Direct API call with kwargs
            response = self.client.listen.v1.media.transcribe_file(
                request=audio_data,
                model="nova-3",
                smart_format=True,
                utterances=True,
                punctuate=True,
                language=language,
                numerals=True,
                paragraphs=True,
                keyterm=keyterms,
            )

            # Extract transcript
            transcript = response.results.channels[0].alternatives[0].transcript

            return transcript

        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
            raise

    def _transcribe_chunk_with_retry(
        self,
        chunk: AudioChunk,
        language: str,
        total_chunks: int
    ) -> Optional[str]:
        """Transkribiert einen Chunk mit Retry-Logik."""

        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Retry {attempt}/{self.MAX_RETRIES} for chunk {chunk.index + 1} "
                        f"(waiting {delay}s)"
                    )
                    time.sleep(delay)

                start_time = time.time()
                transcript = self._transcribe_single(chunk.audio_data, language)
                elapsed = time.time() - start_time

                logger.info(
                    f"Chunk {chunk.index + 1}/{total_chunks} completed in {elapsed:.1f}s"
                )

                return transcript

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Chunk {chunk.index + 1} attempt {attempt + 1} failed: "
                    f"{type(e).__name__}: {str(e)[:100]}"
                )

        # Alle Retries fehlgeschlagen
        logger.error(
            f"Chunk {chunk.index + 1} failed after {self.MAX_RETRIES + 1} attempts: {last_error}"
        )
        return None
    
    def get_api_key(self):
        """Return API key for client-side WebSocket connection."""
        return self.api_key