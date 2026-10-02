# services/google_tts_service.py
import logging
from google.cloud import texttospeech

from services.narration_render import DEFAULT_NARRATION_MODEL, render_turns
from services.wav_concat import is_pydub_available

logger = logging.getLogger(__name__)


class GoogleTTSService:
    def __init__(self, credentials_path):
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is required")
        self.credentials_path = credentials_path
        self.client = texttospeech.TextToSpeechClient()
        # Used by the faithful-narration path for multi-chunk concat (wave fallback).
        self.pydub_available = is_pydub_available()

    def synthesize_narration(self, turns, voices, *, style_prompt=None,
                             mode='two_speaker', language_code='de-DE',
                             model_name=DEFAULT_NARRATION_MODEL):
        """Faithful-narration entry: render structured turns to a WAV via Cloud Gemini TTS.

        The v1 narration render engine. Delegates to the pure
        ``narration_render.render_turns`` with this service's ``texttospeech``
        client. Decouples the speaker *label* (``Turn.speaker``) from the Gemini
        *voice* via the ``voices`` map, and passes any director's note as a
        separate ``SynthesisInput.prompt`` field (never concatenated into the
        transcript).

        Args:
            turns: Non-empty list of ``{'speaker': label, 'text': str}``;
                performance tags stay inline in ``text``.
            voices: ``{label: voice_id}`` covering every distinct speaker.
            style_prompt: Optional director's note (NARR-4 supplies it). Falsy → unset.
            mode: ``'single_speaker'`` or ``'two_speaker'`` (validation; routing is
                decided per chunk).
            language_code: BCP-47 code (default ``'de-DE'``).
            model_name: Gemini-TTS model (default ``DEFAULT_NARRATION_MODEL``).

        Returns:
            str: Path to a temporary WAV file (concatenated when chunked).

        Raises:
            ValueError: on contract violation or empty audio.
            google.api_core.exceptions.GoogleAPICallError: on a non-retryable or
                retry-exhausted Cloud-TTS failure.
        """
        return render_turns(
            self.client, turns, voices,
            style_prompt=style_prompt,
            mode=mode,
            language_code=language_code,
            model_name=model_name,
            pydub_available=self.pydub_available,
        )