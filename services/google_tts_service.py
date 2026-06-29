# services/google_tts_service.py
import logging
import tempfile
from google.cloud import texttospeech

from services.gemini.client import is_pydub_available
from services.narration_render import DEFAULT_NARRATION_MODEL, render_turns

logger = logging.getLogger(__name__)


class GoogleTTSService:
    def __init__(self, credentials_path):
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is required")
        self.credentials_path = credentials_path
        self.client = texttospeech.TextToSpeechClient()
        # Used by the faithful-narration path for multi-chunk concat (wave fallback).
        self.pydub_available = is_pydub_available()
    
    def list_voices(self):
        """
        Get available voices grouped by language.
        
        Returns:
            dict: Voices grouped by language code
        """
        try:
            voices_response = self.client.list_voices()
            
            voices_by_language = {}
            for voice in voices_response.voices:
                for language_code in voice.language_codes:
                    if language_code not in voices_by_language:
                        voices_by_language[language_code] = []
                    
                    voices_by_language[language_code].append({
                        'name': voice.name,
                        'gender': texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                        'natural_sample_rate_hertz': voice.natural_sample_rate_hertz
                    })
            
            return voices_by_language
        
        except Exception as e:
            logger.error(f"Failed to retrieve Google TTS voices: {e}", exc_info=True)
            raise
    
    def synthesize_speech(self, text, voice_name='en-US-Neural2-C', 
                          language_code='en-US', speaking_rate=1.0, pitch=0.0):
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_name: Voice name
            language_code: Language code
            speaking_rate: Speaking rate (0.25 to 4.0)
            pitch: Pitch adjustment (-20.0 to 20.0)
        
        Returns:
            str: Path to temporary MP3 file
        """
        if not text or not text.strip():
            raise ValueError("No text provided for synthesis")
        
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
                pitch=pitch
            )
            
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            temp_audio_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
            
            with open(temp_audio_path, 'wb') as out:
                out.write(response.audio_content)
            
            logger.info(f"Audio synthesized and saved to: {temp_audio_path}")

            return temp_audio_path

        except Exception as e:
            logger.error(f"Google TTS synthesis failed: {e}", exc_info=True)
            raise

    def synthesize_narration(self, turns, voices, *, style_prompt=None,
                             mode='two_speaker', language_code='de-DE',
                             model_name=DEFAULT_NARRATION_MODEL):
        """Faithful-narration entry: render structured turns to a WAV via Cloud Gemini TTS.

        The v1 narration render engine. Delegates to the pure
        ``narration_render.render_turns`` with this service's ``texttospeech``
        client. Decouples the speaker *label* (``Turn.speaker``) from the Gemini
        *voice* via the ``voices`` map, and passes any director's note as a
        separate ``SynthesisInput.prompt`` field (never concatenated into the
        transcript). The standard-neural ``synthesize_speech`` path above is
        untouched and unaffected.

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