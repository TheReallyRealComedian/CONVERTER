# services/gemini/__init__.py
"""Gemini script + TTS package.

Public surface is the ``GeminiService`` class — a thin facade that owns the
genai.Client and delegates ``format_dialogue_with_llm`` and
``generate_podcast`` to the submodule free functions.

Imported by ``services/__init__.py`` (which re-exports ``GeminiService``
under the original ``from services import GeminiService`` path) and by the
worker process via ``tasks.py``.
"""
import logging

from services.gemini.client import create_client, is_pydub_available
from services.gemini.script import format_dialogue_with_llm as _format_dialogue_with_llm
from services.gemini.tts import (
    DEFAULT_TTS_MODEL,
    TTS_MODELS,
    generate_podcast as _generate_podcast,
)

logger = logging.getLogger(__name__)


class GeminiService:
    TTS_MODELS = TTS_MODELS
    DEFAULT_TTS_MODEL = DEFAULT_TTS_MODEL

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

    def generate_podcast(self, dialogue, language='en', tts_model=None):
        return _generate_podcast(
            self.client, dialogue,
            language=language,
            tts_model=tts_model,
            pydub_available=self.pydub_available,
        )


__all__ = ['GeminiService']
