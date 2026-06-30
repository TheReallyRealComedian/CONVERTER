# services/gemini/__init__.py
"""Gemini client package.

Public surface is the ``GeminiService`` class — a thin wrapper that owns a
``genai.Client``. NARR-5 retired the alt-podcast flow that drove it
(``format_dialogue_with_llm`` + ``generate_podcast``), so the class is now
**dormant**: it is still constructed as the ``gemini_service`` singleton in
``app.py`` and re-exported via ``services/__init__.py``, but has no live caller.
Kept intentionally (out of NARR-5 scope) as the seam for a future Gemini
text/script feature — remove it and the singleton if none materialises.
"""
import logging

from services.gemini.client import create_client, is_pydub_available

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = create_client(api_key)
        self.pydub_available = is_pydub_available()


__all__ = ['GeminiService']
