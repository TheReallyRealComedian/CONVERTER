# services/gemini/client.py
"""Gemini SDK client construction and shared environment probes.

Holds the timeout-tweaked ``genai.Client`` factory and the one-time pydub
import probe used by the TTS path. Kept dependency-light so both the
script and TTS submodules can pull from here without cycling through the
top-level facade.
"""
import logging

from google import genai

logger = logging.getLogger(__name__)


def create_client(api_key):
    """Build a ``google.genai.Client`` with an extended HTTP timeout.

    Mirrors the legacy ``GeminiService.__init__`` behaviour: bumps the
    underlying httpx client timeout to 300s when the SDK exposes it. The
    extra timeout is required because podcast TTS generations regularly
    exceed the SDK's default.
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")

    client = genai.Client(api_key=api_key)
    if hasattr(client, '_api_client'):
        if hasattr(client._api_client, '_httpx_client'):
            import httpx
            client._api_client._httpx_client.timeout = httpx.Timeout(timeout=300.0)
            logger.info("✅ Timeout auf 300 Sekunden erhöht")
    return client


def is_pydub_available():
    """Return True if pydub can be imported in this process."""
    try:
        from pydub import AudioSegment  # noqa: F401
        return True
    except ImportError:
        logger.warning("PyDub not available - audio concatenation disabled")
        return False
