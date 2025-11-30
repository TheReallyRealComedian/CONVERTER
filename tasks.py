"""
Background tasks for RQ workers.
These functions run in isolated worker processes.
"""
import os
import shutil
import logging
from services import GeminiService

logger = logging.getLogger(__name__)

# Shared output directory (mounted as Docker volume)
OUTPUT_DIR = '/app/output_podcasts'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_podcast_task(dialogue, language, tts_model):
    """
    Generate a podcast in a background worker process.

    This function runs isolated in the worker container.
    The result file is moved to a shared volume so the web container can serve it.

    Args:
        dialogue: List of dicts with 'speaker', 'style', 'text'
        language: Language code (e.g., 'en', 'de')
        tts_model: TTS model to use (e.g., 'gemini-2.5-flash-preview-tts')

    Returns:
        str: Path to the generated audio file in the shared volume
    """
    try:
        logger.info(f"=== WORKER TASK START ===")
        logger.info(f"Dialogue turns: {len(dialogue)}")
        logger.info(f"Language: {language}")
        logger.info(f"TTS Model: {tts_model}")

        # Initialize service fresh in worker (avoids pickle issues)
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in worker environment")

        gemini_service = GeminiService(api_key)

        # Generate podcast (creates temp file)
        logger.info("Starting podcast generation...")
        temp_path = gemini_service.generate_podcast(dialogue, language, tts_model)

        # Move to shared volume so web container can access it
        filename = os.path.basename(temp_path)
        final_path = os.path.join(OUTPUT_DIR, filename)

        shutil.move(temp_path, final_path)

        logger.info(f"=== WORKER TASK SUCCESS ===")
        logger.info(f"File moved to: {final_path}")

        return final_path

    except Exception as e:
        logger.error(f"=== WORKER TASK FAILED ===")
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        raise
