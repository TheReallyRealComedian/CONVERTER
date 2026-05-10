"""
Background tasks for RQ workers.
These functions run in isolated worker processes.
"""
import os
import shutil
import logging

from rq import get_current_job

from app_pkg.config import OUTPUT_DIR
from services import GeminiService

logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def update_job_stage(stage, **extras):
    """F-4.3 P3: write a stage marker into the current RQ job's meta.

    Worker-side helper; called from tasks.py and the gemini.tts pipeline so
    the polling loop in /podcast-status can render a textual sub-caption
    (and chunk-progress bar in the multi-chunk path) instead of just the
    wall-clock counter.

    No-op outside an RQ worker context (in-process tests, direct calls).
    Wrapped in try/except because a stage update failure must never abort
    the actual TTS pipeline.
    """
    job = get_current_job()
    if job is None:
        return
    try:
        job.meta['stage'] = stage
        for key, value in extras.items():
            job.meta[key] = value
        job.save_meta()
    except Exception as e:
        logger.warning(f"update_job_stage failed for stage={stage}: {e}")


def generate_podcast_task(dialogue, language, tts_model):
    """
    Generate a podcast in a background worker process.

    The output file is named ``{job_id}.wav`` so the web container can locate
    it by job_id alone (used by both /podcast-download and /podcast-cancel
    cleanup). Falls back to the temp basename if no RQ job context is present
    (only happens in direct in-process calls — not the RQ-worker code path).
    """
    try:
        logger.info(f"=== WORKER TASK START ===")
        logger.info(f"Dialogue turns: {len(dialogue)}")
        logger.info(f"Language: {language}")
        logger.info(f"TTS Model: {tts_model}")

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in worker environment")

        gemini_service = GeminiService(api_key)

        logger.info("Starting podcast generation...")
        temp_path = gemini_service.generate_podcast(dialogue, language, tts_model)

        job = get_current_job()
        if job is not None:
            filename = f"{job.id}.wav"
        else:
            filename = os.path.basename(temp_path)
        final_path = os.path.join(OUTPUT_DIR, filename)

        update_job_stage('finalizing')
        shutil.move(temp_path, final_path)

        logger.info(f"=== WORKER TASK SUCCESS ===")
        logger.info(f"File moved to: {final_path}")

        return final_path

    except Exception as e:
        logger.error(f"=== WORKER TASK FAILED ===")
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        raise
