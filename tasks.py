"""
Background tasks for RQ workers.
These functions run in isolated worker processes.
"""
import os
import shutil
import logging

from rq import get_current_job

from app_pkg.config import OUTPUT_DIR
from services import GoogleTTSService
from services.narration_library import narration_audio_path

logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def update_job_stage(stage, **extras):
    """Write a stage marker into the current RQ job's meta.

    Worker-side helper used by ``generate_narration_task`` to record coarse
    progress (e.g. ``finalizing``) on the job for the web side to read.

    No-op outside an RQ worker context (in-process tests, direct calls).
    Wrapped in try/except because a stage update failure must never abort
    the actual render.
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


def generate_narration_task(conversion_id, turns, voices, style_prompt, mode,
                            language_code, model_name):
    """Render a faithful narration to ``narration_<conversion_id>.wav`` (NARR-3).

    **DB-free worker task (Option B).** The worker container mounts only the
    shared ``podcast_data`` volume, never the SQLite DB. So this task renders the
    audio and writes it to the deterministic, id-derived path — it **never**
    flips the Conversion. The web side reconciles ``pending`` →
    ``ready``/``failed`` on the next poll (``reconcile_narration``), keyed on
    this file's existence and the RQ job's terminal state.

    Instantiates the SDK service in-task, renders, ``shutil.move``s the temp WAV
    onto the shared volume, and returns the final path. On any failure it logs
    and re-raises so RQ marks the job ``failed``
    (the Exception lands in ``job.exc_info``, which reconcile surfaces as the
    error). The renderer already cleans up its own temp WAVs on the error path.
    """
    try:
        logger.info("=== NARRATION TASK START ===")
        logger.info(f"conversion_id={conversion_id} mode={mode} model={model_name}")

        creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in worker environment")

        svc = GoogleTTSService(creds)
        temp_path = svc.synthesize_narration(
            turns, voices,
            style_prompt=style_prompt,
            mode=mode,
            language_code=language_code,
            model_name=model_name,
        )

        # id-derived destination on the shared volume (never user input).
        final_path = narration_audio_path(conversion_id)
        update_job_stage('finalizing')
        shutil.move(temp_path, final_path)

        logger.info("=== NARRATION TASK SUCCESS ===")
        logger.info(f"File moved to: {final_path}")

        return final_path

    except Exception as e:
        logger.error("=== NARRATION TASK FAILED ===")
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        raise
