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


def convert_document_task(conversion_id, source_ext):
    """Convert an uploaded document to Markdown → ``result_<id>.json`` (DOC-API).

    **DB-free worker task (Option B, like the narration render).** The worker
    container mounts only the shared volume, never the SQLite DB, so this task
    reads the source from the id-derived path, converts it with the existing
    capability (PDF → ``pdf_extraction``, everything else → ``unstructured`` +
    ``elements_to_markdown``), and writes a **structured** result JSON
    atomically — it never flips the Conversion. The web side reconciles
    ``pending`` → ``ready``/``failed`` on the next poll by *reading* that file
    (a conversion result carries markdown + warnings, so existence alone is not
    enough — the DOC-API extension over the narration mechanic).

    Engines are imported in-task (the web process enqueues without loading
    them; the SDK-singleton convention stays untouched). On any failure it
    logs and re-raises so RQ marks the job ``failed`` (reconcile surfaces the
    ``exc_info`` tail). The source file is scratch: best-effort deleted in
    ``finally`` — there is no retry path that would need it.
    """
    from services.document_conversions import doc_source_path, write_result_file

    source_path = doc_source_path(conversion_id, source_ext)
    try:
        logger.info("=== DOCUMENT CONVERSION TASK START ===")
        logger.info(f"conversion_id={conversion_id} ext={source_ext}")

        if source_ext == 'pdf':
            from services import PDFExtractionService
            svc = PDFExtractionService(os.environ.get('GEMINI_API_KEY'))
            markdown = svc.extract_markdown(source_path)
            warnings = []
        else:
            from unstructured.partition.auto import partition

            from services.unstructured_markdown import elements_to_markdown
            elements = partition(filename=source_path, strategy="fast")
            markdown, warnings = elements_to_markdown(elements, source_ext=source_ext)

        result_path = write_result_file(conversion_id, {
            'markdown': markdown,
            'warnings': list(warnings),
        })

        logger.info("=== DOCUMENT CONVERSION TASK SUCCESS ===")
        logger.info(f"Result written to: {result_path}")
        return result_path

    except Exception as e:
        logger.error("=== DOCUMENT CONVERSION TASK FAILED ===")
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        raise
    finally:
        # Source is scratch once the task ends either way (success has the
        # result on the volume, failure has exc_info in RQ); never let cleanup
        # mask the actual outcome.
        try:
            if os.path.exists(source_path):
                os.remove(source_path)
        except OSError:
            logger.warning(f"Could not remove source file: {source_path}")


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
