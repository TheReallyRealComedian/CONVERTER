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


def _convert_pdf(source_path, mode, budget_eur, page_count):
    """PDF branch: route the legacy engine by mode + budget, attribute honestly.

    The legacy engine is a blackbox (one call, no per-page reporting), so the
    honest provenance depends on HOW it runs:

    * **Local run** (mode ``lokal``, or a cloud job degraded here): the engine
      is instantiated WITHOUT an API key — provably zero model calls, every
      page is deterministic → per-page provenance (trivially honest because
      homogeneous), ``usage`` is a known 0/0.
    * **Cloud run** (mode ``cloud``, key present, budget suffices): the engine
      MAY route pages through Gemini Vision but does not say which — so the
      provenance is ONE document-level entry, rounded UP to ``modell`` (a
      consumer must never read ``deterministisch`` on possibly-generated
      text), plus a ``provenance_document_only`` degradation naming the gap,
      and ``usage`` is ``None`` (unknown — 0 would be a claim). The engine
      sprint replaces this branch with real per-page attribution.

    Cloud degrades to local — never aborts — when the key is missing
    (``cloud_unavailable``) or the pre-flight estimate ``page_count × cent``
    exceeds the job's frozen budget (``budget_exceeded``). The estimate uses
    the bake-off-measured per-page price from config, because the legacy
    engine cannot account for itself mid-flight.
    """
    from app_pkg.config import DOC_CONVERT_CLOUD_CENT_PER_PAGE
    from services import PDFExtractionService
    from services.document_conversions import (
        DEGRADATION_BUDGET_EXCEEDED,
        DEGRADATION_CLOUD_UNAVAILABLE,
        DEGRADATION_PROVENANCE_DOCUMENT_ONLY,
        MODE_CLOUD,
        PROVENANCE_DETERMINISTIC,
        PROVENANCE_MODEL,
        UNIT_DOCUMENT,
        UNIT_PAGE,
        build_result_payload,
        degradation,
    )

    degradations = []
    cloud = mode == MODE_CLOUD
    if cloud:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            cloud = False
            degradations.append(degradation(
                DEGRADATION_CLOUD_UNAVAILABLE,
                'Cloud-Pfad nicht konfiguriert (kein API-Key im Worker). '
                'Lokal konvertiert.'))
        elif budget_eur is not None:
            estimated_eur = (page_count or 1) * DOC_CONVERT_CLOUD_CENT_PER_PAGE / 100
            if estimated_eur > budget_eur:
                cloud = False
                degradations.append(degradation(
                    DEGRADATION_BUDGET_EXCEEDED,
                    f'Erwartete Cloud-Kosten {estimated_eur:.2f} € über dem '
                    f'Kostendeckel {budget_eur:.2f} €. Lokal konvertiert.'))

    if cloud:
        svc = PDFExtractionService(api_key)
        markdown = svc.extract_markdown(source_path)
        degradations.append(degradation(
            DEGRADATION_PROVENANCE_DOCUMENT_ONLY,
            'Die Übergangs-Engine weist Herkunft nicht je Seite aus; '
            'konservativ als modell markiert.'))
        return build_result_payload(
            markdown,
            provenance_unit=UNIT_DOCUMENT,
            provenance=[PROVENANCE_MODEL],
            degradations=degradations,
            usage=None,
        )

    svc = PDFExtractionService(None)  # provably deterministic run
    markdown = svc.extract_markdown(source_path)
    if isinstance(page_count, int) and page_count > 0:
        unit, provenance = UNIT_PAGE, [PROVENANCE_DETERMINISTIC] * page_count
    else:
        # Unknown page count → don't claim page granularity we don't have.
        unit, provenance = UNIT_DOCUMENT, [PROVENANCE_DETERMINISTIC]
    return build_result_payload(
        markdown,
        provenance_unit=unit,
        provenance=provenance,
        degradations=degradations,
        usage={'model_calls': 0, 'cost_eur': 0.0},
    )


def _convert_office(source_path, source_ext):
    """Non-PDF branch: unstructured + serializer — deterministic by construction.

    No page concept in this pipeline (an element stream; DOCX "pages" are a
    renderer artifact) → ONE document-level provenance entry. Serializer
    warnings become structured ``serializer`` degradations: partial success
    is a ready result WITH a degradation list, never a failure.
    """
    from unstructured.partition.auto import partition

    from services.document_conversions import (
        DEGRADATION_SERIALIZER,
        PROVENANCE_DETERMINISTIC,
        UNIT_DOCUMENT,
        build_result_payload,
        degradation,
    )
    from services.unstructured_markdown import elements_to_markdown

    elements = partition(filename=source_path, strategy="fast")
    markdown, warnings = elements_to_markdown(elements, source_ext=source_ext)
    return build_result_payload(
        markdown,
        provenance_unit=UNIT_DOCUMENT,
        provenance=[PROVENANCE_DETERMINISTIC],
        degradations=[degradation(DEGRADATION_SERIALIZER, w) for w in warnings],
        usage={'model_calls': 0, 'cost_eur': 0.0},
    )


def convert_document_task(conversion_id, source_ext, mode, budget_eur,
                          page_count):
    """Convert an uploaded document to Markdown → ``result_<id>.json`` (DOC-API).

    **DB-free worker task (Option B, like the narration render).** The worker
    container mounts only the shared volume, never the SQLite DB, so this task
    reads the source from the id-derived path, converts it with the existing
    capability (PDF → ``pdf_extraction``, everything else → ``unstructured`` +
    ``elements_to_markdown``), and writes a **structured** result JSON
    atomically — it never flips the Conversion. The web side reconciles
    ``pending`` → ``ready``/``failed`` on the next poll by *reading* that file
    (markdown + provenance + degradations + usage — existence alone is not
    enough, the DOC-API extension over the narration mechanic).

    ``mode`` / ``budget_eur`` / ``page_count`` arrive from the web side: the
    mode is the resolved per-job choice (explicit or settings default), the
    budget is frozen at submit time (an env change never re-prices an already
    enqueued job), the page count feeds the pre-flight budget estimate.

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
        logger.info(f"conversion_id={conversion_id} ext={source_ext} "
                    f"mode={mode} budget_eur={budget_eur} pages={page_count}")

        if source_ext == 'pdf':
            payload = _convert_pdf(source_path, mode, budget_eur, page_count)
        else:
            payload = _convert_office(source_path, source_ext)

        result_path = write_result_file(conversion_id, payload)

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
