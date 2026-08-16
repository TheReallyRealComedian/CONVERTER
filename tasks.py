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
    """PDF branch: cloud = page-wise gemini, local = mineru (DOC-LOCAL).

    * **Cloud run** (mode ``cloud``, key present, pre-flight passes):
      ``services.pdf_cloud.run_cloud_pdf`` — one call per page, per-page
      ``modell`` provenance, costs booked from ``usage_metadata``, and the
      REAL mid-flight cap: if actual costs exhaust the budget mid-document,
      the remaining pages come from the local page function with one named
      ``budget_exceeded`` entry (DOC-ENGINE P2; the mid-flight target is the
      mineru engine too since DOC-LOCAL).
    * **Local run** (mode ``lokal``, or a cloud job degraded here):
      ``services.pdf_local.run_local_pdf`` — the mineru VLM in a sibling
      container (DOC-LOCAL replaced the legacy PyMuPDF-text-layer engine that
      returned scans EMPTY). Per-page provenance ``modell`` at cost 0.00 €;
      if the engine itself fails, pages fall back to the text layer with a
      named ``backend_fallback`` entry. The engine needs a real page count —
      unknown at submit → re-derived here via fitz (an unreadable PDF raises,
      exactly as the legacy engine's own fitz.open did).

    Cloud degrades to local — never aborts — when the key is missing
    (``cloud_unavailable``) or the pre-flight estimate ``page_count × cent``
    exceeds the job's frozen budget (``budget_exceeded``): if the ESTIMATE
    already says the budget cannot carry the document, not a single call is
    spent on a result that would be mostly local anyway. The mid-flight cap
    covers the complementary case: the estimate passed, but real per-page
    costs (token-dense pages) exhaust the budget during the run. Pre-flight
    degradation entries are PREPENDED to the local run's own list, so a
    ``backend_fallback`` never hides why the job went local at all.
    """
    from app_pkg.config import DOC_CONVERT_CLOUD_CENT_PER_PAGE
    from services.document_conversions import (
        DEGRADATION_BUDGET_EXCEEDED,
        DEGRADATION_CLOUD_UNAVAILABLE,
        MODE_CLOUD,
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
        from services.pdf_cloud import run_cloud_pdf
        return run_cloud_pdf(source_path, api_key, budget_eur)

    from services.pdf_local import run_local_pdf

    if not (isinstance(page_count, int) and page_count > 0):
        import fitz
        with fitz.open(source_path) as doc:
            page_count = doc.page_count
    payload = run_local_pdf(source_path, page_count)
    payload['degradations'] = degradations + payload['degradations']
    return payload


def _deterministic_document_payload(markdown, warnings):
    """The shared non-PDF result shape: document-level, deterministic.

    Every office/web backend in this task is model-free by construction
    (pandoc/markitdown/trafilatura/unstructured serializer), and none has a
    stable page concept → ONE document-level provenance entry, usage a known
    0/0. Backend warnings become structured ``serializer`` degradations:
    partial success is a ready result WITH a degradation list, never a
    failure.
    """
    from services.document_conversions import (
        DEGRADATION_SERIALIZER,
        PROVENANCE_DETERMINISTIC,
        UNIT_DOCUMENT,
        build_result_payload,
        degradation,
    )

    return build_result_payload(
        markdown,
        provenance_unit=UNIT_DOCUMENT,
        provenance=[PROVENANCE_DETERMINISTIC],
        degradations=[degradation(DEGRADATION_SERIALIZER, w) for w in warnings],
        usage={'model_calls': 0, 'cost_eur': 0.0},
    )


def _convert_office(source_path, source_ext):
    """Legacy non-PDF branch: unstructured + serializer (EML/TXT/MD, and the
    fallback target for an empty trafilatura extraction).

    Since DOC-ENGINE this no longer carries DOCX/PPTX/HTML — those route to
    their measured winners below; EML stays here deliberately (decision doc:
    functional, without competition).
    """
    from unstructured.partition.auto import partition

    from services.unstructured_markdown import elements_to_markdown

    elements = partition(filename=source_path, strategy="fast")
    markdown, warnings = elements_to_markdown(elements, source_ext=source_ext)
    return _deterministic_document_payload(markdown, warnings)


def _convert_docx(source_path):
    """DOCX branch: pandoc (DOC-ENGINE) — the bake-off winner (rule 3, the
    image-footnote-link chain: 4/4 vs 0/4 for every other candidate)."""
    from services.office_backends import convert_docx_pandoc

    return _deterministic_document_payload(*convert_docx_pandoc(source_path))


def _convert_pptx(source_path):
    """PPTX branch: markitdown (DOC-ENGINE) — recall 1.0, only candidate
    carrying speaker notes."""
    from services.office_backends import convert_pptx_markitdown

    return _deterministic_document_payload(*convert_pptx_markitdown(source_path))


def _convert_html(source_path, source_ext):
    """HTML branch: trafilatura + metadata head (DOC-ENGINE, <2 % boilerplate).

    trafilatura finding no main content degrades to the unstructured path
    (named ``backend_fallback`` entry) instead of failing — the legacy path
    could always serve some text for HTML, a hard fail would be a capability
    regression, and re-submitting (this API's retry) would never converge.
    """
    from services.document_conversions import (
        DEGRADATION_BACKEND_FALLBACK,
        degradation,
    )
    from services.office_backends import convert_html_trafilatura

    markdown, warnings = convert_html_trafilatura(source_path)
    if markdown is None:
        payload = _convert_office(source_path, source_ext)
        payload['degradations'].append(degradation(
            DEGRADATION_BACKEND_FALLBACK,
            'Artikel-Extraktion fand keinen Hauptinhalt. '
            'Element-Extraktion übernommen.'))
        return payload
    return _deterministic_document_payload(markdown, warnings)


def convert_document_task(conversion_id, source_ext, mode, budget_eur,
                          page_count):
    """Convert an uploaded document to Markdown → ``result_<id>.json`` (DOC-API).

    **DB-free worker task (Option B, like the narration render).** The worker
    container mounts only the shared volume, never the SQLite DB, so this task
    reads the source from the id-derived path, routes it to the format's
    measured winner (DOC-ENGINE: DOCX → pandoc, PPTX → markitdown, HTML/HTM →
    trafilatura; PDF → gemini page-wise bzw. mineru seit DOC-LOCAL; EML/TXT/MD
    → ``unstructured`` + ``elements_to_markdown``), and writes a **structured**
    result JSON atomically — it never flips the Conversion. The web side
    reconciles ``pending`` → ``ready``/``failed`` on the next poll by
    *reading* that file (markdown + provenance + degradations + usage —
    existence alone is not enough, the DOC-API extension over the narration
    mechanic).

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
        elif source_ext == 'docx':
            payload = _convert_docx(source_path)
        elif source_ext == 'pptx':
            payload = _convert_pptx(source_path)
        elif source_ext in ('html', 'htm'):
            payload = _convert_html(source_path, source_ext)
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
