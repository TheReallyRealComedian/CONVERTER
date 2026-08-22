"""One router for both entrances (DOC-WEB).

Pure module: no Flask, no SDK singleton, no top-level heavy imports (backend
libraries load inside the branch that needs them, exactly like the in-task
imports this logic came from in ``tasks.py``). Both callers share it:

* ``tasks.convert_document_task`` (the API job, async via RQ) wraps
  ``convert_non_pdf`` into the document-level deterministic result payload
  and writes ``convert_pdf``'s payload as is;
* ``app_pkg/documents.py::transform_document`` (the browser button, sync —
  non-PDF only since DOC-WEB-ASYNC; a browser PDF arrives as an API job and
  runs through the task above) takes ``markdown`` + ``degradations`` from
  ``convert_non_pdf``.

ONE place knows which backend serves which format — the web button and the
API can no longer drift into two qualities for the same file (P1: non-PDF;
P2: PDF, which retired ``services/pdf_extraction`` — the detector/ensemble/
multi-page build whose only consumer was the web button).

Routing (the DOC-ENGINE bake-off winners, invocations verbatim in
``services/office_backends.py``):

* DOCX → pandoc, PPTX → markitdown, HTML/HTM → trafilatura (+ metadata head;
  an empty extraction degrades to the unstructured path with a named
  ``backend_fallback`` entry instead of failing),
* everything else (EML/TXT/MD) → ``unstructured`` ``strategy='fast'`` +
  ``elements_to_markdown`` (decision doc: functional, without competition).

Return shape: ``(markdown, degradations)`` with STRUCTURED degradation
entries (``{code, message, pages}`` via ``document_conversions.degradation``),
not bare warning strings — a flat string list would collapse the API's
``backend_fallback``/``serializer`` distinction, i.e. silently change the
answer shape (a DOC-WEB non-goal). Backend warnings become ``serializer``
entries here, at the single spot that knows their origin.
"""
from services.document_conversions import (
    DEGRADATION_BACKEND_FALLBACK,
    DEGRADATION_SERIALIZER,
    degradation,
)


def _serializer_degradations(warnings):
    """Backend warning strings → structured ``serializer`` entries."""
    return [degradation(DEGRADATION_SERIALIZER, w) for w in warnings]


def _convert_unstructured(source_path, source_ext):
    """The unstructured + serializer path (EML/TXT/MD, and the fallback
    target for an empty trafilatura extraction)."""
    from unstructured.partition.auto import partition

    from services.unstructured_markdown import elements_to_markdown

    elements = partition(filename=source_path, strategy="fast")
    markdown, warnings = elements_to_markdown(elements, source_ext=source_ext)
    return markdown, _serializer_degradations(warnings)


def convert_non_pdf(source_path, source_ext):
    """Convert a non-PDF document with its format's measured winner.

    ``source_ext`` is the lowercased extension without dot (both callers
    validate it against their accepted list BEFORE calling — this function
    routes, it does not gatekeep; an unknown extension lands on the
    unstructured path, which is what the legacy task branch did too).

    Returns ``(markdown, degradations)``; raises whatever the backend raises
    (missing binary, empty output) — the caller decides between job-failure
    (API) and HTTP 500 (web).
    """
    if source_ext == 'docx':
        from services.office_backends import convert_docx_pandoc

        markdown, warnings = convert_docx_pandoc(source_path)
    elif source_ext == 'pptx':
        from services.office_backends import convert_pptx_markitdown

        markdown, warnings = convert_pptx_markitdown(source_path)
    elif source_ext in ('html', 'htm'):
        from services.office_backends import convert_html_trafilatura

        markdown, warnings = convert_html_trafilatura(source_path)
        if markdown is None:
            # trafilatura found no main content → the unstructured path
            # serves the result and the switch is NAMED (the legacy path
            # could always serve some text for HTML; a hard fail would be a
            # capability regression, and re-submitting would never converge).
            markdown, degradations = _convert_unstructured(
                source_path, source_ext)
            degradations.append(degradation(
                DEGRADATION_BACKEND_FALLBACK,
                'Artikel-Extraktion fand keinen Hauptinhalt. '
                'Element-Extraktion übernommen.'))
            return markdown, degradations
    else:
        return _convert_unstructured(source_path, source_ext)
    return markdown, _serializer_degradations(warnings)


def convert_pdf(source_path, mode, budget_eur, page_count):
    """PDF branch: cloud = page-wise gemini, local = mineru (DOC-LOCAL).

    Returns the full result payload (``build_result_payload`` shape) — the
    API task writes it verbatim, the web path reads ``markdown`` and
    ``degradations`` from it.

    * **Cloud run** (mode ``cloud``, key present, pre-flight passes):
      ``services.pdf_cloud.run_cloud_pdf`` — one call per page, per-page
      ``modell`` provenance, costs booked from ``usage_metadata``, and the
      REAL mid-flight cap: if actual costs exhaust the budget mid-document,
      the remaining pages come from the local page function with one named
      ``budget_exceeded`` entry (DOC-ENGINE P2; the mid-flight target is the
      mineru engine too since DOC-LOCAL).
    * **Local run** (mode ``lokal``, or a cloud job degraded here):
      ``services.pdf_local.run_local_pdf`` — the mineru VLM in a sibling
      container. Per-page provenance ``modell`` at cost 0.00 €; if the
      engine itself fails, pages fall back to the text layer with a named
      ``backend_fallback`` entry (scan pages additionally named, DOC-WEB
      2.3). The engine needs a real page count — unknown at submit →
      re-derived here via fitz (an unreadable PDF raises).

    Cloud degrades to local — never aborts — when the key is missing
    (``cloud_unavailable``) or the pre-flight estimate ``page_count × cent``
    exceeds the frozen budget (``budget_exceeded``): if the ESTIMATE already
    says the budget cannot carry the document, not a single call is spent
    on a result that would be mostly local anyway. The mid-flight cap covers
    the complementary case (estimate passed, token-dense pages exhaust the
    budget during the run). Pre-flight entries are PREPENDED to the local
    run's own list, so a ``backend_fallback`` never hides why the job went
    local at all.
    """
    import os

    from app_pkg.config import DOC_CONVERT_CLOUD_CENT_PER_PAGE
    from services.document_conversions import (
        DEGRADATION_BUDGET_EXCEEDED,
        DEGRADATION_CLOUD_UNAVAILABLE,
        MODE_CLOUD,
    )

    degradations = []
    cloud = mode == MODE_CLOUD
    if cloud:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            cloud = False
            degradations.append(degradation(
                DEGRADATION_CLOUD_UNAVAILABLE,
                'Cloud-Pfad nicht konfiguriert (kein API-Key). '
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
