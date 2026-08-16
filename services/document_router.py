"""One router for both entrances — the non-PDF format dispatch (DOC-WEB P1).

Pure module: no Flask, no SDK singleton, no top-level heavy imports (backend
libraries load inside the branch that needs them, exactly like the in-task
imports this logic came from in ``tasks.py``). Both callers share it:

* ``tasks.convert_document_task`` wraps the pair into the document-level
  deterministic result payload (``_deterministic_document_payload``);
* ``app_pkg/documents.py::transform_document`` writes the markdown straight
  into the download.

ONE place knows which backend serves which format — the web button and the
API can no longer drift into two qualities for the same file.

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
