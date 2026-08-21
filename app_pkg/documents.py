"""Document → Markdown converter routes (DOCX/PDF/HTML/EML/...).

Since DOC-WEB the browser button and the API run the SAME router
(``services.document_router``): non-PDF formats through the DOC-ENGINE
winners, PDFs through the real engines (gemini page-wise / mineru) in the
mode Oli set for the service (``document_api`` settings namespace — no
second switch) under the same budget cap. The difference to the API is
only the job model: this route is SYNCHRONOUS on the single gunicorn
worker, hence the named page limit ``MAX_SYNC_PDF_PAGES``.
"""
import os
import tempfile
from pathlib import Path

from flask import jsonify, render_template, request
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app_pkg.config import DOC_CONVERT_BUDGET_EUR, MAX_SYNC_PDF_PAGES
from services.document_router import convert_non_pdf, convert_pdf


# Single source of truth for what /transform-document accepts. The template
# reads this via the route context and exposes it to JS as
# window.PageData.acceptedExtensions for client-side prevalidation.
ACCEPTED_EXTENSIONS = ('pdf', 'docx', 'pptx', 'eml', 'html', 'htm', 'txt', 'md')


def _pdf_page_count(path):
    """Page count via fitz, or ``None`` when the file is not a readable PDF
    (the engine then raises its own honest error)."""
    try:
        import fitz
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return None


def _default_pdf_mode(user):
    """Oli's service-side default mode (``document_api`` settings namespace)
    — the browser follows it, no second switch. Lazy import: ``document_api``
    imports ``ACCEPTED_EXTENSIONS`` from this module (one accepted list for
    both entrances), a top-level import here would close the cycle."""
    from app_pkg.document_api import get_doc_api_settings
    return get_doc_api_settings(user)['default_mode']


def _too_many_pages_response(page_count):
    return jsonify({
        'error': f'Dieses PDF hat {page_count} Seiten, im Browser sind bis zu '
                 f'{MAX_SYNC_PDF_PAGES} möglich. Nutze dafür den Dienst unter '
                 f'POST /api/document-conversions.'
    }), 413


def register(app):

    @app.route('/document-converter')
    @login_required
    def document_converter():
        return render_template(
            'document_converter.html',
            accepted_extensions=ACCEPTED_EXTENSIONS,
            accepted_extensions_accept=','.join('.' + ext for ext in ACCEPTED_EXTENSIONS),
        )

    @app.route('/transform-document', methods=['POST'])
    @login_required
    def transform_document():
        if 'document_file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        file = request.files['document_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400

        original_filename = secure_filename(file.filename)
        ext = os.path.splitext(original_filename)[1].lstrip('.').lower()
        if ext not in ACCEPTED_EXTENSIONS:
            return jsonify({
                'error': 'Dieser Dateityp wird nicht unterstützt. '
                         'Erlaubt: PDF, DOCX, PPTX, EML, HTML, TXT, MD.'
            }), 400
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as temp_f:
                file.save(temp_f.name)
                temp_file_path = temp_f.name

            file_ext = Path(original_filename).suffix.lower()

            if file_ext == '.pdf':
                # PDF: the real engines (DOC-WEB P2), mode from Oli's
                # service setting, same budget cap as the API. Synchronous →
                # named page limit instead of a 1800 s gunicorn timeout.
                page_count = _pdf_page_count(temp_file_path)
                if page_count is not None and page_count > MAX_SYNC_PDF_PAGES:
                    return _too_many_pages_response(page_count)
                mode = _default_pdf_mode(current_user)
                app.logger.info("PDF via document_router (mode=%s, pages=%s)...",
                                mode, page_count)
                payload = convert_pdf(temp_file_path, mode,
                                      DOC_CONVERT_BUDGET_EUR, page_count)
                output_markdown = payload['markdown']
                degradations = payload['degradations']
            else:
                # Non-PDF: the SAME router the API task uses (DOC-WEB P1) —
                # one place knows the formats, web and API can't drift apart.
                app.logger.info("Routing document via document_router (ext=%s)...", ext)
                output_markdown, degradations = convert_non_pdf(temp_file_path, ext)
            for entry in degradations:
                app.logger.warning(
                    "Konvertierung (%s) [%s]: %s",
                    original_filename, entry['code'], entry['message'])

            # JSON instead of a file download (DOC-WEB P3): the degradations
            # reach the user, not only the log. The page already built the
            # download client-side from the response text (Blob), so the
            # attachment semantics were never used by the UI.
            return jsonify({
                'markdown': output_markdown,
                'filename': f"{Path(original_filename).stem}.md",
                'degradations': degradations,
            })

        except Exception as e:
            app.logger.error(f"Document conversion failed: {e}", exc_info=True)
            return jsonify({'error': 'Error processing file. Please try again.'}), 500
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
