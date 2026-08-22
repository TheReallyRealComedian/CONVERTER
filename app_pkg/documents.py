"""Document → Markdown converter page and its synchronous route.

Since DOC-WEB the browser button and the API run the SAME router
(``services.document_router``) — one quality per file. Since DOC-WEB-ASYNC
the browser uses BOTH entrances, split by format in
``static/js/document_converter.js``:

* non-PDF formats (DOCX/PPTX/HTML/EML/TXT/MD) stay SYNCHRONOUS on
  ``POST /transform-document`` — the office/web backends are CPU-seconds
  and need no container;
* PDFs go to the service (``POST /api/document-conversions`` + poll). Two
  reasons, both measured/verified 2026-08-21: (1) the single gunicorn
  worker serves NOTHING else while a synchronous conversion runs — asgiref's
  ``WsgiToAsgi`` runs the WSGI app on one ``thread_sensitive`` executor, so
  a 78 s conversion stalled every probe (``/login`` included) for its full
  remaining time; only a job frees that thread. (2) The web container is
  reachable from the internet and holds NO Docker socket anymore; the local
  PDF engine (mineru sibling container) lives on the worker. A PDF arriving
  here is therefore answered with a pointer to the service and never run —
  ``mode=lokal`` on this container would silently degrade to the bare text
  layer, the exact trap DOC-WEB named.
"""
import os
import tempfile
from pathlib import Path

from flask import jsonify, render_template, request
from flask_login import login_required
from werkzeug.utils import secure_filename

from services.document_router import convert_non_pdf


# Single source of truth for what the converter page accepts (both entrances:
# this route for non-PDF, the service for PDF). The template reads this via
# the route context and exposes it to JS as window.PageData.acceptedExtensions
# for client-side prevalidation and the PDF/non-PDF split.
ACCEPTED_EXTENSIONS = ('pdf', 'docx', 'pptx', 'eml', 'html', 'htm', 'txt', 'md')


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
        if ext == 'pdf':
            # Never run a PDF engine on this container (module docstring):
            # the page's JS submits PDFs to the service itself; a PDF landing
            # here is a caller on the wrong route, not a conversion to do.
            return jsonify({
                'error': 'PDF läuft über den Dienst: POST /api/document-conversions.'
            }), 400

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as temp_f:
                file.save(temp_f.name)
                temp_file_path = temp_f.name

            # The SAME router the API task uses (DOC-WEB P1) — one place knows
            # the formats, web and API can't drift apart.
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
