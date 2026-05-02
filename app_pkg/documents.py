"""Document → Markdown converter routes (DOCX/PDF/HTML/EML/...)."""
import os
import tempfile
from io import BytesIO
from pathlib import Path

from flask import jsonify, render_template, request, send_file
from flask_login import login_required
from werkzeug.utils import secure_filename


def register(app):
    # Late import: tests patch ``app.partition`` and
    # ``app.pdf_extraction_service`` on the top-level app.py module, so
    # look those up at call time rather than capturing imports here.
    import app as _app_module

    @app.route('/document-converter')
    @login_required
    def document_converter():
        return render_template('document_converter.html')

    @app.route('/transform-document', methods=['POST'])
    @login_required
    def transform_document():
        if 'document_file' not in request.files:
            return jsonify({'error': 'No file part in the request.'}), 400

        file = request.files['document_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400

        if not file:
            return jsonify({'error': 'No file provided.'}), 400

        original_filename = secure_filename(file.filename)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as temp_f:
                file.save(temp_f.name)
                temp_file_path = temp_f.name

            file_ext = Path(original_filename).suffix.lower()

            if file_ext == '.pdf':
                # PDF: Hybrid-Extraktion mit Tabellenerkennung (PyMuPDF + Gemini Vision)
                app.logger.info("PDF erkannt - verwende PDFExtractionService mit Tabellenerkennung...")
                output_markdown = _app_module.pdf_extraction_service.extract_markdown(temp_file_path)
            else:
                # Andere Formate (DOCX, PPTX, HTML, EML, etc.): bestehende unstructured Pipeline
                app.logger.info("Partitioning document with unstructured (strategy='fast')...")
                elements = _app_module.partition(filename=temp_file_path, strategy="fast")
                output_markdown = "\n\n".join([el.text for el in elements])

            output_path_obj = Path(original_filename)
            output_filename = f"{output_path_obj.stem}.md"

            buffer = BytesIO()
            buffer.write(output_markdown.encode('utf-8'))
            buffer.seek(0)

            return send_file(
                buffer,
                as_attachment=True,
                download_name=output_filename,
                mimetype='text/markdown'
            )

        except Exception as e:
            app.logger.error(f"Unstructured processing failed: {e}", exc_info=True)
            return jsonify({'error': 'Error processing file. Please try again.'}), 500
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
