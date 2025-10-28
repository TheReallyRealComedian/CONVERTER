import os
import asyncio
import tempfile
import logging
import sys
from pathlib import Path
from io import BytesIO
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from markdown_it import MarkdownIt
from playwright.async_api import async_playwright
from werkzeug.utils import secure_filename
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from unstructured.partition.auto import partition
from asgiref.wsgi import WsgiToAsgi
import fitz  # PyMuPDF
from deepgram import DeepgramClient, PrerecordedOptions
from flask import jsonify
import traceback


# ==========================================================
# ===== LOGGING CONFIGURATION =====
# ==========================================================
# Configure logging to output to stdout, which is standard for containers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# ==========================================================


# --- Configuration ---
SECRET_KEY = os.urandom(24)
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
STYLE_DIR = Path('/app/static/css/pdf_styles')

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# --- Markdown Parser Initialization ---
def highlight_code(code, lang, _):
    try:
        lexer = get_lexer_by_name(lang, stripall=True)
    except:
        lexer = get_lexer_by_name('text', stripall=True)
    formatter = HtmlFormatter(style='default', cssclass='highlight', noclasses=True)
    return highlight(code, lexer, formatter)

md = MarkdownIt(
    'default',
    {'breaks': True, 'html': True, 'highlight': highlight_code}
)

# ==========================================================
# ===== ROUTES FOR MARKDOWN TO PDF CONVERTER =====
# ==========================================================
@app.route('/')
def markdown_converter():
    """Renders the main page for the markdown converter."""
    themes = []
    if STYLE_DIR.exists():
        for f in STYLE_DIR.glob('*.css'):
            themes.append(f.stem)
    return render_template('markdown_converter.html', themes=sorted(themes))

@app.route('/convert-markdown', methods=['POST'])
async def convert_markdown():
    """Handles the form submission and PDF conversion."""
    markdown_text = request.form.get('markdown_text')
    markdown_file = request.files.get('markdown_file')

    # --- Input Validation ---
    if markdown_file and markdown_file.filename:
        markdown_text = markdown_file.read().decode('utf-8')
    elif not markdown_text or not markdown_text.strip():
        flash('Error: No Markdown content provided. Please paste text or upload a file.', 'danger')
        return redirect(url_for('markdown_converter'))

    # --- Filename Sanitization ---
    output_filename = request.form.get('output_filename')
    safe_filename = secure_filename(output_filename)
    if not safe_filename:
        flash('Error: Invalid filename provided.', 'danger')
        return redirect(url_for('markdown_converter'))

    # --- Style Selection ---
    style_theme = request.form.get('style_theme', 'default')
    style_path = STYLE_DIR / f"{secure_filename(style_theme)}.css"
    style_content = ''
    if style_theme != 'none':
        try:
            with open(style_path, 'r') as f:
                style_content = f.read()
        except FileNotFoundError:
            flash(f'Warning: Style "{style_theme}" not found. Using no style.', 'warning')
            style_content = ''

    # --- Conversion Logic ---
    html_content = md.render(markdown_text)
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>{style_content}</style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    temp_pdf_path = None # Initialize to avoid UnboundLocalError in finally
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(full_html)
            await page.pdf(
                path=temp_pdf_path,
                format='A4',
                print_background=True,
                margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'}
            )
            await browser.close()

        return send_file(
            temp_pdf_path,
            as_attachment=True,
            download_name=f"{safe_filename}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        app.logger.error(f"PDF generation failed: {e}")
        flash(f'Error: Could not generate PDF. {e}', 'danger')
        return render_template('markdown_converter.html', markdown_text=markdown_text)
    finally:
        try:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
        except Exception as e:
            app.logger.error(f"Error cleaning up temp file {temp_pdf_path}: {e}")


# ==========================================================
# ===== ROUTES FOR MERMAID DIAGRAMS =====
# ==========================================================
@app.route('/mermaid-converter')
def mermaid_converter():
    """Renders the UI for the Mermaid diagram converter."""
    return render_template('mermaid_converter.html')


# ========================================================
# ===== ROUTES FOR UNSTRUCTURED CONVERTER (NEW) =====
# ========================================================

@app.route('/document-converter')
def document_converter():
    """Renders the UI for the document converter."""
    return render_template('document_converter.html')

@app.route('/transform-document', methods=['POST'])
def transform_document():
    """Handles file upload and transformation using a robust hybrid approach."""
    if 'document_file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(url_for('document_converter'))

    file = request.files['document_file']
    if file.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('document_converter'))

    if not file:
        return redirect(url_for('document_converter'))

    original_filename = secure_filename(file.filename)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as temp_f:
            file.save(temp_f.name)
            temp_file_path = temp_f.name

        # ==========================================================
        # ===== NEW HYBRID LOGIC: PyMuPDF + Unstructured
        # ==========================================================
        
        # --- Part 1: Use PyMuPDF to build a definitive link map ---
        link_map = {}
        try:
            app.logger.info("Starting PyMuPDF link extraction...")
            doc = fitz.open(temp_file_path)
            for page_num, page in enumerate(doc):
                links = page.get_links()
                for link in links:
                    if link.get('kind') == fitz.LINK_URI:
                        # The 'from' key is a Rect object defining the clickable area
                        clickable_area = link['from']
                        # Extract the text that falls inside that clickable area
                        link_text = page.get_textbox(clickable_area).strip().replace('\n', ' ')
                        link_url = link.get('uri')
                        
                        if link_text and link_url:
                            # Store the text and URL. We use the text as a key.
                            # This handles cases where the same text links to the same URL multiple times.
                            link_map[link_text] = link_url
                            app.logger.info(f"PyMuPDF found link: '{link_text}' -> '{link_url}'")
            doc.close()
            app.logger.info(f"PyMuPDF finished. Found {len(link_map)} unique links.")
        except Exception as e:
            app.logger.error(f"PyMuPDF failed to process links: {e}", exc_info=True)
            # We can still proceed without links if this fails
            link_map = {}

        # --- Part 2: Use unstructured for the main text body ---
        app.logger.info("Partitioning document with unstructured (strategy='fast')...")
        elements = partition(filename=temp_file_path, strategy="fast")
        # Get the full plain text output from unstructured
        full_text = "\n\n".join([el.text for el in elements])
        
        # --- Part 3: Merge the results ---
        app.logger.info("Merging unstructured text with PyMuPDF link map...")
        output_markdown = full_text
        if link_map:
            # Sort keys by length, longest first, to avoid partial replacements
            # e.g., replace "McKinsey & Company" before "McKinsey"
            for link_text in sorted(link_map.keys(), key=len, reverse=True):
                link_url = link_map[link_text]
                markdown_link = f"[{link_text}]({link_url})"
                # Perform a simple but effective replacement on the entire text block
                output_markdown = output_markdown.replace(link_text, markdown_link)
        
        # ==========================================================
        # ===== END OF NEW HYBRID LOGIC
        # ==========================================================

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
        flash(f'Error processing file: {e}', 'danger')
        return redirect(url_for('document_converter'))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ==========================================================
# ===== ROUTES FOR AUDIO TRANSCRIPTION (DEEPGRAM) =====
# ==========================================================

@app.route('/audio-converter')
def audio_converter():
    """Renders the UI for the audio converter."""
    return render_template('audio_converter.html', deepgram_api_key_set=bool(DEEPGRAM_API_KEY))

@app.route('/api/get-deepgram-token', methods=['GET'])
def get_deepgram_token():
    """Provides the Deepgram API Key to the frontend for live transcription."""
    if not DEEPGRAM_API_KEY:
        app.logger.error("DEEPGRAM_API_KEY not configured on the server.")
        return jsonify({"error": "Audio transcription service is not configured."}), 503
    
    # For a production app with user accounts, you might generate a short-lived key here.
    # For this application, we provide the main key as the app is self-contained.
    return jsonify({"deepgram_token": DEEPGRAM_API_KEY})

@app.route('/transcribe-audio-file', methods=['POST'])
def transcribe_audio_file():
    """Handles audio file upload and transcription via Deepgram."""
    if not DEEPGRAM_API_KEY:
        return jsonify({"error": "Audio transcription service is not configured."}), 503

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request."}), 400

    file = request.files['audio_file']
    language = request.form.get('language', 'en') # Default to English

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        # Initialize Deepgram client
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        # Read file into a buffer
        buffer_data = file.read()
        payload = { "buffer": buffer_data }

        # Configure transcription options
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            punctuate=True,
            language=language
        )

        # Send the request to Deepgram
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        
        # Extract the transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        
        return jsonify({"transcript": transcript})

    except Exception as e:
        app.logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during transcription: {str(e)}"}), 500

# --- ASGI Wrapper ---
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)