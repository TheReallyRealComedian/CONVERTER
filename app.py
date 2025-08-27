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


# ========================================================
# ===== ROUTES FOR UNSTRUCTURED CONVERTER (NEW) =====
# ========================================================

@app.route('/document-converter')
def document_converter():
    """Renders the UI for the document converter."""
    return render_template('document_converter.html')

@app.route('/transform-document', methods=['POST'])
def transform_document():
    """Handles file upload and transformation using unstructured."""
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

        # Partition the document using unstructured
        # Using strategy="hi_res" can improve link detection in some PDFs
        elements = partition(filename=temp_file_path, strategy="hi_res")

        # ==========================================================
        # ===== NEW LOGIC TO PROCESS LINKS AND CREATE MARKDOWN =====
        # ==========================================================
        markdown_parts = []
        for el in elements:
            # Start with the plain text of the element
            text = el.text

            # Check if the element's metadata has link information
            # getattr is used for safety in case .metadata.links doesn't exist
            links = getattr(el.metadata, 'links', [])
            
            if links:
                # If there are links, iterate through them
                for link in links:
                    # The link object has .text and .url attributes
                    link_text = link.get('text')
                    link_url = link.get('url')

                    # To avoid errors, only proceed if we have both text and a URL
                    if link_text and link_url:
                        # Replace the plain text of the link with Markdown format
                        # This preserves text around the link in the same element
                        markdown_link = f"[{link_text.strip()}]({link_url})"
                        text = text.replace(link_text, markdown_link)
            
            markdown_parts.append(text)

        # Join the processed parts into a single Markdown string
        output_markdown = "\n\n".join(markdown_parts)
        
        # Change the output filename to .md
        output_path_obj = Path(original_filename)
        output_filename = f"{output_path_obj.stem}.md"

        buffer = BytesIO()
        buffer.write(output_markdown.encode('utf-8'))
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=output_filename,
            # Set the correct mimetype for Markdown files
            mimetype='text/markdown'
        )
        # ===== END OF NEW LOGIC =====
        # ============================

    except Exception as e:
        app.logger.error(f"Unstructured processing failed: {e}")
        flash(f'Error processing file: {e}', 'danger')
        return redirect(url_for('document_converter'))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- ASGI Wrapper ---
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)