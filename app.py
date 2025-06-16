import os
import asyncio
from pathlib import Path
from flask import Flask, render_template, request, flash, redirect, url_for
from markdown_it import MarkdownIt
from playwright.async_api import async_playwright
from werkzeug.utils import secure_filename

# --- Configuration ---
# Use an environment variable for the secret key in a real production app
SECRET_KEY = os.urandom(24)
# Directory where PDFs will be saved INSIDE the container
OUTPUT_DIR = Path('/app/output')
# Directory where CSS themes are located INSIDE the container
STYLE_DIR = Path('/app/static/css/pdf_styles')

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
OUTPUT_DIR.mkdir(exist_ok=True) # Ensure the output directory exists

# --- Markdown Parser Initialization ---
# ALTERNATIVE APPROACH: Use markdown-it-py built-in table support
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# This function is what markdown-it-py will call for syntax highlighting
def highlight_code(code, lang, _):
    try:
        lexer = get_lexer_by_name(lang, stripall=True)
    except:
        lexer = get_lexer_by_name('text', stripall=True)
    formatter = HtmlFormatter(style='default', cssclass='highlight', noclasses=True)
    return highlight(code, lexer, formatter)

# Initialize the parser - 'default' preset includes table support
md = MarkdownIt(
    'default',
    {'breaks': True, 'html': True, 'highlight': highlight_code}
)

@app.route('/')
def index():
    """Renders the main page with the form."""
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
async def convert():
    """Handles the form submission and PDF conversion."""
    markdown_text = request.form.get('markdown_text')
    markdown_file = request.files.get('markdown_file')

    # --- Input Validation ---
    if markdown_file and markdown_file.filename:
        markdown_text = markdown_file.read().decode('utf-8')
    elif not markdown_text or not markdown_text.strip():
        flash('Error: No Markdown content provided. Please paste text or upload a file.', 'danger')
        return redirect(url_for('index'))

    # --- Filename and Path Sanitization ---
    output_filename = request.form.get('output_filename')
    safe_filename = secure_filename(output_filename)
    if not safe_filename:
        flash('Error: Invalid filename provided.', 'danger')
        return redirect(url_for('index'))

    # Construct a safe, absolute path inside the container
    output_path = OUTPUT_DIR / f"{safe_filename}.pdf"

    # --- Style Selection ---
    style_theme = request.form.get('style_theme', 'default')
    style_path = STYLE_DIR / f"{secure_filename(style_theme)}.css"
    style_content = '' # Default to no style
    if style_theme != 'none': # Allow explicit selection of no style
        try:
            with open(style_path, 'r') as f:
                style_content = f.read()
        except FileNotFoundError:
            flash(f'Warning: Style "{style_theme}" not found. Using no style.', 'warning')
            style_content = '' # Ensure it's empty if file not found


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

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(full_html)
            await page.pdf(
                path=str(output_path), # Ensure path is a string
                format='A4',
                print_background=True,
                margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'}
            )
            await browser.close()
    except Exception as e:
        app.logger.error(f"PDF generation failed: {e}")
        flash(f'Error: Could not generate PDF. {e}', 'danger')
        return redirect(url_for('index'))

    flash(f"Success! Saved PDF as '{output_path.name}' on the server.", 'success')
    return redirect(url_for('index'))


from asgiref.wsgi import WsgiToAsgi
asgi_app = WsgiToAsgi(app)

# Note: Gunicorn will run the app, so this part is for direct execution (e.g., local testing)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)