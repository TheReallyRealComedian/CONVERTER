import os
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone
from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job
from io import BytesIO
from flask import render_template, request, flash, redirect, url_for, send_file, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from markdown_it import MarkdownIt
from playwright.async_api import async_playwright
from html.parser import HTMLParser
from werkzeug.utils import secure_filename
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from unstructured.partition.auto import partition
from asgiref.wsgi import WsgiToAsgi
import nh3
import fitz
import re
import traceback
import time as _time
import requests as http_requests

from app_pkg import create_app
from services import DeepgramService, GeminiService, GoogleTTSService, PDFExtractionService
from tasks import generate_podcast_task
from models import db, User, Conversion


DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GOOGLE_CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
NOTION_MCP_URL = os.environ.get('NOTION_MCP_URL', 'http://localhost:3333')
MCP_AUTH_TOKEN = os.environ.get('MCP_AUTH_TOKEN', '')
NOTION_TOKEN = os.environ.get('NOTION_TOKEN', '')
STYLE_DIR = Path('/app/static/css/pdf_styles')

# --- Notion suggestions cache ---
_notion_cache = {}

def _notion_api(method, path, body=None):
    headers = {'Authorization': f'Bearer {NOTION_TOKEN}', 'Notion-Version': '2022-06-28'}
    url = f'https://api.notion.com/v1{path}'
    if method == 'GET':
        return http_requests.get(url, headers=headers, timeout=15)
    return http_requests.post(url, json=body or {}, headers=headers, timeout=15)

def _cached(key, ttl, fetcher):
    entry = _notion_cache.get(key)
    if entry and _time.time() < entry['exp']:
        return entry['data']
    data = fetcher()
    _notion_cache[key] = {'data': data, 'exp': _time.time() + ttl}
    return data

def _get_notion_db_ids():
    def fetch():
        resp = _notion_api('POST', '/search', {
            'filter': {'value': 'database', 'property': 'object'}, 'page_size': 100
        })
        if resp.status_code != 200:
            return {}
        ids = {}
        for db in resp.json().get('results', []):
            name = ''.join(t.get('plain_text', '') for t in db.get('title', [])).strip().upper()
            if name:
                ids[name] = db['id']
        return ids
    return _cached('db_ids', 3600, fetch)

def _query_db_titles(db_id):
    resp = _notion_api('POST', f'/databases/{db_id}/query', {'page_size': 100})
    if resp.status_code != 200:
        return []
    titles = []
    for page in resp.json().get('results', []):
        for prop in page.get('properties', {}).values():
            if prop.get('type') == 'title':
                t = ''.join(p.get('plain_text', '') for p in prop.get('title', []))
                if t:
                    titles.append(t)
                break
    return sorted(set(titles))

def _get_select_options(db_id, prop_name):
    resp = _notion_api('GET', f'/databases/{db_id}')
    if resp.status_code != 200:
        return []
    for name, prop in resp.json().get('properties', {}).items():
        if name.lower() == prop_name.lower() and prop.get('type') == 'select':
            return [o['name'] for o in prop.get('select', {}).get('options', [])]
    return []

app = create_app()

# Re-export the CSRFProtect instance at module level so test code (and any
# future caller that wants to exempt a route) can reach it via ``app.csrf``.
csrf = app.extensions['csrf']

# Initialize services
deepgram_service = DeepgramService(DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
gemini_service = GeminiService(GEMINI_API_KEY) if GEMINI_API_KEY else None
google_tts_service = GoogleTTSService(GOOGLE_CREDENTIALS_PATH) if GOOGLE_CREDENTIALS_PATH else None
pdf_extraction_service = PDFExtractionService(GEMINI_API_KEY)

# Redis Queue setup
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
redis_conn = Redis.from_url(REDIS_URL)
task_queue = Queue(connection=redis_conn)

# Shared output directory (must match tasks.py and docker-compose volume)
OUTPUT_DIR = '/app/output_podcasts'


# --- Auth Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('markdown_converter'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            if next_page:
                from urllib.parse import urlparse
                parsed = urlparse(next_page)
                if parsed.netloc or parsed.scheme:
                    next_page = None
            return redirect(next_page or url_for('markdown_converter'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


def highlight_code(code, lang, _):
    try:
        lexer = get_lexer_by_name(lang, stripall=True)
    except Exception:
        lexer = get_lexer_by_name('text', stripall=True)
    formatter = HtmlFormatter(style='default', cssclass='highlight', noclasses=True)
    return highlight(code, lexer, formatter)

class _TableColumnCounter(HTMLParser):
    """Counts columns in each <table> to detect wide tables."""

    def __init__(self):
        super().__init__()
        self._in_table = False
        self._in_first_row = False
        self._col_count = 0
        self._tables = []  # list of (start_offset, col_count)
        self._table_start = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._in_table = True
            self._in_first_row = True
            self._col_count = 0
            self._table_start = self.getpos()
        elif tag in ('th', 'td') and self._in_first_row:
            self._col_count += 1
        elif tag == 'tr' and self._in_table and self._col_count > 0:
            # Second <tr> means first row is done
            self._in_first_row = False

    def handle_endtag(self, tag):
        if tag == 'table' and self._in_table:
            self._tables.append((self._table_start, self._col_count))
            self._in_table = False
            self._in_first_row = False


def _wrap_wide_tables(html: str, column_threshold: int = 6) -> str:
    """Wrap tables with >= column_threshold columns in a landscape div."""
    parser = _TableColumnCounter()
    parser.feed(html)

    if not parser._tables:
        return html

    # Process in reverse so string offsets remain valid
    lines = html.split('\n')
    for (line_no, _col_offset), col_count in reversed(parser._tables):
        if col_count < column_threshold:
            continue
        # Find the <table> tag in the source and its closing </table>
        # line_no is 1-based from HTMLParser.getpos()
        table_line_idx = line_no - 1

        # Find the line with </table> starting from table_line_idx
        end_idx = table_line_idx
        for i in range(table_line_idx, len(lines)):
            if '</table>' in lines[i]:
                end_idx = i
                break

        lines[end_idx] = lines[end_idx].replace('</table>', '</table>\n</div>', 1)
        lines[table_line_idx] = '<div class="landscape-table">\n' + lines[table_line_idx]

    return '\n'.join(lines)


md = MarkdownIt(
    'default',
    {'breaks': True, 'html': True, 'highlight': highlight_code}
)


@app.route('/')
@login_required
def markdown_converter():
    themes = []
    if STYLE_DIR.exists():
        for f in STYLE_DIR.glob('*.css'):
            themes.append(f.stem)
    return render_template('markdown_converter.html', themes=sorted(themes))

@app.route('/convert-markdown', methods=['POST'])
@login_required
async def convert_markdown():
    markdown_text = request.form.get('markdown_text')
    markdown_file = request.files.get('markdown_file')

    if markdown_file and markdown_file.filename:
        markdown_text = markdown_file.read().decode('utf-8')
    elif not markdown_text or not markdown_text.strip():
        flash('Error: No Markdown content provided. Please paste text or upload a file.', 'danger')
        return redirect(url_for('markdown_converter'))

    output_filename = request.form.get('output_filename')
    safe_filename = secure_filename(output_filename)
    if not safe_filename:
        flash('Error: Invalid filename provided.', 'danger')
        return redirect(url_for('markdown_converter'))

    orientation = request.form.get('orientation', 'portrait')
    is_landscape = orientation == 'landscape'

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

    html_content = md.render(markdown_text)
    html_content = nh3.clean(
        html_content,
        tags={
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'br', 'hr', 'blockquote', 'pre', 'code',
            'ul', 'ol', 'li', 'dl', 'dt', 'dd',
            'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td', 'caption', 'colgroup', 'col',
            'a', 'img', 'figure', 'figcaption',
            'strong', 'em', 'b', 'i', 'u', 's', 'del', 'ins', 'mark',
            'sub', 'sup', 'small', 'abbr', 'cite', 'q', 'kbd', 'var', 'samp',
            'details', 'summary',
            'div', 'span', 'section', 'article', 'aside', 'header', 'footer', 'nav', 'main',
        },
        attributes={
            '*': {'class', 'id', 'style'},
            'a': {'href', 'title', 'target'},
            'img': {'src', 'alt', 'title', 'width', 'height'},
            'th': {'colspan', 'rowspan', 'scope'},
            'td': {'colspan', 'rowspan'},
            'col': {'span'},
            'colgroup': {'span'},
        },
    )
    html_content = _wrap_wide_tables(html_content, column_threshold=6)
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>{style_content}</style>
        {'<style>@page { size: A4 landscape; }</style>' if is_landscape else ''}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(full_html, wait_until='networkidle')
            # Wait until all web fonts (Google Fonts etc.) are loaded before rendering
            await page.evaluate('document.fonts.ready')
            # Detect tables that overflow the page width (few but very wide columns)
            await page.evaluate('''() => {
                document.querySelectorAll('table').forEach(table => {
                    if (table.scrollWidth > table.parentElement.clientWidth
                        && !table.closest('.landscape-table')) {
                        const wrapper = document.createElement('div');
                        wrapper.className = 'landscape-table';
                        table.parentNode.insertBefore(wrapper, table);
                        wrapper.appendChild(table);
                    }
                });
            }''')
            await page.pdf(
                path=temp_pdf_path,
                format='A4',
                landscape=is_landscape,
                print_background=True,
                margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'}
            )
            await browser.close()

        # Read into buffer so temp file can safely be deleted
        pdf_buffer = BytesIO()
        with open(temp_pdf_path, 'rb') as f:
            pdf_buffer.write(f.read())
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"{safe_filename}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        app.logger.error(f"PDF generation failed: {e}")
        flash('Error: Could not generate PDF. Please try again.', 'danger')
        return render_template('markdown_converter.html', markdown_text=markdown_text)
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                app.logger.error(f"Error cleaning up temp file {temp_pdf_path}: {e}")


@app.route('/mermaid-converter')
@login_required
def mermaid_converter():
    return render_template('mermaid_converter.html')


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
            output_markdown = pdf_extraction_service.extract_markdown(temp_file_path)
        else:
            # Andere Formate (DOCX, PPTX, HTML, EML, etc.): bestehende unstructured Pipeline
            app.logger.info("Partitioning document with unstructured (strategy='fast')...")
            elements = partition(filename=temp_file_path, strategy="fast")
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


@app.route('/audio-converter')
@login_required
def audio_converter():
    return render_template('audio_converter.html', deepgram_api_key_set=bool(DEEPGRAM_API_KEY))

@app.route('/api/get-deepgram-token', methods=['GET'])
@login_required
def get_deepgram_token():
    if not deepgram_service:
        app.logger.error("Deepgram service not configured")
        return jsonify({"error": "Audio transcription service is not configured."}), 503

    try:
        temp_key = deepgram_service.create_temporary_key(ttl_seconds=60)
        return jsonify({"deepgram_token": temp_key})
    except Exception as e:
        app.logger.error(f"Failed to create temporary Deepgram key: {e}")
        return jsonify({"error": "Failed to create transcription token."}), 500

@app.route('/transcribe-audio-file', methods=['POST'])
@login_required
def transcribe_audio_file():
    if not deepgram_service:
        return jsonify({"error": "Audio transcription service is not configured."}), 503

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request."}), 400

    file = request.files['audio_file']
    language = request.form.get('language', 'en')

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        buffer_data = file.read()
        file_size_mb = len(buffer_data) / (1024 * 1024)

        app.logger.info(f"Received audio file: {file.filename} ({file_size_mb:.1f} MB)")

        # transcribe_file handhabt automatisch Splitting wenn nötig
        transcript = deepgram_service.transcribe_file(buffer_data, language)

        return jsonify({
            "transcript": transcript,
            "metadata": {
                "file_size_mb": round(file_size_mb, 2),
                "transcript_length": len(transcript),
                "language": language
            }
        })

    except RuntimeError as e:
        # Chunk-spezifischer Fehler
        app.logger.error(f"Chunked transcription failed: {e}", exc_info=True)
        return jsonify({
            "error": "Transcription of long audio failed. Please try a shorter file."
        }), 500

    except Exception as e:
        app.logger.error(f"Deepgram transcription failed: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during transcription. Please try again."}), 500


@app.route('/generate-podcast', methods=['POST'])
@login_required
def generate_podcast():
    if not google_tts_service:
        return jsonify({"error": "Google Cloud TTS is not configured."}), 503

    temp_audio_path = None
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice_name = data.get('voice_name', 'en-US-Neural2-C')
        language_code = data.get('language_code', 'en-US')
        speaking_rate = float(data.get('speaking_rate', 1.0))
        pitch = float(data.get('pitch', 0.0))

        temp_audio_path = google_tts_service.synthesize_speech(
            text, voice_name, language_code, speaking_rate, pitch
        )

        audio_buffer = BytesIO()
        with open(temp_audio_path, 'rb') as f:
            audio_buffer.write(f.read())
        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            as_attachment=True,
            download_name='podcast.mp3',
            mimetype='audio/mpeg'
        )

    except Exception as e:
        app.logger.error(f"Google TTS synthesis failed: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during synthesis. Please try again."}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                app.logger.error(f"Error cleaning up temp file: {e}")


@app.route('/api/get-google-voices', methods=['GET'])
@login_required
def get_google_voices():
    if not google_tts_service:
        return jsonify({"error": "Google Cloud TTS is not configured."}), 503

    try:
        voices = google_tts_service.list_voices()
        return jsonify(voices)
    except Exception as e:
        app.logger.error(f"Failed to retrieve Google TTS voices: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve voices."}), 500


@app.route('/generate-gemini-podcast', methods=['POST'])
@login_required
def generate_gemini_podcast():
    """Queue a podcast generation job to Redis."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API Key is not configured."}), 503

    try:
        data = request.get_json()
        dialogue = data.get('dialogue', [])
        language = data.get('language', 'en')
        tts_model = data.get('tts_model', None)

        # Enqueue job to Redis (worker will pick it up)
        job = task_queue.enqueue(
            generate_podcast_task,
            args=(dialogue, language, tts_model),
            job_timeout=600,  # 10 minutes max
            meta={'user_id': current_user.id}
        )

        app.logger.info(f"Job {job.get_id()} queued for podcast generation")
        return jsonify({"job_id": job.get_id(), "status": "queued"})

    except Exception as e:
        app.logger.error(f"Failed to queue podcast job: {e}", exc_info=True)
        return jsonify({"error": "Failed to queue podcast job. Please try again."}), 500


@app.route('/podcast-status/<job_id>', methods=['GET'])
@login_required
def podcast_status(job_id):
    """Check the status of a podcast generation job in Redis."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return jsonify({"error": "Job not found"}), 404
    except Exception as e:
        app.logger.error(f"Failed to fetch RQ job {job_id}: {e}", exc_info=True)
        return jsonify({"error": "Job lookup failed"}), 500

    if job.meta.get('user_id') != current_user.id:
        return jsonify({"error": "Job not found"}), 404

    status = job.get_status()

    if status == 'finished':
        return jsonify({"status": "completed", "result": job.result})
    elif status == 'failed':
        error_msg = str(job.exc_info) if job.exc_info else "Unknown error"
        return jsonify({"status": "failed", "error": error_msg})
    elif status in ['queued', 'started']:
        return jsonify({"status": "processing"})
    else:
        return jsonify({"status": status})


@app.route('/podcast-download/<job_id>', methods=['GET'])
@login_required
def podcast_download(job_id):
    """Download the generated podcast file."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return jsonify({"error": "Job not found"}), 404
    except Exception as e:
        app.logger.error(f"Failed to fetch RQ job {job_id}: {e}", exc_info=True)
        return jsonify({"error": "Job lookup failed"}), 500

    if job.meta.get('user_id') != current_user.id:
        return jsonify({"error": "Job not found"}), 404

    if not job.is_finished:
        return jsonify({"error": "Job not ready"}), 400

    file_path = job.result

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found on server"}), 404

    # Prevent path traversal — ensure file is within allowed output directory
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(os.path.realpath(OUTPUT_DIR)):
        app.logger.warning(f"Path traversal attempt blocked: {file_path}")
        return jsonify({"error": "Invalid file path"}), 403

    # Read into buffer and delete file to prevent unbounded disk growth
    podcast_buffer = BytesIO()
    with open(real_path, 'rb') as f:
        podcast_buffer.write(f.read())
    podcast_buffer.seek(0)

    try:
        os.unlink(real_path)
    except Exception as e:
        app.logger.warning(f"Failed to clean up podcast file {real_path}: {e}")

    return send_file(
        podcast_buffer,
        as_attachment=True,
        download_name='gemini_podcast.wav',
        mimetype='audio/wav'
    )


@app.route('/api/get-gemini-voices', methods=['GET'])
@login_required
def get_gemini_voices():
    voices = {
        "male": [
            {"name": "Kore", "description": "Firm and authoritative"},
            {"name": "Charon", "description": "Informative and clear"},
            {"name": "Fenrir", "description": "Excitable and energetic"},
            {"name": "Orus", "description": "Firm and steady"},
            {"name": "Puck", "description": "Upbeat and cheerful"},
            {"name": "Enceladus", "description": "Breathy and soft"},
            {"name": "Iapetus", "description": "Clear and precise"},
            {"name": "Algenib", "description": "Gravelly and deep"},
            {"name": "Achernar", "description": "Soft and gentle"},
            {"name": "Algieba", "description": "Smooth and polished"},
            {"name": "Gacrux", "description": "Mature and experienced"},
            {"name": "Alnilam", "description": "Firm and direct"},
            {"name": "Rasalgethi", "description": "Informative and educational"},
            {"name": "Sadaltager", "description": "Knowledgeable and wise"},
            {"name": "Zubenelgenubi", "description": "Casual and relaxed"}
        ],
        "female": [
            {"name": "Zephyr", "description": "Bright and lively"},
            {"name": "Leda", "description": "Youthful and fresh"},
            {"name": "Laomedeia", "description": "Upbeat and positive"},
            {"name": "Aoede", "description": "Breezy and light"},
            {"name": "Callirrhoe", "description": "Easy-going and friendly"},
            {"name": "Autonoe", "description": "Bright and clear"},
            {"name": "Erinome", "description": "Clear and articulate"},
            {"name": "Umbriel", "description": "Easy-going and calm"},
            {"name": "Despina", "description": "Smooth and flowing"},
            {"name": "Pulcherrima", "description": "Forward and confident"},
            {"name": "Vindemiatrix", "description": "Gentle and warm"}
        ],
        "neutral": [
            {"name": "Kore", "description": "Firm (can be male or female)"},
            {"name": "Achird", "description": "Friendly and approachable"},
            {"name": "Schedar", "description": "Even and balanced"},
            {"name": "Sadachbia", "description": "Lively and animated"},
            {"name": "Sulafat", "description": "Warm and inviting"}
        ]
    }

    return jsonify(voices)


@app.route('/format-dialogue-with-llm', methods=['POST'])
@login_required
def format_dialogue_with_llm():
    if not gemini_service:
        return jsonify({"error": "Gemini API Key is not configured."}), 503

    try:
        data = request.get_json()

        raw_text_received = data.get('raw_text', '')

        if not raw_text_received or not raw_text_received.strip():
            return jsonify({"error": "No text provided for formatting"}), 400

        result = gemini_service.format_dialogue_with_llm(
            raw_text=raw_text_received.strip(),
            num_speakers=int(data.get('num_speakers', 2)),
            speaker_descriptions=data.get('speaker_descriptions', []),
            language=data.get('language', 'en'),
            narration_style=data.get('narration_style', 'conversational'),
            script_length=data.get('script_length', 'medium'),
            custom_prompt=(data.get('custom_prompt') or '').strip() or None
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Dialogue formatting failed: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during dialogue formatting. Please try again."}), 500




# --- Library Routes ---
@app.route('/library')
@login_required
def library():
    conversion_type = request.args.get('type', '')
    search = request.args.get('search', '').strip()
    favorites = request.args.get('favorites', '') == '1'
    sort = request.args.get('sort', 'newest')
    page = request.args.get('page', 1, type=int)
    per_page = 20

    query = Conversion.query.filter_by(user_id=current_user.id)

    if conversion_type:
        query = query.filter_by(conversion_type=conversion_type)
    if favorites:
        query = query.filter_by(is_favorite=True)
    if search:
        escaped_search = re.sub(r'([%_\\])', r'\\\1', search)
        query = query.filter(
            db.or_(
                Conversion.title.ilike(f'%{escaped_search}%'),
                Conversion.content.ilike(f'%{escaped_search}%'),
                Conversion.tags.ilike(f'%{escaped_search}%')
            )
        )

    if sort == 'oldest':
        query = query.order_by(Conversion.created_at.asc())
    elif sort == 'title':
        query = query.order_by(Conversion.title.asc())
    else:
        query = query.order_by(Conversion.created_at.desc())

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    return render_template('library.html',
                           conversions=pagination.items,
                           pagination=pagination,
                           current_type=conversion_type,
                           current_search=search,
                           current_favorites=favorites,
                           current_sort=sort)


@app.route('/library/<int:conversion_id>')
@login_required
def library_detail(conversion_id):
    conversion = Conversion.query.filter_by(id=conversion_id, user_id=current_user.id).first_or_404()
    metadata = json.loads(conversion.metadata_json) if conversion.metadata_json else {}
    return render_template('library_detail.html', conversion=conversion, metadata=metadata)


# --- Conversion API ---
ALLOWED_CONVERSION_TYPES = {'document_to_markdown', 'audio_transcription', 'dialogue_formatting', 'markdown_input'}

@app.route('/api/conversions', methods=['POST'])
@login_required
def api_create_conversion():
    data = request.get_json()
    if not data or not data.get('content'):
        return jsonify({'error': 'Content is required'}), 400

    conversion_type = data.get('conversion_type', 'unknown')
    if conversion_type not in ALLOWED_CONVERSION_TYPES:
        return jsonify({'error': f'Invalid conversion type: {conversion_type}'}), 400

    title = data.get('title', 'Untitled')[:255]

    conversion = Conversion(
        user_id=current_user.id,
        conversion_type=conversion_type,
        title=title,
        content=data['content'],
        source_filename=data.get('source_filename'),
        source_mimetype=data.get('source_mimetype'),
        source_size_bytes=data.get('source_size_bytes'),
        metadata_json=json.dumps(data.get('metadata', {})),
        tags=data.get('tags', ''),
    )
    db.session.add(conversion)
    db.session.commit()
    return jsonify(conversion.to_dict()), 201


@app.route('/api/conversions/<int:conversion_id>', methods=['PUT'])
@login_required
def api_update_conversion(conversion_id):
    conversion = Conversion.query.filter_by(id=conversion_id, user_id=current_user.id).first_or_404()
    data = request.get_json()

    if 'title' in data:
        conversion.title = str(data['title'])[:255]
    if 'tags' in data:
        conversion.tags = str(data['tags'])[:500]
    if 'content' in data:
        conversion.content = data['content']
    if 'is_favorite' in data:
        conversion.is_favorite = bool(data['is_favorite'])

    db.session.commit()
    return jsonify(conversion.to_dict())


@app.route('/api/conversions/<int:conversion_id>', methods=['DELETE'])
@login_required
def api_delete_conversion(conversion_id):
    conversion = Conversion.query.filter_by(id=conversion_id, user_id=current_user.id).first_or_404()
    db.session.delete(conversion)
    db.session.commit()
    return jsonify({'success': True})


# --- Notion Integration ---
@app.route('/api/notion/suggestions')
@login_required
def api_notion_suggestions():
    def fetch():
        if not NOTION_TOKEN:
            return {'people': [], 'projects': [], 'meeting_types': [], 'note_types': []}
        db_ids = _get_notion_db_ids()
        people = _query_db_titles(db_ids['PEOPLE']) if 'PEOPLE' in db_ids else []
        projects = _query_db_titles(db_ids['PROJECT']) if 'PROJECT' in db_ids else []
        meeting_types = _get_select_options(db_ids['MEETINGS'], 'Type') if 'MEETINGS' in db_ids else []
        note_types = _get_select_options(db_ids['NOTES'], 'Type') if 'NOTES' in db_ids else []
        return {'people': people, 'projects': projects, 'meeting_types': meeting_types, 'note_types': note_types}
    try:
        return jsonify(_cached('suggestions', 300, fetch))
    except Exception as e:
        import logging as _logging
        _logging.getLogger(__name__).warning(f'Notion suggestions failed: {e}')
        return jsonify({'people': [], 'projects': [], 'meeting_types': [], 'note_types': []})


@app.route('/api/conversions/<int:conversion_id>/send-to-notion', methods=['POST'])
@login_required
def api_send_to_notion(conversion_id):
    Conversion.query.filter_by(id=conversion_id, user_id=current_user.id).first_or_404()
    data = request.get_json()
    target = data.get('target')
    if target not in ('meetings', 'notes', 'inbox'):
        return jsonify({'error': 'Invalid target'}), 400

    payload = {k: v for k, v in data.get('fields', {}).items() if v}
    try:
        resp = http_requests.post(
            f'{NOTION_MCP_URL}/api/{target}',
            json=payload,
            headers={'Authorization': f'Bearer {MCP_AUTH_TOKEN}',
                     'Content-Type': 'application/json'},
            timeout=30
        )
        result = resp.json()
        if resp.status_code >= 400:
            return jsonify({'error': result.get('error', result.get('detail', 'Notion API error'))}), resp.status_code
        return jsonify(result), resp.status_code
    except http_requests.RequestException as e:
        app.logger.error(f"Failed to reach Notion server: {e}")
        return jsonify({'error': 'Failed to reach Notion server.'}), 502


asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
