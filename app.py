import os
import json
from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job
from io import BytesIO
from flask import render_template, request, redirect, url_for, send_file, jsonify
from flask_login import login_required, current_user
from playwright.async_api import async_playwright
from unstructured.partition.auto import partition
from asgiref.wsgi import WsgiToAsgi
import re
import time as _time
import requests as http_requests

from app_pkg import audio as audio_module
from app_pkg import auth as auth_module
from app_pkg import create_app
from app_pkg import documents as documents_module
from app_pkg import markdown as markdown_module
from app_pkg import mermaid as mermaid_module
from services import DeepgramService, GeminiService, GoogleTTSService, PDFExtractionService
from tasks import generate_podcast_task
from models import db, User, Conversion


DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GOOGLE_CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
NOTION_MCP_URL = os.environ.get('NOTION_MCP_URL', 'http://localhost:3333')
MCP_AUTH_TOKEN = os.environ.get('MCP_AUTH_TOKEN', '')
NOTION_TOKEN = os.environ.get('NOTION_TOKEN', '')

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

auth_module.register(app)
mermaid_module.register(app)
markdown_module.register(app)
documents_module.register(app)
audio_module.register(app)


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
