"""Top-level entry point for the CONVERTER Flask app.

Most of the routing and feature logic lives under ``app_pkg/``; this module
is the bootstrap that builds the Flask instance via ``create_app()``,
constructs the service singletons and Redis/RQ plumbing, registers the
per-feature blueprints, and exposes ``app`` and ``asgi_app`` for Gunicorn /
Uvicorn.

Several names are kept at module level on purpose because the Stage 6
characterization tests patch them by attribute on this module:
``deepgram_service``, ``gemini_service``, ``google_tts_service``,
``pdf_extraction_service``, ``task_queue``, ``Job``, ``async_playwright``,
``partition``, ``GEMINI_API_KEY``, ``DEEPGRAM_API_KEY``, ``redis_conn``.
The blueprints look these up via ``import app as _app_module`` so the
patches reach the route handlers at call time.
"""
import os

from asgiref.wsgi import WsgiToAsgi
from playwright.async_api import async_playwright
from redis import Redis
from rq import Queue
from rq.job import Job
from unstructured.partition.auto import partition

from app_pkg import audio as audio_module
from app_pkg import auth as auth_module
from app_pkg import create_app
from app_pkg import documents as documents_module
from app_pkg import library as library_module
from app_pkg import markdown as markdown_module
from app_pkg import mermaid as mermaid_module
from app_pkg import podcasts as podcasts_module
from app_pkg.integrations import notion as notion_module
from services import DeepgramService, GeminiService, GoogleTTSService, PDFExtractionService


DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GOOGLE_CREDENTIALS_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

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

auth_module.register(app)
mermaid_module.register(app)
markdown_module.register(app)
documents_module.register(app)
audio_module.register(app)
library_module.register(app)
notion_module.register(app)
podcasts_module.register(app)


asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
