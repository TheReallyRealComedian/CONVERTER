# CONVERTER - Multimedia Converter & Podcast Generator

## What is this?
A Flask web app for multimedia conversion: Markdown-to-PDF, document-to-Markdown, audio transcription (Deepgram), and AI podcast generation (Google Gemini TTS). Runs in Docker with Redis/RQ for background jobs.

## Tech Stack
- **Backend**: Flask (async), SQLAlchemy (SQLite), Flask-Login
- **Job Queue**: Redis + RQ (worker container for podcast generation)
- **APIs**: Gemini (script generation + TTS), Deepgram (transcription), Google Cloud TTS
- **Frontend**: Bootstrap + vanilla JS (Jinja2 templates)

## Key Files
- `app.py` — Main Flask app, all routes
- `services/gemini_service.py` — Gemini LLM script generation + TTS podcast synthesis
- `tasks.py` — RQ background tasks (podcast generation)
- `worker.py` — RQ worker process
- `models.py` — SQLAlchemy models (User, ConversionHistory)

## Running
```bash
docker compose up --build
```
App runs on `localhost:5656`. Requires `.env` with `GEMINI_API_KEY`, `DEEPGRAM_API_KEY`, `SECRET_KEY`, and `google-credentials.json` for Google Cloud TTS.

## Gemini Models Used
- **Script generation**: `gemini-2.5-flash` (in `gemini_service.py`)
- **TTS**: `gemini-2.5-flash-preview-tts` / `gemini-2.5-pro-preview-tts` (in `gemini_service.py`)

## Architecture Notes
- Podcast generation is async: web enqueues job via Redis, worker processes it, result shared via `podcast_data` Docker volume
- Long podcasts are chunked (max 80 lines / 3000 chars per chunk) and concatenated with pydub
- Frontend polls `/podcast-status/<job_id>` until complete
