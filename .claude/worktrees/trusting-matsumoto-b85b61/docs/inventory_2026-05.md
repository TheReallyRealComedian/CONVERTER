# CONVERTER — Inventory & Baseline (Stage 0)

Snapshot date: **2026-05-02**. Branch: `main`.

This document is the read-only baseline for the cleanup plan in `CLAUDE.md`. No code was changed. Numbers and call-sites reflect the working tree at HEAD `fb32f8c`.

---

## 1. Codebase Metrics

### 1.1 LOC per Python file (sorted, descending)

| LOC  | File |
|-----:|------|
| 1021 | `services/gemini_service.py` |
|  978 | `app.py` |
|  686 | `services/pdf_extraction/service.py` |
|  377 | `services/audio_chunker.py` |
|  233 | `services/pdf_extraction/detectors.py` |
|  231 | `debug_timeout.py` |
|  216 | `services/deepgram_service.py` |
|  163 | `services/pdf_extraction/multi_page.py` |
|  126 | `services/pdf_extraction/ensemble.py` |
|  114 | `services/pdf_extraction/utils.py` |
|   92 | `services/pdf_extraction/extractors.py` |
|   92 | `services/google_tts_service.py` |
|   77 | `test_full_flow.py` |
|   63 | `tasks.py` |
|   55 | `models.py` |
|   41 | `test_worker_libraries.py` |
|   21 | `test_redis_connection.py` |
|   18 | `worker.py` |
|    6 | `services/__init__.py` |
|    5 | `services/pdf_extraction_service.py` (shim) |
|    4 | `services/pdf_extraction/__init__.py` |
| **4619** | **total** |

### 1.2 Template & static asset sizes

| File | Size |
|------|-----:|
| `templates/audio_converter.html` | 40.7 KB |
| `templates/markdown_converter.html` | 23.4 KB |
| `templates/library_detail.html` | 16.6 KB |
| `templates/base.html` | 12.9 KB |
| `templates/library.html` | 9.4 KB |
| `templates/document_converter.html` | 8.6 KB |
| `templates/mermaid_converter.html` | 5.6 KB |
| `templates/login.html` | 3.3 KB |
| `static/css/style.css` | 29.1 KB |
| `static/css/pdf_styles/default.css` | 8.8 KB |
| `static/css/pdf_styles/newspaper-bodoni.css` | 4.8 KB |
| `static/css/pdf_styles/academic-latex.css` | 3.4 KB |

### 1.3 File count per top-level dir

| Dir | Files (excl. `__pycache__`) |
|-----|---:|
| `services/` (incl. `pdf_extraction/`) | 13 |
| `templates/` | 8 |
| `static/` | 4 (1 main + 3 PDF themes) |
| `output_pdfs/` | 4 |
| `docs/` | 2 |
| `.claude/` | 1 |
| **Top-level files** | 21 |

Tracked files in git: **59**. Of those, 23 live under `services/`.

### 1.4 Dependency count

`requirements.txt`: **26 packages** (excluding the comment header).

Notable bundles:
- Web/runtime: Flask, Flask-SQLAlchemy, Flask-Login, Flask-WTF, gunicorn, uvicorn, gevent, asgiref, nh3
- Markdown→PDF: markdown-it-py, Pygments, playwright
- Doc→Markdown: unstructured[all-docs], pdfminer.six, PyMuPDF, pdfplumber, camelot-py, img2table, opencv-python-headless
- Audio: deepgram-sdk, pydub
- Podcast: google-cloud-texttospeech, google-genai, mermaid
- Background jobs: redis, rq
- Misc: requests

---

## 2. Route Map (`app.py`)

`app.py` defines a single Flask app (no blueprints). 24 endpoints + 2 error handlers + 1 CLI command + 1 `user_loader`.

`Auth` column: ✓ = `@login_required`, ✗ = public, **CSRF**: implicit on all `POST/PUT/DELETE` via `Flask-WTF` `CSRFProtect(app)`.

| Line | Route | Methods | View func | Template / Response | Service / module calls |
|---:|---|---|---|---|---|
| 167 | `/api/csrf-token` | GET | `get_csrf_token` ✓ | JSON | `flask_wtf.generate_csrf` |
| 213 | `/login` | GET, POST | `login` ✗ | `login.html` / redirect | `models.User.check_password`, `flask_login.login_user` |
| 234 | `/logout` | GET | `logout` ✓ | redirect → `/login` | `flask_login.logout_user` |
| 315 | `/` | GET | `markdown_converter` ✓ | `markdown_converter.html` | reads `STYLE_DIR` (`/app/static/css/pdf_styles`) |
| 324 | `/convert-markdown` | POST | `convert_markdown` ✓ (async) | PDF download / `markdown_converter.html` on error | `markdown_it.MarkdownIt`, `nh3.clean`, `_wrap_wide_tables`, `playwright.async_api` |
| 451 | `/mermaid-converter` | GET | `mermaid_converter` ✓ | `mermaid_converter.html` | — |
| 457 | `/document-converter` | GET | `document_converter` ✓ | `document_converter.html` | — |
| 462 | `/transform-document` | POST | `transform_document` ✓ | Markdown download | `PDFExtractionService.extract_markdown` (PDF), `unstructured.partition` (other) |
| 516 | `/audio-converter` | GET | `audio_converter` ✓ | `audio_converter.html` | — |
| 521 | `/api/get-deepgram-token` | GET | `get_deepgram_token` ✓ | JSON | `DeepgramService.create_temporary_key` |
| 535 | `/transcribe-audio-file` | POST | `transcribe_audio_file` ✓ | JSON | `DeepgramService.transcribe_file` |
| 580 | `/generate-podcast` | POST | `generate_podcast` ✓ | MP3 download | `GoogleTTSService.synthesize_speech` |
| 622 | `/api/get-google-voices` | GET | `get_google_voices` ✓ | JSON | `GoogleTTSService.list_voices` |
| 636 | `/generate-gemini-podcast` | POST | `generate_gemini_podcast` ✓ | JSON `{job_id}` | `task_queue.enqueue(generate_podcast_task, …)` (RQ) |
| 665 | `/podcast-status/<job_id>` | GET | `podcast_status` ✓ | JSON status | `rq.job.Job.fetch` |
| 690 | `/podcast-download/<job_id>` | GET | `podcast_download` ✓ | WAV download | `rq.job.Job.fetch`, reads from `OUTPUT_DIR=/app/output_podcasts` |
| 735 | `/api/get-gemini-voices` | GET | `get_gemini_voices` ✓ | JSON (hard-coded list) | — |
| 781 | `/format-dialogue-with-llm` | POST | `format_dialogue_with_llm` ✓ | JSON | `GeminiService.format_dialogue_with_llm` |
| 817 | `/library` | GET | `library` ✓ | `library.html` | `models.Conversion` (paginated query) |
| 861 | `/library/<int:id>` | GET | `library_detail` ✓ | `library_detail.html` | `models.Conversion.first_or_404` |
| 872 | `/api/conversions` | POST | `api_create_conversion` ✓ | JSON 201 | `models.Conversion`, `db.session` |
| 901 | `/api/conversions/<id>` | PUT | `api_update_conversion` ✓ | JSON | `models.Conversion`, `db.session` |
| 920 | `/api/conversions/<id>` | DELETE | `api_delete_conversion` ✓ | JSON | `models.Conversion`, `db.session.delete` |
| 930 | `/api/notion/suggestions` | GET | `api_notion_suggestions` ✓ | JSON | `_notion_api` (Notion REST) — `_get_notion_db_ids`, `_query_db_titles`, `_get_select_options` |
| 949 | `/api/conversions/<id>/send-to-notion` | POST | `api_send_to_notion` ✓ | JSON proxy | `requests.post` to `NOTION_MCP_URL` |
| 140 | `errorhandler(413)` | — | `request_entity_too_large` | JSON 413 | — |
| 147 | `errorhandler(CSRFError)` | — | `handle_csrf_error` | JSON or HTML 400 | — |
| 132 | `user_loader` | — | `load_user` | — | `db.session.get(User, id)` |
| 194 | CLI: `flask create-user` | — | `create_user_cmd` | console | `User.set_password`, `db.session.commit` |

Module-level Flask state initialised at import time (lines 107–190): app, CSRF, `MAX_CONTENT_LENGTH=500 MB`, SQLAlchemy + Login init, four service singletons (`deepgram_service`, `gemini_service`, `google_tts_service`, `pdf_extraction_service`), Redis connection, RQ `task_queue`, `db.create_all()`. ASGI wrapper at line 976.

---

## 3. Service Inventory (`services/`)

Module-level `services/__init__.py` re-exports: `DeepgramService`, `GeminiService`, `GoogleTTSService`, `PDFExtractionService`.

### 3.1 `services/deepgram_service.py` — 216 LOC
| Public symbol | LOC | Called from |
|---|---:|---|
| `class DeepgramService` | (whole file) | `app.py:174` (singleton), `app.py:524-525,529,538,557` |
| `__init__(api_key)` | 15 | `app.py:174` |
| `load_keyterms(language='en')` | 26 | internal (`_transcribe_single`, `_transcribe_chunk_with_retry`) |
| `transcribe_file(audio_data, language)` | 70 | `app.py:557` |
| `_transcribe_single(...)` | 31 | internal |
| `_transcribe_chunk_with_retry(...)` | 43 | internal |
| `create_temporary_key(ttl_seconds=60)` | 6 | `app.py:529` |

Imports `AudioChunker`, `TranscriptMerger`, `AudioChunk` from `.audio_chunker`. Reads `/app/keyterms.json` at runtime.

### 3.2 `services/audio_chunker.py` — 377 LOC
| Public symbol | LOC | Called from |
|---|---:|---|
| `dataclass AudioChunk` | 9 | `deepgram_service.py:11` (re-exported) |
| `class AudioChunker` | ~170 | `deepgram_service.py:32` (instance) |
| `__init__(...)` | 12 | `deepgram_service.py:32` |
| `_get_audio_metadata(file_path)` | 15 | internal |
| `needs_splitting(audio_data)` | 64 | `deepgram_service.py` (chunked path) |
| `split_audio(audio_data)` | 70 | `deepgram_service.py` |
| `_extract_chunk_ffmpeg(...)` | 22 | internal |
| `class TranscriptMerger` | ~155 | `deepgram_service.py:38` (instance) |
| `__init__(...)` | 10 | `deepgram_service.py:38` |
| `merge_transcripts(transcripts)` | 32 | `deepgram_service.py` |
| `_merge_two_transcripts(...)` | 54 | internal |
| `_find_best_overlap(...)` | 36 | internal |
| `_smart_concatenate(...)` | 8 | internal |
| `_clean_merged_text(text)` | 11 | internal |

Shells out to `ffmpeg`/`ffprobe` via `subprocess` (audio_chunker.py:11, 76, 144).

### 3.3 `services/gemini_service.py` — 1021 LOC (largest file in repo)
| Public symbol | LOC | Called from |
|---|---:|---|
| `class GeminiService` | (whole file) | `app.py:175` and `tasks.py:43` (rebuilt in worker) |
| `__init__(api_key)` | 19 | `app.py:175`, `tasks.py:43` |
| `_calculate_tag_guidance(raw_text, narration_style)` | 43 | internal |
| `format_dialogue_with_llm(raw_text, num_speakers, …)` | 194 | `app.py:795` |
| `_build_single_speaker_prompt_v2(...)` | 61 | internal |
| `_build_multi_speaker_prompt_v2(...)` | 86 | internal |
| `_parse_dialogue(formatted_text)` | 35 | internal |
| `generate_podcast(dialogue, language, tts_model)` | 84 | `tasks.py:47` |
| `_generate_single_chunk(...)` | 166 | internal |
| `_generate_with_chunking(...)` | 103 | internal |
| `_create_dialogue_chunks(dialogue_lines)` | 42 | internal |
| `_concatenate_with_pydub(audio_files)` | 28 | internal |
| `_concatenate_with_wave(audio_files)` | 36 | internal |
| `_split_long_dialogue_turns(...)` | 61 | internal |
| `_filter_metadata_lines(dialogue_lines)` | 41 | internal |

Class-level config constants (`MAX_LINES_PER_CHUNK=80`, `MAX_CHARS_PER_CHUNK=3000`, …) at lines 15–20.

### 3.4 `services/google_tts_service.py` — 92 LOC
| Public symbol | LOC | Called from |
|---|---:|---|
| `class GoogleTTSService` | (whole file) | `app.py:176` |
| `__init__(credentials_path)` | 6 | `app.py:176` |
| `list_voices()` | 28 | `app.py:629` |
| `synthesize_speech(text, voice_name, language_code, speaking_rate, pitch)` | 47 | `app.py:595` |

### 3.5 `services/pdf_extraction_service.py` — 5 LOC
Backward-compatibility shim: `from .pdf_extraction.service import PDFExtractionService`. Imported by `services/__init__.py:5`.

### 3.6 `services/pdf_extraction/service.py` — 686 LOC
Single class `PDFExtractionService` (line 25). Public surface:

| Public symbol | LOC | Called from |
|---|---:|---|
| `__init__(gemini_api_key=None)` | 14 | `app.py:177` |
| `extract_markdown(file_path)` | 65 | `app.py:487` |

15 private helpers (`_analyze_pages_tiered`, `_classify_page`, `_get_texts_before_first_table`, `_extract_page_with_ensemble`, `_extract_page_with_gemini_fallback`, `_extract_text_page`, `_extract_scanned_page`, `_add_non_table_text`, `_build_gemini_prompt`, `_extract_with_gemini_vision`, `_validate_gemini_output`, `_content_overlap_score`, `_extract_with_pymupdf_tables`, `_apply_multipage_merges`, `_extract_links`, `_postprocess_markdown`, `_embed_links`).

### 3.7 `services/pdf_extraction/detectors.py` — 233 LOC
Five detector classes: `PyMuPDFDetector`, `PdfplumberDetector`, `CamelotDetector`, `Img2TableDetector`, `TextHeuristicDetector`. Plus `dataclass DetectedTable`. All consumed by `service.py`.

### 3.8 `services/pdf_extraction/ensemble.py` — 126 LOC
- `dataclass ConsensusTable`
- `cluster_detections(...)`, `build_consensus(...)`, `score_extraction(...)`
Consumers: `service.py`, `extractors.py`, `multi_page.py`.

### 3.9 `services/pdf_extraction/extractors.py` — 92 LOC
Free functions: `extract_pymupdf`, `extract_pdfplumber`, `extract_camelot`, `extract_img2table`, `select_best_extraction`. Consumers: `service.py`.

### 3.10 `services/pdf_extraction/multi_page.py` — 163 LOC
- `dataclass TableSpan`
- `detect_continuation_signals`, `is_continuation`, `find_table_spans`, `merge_table_rows`, `_get_column_positions`
Consumers: `service.py`.

### 3.11 `services/pdf_extraction/utils.py` — 114 LOC
- Type alias `BBox`
- `bbox_iou`, `bbox_overlap_ratio`, `parse_markdown_tables`, `table_to_markdown`, `columns_match`, `rows_similar`
Consumers: `detectors.py`, `ensemble.py`, `multi_page.py`, `service.py`.

---

## 4. Anomaly Grep

Grep was scoped to `app.py`, `tasks.py`, `worker.py`, `models.py`, `services/`, `templates/`, `static/`. `.codebuddy/` and `__pycache__/` excluded.

### 4.1 TODO / FIXME / XXX / HACK markers
None found in the application source tree. (Clean.)

### 4.2 Commented-out code blocks (5+ consecutive `#` lines)
None detected by line-based scan in the active modules.

### 4.3 `print(` statements
- `worker.py:15` — `print("Worker gestartet und wartet auf Jobs...")` (startup banner; intentional, but not via `logger`)
- `test_full_flow.py`, `test_redis_connection.py`, `test_worker_libraries.py`, `debug_timeout.py` — heavy `print` usage. These are root-level scripts intended for manual diagnostics; whether they live in `tests/` or get deleted is decided in Stage 1/6.

No `print(` calls in `app.py`, `tasks.py`, `services/*`, or `models.py`.

### 4.4 Hardcoded paths
Container-internal `/app/*` paths (intentional, match Docker volume layout):
- `app.py:49` — `STYLE_DIR = Path('/app/static/css/pdf_styles')`
- `app.py:185`, `tasks.py:13` — `OUTPUT_DIR = '/app/output_podcasts'` (duplicated string constant; both live in modules that share a Docker volume)
- `app.py:189` — `os.makedirs('/app/data', exist_ok=True)` for the SQLite DB
- `services/deepgram_service.py:46` — `Path('/app/keyterms.json')`

No `/tmp`, `/Users/`, `/home/`, or `/var/` literals in application code (`tempfile` is used instead).

### 4.5 Exception handling
Five broad-except sites in app code:

| File:line | Pattern | Notes |
|---|---|---|
| `app.py:244` | `except Exception:` (in `highlight_code`) | Falls back to plain-text lexer; intentional but silent |
| `app.py:671` | `except Exception:` (`/podcast-status/<job_id>`) | Returns 404 if RQ `Job.fetch` throws — swallows reason without logging |
| `app.py:696` | `except Exception:` (`/podcast-download/<job_id>`) | Same pattern as above |
| `services/pdf_extraction/service.py:217` | `except Exception:` | Inside page-classification loop |
| `services/pdf_extraction/detectors.py:134` | `except Exception:` | Inside Img2Table detector retry path |

No bare `except:` clauses. No `except Exception: pass` swallowed-error chains.

### 4.6 Unused-looking imports in `app.py`
`asyncio` (line 2), `traceback` (line 29), `fitz` (line 27) — each is the only occurrence of its identifier in the file. Likely leftover from earlier iterations.

### 4.7 Inline `<script>` blocks in templates
- `markdown_converter.html`: 6 inline `<script>` blocks (lines 165, 180, 265, 317, 456, …)
- `audio_converter.html`: inline `<script>` from line 267 to EOF
- `library.html`, `library_detail.html`, `document_converter.html`, `mermaid_converter.html`, `login.html`, `base.html`: at least one inline `<script>` each
Stage 5 explicitly targets these.

---

## 5. Import Graph

```
worker.py
  └─ rq.Worker, redis        (entry point: `python worker.py`)

tasks.py
  └─ services.GeminiService  (instantiated fresh inside generate_podcast_task)

app.py
  ├─ models           (db, User, Conversion)
  ├─ tasks            (generate_podcast_task — enqueued, never called directly)
  └─ services         (DeepgramService, GeminiService, GoogleTTSService, PDFExtractionService)
                          │
                          ▼
                    services/__init__.py
                          ├─ .deepgram_service.DeepgramService
                          │     └─ .audio_chunker.{AudioChunker, TranscriptMerger, AudioChunk}
                          ├─ .gemini_service.GeminiService           (no internal services deps)
                          ├─ .google_tts_service.GoogleTTSService    (no internal services deps)
                          └─ .pdf_extraction_service.PDFExtractionService   (shim)
                                └─ .pdf_extraction.service.PDFExtractionService
                                      ├─ .detectors  (5 detectors + DetectedTable)
                                      ├─ .ensemble   (build_consensus, ConsensusTable)
                                      ├─ .extractors (select_best_extraction)
                                      ├─ .multi_page (find_table_spans, merge_table_rows)
                                      └─ .utils      (table_to_markdown, parse_markdown_tables, BBox, …)

models.py: standalone (Flask-SQLAlchemy + werkzeug)
```

Observations:
- The web container imports `tasks` only to reference `generate_podcast_task` for enqueueing; the function actually runs inside the `worker` container.
- `worker.py` does **not** import `tasks` — RQ deserialises the task from Redis using its dotted path, so the worker image must have `tasks.py` and `services/` available on `PYTHONPATH`.
- `services/pdf_extraction_service.py` is a thin re-export shim for the package layout move.
- No circular imports.

---

## 6. Top-Level Artefacts with Unclear Purpose

These are tracked in git (verified via `git ls-files`) but not referenced by `app.py`/`tasks.py`/`worker.py`/`Dockerfile`/`docker-compose.yml`. Hypotheses only — Stage 1 decides.

| Artefact | Tracked? | Hypothesis |
|---|---|---|
| `Dockerfile default` | yes | Older Dockerfile from before the Playwright base image was adopted (Python 3.10-slim + Playwright self-install). Active build uses `Dockerfile`. |
| `debug_timeout.py` (231 LOC) | yes | One-off diagnostic for nginx `proxy_read_timeout` 60→300s fix. Standalone (no imports from app modules). |
| `test_full_flow.py` (77 LOC) | yes | Manual end-to-end smoke test that hits a live `BASE_URL` for podcast generation. Not pytest. |
| `test_redis_connection.py` (21 LOC) | yes | Manual connectivity probe for the Redis container. |
| `test_worker_libraries.py` (41 LOC) | yes | Verifies `ffmpeg` + `pydub` + write access to the podcast volume. |
| `output_pdfs/*.pdf` (4 files, ~388 KB) | yes | Old test outputs from June 2025 (`dddd.pdf`, `stuff.pdf`, `test2.pdf`, `test_academic.pdf`). |
| `keyterms.json` | yes | **In active use**. Loaded by `services/deepgram_service.py:46` to feed Nova-3 transcription with German/English domain terms (CAR-NK therapy, mRNA, GMP …). Not unused — but worth documenting. |
| `.codebuddy/` (1065+ JSON files in `db/vectra/`) | only `.gitignore` inside it | Local vector index from a third-party tool ("CodeBuddy"). Bulk content is untracked, but the parent dir leaks into the repo. |
| `google-credentials.json` | gitignored | Real GCP service account credentials present locally. `.gitignore` line 2 protects it; verify before any rewrite of history. |
| `.DS_Store` (root + others) | gitignored | macOS metadata. |

---

## 7. Erste Eindrücke (observations only — not recommendations)

- `app.py` and `services/gemini_service.py` together are 2 000 LOC of the project's 4 600 — and Stage 2/3 already targets them.
- `services/__init__.py` re-exports through a shim (`pdf_extraction_service.py`) instead of pointing directly at the package; the shim adds an extra hop with no callers using it.
- `OUTPUT_DIR='/app/output_podcasts'` is duplicated as a literal string in both `app.py:185` and `tasks.py:13`; Docker-volume layout assumes they stay in lockstep.
- The Notion "suggestions" feature (`_notion_api`, `_get_notion_db_ids`, …, `/api/notion/suggestions`) lives at the top of `app.py` ahead of any web-app concern — strange placement for a feature that isn't markdown/audio/podcast.
- `app.py` imports `asyncio`, `traceback`, and `fitz` but uses none of them (greppable: each name appears exactly once in the file).
- Three broad `except Exception:` clauses in `app.py` swallow errors silently when fetching RQ jobs (lines 244, 671, 696). Whether this is intentional 404-masking is up for Stage 4.
- No `TODO`/`FIXME`/`XXX`/`HACK` markers anywhere in the source — surprisingly clean for a "hotfix-accumulating" codebase.
- `worker.py` is 18 LOC, but it does not import `tasks`. RQ relies on the worker image having `tasks.py` and `services/` available on `PYTHONPATH` — coupling that isn't visible from the source.
- `keyterms.json` is the only "unclear top-level artefact" candidate from the cleanup list that is **actively used** (Deepgram domain-term boost). Stage 1 should not delete it.
- `audio_converter.html` (40.7 KB) and `markdown_converter.html` (23.4 KB) carry the bulk of the embedded JavaScript; both will need careful Stage 5 extraction because they each hold multi-block scripts that share variables across `<script>` blocks (visible in template line numbers).
- The codebase has zero `print()` debug leftovers in production paths — only the worker startup banner and the standalone diagnostic scripts at the repo root. Stage 1 cleanup is mostly about *artefacts*, not *code rot*.
