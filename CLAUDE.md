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

---

## Cleanup Plan (started 2026-05-02)

The app works but has accumulated hotfixes, pasted-in docs, debug leftovers, and god-files. This is a multi-stage cleanup, executed in **separate threads**. Update the **Status** line of each stage as work progresses; this thread is the overseer.

**Method (per stage):** spawn a fresh thread with the stage prompt, do the work, post a short summary back here, mark Status. Each stage has a clear acceptance bar so threads can finish cleanly without scope creep.

**Guardrails (apply to every stage):**
- No behavior changes unless the stage explicitly says so. Refactor ≠ redesign.
- Before any non-trivial restructuring (Stage 2+), the characterization tests from Stage 6 must exist — or the stage is allowed only if it's purely additive/move-only.
- One stage = one PR/branch. Do not bundle stages.
- If a stage uncovers a real bug, file it under "Findings" below — don't silently fix it inside a refactor commit.

---

## Stage 0 — Inventory & Baseline
**Status:** ☑ done 2026-05-02 → see [docs/inventory_2026-05.md](docs/inventory_2026-05.md)
**Goal:** produce a written snapshot of the current codebase so later stages have a baseline to measure against. **No code changes.**
**Tasks:**
- LOC per file, file count per dir, dependency count
- List all routes in `app.py` with method + template + service deps
- List all entries in `services/` with public functions and where they're called
- Grep for obvious anomalies (TODO/FIXME/XXX, commented-out blocks, `print(` debug statements, hardcoded paths)
- Identify import graph between `app.py`, `services/*`, `tasks.py`
**Output:** a file `docs/inventory_2026-05.md` with the above.
**Acceptance:** the doc lets a fresh reader answer "where does X live and what depends on it" in <2 min.

## Stage 1 — Hygiene / Quick Wins
**Status:** ☑ done 2026-05-02 → branch `cleanup/stage-1-hygiene` (commit `70177f2`, +12/−302). Deleted: `Dockerfile default`, `debug_timeout.py`, `output_pdfs/*.pdf`, tracked `.codebuddy/.gitignore`. `.gitignore` tightened (`**/.DS_Store`, `__pycache__/`, `*.pyc`, `.codebuddy/`, `output_pdfs/*`). Root `test_*.py` left in place for Stage 6.
**Goal:** delete clearly dead artifacts, tighten `.gitignore`. Low-risk, no behavior change.
**Candidates (verify before deleting):**
- `Dockerfile default` — confirm unused, then delete
- `debug_timeout.py` (root, 9 KB) — verify no import refs, delete
- `output_pdfs/*.pdf` — old test outputs from June 2025, delete and add `output_pdfs/*` to `.gitignore`
- `.codebuddy/` — add to `.gitignore`, untrack if tracked
- `.DS_Store` — confirm gitignored everywhere
- ~~`keyterms.json`~~ — **KEEP**: actively used by `services/deepgram_service.py:46` for Deepgram domain terms (Stage 0 finding)
- Root-level `test_*.py` files (`test_full_flow.py`, `test_redis_connection.py`, `test_worker_libraries.py`) — decide: move to `tests/` or delete (handled fully in Stage 6)
**Acceptance:** repo `git status` is clean, `git ls-files` no longer includes the deleted artifacts, app still boots (`docker compose up`).

## Stage 2 — Decompose `app.py` (978 LOC) into Flask blueprints
**Status:** ☐ not started
**Prereq:** Stage 6 characterization tests must exist first.
**Goal:** split monolithic `app.py` into per-feature blueprints. Pure restructuring, no logic changes.
**Proposed split:**
- `app/__init__.py` — app factory, db, login, csrf, error handlers
- `app/auth.py` — login/logout/register
- `app/library.py` — library + library_detail routes
- `app/markdown.py` — markdown→PDF converter routes
- `app/documents.py` — doc→markdown routes
- `app/audio.py` — audio transcription routes
- `app/podcasts.py` — podcast routes (incl. `/podcast-status/<job_id>` polling)
- `app/mermaid.py` — mermaid converter
- `app/integrations/notion.py` — `_notion_api`, `/api/notion/suggestions`, `/api/conversions/<id>/send-to-notion` (own blueprint, not under library — Stage 0 surfaced this as a feature cluster)
**Acceptance:** `app.py` shrinks to <100 LOC (factory + run), every blueprint <250 LOC, all routes still respond, characterization tests still green.

## Stage 3 — Decompose `services/gemini_service.py` (1021 LOC)
**Status:** ☐ not started
**Prereq:** Stage 6 characterization tests for podcast generation must exist.
**Goal:** split the god-service into cohesive modules.
**Proposed split:**
- `services/gemini/client.py` — single Gemini client init / shared helpers
- `services/gemini/script.py` — dialogue script generation (text → script)
- `services/gemini/tts.py` — TTS synthesis (script → audio chunks)
- `services/gemini/prompts.py` — prompt templates as constants (no logic)
- `services/gemini/dialogue.py` — dialogue parsing/splitting/metadata filtering
**Acceptance:** no module >300 LOC, all callers updated, podcast smoke test passes end-to-end.

## Stage 4 — Inconsistency Sweep (anomaly detection)
**Status:** ☐ not started
**Goal:** apply the "anomaly = bug" heuristic. Find places that *almost* match a pattern but don't. **Document findings; fix only the clear bugs in this stage, defer larger ones.**
**What to grep for:**
- Logging: every `log.error/warn/info/debug` call — are levels consistent across similar paths?
- Null/empty checks: places where one branch checks `if x` and a sibling branch doesn't
- Exception handling: bare `except:`, `except Exception: pass`, swallowed errors
- Sanitization: `nh3.clean` / `escape` — used uniformly on user input that hits templates?
- Naming drift: `userId` vs `user_id`, `data`, `info`, `temp` as variable names
- CSRF: every state-changing route has the token check?
- File-upload validation: every upload endpoint checks size + extension + content-type?
- Copy-paste: identical 5+ line blocks across files (use `pylint --duplicate-code` or simple grep)
**Output:** a Findings list at the bottom of this CLAUDE.md (not a separate doc), each item: location + severity (Critical/High/Medium/Low) + recommended action.
**Acceptance:** Findings list populated; Critical items have follow-up tickets/stages.

## Stage 5 — Templates & CSS
**Status:** ☐ not started
**Goal:** large templates with embedded `<script>` blocks become unmaintainable. Extract.
**Tasks:**
- `audio_converter.html` (42 KB) and `markdown_converter.html` (24 KB): extract embedded JS into `static/js/<feature>.js`
- Identify shared partials (form headers, flash messages, file-upload widgets) → `templates/_partials/`
- `static/css/style.css` (30 KB): split by feature OR keep single file but document sections; do not invent new design system here
**Acceptance:** no template >15 KB, no inline `<script>` blocks >30 LOC, visual smoke test of each page passes.

## Stage 6 — Characterization Tests (foundation for Stages 2/3)
**Status:** ☐ not started
**Goal:** lock in current behavior of golden paths *before* restructuring. Tests document what the app does today; they are not "should" tests.
**Coverage targets:**
- Auth: login, logout, register, CSRF on POST
- Library: list view, detail view, history record creation on conversion
- Markdown→PDF: small markdown → PDF roundtrip
- Document→Markdown: small docx → markdown
- Audio: short mp3 → transcription (mocked Deepgram OK)
- Podcast: enqueue → status polling state machine (mocked Gemini OK)
**Tooling:** `pytest`, real SQLite test DB, mock external APIs at the HTTP boundary.
**Acceptance:** `pytest` green locally and inside the container; CI not required.

## Stage 7 — Dependency Audit
**Status:** ☐ not started
**Goal:** trim `requirements.txt`. 28 deps; some are likely transitive or unused after past refactors.
**Tasks:**
- `pip-deptree` to see who pulls what
- Grep imports: any top-level dep with zero direct imports? → candidate for removal
- Check version skew: `unstructured[all-docs]==0.14.5` (June 2024) vs `redis==5.0.1` etc. — anything with known CVE?
- `playwright==1.44.0` matches Dockerfile base image — keep in sync
**Acceptance:** `requirements.txt` only contains directly-imported packages, with a one-line comment per non-obvious dep.

---

## Findings (populated by Stage 4)

*(empty — Stage 4 will fill this)*

---

## How to launch a stage in a fresh thread

Open a new Claude Code session in this repo and paste:

> Read `CLAUDE.md` § Cleanup Plan. Execute **Stage N** only. Do not touch other stages. When done, post a short summary (what changed, what you skipped, what you found) and stop — I'll update Status from the overseer thread.
