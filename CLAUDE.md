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
**Status:** ☑ done 2026-05-02 → branch `cleanup/stage-4-anomalies` (3 commits). 18 findings logged below. Critical: 1 (F-001 silent `Job.fetch()` except → masked Redis/auth/pickle errors as 404; fixed in `app.py:670-678` and `697-705`, now distinguishes `NoSuchJobError` → 404 vs other → 500 with `exc_info`). Medium: 5, Low: 12 — all deferred to Stages 2/3/5 where they fold into the natural refactor.
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
**Status:** ☑ done 2026-05-02 → branch `cleanup/stage-6-tests` (7 commits, on top of `cleanup/stage-4-anomalies`). 36 tests in 6 files, runtime 4.4s. Mocking at SDK-singleton boundary (`app.deepgram_service`, `app.gemini_service`, `app.task_queue`, `app.async_playwright`) → survives Stage 3. F-001 explicitly characterized. Run via `pytest tests/`. Test deps added: `pytest>=8.0`, `pytest-asyncio>=0.23`, `responses>=0.25`.
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

**Sweep date:** 2026-05-02. Branch: `cleanup/stage-4-anomalies`. Scope: `*.py` and `templates/*.html` (no `tests/` yet, no `static/`, no `__pycache__`/`.codebuddy`). 18 findings logged. **One Critical fix applied** in this stage — see commit log; all other findings deferred to the named stage. The sweep stayed under the 30-finding cap; no second pass needed.

### F-001: Silent `except Exception` masks Redis errors as 404 in RQ-fetch routes
- **Location:** `app.py:671` (`/podcast-status/<job_id>`), `app.py:696` (`/podcast-download/<job_id>`)
- **Severity:** Critical
- **Pattern:** Exception-handling anomaly
- **Observation:** Both routes wrap `Job.fetch(job_id, connection=redis_conn)` in a bare `except Exception:` and return `{"error": "Job not found"}, 404`. RQ raises `NoSuchJobError` for a missing job, but anything else (Redis connection refused, auth failure, payload-deserialisation error) is also reported to the user as "not found" — and nothing is logged, so the operator has no signal either. Stage 0 already flagged this lead; verified by reading `rq.exceptions`.
- **Action:** Fix-now (Stage 4) — narrow to `NoSuchJobError`, log other exceptions at `warning`, return 500 for transport failures.

### F-002: Silent `except Exception` in `highlight_code` (intentional fallback)
- **Location:** `app.py:244`
- **Severity:** Low
- **Pattern:** Exception-handling anomaly
- **Observation:** Wraps `get_lexer_by_name(lang, …)` to fall back to the `text` lexer when Pygments raises `ClassNotFound` for an unknown code-fence language. The intent is clear from context; no logging is reasonable here.
- **Action:** Defer-to-Stage-2 — narrow to `pygments.util.ClassNotFound` when the markdown route is split into its own blueprint.

### F-003: Three unused module-level imports in `app.py`
- **Location:** `app.py:2` (`asyncio`), `app.py:27` (`fitz`), `app.py:29` (`traceback`)
- **Severity:** Low
- **Pattern:** Naming/dead-code drift
- **Observation:** Each identifier appears exactly once in the file (the import line). `asyncio` is shadowed by `async_playwright`'s implicit loop; `fitz` was likely left from before `pdf_extraction_service` absorbed PyMuPDF; `traceback` is no longer needed because all error sites use `exc_info=True` instead of formatted strings.
- **Action:** Defer-to-Stage-2 — imports get redistributed during the blueprint split anyway.
- **Status:** ☑ resolved in Stage 2 (commit `8d397d6`, the markdown-extraction step) — all three imports dropped from the bootstrap `app.py` along with the rest of the unused stdlib/third-party imports left over after the route moves.

### F-004: `OUTPUT_DIR='/app/output_podcasts'` duplicated as a string literal across three files
- **Location:** `app.py:185`, `tasks.py:13`, `test_worker_libraries.py:26`
- **Severity:** Medium
- **Pattern:** Hardcoded constants
- **Observation:** The web container, the worker container and a manual smoke-test script each carry the same path literal. The Docker volume layout assumes they stay in lockstep; rename in one place and the other two silently break (web returns 404 for completed jobs, smoke-test writes to the wrong place).
- **Action:** Defer-to-Stage-2 — extract to a shared `config.py` (or env var with a single default) when the app factory lands. Stage 0 already flagged this.
- **Status:** ☑ resolved in Stage 2 (commit `e5e8745`) — `app_pkg/config.py` now owns the constant; `app_pkg/podcasts.py` and `tasks.py` both import it. The standalone diagnostic script `test_worker_libraries.py` keeps its inline literal (out-of-scope; not on the production import path).

### F-005: Path-traversal check in `podcast_download` uses `startswith` (prefix-collision-prone)
- **Location:** `app.py:710-714`
- **Severity:** Medium
- **Pattern:** Sanitization / boundary check
- **Observation:** `real_path.startswith(os.path.realpath(OUTPUT_DIR))` accepts `/app/output_podcasts2/foo.wav` if such a sibling directory ever appears. `job.result` is currently set by the worker and is not user-controlled, so this isn't actively exploitable, but the pattern is fragile defence-in-depth. `Path.is_relative_to()` (Py 3.9+) or comparing `os.path.commonpath([real, root]) == root` is the standard.
- **Action:** Defer-to-Stage-2 — fix in the podcast blueprint extraction.

### F-006: File-upload endpoints have inconsistent and incomplete validation
- **Location:** `app.py:328-331` (`convert_markdown` → markdown_file), `app.py:465-473` (`transform_document` → document_file), `app.py:541-548` (`transcribe_audio_file` → audio_file)
- **Severity:** Medium
- **Pattern:** File-upload validation
- **Observation:** Three different defensive shapes for the same problem. `convert_markdown` does `request.files.get(...)` then truthy-checks both file and filename, then blindly `.read().decode('utf-8')` (binary upload → `UnicodeDecodeError` → caught by the broad `except Exception` at 439 and reported as "Could not generate PDF"). `transform_document` checks `'document_file' not in request.files`, then `filename == ''`, then a redundant `if not file:` (FileStorage is always truthy). `transcribe_audio_file` checks the part name and empty filename only. None of the three validate extension or content-type; all three rely on Flask's `MAX_CONTENT_LENGTH=500 MB` for size. App is login-protected and LAN-only, so this is hardening, not an open hole.
- **Action:** Defer-to-Stage-2 — extract a shared `_validate_upload(field_name, allowed_exts)` helper when blueprints land.

### F-007: `secure_filename(None)` will raise `AttributeError` if `output_filename` form field is missing
- **Location:** `app.py:336-340`
- **Severity:** Low
- **Pattern:** Null/empty checks
- **Observation:** `request.form.get('output_filename')` returns `None` when the field is absent; `secure_filename(None)` raises `AttributeError`, which the surrounding broad `except` would later mask as "PDF generation failed". The downstream `if not safe_filename:` check is therefore dead in the None case.
- **Action:** Defer-to-Stage-2 — add explicit None check before `secure_filename`.

### F-008: `app.logger.error(..., exc_info=True)` used inconsistently across symmetric error paths
- **Location:** `app.py:440` (no exc_info) vs `app.py:509, 570, 576, 612, 632, 661, 810` (with exc_info); `app.py:532, 619, 725, 972` (no exc_info)
- **Severity:** Low
- **Pattern:** Logging consistency
- **Observation:** Of 11 `app.logger.error` sites, 6 include `exc_info=True` and 5 do not. The split is not principled: PDF generation (440) skips it but unstructured doc partition (509) — which is the sister conversion path — includes it. "Failed to reach Notion server" (972) and "Failed to create temporary Deepgram key" (532) are both upstream-API failures but the former drops the stacktrace.
- **Action:** Defer-to-Stage-2 — standardise to always-on `exc_info=True` in error paths during the blueprint split.
- **Status:** Partial — the document-routes scope of Step 4c was a no-op: the only error path in `app_pkg/documents.py` (the unstructured-partition catch) already used `exc_info=True` before the move (originally `app.py:510`) and continues to do so after. The four call-sites that still drop the stacktrace (markdown PDF generation, Deepgram-key issuance, podcast TTS-temp cleanup, Notion-MCP transport failure) live in their respective blueprints now and are easy to fix in one pass; left for a future logging-pass commit since each is a behaviour change (different log output) and the prompt narrowed Step 4c to documents only.

### F-009: Local `import re` inside two methods of `gemini_service.py` despite the module being usable at top-level
- **Location:** `services/gemini_service.py:924`, `services/gemini_service.py:982`
- **Severity:** Low
- **Pattern:** Naming/import drift
- **Observation:** `_split_long_dialogue_turns` and `_filter_metadata_lines` each `import re` inside the method body. Re-importing per call doesn't materially cost anything (Python caches), but it signals copy-paste. Module already imports `time`, `wave`, `tempfile` at the top.
- **Action:** Defer-to-Stage-3 — clean up during the gemini decomposition.

### F-010: `Conversion.query.filter_by(id=…, user_id=current_user.id).first_or_404()` repeated four times
- **Location:** `app.py:864, 904, 923, 952`
- **Severity:** Medium
- **Pattern:** Copy-paste
- **Observation:** Identical lookup with identical user-scoping is open-coded in `library_detail`, `api_update_conversion`, `api_delete_conversion`, and `api_send_to_notion`. A future "share with another user" or "soft-delete" feature would have to be added in four places — the kind of drift Stage 4 is meant to catch.
- **Action:** Defer-to-Stage-2 — add `Conversion.get_for_user(id, user_id)` classmethod when library/notion blueprints split out.
- **Status:** ☑ resolved in Stage 2 (commit `5ad218b`) — extracted as `get_owned_conversion(conversion_id)` in `app_pkg/library.py`; both library and notion blueprints call it. Implemented as a module helper rather than a `Conversion` classmethod (per Stage 2 prompt) since the helper relies on the `current_user` Flask proxy, which is request-scoped.

### F-011: `if not <service>: return jsonify({"error": "… not configured"}), 503` repeated across six endpoints
- **Location:** `app.py:524, 538, 583, 625, 640, 784`
- **Severity:** Low
- **Pattern:** Copy-paste
- **Observation:** Three different services (`deepgram_service`, `google_tts_service`, `gemini_service`/`GEMINI_API_KEY`) each gate two-to-three routes with the same five-line check. Trivial to factor as a small decorator (`@require_service('deepgram')`).
- **Action:** Defer-to-Stage-2.

### F-012: `if not file:` after `file = request.files['document_file']` is dead code
- **Location:** `app.py:472-473`
- **Severity:** Low
- **Pattern:** Null/empty checks
- **Observation:** Indexing `request.files[name]` either raises `KeyError` or returns a `FileStorage`, which is always truthy. The check above (`if 'document_file' not in request.files`) already covers the missing case. Looks like a defensive paste from a different framework.
- **Action:** Defer-to-Stage-2 — drop the dead branch when `transform_document` moves to its blueprint.

### F-013: User-supplied config strings flow into upstream API calls without an allowlist
- **Location:** `app.py:545` (Deepgram language), `app.py:590-593` (Google TTS voice/language/rate/pitch), `app.py:797-802` (Gemini narration_style/script_length/num_speakers/custom_prompt)
- **Severity:** Low
- **Pattern:** Null/empty checks (input validation)
- **Observation:** Routes take user-controlled strings/floats and pass them straight to the SDK. No security risk (login-gated, downstream APIs reject bad input), but bad input becomes a 500 instead of a clean 400. `narration_style`, `script_length`, `language` all have well-known allowlists already enumerated inside the service classes — exposing those constants and validating in the route would be more honest.
- **Action:** Defer-to-Stage-2.

### F-014: `services/pdf_extraction/service.py:217` and `services/pdf_extraction/detectors.py:134` swallow exceptions silently with `pass`
- **Location:** `services/pdf_extraction/service.py:217`, `services/pdf_extraction/detectors.py:134`
- **Severity:** Low
- **Pattern:** Exception-handling anomaly
- **Observation:** Both are intentional fallbacks: per-image rect lookup that contributes to coverage heuristics, and an optional Tesseract-OCR import that's absent in many environments. Neither logs, but both are inside heuristic paths where logging on every PDF page would be noisy.
- **Action:** Wontfix — intentional silent fallback in a hot loop. Revisit only if PDF extraction quality regresses.

### F-015: Three different magic timeouts for "API may be slow": 300s (Gemini), 600s (Deepgram), 600s (RQ job)
- **Location:** `services/gemini_service.py:30`, `services/deepgram_service.py:155`, `app.py:653`
- **Severity:** Low
- **Pattern:** Hardcoded constants
- **Observation:** The three numbers were picked at different times and have no documented rationale relative to each other. RQ job timeout (600s) is the outermost wall; if Gemini chunked TTS exceeds 300s but Deepgram allows 600s, only one of them really gates the user wait. Worth aligning when the gemini_service is decomposed.
- **Action:** Defer-to-Stage-3.

### F-016: `_filter_metadata_lines` logs the filtered count twice for the same call
- **Location:** `services/gemini_service.py:519` (caller-side log "Filtered out N metadata lines") and `services/gemini_service.py:1021` (callee-side log "Filtered N metadata lines")
- **Severity:** Low
- **Pattern:** Logging consistency
- **Observation:** Two log lines per invocation reporting the same number — visible noise, harmless. Caller-side wraps in an `if filtered_count > 0` so it's slightly less spammy.
- **Action:** Defer-to-Stage-3 — drop the callee-side line.

### F-017: `data = request.get_json()` results not type-checked before `.get(...)`
- **Location:** `app.py:588, 644, 788, 875, 905, 953`
- **Severity:** Low
- **Pattern:** Null/empty checks
- **Observation:** If a client sends a JSON list (`[]`) or scalar (`"foo"`) instead of an object, `data.get(...)` raises `AttributeError` and the request 500s. Not a real-world risk for a login-gated app whose own JS always sends objects, but inconsistent with `api_create_conversion`'s `if not data or not data.get('content'):` defensive pattern.
- **Action:** Defer-to-Stage-2 — either trust the JS clients (drop all checks) or apply `if not isinstance(data, dict)` uniformly.

### F-018: `templates/audio_converter.html` duplicates the `safeJSON` helper from `templates/document_converter.html`
- **Location:** `templates/audio_converter.html:270-280`, `templates/document_converter.html:71-80`
- **Severity:** Medium
- **Pattern:** Copy-paste
- **Observation:** The helper that gracefully turns redirect-to-login HTML responses into a typed error is reproduced verbatim in both templates. Any change to redirect handling has to be made twice.
- **Action:** Defer-to-Stage-5 — move into `static/js/_helpers.js` when inline scripts are extracted.

---

## How to launch a stage in a fresh thread

Open a new Claude Code session in this repo and paste:

> Read `CLAUDE.md` § Cleanup Plan. Execute **Stage N** only. Do not touch other stages. When done, post a short summary (what changed, what you skipped, what you found) and stop — I'll update Status from the overseer thread.
