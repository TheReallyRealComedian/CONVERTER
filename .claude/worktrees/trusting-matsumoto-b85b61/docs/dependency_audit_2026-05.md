# CONVERTER — Dependency Audit (Stage 7)

Snapshot date: **2026-05-03**. Branch: `cleanup/stage-7-deps` (off `cleanup/stage-3-gemini`).

This document is the read-only audit produced by Stage 7 of the cleanup plan in `CLAUDE.md`. It maps every entry in `requirements.txt` to its direct import (or documented reason to keep), enumerates known CVEs against the pinned versions, and lists the worst version-skew candidates for a future risk-managed upgrade pass.

**No requirements.txt change in this commit.** A second commit on this branch trims the file based on this audit.

---

## 1. Direct-Import Map

Methodology: `grep -rn "^import|^from" --include="*.py"` across the repo (excluding `__pycache__/`, `.codebuddy/`), normalized to top-level module names, then matched against the import-graph in `docs/inventory_2026-05.md` to identify which file consumes each package.

**Lazy imports** (inside `try/except` or function bodies) count as direct imports — they are still required for the optional code path to execute.

| Package | Pinned | Direct import? | Referenced in | Action | Rationale |
|---|---|---|---|---|---|
| `Flask[async]` | 3.0.3 | yes | `app.py`, `app_pkg/*.py`, `tests/conftest.py` | keep | Web framework. The `[async]` extra pulls `asgiref` (also pinned explicitly below). |
| `markdown-it-py` | 3.0.0 | yes | `app_pkg/markdown.py:11` | keep | Markdown→HTML for the PDF converter. |
| `Pygments` | 2.18.0 | yes | `app_pkg/markdown.py:12-14` | keep | Code-fence syntax highlighting in markdown→PDF. CVE-2026-4539 against this pin (see §2). |
| `playwright` | 1.44.0 | yes | `app.py:20` | keep | Headless Chromium for HTML→PDF. Version pinned to match `mcr.microsoft.com/playwright/python:v1.44.0-jammy` base image — bump in lockstep with the Dockerfile only. |
| `gunicorn` | 22.0.0 | **no** | `Dockerfile:62` (CMD) | comment-only | WSGI/ASGI server. Invoked from the Dockerfile CMD line, not imported. |
| `uvicorn` | 0.30.1 | **no** | `Dockerfile:62` (CMD: `--worker-class uvicorn.workers.UvicornWorker`) | comment-only | Provides the ASGI worker class gunicorn loads. Not imported in Python source. |
| `asgiref` | 3.8.1 | yes | `app.py:19` (`from asgiref.wsgi import WsgiToAsgi`) | keep | Wraps Flask (WSGI) into ASGI for the uvicorn worker class. Already pulled transitively via `Flask[async]`, but explicit pinning prevents silent breakage on Flask updates. |
| `unstructured[all-docs]` | 0.14.5 | yes | `app_pkg/documents.py` (via `from unstructured.partition.auto import partition`) | keep | Document→Markdown for non-PDF inputs (.docx, .pptx, .html, …). The `[all-docs]` extra pulls the full doc-handler stack. **Major skew** — see §3. |
| `pdfminer.six` | 20221105 | yes | `services/pdf_extraction/detectors.py:206-207` | keep | Columnar layout fallback in `TextHeuristicDetector._pdfminer_columnar`. 2 CVEs against this pin (§2). |
| `PyMuPDF` | 1.24.1 | yes (as `fitz`) | `services/pdf_extraction/service.py:4`, plus `fitz.Rect`, `fitz.Matrix`, `fitz.LINK_URI` references in same file | keep | Primary PDF parser. |
| `gevent` | 24.2.1 | **no** | nowhere | **remove** | Not imported anywhere. Dockerfile CMD uses `--worker-class uvicorn.workers.UvicornWorker`, not `gevent`. Pure leftover. |
| `mermaid` | 0.3.2 | **no** | nowhere | **remove** | The mermaid feature uses the **JavaScript** `mermaid@10` library from `cdn.jsdelivr.net` (`templates/mermaid_converter.html:77`). The Python package on PyPI does server-side rendering and is not used. As a bonus, this PyPI package's metadata is broken (`torch (>=1.7torchvision)` — invalid version specifier) — `pip-audit` cannot install it. |
| `deepgram-sdk` | 5.1.0 | yes | `services/deepgram_service.py:9` (`from deepgram import DeepgramClient`) | keep | Audio transcription API client. Major skew (latest 7.x) — see §3. |
| `google-cloud-texttospeech` | 2.21.0 | yes | `services/google_tts_service.py:4` (`from google.cloud import texttospeech`) | keep | Google Cloud TTS for the non-Gemini podcast path. |
| `google-genai` | >=1.0.0 | yes | `services/gemini/{client,script,synthesis,tts}.py` (`from google import genai`, `from google.genai import types`), `services/pdf_extraction/service.py:11` | keep | Gemini SDK — script generation, TTS, and PDF Vision fallback. Only unpinned entry in the file (intentional — Gemini SDK is fast-moving). |
| `requests` | 2.31.0 | yes | `app_pkg/integrations/notion.py:6`, `services/deepgram_service.py:8` (both as `http_requests`) | keep | HTTP client for Notion REST and Deepgram temp-key endpoint. 3 CVEs against this pin (§2). |
| `pydub` | 0.25.1 | yes (lazy) | `services/gemini/audio.py:18`, `services/gemini/client.py:39` (probe), `test_worker_libraries.py:18` | keep | Audio chunk concatenation for podcast generation. Lazy-imported with a fallback to the stdlib `wave` module — but the pydub path is the default. |
| `redis` | 5.0.1 | yes | `app.py:21`, `worker.py:6` | keep | Redis client used by both web and worker for the RQ job queue. **Major skew** — see §3. |
| `rq` | 1.16.0 | yes | `app.py:22-23`, `worker.py:7`, `app_pkg/podcasts.py:7` | keep | Job queue. **Major skew** — see §3. |
| `pdfplumber` | 0.10.4 | yes (lazy) | `services/pdf_extraction/detectors.py:57` | keep | One of five detectors in the PDF table-extraction ensemble. |
| `camelot-py` | 0.11.0 | yes (lazy) | `services/pdf_extraction/detectors.py:89` | keep | Lattice/stream table detector in the ensemble. **Major skew** (latest 1.x) — see §3. |
| `img2table` | 1.4.2 | yes (lazy) | `services/pdf_extraction/detectors.py:127,132` | keep | OpenCV-based table detector + optional Tesseract OCR. Pulls `opencv-python-headless` (next row) transitively. |
| `opencv-python-headless` | 4.10.0.84 | **no** | nowhere (no `import cv2`) | comment-only | Transitive dep of `camelot-py` and `img2table`. Explicit pinning here predates `pip-tools`-style lockfiles and prevents wheel ABI surprises (cv2 wheels are ~80 MB and platform-coupled). |
| `Flask-SQLAlchemy` | 3.1.1 | yes | `models.py:3` | keep | ORM. |
| `Flask-Login` | 0.6.3 | yes | `models.py:4`, `app_pkg/{auth,library,markdown,documents,audio,podcasts,mermaid,integrations/notion}.py` | keep | Session auth. |
| `Flask-WTF` | 1.2.1 | yes | `app_pkg/__init__.py:19` (`CSRFError`, `CSRFProtect`, `generate_csrf`) | keep | CSRF protection. |
| `nh3` | 0.2.18 | yes | `app_pkg/markdown.py:8,144` | keep | HTML sanitiser used in markdown→PDF after markdown-it rendering. |
| `pytest` | >=8.0 | yes | `tests/*.py` | keep | Test runner. Stage 6 dep — out-of-scope for this stage. |
| `pytest-asyncio` | >=0.23 | yes | `pytest.ini` (`asyncio_mode = auto`), used by tests/conftest.py | keep | Async test support. |
| `responses` | >=0.25 | yes | `tests/test_*.py` (mock HTTP) | keep | Test-only HTTP mocking for `requests`. |

### 1.1 Summary
- **27 production packages** in `requirements.txt` (excluding the 3 test deps below the comment header).
- **24 are directly imported** by the application source.
- **3 are not directly imported but justified** (`gunicorn`, `uvicorn`, `opencv-python-headless`) — kept with inline comments.
- **2 are not imported and not justified** (`gevent`, `mermaid`) — both removed in the follow-up commit.

---

## 2. CVE Findings

Tooling: `pip-audit --vulnerability-service osv` (OSV.dev), run on 2026-05-03 against `requirements.txt` minus the broken `mermaid==0.3.2` line (pip-audit cannot install it). **Reported, not fixed** — version bumps that pull in CVE patches are explicitly out-of-scope for Stage 7.

| Package | Pinned | CVE | Severity hint | Fix version |
|---|---|---|---|---|
| `Flask` | 3.0.3 | CVE-2026-27205 | unrated | 3.1.3 |
| `Pygments` | 2.18.0 | CVE-2026-4539 | unrated | 2.20.0 |
| `unstructured` | 0.14.5 | CVE-2025-64712 | unrated | 0.18.18 |
| `pdfminer.six` | 20221105 | CVE-2025-70559 | unrated | 20251230 |
| `pdfminer.six` | 20221105 | CVE-2025-64512 | unrated | 20251107 |
| `requests` | 2.31.0 | CVE-2024-47081 | unrated | 2.32.4 |
| `requests` | 2.31.0 | CVE-2024-35195 | unrated | 2.32.0 |
| `requests` | 2.31.0 | CVE-2026-25645 | unrated | 2.33.0 |

**Total:** 8 known vulnerabilities across 5 packages.

Context for triage:
- The app is **login-gated and LAN-only** (no public exposure documented in `CLAUDE.md`). All eight CVEs require an attacker to either reach the app or feed crafted input through one of the upload endpoints.
- The two `pdfminer.six` CVEs and the `unstructured` CVE involve crafted input documents — these are the highest-relevance items for a converter app that ingests user-supplied PDFs/Office docs, even from authenticated users.
- The three `requests` CVEs and the `Flask`/`Pygments` ones are network/HTML-rendering bugs respectively. Lower priority for an internal-only deployment.

OSV.dev does not consistently emit CVSS scores for these IDs, hence the "unrated" column. A future upgrade stage should re-run `pip-audit` against an updated pin set and triage from there.

---

## 3. Major-Skew Candidates

For each package, "Latest" is the version returned by `pip index versions <pkg>` on 2026-05-03 against the public PyPI index. **Listing only — no upgrade attempt in this stage.**

### Top 3 by skew + risk

| Rank | Package | Pinned | Latest | Why it's risky to upgrade |
|---|---|---|---|---|
| 1 | `unstructured[all-docs]` | 0.14.5 | 0.22.26 | 8 minor versions of API churn (Jun 2024 → 2026). The `[all-docs]` extra is the doc-partition entry point — every conversion path through `app_pkg/documents.py` flows through it. Has a CVE (see §2). Upgrades historically broken `partition()` defaults; needs the Stage 6 doc characterization tests as a safety net. |
| 2 | `rq` | 1.16.0 | 2.8.0 | Major version bump. RQ 2.x changed the default job-result backend (now uses Redis Streams optionally) and bumped the Redis client floor. Paired with the `redis` 5.x → 7.x bump below — both have to move together. Worker container needs full re-test. |
| 3 | `deepgram-sdk` | 5.1.0 | 7.0.0 | Two major versions. SDK 6 introduced async-first clients; SDK 7 reorganised `DeepgramClient` and the `keyterms` config surface (which `services/deepgram_service.py:46` uses via `keyterms.json`). |

### All entries with non-trivial skew

| Package | Pinned | Latest | Skew |
|---|---|---|---|
| `markdown-it-py` | 3.0.0 | 4.0.0 | major |
| `playwright` | 1.44.0 | 1.59.0 | minor (locked to Dockerfile base image) |
| `gunicorn` | 22.0.0 | 25.3.0 | major |
| `uvicorn` | 0.30.1 | 0.46.0 | minor (0.x) |
| `asgiref` | 3.8.1 | 3.11.1 | minor |
| `unstructured` | 0.14.5 | 0.22.26 | major |
| `pdfminer.six` | 20221105 | 20260107 | calendar-versioned, ~3 years |
| `PyMuPDF` | 1.24.1 | 1.27.2.3 | minor |
| `deepgram-sdk` | 5.1.0 | 7.0.0 | major (×2) |
| `google-cloud-texttospeech` | 2.21.0 | 2.36.0 | minor |
| `google-genai` | >=1.0.0 | 1.74.0 | (unpinned — floats with each `pip install`) |
| `requests` | 2.31.0 | 2.33.1 | minor |
| `redis` | 5.0.1 | 7.4.0 | major (×2) |
| `rq` | 1.16.0 | 2.8.0 | major |
| `pdfplumber` | 0.10.4 | 0.11.9 | minor (0.x) |
| `camelot-py` | 0.11.0 | 1.0.9 | major (0.x → 1.x) |
| `opencv-python-headless` | 4.10.0.84 | 4.13.0.92 | minor |
| `nh3` | 0.2.18 | 0.3.5 | minor (0.x) |
| `Flask-WTF` | 1.2.1 | 1.3.0 | minor |
| `Pygments` | 2.18.0 | 2.20.0 | minor |
| `Flask` | 3.0.3 | 3.1.3 | minor |

Packages **at or near latest** and not skew candidates: `gevent` (removed), `mermaid` (removed), `Flask-SQLAlchemy`, `Flask-Login`, `pydub`, `img2table`, `pytest*`, `responses`.

### Recommended sequencing for a future upgrade stage (NOT this stage)

1. **CVE-driven minor bumps first**: `Flask` 3.0.3 → 3.1.3, `Pygments` 2.18.0 → 2.20.0, `requests` 2.31.0 → 2.33.x, `pdfminer.six` → 20251230. Low blast radius, fixes 7 of the 8 CVEs.
2. **`unstructured`** by itself: it is its own conversion path and has a CVE. Test against the Stage 6 docs characterization tests + a manual smoke on a real .docx and .pptx.
3. **`redis` + `rq` together**: paired major bumps. Worker container must be rebuilt and the podcast end-to-end smoke run.
4. **`deepgram-sdk`** by itself: API reorganisation. Read the v6/v7 release notes before migrating `services/deepgram_service.py`.
5. Everything else (camelot 0.x→1.x, markdown-it-py 4.0, gunicorn 25.x): pure tech-debt sweep, lowest priority.

---

## 4. Out-of-Scope Confirmations

The following were deliberately *not* changed in this stage, per the Stage 7 prompt's anti-heuristics:

- **No major version bumps**, even where a CVE has a fix. Risk-managed in a separate stage.
- **No tooling switch** — `requirements.txt` stays flat, no `pyproject.toml` / `poetry` / `uv` / `pip-tools` migration.
- **No `requirements-dev.txt` split** — the test deps remain in the same file under the existing comment header.
- **No pin-strategy change** — exact pins (`==`) stay exact, the floor pin on `google-genai` (`>=1.0.0`) stays as a floor.
- **No CVE auto-fix** — only reported in §2 above.
