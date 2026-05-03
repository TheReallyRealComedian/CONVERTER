# CONVERTER - Multimedia Converter & Podcast Generator

## What is this?
A Flask web app for multimedia conversion: Markdown-to-PDF, document-to-Markdown, audio transcription (Deepgram), and AI podcast generation (Google Gemini TTS). Runs in Docker with Redis/RQ for background jobs.

## Tech Stack
- **Backend**: Flask (async), SQLAlchemy (SQLite), Flask-Login
- **Job Queue**: Redis + RQ (worker container for podcast generation)
- **APIs**: Gemini (script generation + TTS), Deepgram (transcription), Google Cloud TTS
- **Frontend**: Bootstrap + vanilla JS (Jinja2 templates)

## Key Files
- `app.py` — Bootstrap shim (~70 LOC); imports `app_pkg`, holds service singletons that tests patch at `app.<name>`
- `app_pkg/` — App factory + per-feature route modules (auth, library, markdown, documents, audio, podcasts, mermaid, `integrations/notion`). Each module exposes a `register(app)` function; **no Flask `Blueprint(...)`** — endpoint names stay flat so templates' `url_for("login")` etc. don't need rewrites
- `app_pkg/config.py` — Shared constants (e.g. `OUTPUT_DIR`)
- `services/gemini/` — Gemini package: `client`, `script`, `tts`, `synthesis`, `audio`, `dialogue`, `prompts`. Public API as `GeminiService` via `services/gemini/__init__.py`
- `services/deepgram_service.py`, `services/google_tts_service.py`, `services/audio_chunker.py`, `services/pdf_extraction/` — Other services (untouched in cleanup)
- `tasks.py` — RQ background tasks (podcast generation)
- `worker.py` — RQ worker process
- `models.py` — SQLAlchemy models (User, ConversionHistory)
- `tests/` — Characterization tests (36 tests, ~5s); mocks at SDK-singleton boundary so they survive future service splits
- `static/js/` — Per-feature JS modules + shared `_utils.js`; templates inline only small `window.PageData = {…}` blocks
- `static/css/style.css` — Single stylesheet with TOC + section comments (not split by design)

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
- **Routing pattern:** route modules in `app_pkg/` expose a `register(app)` function rather than Flask `Blueprint(...)` objects, to keep endpoint names flat. New routes must follow the same pattern (decided in Stage 2 of the 2026-05 cleanup; see [docs/cleanup_plan.md](docs/cleanup_plan.md)).
- **Service-singleton pattern:** SDK clients (`deepgram_service`, `gemini_service`, `google_tts_service`, `task_queue`, `async_playwright`, `partition`) are bound at the top level of `app.py` so tests patch them at `app.<name>`. New SDK integrations must follow this convention or update test patches.

## Historical
The 2026-05 cleanup wave (Stages 0–7, 18 Findings F-001…F-018, deferred CVE-upgrade items, remaining-work list) is archived to [docs/cleanup_plan.md](docs/cleanup_plan.md). Architectural decisions made there are mirrored in *Architecture Notes* above.

---

## UX Review Plan (started 2026-05-03)

UX-Verbesserung läuft als 3-Schritt-Kaskade pro Feature (Methodik: Duan et al. *Heuristic Evaluation with LLMs*, CHI 2024; gefiltert durch Nielsens Heuristiken H1, H4, H6, H9). Begründung Kaskade > Monster-Prompt: jede Stufe hat engen Fokus, der Overseer kann zwischen Stufen manuell korrigieren, und der strukturierte Output von Stufe N wird Input von Stufe N+1.

**Per Feature drei Stufen:**

1. **Inventur** — alle interaktiven Elemente kartieren (Element-Typ, Label, Aktion, vorhandene/fehlende States: default/hover/focus/disabled/loading/error/success/empty). Kein Bewerten, nur Mapping.
2. **Heuristik-Review** — Inventur durch Nielsen H1/H9/H4/H6 filtern (System-Status, Hilfe-bei-Fehlern, Konsistenz, Wiedererkennen-statt-Erinnern). Output: severity-ranked Findings-Tabelle (1=kosmetisch … 4=kritisch).
3. **Patterns + Microcopy** — pro Finding konkretes UI-Pattern, deutsche Microcopy (Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern), visuelle Hinweise, Aufwand XS/S/M/L. Top-5 Quick-Wins per Impact-Score (Schweregrad × 5 / Aufwand-Gewicht).

**Produkt-Kontext (für Stufe-2-Prompts):** Single-User-App (nur Oliver), kein Multi-User, kein öffentlicher Zugang. LAN-only, login-protected. Primäre Aufgaben unterscheiden sich pro Feature und gehen in den Stufe-2-Prompt der jeweiligen Feature-Stage.

**Methodik:** Sub-Threads kombinieren statisches Code-Reading (templates + JS + Route-Handler) mit Live-Walkthrough am laufenden Container (`docker compose up`, dann Browser-Interaktion via verfügbare MCP-Tools wenn möglich). Viele UI-States (loading, error, hover, validation feedback, network-failure-recovery) sind runtime-only und nicht aus Code-Reading allein ableitbar — daher beide Quellen erforderlich.

**Guardrails (jede Stufe):**
- Stufen 1–3 sind reine Analyse, **kein Code-Change**.
- Implementierung der Quick-Wins läuft als separater Sub-Thread pro Cluster, eine PR pro Cluster.
- Charakterisierungstests aus Cleanup Stage 6 müssen grün bleiben (`pytest tests/`).
- Wenn die Inventur Bugs aufdeckt (z.B. fehlerhafte Validierung, Crashes), als separates Issue notieren — nicht in der Inventur-Phase patchen.
- Eine Feature-Stage = ein Branch mit drei Commits (einer pro Stufe-Output-Datei).

---

### F-1: document_converter
**Status:**
- F-1.1 Inventur: ☑ done 2026-05-03 → [docs/ui_inventory_document_converter_2026-05.md](docs/ui_inventory_document_converter_2026-05.md). Live-Walkthrough via Claude in Chrome auf Mintbox. 24 Elemente, 6 fehlende States im Code, 5 Code↔live-Divergenzen, 9 separate Befunde (3 davon vom Sub-Thread als wahrscheinliche Bugs flagged: Drop-Zone-Active-Highlight transparent, Save-Btn `.saved`-Klasse stale, Empty-Submit visuell silent). Disposition Bug-vs-Heuristik-Finding wird in F-1.2 entschieden.
- F-1.2 Heuristik-Review: ☑ done 2026-05-03 → [docs/ui_findings_document_converter_2026-05.md](docs/ui_findings_document_converter_2026-05.md). 19 Findings (Sev 4: 2, Sev 3: 7, Sev 2: 7, Sev 1: 3) + 3 reine Bug-Tickets (B1 CSS-Override, B2 `.saved`-Klasse-Reset, B3 clear-file-Handler-Scope). Disposition der 9 Stufe-1-Bemerkungen: 6 nur Findings, 3 Findings + Bug-Tickets, 0 nur Bugs. Schwere Findings konzentrieren sich auf Empty-Submit-Silent (F1/F2 H1+H9 Sev 4), Result-Persistenz nach Clear (F3/F4), Save-Button Stale-Visual (F5/F6), Save-Failure-Inkonsistenz `alert()` vs Banner (F7/F8), Format-Label-Mismatch (F9).
- F-1.3 Patterns + Microcopy: ☑ done 2026-05-03 → [docs/ui_patterns_document_converter_2026-05.md](docs/ui_patterns_document_converter_2026-05.md). 14 Pattern-Blöcke (5 konsolidiert + 9 einzeln). Alle Patterns mit existierenden Neomorphism-Komponenten realisierbar; höchster Aufwand M (Pattern 9 Drop-Zone-Loading), Rest S/XS. Top-5 Quick-Wins (nach Impact-Score): Pattern 2 (Clear-Reset, 15.0, XS), Pattern 3 (Save-Btn-Lifecycle, 15.0, XS), Pattern 1 (Empty-Submit-Banner, 10.0, S), Pattern 10 (Auto-Scroll, 10.0, XS), Pattern 4 (Save-Failure-Banner, 7.5, S).
- Implementation: ☐ awaiting Oliver's selection of which Quick-Wins to implement (and in which clusters)

### F-2..F-N: queued
Reihenfolge wird nach Abschluss von F-1.3 entschieden, basierend auf Pilot-Erfahrung. Kandidaten in grober Komplexitäts-Reihenfolge: `audio_converter` (komplex, Recording + Upload + Live-Transkription), `markdown_converter` (Editor + Preview + PDF + Reader-Mode), `library_detail` (View/Edit/Delete/Notion-Integration), `library` (Liste + Filter + Empty-State), `mermaid_converter`, `login`, podcast-generation flow (cross-template async).

---

## How to launch a UX cascade step in a fresh thread

Open a new Claude Code session in this repo and paste the prompt drafted by the overseer thread for the current Stufe. The prompt is self-contained — for Stufe 2/3 it embeds the prior Stufe's output (Tabelle/Findings) so no context-load from CLAUDE.md is needed.

When done: post the resulting Tabelle/Findings/Patterns back to the overseer thread; the overseer drafts the next Stufe's prompt with the new output embedded, and updates the Status checkboxes here.
