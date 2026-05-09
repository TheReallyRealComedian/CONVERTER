# CONVERTER — Multimedia Converter & Podcast Generator

## What is this?
Flask web app for multimedia conversion: Markdown-to-PDF, document-to-Markdown, audio transcription (Deepgram), and AI podcast generation (Google Gemini TTS). Runs in Docker with Redis/RQ for background jobs. Single-User-App (nur Oliver), LAN-only, login-protected.

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
- `services/deepgram_service.py`, `services/google_tts_service.py`, `services/audio_chunker.py`, `services/pdf_extraction/`
- `tasks.py` — RQ background tasks (podcast generation)
- `worker.py` — RQ worker process
- `models.py` — SQLAlchemy models (User, ConversionHistory)
- `tests/` — Characterization tests (~37 tests, ~5s); mocks at SDK-singleton boundary so they survive future service splits
- `static/js/` — Per-feature JS modules + shared `_utils.js` (Helper: `showAlert`, `showToast`, `formatFileSize`); templates inline only small `window.PageData = {…}` blocks
- `static/css/style.css` — Single stylesheet with TOC + section comments (not split by design)

## Running
```bash
docker compose up --build
```
App runs on `localhost:5656`. Requires `.env` with `GEMINI_API_KEY`, `DEEPGRAM_API_KEY`, `SECRET_KEY`, and `google-credentials.json` for Google Cloud TTS.

## Gemini Models Used
- **Script generation**: `gemini-2.5-flash`
- **TTS**: `gemini-2.5-flash-preview-tts` / `gemini-2.5-pro-preview-tts`

## Architecture Notes
- Podcast generation is async: web enqueues job via Redis, worker processes it, result shared via `podcast_data` Docker volume.
- Long podcasts are chunked (max 80 lines / 3000 chars per chunk) and concatenated with pydub.
- Frontend polls `/podcast-status/<job_id>` until complete.
- **Routing pattern:** route modules in `app_pkg/` expose a `register(app)` function rather than Flask `Blueprint(...)` objects, to keep endpoint names flat. New routes must follow the same pattern.
- **Service-singleton pattern:** SDK clients (`deepgram_service`, `gemini_service`, `google_tts_service`, `task_queue`, `async_playwright`, `partition`) are bound at the top level of `app.py` so tests patch them at `app.<name>`. New SDK integrations must follow this convention or update test patches.
- **Test-Suite-Limit:** Charakterisierungstests rendern keine Templates und mocken SDK-Boundaries. UI-/Template-Bugs (z.B. Jinja2-Syntax) werden nicht gefangen — Live-Smoke nach Template-Änderungen erforderlich.

## Code-Stil
- UI-Strings deutsch (Microcopy: Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern).
- Helper aus `static/js/_utils.js` wiederverwenden, nicht reimplementieren.
- Pre-Flight für Sprints: `pytest tests/` muss grün bleiben.

---

## Working Practice — Master = Dispatch, Sub-Thread = Execute

**Prinzip in einem Satz**: Master-Thread plant, pflegt Backlog, schreibt Sprint-Prompts. Sub-Thread pro Sprint führt aus. **Master macht keine Code-Edits.**

**Mechanik**:
1. Master schreibt Sprint-Prompt-Doc unter `docs/archive/sprint-prompts/SPRINT_<CODE>_<NAME>_<datum>.md` — imperativ, in Phasen geschnitten, Stop-Regel nach jeder Phase.
2. Master liefert **direkt im Chat einen copy-paste-fähigen Anfangs-Prompt** in folgendem Format:

   ```
   Du übernimmst den Sub-Thread für Sprint <CODE> (<KURZ-TITEL>).

   Sprint-Prompt vollständig lesen: <ABSOLUTER-PFAD-ZUR-SPRINT-DOC>

   Dann Phase 1 direkt starten — nicht zusammenfassen, nicht planen, nicht beurteilen. Sprint-Prompt = ausführen. Master macht Dispatch, du machst Execute. Nach jeder Phase Stop + Bericht.

   Working-Practice in /Volumes/MintHome/CODE/CONVERTER/CLAUDE.md unter Sektion "Working Practice".
   ```

   Kein `cat ... | pbcopy`-Indirection-Schritt mehr — der direkte Prompt-Block pinnt den Sub-Thread auf Executor-Rolle und verhindert das „lies und plane"-Missverständnis.
3. Oliver kopiert den Block aus dem Chat, öffnet einen frischen Sub-Thread, paste-t.
4. Sub-Thread arbeitet Phasen ab, berichtet zwischen Phasen, wartet auf Sign-off.
5. Sub-Thread pflegt am Ende STATUS.md + BACKLOG.md + ggf. Memory.
6. Master nimmt Stand-Update zurück und schreibt nächsten Sprint-Prompt.

**Master-erlaubte Edits** (Dispatch-Artefakte, kein Code):
- `CLAUDE.md`, `STATUS.md`, `BACKLOG.md`
- `docs/archive/sprint-prompts/SPRINT_*.md`, `_TEMPLATE.md`, `README.md`
- Hand-off-Doc (`MASTER_BACKLOG_HANDOFF_<datum>.md`) beim Master-Wechsel
- Memory-Zone (`feedback_*.md`, `reference_*.md`, `MEMORY.md`)

Alles andere — Feature-Code, Templates, Tests, Services — gehört in einen Sprint.

**Disziplin**:
- Master sagt nie „mach jetzt einfach". Bei Frust: max. 1 Klärungsfrage, dann Sub-Session.
- Sprint-Prompt-Doc ist Ausführungs-Doku, nicht Diskussions-Doku. Review/Diskussion ist ein eigener Sprint.
- Phase 0 (Workshop / Mechanik-Fragen) ist **Ausnahme**, nicht Default — nur wenn echt eine Mechanik-Wahl offen ist. Triviale Sprints starten direkt in Phase 1.
- BACKLOG.md = Source-of-Truth offene Items. STATUS.md = aktueller Stand.
- Parallele Sprints sind erlaubt, solange ihre Backlog-Items disjunkt sind.

**Sprint-Sizes**: S/M/L/XL — Daumenregel: S = ein File-Touch, M = ein Feature-Cluster, L = mehrere Features, XL = Schema/Migration. Größer als L → splitten.

**Bootstrap für neuen Master-Thread**: lies `CLAUDE.md` → `STATUS.md` → `BACKLOG.md` → ggf. `MASTER_BACKLOG_HANDOFF_<datum>.md` (sofern vorhanden). Dann Sprint-Prompt schreiben.

---

## Historical
- 2026-05 Cleanup-Welle (Stages 0–7, 18 Findings F-001…F-018): [docs/cleanup_plan.md](docs/cleanup_plan.md). Architektur-Entscheidungen daraus sind oben in *Architecture Notes* gespiegelt.
- 2026-05 UX-Cascade F-1 (`document_converter`) + F-2 (`audio_converter`) Cluster I: dreistufige Methodik (Inventur → Heuristik-Review → Patterns + Microcopy) basierend auf Duan et al. *Heuristic Evaluation with LLMs* (CHI 2024) + Nielsen H1/H4/H6/H9. Outputs in [docs/ui_inventory_*.md](docs/), [docs/ui_findings_*.md](docs/), [docs/ui_patterns_*.md](docs/). Methodik-Lehren: Helper-Reuse aus F-1 senkt F-2-Aufwand (~41% Cross-Feature-H4); Holistic-Rewrite mit Sub-Batches schlägt sequentielle Edits bei stark verkoppelten Pattern-Gruppen.
- Vorgänger-Hand-off (2026-05-03 Maschinen-Wechsel): [OVERSEER_HANDOFF.md](OVERSEER_HANDOFF.md) — historisch, neue Hand-offs folgen `MASTER_BACKLOG_HANDOFF_<datum>.md`-Konvention.
