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
- Implementation:
  - Cluster A (P2 + P3 + P10, schließt B2 + B3): ☑ done 2026-05-03 → commit `153d418`. Statisch verifiziert (`pytest tests/` 36/36 grün, 6.05s); Live-Smoke ausstehend (Container hat keinen Source-Bind-Mount, Sub-Thread hielt `docker cp`/Rebuild zurück). Eine Side-Note: Template-Static-Text "Save to Library" in `templates/document_converter.html` wird zwar nie sichtbar (JS überschreibt vor Anzeige), aber inhaltlich falsch → folded into Cluster B.
  - Cluster B (P1 + P4 + Helper `showAlert`): ☑ done 2026-05-03 → commit `1242e48`. Pytest 36/36 grün; `docker cp`-Smoke verifizierte ausgelieferte Assets (Strings + CSS-Regel + Helper-Export). Live-Browser-Walkthrough zurückgehalten (Login-Flow als oversized für client-only-Patch). Sub-Thread-Befund: Conversion-Error-Pfad nutzt noch raw `innerHTML` statt `showAlert` — sollte zu Cluster C gefoldet werden (gleiche Code-Pfade wie P7).
  - Cluster C (P5 + P7 + B1 + Conversion-Error-Fold): ☑ done 2026-05-03 → commit `a96eb93`. Pytest 36/36 grün; statisch verifiziert. B1-Root-Cause war eine Box-Shadow-Transition-Interpolations-Artefakt (kürzere Shadow-Liste wird mit `transparent 0 0 0 0` gefüllt — keine CSS-Variable-Bug, keine Spezifitäts-Übersteuerung; rgba-Literal war einfach unsichtbar gegen den Hintergrund derselben Farbfamilie). showAlert hat jetzt closable + autoDismissMs Options, default-Verhalten passt für alle Cluster-B-Calls automatisch (Danger persistent, mit ×).
  - Cluster D (P6 + P8 — Format-Honesty + Unsupported-Drag-Warning): ☑ done 2026-05-03 → commit `e68b6dd`. Pytest 37/37 grün (1 neuer Test `test_transform_document_unsupported_extension_returns_400`). Single-Source-of-Truth `ACCEPTED_EXTENSIONS` in `app_pkg/documents.py`, fließt nach Template (`accept`-Attribut + `window.PageData`) und JS. Backend liefert jetzt 400+DE-JSON für unsupported Extensions — schließt F-006 (aus Cleanup-Stage 4) **für `document_converter`**. Andere Upload-Endpoints (markdown, audio) bleiben auf F-006-Stand und werden in ihren F-2/F-N-Stages adressiert. Drag-MIME-Detection ist Best-Effort (Browser-abhängig) — Drop-Pfad ist die zuverlässige Validierung.
  - Cluster E (P11 + P13 — a11y): ☑ done 2026-05-03 → commit `990d1d3`. Pytest 37/37 grün; statisch verifiziert. Drop-Zone ist jetzt Tab-Stop mit Enter/Space-Trigger; Result-`<pre>` ist fokussierbarer `region` mit aria-label. `:focus-visible`-Vokabular reuse von `c-btn`. Bonus: Cluster B's `try { dropZone.focus(); } catch(_) {}` Forward-Reference jetzt zu plain `dropZone.focus()` aufgeräumt (Drop-Zone ist verlässlich fokussierbar).
  - Polish-1 (P12 Filename-Format + P14 Download-Toast + DE-string-pass): ☑ done 2026-05-03 → commit `ea9db78`. Pytest 37/37 grün. `formatFileSize` + `showToast` (singleton-Strategie ohne extra Container) als geteilte Helper in `_utils.js`. 7 DE-Strings übersetzt (5 Template + 2 JS). Sub-Thread fand existierende `showToast`-Calls in `library.js`/`library_detail.js` und wahrte Backward-Compat (Display-Zeit 2000→2500 ms ist die einzige Verhaltens-Änderung — harmlos). Englische Strings in `library.js` (2) und `library_detail.js` (6) wurden namentlich aufgelistet als F-N-Material — nicht hier gefixt.
  - Polish-2 (P9 Drop-Zone-Loading-Indikation, Single-Phase): ☑ done 2026-05-03 → commit `37f8420`. Pytest 37/37 grün. CSS-only Spinner + Overlay analog Cluster-C-Pattern; `pointer-events: none` während Loading verhindert dass Cluster-B/D-States sich überlagern. Defense-in-depth Cascade falls eine State-Klasse aus früherer Interaktion stehengeblieben ist. Microcopy Drop-Zone "Wird verarbeitet …" ergänzt Convert-Btn "Wird umgewandelt …" (verschiedene Metaphern, beide gleichzeitig sichtbar sinnvoll).
- **F-1 strukturell abgeschlossen 2026-05-03.** Alle 14 Patterns + 3 Bug-Tickets implementiert. 6 Implementierungs-Cluster (A, B, C, D, E, Polish-1, Polish-2). Höchster Schweregrad-Befund (F1/F2 Empty-Submit silent, Sev 4) gefixt durch Pattern 1 in Cluster B. Alle Findings, deren Bug-Bestandteil F-006 für `document_converter` adressiert war, sind durch Cluster D's Backend-Whitelist mitgelöst. Pytest 37/37 grün durchgängig. **F-2 Feature-Wahl ausstehend.**

### F-2: audio_converter
**Status:**
- F-2.1 Inventur: ☑ done 2026-05-03 → [docs/ui_inventory_audio_converter_2026-05.md](docs/ui_inventory_audio_converter_2026-05.md). 47 Elemente (42 ohne Headings/Feedback), 13 fehlende States im Code, 7 live-verifizierte Code↔live-Divergenzen, 6 unverifizierbare States, 6 audio-spezifische States. **Vier prominente Surprises:** (1) Drag-Drop ist eine Lüge — Label verspricht es, JS hat keine Handler; deutliche Inkonsistenz vs document_converter. (2) 11+ `alert()`-Call-Sites + Inline-Span + c-alert-Banner → drei konkurrierende Error-UI-Patterns ohne Verwendung des in F-1 etablierten `showAlert()`-Helpers. (3) Configuration-Error-Banner kippt die ganze Seite (Podcast-Tab tot obwohl der nur Gemini braucht). (4) Englische Strings flächendeckend.
- F-2.2 Heuristik-Review: ☑ done 2026-05-03 → [docs/ui_findings_audio_converter_2026-05.md](docs/ui_findings_audio_converter_2026-05.md). 32 Findings (Sev 4: 5, Sev 3: 13, Sev 2: 9, Sev 1: 5) + 9 reine Bug-Tickets (B1–B9). **13 von 32 Findings (~41%) sind Cross-Feature-H4** — d.h. lösbar durch existing-helper-reuse aus F-1 (`showAlert`, `showToast`, `formatFileSize`, `resetSaveBtn`-Pattern, Drop-Zone-Pattern, accept-Whitelist, `:focus-visible`-Vokabular). Disposition der vier Stufe-1-Surprises: alle vier wurden zu Heuristik-Findings (Drag-Drop-Lüge → Sev 4 + Bug B1; 11+ alert() → 4 Findings Sev 3; Config-Error-Global → 2 Findings Sev 3; Englische Strings → Sev 3). Zwei zusätzliche Surprises im Review: **Surprise 5** (Silent-fail-Familie F1+F5+F15 = gemeinsames Anti-Pattern); **Surprise 6** (Live-Textarea-Append-Race aktiv überschreibt User-Edits, subtiler als gedacht).
- F-2.3 Patterns + Microcopy: ☑ done 2026-05-03 → [docs/ui_patterns_audio_converter_2026-05.md](docs/ui_patterns_audio_converter_2026-05.md). 21 Pattern-Blöcke (konsolidiert aus 32 Findings, 11 Konsolidierungen). 4 Cross-Feature-Konvergenz-Blöcke gebündelt (P5 Alert-Sweep, P6 Save-Lifecycle, P12 DE-Microcopy + Date-Format, P19 formatFileSize). Top-3 Quick-Wins alle Score 15.0 / Sev 3 / XS: P6 Save-Btn-Lifecycle, P9 Lang-Disabled-sichtbar, P11 Copy-Guard-Sentinel. **Cluster-Vorbereitung vom Sub-Thread:** I = 12 Patterns (P1–P12, Sev 4+3), II = 9 Patterns (P13–P21, Sev 2+1). Sub-Thread bot 3-Cluster-Split-Fallback an (Ia/Ib/Ic) falls 12 zu groß. Surprise 5 (Silent-fail-Familie) als drei separate Patterns mit kohärenter Sub-Gruppe gehandhabt. Surprise 6 (Live-Textarea-Race) als P10 mit "Stream hat Vorrang während Recording, User-Edit nur nach Stop" gelöst.
- Implementation:
  - Cluster I (P1–P12, Sev 4+3, schließt B1–B8): ☑ done 2026-05-03 → commit `ef78508`. Pytest 37/37 grün. Helper-Reuse durchgängig (`showAlert`/`showToast`/`formatFileSize` aus `_utils.js`); `grep -c "alert(" static/js/audio_converter.js` = 0; keine englischen UI-Strings mehr in Audio-spezifischen Files. Sub-Thread-Befund: 12-in-einem war "gerade noch handhabbar" — Drei-Sub-Batch-Strategie (Foundation → Critical-UX → State-Lifecycle) hat es ermöglicht, weil die Patterns stark verkoppelt waren. Holistic-Rewrite war effizienter als sequentielle 12 Edits. Empfehlung für Cluster II / F-3: 12 OK bei ähnlich hoher Verkopplung + ~40% Cross-Feature-H4; sonst 3-Cluster-Split. Surprises: bestehender `getUserMedia`-in-`socket.onopen`-Bug (Permission-Prompt erst nach WS-Handshake) als Out-of-Scope respektiert; B7 Backend-Flag-Split (P8) machte einen `{% if not deepgram_api_key_set %}`-Wrapper überflüssig (entfernt). **⚠️ Live-Smoke-Gate vor Cluster II — gemäß F-2-Sprint-Strategie obligatorisch.**
  - Cluster II (P13–P21, Sev 2+1): ☐ awaiting Live-Smoke-Confirmation für Cluster I

### F-3..F-N: queued
Wird nach F-2-Abschluss entschieden. Kandidaten: `markdown_converter`, `library_detail`, `library`, `mermaid_converter`, `login`, podcast-flow.

---

## How to launch a UX cascade step in a fresh thread

Open a new Claude Code session in this repo and paste the prompt drafted by the overseer thread for the current Stufe. The prompt is self-contained — for Stufe 2/3 it embeds the prior Stufe's output (Tabelle/Findings) so no context-load from CLAUDE.md is needed.

When done: post the resulting Tabelle/Findings/Patterns back to the overseer thread; the overseer drafts the next Stufe's prompt with the new output embedded, and updates the Status checkboxes here.
