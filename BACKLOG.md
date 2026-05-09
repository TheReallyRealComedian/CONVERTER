# BACKLOG

Source-of-Truth für offene Items. Geordnet als **sequenzielle Sprint-Roadmap** — wir arbeiten von oben nach unten ab. Codes (`SEC`, `CVE-LOW`, …) sind stabil und gehen 1:1 in die Sprint-Prompt-Doc-Namen unter [docs/archive/sprint-prompts/](docs/archive/sprint-prompts/).

Prio-Skala: **P0** kritisch (Production kaputt) · **P1** als nächstes dran · **P2** in der Sequenz weiter unten · **P3** nice-to-have / nicht in Sequenz / Reminder.

---

## In Sequenz — Cleanup-Abschluss

### 1. `SEC` — Security-Hardening-Sweep · P1 · M
Bündelt verbleibende Sicherheits-Findings aus Cleanup Stage 4:

- **F-005 Path-Traversal** in `podcast_download` — `startswith` durch `Path.is_relative_to()` oder `os.path.commonpath` ersetzen ([app_pkg/podcasts.py](app_pkg/podcasts.py)).
- **F-006 für `markdown_converter`** — Backend-Whitelist analog F-1 Cluster D in `app_pkg/documents.py:30` (precomputed `accept`-String + 400+DE-JSON für unsupported extensions).
- ~~**F-006 für `audio_converter`**~~ — durch F-2 Cluster II Pattern 13 erledigt (2026-05-09).
- **F-013 Input-Allowlist** — User-supplied Config-Strings vor Upstream-API-Calls validieren (Deepgram language, Google TTS voice/language/rate/pitch, Gemini narration_style/script_length/num_speakers).

Vorlagen: F-1 Cluster D ([app_pkg/documents.py:30](app_pkg/documents.py#L30)) für die Backend-Whitelist-Mechanik.

### 2. `HYG` — Code-Quality-Sweep der Stage-4-Restposten · P1 · M
Bündelt verbleibende Hygiene-Findings (alle Low-Severity, kein Behavior-Change):

- **F-002** narrow `except` in `highlight_code` zu `pygments.util.ClassNotFound` ([app_pkg/markdown.py](app_pkg/markdown.py)).
- **F-007** `secure_filename(None)` AttributeError absichern ([app_pkg/markdown.py](app_pkg/markdown.py)).
- **F-008 partial** — 4 logging-Sites ohne `exc_info=True`: markdown PDF generation, Deepgram-key issuance, podcast TTS-temp cleanup, Notion-MCP transport failure.
- **F-011** Service-availability-Decorator (`@require_service('deepgram')` etc.) für 6 Endpoints.
- **F-012** Dead `if not file:` check entfernen ([app_pkg/documents.py](app_pkg/documents.py)).
- **F-015** Timeout-Alignment — drei Magic-Timeouts (Gemini 300s, Deepgram 600s, RQ 600s) dokumentieren oder konsolidieren.
- **F-016** Doppel-Log in `_filter_metadata_lines` ([services/gemini/dialogue.py](services/gemini/dialogue.py)) — callee-side line droppen.
- **F-017** `isinstance(data, dict)`-Check für `request.get_json()` an 6 Stellen einheitlich.

Vollständige Beschreibung jedes Findings: [docs/cleanup_plan.md#findings-populated-by-stage-4](docs/cleanup_plan.md).

### 3. `CVE-LOW` — Minor-Bumps mit CVE-Fixes · P2 · S
Drei Pakete mit jeweils einem CVE, Minor-Bump, kein API-Break erwartet:

- **Pygments** 2.18.0 → 2.20.0 (CVE-2026-4539)
- **requests** 2.31.0 → 2.33.0 (3 CVEs)
- **Flask** 3.0.3 → 3.1.3 (CVE-2026-27205)

Pre-Test-Welle nach jedem Bump (`pytest tests/` 38/38). Reihenfolge: Pygments → requests → Flask (lowest blast radius first).

### 4. `CVE-PDF` — User-Upload-Pfad-CVEs + Major-Skew · P2 · L
Beide auf User-Upload-Pfad, beide mit echten CVEs:

- **pdfminer.six** 20221105 → 20251230 (2 CVEs)
- **unstructured[all-docs]** 0.14.5 → 0.22.26 (CVE-2025-64712, **8 Minor-Versionen**, doc-partition API hat sich geändert)

Eigener Sprint weil Bibliothek-API potenziell ändert. Pre-Flight: dokumentieren welche Test-Cases User-Upload-Pfad treffen, dann Re-Run nach Bump.

### 5. `CVE-RQ` — Job-Queue Major-Bump · P2 · L
**rq** 1.16.0 → 2.8.0 + parallel **redis** Major-Bump. Worker-Container + Web-Container müssen synchron. Charakterisierungstests für Podcast-Generation (Stage-6, 11 Tests inkl. F-001) sind die wichtigste Verteidigung.

### 6. `CVE-DG` — Deepgram-SDK Major-Bump · P2 · L
**deepgram-sdk** 5.1.0 → 7.0.0 (zwei Majors, Client-Surface reorganisiert). Audio-Transcription-Tests (Stage-6, 4 Tests) als Re-Run-Basis.

---

## In Sequenz — UX-Cascade-Fortsetzung

### 7. `F3-PICK` — F-3 Feature-Wahl + Inventur · P2 · S
Kandidaten: `markdown_converter`, `library`, `library_detail`, `mermaid_converter`, `login`, podcast-flow. Master entscheidet vor Sprint-Start basierend auf Schmerz/Aufwand. Sprint führt dann Stufe 1 (Inventur) durch.

### 8. `F3-REVIEW` — F-3 Heuristik-Review · P2 · S
Stufe 2: Findings-Tabelle Sev 1–4 nach Nielsen H1/H4/H6/H9. Erwartung: ~30–50% Cross-Feature-H4 (Helper-Reuse aus F-1/F-2).

### 9. `F3-PATTERNS` — F-3 Patterns + Microcopy · P2 · S
Stufe 3: Pattern-Blöcke + DE-Microcopy + Top-N Quick-Wins per Impact-Score. Cluster-Vorschlag für Implementation am Ende.

### 10. `F3-IMPL-*` — F-3 Implementation-Cluster · P2 · M-L
1 bis N Sprints je nach Pattern-Menge (F-1 hatte 6 Cluster, F-2 Cluster I bündelte 12). Code-Sprint-Erfahrung: bei stark verkoppelten Patterns + ~40% Cross-Feature-H4 ist Holistic-Rewrite effizienter als sequentielle Edits.

### 11. `F-N…` — Folge-Wellen für Restfeatures · P2 · je L
Pro Restfeature wieder die 3 Methodik-Stufen + Implementation-Cluster. Reihenfolge wird am Ende von F-3 entschieden.

---

## In Sequenz — Wave-Close

### 12. `WAVE-CLOSE` — Strukturelles Closing · P3 · XS
- `docs/cleanup_plan.md` Header auf "fully closed" updaten (alle Findings + Outstanding work durch).
- `OVERSEER_HANDOFF.md` archivieren oder löschen (durch CLAUDE.md/STATUS.md/BACKLOG.md ersetzt).
- Ggf. UX-Cascade-Doku-Convention dokumentieren (für künftige Wellen).

---

## P3 — nicht in Sequenz / Reminder

- **`getUserMedia`-in-`socket.onopen`-Bug** im audio_converter (Permission-Prompt erst nach WS-Handshake, in F-2 Cluster I als Out-of-Scope respektiert). Größe S. Fold-Kandidat für `HYG` falls billig, sonst eigener Sprint.
- **Englische UI-Strings in [static/js/library.js](static/js/library.js)** (2 Strings) und **[static/js/library_detail.js](static/js/library_detail.js)** (6 Strings). Größe XS. Wahrscheinlich mit F-3-`library`/`library_detail`-Welle gefolded — lasse hier nur als Sichtbarkeit, kein eigener Sprint.
- **Playwright-UI-Tests einführen** — schließt Test-Coverage-Lücke (Charakterisierungstests rendern keine Templates, mocken SDK-Boundaries). Größe L. Eigener Sprint, aber außerhalb der Cleanup-Sequenz — erst wenn UX-Wellen durch sind und stabiler State erreicht ist.
- **Tesseract-NC-33-Workaround** (Mintbox-System-Kontext, kein CONVERTER-Touch) — Reminder, kein aktiver Sprint. Quelle: [/Volumes/MintHome/CLAUDE.md](/Volumes/MintHome/CLAUDE.md), Stand 2026-05-01.

---

## Erledigt (rolling, älteste fallen raus)

- ☑ F-2 Cluster II (audio_converter Sev 2+1, P13–P21; F-2 strukturell abgeschlossen) — 2026-05-09
- ☑ Working-Practice-Bootstrap (CLAUDE.md slim, STATUS.md, BACKLOG.md, Sprint-Prompt-Template) — 2026-05-09
- ☑ F-2 Cluster I (audio_converter Sev 4+3, P1–P12) — 2026-05-03
- ☑ F-1 Hot-Fix (Jinja2 Generator-Expression) — 2026-05-03
- ☑ F-1 (document_converter, alle 14 Patterns + 3 Bug-Tickets, 6 Cluster) — 2026-05-03
- ☑ Cleanup-Wave Stages 0–7 (cleanup_plan.md, F-001…F-018 strukturell durch) — 2026-05-02/03
