# STATUS

**Stand**: 2026-05-09
**Live**: `localhost:5656` im Docker-Stack auf Mintbox + `converter.smallpieces.de`. Pytest 48/48 grün.

## Aktueller Sprint
_Keiner aktiv. HYG ☑ done 2026-05-09. Stage-4-Findings-Block damit durch (F-014 wontfix). Verbleibende Sequenz: CVE-LOW → CVE-PDF → CVE-RQ → CVE-DG → F3-* → WAVE-CLOSE._

## Sequenz-Plan
Wir arbeiten den **Cleanup-Abschluss** als sequenzielle Sprint-Roadmap ab (Reihenfolge in [BACKLOG.md](BACKLOG.md)):

```
[1] CVE-LOW  →   [2] CVE-PDF  →   [3] CVE-RQ  →   [4] CVE-DG   →
[5] F3-PICK  →   [6] F3-REVIEW →  [7] F3-PATTERNS  →  [8] F3-IMPL-* →
[9] F-N…     →   [10] WAVE-CLOSE
```

Sprints werden jeweils als Sprint-Prompt-Doc unter [docs/archive/sprint-prompts/](docs/archive/sprint-prompts/) angelegt, dann als frische Sub-Session ausgeführt. Master macht keine Code-Edits.

## Zuletzt durch
- **HYG** (F-002 Pygments narrow-except, F-007 secure_filename(None) guard, F-008 5 Logging-Sites mit exc_info, F-011 `require_service`-Decorator + DE-Microcopy für 3 Services × 6 Endpoints, F-012 dead `if not file:` raus, F-015 Timeout-Konstanten in `app_pkg/config.py` zentralisiert, F-016 Doppel-Log raus, F-017 `isinstance(data, dict)`-Inline-Check an 6 Stellen; +5 Tests, 48/48 grün) — 2026-05-09.
- **SEC** (F-005 Path-Traversal `Path.is_relative_to`, F-006 markdown Backend-Whitelist, F-013 Input-Allowlists für Deepgram/Google-TTS/Gemini; +5 Tests, 43/43 grün) — 2026-05-09, commit `6a18086`.
- **F-2 Cluster II** (audio_converter Sev 2+1 Patterns P13–P21, F-2 strukturell abgeschlossen) — 2026-05-09.
- **Working-Practice-Bootstrap** (CLAUDE.md slim, STATUS.md, BACKLOG.md als Roadmap, Sprint-Prompt-Template) — 2026-05-09.
- **F-2 Cluster I** (audio_converter UX, Sev 4+3 Patterns P1–P12) — 2026-05-03, Live-Smoke ☑.
- **F-1 Hot-Fix** (`document_converter` Jinja2 Generator-Expression Production-Regression) — 2026-05-03.
- **F-1 strukturell abgeschlossen** (alle 14 Patterns + 3 Bug-Tickets, 6 Cluster) — 2026-05-03.
- **Cleanup-Wave Stages 0–7** (cleanup_plan.md, F-001…F-018 strukturell durch) — 2026-05-02/03.

## Methodik
**Master = Dispatch, Sub-Thread = Execute** (Working-Practice umgestellt 2026-05-09 — siehe CLAUDE.md, Sektion *Working Practice*). Master macht keine Code-Edits; jeder Sprint läuft als separater Sub-Thread mit Sprint-Prompt-Doc.
