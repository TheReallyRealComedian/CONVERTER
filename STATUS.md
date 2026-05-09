# STATUS

**Stand**: 2026-05-09
**Live**: `localhost:5656` im Docker-Stack auf Mintbox + `converter.smallpieces.de`. Pytest 43/43 grün.

## Aktueller Sprint
_Keiner aktiv. SEC ☑ done 2026-05-09 → commit `6a18086`. Master entwirft als nächstes `HYG`._

## Sequenz-Plan
Wir arbeiten den **Cleanup-Abschluss** als sequenzielle Sprint-Roadmap ab (Reihenfolge in [BACKLOG.md](BACKLOG.md)):

```
[1] HYG      →
[2] CVE-LOW  →   [3] CVE-PDF  →   [4] CVE-RQ  →   [5] CVE-DG   →
[6] F3-PICK  →   [7] F3-REVIEW →  [8] F3-PATTERNS  →  [9] F3-IMPL-* →
[10] F-N…    →   [11] WAVE-CLOSE
```

Sprints werden jeweils als Sprint-Prompt-Doc unter [docs/archive/sprint-prompts/](docs/archive/sprint-prompts/) angelegt, dann als frische Sub-Session ausgeführt. Master macht keine Code-Edits.

## Zuletzt durch
- **SEC** (F-005 Path-Traversal `Path.is_relative_to`, F-006 markdown Backend-Whitelist, F-013 Input-Allowlists für Deepgram/Google-TTS/Gemini; +5 Tests, 43/43 grün) — 2026-05-09, commit `6a18086`.
- **F-2 Cluster II** (audio_converter Sev 2+1 Patterns P13–P21, F-2 strukturell abgeschlossen) — 2026-05-09.
- **Working-Practice-Bootstrap** (CLAUDE.md slim, STATUS.md, BACKLOG.md als Roadmap, Sprint-Prompt-Template) — 2026-05-09.
- **F-2 Cluster I** (audio_converter UX, Sev 4+3 Patterns P1–P12) — 2026-05-03, Live-Smoke ☑.
- **F-1 Hot-Fix** (`document_converter` Jinja2 Generator-Expression Production-Regression) — 2026-05-03.
- **F-1 strukturell abgeschlossen** (alle 14 Patterns + 3 Bug-Tickets, 6 Cluster) — 2026-05-03.
- **Cleanup-Wave Stages 0–7** (cleanup_plan.md, F-001…F-018 strukturell durch) — 2026-05-02/03.

## Methodik
**Master = Dispatch, Sub-Thread = Execute** (Working-Practice umgestellt 2026-05-09 — siehe CLAUDE.md, Sektion *Working Practice*). Master macht keine Code-Edits; jeder Sprint läuft als separater Sub-Thread mit Sprint-Prompt-Doc.
