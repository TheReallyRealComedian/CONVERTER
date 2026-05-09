# STATUS

**Stand**: 2026-05-09
**Live**: `localhost:5656` im Docker-Stack auf Mintbox + `converter.smallpieces.de`. Pytest 48/48 grün.

## Aktueller Sprint
_Keiner aktiv. CVE-RQ ☑ done 2026-05-09 → commits `513844e` (redis-py 5.0.1 → 7.4.0; redis 6.x übersprungen wegen rq-2.x-Constraint `redis!=6`), `3cfebbe` (rq 1.16.0 → 2.8.0 + atomic worker.py-Connection-Refactor + app_pkg/podcasts.py Job.id-Patch + tests/conftest.py mock_job-Fixture-Sync). Pytest 48/48 grün im Container, Live-Smoke clean inkl. F-001 narrow (NoSuchJobError → 404) und F-001 broad (ConnectionError bei redis-down → 500). Verbleibende Sequenz: CVE-DG → F3-* → WAVE-CLOSE._

## Sequenz-Plan
Wir arbeiten den **Cleanup-Abschluss** als sequenzielle Sprint-Roadmap ab (Reihenfolge in [BACKLOG.md](BACKLOG.md)):

```
[1] CVE-DG   →
[2] F3-PICK  →   [3] F3-REVIEW →  [4] F3-PATTERNS  →  [5] F3-IMPL-* →
[6] F-N…     →   [7] WAVE-CLOSE
```

Sprints werden jeweils als Sprint-Prompt-Doc unter [docs/archive/sprint-prompts/](docs/archive/sprint-prompts/) angelegt, dann als frische Sub-Session ausgeführt. Master macht keine Code-Edits.

## Zuletzt durch
- **CVE-RQ** (redis-py 5.0.1 → 7.4.0 — redis 6.x übersprungen wegen rq-2.x-Constraint `redis!=6,>=3.5`; rq 1.16.0 → 2.8.0 mit atomic worker.py-Refactor wegen `Connection`-Removal in rq 2.0; während Live-Smoke API-Drift entdeckt: `Job.get_id()` in rq 2.x entfernt, 2-Zeilen-Patch in `app_pkg/podcasts.py` plus 1-Zeile-Sync in `tests/conftest.py` weil MagicMock-Auto-Attribute den Drift in der Test-Suite versteckt hatten; pytest 48/48 grün im Container; Live-Smoke clean inkl. F-001 narrow → 404 und F-001 broad → 500; size mid-sprint von L auf XL hochgestuft) — 2026-05-09, commits `513844e` / `3cfebbe`.
- **CVE-PDF** (unstructured 0.14.5 → 0.18.32 statt 0.22.26 — gecappt weil 0.20.2+ Python ≥3.11 fordert, Container-Base auf 3.10.12; pdfminer.six 20221105 → 20251230 + forced co-bump pdfplumber 0.10.4 → 0.11.9 wegen `pdfplumber 0.10.4` Hard-Pin; Reihenfolge unstructured-zuerst weil unstructured 0.14.5 hard-importierte `PSSyntaxError` aus pdfminer, das in 20251230 entfernt ist; pytest 48/48 grün im Container, Live-Smoke clean über PDF normal + columnar + DOCX + PPTX + DOCX m. Bildern + Tabellen) — 2026-05-09, commits `c821615` / `85f1caf`.
- **CVE-LOW** (Pygments 2.18.0 → 2.20.0 / CVE-2026-4539; requests 2.31.0 → 2.33.0 / 3 CVEs; Flask 3.0.3 → 3.1.3 / CVE-2026-27205, Werkzeug zieht transitiv auf 3.1.6 nach; pytest 48/48 grün, Live-Smoke gegen rebuilt Mintbox-Container clean) — 2026-05-09, commits `fa98b35` / `0698748` / `73a45b9`.
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
