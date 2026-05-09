# BACKLOG

Source-of-Truth für offene Items. Geordnet als **sequenzielle Sprint-Roadmap** — wir arbeiten von oben nach unten ab. Codes (`SEC`, `CVE-LOW`, …) sind stabil und gehen 1:1 in die Sprint-Prompt-Doc-Namen unter [docs/archive/sprint-prompts/](docs/archive/sprint-prompts/).

Prio-Skala: **P0** kritisch (Production kaputt) · **P1** als nächstes dran · **P2** in der Sequenz weiter unten · **P3** nice-to-have / nicht in Sequenz / Reminder.

---

## In Sequenz — Cleanup-Abschluss

### 1. `CVE-RQ` — Job-Queue Major-Bump · P2 · XL
**rq** 1.16.0 → 2.8.0 + **redis** 5.0.1 → 7.4.0 (redis 6.x übersprungen wegen rq-2.x-Constraint `redis!=6`). Worker-Container + Web-Container müssen synchron. **worker.py-Refactor erforderlich** (rq 2.0 hat `Connection`-Context-Manager entfernt — Master-Pre-Flight 2026-05-09 hat das verifiziert und im Sprint-Prompt verankert; Größe daher von L auf XL hochgestuft). Charakterisierungstests für Podcast-Generation (Stage-6, 11 Tests inkl. F-001) sind die wichtigste Verteidigung.

### 2. `CVE-DG` — Deepgram-SDK Major-Bump · P2 · L
**deepgram-sdk** 5.1.0 → 7.0.0 (zwei Majors, Client-Surface reorganisiert). Audio-Transcription-Tests (Stage-6, 4 Tests) als Re-Run-Basis.

---

## In Sequenz — UX-Cascade-Fortsetzung

### 3. `F3-PICK` — F-3 Feature-Wahl + Inventur · P2 · S
Kandidaten: `markdown_converter`, `library`, `library_detail`, `mermaid_converter`, `login`, podcast-flow. Master entscheidet vor Sprint-Start basierend auf Schmerz/Aufwand. Sprint führt dann Stufe 1 (Inventur) durch.

### 4. `F3-REVIEW` — F-3 Heuristik-Review · P2 · S
Stufe 2: Findings-Tabelle Sev 1–4 nach Nielsen H1/H4/H6/H9. Erwartung: ~30–50% Cross-Feature-H4 (Helper-Reuse aus F-1/F-2).

### 5. `F3-PATTERNS` — F-3 Patterns + Microcopy · P2 · S
Stufe 3: Pattern-Blöcke + DE-Microcopy + Top-N Quick-Wins per Impact-Score. Cluster-Vorschlag für Implementation am Ende.

### 6. `F3-IMPL-*` — F-3 Implementation-Cluster · P2 · M-L
1 bis N Sprints je nach Pattern-Menge (F-1 hatte 6 Cluster, F-2 Cluster I bündelte 12). Code-Sprint-Erfahrung: bei stark verkoppelten Patterns + ~40% Cross-Feature-H4 ist Holistic-Rewrite effizienter als sequentielle Edits.

### 7. `F-N…` — Folge-Wellen für Restfeatures · P2 · je L
Pro Restfeature wieder die 3 Methodik-Stufen + Implementation-Cluster. Reihenfolge wird am Ende von F-3 entschieden.

---

## In Sequenz — Wave-Close

### 8. `WAVE-CLOSE` — Strukturelles Closing · P3 · XS
- `docs/cleanup_plan.md` Header auf "fully closed" updaten (alle Findings + Outstanding work durch).
- `OVERSEER_HANDOFF.md` archivieren oder löschen (durch CLAUDE.md/STATUS.md/BACKLOG.md ersetzt).
- Ggf. UX-Cascade-Doku-Convention dokumentieren (für künftige Wellen).

---

## P3 — nicht in Sequenz / Reminder

- **`getUserMedia`-in-`socket.onopen`-Bug** im audio_converter (Permission-Prompt erst nach WS-Handshake, in F-2 Cluster I als Out-of-Scope respektiert). Größe S. Eigener Sprint, Fold-Kandidat in eine künftige Audio-UX-Welle.
- **Englische UI-Strings in [static/js/library.js](static/js/library.js)** (2 Strings) und **[static/js/library_detail.js](static/js/library_detail.js)** (6 Strings). Größe XS. Wahrscheinlich mit F-3-`library`/`library_detail`-Welle gefolded — lasse hier nur als Sichtbarkeit, kein eigener Sprint.
- **Playwright-UI-Tests einführen** — schließt Test-Coverage-Lücke (Charakterisierungstests rendern keine Templates, mocken SDK-Boundaries). Größe L. Eigener Sprint, aber außerhalb der Cleanup-Sequenz — erst wenn UX-Wellen durch sind und stabiler State erreicht ist.
- **Tesseract-NC-33-Workaround** (Mintbox-System-Kontext, kein CONVERTER-Touch) — Reminder, kein aktiver Sprint. Quelle: [/Volumes/MintHome/CLAUDE.md](/Volumes/MintHome/CLAUDE.md), Stand 2026-05-01.

---

## Erledigt (rolling, älteste fallen raus)

- ☑ CVE-PDF (unstructured 0.14.5 → **0.18.32 statt sprint-soll 0.22.26** — gecappt weil unstructured 0.20.2+ Python ≥3.11 fordert und Container-Base auf Python 3.10.12 läuft; pdfminer.six 20221105 → 20251230 + **forced co-bump pdfplumber 0.10.4 → 0.11.9** wegen pdfplumber 0.10.4 Hard-Pin auf pdfminer.six==20221105; **Reihenfolge-Tausch unstructured-zuerst** weil unstructured 0.14.5 `PSSyntaxError` aus pdfminer importierte, das in 20251230 entfernt ist; pytest 48/48 grün im Container, Live-Smoke clean über PDF normal + columnar + DOCX + PPTX + DOCX m. Bildern + Tabellen) — 2026-05-09, commits `c821615` / `85f1caf`
- ☑ CVE-LOW (Pygments 2.18.0 → 2.20.0 / CVE-2026-4539, requests 2.31.0 → 2.33.0 / 3 CVEs, Flask 3.0.3 → 3.1.3 / CVE-2026-27205; Werkzeug zog transitiv auf 3.1.6 nach; pytest 48/48 grün; Live-Smoke gegen rebuilt Mintbox-Container clean) — 2026-05-09, commits `fa98b35` / `0698748` / `73a45b9`
- ☑ HYG (F-002 Pygments narrow-except, F-007 secure_filename(None) guard, F-008 5 Logging-Sites mit exc_info, F-011 `require_service`-Decorator + DE-Microcopy für 6 Endpoints, F-012 dead `if not file:` raus, F-015 Timeout-Konstanten in `app_pkg/config.py` zentralisiert, F-016 Doppel-Log raus, F-017 `isinstance(data, dict)`-Inline-Check an 6 Stellen; +5 Tests, 48/48 grün) — 2026-05-09
- ☑ SEC (F-005 Path-Traversal, F-006 markdown Backend-Whitelist, F-013 Input-Allowlists; +5 Tests, 43/43 grün) — 2026-05-09, commit `6a18086`
- ☑ F-2 Cluster II (audio_converter Sev 2+1, P13–P21; F-2 strukturell abgeschlossen) — 2026-05-09
- ☑ Working-Practice-Bootstrap (CLAUDE.md slim, STATUS.md, BACKLOG.md, Sprint-Prompt-Template) — 2026-05-09
- ☑ F-2 Cluster I (audio_converter Sev 4+3, P1–P12) — 2026-05-03
- ☑ F-1 Hot-Fix (Jinja2 Generator-Expression) — 2026-05-03
- ☑ F-1 (document_converter, alle 14 Patterns + 3 Bug-Tickets, 6 Cluster) — 2026-05-03
- ☑ Cleanup-Wave Stages 0–7 (cleanup_plan.md, F-001…F-018 strukturell durch) — 2026-05-02/03
