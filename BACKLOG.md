# BACKLOG

Source-of-Truth für offene Items. Geordnet als **sequenzielle Sprint-Roadmap** — wir arbeiten von oben nach unten ab. Codes (`SEC`, `CVE-LOW`, …) sind stabil und gehen 1:1 in die Sprint-Prompt-Doc-Namen unter [docs/archive/sprint-prompts/](docs/archive/sprint-prompts/).

Prio-Skala: **P0** kritisch (Production kaputt) · **P1** als nächstes dran · **P2** in der Sequenz weiter unten · **P3** nice-to-have / nicht in Sequenz / Reminder.

---

## In Sequenz — UX-Cascade-Fortsetzung

### 1. `F3-REVIEW` — F-3 Heuristik-Review (`library_detail`) · P2 · S
Stufe 2: Findings-Tabelle Sev 1–4 nach Nielsen H1/H4/H6/H9. Eingabe: [docs/ui_inventory_library_detail_2026-05.md](docs/ui_inventory_library_detail_2026-05.md) (18 Befunde, davon 6 Bug-Ticket-Kandidaten + 11 Finding-only + 1 Pre-Existing). Erwartung: ~30–50% Cross-Feature-H4 (Helper-Reuse aus F-1/F-2). Master macht ggf. Live-Walkthrough-Nachreichung für die 10 Code-deduced-Punkte aus der Inventur-Doc — Schwerpunkt Befund 1 (Sidebar-Active-State auf Detail-Seite, vermutet fehlend).

### 2. `F3-PATTERNS` — F-3 Patterns + Microcopy · P2 · S
Stufe 3: Pattern-Blöcke + DE-Microcopy + Top-N Quick-Wins per Impact-Score. Cluster-Vorschlag für Implementation am Ende.

### 3. `F3-IMPL-*` — F-3 Implementation-Cluster · P2 · M-L
1 bis N Sprints je nach Pattern-Menge (F-1 hatte 6 Cluster, F-2 Cluster I bündelte 12). Code-Sprint-Erfahrung: bei stark verkoppelten Patterns + ~40% Cross-Feature-H4 ist Holistic-Rewrite effizienter als sequentielle Edits.

### 4. `F-N…` — Folge-Wellen für Restfeatures · P2 · je L
Pro Restfeature wieder die 3 Methodik-Stufen + Implementation-Cluster. Verbleibende Kandidaten: `markdown_converter`, `library` (List-View), `mermaid_converter`, `login`, podcast-flow. Reihenfolge wird am Ende von F-3 entschieden.

---

## In Sequenz — Wave-Close

### 5. `WAVE-CLOSE` — Strukturelles Closing · P3 · XS
- `docs/cleanup_plan.md` Header auf "fully closed" updaten (alle Findings + Outstanding work durch).
- `OVERSEER_HANDOFF.md` archivieren oder löschen (durch CLAUDE.md/STATUS.md/BACKLOG.md ersetzt).
- Ggf. UX-Cascade-Doku-Convention dokumentieren (für künftige Wellen).

---

## P3 — nicht in Sequenz / Reminder

- **`getUserMedia`-in-`socket.onopen`-Bug** im audio_converter (Permission-Prompt erst nach WS-Handshake, in F-2 Cluster I als Out-of-Scope respektiert). Größe S. Eigener Sprint, Fold-Kandidat in eine künftige Audio-UX-Welle.
- **Englische UI-Strings in [static/js/library.js](static/js/library.js)** (2 Strings). Größe XS. Wird mit der `library`-(List-View-)Welle (`F-N…`) gefolded — kein eigener Sprint. (Die 6 Strings in `library_detail.js` sind jetzt in [docs/ui_inventory_library_detail_2026-05.md](docs/ui_inventory_library_detail_2026-05.md) Befund 4 als Pre-Existing-Item festgehalten und werden mit F3-IMPL-* gefolded.)
- **Playwright-UI-Tests einführen** — schließt Test-Coverage-Lücke (Charakterisierungstests rendern keine Templates, mocken SDK-Boundaries). Größe L. Eigener Sprint, aber außerhalb der Cleanup-Sequenz — erst wenn UX-Wellen durch sind und stabiler State erreicht ist.
- **Tesseract-NC-33-Workaround** (Mintbox-System-Kontext, kein CONVERTER-Touch) — Reminder, kein aktiver Sprint. Quelle: [/Volumes/MintHome/CLAUDE.md](/Volumes/MintHome/CLAUDE.md), Stand 2026-05-01.

---

## Erledigt (rolling, älteste fallen raus)

- ☑ F3-PICK (UX-Inventur `library_detail`, Stufe 1 der Duan-Kaskade — analog F-1.1 / F-2.1; 21 interaktive Elemente in 32 Tabellenzeilen kartiert, ~12 fehlende States identifiziert, ~7 vermutete Code↔live-Divergenzen, 18 separate Befunde mit Code-Ankern; Disposition vorläufig: 6 Bug-Ticket-Kandidaten + 11 Finding-only + 1 Pre-Existing-Erwähnung; Sub-Thread ohne Browser-Access → 10 Live-Walkthrough-Lücken zur Master-Nachreichung gelistet, Schwerpunkt Befund 1 Sidebar-Active-State auf Detail-Seite; Output [docs/ui_inventory_library_detail_2026-05.md](docs/ui_inventory_library_detail_2026-05.md)) — 2026-05-09, commit `e9cfd1a`
- ☑ CVE-DG (deepgram-sdk 5.1.0 → 7.1.0 — drei Majors über 6.0/7.0/7.1, **kein Service-Refactor nötig** weil `client.listen.v1.media.transcribe_file` mit allen 8 verwendeten Kwargs surface-stable ist; die 6.0/7.0-Major-Breakages liegen WebSocket-seitig (regenerated WS-Clients, `send_media`-bytes, `deepgram.listen.v1.types`-Reorg), das nutzt CONVERTER nicht; `create_temporary_key` returnt nur `self.api_key` ohne SDK-Call; Pre-Flight: `requires_python<4.0,>=3.10` ✓, keine Hard-Pin-Konflikte mit installierten Packages; pytest 48/48 grün im Container, Container-SDK-Smoke clean (echte `client.listen.v1.media`-Surface, kein MagicMock-Pseudo-Pass), Live-Roundtrip 25.9s real-speech → 334 Char + Chunking-Pfad 34.6min concat → 2 Chunks à HTTP 200 → 26621 Char Final-Transcript, beide Container-Logs clean; Live-Tab-Browser-WS-Pfad nicht durchgeführt mangels Browser-Access — `create_temporary_key` ist tested-trivial) — 2026-05-09, commit `74589e3`
- ☑ CVE-RQ (redis-py 5.0.1 → 7.4.0 — redis 6.x übersprungen wegen rq-2.x-Constraint `redis!=6,>=3.5`; rq 1.16.0 → 2.8.0 mit atomic worker.py-Refactor wegen `Connection`-Removal in rq 2.0; während Live-Smoke API-Drift entdeckt: `Job.get_id()` in rq 2.x entfernt → 2-Zeilen-Patch in `app_pkg/podcasts.py` plus 1-Zeile-Sync in `tests/conftest.py` weil MagicMock-Auto-Attribute den Drift in der Test-Suite versteckt hatten; pytest 48/48 grün; Live-Smoke clean inkl. F-001 narrow → 404 und F-001 broad → 500; Größe mid-sprint von L auf XL hochgestuft) — 2026-05-09, commits `513844e` / `3cfebbe`
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
