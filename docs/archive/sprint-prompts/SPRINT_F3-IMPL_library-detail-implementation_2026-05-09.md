# Sprint F3-IMPL — F-3 Implementation `library_detail` (alle 15 Patterns, 3 Sub-Batches)

**Datum**: 2026-05-09

**Ziel**: Alle 15 Patterns aus F3-PATTERNS in einem Sprint implementieren, intern strukturiert als drei Sub-Batches (Foundation → Notion → Polish). Dies ist ein **bewusst aggressiv geschnittener Sprint** — User hat 1-Sprint statt 2- oder 3-Sprint-Split gewählt, in voller Kenntnis dass 15 Patterns über der F-2-Cluster-I-Schmerzgrenze (12) liegen. Die Sub-Batch-Mechanik dient als Schutz vor Überforderung — nach jedem Sub-Batch ein STOP-Punkt mit Bericht.

**Vorbedingung**:
- Pytest 48/48 grün auf `main`. Letzter Code-Touch: F3-PATTERNS (commits `f010cc7` + `42bf15f`, 2026-05-09).
- **Eingabe**: [docs/ui_patterns_library_detail_2026-05.md](docs/ui_patterns_library_detail_2026-05.md). Sub-Thread liest **vor jedem Sub-Batch** die zugehörigen Pattern-Blöcke nochmal.
  - 15 Pattern-IDs (P1-P14 + P15 strukturell).
  - 4 Patterns mit `🔥 Smoke-Pflicht in F3-IMPL`-Tag: P1, P3, P4, P5.
  - 2 Helper-Vorschläge in dedizierter Doc-Sektion: `formatDatetimeLocalNow` (für P5), `renderTagChips` (für P12 vermutlich).
- **Methodik-Vorlagen** (F-2 Cluster I als direkte Mechanik-Vorlage für Multi-Pattern-Sprints):
  - F-2 Cluster I commit `ef78508` (12 Patterns Sev 4+3, holistic-Rewrite mit Foundation → Critical-UX → State-Lifecycle Sub-Batches). Lehre aus F-2-Sub-Thread: „12-in-einem war 'gerade noch handhabbar'" — bei stark verkoppelten Patterns Holistic-Rewrite, bei additiven Patterns sequenziell.
  - F-1 Cluster D commit `e68b6dd` als Vorlage für Backend-Whitelist-Mechanik (falls in P-Patterns für Validation gebraucht).
  - F-1 Cluster A-E + Polish als Vorlagen für isolierte Pattern-Cluster.
- **Test-Coverage**: aktuelle Suite 48/48 grün. Erwartete neue Tests:
  - 1-2 für P1 (Auto-Save-Failure-Banner) — z.B. `test_update_field_failure_shows_banner`.
  - 1-2 für P3 (Delete-Failure-Banner) — z.B. `test_delete_conversion_failure_shows_banner`.
  - Eventuell 1 für P4 (Notion-State-Recovery) wenn Backend-side testbar.
  - Erwartete Final-Anzahl: **49–53 Tests grün**.
- **Memory-Layer-Pflicht-Lese**: `feedback_no_silent_fixes.md` (Bugs während Implementation als separates Bug-Ticket dokumentieren), `feedback_pragmatic_merge.md` (Risiko-Kalibrierung). Plus implizit alle anderen Memory-Einträge.

**Out-of-scope**:
- Weitere F-3-Wellen für andere Features.
- WAVE-CLOSE.
- Bug-Tickets BT7 + BT8 (pure Bug-Tickets aus F-3.2) — separate Aktion, nicht in F3-IMPL.
- Konstitutive Befunde aus F-3.1 (api_create_conversion-Strict-Validation, Notion-MCP-String-Doppelung) — gehören zur library-Welle bzw. Notion-Konsolidierung.

---

## Sub-Batch-Strategie (Pflicht-Reihenfolge)

| Sub-Batch | Patterns | Anzahl | Smoke-Pflicht | Begründung |
|-----------|----------|--------|---------------|------------|
| **A — Foundation + Silent-Failure** | P15, P1, P3, P6, P7, P8, P14 | 7 | P1, P3 | Schafft Helper-Reuse-Voraussetzungen + behebt daily-usage-Schmerz (Auto-Save/Delete silent). Blockiert nicht Sub-Batch B oder C. |
| **B — Notion-Form-Stability** | P4, P5 | 2 | P4, P5 | Eng gescoped (renderNotionFields-Wurzel + Datum-Default). Beide Smoke-Pflicht mit eigener Smoke-Mechanik. |
| **C — Polish + a11y** | P2, P9, P10, P11, P12, P13 | 6 | — | Kosmetisch + a11y, keine Smoke-Pflicht weil statisch verifizierbar. |

**Pflicht-Reihenfolge**: A → B → C, **ohne Auslassen**. STOP-Punkt mit Bericht nach jedem Sub-Batch — Master kann zwischen Sub-Batches korrigieren oder Sprint abbrechen.

**Holistic vs. sequenziell pro Sub-Batch**: Sub-Thread entscheidet pragmatisch. F-2-Cluster-I-Lehre: „bei stark verkoppelten Patterns Holistic-Rewrite, bei additiven Patterns sequenziell". Sub-Batch A hat starke Verkopplung (Helper-Reuse-Foundation), B ist sehr klein (2 Patterns), C ist additiv. Default-Empfehlung: A holistic, B atomic (2-Pattern-Slot), C sequenziell.

---

## Phase 1 — Implementation (drei Sub-Batches mit STOP-Punkten)

### Pre-Flight (vor Sub-Batch A)

1. `pytest tests/` — muss 48/48 grün sein.
2. `git status -s` → clean tree erwartet.
3. **Pattern-Doc + Findings-Doc + Inventur-Doc kurz überfliegen**.
4. **`_utils.js`-Helper-Bestand verifizieren**: `grep -n "^function\|window\\." static/js/_utils.js` zeigt aktuelle Helper.

---

### Sub-Batch A — Foundation + Silent-Failure (7 Patterns)

**Patterns**: P15 (struktureller Vorbedingung-Block, vermutlich Banner-Mountpoint o.ä.), P1 (Auto-Save Failure-Banner, **🔥 Smoke-Pflicht**), P3 (Delete Failure-Banner, **🔥 Smoke-Pflicht**), P6 (DE-Microcopy-Pass, deckt 6 EN-Strings aus Polish-1-BACKLOG mit), P7 (showAlert-statt-Inline-Code), P8 (Toast-Level pro Call-Site), P14 (safeJSON für Redirect-Login-Schutz).

**Mechanik (Holistic-Rewrite empfohlen)**:

1. P15 als erstes (strukturelle Vorbedingung — Banner-Mountpoint o.ä. ins Template, falls noch nicht vorhanden).
2. P6 als zweites (DE-Microcopy-Pass alle 6 EN-Strings aus library_detail.js übersetzt — schafft DE-Foundation für nachfolgende Patterns).
3. P1 + P3 + P7 + P8 + P14 holistic — alle nutzen `_utils.js`-Helper, gemeinsame Code-Pfade.
4. **🔥 Smoke-Pflicht** für P1 + P3 **vor** dem Pattern-Apply:
   - P1: DevTools Network-Throttle (Slow 3G oder Offline) → Notiz-Textarea fokussieren → tippen → Auto-Save sollte fehlschlagen → vor Patch: silent / nach Patch: Failure-Banner. Wenn Smoke zeigt P1 ist nicht reproduzierbar (z.B. weil Auto-Save eigentlich kein silent-fail mehr hat seit irgendeinem Commit): STOP, Master fragen.
   - P3: DevTools Network-Throttle → Delete-Button → vor Patch: silent / nach Patch: Failure-Banner. Bei nicht reproduzierbar: STOP.
5. Tests: 1-2 neue Tests für P1, 1-2 für P3 in `tests/test_library.py`.
6. `pytest tests/` muss grün bleiben (49-52 erwartet).

**Live-Smoke nach Sub-Batch A**:

- DE-Microcopy: alle 6 betroffenen Strings im Browser sichtbar.
- Auto-Save-Pfad: normaler Save zeigt Toast (nicht Banner). Network-Failure zeigt Banner.
- Delete-Pfad: normaler Delete navigiert weg. Network-Failure zeigt Banner.
- showAlert-Reuse: keine raw `alert()`-Calls mehr in library_detail.js (`grep -c "alert(" static/js/library_detail.js` = 0).
- DevTools-Console clean.

**STOP nach Sub-Batch A** — Bericht: welche der 7 Patterns durch, ob Smoke-Pflicht für P1+P3 verifizierbar war, neue Test-Anzahl, ob neue Helper in `_utils.js` angelegt wurden, irgendwelche Mid-Sub-Batch-Drifts.

---

### Sub-Batch B — Notion-Form-Stability (2 Patterns)

**Patterns**: P4 (Notion-State-Wipe-Recovery: Re-Toggle + Target-Switch löschen User-Inputs nicht mehr, **🔥 Smoke-Pflicht**), P5 (Datum-Default lokal statt UTC, **🔥 Smoke-Pflicht**).

**Mechanik (atomic — beide Patterns in einem Apply-Schritt)**:

1. P5 zuerst (Datum-Helper `formatDatetimeLocalNow` aus Helper-Vorschlag-Sektion in `_utils.js` anlegen, dann in `library_detail.js` Datum-Default ersetzen). **🔥 Smoke-Pflicht** vorab: Browser auf der Detail-Seite öffnen, Notion-Panel öffnen, Datum-Input-Default ansehen — UTC-Datum (z.B. "2026-05-09T08:00") vs. erwartetes lokales Berlin-Datum (z.B. "2026-05-09T10:00"). Wenn Smoke zeigt P5 ist nicht reproduzierbar (Datum ist schon lokal): STOP, Master fragen.
2. P4 als zweites (`renderNotionFields`-Logik so ändern dass Re-Toggle/Target-Switch User-Inputs nicht löscht — entweder State-Save vor Re-Render, oder DOM-Diff-statt-Wipe). **🔥 Smoke-Pflicht** vorab: Browser → Notion-Panel öffnen → Inputs füllen → Re-Toggle (Panel zu, dann wieder auf) → vor Patch: Inputs leer / nach Patch: Inputs erhalten. Analog für Target-Switch (Page-Target → Database-Target umschalten). Bei nicht reproduzierbar: STOP.
3. Tests: optional 1 Backend-side-Test für P4 wenn die State-Logik testbar ist.
4. `pytest tests/` muss grün bleiben.

**Live-Smoke nach Sub-Batch B**:

- Datum-Default ist lokales Berlin-Datum.
- Notion-Panel-Re-Toggle erhält Inputs.
- Target-Switch (Page ↔ Database) erhält Inputs.
- Vorhandene Notion-Submit-Pfade weiterhin funktional (smoke gegen real Notion oder gegen mock — je nach Umgebung).

**STOP nach Sub-Batch B** — Bericht: beide Patterns durch, Smoke-Verifikation für P4+P5, Test-Stand, ob `formatDatetimeLocalNow` als neuer Helper in `_utils.js` gelandet ist.

---

### Sub-Batch C — Polish + a11y (6 Patterns)

**Patterns**: P2 (Sidebar-Active-State auf Detail-Seite), P9 (Empty-State + a11y für Tag-Liste), P10 (a11y-Annotations für `<pre>`-Content + suggestions-status), P11 (dirty-indicator für ungespeicherte Edits), P12 (Tag-Chip-Rendering — `renderTagChips`-Helper aus Vorschlag-Sektion), P13 (Page-Title mit Conversion-Name).

**Mechanik (sequenziell, pro Pattern atomic)**:

1. P2: Sidebar-Active-State-Pflege in Template/CSS — sehr klein, statisch verifizierbar.
2. P12 + P9: Tag-Chip-Rendering inkl. Empty-State, eventuell `renderTagChips`-Helper in `_utils.js`.
3. P10: aria-label, aria-live, role-Annotations für `<pre>` und suggestions-status.
4. P11: dirty-indicator-Mechanik (z.B. Unsaved-Changes-Warning beim Tab-Wechsel).
5. P13: Page-Title-Update mit Conversion-Name.

**Live-Smoke nach Sub-Batch C** (statisch — kein Throttle nötig):

- Sidebar zeigt aktiven Detail-Eintrag (oder „neutralen" Zustand, je nach Pattern-Spec).
- Tag-Liste zeigt Empty-State wenn leer, sonst Chip-Rendering.
- Screen-Reader-Tools (oder DevTools-Inspect) bestätigen aria-Annotations.
- Dirty-indicator erscheint nach Edit, verschwindet nach Save.
- Page-Title enthält Conversion-Name (Browser-Tab + History).

**STOP nach Sub-Batch C** — Bericht: alle 6 Patterns durch, ob `renderTagChips` als neuer Helper in `_utils.js` gelandet ist, ob beim Code-Reading weitere Findings/Bugs auffielen die nicht im F-3.2-Findings-Doc waren.

---

## Phase 2 — Verify (gesamter Sprint)

1. `pytest tests/` final grün (49-53 erwartet, je nach P1/P3-Test-Schnitt).
2. `grep -c "alert(" static/js/library_detail.js` → 0.
3. Englische UI-Strings in library_detail.js → 0 (DE-Microcopy-Pass aus P6 deckt das ab).
4. **End-to-End-Smoke** beidpfadig:
   - Standard-Pfad: Conversion öffnen → Notiz tippen (Auto-Save Toast) → Tag editieren → Notion-Submit (Datum-Default lokal, alle Inputs erhalten beim Re-Toggle) → Delete → Sidebar-Active-State korrekt.
   - Fehler-Pfade: Network-Throttle → Auto-Save-Banner + Delete-Banner.
5. DevTools-Console final clean.
6. Cluster I/II/III sind alle drei in `git diff` reflektiert — keine Sub-Batch-Auslassung.

Nach Phase 2: STOP — Bericht. Liste der gesmokten Pfade, Final-Test-Anzahl, etwaige Drift-Befunde.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- **Default: drei Commits, einer pro Sub-Batch** (analog F-1 Cluster A-E mit separaten Commits). Jeder Commit-Subject z.B. „F-3 Sub-Batch A: Foundation + Silent-Failure (P15, P1, P3, P6, P7, P8, P14)".
- Falls Sub-Thread alle Sub-Batches in einem Commit bündeln will (z.B. weil Helper-Konvergenz cross-Sub-Batch lief): kurz im Bericht erwähnen, Default ist drei Commits.
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commits ist Teil des Sprints (Single-User-Single-Instance-Repo). Wenn der Auto-Mode-Classifier blockt: im Phase-3-Bericht erwähnen, Master pusht von Hand.

---

## Stop-Regel

Nach **jeder Phase** UND **nach jedem Sub-Batch** Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für F3-IMPL** (15-Pattern-Sprint, höher als F-2-Cluster-I-Schmerzgrenze):
- Wenn der Sub-Thread während Sub-Batch A merkt, dass die Verkopplung höher ist als gedacht und Sub-Batch B oder C nicht mehr handhabbar wäre: **STOP nach Sub-Batch A**, Master fragen — Sprint kann re-skopt werden auf 1-Sprint-Cluster-I, mit B+C als Folge-Sprints.
- Wenn ein 🔥 Smoke-Pflicht-Pattern (P1/P3/P4/P5) sich als nicht reproduzierbar erweist: STOP, **nicht** silent-fixen — Master entscheidet ob Pattern aus Scope fällt oder Befund neu zu bewerten ist.
- Wenn beim Code-Reading weitere Findings auffallen die nicht in F-3.2 dokumentiert sind: als „aufgefallen, nicht gefixt" in den Bericht — siehe Memory `feedback_no_silent_fixes.md`.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**XL** — 15 Patterns in einem Sprint, 3 Sub-Batches, 4 Smoke-Pflicht-Patterns, 2 neue Helper, 2-4 neue Tests, mehrere Code-Bereiche (library_detail.js, library_detail.html, _utils.js, ggf. style.css und app_pkg/library.py). Über der F-2-Cluster-I-Schmerzgrenze (12) — bewusste User-Entscheidung. Sub-Batch-Mechanik ist die Schutz-Vorkehrung gegen Überforderung.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Code-Reading von `app_pkg/library.py` Auffälligkeiten in den **anderen** Routen auffallen (`library`, `api_*`-Endpoints) die **nicht** zu `library_detail` gehören: kurz im Bericht aufzählen, **nicht** in den Sprint-Diff. Master fold-et bei der `library`-Welle.
- Wenn ein Helper-Vorschlag aus F-3.3 im Verlauf des Sprints überflüssig wird (z.B. weil ein anderer Pattern dieselbe Logik schon gelöst hat): kurz im Bericht aufzählen + in `_utils.js` nicht anlegen.
- Wenn beim Live-Smoke ein Befund aus F-3.1 plötzlich anders aussieht als beschrieben (z.B. Sidebar-Active-State funktioniert doch): im Bericht aufnehmen — code-deduced-Inventur kann irren.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F3-IMPL ☑ done 2026-05-XX → commits `<hash-A>` (Sub-Batch A), `<hash-B>` (Sub-Batch B), `<hash-C>` (Sub-Batch C). Pytest <neue Anzahl>/<neue Anzahl> grün. Live-Smoke clean (Standard + Fehler-Pfade). 15 Patterns implementiert + 6 BTs gefoldet + 2 neue Helper in `_utils.js`. **F-3 strukturell abgeschlossen** für `library_detail`. Verbleibende Sequenz: F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F3-IMPL-*" raus → Erledigt-Liste mit allen 15 Pattern-IDs zur Traceability. Sektion „2. F-N…" rückt auf Position 1 mit Hinweis dass die nächste F-3-Welle ein neues Feature anpicken muss (Master-Entscheidung). Folge-Sprint-Nummern -1.
- **Memory**: nur wenn übertragbare Lehren auftauchen (z.B. „15-Patterns-in-einem-Sprint mit 3 Sub-Batches: was zog gut, was war hart"). Defensiv.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — alle Patterns sind in F-3.3 vollständig spezifiziert, Microcopy steht, Aufwandsschätzung ist da, Sub-Batch-Strategie ist im Sprint-Prompt verankert.)_
