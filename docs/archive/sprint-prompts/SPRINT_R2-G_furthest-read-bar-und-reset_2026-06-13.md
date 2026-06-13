# Sprint R2-G — Fortschritts-Bar = furthest-read + „Als ungelesen"-Reset (S)

> **Executor-Doc.** Phasen strikt nacheinander, nach jeder Phase **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün. UI-Strings deutsch (Microcopy-Konventionen in CLAUDE.md). Helper aus `static/js/_utils.js` wiederverwenden. **Zeilennummern unten sind Orientierung (ca.) — sie driften, geh über die Funktions-/Element-Namen.**

## Kontext / Warum

User-Feedback (Oliver, 2026-06-13) zur Reader-Fortschrittsanzeige, zwei Punkte:
1. *„der maximale fortschritt wird gespeichert — sprich wenn ich wieder hoch scrolle geht er nicht zurück, wenn ich über den bisherigen fortschritt scrolle, dann erweitert sich fortschritt max"*
2. *„es muss einen button geben um den lese-fortschritt zurückzusetzen oder als ungelesen zu markieren"*

**Punkt 1 ist eine bewusste Revision der R2-F-Entscheidung.** In R2-F (Knoten 8 in `docs/reader_architecture.md`) blieb die Bar absichtlich **Positions-Anzeige** (läuft beim Hochscrollen zurück) und der Max wanderte nur in ein separates „Gelesen"-Label. Oliver will jetzt: die **Bar selbst zeigt den Max** — bleibt beim Hochscrollen stehen, wächst nur vorwärts (Readwise-Verhalten). Der gespeicherte Max ist **serverseitig bereits garantiert** (R2-F Forward-Clamp, [library.py](../../app_pkg/library.py) `api_update_conversion_progress` ~Z.293-320) — hier geht es um die **Anzeige** plus den **Reset-Pfad**, der den Clamp bewusst durchbricht.

**Workshop-Entscheidungen (Master + Oliver 2026-06-13, nicht neu diskutieren):**
- **Bar = furthest-read**, nur gefüllt, **kein** separater Positions-Marker (YAGNI — die aktuelle Position ist durch den Viewport gegeben; ein Marker wäre Folge-Polish).
- **Reset-Button** lebt in der **Reader-Detail-Sidebar** (kleinster Scope, immer erreichbar — nicht in der Abschluss-Leiste, nicht library-list-seitig). Microcopy: **„Als ungelesen markieren"**.
- **„Gelesen"-Label bleibt** (Olivers Wahl), folgt aber jetzt dem Max: erscheint ab 95 %, **verschwindet beim Reset** (heute ist es nur-monoton-an → das ändert sich).
- **Reset-Ziel = `last_read_percent = NULL`** (nicht 0) — sauberer „nie gelesen"-Zustand, identisch zu frisch-ingestet: kein Karten-Balken (Template-Bedingung `> 0`), nicht in der Weiterlesen-Sektion (`> 0`).

## Phase 1 — Backend: Reset-Pfad im Progress-Endpoint (+ Tests)

Datei: `app_pkg/library.py` (`api_update_conversion_progress`, ~Z.293), `tests/test_conversion_progress.py`.

1. Reset über ein **explizites Flag** im bestehenden `PATCH /api/conversions/<id>/progress` (das ist genau die „künftiges Reset braucht ein explizites Flag, um den Forward-Clamp zu umgehen"-Notiz, die R2-F im BACKLOG hinterlegt hat):
   - Body `{"reset": true}` → setzt `last_read_percent = None`, **umgeht den `max(stored, …)`-Clamp**, ignoriert ein evtl. mitgeschicktes `percent`. Response `200 {"success": true, "last_read_percent": null}`.
   - Ohne `reset` (bzw. `reset` falsy): der bestehende Pfad unverändert (Bool-Reject, range-clamp, Forward-Clamp `max(stored, percent)`).
   - `reset` muss **echtes** `true` sein (nicht truthy-Garbage missbrauchbar) — validiere wie der bestehende Bool-Check-Stil im Endpoint; ein nicht-bool `reset` ist ein 400 oder wird ignoriert (entscheide konsistent zur vorhandenen Validierungs-Strenge, begründe im Bericht).
2. Tests (am bestehenden `test_conversion_progress.py`-Stil):
   - Reset setzt NULL **trotz** gespeichertem Wert (vorher 80 → `{reset:true}` → NULL, **nicht** `max(80,0)=80`) — beweist den Clamp-Bypass.
   - Reset + mitgeschicktes `percent` → reset gewinnt (NULL).
   - Normale Updates clampen weiterhin vorwärts (Regression: kleinerer Wert = no-op).
   - `reset:false` / fehlend → normaler Clamp-Pfad.

**Stop + Bericht** (Validierungs-Entscheidung für `reset`, Test-Delta).

## Phase 2 — Frontend: Bar→max, Label→folgt-max, Reset-Button (+ Live-Smoke)

Dateien: `static/js/library_detail.js`, `templates/library_detail.html`, `static/css/style.css`.

1. **Bar zeigt furthest-read** — in `update()` (~Z.1463): die Fill-Breite an `Math.max(percent, maxReached)` hängen statt nur `percent` (~Z.1473). Folge: Hochscrollen lässt die Bar bei `maxReached` stehen; über den Max hinaus wächst `maxReached` (bestehendes Gate `persistArmed && percent > maxReached`, ~Z.1476, **unverändert**) und die Bar mit. `maxReached` ist aus `PageData.lastReadPercent` geseedet → beim Öffnen eines teilgelesenen Docs zeigt die Bar **sofort** den gespeicherten Max, auch vor dem rAF-Arming.
2. **„Gelesen"-Label folgt jetzt dem Max** — `syncReadFlag()` (~Z.1428) von nur-monoton-an auf **beидиректional**: `readFlag.hidden = !(maxReached >= READ_COMPLETE_PERCENT)`. So verschwindet es beim Reset und erscheint wieder, wenn erneut ≥95 gelesen wird. (Element `#reader-read-flag`, ~Z.57.)
3. **Reset-Button in der Detail-Sidebar** — bei der Status-/Lese-Liste-Card-Gruppe (`c-surface--flat`, ~Z.105-131). Da Fortschritt orthogonal zu Ort (R2-C) und Priorität (R2-D) ist: entweder eine eigene kleine „Lese-Fortschritt"-Card oder an die Lese-Liste-Card angehängt — konsistente Card-Optik, deine Wahl, begründe knapp. Button-Text „Als ungelesen markieren" (≤3 Wörter-Regel ist hier für Buttons gedacht; dieser beschreibende Label ist ok, aber halte ihn knapp). Sichtbar unabhängig vom Fortschritt (man kann auch mitten im Doc zurücksetzen).
4. **JS-Handler `resetProgress()`**: PATCH `{reset:true}` über den globalen CSRF-fetch-Wrapper (wie `setStatus`/`finishArchive`). Bei Erfolg: `maxReached = 0`, `update()` neu zeichnen (Bar fällt auf die aktuelle Scroll-Position = `max(currentPercent, 0)`), `syncReadFlag()` (Label aus), Toast „Als ungelesen markiert." (`showToast` aus `_utils.js`). **Kein Auto-Scroll** — der User bleibt, wo er ist. Bei Fehler: `showToast`-Error, State unangetastet. `window.resetProgress` exposen wie die anderen Handler.
5. **Live-Smoke mit echtem Scrollen** (lokale Instanz; Code ist im Image gebacken → vorher `docker compose up -d --build`; Memory-Caveats `reference_scroll_progress_persistence` beachten — der „bleibt beim Hochscrollen"-Kern-Fix ist **nicht** rAF-gated, weil `maxReached` geseedet ist, also wie R2-F Path A direkt beobachtbar):
   - teilgelesenes Doc öffnen → Bar steht beim geseedeten Max; **hochscrollen → Bar bleibt** (Kern-Fix); über den Max scrollen → Bar wächst.
   - Reset → Bar fällt auf aktuelle Position, „Gelesen"-Label weg (falls vorher ≥95), Toast.
   - zurück zur Library → **kein Karten-Fortschrittsbalken** mehr, Doc **nicht** in Weiterlesen (NULL-Zustand verifizieren, ideal per DB-Probe).
   - dark + light, keine Console-Errors, DB/Theme nach Smoke restaurieren.

**Stop + Bericht.**

## Phase 3 — Wrap-up

**Commit-Disziplin wie R2-F: erst den Code committen (eigener Hash), dann den Doc-Wrap als separaten Commit, beide pushen (HEAD == origin halten).**

1. `STATUS.md` + `BACKLOG.md`: R2-G ☑ done mit beiden Hashes. **Die R2-F-BACKLOG-Notiz „Fortschritt zurücksetzen bräuchte explizites Flag" als erledigt markieren** (jetzt gebaut). Vermerken, dass die R2-F-Bar-Entscheidung (Bar = Position) durch R2-G **revidiert** ist (Bar = furthest-read).
2. `docs/reader_architecture.md`: **Knoten 9** — Bar zeigt jetzt furthest-read (explizite Revision der Knoten-8-Entscheidung, mit einem Satz warum: User-Feedback, Readwise-Verhalten), Reset-Pfad (explizites `reset`-Flag umgeht den Forward-Clamp, NULL = ungelesen), „Gelesen"-Label folgt jetzt dem Max statt nur-monoton-an. Decision-Log-Zeile + Workshop-Datum.
3. **Bullet-Guard** vor dem Doc-Commit: `grep -nE '(- \*\*.*){2,}' BACKLOG.md STATUS.md` (Memory `reference_markdown_bullet_delete_newline`).
4. `pytest tests/` final grün.

**Stop + Schluss-Bericht** — inkl. Olivers offenem Schritt: **Mintbox-Deploy** (`git pull` + `docker compose up -d --build`, **keine Migration** — `last_read_percent` existiert seit R2-B; danach Browser-Hard-Reload wegen Static-Cache). Falls R2-F dort noch nicht deployed ist, zieht **ein** Deploy R2-F + R2-G zusammen.

## Out of scope
- Positions-Marker zusätzlich zur furthest-read-Bar (YAGNI; Folge-Polish falls je gewünscht).
- Reset von der Library-Liste aus (per-Karte) — Detail-Sidebar-only in v1 (war die Scope-Wahl).
- Bestätigungs-Dialog vor dem Reset — direkt + Toast reicht (Fortschritt ist billig wiederherstellbar durch erneutes Lesen).
- Lifecycle/Queue/Favoriten — unangetastet (Fortschritt ist die orthogonale dritte/Lese-Achse).
