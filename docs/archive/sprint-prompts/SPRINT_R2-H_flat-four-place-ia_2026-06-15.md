# Sprint R2-H — Flache Vier-Orte-IA (Inbox · Lese-Liste · Bibliothek · Archiv)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **273**). Visueller/IA-Sprint → **Live-Smoke nach jedem Frontend-Touch Pflicht** (Tests fangen CSS/Template nicht). Working Practice: Master dispatcht, du executest; du committest jede Phase selbst (eigener Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER` (Mac-lokal = Source-of-Truth; Mintbox-Mount nur Runtime). Zeilennummern driften → geh über Symbol-/Klassennamen.

## Warum (Kontext, knapp)

Die Library hat heute **drei sich überlappende Achsen** als Bedien-Oberfläche: Ort (`lifecycle_status` inbox/later/archive, R2-C) · Lese-Liste-Flag (`queue_position`, R2-D) · Favorit (`is_favorite`). Plus zwei konkurrierende Nav-Listen (Tabs Inbox/Lese-Liste/Bibliothek **und** Ort-Chips Inbox/Später/Archiv). Der User („ich check unsere zwei Listen nicht / wofür ist die Flag / Später=WTF") will **eine flache Achse**.

**Workshop-Entscheidung (Master + Oliver 2026-06-15, gesetzt):** Kollabiere die drei Achsen auf **eine flache Achse mit vier sich gegenseitig ausschließenden Orten** — **Inbox · Lese-Liste · Bibliothek · Archiv** — und zeige sie **identisch** in der Top-Nav (Tabs) **und** auf jeder Karte (Move-Buttons) **und** im Detail-View. „Später", der Favoriten-Stern und das Queue-Flag verschwinden.

## Das Modell (verbindlich)

Jedes Item ist in **genau einem** Ort. Die vier Orte werden **aus den bestehenden Spalten abgeleitet — kein Schema-Touch, keine Migration** (`lifecycle_status` + `queue_position` bleiben die Wahrheit, „place" ist eine abgeleitete Single-Select-Sicht). Ableitung mit dieser **Präzedenz**:

| Ort | Bedingung (Präzedenz von oben) |
|---|---|
| **Archiv** | `lifecycle_status == 'archive'` |
| **Lese-Liste** | sonst, wenn `queue_position IS NOT NULL` (geordnet nach `queue_position`) |
| **Inbox** | sonst, wenn `lifecycle_status == 'inbox'` |
| **Bibliothek** | sonst (das neutrale Regal: `lifecycle_status == 'later'`, `queue_position NULL`) |

**Move-Aktion** (das eine Control setzt die richtige Kombi, hält Exklusivität):

| Klick auf | setzt |
|---|---|
| **Inbox** | `lifecycle_status='inbox'`, `queue_position=NULL` |
| **Lese-Liste** | `queue_position = max(queue_position)+1.0` (ans Ende), `lifecycle_status='later'` |
| **Bibliothek** | `lifecycle_status='later'`, `queue_position=NULL` |
| **Archiv** | `lifecycle_status='archive'`, `queue_position=NULL` *(Archivieren nimmt es von der Lese-Liste — bewusst, im exklusiven Modell zwingend; ändert R2-D's „kein Auto-Dequeue")* |

**Views/Tabs** = die vier Orte (Filter auf die Ableitung oben). **Such-Verhalten**: die Suche ist der globale Finder — eine aktive Suche spannt über **alle Nicht-Archiv-Orte** (Inbox+Lese-Liste+Bibliothek); der Bibliothek-Tab ohne Query zeigt **nur das neutrale Regal** (Ort=Bibliothek). („Alles sehen" = Suche, das war Olis explizite Wahl „alles via Suche".) Such-Scope ist der eine im Smoke fein-justierbare Punkt; Default wie hier.

**Lese-Fortschritt bleibt orthogonal** (kein 5. Ort): Karten-Fortschrittsbalken (R2-B/F/G) + Resume-on-Open + die Detail-„Als ungelesen"-Reset-Card (R2-G) **bleiben**. **Die „Weiterlesen"-Sektion entfällt** (sie überlappte Orte — bricht „flat"; Fortschritt lebt als Balken auf der Karte weiter, wo immer das Item liegt).

## Was raus / was bleibt

- **RAUS**: Favoriten-Stern (`favorite-btn`, `toggleFavorite`) — **`is_favorite`-Spalte bleibt liegen (unsichtbar, nicht löschen, reversibel)**; das `?favorites`-Filter + die UI. Queue-Flag (`queue-btn` Karte / `queue-toggle-btn` Detail). Das per-Karte `status-segmented` (R2-C). Die Ort-Chip-Zeile (das zweite `status-segmented` mit `?status`, R2-E). Die Weiterlesen-Sektion.
- **BLEIBT**: Lese-Liste-**Reorder** (Hoch/Runter `moveQueue` + der R2-D-Swap-über-die-sichtbare-Menge); Fortschritts-Balken + R2-G-Reset; Tag-Leiste + Typeahead (R2-E) im Bibliothek-Tab; Such-/Typ-Filter; die gesegneten Unicode-Glyphen (▲▼ Reorder etc.).

---

## Phase 1 — Backend: das Vier-Orte-Modell

Dateien: `app_pkg/library.py`, `tests/`.
1. **Move-Aktion** — ein atomarer „setze Ort"-Pfad, der die Kombi-Tabelle oben umsetzt (Empfehlung: neuer `POST /api/conversions/<id>/place` mit `{place: 'inbox'|'leseliste'|'bibliothek'|'archiv'}`, owner-scoped, 404/400 wie üblich). Subsumiert `setStatus` (PUT lifecycle) + den add/remove-Teil von `api_update_conversion_queue`. Der **Hoch/Runter-Swap bleibt** (eigener Pfad oder im selben Endpoint, über die sichtbare Lese-Liste-Menge — R2-D-Swap-Fix nicht brechen).
2. **Views** (`library()`): `view` ∈ `{inbox, leseliste, bibliothek, archiv}` über die Präzedenz-Ableitung; `bibliothek` ohne `search` = nur Ort=Bibliothek; **`search` spannt über alle Nicht-Archiv-Orte**. Den `?status`-Ort-Filter (R2-E) + `?favorites` entfernen; `pagination_args` entsprechend abspecken.
3. **Inbox-Count**-Badge weiter mitliefern (jetzt = Ort=Inbox). Lese-Liste-Reorder-Logik unverändert lassen.
4. **Tests**: Ableitung jedes Orts (inkl. Präzedenz: queued+inbox→Lese-Liste, archiv schlägt alles); Move-Aktion setzt die Kombi + hält Exklusivität (Archiv nullt queue; Lese-Liste appendet); Suche spannt alle Nicht-Archiv-Orte; bibliothek-ohne-query = neutrales Regal; Reorder weiter grün. Bestehende `?status`/`?favorites`/queue-add/remove-Tests umbauen statt nur löschen. `pytest` grün ≥ Baseline.

**Stop + Bericht** (Ableitungs-Präzedenz, Move-Kombis, Test-Delta).

## Phase 2 — Frontend: das eine Vier-Orte-Control (Liste + Detail)

Das **identische** 4-Buttons-Control oben (Tabs) und auf jeder Karte und im Detail. Single-Select, aktiver Ort gedrückt (das eine gepresste Element der Gruppe — Elevation-Budget). Sub-batchbar 2A Liste / 2B Detail.

**2A Liste** (`templates/library.html`, `static/js/library.js`, CSS):
- **Tab-Leiste**: vier Tabs **Inbox · Lese-Liste · Bibliothek · Archiv** (Archiv neu dazu; Ort-Chip-Zeile raus). Aktiver Tab `aria-current`.
- **Karte**: das per-Karte `status-segmented` + `queue-btn` + `favorite-btn` ersetzen durch **ein** 4-Orte-Move-Control (gleiche Optik wie die Tabs, aktiver Ort gedrückt, Klick = `place`-Move → wie nötig reload/in-place wie das bestehende `setStatus`-Reload-Gate, Memory `reference_reorder_over_filtered_set`: wenn das Item den sichtbaren View verlässt → reload, sonst in-place). Lese-Liste-Karten behalten die Reorder-Rail.
- Favoriten-Stern + Queue-Flag-Markup + Weiterlesen-Sektion entfernen.
- **Live-Smoke** (echte Klicks, dark+light, 0 Console-Errors): alle 4 Tabs; Move zwischen allen 4 Orten von einer Karte (inkl. Inbox→Lese-Liste→Bibliothek→Archiv→zurück); Reorder in der Lese-Liste; Archiv nimmt von der Lese-Liste; Suche spannt über Orte; Tag-Leiste im Bibliothek-Tab; Fortschrittsbalken sichtbar.

**Stop + Bericht.** (Dann 2B.)

**2B Detail** (`templates/library_detail.html`, `static/js/library_detail.js`, CSS):
- Die Detail-Sidebar `status-segmented--full` + `queue-toggle-btn` + `favorite-btn` durch **dasselbe** 4-Orte-Control ersetzen (synchron zur Liste). Die **R2-G „Lese-Fortschritt"/„Als ungelesen"-Card bleibt** (orthogonal). `resetProgress` unangetastet.
- **Live-Smoke**: Ort-Move aus dem Detail (alle 4), R2-G-Reset weiter funktional, Abschluss-Leiste/„Gelesen"-Label weiter da, dark+light, 0 Console-Errors.

**Stop + Bericht.**

## Phase 3 — Wrap-up

**Code pro Phase committen, Doc-Wrap separat, alle pushen.**
1. `STATUS.md` + `BACKLOG.md`: R2-H ☑ done mit Hashes; festhalten — die drei Achsen sind auf eine flache 4-Orte-Achse kollabiert, „Später"/Favorit/Queue-Flag retired, vier Tabs synchron zu den Karten-Controls. Den **P3-Reminder „Favoriten-Retire-Frage"** als durch R2-H erledigt markieren (Stern ist weg, Spalte liegt brach). R2-B-Weiterlesen/Resume-95%-Reminder ggf. anpassen (Weiterlesen-Sektion entfällt).
2. `docs/reader_architecture.md`: neues Kapitel — **das flache Vier-Orte-Modell** ersetzt das R2-C/D/E-Drei-Achsen-Bild (Ableitung aus den bestehenden Spalten, Move-Kombis, Suche=globaler Finder, Fortschritt bleibt orthogonal, Archiv dequeued). Decision-Log + Datum.
3. **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md STATUS.md`), finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Olis offener Schritt: **Mintbox-Deploy** (`git pull` + `docker compose up -d --build`, **keine Migration**, danach Browser-Hard-Reload). Hinweis: der `converter-mcp` liest weiter sauber (die GET-API + `recorded_at` sind unberührt; `lifecycle_status` bleibt als Spalte, nur die UI-Achsen sind kollabiert).

## Out of scope
- Schema-Migration / neue Spalte (die vier Orte sind abgeleitet).
- `is_favorite`-Spalte/Daten löschen (nur UI raus).
- Read-Progress-Mechanik ändern (bleibt 1:1).
- `converter-mcp` / GET-API-Shape ändern (unberührt; `lifecycle_status`/`queue_position` bleiben in `to_dict`/Summary).
