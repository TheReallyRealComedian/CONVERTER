# Sprint R2-E — Library-IA „Readwise-3er" + Tag-Leiste + Tag-Daten (L)

> **Executor-Doc.** Phasen strikt nacheinander, nach jeder Phase **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` muss grün sein, bevor du beginnst. UI-Strings deutsch (Microcopy-Konventionen in CLAUDE.md). Helper aus `static/js/_utils.js` wiederverwenden.

## Kontext / Warum

Live-Befund nach dem Mintbox-Deploy von R2-C+R2-D (Olivers Worte): *„ich komm mit unserem archiv, später inbox vs Alle, Lese-Liste, Weiterlesen nicht klar — das ist total verwirrend - wie kommt was wohin, wo finde ich was."* Dazu: Die AI-Newsletter (NL1/NL2) haben ~100 Tags in die Chip-Leiste gespült, darunter **kaputte Tags** mit Markdown-Artefakten (`** [anthropic`, `** ai-agenten`, `** nvidia`, …) und viele Quasi-Duplikate (`ki-ethik`/`ai-ethik`).

Die drei Daten-Achsen bleiben unangetastet und orthogonal (siehe `docs/reader_architecture.md`):
- **Ort** `lifecycle_status` (inbox/later/archive, R2-C)
- **Lese-Zustand** `last_read_percent` (R2-B)
- **Priorität** `queue_position` (R2-D)

Was sich ändert, ist die **Präsentation**: heute zwei konkurrierende Leisten (View-Switcher Alle/Lese-Liste/Weiterlesen **plus** Status-Chips Inbox/Später/Archiv). Workshop-Entscheidung (2026-06-12, Oliver): **„Readwise-3er"** — genau drei Top-Level-Ziele.

## Ziel-IA (entschieden, nicht neu diskutieren)

**Eine Tab-Leiste mit drei Tabs**, ersetzt View-Switcher UND die Top-Level-Status-Chip-Zeile:

| Tab | Beantwortet | Filter-Semantik |
|---|---|---|
| **Inbox** | „Was ist neu / untriagiert?" | `lifecycle_status='inbox'` **AND** `queue_position IS NULL` |
| **Lese-Liste** | „Was lese ich jetzt, in welcher Reihenfolge?" | R2-D-Queue (geordnet, Archiv ausgeblendet) + „Weiterlesen"-Sektion oben |
| **Bibliothek** | „Wo finde ich alles?" | Default-View: alles + Suche/Filter; Status wird hier zum **Ort-Filter** |

Kern-Semantik der Triage (wichtigste Design-Entscheidung, bitte exakt so umsetzen):
- **Inbox zeigt nur Untriagiertes**: Status inbox **und** nicht in der Queue. „Zur Lese-Liste" (= Queue-Add) lässt `lifecycle_status` unverändert — das Item verschwindet trotzdem aus der Inbox, weil es jetzt gequeued ist. Wird es später de-queued und ist immer noch `inbox`, **fällt es zurück in die Inbox** (bewusst: un-gequeued + untriagiert = wieder Triage).
- „Später" / „Archiv" auf der Karte setzen wie bisher den Status (bestehende R2-C-API, keine Änderung).
- **Keine Auto-Kopplung der Achsen** (kein Auto-Dequeue bei Archiv etc.) — R2-D-Entscheidung bleibt.

**Weiterlesen-Sektion** (in Lese-Liste, oberhalb der Queue):
- Items mit `0 < last_read_percent < 95`, **nicht** in der Queue (`queue_position IS NULL`), **nicht** archiviert.
- Sortierung 1:1 vom bisherigen `view=reading` übernehmen (im Code nachschauen, nicht neu erfinden).
- Sektion nur rendern, wenn nicht leer; keine Kappung (praktisch selten >3 Items).
- Queue-Items mit Fortschritt brauchen keine Doppelung — ihr Karten-Fortschrittsbalken (R2-B) zeigt das schon.

**Bibliothek**:
- Default beim Aufruf von `/library` ohne Params.
- Such-Panel (Typ/Suche/Favoriten/Sort/per-page), Tag-Leiste und der **Ort-Filter** (bisherige Status-Chips Inbox/Später/Archiv, umgezogen, Label „Ort:") leben **nur hier**. Inbox- und Lese-Liste-Tab sind kuratierte Listen ohne Filter-Panel.
- Tag-Klick auf Karten (alle Views) navigiert wie bisher in die Tag-gefilterte Bibliothek.

**View-Param**: `view=inbox` (neu) · `view=queue` (= Lese-Liste, bleibt) · default = Bibliothek. **`view=reading` entfällt ersatzlos** (Single-User-App, keine Kompat nötig — unbekannte view-Werte fallen wie bisher auf default).

**Inbox-Tab-Badge**: Count der untriagierten Items am Tab, nur wenn > 0.

---

## Phase 1 — Backend (`app_pkg/library.py` + Tests)

1. View-Param-Semantik umbauen: `view in ('inbox', 'queue')`, sonst default. Inbox-Filter wie oben (Status inbox + `queue_position IS NULL`).
2. `view=queue` liefert zusätzlich die Weiterlesen-Sektion-Daten (separate Query, Sortierung vom alten `view=reading` übernehmen) ans Template.
3. `view=reading`-Zweig entfernen (inkl. zugehörige `pagination_args`-Pfade prüfen).
4. Inbox-Count (untriagiert) für das Tab-Badge mitliefern — in **jedem** View (die Tab-Leiste ist immer sichtbar).
5. Tag-Daten für Phase 3 vorbereiten: `available_tags` um Nutzungs-Count erweitern (Count über die `conversion_tags`-Junction, absteigend, tie-break alphabetisch). Template-Kontrakt: Top-N-fähige, count-sortierte Liste.
6. Tests: bestehende `view=reading`-Tests umbauen statt löschen (die Filter-Logik lebt in der Weiterlesen-Sektion weiter); neue Tests für Inbox-Semantik (gequeuedes inbox-Item ist NICHT in view=inbox; de-queue → wieder drin) und Inbox-Count. `pytest tests/` grün.

**Stop + Bericht** (was geändert, Test-Delta, offene Risiken).

## Phase 2 — Frontend IA (`templates/library.html`, `static/js/library.js`, CSS)

1. Tab-Leiste: Inbox (+Badge) · Lese-Liste · Bibliothek. Aktiver Tab klar erkennbar, `aria-current`.
2. Such-Panel + Ort-Filter-Chips („Ort: Inbox · Später · Archiv") + Tag-Leiste nur im Bibliothek-View rendern. Die Ort-Chips dürfen optisch **nicht** wie Tag-Chips aussehen (das war Teil der Verwirrung) — abgesetzte Filter-Gruppe mit Label.
3. Inbox-View: Triage-Affordances direkt auf der Karte sicherstellen — „Lese-Liste" (Queue-Toggle, R2-D), „Später", „Archiv" (R2-C). Vorhandene Buttons reichen funktional; prüfe, dass sie in der Inbox prominent/verständlich sind (Microcopy, Reihenfolge), kein Neubau.
4. Lese-Liste-View: Weiterlesen-Sektion (Überschrift „Weiterlesen") über der Queue; Queue-Darstellung (#Rang, Hoch/Runter, R2-D) unverändert.
5. Client-Verhalten prüfen: `setStatus`-Reload-Gate (R2-D) und Queue-Reload-Verhalten gegen die neuen Views — `window.PageData.currentView` Werte konsistent halten. Inbox-View braucht analoge Behandlung: Triage-Aktion entfernt die Karte aus der sichtbaren Menge → Anzeige-Konsistenz (Reload oder sauberes Entfernen, Muster aus R2-D: Reorder/Mutation über die sichtbare Menge, abgeleitete States nicht stale lassen).
6. Empty-States (deutsch, max 3 Sätze): Inbox leer („alles triagiert"-Ton), Lese-Liste leer (R2-D-Text prüfen/anpassen: Hinweis, dass man aus Inbox/Bibliothek befüllt), Weiterlesen-Sektion leer = nicht rendern.
7. **Live-Smoke** (Templates geändert → Pflicht, Test-Suite fängt das nicht): alle drei Tabs, Triage-Flow Inbox→Lese-Liste→(de-queue)→zurück in Inbox, Ort-Filter in Bibliothek, dark + light. Browser-Smoke mit echten Klicks.

**Stop + Bericht** (inkl. Screenshot-Beschreibung der drei Tabs).

## Phase 3 — Tag-Leiste (Bibliothek)

Workshop-Entscheidung: **Top-N + Aufklappen UND Typeahead.**

1. Top-15-Chips nach Nutzungs-Count (Phase-1-Daten). Der aktive `?tag=`-Filter-Chip ist **immer** sichtbar, auch wenn nicht Top-15.
2. „+N weitere"-Toggle: expandiert die volle, alphabetisch sortierte Chip-Liste; collapsed ist Default; Zustand muss nicht persistiert werden.
3. Typeahead: Eingabefeld „Tag filtern …" mit nativer `<datalist>` über alle Tag-Namen; Auswahl/Enter navigiert auf `?tag=<name>` (gleiche URL-Mechanik wie Chip-Klick, `pagination_args` respektieren). Vanilla JS, keine Library.
4. Smoke: Chip-Klick, Typeahead-Auswahl, „+N weitere", aktiver Nicht-Top-15-Tag, dark + light.

**Stop + Bericht.**

## Phase 4 — Tag-Daten-Bereinigung (CONVERTER-seitig)

Workshop-Entscheidung: nur CONVERTER-seitig; die Quasi-Duplikate (`ki-ethik`/`ai-ethik`) bleiben vorerst (manuell über „Tags verwalten" mergebar). Ein email-automation-Vokabular-Brief (NL3) ist explizit **out of scope**.

1. **Normalisierung zentral in `Tag.get_or_create`** (models.py) härten — ein Ort für alle Pfade (Ingest, UI, künftige). Zuerst lesen, was dort heute schon normalisiert wird (lowercase?), dann ergänzen: Markdown-Artefakte strippen (`*`, `` ` ``, führende/schließende `[`/`]`), trim, Mehrfach-Whitespace → ein Space. Ergebnis leer → Aufrufer muss skippen können (Rückgabe-Kontrakt klären, `app_pkg/ingest.py:184-189` ist lenient und skippt heute schon nicht-Strings).
2. Tests: `"** [anthropic"` → `anthropic`; `"  KI-Agenten  "` → normalisiert; nur-Artefakt-String → skip ohne Tag-Anlage; Ingest-Pfad-Test mit kaputten Topics im Payload.
3. **One-off-Script `scripts/cleanup_tags.py`** für den Bestand: läuft im Container, `--dry-run` ist Default (echter Lauf nur mit `--apply`), idempotent.
   - Inventur: alle Tags, deren Name sich unter der neuen Normalisierung ändern würde, mit Nutzungs-Counts listen.
   - Apply: Name normalisieren; kollidiert das Ergebnis mit einem bestehenden Tag → **mergen** (Junction-Rows umhängen, Duplikat-Paare dabei abfangen, leeres Duplikat-Tag löschen); wird der Name leer → Tag samt Junction-Rows löschen.
   - Ausgabe: pro Tag eine Zeile alt → neu / merged-into / deleted.
4. Lokal verifizieren: kaputte Tags in der Dev-DB erzeugen (über den gehärteten Pfad geht das nicht mehr — direkt per SQL/Shell anlegen), dry-run + apply + zweiter apply (no-op) durchspielen.
5. **Mintbox-Lauf ist NICHT Teil dieses Sprints** — der echte Bestand liegt auf der Mintbox und das Script kommt erst per Deploy (`git pull` + `docker compose up --build`) in das Prod-Image. In den Phase-Bericht gehören die fertigen Kommandos für Oliver:
   ```
   ssh mintbox
   cd /home/oliver/CODE/CONVERTER && git pull && docker compose up -d --build
   docker compose exec markdown-converter python scripts/cleanup_tags.py            # dry-run, Bericht lesen
   docker compose exec markdown-converter python scripts/cleanup_tags.py --apply
   ```
   (Pfad auf der Mintbox ggf. verifizieren, nicht blind übernehmen.)

**Stop + Bericht** (Inventur-Logik, Merge-Fälle, dry-run-Output lokal).

## Phase 5 — Wrap-up

1. `STATUS.md` + `BACKLOG.md`: R2-E als erledigt mit Commit-Hashes; offene Real-Welt-Aktion „Mintbox-Deploy + Tag-Cleanup-Run" als Olivers Schritt notieren.
2. `docs/reader_architecture.md`: neues IA-Kapitel — die drei Tabs, die Inbox-Triage-Semantik (`inbox AND unqueued`, Rückfall-Verhalten), Wegfall `view=reading`, Tag-Leisten-Mechanik (Top-N + Typeahead), zentrale Tag-Normalisierung.
3. Optionaler Memory-Eintrag, falls eine verallgemeinerbare Lehre entstand (nicht erzwingen).
4. **Bullet-Guard** vor dem Doc-Commit: `grep -nE '(- \*\*.*){2,}' BACKLOG.md` (Memory `reference_markdown_bullet_delete_newline`).
5. `pytest tests/` final grün. Commits pro Phase waren Pflicht; hier nur noch Doc-Commit.

**Stop + Schluss-Bericht.**

## Out of scope
- Auto-Kopplung der Achsen (Archiv ↔ Queue ↔ Fortschritt) — bleibt wie in R2-C/R2-D entschieden.
- Quasi-Duplikat-Merge per Heuristik (`ki-ethik` vs `ai-ethik`) — manuell via „Tags verwalten".
- email-automation-Seite (Topics-Vokabular) — ggf. späterer NL3-Brief.
- Reader-/Detail-View (`library_detail.*`) — läuft parallel als R2-F, **nicht anfassen** (einzige Ausnahme: keine).
- `app_pkg/library.py`-Progress-Endpoint (`api_update_conversion_progress`) — gehört R2-F.
