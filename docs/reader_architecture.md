# Reader-Architecture — Entscheidungs-Memo

**Stand**: 2026-06-04
**Workshop-Datum**: 2026-05-25 (Master-Workshop nach R1-A done); R2-B-Workshop 2026-05-29; READER-FIX-B Anker-Korrektur 2026-05-31; R2-C-Workshop 2026-06-03; R2-D-Workshop 2026-06-04
**Status**: Aktive Referenz für R2 (☑ A/B/C/D komplett — die drei Reader-Achsen Ort · Lese-Zustand · Priorität sind voll getrennt) + R3 + R4-LEARN Sprints — nicht archiviert.

---

## Kontext

CONVERTER wandert vom Multimedia-Konverter zum **Readwise-Reader-Replacement**. Bestehende
Features (Markdown→PDF, Document→Markdown, Audio-Transcript, Podcast-Generation, Notion-Send)
bleiben — die Library wird zum Reader-Layer obendrauf. Strategischer Pivot dokumentiert in
Master-Thread vom 2026-05-25, R1-A Foundation steht (commit `c84e469`).

Dieses Memo persistiert die Architektur-Entscheidungen aus dem Master-Workshop vom selben Tag,
damit folgende Sprints (R1-B Highlight-Cluster, R2 Library-Power, R3 Web-Article-Save) sich auf
einen Plan referenzieren statt jeweils neu zu diskutieren.

## Migration-Realität

Das Projekt nutzt **kein** Alembic / Flask-Migrate. Schema-Bootstrap läuft per `db.create_all()`
in [app_pkg/__init__.py:76](app_pkg/__init__.py). Konsequenzen:

- **Neue Tabellen** sind trivial: beim nächsten Container-Start automatisch angelegt.
- **Neue Spalten an bestehender Tabelle** brauchen manuelles `ALTER TABLE` (SQLite mit eingeschränktem
  ALTER-Support, ggf. Table-Rebuild via Rename+Create+Copy+Drop).

Daraus folgt: **wo möglich, neue Tabellen statt neuer Spalten an `Conversion`**.

## Knoten 1 — Highlight-Schema

**Entscheidung**: Eigene Tabelle `Highlight` mit FK auf `Conversion`.

**Begründung**: Highlights sind erste-Klasse-Bürger im Reader-Paradigma — Cross-Doc-Queries (z.B.
„alle Highlights mit Tag X"), Daily-Review-Speisung (R4), Index-Performance. Plus: **leichtere
Migration** als die JSON-Spalten-Alternative (neue Tabelle = `create_all()` trivial vs. `ALTER
TABLE Conversion` manuell auf dem Server).

## Knoten 2 — Highlight-Anker

**Entscheidung**: Text-Quote + Prefix + Suffix (W3C Web Annotation Data Model, à la Hypothes.is).

**Schema-Felder pro Highlight**:

| Feld | Typ | Inhalt |
|---|---|---|
| `exact` | TEXT | Der markierte Text-String, exakt wie er im Roh-Markdown steht. |
| `prefix` | TEXT | Bis zu ~32 Zeichen direkt vor `exact` (für Disambiguation). |
| `suffix` | TEXT | Bis zu ~32 Zeichen direkt nach `exact` (für Disambiguation). |

**Begründung**: selbst-heilend (Re-Apply beim Render funktioniert ohne Position-Mapping), Browser-Selection
liefert alle drei Werte direkt (Range-Context-Walks), W3C-Standard für
Web-Annotations (Hypothes.is-Kompatibilität als Future-Option), pure client-side.

> **Korrektur 2026-05-31 (READER-FIX-B)**: `Selection.toString()` darf **nicht** der gespeicherte
> `exact`-Such-Key sein. An Block-Grenzen fügt `toString()` Separator-Newlines ein (am P→P-Übergang
> empirisch `\n\n`), die der `readerRawText`-Concat (purer `nodeValue`, ein `\n`) nicht enthält →
> `indexOf` beim Re-Apply findet nie → Block-übergreifende Highlights wurden unsichtbar (Cross-Format).
> `exact`/`prefix`/`suffix` werden seit `59eb0cd` **alle aus `readerRawText` gesliced** (siehe
> Decision-Log unten + Memory `feedback_selection_anchor_coordinate_system.md`). Die W3C-Anker-Idee
> bleibt korrekt — nur die Quelle des `exact`-Strings musste auf das Locate-Koordinatensystem
> umgestellt werden.

**Re-Apply-Algorithmus** beim Doc-Load:
1. Im Reader-View-DOM rekursiv alle Text-Nodes durchgehen.
2. Pro Highlight: nach `exact` suchen, plus Match-Disambiguation über `prefix`/`suffix`.
3. Bei eindeutigem Match: Text-Range mit `<span class="highlight" data-highlight-id="...">` wrappen.
4. Bei Mehrfach-Match trotz Prefix/Suffix: ersten nehmen (selten, akzeptabel).

## Knoten 3 — Notes-Storage

**Entscheidung**: Single-`note`-Feld direkt am Highlight (nullable TEXT).

**Begründung**: 1:1, YAGNI gegenüber Multi-Note-Threads. Wenn später Bedarf: non-breaking
Migration zu eigener `Note`-Tabelle möglich.

## Knoten 4 — Tag-Verzweigung

**Entscheidung**: `Tag` + zwei Junction-Tabellen `conversion_tags` + `highlight_tags`.

**Schema**:

```
Tag
  id            integer primary key
  user_id       integer FK -> user.id (per-user-Tag-Namespace)
  name          string(80) not null
  created_at    datetime
  unique(user_id, name)

conversion_tags  (Junction)
  conversion_id integer FK -> conversion.id
  tag_id        integer FK -> tag.id
  primary key(conversion_id, tag_id)

highlight_tags  (Junction)
  highlight_id  integer FK -> highlight.id
  tag_id        integer FK -> tag.id
  primary key(highlight_id, tag_id)
```

**Begründung**: eindeutiger Tag-Namespace pro User (kein doppelter „KI"-String), beidseitige
FK-Integrität, Cross-Query trivial (`Tag.conversions + Tag.highlights`), klassisches normalisiertes
Design.

**CSV-Migration**: die existierende `Conversion.tags`-CSV-Spalte ([models.py:35](models.py:35))
wurde in R2-A (☑ done 2026-05-25) migriert — `_migrate_conversion_tags_csv_to_junction(app)`-Helper
in `_run_pending_migrations` parst die CSV-Strings über `Tag.get_or_create(user_id, name)` in die
`conversion_tags`-Junction, setzt danach `Conversion.tags = ''` als Idempotenz-Marker. CSV-Spalte
**bleibt als Dead-Column** liegen (Master-Disposition zu R2-A-Zeit) — SQLite `DROP COLUMN` ist ein
Table-Rebuild, das gehört in einen separaten Cleanup-Sprint.

## Knoten 5 — Class-Naming

**Entscheidung**: `Conversion`-Class-Name bleibt unverändert.

**Begründung**: YAGNI. UI-Strings sagen eh „Library" / „Dokument", der Class-Name `Conversion` lebt
nur im Code. Rename wäre großer Code-Churn ohne UI-Wert. Wenn der Name irgendwann ärgert: eigener
Cleanup-Sprint mit clean ALTER TABLE.

## Knoten 6 — Reader-Achsen: Ort · Lese-Zustand · Priorität (R2-D)

**Entscheidung**: Drei **orthogonale** Reader-Achsen, je eine eigene Skalar-Spalte an `Conversion`, nie ein Misch-Enum:

| Achse | Spalte | Werte | Sprint |
|---|---|---|---|
| **Ort** (Triage-Lifecycle) | `lifecycle_status` | `inbox` / `later` / `archive` | R2-C |
| **Lese-Zustand** (Fortschritt) | `last_read_percent` | `NULL` / 0–100 (Weiterlesen = `0<%<95`) | R2-B |
| **Priorität** (geordnete Lese-Liste) | `queue_position` | `NULL` = nicht drauf / Float (`asc` = Reihenfolge) | R2-D |

**Begründung**: Ein Doc kann gleichzeitig *archiviert* + *halb gelesen* + *nicht auf der Lese-Liste* sein — oder *Inbox + ungelesen + Lese-Liste-Platz-3*. Die drei Fragen „wo lebt es?", „wie weit bin ich?", „was als Nächstes?" sind unabhängig; ein einzelnes Status-Enum würde sie kollabieren. Die **Lese-Liste (R2-D)** ist bewusst eine **eigene Dimension**, **kein** Repurpose der Favoriten — der Stern bleibt orthogonal und unsortiert (Archiv-Items dürfen Favorit sein), liefert also kein echtes „lese-als-Nächstes".

**`queue_position` als Float**: `NULL` = nicht auf der Liste, Sortierung `asc` = Lese-Reihenfolge. **Float** (nicht Integer), damit ein künftiges Drag-Einsortieren *zwischen* zwei Nachbarn nur den Mittelwert setzen muss, ohne die ganze Liste neu zu nummerieren. v1 macht **Hoch/Runter** = `queue_position`-**Swap** der zwei betroffenen Zeilen (eine commit-Boundary). Kein Backfill — alle starten `NULL` (leere Liste).

**Weiterlesen-View** ist rein abgeleitet (`0 < last_read_percent < 95`, sort `updated_at desc`) — kein eigenes Schema, die billige Hälfte von R2-D.

**Archiv ∩ Lese-Liste**: der queue-View zeigt nur **queued + nicht-archiviert** (`queue_position IS NOT NULL AND lifecycle_status != 'archive'`). **Kein Auto-Dequeue** beim Archivieren — reine View-Filterung, un-archivieren bringt das Item an seiner alten Position zurück. **Konsequenz fürs Reorder**: der up/down-Swap muss über **dieselbe gefilterte (sichtbare) Menge** laufen wie der View, sonst tauscht „hoch" mit einem unsichtbaren archivierten Nachbarn und tut scheinbar nichts (R2-D-Swap-Fix `d7f5097`; verallgemeinert in Memory `reference_reorder_over_filtered_set.md`). Dasselbe Prinzip greift im Frontend: queue-View-Membership-Änderungen (de-queue / archivieren) reloaden, damit `#Rang` + Edge-Arrows der Restliste korrekt bleiben.

## Vollständiges Reader-Schema-Diagramm

```
User (existiert)
 └── conversions  (existing 1:N)

Conversion (existing, unverändert in R1-B)
 ├── highlights   (NEW 1:N — R1-B-A)
 └── tags (CSV)   (existing, in R2-A zu Junction migriert)

Highlight (NEW — R1-B-A)
 ├── conversion_id FK
 ├── exact, prefix, suffix  (Anker, R1-B-A)
 ├── note                   (nullable, R1-B-B)
 └── tags                   (M:N via highlight_tags, R1-B-C)

Tag (NEW — R1-B-C)
 ├── user_id FK
 ├── name (unique per user)
 ├── conversions  (M:N via conversion_tags — R2-A migriert von CSV)
 └── highlights   (M:N via highlight_tags — R1-B-C)
```

## Sprint-Schneidung R1-B (statt eines L-Sprints)

R1-B war als L im BACKLOG → splittet in drei kleinere Sub-Sprints:

| Sprint | Inhalt | Größe |
|---|---|---|
| **R1-B-A** | Highlight-Core: Tabelle `Highlight` (ohne `note`, ohne Tags), Backend-API (POST/GET/DELETE), Frontend-Selektion-UX im Reader-View, Save, Re-Apply beim Reload. **Foundation.** | M |
| **R1-B-B** | Highlight-Notes: `note`-Feld-Migration (`ALTER TABLE Highlight ADD COLUMN note TEXT`), API-PATCH, Frontend-Note-Edit-Popup, Sidebar-Notes-Anzeige. | S |
| **R1-B-C** | Highlight-Tags + Tag-Foundation: Tabellen `Tag` + `highlight_tags`, Tag-API, Tag-Picker im Highlight-UI, Tag-Manager-Page. **`conversion_tags`-Junction baut R2-A**, nicht R1-B-C — CSV-`Conversion.tags` bleibt in R1-B-C unangetastet. | M |

R2-A erbt das Tag-Schema aus R1-B-C: Tabellen existieren bereits, R2-A migriert nur die existierende
CSV-Spalte in das `conversion_tags`-Junction.

## Sprint-Schneidung R2 (2026-05-25 nachgeführt)

Beim Schreiben des R2-A-Sprint-Prompts wurde R2-A in zwei Sprints gesplittet — der ursprüngliche
Plan hatte Tag-Migration + Lifecycle-Status (Inbox/Later/Archive) zusammen, mit Tag-Migration plus
Frontend-Umstellung war R2-A aber schon L. Lifecycle ist in einen eigenen R2-C-Sprint
ausgegliedert worden.

| Sprint | Inhalt | Größe | Status |
|---|---|---|---|
| **R2-A** | `conversion_tags`-Junction + CSV-Migration + `Tag.get_or_create`-DRY-Anker + Frontend Library-Card-Strip + Detail-Sidebar-Picker + GET-/api/tags-Erweiterung + Tag-Manager-Cascade beide Junctions + Pre-Commit-Patch Library-Search Junction-Branch. | L | ☑ done 2026-05-25 |
| **R2-B** | Filtered Views + Reading-Progress. Tag-Filter-Chip-Row in der Library-List mit URL-`?tag`-Persistierung (Junction-Pfad `Conversion.tag_refs.any(Tag.name == …)`, `==` statt `ilike`). Reading-Progress pro Card via nullable `Conversion.last_read_percent` (Prozent 0–100, furthest-read), `PATCH /api/conversions/<id>/progress` + Resume-on-Open + throttle/keepalive-Flush. | M | ☑ done 2026-05-29 (`4ff36a8` + `8b7e4f3`) |
| **R2-C** | Lifecycle-Status (Inbox/Later/Archive). **Eine Spalte** `Conversion.lifecycle_status` (String(20), Default `'inbox'`, indexed) via Inline-ALTER-TABLE-Helper + einmaliger differenzierter Backfill (`ai_newsletter→inbox`, Rest→`archive`). Orthogonal zum R2-B-Progress (**kein** 4. Status). Frontend: Status-Badge + Segmented-Toggle in Card + Detail, `?status`-Filter-Chips (kombinierbar mit `?tag`). | M | ☑ done 2026-06-04 (`f29c9cd` + `3350b89`) |
| **R2-D** | Lese-Liste / Priority-Shortlist (geordnet). **Eine Float-Spalte** `Conversion.queue_position` (nullable, `NULL`=nicht drauf) via Inline-ALTER (kein Backfill). Queue-API `POST /api/conversions/<id>/queue` (add / remove / up-down-Swap). `?view=queue` (queued+nicht-archiviert, sort `position asc`) + abgeleiteter `?view=reading` (`0<%<95`, kein neues Schema). Frontend: View-Switcher (Modus) + Queue-Flag-Toggle (Card+Detail) + Hoch/Runter-Reorder. **Dritte orthogonale Achse** (Priorität, Knoten 6). | M | ☑ done 2026-06-04 (`d466ec1` + `d7f5097` Swap-Fix + `4824a5c`) |

## Foundation-Voraussetzung für R1-B-A

R1-A liefert die kritischen Anker-Voraussetzungen:

- **Reader-View** mit lesbarem HTML statt Roh-Markdown-Pre-Block → Selektion ist UX-tauglich.
- **`<script type="text/markdown" id="content-source">`** hidden raw-source → Highlight-Save kann das
  exakt-markierte Text-Fragment plus Kontext aus dem Roh-Markdown ziehen (nicht aus dem HTML-Strip).
- **`script_safe`-Filter** mit `markupsafe.Markup`-Return verhindert HTML-Entity-Corruption im
  raw-source-Channel.

## Out-of-scope für alle R1-B-Sprints

- **R2-A** — `conversion_tags`-Junction + CSV-Migration. R1-B touched `Conversion.tags` nicht.
- **R2-B** — Filtered Views, Reading-Progress.
- **R3** — Web-Article-Save.
- **R4** — Ask-this-Document, RSS/Newsletter, Daily-Review (das ist die natürliche Highlight-Konsumenten-Ebene aber ein eigener Cluster), EPUB-Reader.
- **Conversion-Rename** — siehe Knoten 5.

---

## Decision-Log

| Datum | Entscheidung | Begründung |
|---|---|---|
| 2026-05-25 | Highlights als eigene Tabelle | Erste-Klasse-Bürger + leichtere Migration |
| 2026-05-25 | Text-Quote-Anker (W3C-Style) | Selbst-heilend, Standard, client-side |
| 2026-05-25 | Single-`note`-Feld | YAGNI, später migrierbar |
| 2026-05-25 | `Tag` + 2 Junction-Tabellen | Eindeutiger Namespace, FK-Integrität |
| 2026-05-25 | `Conversion`-Name bleibt | YAGNI |
| 2026-05-25 | R1-B splittet in A/B/C | L-Sprint zu groß für Sub-Thread |
| 2026-05-25 | R2-A splittet in Tag-Migration (R2-A) + Lifecycle-Status (R2-C) | Tag-Migration + Frontend-Umstellung schon L, Lifecycle hätte XL daraus gemacht |
| 2026-05-25 | `Tag.get_or_create`-Classmethod als DRY-Anker | 3 Call-Sites (Highlight-POST, Conversion-POST, Migration-Helper) — Single-Source-of-Truth für Normalisierung statt 3-fach inline |
| 2026-05-25 | CSV-Migration via leerer-Spalte-als-Marker idempotent | Kein zusätzlicher Marker-Column nötig, alte Spalte trägt den State („leer ↔ migriert"); Memory-Eintrag `reference_data_migration_idempotency.md` |
| 2026-05-29 | Reading-Progress als Prozent 0–100 (`Conversion.last_read_percent`, nullable Float) | Eine Spalte dient Card-Indikator **und** Resume; robust gegen Content-Längen-Änderung; subsumiert gelesen/ungelesen. Statt `last_read_scroll_top` (pixel-/char-genau, fragil bei Re-Render) |
| 2026-05-29 | Tag-Filter nur über URL `?tag=<name>` | Web-native Source, bookmarkbar, Back-Button; frischer `/library`-Aufruf zeigt Default (kein klebriger Filter). Kein localStorage |
| 2026-05-29 | Furthest-read statt aktueller Scroll | Höchster erreichter Prozent-Wert wird persistiert (Session-Max aus gespeichertem Wert geseedet) — Zurückscrollen resettet den Fortschritt nicht |
| 2026-05-29 | Persist via throttled fetch + keepalive-Flush, **nicht** `navigator.sendBeacon` | CSRFProtect ist global aktiv; sendBeacon kann den `X-CSRFToken`-Header nicht setzen. Throttle (~2s) über den globalen fetch-Wrapper, Flush bei `visibilitychange→hidden` via `fetch(..., {keepalive:true})` |
| 2026-05-31 | Highlight-Anker in EINEM Koordinatensystem (`readerRawText`), nie `selection.toString()` als Such-Key (READER-FIX-B) | Save und Locate müssen denselben Text-Raum teilen. `selection.toString()` fügt an Block-Grenzen Separator-Newlines ein (`\n\n` empirisch), `readerRawText` (`nodeValue`-Concat) nicht (`\n`) → `indexOf` beim Re-Apply schlug fehl, Block-übergreifende Highlights unsichtbar. Fix: `exact`/`prefix`/`suffix` alle aus `readerRawText` slicen (`rawOffsetForPoint`-Helper), Backward-Compat-Fallback `locateWhitespaceTolerant` rettet Alt-Highlights. Korrektur-Notiz: READER-FIX-A reparierte nur Inline-Grenzen, die „Cross-Format-verschwunden"-Bewertung beruhte auf synthetischen DevTools-Ranges — Selection-Features nur mit echtem Maus-Drag smoken. Memory `feedback_selection_anchor_coordinate_system.md` |
| 2026-06-03 | R2-C: Lifecycle als **3 Orte** (`inbox`/`later`/`archive`), nicht Star/Shortlist | Triage-Workflow braucht „wo lebt das Doc", nicht nur „wichtig ja/nein". Star/Shortlist bewusst verworfen zugunsten des 3-Ort-Modells |
| 2026-06-03 | R2-C: „gelesen" bleibt der R2-B-Progress, **kein** 4. Lifecycle-Status | Lese-Fortschritt (`last_read_percent`) und Triage-Ort sind orthogonale Achsen — ein Doc kann „gelesen + archive" oder „ungelesen + later" sein. Kein Misch-Enum |
| 2026-06-03 | R2-C: **eine Spalte** `lifecycle_status` statt eigener Tabelle | YAGNI — keine Status-Historie/Timestamps gebraucht. Inline-ALTER-TABLE (`reference_inline_sqlite_migration.md`) statt Table-Join; migrierbar zu eigener Tabelle wenn später Historie nötig |
| 2026-06-03 | R2-C: differenzierter Einmal-Backfill beim Spalten-Add (`ai_newsletter→inbox`, Rest→`archive`) | Bestehende Newsletter sind echtes Triage-Material (Inbox), alte Tool-Outputs sind erledigt (Archive). Backfill-`UPDATE` **innerhalb** des Spalten-Existenz-Guards → genau einmal, kein Re-Clobber bei späterem manuellem Verschieben |
| 2026-06-04 | R2-D: Lese-Liste als **eigene Dimension** (`queue_position`), kein Favoriten-Repurpose | Der Favoriten-Stern ist orthogonal (Archiv-Items dürfen Favorit sein) **und** unsortiert → kein echtes „lese-als-Nächstes". Eigene Spalte statt den Stern umzudeuten; keine Favoriten-Migration (Knoten 6) |
| 2026-06-04 | R2-D: `queue_position` als **Float**, nicht Integer | Float erlaubt späteres Drag-Einsortieren *zwischen* zwei Nachbarn via Mittelwert ohne Neu-Nummerierung der ganzen Liste. v1 macht Hoch/Runter = `queue_position`-Swap zweier Zeilen (eine commit-Boundary) |
| 2026-06-04 | R2-D: Reorder-Swap läuft über die **sichtbare (gefilterte) Menge**, nicht alle queued Items | Decision #5 dequeued archivierte Items nicht → sie behalten `queue_position`. Gegen die ungefilterte Menge tauschte „hoch" mit einem unsichtbaren Archiv-Nachbarn (scheinbarer No-op). Der `queued`-Set im up/down-Zweig kriegt denselben `lifecycle_status != 'archive'`-Filter wie der View (Swap-Fix `d7f5097` nach Master-Diff-Read, rot→grün-Regressionstest). Verallgemeinert: jede Reorder/Mutation über dieselbe gefilterte Menge wie der View — Memory `reference_reorder_over_filtered_set.md` |
| 2026-06-04 | R2-D: Archiv ∩ Lese-Liste = **View-Filter** statt Auto-Dequeue | Archivieren fasst `queue_position` nicht an; der queue-View filtert `lifecycle_status != 'archive'` raus. Un-archivieren bringt das Item an alter Position zurück — Lifecycle (Ort) und queue_position (Priorität) bleiben unabhängige Achsen |
