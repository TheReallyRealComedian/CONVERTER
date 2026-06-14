# Reader-Architecture — Entscheidungs-Memo

**Stand**: 2026-06-14
**Workshop-Datum**: 2026-05-25 (Master-Workshop nach R1-A done); R2-B-Workshop 2026-05-29; READER-FIX-B Anker-Korrektur 2026-05-31; R2-C-Workshop 2026-06-03; R2-D-Workshop 2026-06-04; R2-E/R2-F-Workshop 2026-06-12; VIS1-DS-Angleichung 2026-06-14 (visuell, Decision-Log)
**Status**: Aktive Referenz für R2 (☑ A/B/C/D komplett — die drei Reader-Achsen Ort · Lese-Zustand · Priorität sind voll getrennt) + R3 + R4-LEARN Sprints — nicht archiviert. Reader-Screens visuell auf die nachgeschärften DS-Regeln angeglichen (VIS1, Decision-Log unten).

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

## Knoten 7 — Library-IA „Readwise-3er" + Tag-Daten (R2-E)

**Entscheidung**: Die drei orthogonalen Achsen aus Knoten 6 bleiben im Datenmodell **unangetastet** — R2-E ist eine reine **Präsentations-/Hygiene-Welle**. Statt zweier konkurrierender Leisten (View-Switcher Alle/Lese-Liste/Weiterlesen **plus** Status-Chips Inbox/Später/Archiv — der „wie kommt was wohin"-Live-Befund) gibt es **eine Tab-Leiste mit genau drei Top-Level-Zielen**:

| Tab | Beantwortet | Filter-Semantik | `view=` |
|---|---|---|---|
| **Inbox** | „Was ist neu / untriagiert?" | `lifecycle_status='inbox'` **AND** `queue_position IS NULL` | `inbox` |
| **Lese-Liste** | „Was lese ich jetzt, in welcher Reihenfolge?" | R2-D-Queue (geordnet, Archiv aus) + Weiterlesen-Sektion oben | `queue` |
| **Bibliothek** | „Wo finde ich alles?" | Default: alles + Suche/Filter; Status wird hier zum **Ort-Filter** | (default) |

**Inbox-Triage-Semantik** (die zentrale Design-Entscheidung): Die Inbox zeigt nur **Untriagiertes** = Ort `inbox` **und** nicht in der Queue. „Zur Lese-Liste" (Queue-Add) lässt `lifecycle_status` unverändert — das Item verschwindet trotzdem aus der Inbox, weil es jetzt gequeued ist. Wird es später de-queued und ist immer noch `inbox`, **fällt es zurück in die Inbox** (bewusst: un-gequeued + untriagiert = wieder Triage). „Später"/„Archiv" setzen wie gehabt den Ort (R2-C-API). **Keine Auto-Kopplung der Achsen** — die R2-D-Entscheidung (kein Auto-Dequeue bei Archiv etc.) bleibt; die Inbox-Verknüpfung ist eine reine *View-Filter*-Konjunktion, kein Schema-Trigger.

**Wegfall `view=reading`**: Der eigenständige Weiterlesen-View entfällt ersatzlos (Single-User, keine Kompat nötig). Die Filter-Logik lebt als **Weiterlesen-Sektion** oben in der Lese-Liste weiter (`0<last_read_percent<95`, **nicht** gequeued, **nicht** archiviert, sort `updated_at desc` — 1:1 vom alten View). Gequeuete Items mit Fortschritt doppeln nicht, ihr Karten-Fortschrittsbalken (R2-B) zeigt das schon. Unbekannte `view`-Werte (inkl. `reading`) fallen wie immer auf die Default-Bibliothek.

**Tag-Leiste** (nur Bibliothek): Top-15-Chips nach **Nutzungs-Count** (Count über die `conversion_tags`-Junction, absteigend, tie-break alphabetisch), der aktive `?tag=`-Chip ist **immer** sichtbar (auch jenseits Top-15 und für Phantom-Tags ohne Bestand). „+N weitere" klappt die volle alphabetische Liste auf (collapsed Default, kein Persist). Typeahead über native `<datalist>`; Auto-Submit am **`change`**-Event (nicht `input` — sonst snappt ein Tag, das striktes Präfix eines anderen ist, beim Tippen weg, z.B. `spacex` vs. `spacex ipo`). Das GET-Formular repliziert die `pagination_args`-Semantik der Chips via Hidden-Inputs (page→1, Defaults bleiben aus der URL). CSS-Falle dokumentiert: die `display:contents`-Chip-Listen überschreiben das UA-`hidden`-Attribut → genereller `[hidden]{display:none!important}`-Guard nötig.

**Zentrale Tag-Normalisierung**: `Tag.normalize_name` (models.py) ist **der eine Ort** für alle Schreibpfade (UI-Attach, Ingest-Topics, Cleanup-Script) — lowercase+trim (R2-A) plus Strippen der LLM-Newsletter-Artefakte (`*` und `` ` `` überall, `[`/`]` an den Rändern, Whitespace-Läufe → ein Space). `get_or_create` ruft sie auf, Rückgabe-Kontrakt unverändert (None → Aufrufer skippt/400). Den **Bestand** zieht `scripts/cleanup_tags.py` nach (dry-run-Default, `--apply`, idempotent): rename / merge bei Kollision (Junction-Rows beider M:N umhängen, Duplikat-Paare abfangen) / delete bei Leer-Normalisierung. Das Script **importiert** `normalize_name` (reimplementiert sie nicht) und garantiert via same-path + `rollback()` Dry-run-Treue. Verallgemeinerte Lehre: Memory `reference_tag_vocab_central_gate_plus_backfill_script.md` (Geschwister zu `reference_data_migration_idempotency.md`).

## Knoten 8 — Reader-Abschluss-Leiste + Fortschritts-Klarheit (R2-F)

**Live-Befund** (Oliver, erster Nutzungs-Burst): „nach dem fertig lesen muss ich wieder hoch … aber das setzt den fortschritt zurück." **Phase-1-Diagnose (read-only) auf der echten Mintbox: der DB-Wert sinkt nie.** Die R2-B-furthest-read-Persistierung sendet nur `maxReached` (aus dem gespeicherten Wert geseedet, nur vorwärts); beim Hochscrollen läuft die *sichtbare* Progress-Bar als Positions-Anzeige zurück und *suggeriert* einen Reset, aber der Client sendet **null** PATCHes (Gate `percent > maxReached`) und die DB bleibt stehen. Kein Bug — ein Anzeige-/Affordance-Loch. R2-F adressiert beides:

- **„Gelesen"-Label** (entkoppelt „wo bin ich" von „wie weit war ich"): sobald der *persistierte* furthest-read `maxReached >= 95` ist, zeigt ein dezentes, **dauerhaftes** Label oben-rechts — gebunden an `maxReached`, **nicht** an die Scroll-Position. Die Bar darf beim Hochscrollen zurücklaufen (das ist ihr Job), das Label bleibt stehen. `syncReadFlag()` schaltet nur **an** (maxReached wächst monoton), bei init + nach jedem Bump; sticky height:0-Wrapper + absolute Pill → gepinnt ohne Flow-Höhe (token-driven → Dark gratis). Schwelle `READ_COMPLETE_PERCENT = 95` einmal benannt (Label **und** Resume-Check — dieselbe „gelesen"-Grenze wie die Karte). **Kein** zweites Mid-Progress-Label — der Karten-Fortschrittsbalken (R2-B) + Resume-on-Open decken den 40%-Fall ab.
- **Abschluss-Leiste** am Content-Ende (nur bei gerendertem Markdown): „Zurück zur Library" (`history.back()` wenn der direkte Referrer die Liste war → erhält Scroll-/Filter-State, sonst `href`-Fallback `/library`) + „Archivieren" (nutzt den **bestehenden** R2-C-Status-PUT, dann Erfolgs-Toast + Navigation). Kein neuer Endpoint, kein neuer Status — reine Affordance, damit man am Doc-Ende nicht hochscrollen muss, um wegzukommen.
- **Server-Forward-Clamp** im Progress-PATCH: `last_read_percent = max(stored, range_clamp(percent))` (range-clamp zuerst, `or 0.0`-None-Schutz, Response liefert den effektiven Wert). Damit ist furthest-read **doppelt garantiert** — Client-Gate **und** Server. Das Vorwärts-Gate hängt nicht mehr allein am korrekten Client-Seeding; jeder künftige Seeding-/Client-Bug kann den Wert serverseitig nicht mehr senken. **Konsequenz**: ein „Fortschritt zurücksetzen" bräuchte künftig einen expliziten Pfad (Flag, das den Clamp umgeht, oder dedizierter Reset-Endpoint) — als BACKLOG-Notiz festgehalten, nicht gebaut.

## Knoten 9 — Fortschritts-Bar = furthest-read + „Als ungelesen"-Reset (R2-G)

**Folge-Feedback** (Oliver, 2026-06-13, nach R2-F): Knoten 8 ließ die Bar bewusst **Positions-Anzeige** (läuft beim Hochscrollen zurück) und steckte den Max nur ins „Gelesen"-Label. Oliver will die **Bar selbst** als furthest-read — beim Hochscrollen stehenbleibend, nur vorwärts wachsend (Readwise-Verhalten). **Explizite Revision der Knoten-8-Bar-Entscheidung** (Bar = Position → Bar = Max); Grund: das Zurücklaufen las sich weiter als „Reset", obwohl Phase 1 von R2-F bewies, dass der DB-Wert nie sinkt. R2-G ist reine Anzeige + ein Reset-Pfad, **kein Schema-Touch**:

- **Bar zeigt furthest-read**: `fill = max(position, maxReached)` (vorher nur `position`). `maxReached` ist aus `last_read_percent` geseedet → die Bar steht beim Öffnen sofort auf dem gespeicherten Max, auch vor dem rAF-Arming. Das R2-F-Vorwärts-Gate (`persistArmed && percent > maxReached`) bleibt **unverändert** — nur die Fill-Zeile koppelt jetzt an den Max statt an die Position. **Kein** separater Positions-Marker (YAGNI — die Position gibt der Viewport).
- **„Gelesen"-Label folgt jetzt dem Max bidirektional**: in Knoten 8 schaltete `syncReadFlag()` nur **an** (maxReached wuchs monoton). Mit dem Reset kann `maxReached` auf 0 fallen → das Label muss wieder verschwinden. Jetzt `readFlag.hidden = !(maxReached >= READ_COMPLETE_PERCENT)` — erscheint ab 95, verschwindet beim Reset, kommt bei erneutem ≥95 zurück.
- **Reset-Pfad** „Als ungelesen markieren" (Detail-Sidebar, eigene „Lese-Fortschritt"-Card — orthogonal zu Status/Lese-Liste): explizites `{"reset": true}` im **bestehenden** Progress-PATCH setzt `last_read_percent = NULL` und **umgeht bewusst den R2-F-Forward-Clamp** — die eine sanktionierte Art, den Wert zu senken (genau der „explizite Pfad nötig", den Knoten 8 als BACKLOG-Notiz hinterließ). `NULL` = „nie gelesen", identisch zu frisch-ingestet (kein Karten-Balken, nicht in Weiterlesen). Non-bool `reset` = 400 (gleiche strikte Bool-Validierung wie `percent`). Clientseitig disarmt `resetProgress()` vor dem Redraw **und** cancelt den pending Persist-Timer, damit das sofortige `update()` die aktuelle Position nicht als neuen Fortschritt zurückpersistiert (Self-Persist-Schutz wie beim Resume-Scroll; Memory `reference_scroll_progress_persistence`), rearmt im Folge-rAF.

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
| 2026-06-12 | R2-E: **eine Tab-Leiste** (Inbox/Lese-Liste/Bibliothek) ersetzt View-Switcher UND Top-Level-Status-Chips | Zwei konkurrierende Leisten waren der „wie kommt was wohin"-Live-Befund. Drei Top-Level-Ziele („Readwise-3er") statt orthogonaler Achsen direkt an die Oberfläche zu legen. Datenmodell unangetastet — reine Präsentation |
| 2026-06-12 | R2-E: Inbox = `lifecycle_status='inbox'` **AND** `queue_position IS NULL` (Triage-Konjunktion) | Die Inbox beantwortet „was ist untriagiert?". Queueing triagiert implizit (Status bleibt inbox, Item verlässt die Inbox); De-queue eines noch-inbox-Items fällt zurück in die Triage. Reine View-Filter-Konjunktion, **kein** Schema-Trigger/Auto-Kopplung (R2-D-Entscheidung bleibt) |
| 2026-06-12 | R2-E: `view=reading` entfällt → Weiterlesen wird **Sektion** in der Lese-Liste | Single-User, keine Kompat nötig. Dieselbe Filter-Logik (`0<%<95`, unqueued, nicht-archiviert, `updated_at desc`) lebt als Sektion über der Queue weiter; gequeuete Progress-Items doppeln nicht (Karten-Balken zeigt's). Unbekannte `view`-Werte → Default-Bibliothek |
| 2026-06-12 | R2-E: Tag-Leiste Top-15 nach Count + „+N weitere" + `<datalist>`-Typeahead, Auto-Submit am **`change`**-Event | Count-Sortierung hebt die genutzten Tags; aktiver Chip immer sichtbar (auch Phantom-Tag). `change` statt `input`, sonst snappt ein Präfix-Tag (`spacex` vor `spacex ipo`) beim Tippen weg. `[hidden]`-Guard nötig, weil `display:contents` das UA-`hidden` überschreibt |
| 2026-06-12 | R2-E: Tag-Normalisierung **zentral** in `Tag.normalize_name` + Bestand per `scripts/cleanup_tags.py` nachziehen | Ein Gate für alle Schreibpfade (UI/Ingest/Script) statt N-fach inline — härtet gegen LLM-Newsletter-Artefakte (`** [anthropic`). Das Cleanup-Script **importiert** die Runtime-Normalisierung (reimplementiert sie nicht) und garantiert via same-path+rollback Dry-run-Treue. Memory `reference_tag_vocab_central_gate_plus_backfill_script.md`. Quasi-Duplikate (`ki-ethik`/`ai-ethik`) bleiben bewusst manuell mergebar (out-of-scope, kein Auto-Merge wegen Sprach-/Bedeutungs-Risiko) |
| 2026-06-12 | R2-F: „Gelesen"-Label an `maxReached` (persistierter furthest-read), **nicht** an der Scroll-Position | Die Bar ist Positions-Anzeige und läuft beim Hochscrollen zurück — genau das erzeugte Olivers „Reset"-Eindruck (Phase 1 belegte: DB-Wert sinkt nie, der Client sendet null PATCHes beim Hochscrollen). Das Label entkoppelt „wie weit war ich" von „wo bin ich" und bleibt ab `>=95` dauerhaft. Schwelle = dieselbe „gelesen"-Grenze wie Karte/Resume (`READ_COMPLETE_PERCENT=95`); kein zweites Mid-Progress-Label (Karten-Balken + Resume decken's) |
| 2026-06-12 | R2-F: **Server-Forward-Clamp** im Progress-PATCH (`max(stored, percent)`) statt bedingungslosem Überschreiben | Phase-1-Repro auf der Mintbox bewies: der DB-Wert sinkt clientseitig nie (nur `maxReached` gesendet), aber der Server clampte nicht — das Vorwärts-Gate hing allein am Client-Seeding. Server-Clamp macht furthest-read doppelt garantiert (Defense-in-Depth). Konsequenz: „Fortschritt zurücksetzen" bräuchte ein explizites Flag (BACKLOG-Notiz, nicht gebaut) |
| 2026-06-12 | R2-F: Abschluss-Leiste am Content-Ende statt Auto-Archivieren bei 100% | Reine Affordance (Zurück via `history.back`/href-Fallback + Archivieren über den bestehenden R2-C-PUT), damit man am Doc-Ende nicht hochscrollen muss, um wegzukommen. Auto-Prompt/Auto-Archiv bei 100% im Workshop verworfen (zu invasiv; Single-User behält Kontrolle) |
| 2026-06-13 | R2-G: Bar zeigt **furthest-read** (`fill = max(position, maxReached)`), nicht die Scroll-Position — **Revision** der R2-F-Knoten-8-Entscheidung | Folge-Feedback: das Zurücklaufen beim Hochscrollen las sich weiter als „Reset". Bar bleibt jetzt am Max stehen, wächst nur vorwärts (Readwise-Verhalten). Das R2-F-Vorwärts-Gate (`persistArmed && percent>maxReached`) bleibt unverändert — nur die Fill-Zeile koppelt an den Max; „Gelesen"-Label folgt dem Max jetzt bidirektional (nicht mehr nur-monoton-an, sonst bliebe es nach Reset hängen). Kein separater Positions-Marker (YAGNI) |
| 2026-06-13 | R2-G: Reset „Als ungelesen" via explizitem `{"reset":true}`-Flag im bestehenden Progress-PATCH → `last_read_percent = NULL`, umgeht den R2-F-Forward-Clamp | Genau der „explizite Pfad", den Knoten 8/R2-F als BACKLOG-Notiz hinterließ — eine sanktionierte Art, den Wert zu senken. `NULL` (nicht 0) = „nie gelesen", identisch zu frisch-ingestet (kein Karten-Balken, nicht in Weiterlesen). Non-bool `reset` = 400 (strikte Bool-Validierung wie `percent`). `resetProgress()` disarmt + cancelt den pending Persist-Timer vor dem Redraw, sonst persistiert das sofortige `update()` die aktuelle Position zurück (Self-Persist-Schutz, 2. Präzedenzfall in Memory `reference_scroll_progress_persistence` neben dem Resume-Scroll) |
| 2026-06-14 | VIS1: Reader/Detail (+ alle Screens) auf die im Design-Review **nachgeschärften** DS-Regeln angeglichen — Nunito, Elevation-Budget, ≥32px-Spacing, Status-als-Tint, kein Hardcode | Das DS wurde **aus CONVERTER extrahiert** und im Review nachgeschärft → die Rück-Angleichung ist ein **Elevation-Budget-Audit, kein Reskin** (Farben/Radii/Shadow-Primitives identisch). Reader-Detail-Sidebar-Cards flach mit **einem** gepressten aktiven Element (Status-Segment/Toggle), Highlight-Popover (border→`--nm-raised`) + Highlight-Sidebar-Cards (raised→Tint) entpillowed, `queue-toggle` Regel-5 (flat-off/pressed-on), Divider→`--nm-sep-top`, Cross-Format-`border-left`→Tint (Regel 7). **R2-F/G-Verhalten unberührt** (`library_detail.js` nicht angefasst — reine CSS/Markup-Angleichung). Memory `reference_design_system_realignment_is_budget_audit`. Reader/Detail = VIS1-Phase 3 (`b598c0e`); Tools-Rest/Login/Tags → VIS2 |
| 2026-06-14 | VIS2: Audio + Mermaid + Login + Tags angeglichen — **damit stehen alle Screens (VIS1 + VIS2) auf den nachgeschärften DS-Regeln, die App-weite Angleichung ist komplett** | Fortsetzung VIS1, dieselbe Methodik (Elevation-Budget · ≥32px · Status-als-Tint · Hardcode→Token · Hover-Reveal). Audio: Mic-Recording gesättigter Fill → `tint-danger`-Wash + danger-Ink (Regel 7/8), Mode-Radios echtes Segmented (rule-7-verbotene farbige inset-Left-Bar weg), Podcast-Stage-Bar `var(--nm-pressed-sm,…)`-als-`background`-Bug gefixt (Shadow-Token war ungültig als bg). Tags: `.tag-manager-card`-Reihen von N raised-Pillows → flach getönt (`surface-grad`), nur die gehoverte/fokussierte Reihe hebt + enthüllt das Delete (card-actions-Muster). Login: ≥32px-Spacing. Mermaid: `pane-header`-Separator → `--nm-sep-bottom`-Token. Plus Carry-forward `.save-library-btn.saved`-Grün → `--nm-alert-success-ink`. **Rein visuell, kein Verhaltens-Touch**, pytest 217/217, Live-Smoke dark+light je Screen. `93c85bf` P1 (Audio) + `2e033aa` P2 (Mermaid/Login/Tags) + Doc-Wrap. Memory `reference_design_system_realignment_is_budget_audit` (VIS1+VIS2 ein Präzedenzfall) |
