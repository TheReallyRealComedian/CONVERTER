# Reader-Architecture — Entscheidungs-Memo

**Stand**: 2026-06-22
**Workshop-Datum**: 2026-05-25 (Master-Workshop nach R1-A done); R2-B-Workshop 2026-05-29; READER-FIX-B Anker-Korrektur 2026-05-31; R2-C-Workshop 2026-06-03; R2-D-Workshop 2026-06-04; R2-E/R2-F-Workshop 2026-06-12; VIS1-DS-Angleichung 2026-06-14 (visuell, Decision-Log); R2-H-Workshop 2026-06-15 (flache Vier-Orte-IA, Knoten 10); R4-LEARN-Build 2026-06-19 (SR-/Recall-Layer, Knoten 11); R4-LEARN-FIX 2026-06-20 (Card-DELETE + Lösch-Affordanz, Knoten 11); READER-ADJ 2026-06-22 (Reader-Mode + geteiltes „Aa"-Popover, Knoten 12)
**Status**: Aktive Referenz für R2 (☑ A–H komplett) + R4-LEARN (☑ done 2026-06-19, Knoten 11) + R3 — nicht archiviert. **Das Datenmodell hat drei orthogonale Reader-Achsen (Ort · Lese-Zustand · Priorität, Knoten 6); die *Bedien-Schicht* darüber ist seit R2-H eine EINE flache Vier-Orte-Achse (Inbox · Lese-Liste · Bibliothek · Archiv, Knoten 10) — die Achsen bleiben im Schema, „place" ist eine abgeleitete Single-Select-Sicht.** Reader-Screens visuell auf die nachgeschärften DS-Regeln angeglichen (VIS1, Decision-Log unten).

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

## Knoten 10 — Flache Vier-Orte-Achse (R2-H) — kollabiert die Bedien-Schicht von Knoten 6 + 7

**Folge-Feedback** (Oliver, 2026-06-15): die drei *orthogonalen* Achsen aus Knoten 6 (Ort · Lese-Zustand · Priorität) plus der Favorit waren als **Bedien-Oberfläche** drei sich überlappende Achsen, und die Readwise-3er-IA aus Knoten 7 hatte zusätzlich **zwei konkurrierende Nav-Listen** (Tabs Inbox/Lese-Liste/Bibliothek **und** Ort-Chips Inbox/Später/Archiv) → „ich check unsere zwei Listen nicht / wofür ist die Flag / Später=WTF". **Explizite Revision der Bedien-Schicht von Knoten 6 + 7**: die Achsen kollabieren auf **EINE flache Single-Select-Achse mit vier sich gegenseitig ausschließenden Orten** — **Inbox · Lese-Liste · Bibliothek · Archiv** —, **identisch** in Top-Nav (Tabs), auf jeder Karte (Move-Control) und im Detail (`.place-control`). **Kein Schema-Touch, keine Migration**: `lifecycle_status` + `queue_position` bleiben die Wahrheit (das Schema von Knoten 6 ist unangetastet); „place" ist eine **abgeleitete** Single-Select-Sicht. `is_favorite` liegt als Spalte brach (UI raus, reversibel).

**Ableitung** (Präzedenz von oben, erste zutreffende Bedingung gewinnt):

| Ort | Bedingung |
|---|---|
| **Archiv** | `lifecycle_status == 'archive'` |
| **Lese-Liste** | sonst, `queue_position IS NOT NULL` (geordnet `queue_position asc`) |
| **Inbox** | sonst, `lifecycle_status == 'inbox'` |
| **Bibliothek** | sonst — neutrales Regal (`lifecycle_status == 'later'`, `queue_position NULL`) |

**Move-Aktion** — ein atomarer `POST /api/conversions/<id>/place {place}` setzt die `(lifecycle_status, queue_position)`-Kombi und hält Exklusivität: `inbox → ('inbox', NULL)` · `bibliothek → ('later', NULL)` · `archiv → ('archive', NULL)` · `leseliste → ('later', max(queue)+1.0)` (ans Ende, idempotent für ein bereits gelistetes Item). Subsumiert den R2-C-PUT-`lifecycle_status` **und** den add/remove-Teil von `/queue`; der **Hoch/Runter-Swap bleibt auf `/queue`** (Reorder über die sichtbare Menge, Knoten 6). **Archivieren dequeued jetzt** (`queue_position=NULL`) — **das ersetzt Knoten 6's „kein Auto-Dequeue"**: im exklusiven Modell kann ein Item nicht zugleich Archiv und Lese-Liste sein.

**Views/Tabs** = die vier Orte (Filter auf die Ableitung). **Suche/Tag/Typ = globaler Finder**: jeder aktive Content-Filter (`search OR tag OR gültiger type`) spannt über **alle Nicht-Archiv-Orte** (Inbox+Lese-Liste+Bibliothek) und ignoriert den Ort-Tab; der Bibliothek-Tab **ohne** Filter ist nur das neutrale Regal. „Alles sehen" = filtern (Olis Wahl); die Tag-Leiste lebt im Bibliothek-Tab, wirkt aber global. *(Such-/Filter-Scope ist der eine bewusst smoke-justierbare Punkt; ursprüngliche R2-H-Default war „nur Suche global", per Master-Umlenkung in P2A auf „jeder Content-Filter" erweitert.)*

**Lese-Fortschritt bleibt orthogonal** (kein 5. Ort): Karten-Fortschrittsbalken + Resume-on-Open + die R2-G-„Als ungelesen"-Card (Knoten 9) bleiben unverändert. **Die Weiterlesen-Sektion (Knoten 7) entfällt** — sie überlappte Orte (brach „flat"); Fortschritt lebt als Balken auf der Karte weiter, wo immer das Item liegt.

**converter-mcp / Read-API unberührt**: `to_dict()` + die GET-`/api/conversions`-Endpoints + die Spalten `lifecycle_status`/`queue_position` bleiben exakt; nur die UI-Bedien-Achsen sind kollabiert. Der MCP-Treiber-Use-Case (`list_audio_transcripts` mit `exclude_status=archive`) hält weiter.

**Knoten 6 + 7 bleiben als Daten-/Historie-Referenz gültig** (das Schema ist unverändert orthogonal); ihre *Bedien-Schicht* (Tabs/Chips/Weiterlesen/Favorit-Stern) ist durch Knoten 10 ersetzt. Verallgemeinerbare Lehre → Memory `reference_collapse_orthogonal_axes_to_flat_single_select`.

## Knoten 11 — SR-/Recall-Layer: Karten · FSRS · Review-UI (R4-LEARN)

**Entscheidung** (R4-LEARN, 2026-06-19, aus einem 2×-code-review'ten Produkt-Brief geschnitten): ein **Spaced-Repetition-/Recall-Layer über den bestehenden Highlights** — das ursprüngliche Readwise-Herz. **Teilung**: der externe **Agent erzeugt** Karten (CONVERTER **nicht** — keine Generierung im Stack), CONVERTER **speichert/plant/zeigt** und liefert die Schreib-Endpoints + einen globalen Highlights-Reader. Die Wissenslandkarte ist bewusst **out** (spätere Phase).

**Schema** (neue Tabellen via `db.create_all`, **keine Migration** — `_run_pending_migrations` nur für Spalten-Adds an *bestehenden* Tabellen, hier nichts):

- **`Card`** — **self-contained**: `front`/`back`/`cloze_text`/`prompt`/`note` stehen auf der Karte, das Review liest das Highlight **nie live**. `user_id` FK (Owner-Scope — eine Karte überlebt ihr Highlight, scopet also nicht über das nullbare Highlight; wie `Conversion`/`Tag`). `highlight_id` FK **nullable** = Best-Effort-Provenienz (bare Column, **keine** `Highlight`↔`Card`-Relationship). `source_snapshot`/`source_doc_title` = Authoring-Snapshot. `type` (`atomic`|`generative`), `state` (`ok`|`wackelt`), `created_by` (default `agent`). Tags M:N via **`card_tags`** (analog `highlight_tags`).
- **`Review`** — eigene **1:1**-Tabelle (sauber für eine spätere Review-History, nicht auf die Karte gemerged): `due`/`stability`/`difficulty`/`last_reviewed`/`reps`/`lapses`/`rating_history`. ORM-cascade `Card.review = relationship(uselist=False, cascade='all, delete-orphan')`. `POST /api/cards` legt sie gleich im FSRS-„new"-Zustand mit an (`due=jetzt`, `reps`/`lapses`=0, Rest NULL).

**Lösch-Mechanik = ORM-`before_delete`-Event** (der Muss-Fix): CONVERTER läuft SQLite **ohne `PRAGMA foreign_keys=ON`** → ein deklariertes `ON DELETE SET NULL`/`CASCADE` ist am DB-Level **inert**. Die echte Mechanik ist ein `@event.listens_for(Highlight, 'before_delete')`, der `card.highlight_id` per Core-UPDATE auf der Flush-`connection` (nicht der Session — mid-flush) nullt. Feuert **direkt** (DELETE-Endpoint) **und** über den Conversion-`delete-orphan`-Cascade (pro Highlight-Delete) → Karte+Review überleben, nur der Provenienz-Link bricht. **Alle Cascades in diesem Stack sind ORM-Level**, nie DB-Level. Memory `reference_sqlite_no_fk_pragma_orm_delete`.

**FSRS hinter einer swappable Scheduler-Schnittstelle** (`services/scheduler/`): kleine ABC `new_card_state()` + `apply_rating(review_state, rating)->dict` (deals in plain dicts, storage-agnostisch). Zwei Impls — **FSRS** (Default, via `fsrs==6.3.1`, `enable_fuzzing=False` für deterministische Intervalle) + **SM-2-Fallback** hinter derselben ABC; `get_scheduler()` wählt per Config `SCHEDULER_ENGINE` (Default `fsrs`) / `FSRS_DESIRED_RETENTION` (Default 0.9). **Kein Auto-Grading** (Rating kommt immer vom User). `reps`/`lapses` besitzen wir (FSRS-6-`Card` trackt sie nicht mehr; „again = lapse"). **Dokumentierte Vereinfachung**: das gesperrte `Review`-Schema hat keine Spalte für FSRS-`state`/`step` → eine bereits bewertete Karte wird im Review-State **rekonstruiert** (graduiert); die stability/difficulty-getriebene Intervall-Math bleibt voll erhalten, nur die Sub-Day-Learning-Step-Rampe kollabiert. Memory `reference_swappable_scheduler_interface`.

**Drei-Wege-Auth-Split** (gesperrt): die Schreib-/Lese-Pfade haben drei verschiedene Auth-Modelle, je nach Akteur:

| Pfad | Endpoint(s) | Auth | Akteur |
|---|---|---|---|
| **Agent-Write** | `POST` / `PATCH /api/cards` | **Token** (`CARD_TOKEN`, Ingest-Muster — eigenes Env-Token, unabhängige Rotation; constant-time, fail-closed, nur diese 2 Views CSRF-exempt, Token nie geloggt; Ziel-User via `INGEST_USER`/first()) | externer Agent |
| **User-Rate** | `POST /api/cards/<id>/review` | **Session** (`@login_required`, CSRF-protected) | User in der Review-UI |
| **User-Annotate** | `POST /api/cards/<id>/annotate` (Vertiefen→`wackelt` + Inline-Notiz) | **Session** (`@login_required`) — eigener Pfad, weil PATCH token-only ist | User in der Review-UI |
| **User-Delete** | `DELETE /api/cards/<id>` (Karte löschen, R4-LEARN-FIX) | **Session** (`@login_required`, owner-scoped → 404) — ORM-Cascade nimmt `review`+`card_tags` mit, **nicht** CSRF-exempt | User in der Review-UI |
| **Reads** | `GET /api/highlights/recent`, `/api/cards`, `/api/cards/<id>`, `/api/review-state` | **Session** (`@login_required`, owner-scoped) | MCP/Agent **und** UI |

**Für den converter-mcp-Owner (Koordinator-Scope, nicht CONVERTER)**: zu wrappen sind die **Writes** (`POST` + `PATCH /api/cards`, mit `CARD_TOKEN`) und die **vier Reader** (`/api/highlights/recent`, `/api/cards`, `/api/cards/<id>`, `/api/review-state`, Session). **`POST /review`, `POST /annotate` und `DELETE /api/cards/<id>` sind UI-only** (der User bewertet/vertieft/löscht in der Oberfläche) — **nicht** für den Agent wrappen.

**Review-UI** (`/review`, Jinja/Vanilla, `@login_required`): läuft die `due<=jetzt`-Queue aus `/api/review-state` ab — atomar (`front` oder Cloze-Lücke) → Aufdecken → Rating `again/hard/good/easy` → `POST /review` → nächste; generativ (`prompt` → erklären → Musterantwort als Stichpunkte). „Vertiefen" + Inline-Notiz über `POST /annotate`; **„Löschen" im Footer** (danger-toned `--nm-danger`, natives `confirm()`) über `DELETE /api/cards/<id>` → rückt zur nächsten Karte vor bzw. re-lädt zum Empty-State; alle state-changing Requests (inkl. DELETE) über den globalen `base.html`-CSRF-fetch-Wrapper. DS-konform (`c-surface--flat`, „Gut"-primary, Status-als-Tint, Cloze XSS-safe via DOM-Nodes, token-driven). **Card-Lifecycle damit komplett** (R4-LEARN-FIX): erzeugen/patchen = **Agent** (Token), bewerten/annotieren/**löschen** = **User** (Session).

## Knoten 12 — Reader-Mode & geteiltes „Aa"-Popover (READER-ADJ)

**Entscheidung** (READER-ADJ, 2026-06-22, reine Frontend-Welle): die Lese-Steuerung ist in **drei klar getrennte Schichten** aufgeteilt. Der langjährige Missstand war eine floating `.reader-toolbar` im markdown-converter, die **dauerhaft über dem Text hing**. Klärung mit Oli: „Reader" war historisch als die Distraction-Free-Ansicht umgesetzt worden (Sidebars ein/aus); gemeint war eigentlich der markdown-converter-artige Reader-Mode (Spaltenbreite + Textgröße). Die drei Schichten koexistieren jetzt:

| Schicht | Was | Wo | Mechanik |
|---|---|---|---|
| **Distraction-Free-Floater** | feingranulares Sidebar-Collapse (global links / Detail rechts) | `library_detail` + base.html | `.sidebar-toggle-floater`, `body.global-sidebar--collapsed`/`detail-sidebar--collapsed` — **unangetastet** |
| **Reader-Mode** | fokussierte, zentrierte Leseansicht mit **Breite + Textgröße**, umgebende Chrome ausgeblendet | markdown-converter (Preview-iframe) **+** `library_detail` (`.reader-view`) | `body.reader-active` (Chrome-Hide) + `--reader-width`/`--reader-font-size` |
| **„Aa"-Popover** | on-demand-Controls (A−/A+ · 4 Breite-Presets · [nur markdown: Dark] · Exit) | beide Flächen, geteilt | dezenter Eck-Trigger → Popover, **Outside-Click + Esc** schließt; **nie** eine Dauer-Leiste überm Text (Safari-/Kindle-Muster) |

**Geteilte Komponente** `static/js/reader_settings.js` — `ReaderSettings.create({target, trigger, popover, onChange?, onDark?, onExit?})`: extrahiert die vorhandene markdown-converter-Logik (`changeFontSize`/`changeWidth`/`WIDTH_MAP`/`applyWidth`/`updateWidthButtons`/`getReaderPrefs`/`saveReaderPrefs`) **generisch** über einen **Ziel-Container als Parameter** statt fix `.main-container`. Deklarative `data-reader-*`-Verdrahtung (kein inline-onclick); `handleEscape()` mit **Zwei-Stufen-Contract** (offenes Popover → erst schließen, ein zweites Esc verlässt erst dann den Reader-Mode); `onDark`/`onExit` optional → der jeweilige Button wird `.remove()`d, wenn der Consumer ihn nicht liefert (so kriegt die Library automatisch **keinen** Dark-Button). Markup geteilt via `templates/_partials/reader_aa.html` (Jinja-Flags `reader_aa_dark`/`reader_aa_exit`), CSS token-driven `.reader-aa-*` (kein per-Theme-Override → Dark gratis).

**Consumer-Unterschiede**:
- **markdown-converter** (`target = .main-container`): `onChange = renderIframe` (Textgröße ist in das Preview-iframe-`srcdoc` gebacken → Re-Render bei Font-Change; Width ist reine äußere CSS-Var, kein Re-Render), `onDark` (reader-scoped Dark-Toggle des Preview-iframe), `onExit`; **persistiert `readerPrefs.modeOn`** → die Seite stellt den Reader-Mode beim Laden wieder her.
- **library_detail** (`target = #content-body`): **kein `onDark`** (folgt dem **globalen** Theme), `onExit`; `body.library-reader`-Layout blendet Header/Btn-Row/Sidebar-Spalte/Floater aus (reuse von `body.reader-active` + library-spezifische Regeln, `.max-w-6xl`-Cap aufgehoben damit ultrawide viewport-relativ wirkt) und zentriert `#content-body` auf `--reader-width`; **kein `modeOn`-Write** (gehört dem markdown-converter-Auto-Restore) → jede Detailseite öffnet in Normalansicht, nur `width`/`fontSize` sind geteilt (**Cross-Reader-Konsistenz**). **Highlighting · Reading-Progress · Distraction-Free-Floater bleiben im Reader-Mode aktiv** (koexistieren).

**Esc-Guard-Lehre** (per echtem Maus-Drag gefangen): der vom markdown-converter übernommene Esc-Guard listete `BUTTON` — nach einem **echten** Klick auf „Reader Mode" behält der (dann versteckte) Button den Fokus → der erste Esc wurde geschluckt, primärer Exit kaputt. Synthetisches `.click()` maskierte es (es bewegt den Fokus nicht). Fix in der Library: `BUTTON` aus dem Guard raus (Esc verlässt auch bei fokussiertem Control) + Blur-on-Enter; `TEXTAREA`/`INPUT`/`SELECT` bleiben geschützt (Highlight-Notiz/Tag-Input). Verstärkt Memory `feedback_selection_anchor_coordinate_system` (echter Drag deckt auf, was synthetisch maskiert bleibt); verallgemeinerte Lehre → Memory `reference_shared_reader_settings_aa_popover`.

## Knoten 13 — LaTeX-Mathe-Rendering: schützen + rendern (MATH-RENDER)

**Problem** (Oli 2026-06-23, Reader-Screenshot): Markdown mit LaTeX-Mathe (`$…$` inline, `$$…$$` Block) wurde **roh** angezeigt — `\frac{dC}{dt}`, `C_{\text{in}}` etc. als Quelltext. Der geteilte `render_markdown_to_html` hatte kein Mathe-Plugin, im Frontend kein KaTeX. Eine **rein client-seitige** Lösung wäre fragil: der Markdown-Inline-Parser frisst `_`/`\`/`{}` schon vor jedem Client-Render.

**Entscheidung: schützen (Server) + rendern (KaTeX client), drei Asset-Strategien je Render-Kontext.**

**1. Schützen — `dollarmath` im geteilten Renderer** ([app_pkg/markdown_render.py](../app_pkg/markdown_render.py)): `dollarmath_plugin` aus `mdit-py-plugins` tokenisiert die Mathe **vor** der Markdown-Inline-Zerlegung. **Konservativ konfiguriert** (`allow_space=False`, `allow_digits=False`, `allow_labels=False`, `double_inline=False`) → Streu-`$` (Preise wie „5$"/„10$"/„$ 5 $") bleibt **Text**, wird nicht zu Mathe. **Eigene Render-Rules** (statt der Plugin-Defaults `math inline`/`math block`) erzeugen class-getaggte Spans: `<span class="math-inline">` / `<span class="math-display">` mit dem **rohen LaTeX** als `escapeHtml`'ter Body. Zwei Eigenschaften fallen daraus:
- **nh3 ohne Allow-List-Touch**: `span` + `class` sind in `_ALLOWED_TAGS`/`_ALLOWED_ATTRIBUTES` schon erlaubt → das Markup übersteht die Sanitisierung unverändert; der Display-Modus reist über die **Klasse**, nicht ein `data-`-Attr.
- **XSS-safe**: der `escapeHtml`'te Body neutralisiert `$<script>$`; KaTeX liest später den **dekodierten `textContent`** und rendert ihn als Mathe, nie als HTML.

**2. Rendern — KaTeX pro Fläche** (vendored `static/vendor/katex/`, 0.16.11, CSS+JS+20 woff2-Fonts; woff2-first → keine Font-404). Die Render-JS ist überall identisch (`.math-inline`/`.math-display` → `katex.render(el.textContent, el, {displayMode, throwOnError:false})`), aber **die Asset-Auflösung unterscheidet sich je Kontext**:

| Fläche | Render-Trigger | Asset-Strategie | Warum |
|---|---|---|---|
| **Reader** (`library_detail`) | `renderMath()` in [library_detail.js](../static/js/library_detail.js) on load, **vor den Highlights** | KaTeX-CSS als `<link>` (`head_extra`) + JS vendored | normale Static-URL löst auf; `throwOnError:false` = kaputte Formel bleibt Quelltext statt Seiten-Crash |
| **markdown-converter-Preview** | `markdown-it-texmath` (vendored) rendert `$…$`/`$$…$$` direkt mit `katex.renderToString` in den Token-Stream | iframe-`<head>` kriegt **nur** den KaTeX-CSS-`<link>` (absolute URL via `PageData.katexCssUrl`) | die gerenderte KaTeX-HTML kommt aus dem **Parent** → der iframe braucht kein JS, nur CSS; absolute URL → `@font-face`-`fonts/`-URLs lösen relativ zur **CSS-Datei** auf (nicht zur iframe-Base), kein 404 |
| **PDF-Export** (Playwright) | `page.evaluate` rendert die Spans nach `set_content`, **vor** `page.pdf()` (synchron, keine Race) | `_katex_pdf_assets()` ([app_pkg/markdown.py](../app_pkg/markdown.py)) inlinet **alles**: CSS mit woff2-Fonts als **data-URIs** (woff/ttf-Fallbacks rausregext) + JS als Inline-Script | `set_content` lädt mit **about:blank-Base** → externe/relative URLs lösen **nicht** auf; nur ein self-contained Bundle rendert mit Fonts. Render **vor** `document.fonts.ready`, damit KaTeX' data-URI-Fonts in den Fonts-Wait einfließen |

**Render-vor-Highlights-Ordnung** (Reader): `renderMath()` läuft **vor** `initHighlights()` — sonst säßen die Text-Quote-Anker der Highlights auf rohem LaTeX, das KaTeX dann unter ihnen wegrendert (Anker-Drift auf Mathe-Docs). Bewusst geordnet.

**`texmath`-Config spiegelt den Server**: `delimiters:'dollars'` + `outerSpace:false` → dieselbe konservative `$`-Semantik wie `dollarmath` (Preise/Spaces bleiben Text), damit Preview == Reader/PDF.

**Bewusst out — EPUB/Kindle-Math** (Folge-Item, BACKLOG P3): das EPUB wird aus demselben `render_markdown_to_html` gebaut und trägt die Mathe als class-getaggte LaTeX-Spans — also **roh wie bisher, nicht schlechter**. Client-KaTeX im EPUB ist unzuverlässig (E-Reader-JS); echter EPUB-Math bräuchte **Server-Render zur Build-Zeit** (KaTeX→MathML oder KaTeX-HTML+inline-CSS) im `epub_service` → eigener Sprint. Verallgemeinerte Lehre → Memory `reference_math_protect_then_render`.

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
 ├── highlights   (M:N via highlight_tags — R1-B-C)
 └── cards        (M:N via card_tags — R4-LEARN)

Card (NEW — R4-LEARN)
 ├── user_id FK              (Owner-Scope — Karte überlebt ihr Highlight)
 ├── highlight_id FK         (nullable = Best-Effort-Provenienz; before_delete-Event nullt es)
 ├── source_snapshot/_doc_title  (Authoring-Snapshot — Review liest das Highlight nie live)
 ├── type (atomic|generative), front/back/cloze_text/prompt, note
 ├── state (ok|wackelt), created_by
 ├── review                  (NEW 1:1 — ORM cascade delete-orphan)
 └── tags                    (M:N via card_tags)

Review (NEW — R4-LEARN, 1:1 zu Card)
 └── due/stability/difficulty/last_reviewed/reps/lapses/rating_history  (FSRS-State)
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
| 2026-06-15 | R2-H: die drei *Bedien*-Achsen (Ort/Lese-Liste-Flag/Favorit) + die zwei Nav-Listen (Tabs+Ort-Chips) kollabieren auf **EINE flache Single-Select-Achse mit vier exklusiven Orten** (Inbox/Lese-Liste/Bibliothek/Archiv), identisch Tab↔Karte↔Detail (Knoten 10) | Drei orthogonale Achsen + zwei Listen waren als Oberfläche verwirrend („wofür ist die Flag / Später=WTF"). Die Orte werden aus `lifecycle_status`+`queue_position` **abgeleitet** (Präzedenz archive>queued>inbox>neutrales Regal) — **kein Schema-Touch** (das Knoten-6/7-Schema bleibt, nur die Bedien-Schicht ist ersetzt). Eine `POST /place`-Move-Aktion setzt die Kombi + hält Exklusivität; `is_favorite` liegt brach (reversibel). Memory `reference_collapse_orthogonal_axes_to_flat_single_select` |
| 2026-06-15 | R2-H: **Archivieren dequeued** (`queue_position=NULL`) — **Revision** der R2-D/Knoten-6-Entscheidung „Archiv ∩ Lese-Liste via View-Filter, kein Auto-Dequeue" | Im exklusiven Vier-Orte-Modell kann ein Item nicht zugleich Archiv und Lese-Liste sein — die Orte schließen sich aus, also nullt jeder Move (auch Archiv) die fremden Spalten. Der Hoch/Runter-Swap bleibt über die sichtbare Menge (Knoten 6 unverändert) |
| 2026-06-15 | R2-H: **Suche/Tag/Typ = globaler Finder** über alle Nicht-Archiv-Orte — Master-Umlenkung in P2A: **jeder** Content-Filter, nicht nur Suche | „Alles sehen" = filtern (Olis Wahl) statt eines „Alle"-Tabs. Bibliothek-Tab ohne Filter = neutrales Regal; mit Filter spannt er Inbox+Lese-Liste+Bibliothek. Der Default-Scope war zuerst „nur Suche global", per Master-Diff-Read auf `search OR tag OR type` erweitert — der eine bewusst smoke-justierbare Punkt |
| 2026-06-19 | R4-LEARN: `card`/`review`/`card_tags` als neue Tabellen via `db.create_all`, **keine Migration**; Karte **self-contained** + `Review` als eigene 1:1-Tabelle | Neue Tabellen brauchen keinen `_run_pending_migrations`-Eintrag (der ist nur für Spalten-Adds an *bestehenden* Tabellen). Self-contained, weil das Review das Highlight nie live lesen darf (Highlight kann gelöscht/editiert werden); `Review` eigen statt auf die Karte gemerged → spätere Review-History sauber. `Card.user_id` ergänzt (Owner-Scope), weil die Karte ihr nullbares Highlight überlebt (Knoten 11) |
| 2026-06-19 | R4-LEARN: Lösch-Mechanik **ORM-`before_delete`-Event** statt DB-`ON DELETE SET NULL` | Kein `PRAGMA foreign_keys=ON` in SQLite → deklarierte FK-Actions sind inert; das Event nullt `card.highlight_id` per Core-UPDATE auf der Flush-`connection` (greift direkt + über den Conversion-`delete-orphan`-Cascade). Generalisierter, wiederkehrender Trap → Memory `reference_sqlite_no_fk_pragma_orm_delete` |
| 2026-06-19 | R4-LEARN: **Drei-Wege-Auth-Split** — Agent-Write=Token (`CARD_TOKEN`), User-Rate=Session, User-Annotate=Session; Reader=Session | Drei Akteure/Trust-Boundaries: der Agent schreibt session-los (Ingest-Token-Muster, eigenes Token für unabhängige Rotation), der User bewertet/vertieft in der UI (Session+CSRF). `POST /annotate` ist ein eigener Session-Pfad, weil `PATCH /api/cards` token-only ist — der Browser hat kein `CARD_TOKEN`. **Rate + Annotate sind UI-only** (nicht für den converter-mcp-Agent wrappen) |
| 2026-06-19 | R4-LEARN: **FSRS hinter swappable Scheduler-Schnittstelle** (SM-2-Fallback, Config), FSRS-`state`/`step` nicht persistiert | Engine austauschbar (`SCHEDULER_ENGINE`/`FSRS_DESIRED_RETENTION`) hinter einer dict-basierten ABC, damit der Endpoint+Schema engine-agnostisch bleiben; `fuzzing` aus für deterministische Intervalle. `reps`/`lapses` besitzen wir (FSRS-6-Card trackt sie nicht). Das gesperrte `Review`-Schema hat keine `state`/`step`-Spalte → graduierte Rekonstruktion (Intervall-Math erhalten, nur Sub-Day-Learning-Rampe kollabiert). Memory `reference_swappable_scheduler_interface` |
| 2026-06-20 | R4-LEARN-FIX: **`DELETE /api/cards/<id>`** (Session, owner-scoped, ORM-Cascade) + „Löschen" im Review-Footer → **Card-Lifecycle komplett** | Die R4-LEARN-Lücke aus dem end-to-end-Smoke: „Agent erzeugt / User bewertet" gebaut, „User löscht" vergessen. `@login_required` + `first_or_404` über `Card.user_id` (fremd/fehlend 404); `db.session.delete(card)` → ORM nimmt `review` (`cascade='all, delete-orphan'`) + `card_tags` (secondary) mit — **kein Raw-SQL** (sonst Orphans, FK-Pragma inert, vgl. Knoten 11 / `reference_sqlite_no_fk_pragma_orm_delete`). Session-Write, **nicht** CSRF-exempt (DELETE läuft über den `base.html`-fetch-Wrapper). UI: danger-toned Footer-Button + natives `confirm()`, vorrücken/Empty-State. **Nicht** in den converter-mcp wrappen (UI-only) |
