# Reader-Architecture — Entscheidungs-Memo

**Stand**: 2026-05-25
**Workshop-Datum**: 2026-05-25 (Master-Workshop nach R1-A done)
**Status**: Aktive Referenz für R1-B + R2 + R3 Sprints — nicht archiviert.

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
liefert alle drei Werte direkt (`Selection.toString()` plus Range-Context-Walks), W3C-Standard für
Web-Annotations (Hypothes.is-Kompatibilität als Future-Option), pure client-side.

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
wird in R2-A migriert — Parsing der CSV-Strings → Tag-Inserts → `conversion_tags`-Junction-Inserts.
CSV-Spalte kann danach gelöscht werden (manuelles ALTER TABLE) oder als Dead-Column liegenbleiben.
Master-Disposition zu R2-A-Zeit.

## Knoten 5 — Class-Naming

**Entscheidung**: `Conversion`-Class-Name bleibt unverändert.

**Begründung**: YAGNI. UI-Strings sagen eh „Library" / „Dokument", der Class-Name `Conversion` lebt
nur im Code. Rename wäre großer Code-Churn ohne UI-Wert. Wenn der Name irgendwann ärgert: eigener
Cleanup-Sprint mit clean ALTER TABLE.

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
