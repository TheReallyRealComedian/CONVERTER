# Sprint R2-A — `conversion_tags`-Junction + CSV-Migration

**Datum**: 2026-05-25

**Ziel**: Die existierende `Conversion.tags`-CSV-Spalte durch eine `conversion_tags`-Junction-Tabelle ablösen — Conversions teilen sich denselben `Tag`-Namespace mit Highlights (aus R1-B-C). Bestehende CSV-Daten werden idempotent in die Junction migriert. Frontend wird auf den neuen Tag-Picker umgestellt (Sidebar + Library-Card-Strip). Erster R2-Cluster-Sprint, der die Tag-Foundation aus R1-B-C aufgreift und auf Conversions ausdehnt.

**Vorbedingung**:
- Pytest 106/106 grün auf `main` (zuletzt R1-B-C done, commit `099f041`).
- Architektur-Memo [docs/reader_architecture.md](docs/reader_architecture.md) Knoten 4: Tag + zwei Junction-Tabellen, beide M:N mit `Tag`. R1-B-C hat `highlight_tags` geliefert. R2-A liefert `conversion_tags` plus CSV-Migration.
- Tag-API-Pattern aus R1-B-C ([app_pkg/tags.py](app_pkg/tags.py)) ist Source-of-Truth für find-or-create + Normalisierung (lowercase+trim). Pattern wird für Conversion-Tag-Endpoints wiederverwendet.
- Memory `reference_inline_sqlite_migration.md` ist Source-of-Truth für den Migration-Helper-Pattern. `_run_pending_migrations(app)` wird hier erweitert.
- Existing Code-Stellen, die CSV-Tags nutzen (zu migrieren):
  - [templates/library.html:76-78](templates/library.html:76) — Card-Strip via `conv.tags.split(',')`
  - [templates/library_detail.html:159-161](templates/library_detail.html:159) — `tags-input` + `tag-chip-container`
  - [static/js/library_detail.js:20](static/js/library_detail.js:20) — `AUTOSAVE_INPUTS = { title: 'detail-title', tags: 'tags-input' }`
  - [static/js/library_detail.js:281,424,427,469-470](static/js/library_detail.js:281) — vier CSV-Mechanik-Stellen
  - [app_pkg/library.py:156](app_pkg/library.py:156) — PATCH-Handler `conversion.tags = str(data['tags'])[:500]`
- Container-DB-Smoke-Foundation: einige bestehende Conversions haben CSV-Tags (z.B. doc 2 evtl. mit "ki, produkt" o.ä.). **Sub-Thread soll in Pre-Flight die CSV-Tag-Population checken** (`SELECT id, tags FROM conversion WHERE tags != ''`) — das ist die Migration-Smoke-Foundation.

**Out-of-scope**:
- **Lifecycle-Status (Inbox/Later/Archive)** — ursprünglich im Architektur-Memo bei R2-A erwähnt, hier aber bewusst ausgegliedert in einen separaten **R2-C-Sub-Sprint** (Sprint-Schneidung-Entscheidung beim Schreiben dieses Prompts: R2-A ist mit Tag-Migration + Frontend-Umstellung schon L, Lifecycle würde XL daraus machen). Sub-Thread legt R2-C als BACKLOG-Item nach R2-A-Abschluss an, Architektur-Memo wird entsprechend aktualisiert.
- **R2-B Filtered Views + Reading-Progress** — eigener Sub-Sprint nach R2-A.
- **CSV-Column komplett DROP** (`ALTER TABLE Conversion DROP COLUMN tags`) — SQLite-DROP-Column ist Table-Rebuild, riskant, in einem separaten Cleanup-Sprint später. Hier bleibt die Spalte als Dead-Column liegen (nach Migration leer).
- **Tag-Rename, Tag-Color-Coding, Tag-Bulk-Operations** — wie R1-B-C: YAGNI für R2-A.
- **CSV-Backward-Compat im PATCH-Handler** — nach R2-A nutzt das Frontend keine CSV-Updates mehr; PATCH-Handler-Behandlung des `tags`-Felds kann bleiben (Dead-Path) oder entfernt werden — Sub-Thread-Disposition.

---

## Phase 1 — Implementation

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. `pytest tests/` → 106/106 grün als Baseline.
3. Mac-Stack live. Container-DB-CSV-Tag-Status checken:
   ```bash
   docker compose exec markdown-converter python -c "from app import app; from models import Conversion; app.app_context().push(); [print(c.id, c.tags) for c in Conversion.query.filter(Conversion.tags != '').all()]"
   ```
4. **Architektur-Memo Knoten 4** lesen + **Memory `reference_inline_sqlite_migration.md`** für Migration-Helper-Pattern lesen.

### Files

```
models.py                     # EDIT — conversion_tags-Junction-Table, Conversion.tag_refs-M:N, to_dict erweitert, Tag.get_or_create-classmethod
app_pkg/__init__.py           # EDIT — _run_pending_migrations um _migrate_conversion_tags_csv_to_junction(app) erweitert
app_pkg/library.py            # EDIT — neue Routes für Conversion-Tag-Attach/Detach analog R1-B-C; bestehender PATCH-Handler-tags-Pfad Dead-Code-Disposition
app_pkg/tags.py               # EDIT — get_or_create-Logik aus POST-Route in models.py umziehen, beide Call-Sites (Highlights + Conversions) konsumieren den Helper; GET /api/tags zählt nun Highlights + Conversions je Tag
app_pkg/highlights.py         # EDIT — to_dict bleibt; falls find-or-create-Logik dort indirekt war: jetzt Tag.get_or_create()
templates/library.html        # EDIT — Card-Strip aus conv.tag_refs (Tag-Objects) statt conv.tags-CSV-Split
templates/library_detail.html # EDIT — tags-input + tag-chip-container ersetzt durch Inline-Tag-Picker (analog R1-B-C Popover-Variante, aber inline in der Sidebar): Chips + Add-Input mit datalist + X-Detach
static/js/library_detail.js   # EDIT — CSV-Mechanik raus (AUTOSAVE_INPUTS.tags weg, 4 CSV-Stellen weg), neue Tag-Picker-Logik analog Highlight-Tag-Picker (loadConversionTagSuggestions, addTagToConversion, removeTagFromConversion, renderConversionTagChips)
static/css/style.css          # EDIT — bestehende `.tag-chip`-Klasse evtl. konsolidieren mit `.highlight-tag-chip` (R1-B-C-Disposition besagte: Namespace-Trennung bis R2-A — jetzt Konsolidations-Punkt erreicht); oder weiterhin getrennte Klassen-Familien wenn das Detail-View visuell anders sein soll. Sub-Thread disponiert.
tests/test_conversion_tags.py # NEU — Migration-Idempotenz-Tests, API-Tests für Conversion-Tag-Routes
tests/test_tags.py            # EDIT — GET /api/tags-Test um Conversion-Count erweitern, get_or_create-Helper-Test
```

Elf Files: eine Neuanlage (`test_conversion_tags.py`), zehn Edits. Großer Sprint.

### Mechanik

**1. Schema-Touch ([models.py](models.py))**:

Neue Junction-Tabelle (analog `highlight_tags`):

```python
conversion_tags = db.Table(
    'conversion_tags',
    db.Column('conversion_id', db.Integer, db.ForeignKey('conversion.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True),
)
```

Backref am `Conversion`-Model:

```python
tag_refs = db.relationship('Tag', secondary=conversion_tags, lazy='joined',
                           backref=db.backref('conversions', lazy='dynamic'))
```

**Naming**: `tag_refs` (nicht `tags`), weil die existierende CSV-Spalte `tags = db.Column(db.String(500))` weiterhin existiert (Dead-Column nach Migration). Klare semantische Trennung. Wenn die CSV-Spalte später gedroppt wird (Cleanup-Sprint), kann `tag_refs` zu `tags` umbenannt werden — bis dahin koexistieren beide Namen.

`Conversion.to_dict()` erweitert um:
```python
'tag_refs': [t.to_dict() for t in self.tag_refs]
```

CSV-Spalte `tags` bleibt im `to_dict()` (Backward-Compat für nicht-touched Code-Pfade), aber das Frontend ignoriert sie nach Migration.

**Tag-Helper als classmethod**:

`get_or_create` von der inline-Logic in `app_pkg/tags.py:POST` in `models.py:Tag` umziehen:

```python
@classmethod
def get_or_create(cls, user_id, name):
    normalized = (name or '').strip().lower()
    if not normalized:
        return None
    tag = cls.query.filter_by(user_id=user_id, name=normalized).first()
    if tag is None:
        tag = cls(user_id=user_id, name=normalized)
        db.session.add(tag)
        db.session.flush()  # für tag.id ohne commit
    return tag
```

Beide Call-Sites (Highlight-Tag-POST in `app_pkg/tags.py` und neue Conversion-Tag-POST in `app_pkg/library.py`) plus der Migration-Helper konsumieren denselben classmethod. **DRY-Pflicht**.

**2. Migration-Helper ([app_pkg/__init__.py](app_pkg/__init__.py))**:

`_run_pending_migrations(app)` wird um einen neuen Schritt erweitert:

```python
def _migrate_conversion_tags_csv_to_junction(app):
    """R2-A: CSV-Tags in conversion_tags-Junction migrieren.
    
    Idempotent: Migration läuft nur für Conversions mit non-empty tags-CSV.
    Nach erfolgreicher Migration wird tags-CSV auf '' gesetzt (Migration-Marker).
    Beim nächsten Container-Start: tags == '' → no-op.
    """
    from models import Conversion, Tag, db
    candidates = Conversion.query.filter(
        Conversion.tags != None,
        Conversion.tags != ''
    ).all()
    if not candidates:
        return
    migrated = 0
    for conv in candidates:
        names = [n.strip() for n in conv.tags.split(',') if n.strip()]
        for name in names:
            tag = Tag.get_or_create(conv.user_id, name)
            if tag and tag not in conv.tag_refs:
                conv.tag_refs.append(tag)
        conv.tags = ''  # Migration-Marker — nicht mehr antasten
        migrated += 1
    db.session.commit()
    app.logger.info(f"R2-A: migrated {migrated} conversions from CSV to conversion_tags junction")
```

Aufruf-Stelle: nach dem bestehenden Highlight-Note-Migrations-Block in `_run_pending_migrations(app)`.

**Idempotenz-Mechanik via leerer CSV**:
- Nach Migration einer Conversion: `tags = ''` setzt sie auf no-match für `Conversion.tags != ''`-Filter.
- Bei zweitem Container-Start: `candidates` ist leer (oder enthält nur neue Conversions, die seit dem letzten Run mit nicht-leerer CSV erstellt wurden — was es nach Frontend-Migration nicht mehr geben sollte).
- **Defense gegen User-Detach-dann-Restart-Race**: wenn ein User einen Conversion-Tag im Frontend detached und Container neu gestartet wird, würde die alte CSV-Backup-Spalte den Tag NICHT wiederherstellen — denn die CSV ist nach erstem Migration-Lauf leer.
- Edge-Case: wenn nach R2-A noch jemand händisch via SQL die `tags`-Spalte einer Conversion füllt, würde der nächste Container-Start die wieder migrieren. Akzeptiert (theoretisches Szenario).

**3. Backend-Routes ([app_pkg/library.py](app_pkg/library.py))**:

Neue Routes innerhalb des `register(app)`-Blocks (oder als separater `library`-naher Modul falls Sub-Thread es übersichtlicher findet):

| Route | Methode | Body | Response |
|---|---|---|---|
| `/api/conversions/<int:conversion_id>/tags` | POST | `{"name": str}` | 201 + Tag.to_dict() (find-or-create + attach), 200 + Tag.to_dict() bei Already-Attached (no-op-Pattern aus R1-B-C) |
| `/api/conversions/<int:conversion_id>/tags/<int:tag_id>` | DELETE | — | 200 + `{"success": true}` (detach, Tag bleibt) |

Validation identisch zu R1-B-C Highlight-Tag-Routes: `name` normalisiert (lowercase+trim, via `Tag.get_or_create`), Ownership via `get_owned_conversion`, max 80 chars.

**Bestehender PATCH-Handler-`tags`-Pfad** ([library.py:147-148](app_pkg/library.py:147)):
```python
if 'tags' in data:
    conversion.tags = str(data['tags'])[:500]
```

Sub-Thread-Disposition: **Default = entfernen** (Dead-Path nach Migration, kein Frontend nutzt es mehr). Alternative wenn Sub-Thread konservativ bleiben will: belassen als no-op-Akzeptanz (verhindert API-Bruch bei alten Clients), aber dann mit Kommentar markieren.

**4. Tag-API erweitert ([app_pkg/tags.py](app_pkg/tags.py))**:

`GET /api/tags` zählt aktuell nur Highlights. Erweitern um Conversion-Count:

```python
# Pseudo: zwei Subqueries oder ein Join mit beiden Junctions
{ 'id': ..., 'name': ..., 'highlight_count': N, 'conversion_count': M }
```

Tag-Manager-Page rendert beide Counts.

**5. Frontend Library-List ([templates/library.html](templates/library.html))**:

Bestehende Schleife (Z.76-78):
```html
{% if conv.tags %}
  {% for tag in conv.tags.split(',') %}
    <span class="tag-chip">{{ tag.strip() }}</span>
  {% endfor %}
{% endif %}
```

Migrieren auf:
```html
{% if conv.tag_refs %}
  {% for tag in conv.tag_refs %}
    <span class="tag-chip">{{ tag.name }}</span>
  {% endfor %}
{% endif %}
```

Backend `library`-Route übergibt Conversions schon via `pagination.items` ans Template — `tag_refs`-Relationship lädt automatisch via `lazy='joined'`.

**6. Frontend Library-Detail Tag-Picker ([templates/library_detail.html](templates/library_detail.html))**:

Bestehender Sidebar-Tags-Block (Z.157-161):
```html
<input type="text" id="tags-input" value="{{ conversion.tags }}"
       placeholder="kommagetrennte Tags" onchange="updateField('tags', this.value)">
<div id="tag-chip-container"></div>
```

Ersetzen durch:
```html
<div id="conversion-tag-chips" class="flex flex-wrap gap-1 mb-2"></div>
<div class="flex gap-1">
  <input type="text" id="conversion-tag-input" list="conversion-tag-suggestions"
         placeholder="Tag hinzufügen..." class="c-input text-sm flex-1"
         autocomplete="off" maxlength="80">
  <button type="button" class="c-btn text-xs" onclick="addTagToConversion()">+</button>
</div>
<datalist id="conversion-tag-suggestions"></datalist>
```

Initial-Befüllung: `conversion-tag-chips` aus `{{ conversion.tag_refs }}` server-rendered, oder per JS bei page-load via `GET /api/conversions/<id>` (oder direkt aus PageData). Sub-Thread disponiert — pragmatischer Default: PageData-embed der Tag-Liste aus `conversion.to_dict()`, JS rendert.

**7. JS ([static/js/library_detail.js](static/js/library_detail.js))**:

*Entfernen* (Dead-Code nach Migration):
- `AUTOSAVE_INPUTS.tags` (Z.20) raus
- 4 CSV-Stellen (Z.281, 424, 427, 469-470) raus oder refactor
- `tag-chip-container`-Render-Logik (bisherige CSV-zu-Chip-Funktion)

*Neu* (analog Highlight-Tag-Picker aus R1-B-C):
- `loadConversionTagSuggestions()` — fetch `GET /api/tags`, befüllt `<datalist id="conversion-tag-suggestions">`
- `addTagToConversion()` — POST `/api/conversions/<id>/tags`, state-Update, re-render Chips
- `removeTagFromConversion(tagId)` — DELETE, state-Update, re-render
- `renderConversionTagChips()` — baut Chips aus state
- Init beim DOM-Ready: `loadConversionTagSuggestions()` + initiales Chip-Render aus PageData

PageData-Erweiterung: `window.PageData.conversionTags = [{id, name}, ...]` (vom Template injiziert aus `conversion.tag_refs`).

**8. CSS ([static/css/style.css](static/css/style.css))**:

**Konsolidations-Entscheidung**: R1-B-C hat bewusst zwei Tag-Chip-Familien getrennt (`.tag-chip` für CSV-Strip in Library-List + Detail-Sidebar, `.highlight-tag-chip` für die neuen Highlight-Tags). Begründung damals: das CSV-Pattern blieb noch unangetastet.

Jetzt mit R2-A wird die CSV abgelöst. Drei Optionen:

- **A**: Klassen-Familien konsolidieren — `.tag-chip` und `.highlight-tag-chip` werden eine Familie. Vorteil: weniger CSS, konsistent. Risiko: visueller Unterschied zwischen Conversion-Tags und Highlight-Tags geht verloren.
- **B**: Klassen-Familien behalten — Conversion-Tags nutzen `.tag-chip` (alt), Highlight-Tags nutzen `.highlight-tag-chip` (neu). Visueller Unterschied bleibt (primäre Doc-Tags vs. sekundäre Annotations-Tags). Aber: doppelter Maintenance.
- **C**: `.tag-chip` als Base-Klasse + `.tag-chip--highlight` als Modifier. Sauber, aber Refactor von R1-B-C-CSS.

Master-Empfehlung: **B**. Keine Konsolidation in R2-A — das ist eine UI-Design-Entscheidung, die in einem Polish-Sprint später kommen sollte. R2-A behält die getrennten Familien, Memory `feedback_css_class_collision_in_markdown_views.md` ist nicht verletzt (Konsolidation ist optional, nicht Pflicht).

Plus: CSS-TOC-Eintrag falls neu nötig (vermutlich nicht, wenn beide Klassen schon existieren).

**9. Tests ([tests/test_conversion_tags.py](tests/test_conversion_tags.py) NEU + [tests/test_tags.py](tests/test_tags.py) EDIT)**:

**`tests/test_conversion_tags.py` neu**:

| Test | Erwartung |
|---|---|
| POST tag to conversion (new) → 201, Tag erstellt, attached an Conversion | grün |
| POST tag to conversion (existing, anderer Highlight oder Conversion) → 201/200, Tag reused | grün |
| POST tag to conversion (already attached) → 200 no-op | grün |
| POST oversized/empty name → 400 | grün |
| POST foreign conversion → 404 | grün |
| DELETE detach → 200, Junction weg, Tag bleibt | grün |
| DELETE foreign-user-Tag-detach → 404 | grün |
| Migration-Helper Idempotenz: Pytest mit pre-seeded CSV → run → Tag-Tabelle + Junction-Counts korrekt, CSV-Spalte leer | grün |
| Migration-Helper Idempotenz: second run → no-op, kein Duplikat | grün |
| Migration-Helper CSV-Trim-und-Lowercase: " KI , produkt " → 2 Tags „ki" + „produkt" | grün |
| Migration-Helper leere/whitespace-only Einträge in CSV werden gefiltert | grün |
| Cross-Domain-Tag-Reuse: derselbe Tag-String wird vom Highlight-Pfad UND Conversion-Pfad als dieselbe Tag-Row genutzt | grün |
| Conversion.to_dict() enthält tag_refs-Liste | grün |
| Tag.to_dict() für GET /api/tags zeigt highlight_count + conversion_count | grün |

Mindestens **10 neue Tests**. Plus Anpassungen in `tests/test_tags.py` für den erweiterten `GET /api/tags`-Test (jetzt mit `conversion_count`). Plus `Tag.get_or_create`-Helper-Test (falls als separate Unit testbar).

Pytest auf **116+** grün.

### Code-Quality-Gates

- UI-Strings deutsch: „Tag hinzufügen...", „Tags", „Tag entfernt." — Microcopy konsistent mit R1-B-C.
- `showToast` für Banner.
- CSRF transparent via base.html-fetch-Wrapper.
- DRY-Pflicht: `Tag.get_or_create()` Single-Source-of-Truth, 3 Call-Sites (Highlight-Tag-POST, Conversion-Tag-POST, Migration-Helper).
- Live-Smoke nach Frontend-Änderung Pflicht.
- Pytest 116+/116+ grün vor Phase-Ende.

### Phase-1-Stop

Nach Phase 1: STOP — Bericht. Welche Conversions in der Container-DB hatten CSV-Tags (Migration-Output-Log), wie viele Tag-Rows wurden via Migration erstellt (vs. wieviele existierten schon aus R1-B-C), CSS-Konsolidations-Disposition (A/B/C), Dead-Path-PATCH-Disposition (entfernt oder belassen), Test-Count vorher/nachher.

---

## Phase 2 — Verify

**Pytest**:

1. `docker compose exec markdown-converter pytest tests/` → 116+/116+ grün.

**Migration-Idempotenz-Smoke**:

2. Vor Phase 2: Container-Logs der ersten Start-Sequenz nach Image-Rebuild prüfen — Log-Line `R2-A: migrated N conversions from CSV to conversion_tags junction` muss erscheinen (oder nicht, falls keine CSV-Tags in der DB). Container neu starten — Log-Line darf beim Restart NICHT erneut erscheinen.
3. DB-Probe: `SELECT id, tags FROM conversion WHERE tags != ''` muss leer sein. `SELECT * FROM conversion_tags` muss die migrierten Junction-Rows zeigen.

**Live-Smoke**:

4. **Library-List**: doc-Cards mit migrierten Tags müssen die Chips im Card-Strip zeigen. Vergleich mit DB-Probe: jeder Tag aus der Junction muss als Chip erscheinen.
5. **Library-Detail** doc mit Tag öffnen: Sidebar-Block zeigt die Tag-Chips aus `tag_refs`, kein CSV-Input mehr sichtbar.
6. **Tag hinzufügen**: Tag-Input „neuer-tag" → + → Toast „Tag hinzugefügt." → Chip erscheint inline + nach Reload + in Library-Card-Strip.
7. **Tag entfernen**: X-Click auf Chip → DELETE → Chip weg inline + nach Reload + in Library-Card-Strip.
8. **Datalist-Autocomplete**: Tag-Input „k" → datalist zeigt existierende Tag-Namen (Highlight-Tags UND Conversion-Tags im selben Namespace).
9. **Cross-Domain-Tag-Reuse**: einen Tag-Namen, der schon als Highlight-Tag existiert (z.B. „produkt" aus R1-B-C-Smoke), zu einer Conversion hinzufügen → Tag-ID identisch (kein Duplikat in `Tag`-Tabelle).
10. **Tag-Manager-Page** `/tags`: jeder Tag zeigt jetzt `highlight_count` UND `conversion_count`. Bei DELETE eines Tags: cascade auf BEIDE Junctions (Highlight-Junction + Conversion-Junction) — verifiziere via DB-Probe nach Delete.
11. **Cross-Doc-Isolation**: Tag „test" an Conversion A muss nicht in Conversion B erscheinen (Library-List).
12. **Dark-Mode**: Tag-Picker im Detail-View + Card-Strip in Library-List lesbar in beiden Themes.

**Edge-Case-Smoke**:

13. Migration-Pre-Status: einen Test-Conversion mit künstlicher CSV erstellen (`UPDATE conversion SET tags = 'eins, zwei, eins, drei' WHERE id = X`), Container neu starten, Migration soll laufen: Tags „eins", „zwei", „drei" (dedupliziert) als Junction-Rows, CSV leer. Sub-Thread kann das via Pytest oder Live-Sequenz prüfen.
14. **Highlight-Tag-Picker-Regression**: R1-B-C Highlight-Tag-Picker muss weiterhin funktionieren (Add, Remove, Manager-Page). Tag-Manager-Page-Delete cascadiert nun auf beide Junctions.
15. **CSV-Spalte unleer machbar?** — wenn das Frontend keine CSV-Updates mehr macht (Sub-Thread hat den PATCH-Handler-tags-Pfad entfernt oder no-opped), dann kann nur SQL die CSV-Spalte wieder füllen. Edge-Case akzeptiert.

Nach Phase 2: STOP — Bericht. Migration-Run-Log, DB-Probe-Output, Smoke-Pfad-Übersicht, Cross-Domain-Reuse-Verifikation.

---

## Phase 3 — Commit + Push + Image-Rebuild

- Plain-prose Commit-Message, mehrere `-m`-Flags.
- Ein Commit. Subject z.B. „R2-A: conversion_tags-Junction + CSV-Migration -- M:N relationship, find-or-create Tag.get_or_create, idempotent migration helper, frontend Tag-Picker statt CSV-Input, Library-Card-Strip aus tag_refs".
- Body soll erwähnen: Schema (neue Junction), Migration-Mechanik (leere-CSV-als-Marker, idempotent), `Tag.get_or_create`-Classmethod als DRY-Anker, Frontend-Umstellung (Card-Strip + Detail-Sidebar-Picker), CSS-Konsolidations-Disposition, Dead-Path-Entscheidung, Test-Count.
- Image-Rebuild via `docker compose up -d --build` damit Production-Image den neuen Stand bekommt.
- Branch direkt auf `main`. Push direkt nach Commit.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für R2-A**:
- Wenn die Migration in Phase 1 fehlerhaft läuft (z.B. ein CSV-Tag-String enthält Sonderzeichen, die `Tag.get_or_create` rejectet, oder commit-Crash mid-migration): **sofort STOP**, Master fragen. Migration-Bugs sind deploy-blocking — die existierende DB könnte in inkonsistentem Zustand landen.
- Wenn der `Tag.get_or_create`-Classmethod-Umzug Reibung mit R1-B-C-Tests erzeugt (z.B. weil dort die Logik anders implementiert war als jetzt im Helper): STOP, Master fragen. Konsolidation darf R1-B-C-Tests nicht brechen.
- Wenn das Frontend-Refactoring (`AUTOSAVE_INPUTS` Cleanup) andere Felder mit-bricht (das `tags`-Feld war Teil eines generischen Autosave-Patterns): **nicht aggressiv refactorn**, Bericht-Item, Master entscheidet.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**L** — zwei Schema-Touches (neue Junction, Tag.get_or_create), Migration-Helper-Erweiterung mit Idempotenz-Mechanik, 2 neue API-Routes, Frontend-Umstellung in zwei Templates + viel JS-Refactor (CSV-Mechanik raus, Tag-Picker rein), `GET /api/tags`-Erweiterung, neuer Test-Modul mit 10+ Tests plus test_tags.py-Anpassungen. Wenn Migration in der Praxis Edge-Cases zeigt (z.B. Tag-String-Format-Bugs) oder das Frontend-Refactoring größer als gedacht wird: eskaliert auf XL, Master entscheidet ob splitten in „R2-A-1 Schema + Migration + API" plus „R2-A-2 Frontend-Umstellung".

---

## Konstitutiv mit-genommen, falls berührt

- Wenn `AUTOSAVE_INPUTS` aus `library_detail.js` beim Cleanup als Pattern erkennbar wird, das auch für andere Felder (z.B. `title`) sinnvoll bleibt: Pattern beibehalten, nur den `tags`-Eintrag rausnehmen.
- Wenn `Conversion.to_dict()`-Erweiterung um `tag_refs` Performance-Implikationen hat (lazy='joined' bei großen Tag-Listen): vor Skalierungs-Sorge kein Sprint-Item, im Bericht erwähnen.
- Wenn die Tag-Manager-Page nach R2-A-Counts-Erweiterung visuell überladen wirkt (zwei Zahlen pro Tag): kleiner Style-Tweak im selben Sprint OK.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „R2-A ☑ done 2026-05-25 → commit `<hash>` (conversion_tags-Junction + CSV-Migration: neue Junction-Table mit M:N-Relationship Conversion.tag_refs, Tag.get_or_create-Classmethod als DRY-Anker für 3 Call-Sites, _migrate_conversion_tags_csv_to_junction-Helper im _run_pending_migrations mit leerer-CSV-Marker-Idempotenz, 2 neue API-Routes für Conversion-Tag-Attach/Detach, Frontend Library-Card-Strip + Detail-Sidebar-Picker auf tag_refs umgestellt, AUTOSAVE_INPUTS-Cleanup, GET /api/tags erweitert um conversion_count). Pytest 116+/116+ grün. Migration-Log: N Conversions migriert. **Master-Aktivität nächste**: R2-B Filtered Views + Reading-Progress oder R2-C Lifecycle-Status (Inbox/Later/Archive)."
- **BACKLOG.md**: R2-A in Erledigt-Liste; R2-A-Bullet aus P1-Inbox raus; **R2-C-Sub-Sprint** „Lifecycle-Status (Inbox/Later/Archive) M" als neues P1-Item anlegen (aus Out-of-scope-Disposition); R2-B bleibt P1.
- **Memory**: wenn die CSV-Migration-Idempotenz-Mechanik (leere-CSV-als-Marker) als wiederverwendbares Pattern reift (z.B. künftig andere Feld-Migrations): `reference_csv_to_junction_migration.md` mit Begründung. Plus: wenn `Tag.get_or_create`-Classmethod-Pattern in weiteren Domain-Modellen wiederverwendbar ist: ggf. Memory. Sub-Thread-Disposition.
- **Architektur-Memo**: `docs/reader_architecture.md` Knoten 4 + Sprint-Schneidungs-Tabelle aktualisieren: R2-A war Tag-only, R2-C ist Lifecycle separat. Master macht das beim R2-A-Closing oder beim nächsten Master-Burst — Sub-Thread kann es im Phase-3-Wrap auch direkt erledigen.

---

## Phase-0-Entscheidungen

_(Phase 0 nicht aktiviert — alle Architektur-Knoten geklärt: Junction-Schema aus Memo Knoten 4, Migration-Mechanik aus Memory `reference_inline_sqlite_migration.md`, Tag-Helper-DRY-Pflicht selbsterklärend, Lifecycle-Auslagerung nach R2-C als Master-Disposition beim Schreiben dieses Prompts.)_
