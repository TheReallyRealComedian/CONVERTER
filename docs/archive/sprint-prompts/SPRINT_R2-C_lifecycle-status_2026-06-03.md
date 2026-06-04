# Sprint R2-C — Lifecycle-Status (Inbox/Later/Archive)

**Datum**: 2026-06-03

**Ziel**: Reader-Triage-Workflow. Jede Conversion bekommt einen **Lifecycle-Ort** (`inbox` / `later` / `archive`) — orthogonal zum Lese-Fortschritt aus R2-B (`last_read_percent` = „gelesen", kein eigener Status). Status-Toggle in Library-Card + Detail-View, Status-Filter-Chips in der List-View (kombinierbar mit dem R2-B-`?tag=`-Filter). Newsletter (echtes Triage-Material) landen im Inbox, alte Konverter-Outputs im Archive.

**Vorbedingung**: HEAD `5aafa0a`, lokal+remote synchron. Pytest **171/171 grün**. R2-B (Tag-Filter-Chips + Reading-Progress) ☑, NL1 (`ai_newsletter`-Typ + Ingestion-Endpoint) ☑, NL-Bridge live. Mac-Dev-Stack mit HYG3-Live-Mount (`static/`+`templates/` → Template-Smoke via `docker compose restart markdown-converter`).

**Out-of-scope**:
- **R4-LEARN** (Spaced-Repetition / Anki-like Lernkarten über Highlights) — eigenes Future-Cluster, siehe BACKLOG.
- **R3 Web-Article-Save**.
- **Shortlist / Star** — bewusst verworfen zugunsten des 3-Ort-Modells.
- Reading-Progress (R2-B, fertig) — „gelesen" bleibt der Progress, **kein** 4. Status.

---

## Workshop-Entscheidungen (Master, 2026-06-03) — nicht neu diskutieren

| # | Frage | Entscheidung |
|---|-------|--------------|
| 1 | Status-Set | **3 Orte**: `inbox` / `later` / `archive`. „Gelesen" = R2-B-Progress (orthogonal). |
| 2 | Schema | **Eine Spalte** `Conversion.lifecycle_status` (String, Default `'inbox'`) via Inline-ALTER-TABLE-Helper. Kein eigenes Table (YAGNI, keine Historie). |
| 3 | Migration (Bestand) | Einmalig beim Spalten-Add: **`ai_newsletter` → `inbox`, alles andere → `archive`** (Conditional auf `conversion_type`). |
| 4 | Neue Conversions | Default `'inbox'` (Spalten-Default greift für Konverter-Creates **und** den Ingest-Endpoint). |
| 5 | List-Default | „Alle zeigen" + Status-Filter-Chips zum Eingrenzen (kein erzwungener Inbox-View). |
| 6 | UI-Labels (deutsch) | **Inbox / Später / Archiv** (intern `inbox`/`later`/`archive`). |

---

## Phase 0 — entfällt

Workshop ist gelaufen (Tabelle oben), Anker Master-grounded. **Direkt Phase 1.**

---

## Phase 1 — Schema + Migration + API + Backend-Filter

Pre-Flight: `pytest tests/` grün (171).

### Erwartete Files
```
models.py             # EDIT — lifecycle_status-Spalte + to_dict
app_pkg/__init__.py   # EDIT — 3. idempotenter ALTER-Block + einmaliger differenzierter Backfill
app_pkg/library.py    # EDIT — LIFECYCLE_STATUSES, ?status-Filter, pagination_args(...status=''), PUT-Handler
tests/test_lifecycle.py  # NEU — Migration/Filter/PUT-Validierung
```

### Schritte
1. **`models.py`** (`Conversion`): `lifecycle_status = db.Column(db.String(20), default='inbox', index=True)` (index — wird oft gefiltert). `to_dict` += `'lifecycle_status': self.lifecycle_status`.
2. **`app_pkg/__init__.py`** `_run_pending_migrations` — 3. Block direkt nach dem `last_read_percent`-Block (~Z.100), **vor** `_migrate_conversion_tags_csv_to_junction`:
   ```python
   cols = {c['name'] for c in inspector.get_columns('conversion')}
   if 'lifecycle_status' not in cols:
       db.session.execute(text("ALTER TABLE conversion ADD COLUMN lifecycle_status VARCHAR(20) DEFAULT 'inbox'"))
       # Einmaliger differenzierter Backfill (läuft nur beim Spalten-Add → idempotent):
       # Newsletter bleiben im Inbox-Triage, alte Tool-Outputs ins Archive.
       db.session.execute(text("UPDATE conversion SET lifecycle_status='archive' WHERE conversion_type != 'ai_newsletter'"))
       db.session.commit()
       app.logger.info("R2-C: conversion.lifecycle_status added + backfilled (ai_newsletter→inbox, rest→archive)")
   ```
   Der `cols`-Block existiert schon für `last_read_percent` — nur das `if 'lifecycle_status' not in cols`-Glied ergänzen (nicht den `get_columns`-Call duplizieren). Idempotent: läuft nur wenn die Spalte fehlt. **Container-Restart-Smoke**: Log-Line nur beim ersten Start.
3. **`app_pkg/library.py`**:
   - Konstante `LIFECYCLE_STATUSES = {'inbox', 'later', 'archive'}` (neben `ALLOWED_CONVERSION_TYPES`).
   - `library()`-Route: `status = request.args.get('status', '').strip()`; wenn `status in LIFECYCLE_STATUSES` → `query = query.filter(Conversion.lifecycle_status == status)`. `status` in `has_active_filter` (Z.101) aufnehmen. `current_status=status` ans Template.
   - **`pagination_args`** (Z.25) um `status=''` erweitern (leeren droppen, analog `tag`). **Alle Call-Sites** in `library.html` (Pagination-Nav + Tag-Chip-Links + Status-Chip-Links) um `current_status` ergänzen.
   - **`api_update_conversion`** (PUT, Z.175): `if 'lifecycle_status' in data:` → gegen `LIFECYCLE_STATUSES` validieren (ungültig → 400 „Ungültiger Lifecycle-Status."), sonst setzen. (Der Toggle nutzt diesen PUT, analog `is_favorite`.)
4. **Tests** (`tests/test_lifecycle.py`): Migration (Spalte da nach `_run_pending_migrations`; differenzierter Backfill: eine `ai_newsletter`-Zeile bleibt `inbox`, eine Nicht-Newsletter-Zeile wird `archive`; zweiter Lauf no-op); PUT setzt `lifecycle_status` + Persistenz; PUT ungültiger Wert → 400; `GET /library?status=later` liefert nur later-Zeilen; `has_active_filter` true bei status; Pagination bewahrt `status`. Stil an `tests/test_conversion_tags.py`.

### Verify + Commit Phase 1
- `pytest tests/` grün (171 → ~+8).
- Container-Restart-Smoke: Migration-Log-Line einmalig; DB-Probe: Newsletter (id=9 + die 11 Backfill) auf `inbox`, alte Conversions auf `archive`.
- **Konstitutiv mit-genommen** (du bist eh in `library.py`s Filter-Bereich): den P3-`library.py:66`-LIKE-Escaping-Trap gleich mit-fixen — den Search-Branch `Conversion.title.ilike(f'%{escaped_search}%')` / `tag_refs.any(Tag.name.ilike(...))` auf `.contains(search, autoescape=True)` umstellen (Memory `reference_sqlalchemy_like_escape.md`), `re.sub`-Handarbeit raus. Kurz im Bericht vermerken.
- **Commit** (plain-prose, mehrere `-m`): `R2-C-backend: lifecycle_status Spalte + Migration + ?status-Filter + PUT`. Push direkt.

**STOP — Bericht an Master. Nicht in Phase 2 bis Sign-off.**

---

## Phase 2 — Frontend (Status-Toggle + Filter-Chips + Badge)

Pre-Flight: `pytest tests/` grün.

### Erwartete Files
```
templates/library.html        # EDIT — Status-Filter-Chip-Row + Per-Card-Status-Control + Badge
templates/library_detail.html # EDIT — Status-Toggle in der Detail-View
static/js/library.js          # EDIT — setStatus(id, status) → PUT
static/js/library_detail.js   # EDIT — Status-Toggle-Handler
static/css/style.css          # EDIT — Status-Badge/Chip-Styles + TOC
```

### Schritte
1. **Status-Filter-Chip-Row** (`library.html`, analog der R2-B-Tag-Chip-Row): drei Chips **Inbox / Später / Archiv** → `?status=inbox|later|archive`, aktiver Chip akzentuiert (`aria-current` + `.c-tag--active` wiederverwenden oder Status-Variante), bewahrt die übrigen Filter via `pagination_args`. „× Status-Filter aufheben" wenn aktiv (Microcopy-Disziplin aus HYG3: scope-genaues Label, nicht „Filter zurücksetzen").
2. **Per-Card-Status-Control** (`library.html`): kompaktes 3-Wege-Control (Segmented oder Dropdown) auf jeder Card → ruft `setStatus(id, 'later')` etc. Plus dezenter Status-**Badge** (Inbox/Später/Archiv). DE-Labels, keine Emojis.
3. **Detail-View** (`library_detail.html`): derselbe Status-Toggle (z.B. in der rechten Sidebar bei Tags/Details).
4. **JS** (`library.js` + ggf. `library_detail.js`): `setStatus(id, status)` → `fetch PUT /api/conversions/<id> {lifecycle_status: status}` → bei ok UI updaten (Badge + aktives Control), `showToast` aus `_utils.js` bei Fehler (Erfolg still oder kurzer Toast — deine Wahl, konsistent mit den Nachbar-Mutationen). Kein neuer `_utils`-Helper für Single-Call-Site.
5. **CSS** (`style.css`): Status-Badge/Chip-Styles, token-driven, Dark-Mode-konsistent; TOC-Eintrag. Drei distinkte, dezente Töne (Inbox/Später/Archiv) — nicht mit den `type-*`-Badges verwechselbar.

### Verify + Commit Phase 2
- `pytest tests/` grün.
- **Live-Smoke** über den HYG3-Mount (`docker compose restart markdown-converter`): Status-Chips filtern (`?status=inbox` zeigt die Newsletter); Toggle bewegt eine Conversion inbox→later→archive, persistiert (Reload), Badge folgt; Default-View zeigt alle; Status-Filter **kombiniert mit `?tag=`** (beide aktiv → URL trägt beide, Pagination bewahrt beide); Dark+Light Badge/Chips lesbar.
- **Commit** (plain-prose): `R2-C-frontend: Status-Toggle in Card/Detail + Status-Filter-Chips + Badge`. Push direkt.

**STOP — Bericht an Master.**

---

## Phase 3 — Verify + Wrap-up

1. `pytest tests/` final grün.
2. Cross-Feature-Sanity: Status + Tag + Progress koexistieren auf einer Card (Badge + Tag-Chips + Progress-Balken), `?status=inbox&tag=ki-agenten` schneidet korrekt.
3. **STATUS.md** + **BACKLOG.md**: R2-C ☑ done mit Commit-Hashes; R2-C aus P1 entfernen; **R2-Cluster damit komplett** (R2-A/B/C). Nächstes P1 = ? (R3 Web-Article-Save oder R4-LEARN — Master entscheidet später). `library.py:66`-P3 entfernen (in Phase 1 mit-gefixt).
4. **`docs/reader_architecture.md`**: R2-Tabelle R2-C → done; Decision-Log-Einträge (3-Ort-Modell, Progress-statt-4.-Status, differenzierter Migration-Default).
5. **Memory**: vermutlich kein neuer Eintrag (etablierte Patterns: Inline-Migration, R2-B-Filter-Reuse). Nur falls Unerwartetes.
6. Push bestätigen.

**STOP — Schluss-Bericht an Master.**

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute. **Doc-Wrap-up-Guard** (Memory `reference_markdown_bullet_delete_newline.md`): nach Bullet-Löschungen in STATUS/BACKLOG vor dem Commit `grep -nE '(- \*\*.*){2,}' BACKLOG.md`.

---

## Größe

**M** — ein Schema-Spalten-Add via Inline-ALTER + einmaliger Backfill-UPDATE, Backend-Filter + PUT-Erweiterung, Frontend Toggle + Filter-Chips + Badge, Test-Welle (~8). Spiegelt die R2-B-Form (Backend/Frontend-Split). Kein Service-Boundary-Touch.

---

## Konstitutiv mit-genommen, falls berührt

- **`library.py:66` LIKE-Escaping-Trap** (P3) — in Phase 1 mit-fixen, da du eh den Filter-Bereich anfasst (`.contains(autoescape=True)`).

---

## BACKLOG- und STATUS-Updates nach Abschluss

- ✓ Sprint R2-C durch (2026-06-03), zwei Code-Commits.
- R2-Cluster (A/B/C) komplett; `library.py:66`-P3 raus (mit-gefixt).
- 📋 evtl. Follow-ups aus Smoke.
- STATUS.md / BACKLOG.md / reader_architecture.md wie Phase 3.
