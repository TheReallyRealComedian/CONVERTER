# Sprint R2-B — Filtered Views + Reading-Progress

**Datum**: 2026-05-29

**Ziel**: Die Library-List wird Reader-tauglich. Zwei Cluster: (1) **Reading-Progress** — Lese-Position pro Conversion als Prozent persistieren, beim Öffnen resumen, in der List-View als Card-Fortschrittsbalken zeigen; schließt den explizit aufgeschobenen READER-MODE-„R2-B-Vorbehalt" (last-read-Position). (2) **Filtered Views** — Tag-Filter-Chip-Row in der List-View, Filter-State in der URL (`?tag=ki`), baut auf dem in R2-A etablierten `Conversion.tag_refs.any(...)`-Junction-Pfad auf.

**Vorbedingung**: HEAD `c9a4516`, lokal+remote synchron. Pytest **131/131 grün** im Container. R1 Reader-Core ☑, R2-A Tag-Junction ☑, READER-MODE ☑, READER-FIX-A ☑. Mac-Dev über `docker-compose.override.yml`. Container-DB hält Smoke-Foundation: doc 2 (4 Highlights, langer Inhalt), doc 4 (Cross-Format + Tags „test"/„lakmus"), doc 7 (Multi-Anker), doc 9 (Tag „produkt") — direkt als Test-Daten wiederverwendbar (kein Neu-Seed nötig).

**Out-of-scope** (Scope-Creep-Schutz):
- **R2-C Lifecycle** (Inbox/Later/Archive) — eigener Folge-Sprint.
- **Tag-Cloud / Tag-Statistik** in der Tag-Manager-Page (der „Optional"-Punkt aus dem BACKLOG) — bewusst aufgeschoben, hält R2-B bei M.
- **localStorage-Filter-Persistenz** — Master-Workshop-Entscheidung: nur URL als Source.
- **Pixel-/Char-genaues Resume** — Master-Workshop-Entscheidung: Prozent-basiert.
- **Multi-Tag-Filter** (mehrere Tags AND/OR gleichzeitig), **Tag-Chip-Overflow/Cloud-UI** bei vielen Tags, **Tag-Rename / Tag-Bulk-Ops / Tag-Color-Coding** — alles spätere Polish-Kandidaten.
- **SQLite DROP der Dead-CSV-Spalte** `Conversion.tags` — bleibt liegen (separater Cleanup-Sprint).

---

## Workshop-Entscheidungen (Master, 2026-05-29) — nicht neu diskutieren

| # | Frage | Entscheidung | Konsequenz |
|---|-------|--------------|------------|
| 1 | Reading-Progress-Schema | **Prozent 0–100** | Eine nullable `Float`-Spalte `Conversion.last_read_percent` via Inline-ALTER-TABLE-Helper. Dient Card-Indikator **und** Resume. Robust gegen Content-Längen-Änderung. Subsumiert gelesen/ungelesen. |
| 2 | Tag-Filter-Persistierung | **Nur URL** `?tag=<name>` | Web-native Source, bookmarkbar, Back-Button, frischer `/library`-Aufruf zeigt Default (kein klebriger Filter). Kein localStorage. |
| 3 | Persist-Trigger-Mechanik | **Throttled fetch + keepalive-Flush** | Throttle (~2 s) während Scroll über den globalen `fetch`-Wrapper (CSRF kommt automatisch). Flush bei `visibilitychange`→hidden via `fetch(..., {keepalive: true})`. **Nicht** `navigator.sendBeacon` (kann CSRF-Header nicht setzen; CSRFProtect ist global aktiv). |
| 4 | Fortschritt = max erreicht | **Furthest-read** | Es wird der **höchste** erreichte Prozent-Wert dieser Session persistiert, nicht der aktuelle Scroll — Zurückscrollen darf den Fortschritt nicht resetten. Seed des Session-Max aus dem gespeicherten Wert. |

---

## Phase 0 — entfällt

Workshop ist gelaufen (Tabelle oben), Mechanik ist durchprescribed. **Direkt mit Phase 1 starten.** Kein Phase-0-Audit.

---

## Phase 1 — Cluster A: Reading-Progress-Persist

Pre-Flight: `pytest tests/` grün (Baseline 131), im Container wie etabliert.

### Erwartete Files

```
models.py                       # EDIT — last_read_percent-Spalte + to_dict
app_pkg/__init__.py             # EDIT — dritter idempotenter ALTER-TABLE-Block in _run_pending_migrations
app_pkg/library.py              # EDIT — neue Route PATCH /api/conversions/<id>/progress
templates/library_detail.html   # EDIT — PageData um lastReadPercent erweitern
static/js/library_detail.js     # EDIT — Persist (throttle + keepalive-Flush) + Resume-on-Open in/an initReadingProgress
templates/library.html          # EDIT — Card-Fortschrittsbalken (conv.last_read_percent)
static/css/style.css            # EDIT — .card-progress{,__fill} Block + TOC-Eintrag
tests/test_conversion_progress.py  # NEU — Endpoint + Migration + to_dict
```

### Schritte

1. **Schema** (`models.py`, `Conversion`-Model ~Z.48): neue Spalte
   `last_read_percent = db.Column(db.Float, nullable=True)` (sinnvoll direkt nach `is_favorite`). In `to_dict()` (Z.58) ergänzen: `'last_read_percent': self.last_read_percent,`.

2. **Migration** (`app_pkg/__init__.py`, `_run_pending_migrations` Z.83): dritter idempotenter Block, **vor** dem `_migrate_conversion_tags_csv_to_junction(app)`-Call (Spalten-Adds gruppiert, dann Daten-Migration). Pattern exakt wie der `highlight.note`-Block:
   ```python
   if 'conversion' in inspector.get_table_names():
       cols = {c['name'] for c in inspector.get_columns('conversion')}
       if 'last_read_percent' not in cols:
           db.session.execute(text('ALTER TABLE conversion ADD COLUMN last_read_percent FLOAT'))
           db.session.commit()
           app.logger.info("R2-B: conversion.last_read_percent column added via ALTER TABLE")
   ```
   Single-Source-of-Truth-Pattern siehe Memory `reference_inline_sqlite_migration.md`. Container-Restart-Smoke: Log-Line **nur beim ersten** Start (idempotent).

3. **API** (`app_pkg/library.py`, neue Route neben den anderen `/api/conversions/<id>/…`): `PATCH /api/conversions/<int:conversion_id>/progress`.
   - Ownership via `get_owned_conversion(conversion_id)` (foreign → 404).
   - Body nicht-dict → 400 „Ungültiger Request-Body. JSON-Objekt erwartet." (gleiche Microcopy wie die Nachbar-Routes).
   - `percent = data.get('percent')`. **Pflicht-Key**: fehlt / `None` / nicht numerisch → 400 „Feld \"percent\" fehlt oder ist ungültig." (atomarer Endpoint, kein verstecktes no-op — wie R1-B-B note-PATCH).
   - **Bool-Falle**: `isinstance(True, int)` ist `True` → `isinstance(percent, bool)` explizit ablehnen (400).
   - Numerisch-aber-out-of-range → **clampen** statt 400: `percent = max(0.0, min(100.0, float(percent)))` (Fire-and-forget-Signal, Rundungsartefakte wie 100.0001 sollen nicht 400en).
   - Persistieren, commit, `return jsonify({'success': True, 'last_read_percent': percent}), 200`.

4. **PageData** (`templates/library_detail.html`, der inline `window.PageData = {…}`-Block): `lastReadPercent: {{ conversion.last_read_percent if conversion.last_read_percent is not none else 'null' }}` (oder via `tojson`-Filter sauber als Zahl/null serialisieren — der `conversion`-Object ist im Template-Scope, **keine Route-Änderung nötig**). Ziel: JS liest den gespeicherten Wert beim Load.

5. **Frontend Persist + Resume** (`static/js/library_detail.js`, an/in `initReadingProgress` Z.1203):
   - **Session-Max**: `let maxReached = <seed aus PageData.lastReadPercent oder 0>;`. In `update()` nach der Fill-Width-Berechnung: `if (percent > maxReached) { maxReached = percent; schedulePersist(maxReached); }`. **Nur** wenn `scrollable >= MIN_SCROLLABLE_PX` (kurze Docs nicht mit 0 clobbern).
   - **Throttle** `schedulePersist(p)`: simpler timestamp-/timer-basierter Throttle (~2 s), inline in `library_detail.js`. **Kein** neuer `_utils.js`-Helper — Single-Call-Site, Memory `feedback_helper_reuse_design_choice.md` (Inline ≠ H4-Verletzung bei einer Call-Site).
   - **Persist-Call** `persistProgress(p, useKeepalive)`: `fetch(\`/api/conversions/${CONVERSION_ID}/progress\`, {method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({percent: p}), keepalive: useKeepalive})`. Geht über den globalen Wrapper aus `base.html` → CSRF-Header automatisch (Sub-Thread bestätigt das im Smoke: kein 400). Fehler still schlucken (Reader bleibt ruhig, kein Toast bei Progress-Save-Fail).
   - **Flush** bei `document.addEventListener('visibilitychange', …)`: wenn `document.hidden` → `persistProgress(maxReached, true)` (keepalive überlebt Tab-Close/Navigation).
   - **Resume-on-Open**: nach Layout-Settle (`requestAnimationFrame` o. kleiner Timeout, damit Code-Blocks/Bilder gerendert sind) den `scroller` auf den gespeicherten Prozent scrollen — **nur** wenn `1 < lastReadPercent < 95` (bei ≥95 als „gelesen" oben öffnen, nicht ans Ende zwingen; ≤1 = Top sowieso). `scroller.scrollTop = (lastReadPercent / 100) * (scroller.scrollHeight - scroller.clientHeight)`.
   - **Resume-Scroll nicht als Fortschritt persistieren**: Flag während des programmatischen Scrolls setzen **oder** Persist-Tracking erst nach dem ersten echten User-Scroll aktivieren. Sonst feuert der Resume-Scroll selbst einen Persist.

6. **Card-Indikator** (`templates/library.html`, Card `c-card flex flex-col` Z.52): dünner Fortschrittsbalken (3 px) am **unteren Card-Rand** (letztes Kind der Card), Fill-Width = `conv.last_read_percent`, **versteckt** wenn `last_read_percent` null/0. Bei ≥95 dezenter „gelesen"-Zustand (z.B. voller Balken in gedämpftem Ton). `title`-Attribut DE, z.B. „37 % gelesen" (gerundet). `conv.last_read_percent` ist im Jinja-Scope (Route übergibt `conversions=pagination.items` als Conversion-Objekte) — keine Route-Änderung nötig.

7. **CSS** (`static/css/style.css`): Block `.card-progress` / `.card-progress__fill` token-driven (`--nm-*`), Dark-Mode-konsistent (gleiche Token-Disziplin wie `.reading-progress__fill` aus READER-MODE). TOC-Eintrag ergänzen.

8. **Tests** (`tests/test_conversion_progress.py`, neu): PATCH valid → 200 + persistiert; clamp >100 → 100; clamp <0 → 0; non-numeric (`"x"`) → 400; bool → 400; missing key → 400; non-dict body → 400; foreign conversion → 404; `to_dict` enthält `last_read_percent`; Migration-Idempotenz (Spalte existiert nach `_run_pending_migrations`, zweiter Lauf no-op). Stil an `tests/test_conversion_tags.py` / `tests/test_highlights.py` anlehnen (App-Context-Fixtures wiederverwenden).

### Quality-Gates Phase 1

- `pytest tests/` grün, neue Tests zählen hoch (Baseline 131 → ~131+10).
- UI-Strings deutsch (CLAUDE.md Code-Stil). Keine `alert()` — `showToast`/`showAlert` aus `_utils.js`.
- Keine neuen `_utils.js`-Helper für Single-Call-Site-Logik (Throttle inline).
- Template-Touch → Live-Smoke Pflicht (Suite rendert keine Templates).

### Verify + Commit Phase 1

1. `pytest tests/` grün.
2. **Live-Smoke** (Browser `localhost:5656`):
   - Langes Doc öffnen (z.B. doc 2), auf ~40 % scrollen, weg-navigieren, neu öffnen → resumed nahe 40 %. Card in `/library` zeigt ~40 %-Balken.
   - Bis ganz unten scrollen → ≥95 % → Card zeigt „gelesen"-Zustand.
   - Zurückscrollen nach oben, dann schließen → Fortschritt **bleibt** beim Max (kein Reset).
   - Kurzes Doc (kein Scroll) → kein Balken, kein Clobber eines evtl. vorhandenen Werts.
   - Network-Tab: Progress-PATCH liefert 200 (CSRF ok), `visibilitychange`-Flush feuert beim Tab-Wechsel.
   - Dark + Light: Card-Balken lesbar.
3. **Container-Restart-Smoke**: Migration-Log-Line nur beim ersten Start, beim zweiten nicht (idempotent).
4. **Commit** (plain-prose, mehrere `-m`, keine Backticks/Unicode-Pfeile/HEREDOCs), z.B.:
   `git commit -m "R2-B-progress: last_read_percent Spalte plus PATCH-progress-Endpoint" -m "Resume-on-Open prozent-basiert, Furthest-read-Persist via throttle plus keepalive-Flush, Card-Fortschrittsbalken in der List-View, Inline-ALTER-TABLE-Migration idempotent" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`
   Push direkt nach Commit (`git push origin main`). Bei Push-Auth-Block (macOS-Keychain): an Master melden — Master-Session kann pushen (Handoff-Gotcha #2).

**STOP — Bericht an Master. Nicht in Phase 2 weiter bis Sign-off.**

---

## Phase 2 — Cluster B: Filtered Views (Tag-Filter)

Pre-Flight: `pytest tests/` grün (jetzt inkl. Phase-1-Tests).

### Erwartete Files

```
app_pkg/library.py        # EDIT — ?tag-Filter-Branch + has_active_filter + pagination_args + available_tags ans Template
templates/library.html    # EDIT — Tag-Filter-Chip-Row + Pagination-Args-Call-Sites
static/css/style.css       # EDIT — Filter-Chip-Styles (falls nicht via .c-tag abgedeckt) + TOC
tests/test_conversion_tags.py  # EDIT — Tag-Filter-Route-Tests (an den R2-A-Search-Patch-Stil anlehnen)
```

### Schritte

1. **Backend** (`app_pkg/library.py`, `library()`-Route Z.49):
   - `tag = request.args.get('tag', '').strip().lower()` (Tags sind lowercase+trim gespeichert — incoming normalisieren).
   - Wenn `tag`: `query = query.filter(Conversion.tag_refs.any(Tag.name == tag))` (exakter Match auf normalisierten Namen; **derselbe Junction-Pfad** wie der R2-A-Search-Branch Z.72-78, nur `==` statt `ilike`).
   - `tag` in `has_active_filter` (Z.89) aufnehmen.
   - **`pagination_args` erweitern** (Z.24): Parameter `tag` ergänzen, in den `args`-Dict aufnehmen (nur wenn gesetzt, analog `favorites`). **Alle 4 Call-Sites** in `library.html` (Pagination-Nav Z.96/101/109) um `current_tag` erweitern.
   - **`available_tags`** ans Template: die Tags des Users, die an ≥1 Conversion hängen, für die Chip-Row. Cleanste Query wählen (z.B. `Tag.query.filter(Tag.conversions.any(Conversion.user_id == current_user.id)).order_by(Tag.name)` — `Tag.conversions` ist der dynamische Backref aus `conversion_tags`). `current_tag=tag` ebenfalls übergeben.

2. **Template** (`templates/library.html`): Tag-Filter-Chip-Row unter der Filter-Bar (nach `</form>`/`</div>` Z.43, vor `#library-alert-container`). Jede Chip = Link auf `?tag=<name>` der die **übrigen** aktiven Filter (`type`/`search`/`favorites`/`sort`/`per_page`) bewahrt (via `pagination_args` mit `page=1`, oder ein analoger Helper). Aktiver Tag-Chip akzentuiert (`aria-current` o. Accent-Klasse). Wenn ein Tag aktiv ist: ein „× Filter zurücksetzen"-Affordance (Link auf `?` ohne `tag`, übrige Filter bewahrt — oder schlicht `url_for('library')`). Optisch `.c-tag`-Pille wiederverwenden (Memory `feedback_helper_reuse_design_choice.md`: bewusste Wiederverwendung, neue Variante nur wenn nötig). Chip-Row nur rendern wenn `available_tags` nicht leer.
   - **Empty-State**: bereits vorhanden (`library.html:115-126`, `has_active_filter`-Zweig mit „Filter zurücksetzen"). Da `tag` jetzt in `has_active_filter` ist, deckt der bestehende Empty-State den 0-Treffer-Fall des Tag-Filters automatisch ab — **kein neuer Empty-State nötig**.

3. **Tests** (`tests/test_conversion_tags.py`, erweitern): `GET /library?tag=ki` surfaced nur ki-Conversions; `?tag=` (leer) = alle; Tag-Filter kombiniert mit `type`/`favorites`; Pagination bewahrt `tag`; `?tag=<unbekannt>` → 0 Treffer + Empty-State-Pfad. **Assertion-Stil 1:1 wie die bestehenden R2-A-Library-Search-Tests** in derselben Datei (Response-Body auf Titel-Präsenz prüfen) — match surrounding code.

### Quality-Gates Phase 2

- `pytest tests/` grün, neue Tests zählen hoch.
- Pagination-Args dürfen `?tag=` über Seitenwechsel **nicht verlieren** (alle 4 Call-Sites angefasst).
- UI-Strings deutsch.

### Verify + Commit Phase 2

1. `pytest tests/` grün.
2. **Live-Smoke** (`localhost:5656`): Tag-Chip klicken → URL `?tag=produkt`, nur passende Docs; mit Typ-Filter kombinieren bleibt konsistent; auf Seite 2 blättern bewahrt `?tag=`; Reset leert; `?tag=gibtsnicht` → Empty-State mit Reset-Button.
   - **Konstitutiv mit-genommen** (R2-A-Dark-Mode-Smoke, P3-Reminder): bei diesem Smoke gleich die `.c-tag`-Card-Strip-Chips **und** die neuen Filter-Chips in **Dark + Light** auf Lesbarkeit prüfen — schließt den offenen R2-A-Dark-Mode-Reminder billig mit.
3. **Commit** (plain-prose, mehrere `-m`), z.B.:
   `git commit -m "R2-B-filter: Tag-Filter-Chip-Row in der Library-List" -m "URL-persistierter ?tag-Filter via Conversion.tag_refs.any-Junction, pagination_args um tag erweitert, available_tags ans Template, bestehender Empty-State deckt 0-Treffer ab" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`
   Push direkt nach Commit.

**STOP — Bericht an Master. Nicht in Phase 3 weiter bis Sign-off.**

---

## Phase 3 — Abschluss

1. `pytest tests/` final grün (volle Suite).
2. Kurzer Cross-Cluster-Sanity-Smoke: Doc mit Fortschritt **und** Tag öffnen/filtern — beide Cluster koexistieren ohne Konflikt (Card zeigt Balken **und** ist im Tag-Filter sichtbar).
3. **STATUS.md** + **BACKLOG.md** nachziehen:
   - R2-B als ☑ done 2026-05-29 mit Commit-Hashes in STATUS „Aktueller Sprint" + BACKLOG „Erledigt".
   - BACKLOG P1: R2-B-Item entfernen, R2-C als nächstes P1 stehen lassen.
   - P3-Reminder: R2-A-Dark-Mode-Smoke entfernen (in Phase 2 mit-erledigt); READER-MODE-Reading-Progress-Persistierung-Vorbehalt entfernen (jetzt umgesetzt).
   - Neue Follow-ups eintragen falls Smoke welche aufwirft (P-Stufe + Größe).
   - `docs/reader_architecture.md` Decision-Log + R2-Sprint-Schneidungs-Tabelle: R2-B auf done, Workshop-Entscheidungen (Prozent-Schema, URL-Filter, furthest-read) als Einträge ergänzen.
4. **Memory** prüfen: vermutlich kein neuer Eintrag nötig (Mechaniken sind bereits in `reference_inline_sqlite_migration.md` + `feedback_helper_reuse_design_choice.md` abgedeckt). Falls Phase 1/2 eine **verallgemeinerbare** Lehre aufwirft (z.B. keepalive-Flush-Pattern als Reader-weit wiederverwendbar) → kurzer `reference_*`-Eintrag + MEMORY.md-Pointer. Sonst weglassen.
5. Push bestätigen (beide Commits auf `origin/main`).

**STOP — Schluss-Bericht an Master.**

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute. Bei „mach jetzt einfach"/Frust-Signal: einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — ein Schema-Spalten-Add via Inline-ALTER (kein Table-Rebuild), zwei Frontend-Cluster (Progress: Persist/Resume/Card; Filter: Chip-Row/Backend), Test-Welle ~12–16 Tests, zwei Commits. Schema-Touch begrenzt (eine nullable Spalte), kein Service-Boundary-Touch.

---

## Konstitutiv mit-genommen, falls berührt

- **R2-A-Dark-Mode-Smoke** (`.conversion-tag-*` / `.c-tag`): in Phase 2 beim Filter-Chip-Dark/Light-Smoke mit-erledigt — der offene P3-Reminder fällt damit.
- **READER-MODE-Reading-Progress-Persistierung-Vorbehalt**: durch Cluster A faktisch umgesetzt — P3-Reminder entfernen.

Alles andere aus dem BACKLOG bleibt liegen (insb. R2-C, MAC1-FOLLOWUPs, die übrigen P3-Smokes).

---

## BACKLOG- und STATUS-Updates nach Abschluss

- ✓ Sprint R2-B durch (2026-05-29), zwei Commit-Hashes.
- 📋 evtl. Follow-ups aus Smoke (z.B. Multi-Tag-Filter, Tag-Chip-Overflow bei vielen Tags, Resume-UX-Feintuning der 95 %-Schwelle).
- STATUS.md: aktueller Stand auf R2-B-Output.
- BACKLOG.md: R2-B raus, R2-A-Dark-Mode + Reading-Progress-Vorbehalt-Reminder raus, R2-C bleibt nächstes P1.
- `docs/reader_architecture.md`: R2-Tabelle + Decision-Log fortschreiben.
