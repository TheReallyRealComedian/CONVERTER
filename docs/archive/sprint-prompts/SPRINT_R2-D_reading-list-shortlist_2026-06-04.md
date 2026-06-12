# Sprint R2-D — Lese-Liste / Priority-Shortlist (geordnet)

**Datum**: 2026-06-04

**Ziel**: Die fehlende Reader-Dimension nachrüsten — eine **manuell geordnete Prioritäts-Lese-Liste** („Was will ich als Nächstes lesen, in welcher Reihenfolge?"). Orthogonal zu Lifecycle-Ort (R2-C) und Lese-Fortschritt (R2-B). Plus ein billiger **„Weiterlesen"-View** (was gerade angefangen ist), abgeleitet aus dem R2-B-Progress. **User-Befund 2026-06-04**: Inbox/Later/Archive sind Eimer ohne Reihenfolge; Favoriten sind orthogonal (Archiv-Items können Favorit sein) **und** nicht sortierbar → kein echtes „Lese-als-Nächstes".

**Vorbedingung**: HEAD `f35a22c` (R2-C komplett), lokal+remote synchron. Pytest **181/181 grün**. R2-A/B/C (Tags, Progress+Filter, Lifecycle) ☑, NL-Bridge live. Mac-Dev-Stack mit HYG3-Live-Mount (`static/`+`templates/` → Template-Smoke via `docker compose restart markdown-converter`).

**Out-of-scope**:
- **Drag-and-Drop-Reorder** — v1 macht Hoch/Runter-Buttons; Drag ist späterer Polish (eigener Mini-Sprint, wenn's sich zäh anfühlt).
- **Favoriten anfassen/retiren** — bleibt unverändert als orthogonaler Stern; falls er nach R2-D redundant wirkt, separater Cleanup.
- **R4-LEARN** (Lernkarten), **R3 Web-Article-Save** — eigene Cluster.

---

## Workshop-Entscheidungen (Master, 2026-06-04) — nicht neu diskutieren

| # | Frage | Entscheidung |
|---|-------|--------------|
| 1 | Verhältnis zu Favoriten | **Eigene neue Dimension**. Favoriten-Stern bleibt unangetastet. Keine Migration der Favoriten (deine archivierten Favoriten verschmutzen die Liste **nicht**). |
| 2 | Schema | Eine Spalte `Conversion.queue_position` (**Float**, nullable, `NULL` = nicht auf der Liste) via Inline-ALTER. **Float**, damit künftiges Drag-Einsortieren *zwischen* zwei Items ohne Neu-Nummerierung geht. **Kein Backfill** (alle starten NULL = leere Liste). |
| 3 | Reorder-UX | **Hoch/Runter-Buttons** (v1). „Move up" = `queue_position` mit dem direkt darüber liegenden Listen-Item **tauschen** (Float-Swap, serverseitig). |
| 4 | „Weiterlesen"-View | Abgeleitet aus R2-B-Progress (`0 < last_read_percent < 95`), sortiert nach `updated_at` desc. Kein neues Schema. |
| 5 | Archiv ∩ Liste | Der Lese-Liste-View zeigt nur **queued + nicht-archiviert** (Archiv = erledigt, gehört nicht in die To-Read-Liste). **Kein** Auto-Dequeue beim Archivieren (reine View-Filterung — un-archivieren bringt's an alter Position zurück). |

---

## Phase 0 — entfällt

Workshop ist gelaufen, Anker Master-grounded. **Direkt Phase 1.**

---

## Phase 1 — Schema + Queue-API + View-Backend

Pre-Flight: `pytest tests/` grün (181).

### Erwartete Files
```
models.py             # EDIT — queue_position-Spalte + to_dict
app_pkg/__init__.py   # EDIT — 4. idempotenter ALTER-Block (kein Backfill)
app_pkg/library.py    # EDIT — Queue-Endpoint + ?view=queue/reading + pagination_args(...view='')
tests/test_reading_list.py  # NEU — Schema/Queue-API/Views
```

### Schritte
1. **`models.py`**: `queue_position = db.Column(db.Float, nullable=True, index=True)`. `to_dict` += `'queue_position': self.queue_position`.
2. **`app_pkg/__init__.py`** `_run_pending_migrations` — 4. Glied im Conversion-`cols`-Block (neben `last_read_percent`/`lifecycle_status`): `if 'queue_position' not in cols: ALTER TABLE conversion ADD COLUMN queue_position FLOAT` + commit + Log `R2-D: conversion.queue_position added via ALTER TABLE`. **Kein Backfill** (NULL = leere Liste). Idempotent über den Guard.
3. **`app_pkg/library.py`** — neuer Endpoint **`POST /api/conversions/<int:id>/queue`** (Muster wie der progress-PATCH / tag-POST, `get_owned_conversion`-Ownership, Body-dict-Check):
   - `action = data.get('action')`, validieren gegen `{'add','remove','up','down'}` (sonst 400).
   - **add**: wenn `queue_position is None` → setze auf `(max queue_position aller queued Conversions des Users) + 1.0` (sonst `1.0`). Bereits drauf → no-op.
   - **remove**: `queue_position = None`.
   - **up/down**: unter den queued Conversions des Users (sortiert `queue_position asc`) den direkten Nachbarn finden und die beiden `queue_position`-Werte **tauschen** (zwei-Zeilen-Update, eine commit-Boundary). Am Rand (oberstes/unterstes) → no-op.
   - Response: `jsonify(conversion.to_dict())` (oder `{success, queue_position}`); bei up/down ggf. beide betroffenen IDs zurückgeben fürs Frontend-Reorder.
4. **`library()`-Route** — `view = request.args.get('view', '')`:
   - `view == 'queue'`: `query = query.filter(Conversion.queue_position.isnot(None), Conversion.lifecycle_status != 'archive')`, **Sort überschreiben** `order_by(Conversion.queue_position.asc())` (statt des `sort`-Params).
   - `view == 'reading'`: `query = query.filter(Conversion.last_read_percent.isnot(None), Conversion.last_read_percent > 0, Conversion.last_read_percent < 95)`, Sort `order_by(Conversion.updated_at.desc())`.
   - sonst: bestehende Sort-Logik.
   - `view` **nicht** in `has_active_filter` (es ist ein Modus, kein Filter-Chip — der „Filter zurücksetzen"-Empty-State bleibt für type/search/tag/status). Bestehende Filter (type/tag/status/search) dürfen **on top** AND-en.
   - `view` in **`pagination_args`** (`...view=''`) + alle Call-Sites; `current_view=view` ans Template.
5. **Tests** (`tests/test_reading_list.py`): Migration (Spalte da, kein Backfill = alle NULL); Queue-API add (an-die-Liste, ans-Ende; idempotent), remove (NULL), up/down (Swap + Rand-no-op + Ownership-404 + ungültige action 400); `?view=queue` ordnet nach position + schließt Archiv aus; `?view=reading` filtert 0<%<95 (NULL/0/≥95 ausgeschlossen); `to_dict` enthält queue_position. Stil an `tests/test_lifecycle.py`.

### Verify + Commit Phase 1
- `pytest tests/` grün (181 → ~+10).
- Container-Restart-Smoke: Migration-Log einmalig.
- **Commit** (plain-prose): `R2-D-backend: queue_position Spalte + Queue-API (add/remove/up/down) + view=queue/reading`. Push direkt.

**STOP — Bericht an Master.**

---

## Phase 2 — Frontend (View-Switcher + Liste-Toggle + Reorder + Weiterlesen)

Pre-Flight: `pytest tests/` grün.

### Erwartete Files
```
templates/library.html        # EDIT — View-Switcher + Queue-Add/Remove-Control + Reorder-Controls (nur im queue-View) + View-Empty-States
templates/library_detail.html # EDIT — „Auf die Lese-Liste"-Toggle
static/js/library.js          # EDIT — toggleQueue + moveQueue
static/js/library_detail.js   # EDIT — toggleQueue (1-Arg)
static/css/style.css          # EDIT — View-Switcher + Queue-Controls + TOC
```

### Schritte
1. **View-Switcher** (`library.html`, oben, prominent): `Alle · Lese-Liste · Weiterlesen` als Links (`?view=` / `?view=queue` / `?view=reading`), aktiver Zustand markiert (`aria-current` + Accent). Eigenständig von der Filter-/Status-/Tag-Chip-Row (das ist ein Modus, kein Filter).
2. **Queue-Add/Remove-Control** pro Card (nahe dem Favoriten-Stern `library.html:102`): Toggle „auf die Lese-Liste" ↔ „von der Liste" (Icon + title), `onclick="toggleQueue({{ conv.id }}, this)"`. Sichtbar in jedem View.
3. **Reorder-Controls** (nur im **queue-View**): pro Card Hoch/Runter-Buttons (`moveQueue(id,'up'|'down')`) + Positions-Hinweis. v1-Layout: bestehendes Grid ist ok, aber im queue-View **1-spaltig + nummeriert** (klarere Reihenfolge) ist nicer — deine Wahl, halte es simpel. Nach up/down: simpel **Reload des queue-Views** (Liste ist klein, robust) — DOM-Swap als optionaler Polish.
4. **View-Empty-States**: queue leer → „Deine Lese-Liste ist leer. Auf einer Karte ‚Auf die Lese-Liste' tippen." · reading leer → „Nichts angefangen. Sobald du etwas liest, taucht dein aktueller Stapel hier auf." (DE, ≤3 Sätze, keine Emojis).
5. **Detail-View** (`library_detail.html`): „Auf die Lese-Liste"-Toggle (z.B. in der Sidebar bei Status/Details).
6. **JS** (`library.js`): `toggleQueue(id, btn)` → `POST /api/conversions/<id>/queue {action:'add'|'remove'}` über den globalen CSRF-fetch-Wrapper, bei ok Control-State + Toast/still updaten; `moveQueue(id, dir)` → `POST … {action:'up'|'down'}` → bei ok queue-View reloaden. Fehler `showToast`. Kein neuer `_utils`-Helper (Single-Call-Site). `library_detail.js`: `toggleQueue(status)`-Pendant mit fixer `CONVERSION_ID`.
7. **CSS**: View-Switcher (Tab/Pill aktiv-Zustand), Queue-Control + Reorder-Buttons, token-driven, Dark-Mode-konsistent; TOC-Eintrag. `view` an alle `pagination_args`-Call-Sites in `library.html` (analog `current_status` aus R2-C).

### Verify + Commit Phase 2
- `pytest tests/` grün.
- **Live-Smoke** über den HYG3-Mount (`docker compose restart markdown-converter`):
  - „Auf die Lese-Liste" auf 3 Karten → erscheinen im `?view=queue` (Reihenfolge = Add-Reihenfolge); Hoch/Runter ordnet um + persistiert über Reload; „von der Liste" entfernt.
  - Archiv ∩ Liste: ein queued Item auf `archive` setzen → verschwindet aus dem Lese-Liste-View, un-archivieren → zurück an alter Position.
  - `?view=reading` zeigt nur Angefangenes (0<%<95), sortiert nach zuletzt.
  - View-Switcher aktiv-Zustände; leere Liste/Reading → die View-Empty-States.
  - Detail-Toggle funktioniert (separater JS-Pfad).
  - Dark + Light lesbar. **Reale DB nach Smoke sauber zurücklassen** (queue_position der Test-Items wieder NULL).
- **Commit** (plain-prose): `R2-D-frontend: View-Switcher (Alle/Lese-Liste/Weiterlesen) + Queue-Toggle + Hoch/Runter-Reorder`. Push direkt.

**STOP — Bericht an Master.**

---

## Phase 3 — Verify + Wrap-up

1. `pytest tests/` final grün.
2. Cross-Feature-Sanity: queue_position koexistiert mit Lifecycle/Tag/Progress/Favorit auf einer Card; ein Item gleichzeitig in Lese-Liste + mit Tag + Progress.
3. **STATUS.md** + **BACKLOG.md**: R2-D ☑ done mit Hashes; als P1 eintragen/abräumen. **`reader_architecture.md`**: neue Dimension „Lese-Liste/Queue (orthogonal zu Ort + Lese-Zustand)" + Decision-Log (eigene Dimension statt Favoriten-Repurpose, Float-queue_position für Reorder, up/down-Swap v1, Archiv-View-Filter statt Auto-Dequeue).
4. **Memory**: evtl. `reference_*` zum Float-Position-Reorder-Pattern (Swap v1 / fractional für Drag später), falls verallgemeinerbar — deine Einschätzung.
5. **Bullet-Guard** vor BACKLOG-Commit (`grep -nE '(- \*\*.*){2,}' BACKLOG.md`).
6. Push bestätigen.

**STOP — Schluss-Bericht an Master.**

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute. Doc-Wrap-up-Guard (Memory `reference_markdown_bullet_delete_newline.md`).

---

## Größe

**M** — eine Float-Spalte via Inline-ALTER (kein Backfill), ein Queue-Endpoint (4 Actions, Swap-Reorder), zwei View-Modi in der Route, Frontend View-Switcher + Toggle + Reorder + zwei Empty-States, Test-Welle (~10). Spiegelt die R2-B/R2-C-Form (Backend/Frontend-Split).

---

## Konstitutiv mit-genommen, falls berührt
- Nichts Zusätzliches — eng geschnitten. Der „Weiterlesen"-View ist die billige Hälfte (reine Route-Ableitung aus R2-B-Progress, kein Schema).

---

## BACKLOG- und STATUS-Updates nach Abschluss
- ✓ Sprint R2-D durch (2026-06-04), zwei Code-Commits.
- 📋 Drag-and-Drop-Reorder (P3, Polish, wenn Hoch/Runter zäh wird); ggf. Favoriten-Retire-Frage (falls nach R2-D redundant).
- STATUS.md / BACKLOG.md / reader_architecture.md wie Phase 3.
