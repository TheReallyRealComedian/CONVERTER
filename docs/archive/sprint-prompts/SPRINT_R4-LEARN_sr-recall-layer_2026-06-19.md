# Sprint R4-LEARN — SR-/Recall-Layer (Karten · FSRS · Review-UI · Agent-Schreib-Haken)

> **Executor-Doc (L, 5 Phasen).** Phasen strikt nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **277**). Du committest jede Phase selbst (eigener Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER` (Mac-lokal = Source-of-Truth; Mintbox-Mount nur Runtime). Visuelle Phasen (4) → **Live-Smoke Pflicht**. Zeilennummern driften → über Symbolnamen gehen.

## Kontext / Warum

CONVERTER bekommt einen **Spaced-Repetition-/Recall-Layer** über die bestehenden **Highlights**: **Karten** (vom externen Agent erzeugt, CONVERTER speichert/plant/zeigt), **FSRS-Scheduling**, eine **Review-UI**, plus die **Schreib-Endpoints** und einen **globalen Highlights-Reader** für den Agent. Produkt-Brief (v3, Kontext): die zugehörige Diskussion ist eingearbeitet; dieser Sprint ist der **Code-Schnitt** davon. **Karten-Generierung passiert NICHT in CONVERTER** (macht der Agent) — CONVERTER liefert die Schreib-API + Speicher + Scheduling + UI.

## Stack-Realität (verbindlich, schon verifiziert)

- **`Conversion`** = Dokument (Markdown-Volltext). **`Highlight`** ([models.py](../../models.py)) hängt per FK an `Conversion`, hat `exact`/`prefix`/`suffix`, `note`, `tags` (`highlight_tags`-Junction), **`created_at` (existiert + `index=True`)** und cascadet weg (`cascade='all, delete-orphan'`) wenn die `Conversion` gelöscht wird.
- **Kein `PRAGMA foreign_keys=ON`** irgendwo → DB-Level-FK-Actions sind **inert**; die echten Cascades laufen **ORM-seitig**. ⇒ siehe Phase-1-Muss-Fix.
- **Token-Auth-Ingest existiert** ([app_pkg/ingest.py](../../app_pkg/ingest.py), Memory `reference_token_auth_ingest_endpoint`): Bearer gegen Env-Token via `hmac.compare_digest`, **CSRF-exempt nur diese View** (`app.extensions['csrf'].exempt`), fail-closed (503 ohne Token, 401 falsch), Token nie geloggt. **Schreib-Endpoints nutzen dieses Muster.**
- **Routing**: `register(app)`-Pattern, **kein Blueprint**, flache Endpoint-Namen. Neues Modul `app_pkg/cards.py` (in `app.py` einhängen wie die anderen).
- **Keine Alembic-Migrationen**: neue Tabellen entstehen via `db.create_all()` (in `create_app`) → für `card`/`review`/`card_tags` **null Migrations-Aufwand**. Nur Spalten-Adds an *bestehenden* Tabellen laufen über `_run_pending_migrations` (Memory `reference_inline_sqlite_migration`) — hier nicht nötig (Highlight hat alles).
- **Tags**: `Tag.get_or_create(user_id, name)` ist der DRY-Anker (normalisiert via `Tag.normalize_name`). Karten-Tags laufen darüber.
- **`converter-mcp`** = separates Projekt, anderer Owner — **out of scope**. CONVERTER exponiert nur die Endpoints; das MCP-Tool-Wrapping macht der Koordinator.
- UI: Flask/Jinja/Vanilla-JS, kein neues Framework.

## Gesperrte Entscheidungen (Master, nicht neu diskutieren)

- **Reads `@login_required`** (Session, wie die bestehende GET-API, die der MCP schon konsumiert). **Card-Writes Token-Auth** (Ingest-Muster, **eigenes** Env-Token `CARD_TOKEN`, getrennt von `INGEST_TOKEN` — unabhängige Rotation). **Rate-Endpoint `@login_required`** (das ist der *User* in der Review-UI, kein Agent).
- **Karte ist self-contained**: `front`/`back`/`cloze_text` stehen auf der Karte; Review liest **nie** das Highlight live. `highlight_id` ist Best-Effort-Provenienz.
- **`review` als eigene 1:1-Tabelle** (sauber für History) — nicht auf die Karte mergen.
- **`POST /card` legt die `review`-Zeile gleich mit an** (Initial-Zustand „new", `due = jetzt`).
- **Knowledge-Map ist NICHT in diesem Build** (eigene spätere Phase).

---

## Phase 1 — Schema + Highlight-`before_delete` + globaler `since`-Reader (Backend-Fundament)

Dateien: `models.py`, neues `app_pkg/cards.py` (+ in `app.py` registrieren), `tests/`.

1. **Models** (via `db.create_all()`): `Card`, `Review`, `card_tags`-Junction.
   - **`Card`**: `id`; `highlight_id` FK→`highlight.id` **nullable** (Provenienz); `source_snapshot` (Text, nullable) + `source_doc_title` (Text, nullable); `type` (`atomic`|`generative`); `front`/`back`/`cloze_text`/`prompt` (alle Text nullable); `note` (Text nullable); `state` (`ok`|`wackelt`, default `ok`); `created_by` (default `agent`); `created_at`/`updated_at`. `to_dict()`. Relationship `tags` über `card_tags` (sekundär, wie `Highlight.tags`).
   - **`Review`**: `id`; `card_id` FK→`card.id` (CASCADE via ORM-relationship: `Card.review = relationship(..., uselist=False, cascade='all, delete-orphan')`); FSRS-Felder `due` (DateTime, index), `stability` (Float, null), `difficulty` (Float, null), `last_reviewed` (DateTime, null), `reps` (Int, default 0), `lapses` (Int, default 0), `rating_history` (Text/JSON, null).
   - **`card_tags`**: `(card_id, tag_id)` analog `highlight_tags`.
2. **⚠ Muss-Fix — die Lösch-Mechanik ORM-seitig** (DB-Level `ON DELETE SET NULL` feuert hier NICHT, s. Stack-Realität): ein SQLAlchemy-**`before_delete`-Event auf `Highlight`**, der alle Karten mit `highlight_id == <gelöschtes Highlight>` auf `highlight_id = None` setzt (Provenienz-Link bricht, **Karte + Review überleben**). Feuert auch wenn das Highlight via Conversion-`delete-orphan` mit-gelöscht wird (Event greift pro Highlight-Delete). **Nicht** das FK-Pragma global anschalten.
3. **`GET /api/highlights/recent`** (`@login_required`, owner-scoped, neues Modul oder in `cards.py`): globaler Reader über **alle** Docs des Users, `?since=<ISO>`-Filter auf `Highlight.created_at` (existiert+indexed), `?limit=` (Default 100, Cap 500). Response je Highlight: `id`, `exact`, `note`, `tags` (`[{id,name}]`), `created_at`, plus Eltern-`{conversion_id, title}`. Sort `created_at desc`. (Die heutige Highlight-API ist strikt pro-Doc — das hier ist der fehlende globale.)
4. **Tests**: Tabellen entstehen; **`before_delete` nullt Karten** (Highlight löschen → `card.highlight_id` NULL, Karte+Review da; Conversion löschen → Highlights weg, Karten überleben mit NULL); `recent`-Reader (global, `since`-Filter, owner-Scoping, Cap, Sort). `pytest` grün ≥ Baseline.

**Stop + Bericht** (Schema, der Event-Mechanismus + sein Test rot→grün, Reader-Demo).

## Phase 2 — Card-Write-API (Token) + Card/Review-Reads (Session)

Dateien: `app_pkg/cards.py`, `tests/`, `.env.example` (`CARD_TOKEN=`).

1. **`POST /api/cards`** — **Token-Auth** (Ingest-Muster aus `ingest.py`: Bearer gegen `CARD_TOKEN`, `hmac.compare_digest`, CSRF-exempt nur diese View, fail-closed, Token nie geloggt; Ziel-User wie Ingest via `INGEST_USER`/`first()`). Body: `highlight_id` (optional, validiere Ownership wenn gesetzt), `type`, `front`/`back`/`cloze_text`/`prompt`, `tags[]`, `note`, `source_snapshot`, `source_doc_title`. **Pro-Typ-Validierung** (400 bei Verstoß): `atomic` braucht (`front` UND `back`) ODER `cloze_text`; `generative` braucht `prompt`. Tags via `Tag.get_or_create` → `card_tags`. **Legt die `Review`-Zeile gleich mit an**: `due = jetzt`, `reps=0`, `lapses=0`, Rest NULL (FSRS-„new"). 201 + `card.to_dict()`.
2. **`PATCH /api/cards/<id>`** — Token-Auth. Verfeinert Felder, kann `state` zurücksetzen (`wackelt`→`ok` / setzen). Tags ersetzbar. `updated_at` bump.
3. **Reads `@login_required`** (owner-scoped, für den MCP/Agent konsistent zur GET-API): `GET /api/cards?state=&highlight_id=&limit=&offset=` (schlankes Summary), `GET /api/cards/<id>` (voll), `GET /api/review-state` (fällige Karten / Zähler — was ist `due <= jetzt`).
4. **Tests**: Token-Auth (503 ohne Token, 401 falsch, 201 echt; CSRF-exempt unter erzwungenem CSRF wie der Ingest-Test); Pro-Typ-Validierung (atomic/generative gültig/ungültig→400); **Review-Zeile wird mit angelegt** (`due` gesetzt); Tag-Normalisierung über `card_tags`; Ownership des `highlight_id`; Reads filtern/scopen. `pytest` grün.

**Stop + Bericht** (Auth-Selbstreview wie bei NL1: constant-time, fail-closed, nur diese View exempt, Token nie geloggt).

## Phase 3 — FSRS-Scheduling-Engine (swappable) + Rate-Endpoint

Dateien: `services/scheduler/` (neu, kleines Paket), `requirements.txt`/`Dockerfile` (py-fsrs), `app_pkg/cards.py`, `tests/`.

1. **`Scheduler`-Schnittstelle** (klein, austauschbar): `new_card_state() -> dict` (Initial) und `apply_rating(review_state: dict, rating: str) -> dict` (neuer Zustand: `due`, `stability`, `difficulty`, `last_reviewed`, `reps`, `lapses`). Ratings `again|hard|good|easy`.
2. **FSRS-Impl** via **`py-fsrs`** (Dependency adden) + **SM-2-Fallback-Impl** hinter derselben Schnittstelle. Welche aktiv ist → Config (Default FSRS); Ziel-Behaltensrate konfigurierbar (Default ~0.9). **Kein** Auto-Grading.
3. **`POST /api/cards/<id>/review`** — **`@login_required`** (der User bewertet in der UI), Body `{rating}`. Lädt die `Review`-Zeile, ruft `scheduler.apply_rating`, schreibt zurück (`due`/`stability`/`difficulty`/`reps`/`lapses`/`last_reviewed`, optional `rating_history` anhängen). **Generativ + schwache Bewertung** (`again`/`hard`) darf optional `card.state = wackelt` setzen.
4. **Tests**: Scheduler-Math über py-fsrs (new→erste Bewertung verschiebt `due` vorwärts; `again` = lapse, `reps`/`lapses` ticken; `good`/`easy` längeres Intervall); Schnittstellen-Swap (SM-2-Fallback liefert plausible `due`); Rate-Endpoint aktualisiert die Review-Zeile + `wackelt`-Pfad. `pytest` grün.

**Stop + Bericht.**

## Phase 4 — Review-UI (Jinja/Vanilla, Live-Smoke)

Dateien: neue Route + `templates/review.html` + `static/js/review.js` + CSS, Nav-Eintrag.

1. **Review-Seite** (`@login_required`): fällige Karten (`due <= jetzt`) der Reihe nach.
   - *Atomar:* Vorderseite (`front` oder Cloze-Lücke) → „aufdecken" (`back`/Lösung) → Rating-Buttons (again/hard/good/easy) → `POST /review` → nächste.
   - *Generativ:* `prompt` → User erklärt (laut/Papier) → „Musterantwort aufdecken" (`back` als Stichpunkte) → Selbst-Bewertung → `POST /review`.
   - **„Vertiefen"-Knopf** je Karte: setzt `state = wackelt` (+ optional Notiz via `PATCH`/eigener Pfad) — Einstieg in den Agent-Dialog-Recall.
   - Inline-Annotation (Notiz an der Karte).
   - DS-konform (Neomorphism: das gelandete `.place-control`/Karten-Muster als Vorlage, Elevation-Budget, Token-driven — kein Hardcode; Memory `reference_design_system_realignment_is_budget_audit`).
2. **Live-Smoke** (lokale Docker-Instanz, echte Klicks, dark+light, 0 Console-Errors): atomare Karte bewerten → `due` rückt vor, nächste Karte; generative Karte (erklären→aufdecken→bewerten); „Vertiefen" setzt `wackelt`; leere-Queue-Empty-State. (Karten zum Smoken via `POST /api/cards` mit `CARD_TOKEN` seeden.)

**Stop + Bericht** (Screenshot-Beschreibung atomar + generativ).

## Phase 5 — Wrap-up

1. `STATUS.md` + `BACKLOG.md`: R4-LEARN ☑ done mit Phasen-Hashes. Festhalten: SR-Layer über den Highlights, Agent schreibt (Token), User reviewt (Session/FSRS).
2. `docs/reader_architecture.md`: neues Kapitel — der SR-/Recall-Layer (Card/Review/card_tags, self-contained Karte + `before_delete`-Provenienz-Null, FSRS hinter Swappable-Schnittstelle, Auth-Split Token-Write/Session-Read+Rate). Decision-Log + Datum.
3. **Memory**: m.E. einen `reference_*`-Eintrag wert — „SQLite ohne FK-Pragma: deklariertes `ON DELETE SET NULL`/`CASCADE` ist inert, Lösch-Mechanik muss ORM-seitig (`before_delete`-Event); Cascades sind hier durchweg ORM-Level" (konkreter, wiederkehrender Trap). Deine Einschätzung.
4. **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md STATUS.md`), finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Olis offene Schritte: **Mintbox-Deploy** (`git pull` + `up -d --build` — `py-fsrs` ist neu → Image baut neu; `CARD_TOKEN` in `.env` setzen; **keine Migration** — neue Tabellen via `db.create_all`), und die **converter-mcp-Tool-Erweiterung** (separates Projekt/Koordinator: `POST /card` + die neuen Reader wrappen).

## Out of scope
- Karten-**Generierung** in CONVERTER (macht der Agent).
- **Wissenslandkarte** / Notion-Projektion (spätere Phase).
- Altbestands-Migration (Erklärbär-Docs, Anki) · Mobile · Multi-User.
- converter-mcp-Tool-Code (anderes Repo).
- FK-Pragma global anschalten (out — ORM-Event ist der Weg).
