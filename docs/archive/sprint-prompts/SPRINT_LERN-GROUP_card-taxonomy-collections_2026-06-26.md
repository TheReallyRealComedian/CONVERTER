# Sprint LERN-GROUP — Karten gruppieren: Taxonomie-Baum + Sammlungen + Review-Filter (L, 4 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **443**). Du committest jede Phase selbst (Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. Working-Practice in `CLAUDE.md`. **Schema-Touch** (Spalte + 2 Tabellen) → Backend pytest-getestet; **Frontend-Phasen** brauchen **Live-Smoke** (dark+light, 0 Console-Errors).

## Ziel & Entscheidungen (gesetzt, Oli 2026-06-26)

Lernstoff über **zwei orthogonale Achsen** gruppieren, beide als **Filter auf die Wiederhol-Queue** — „heute nur Transformer-Modelle" (Taxonomie-Ast) bzw. „die nächsten Tage nur Kurs X" (Sammlung). **Gilt nur für Karten** (R4-LEARN), nicht die ganze Conversion-Kette. **Keine Notion-Sync** (eigenständig in CONVERTER).

- **Achse A — Taxonomie (Baum)**: hierarchische **Tags** via **`parent_id`** auf dem bestehenden `Tag`-Modell. Reuse: Karten hängen schon an Tags (`card_tags`, vom Agent gesetzt) → du ordnest die Tags **einmal** in einen Wald, **keine Karte muss neu getaggt werden**. Filter = „fällige Karten, deren Tag im **Teilbaum** von Knoten X liegt".
- **Achse B — Sammlungen (kuratiert, flach)**: neue **`Collection`**-Entität + `card_collections`-M2M. Eine Sammlung = ein benanntes Karten-Bündel für einen Zweck (Horizont/Kurs/Themenpaket — der User benennt jede Instanz selbst; **eine** Entität, kein Typ-Unterschied v1). Cross-cutting, Karte in beliebig vielen.
- **Payoff**: `/api/review-state?tag=<id>` (Teilbaum) und/oder `?collection=<id>` + ein **Selektor** in der „Lernen"-UI; Default „alles fällig" bleibt.
- **converter-mcp unberührt** — der Agent setzt weiter nur Tags; Hierarchie + Sammlungen sind User-Seite (UI). Kein neuer Agent-Token/Tool.

## Verifizierte Code-Fakten (Master-gegroundet)

- **Schema-Andockpunkt**: `db.create_all()` + `_run_pending_migrations(app)` in [app_pkg/__init__.py:77](app_pkg/__init__.py) (`inspect(db.engine).get_columns(...)`-Check + `ALTER TABLE … ADD COLUMN`, idempotent — Memory `reference_inline_sqlite_migration`). Neue **Tabellen** kommen via `db.create_all` (Modell definieren reicht); die **Spalte** `tag.parent_id` braucht einen ALTER-Block.
- **`Tag`** ([models.py:131](models.py)): flach (id/user_id/name/created_at, `UniqueConstraint(user_id,name)`), **geteilt** über `card_tags`/`highlight_tags`/`conversion_tags`. `Tag.get_or_create`/`normalize_name` = der Vokabular-Gate.
- **`Card`** hat `tags` via `card_tags` (M2M, [models.py:238](models.py)).
- **Review-Queue**: `api_review_state` ([app_pkg/cards.py:414](app_pkg/cards.py)) = `Card.query.filter_by(user_id).join(Card.review).filter(Review.due<=now).order_by(Review.due.asc())` — **kein Filter**. Hier kommt der Scope-Filter rein.
- **`/tags`**-Seite + `/api/tags` (GET, listet Tags + highlight/conversion-Counts) in [app_pkg/tags.py](app_pkg/tags.py) — Heimat fürs Hierarchie-UI + `PATCH /api/tags/<id>`.
- **SQLite ohne FK-Pragma** (Memory `reference_sqlite_no_fk_pragma_orm_delete`): deklariertes `ON DELETE` ist inert → Lösch-Mechanik ORM-seitig (Tag-Delete: Kinder reparenten; Card/Collection-Delete: `card_collections` via `relationship(secondary=…)`-Cascade).
- **„Lernen"-UI**: [static/js/review.js](static/js/review.js) walkt `data.due_cards` aus `/api/review-state`; [templates/review.html](templates/review.html) ist die Shell.

## Phase 1 — Schema + Taxonomie-Backend (Achse A)

1. **`Tag.parent_id`** ([models.py](models.py)): `db.Column(db.Integer, db.ForeignKey('tag.id'), nullable=True, index=True)` (null = Wurzel) + Self-Relationship (`children`/`parent` via `remote_side=[id]`). **Inline-Migration** in `_run_pending_migrations`: `ALTER TABLE tag ADD COLUMN parent_id INTEGER` (idempotent, `get_columns('tag')`-Check).
2. **Subtree-Helper** `_tag_subtree_ids(root_id, user_id) -> set[int]`: die User-Tags (id, parent_id) laden, parent→children-Map bauen, **BFS** ab `root_id`, `{root + Nachfahren}` zurück. (Tag-Zahl klein → load-all + BFS reicht; keine rekursive SQL nötig.)
3. **`PATCH /api/tags/<int:tag_id>`** (Session, owner-scoped, in tags.py): Body `{parent_id: int|null}`. Validieren: `parent_id` ist ein **eigenes** Tag (oder null); **kein Zyklus** (`parent_id` darf nicht das Tag selbst oder im Teilbaum des Tags sein → `parent_id not in _tag_subtree_ids(tag_id, user)`, sonst 400). Setzen, commit. Response = das Tag.
4. **`/api/tags` GET erweitern**: `parent_id` + ein **`card_count`** pro Tag (Karten je Tag, analog den bestehenden highlight/conversion-Counts) in die Response, damit das UI Baum + Zahlen zeigt.
5. **Tag-Delete** (`DELETE /api/tags/<id>`, existiert): vor/bei Delete die **Kinder reparenten** (`parent_id = NULL`, an die Wurzel), kein Orphan-mit-totem-Parent. ORM-Event oder explizites UPDATE.
6. **`/api/review-state?tag=<id>`-Filter**: `tag` optional; wenn gesetzt → owner-validieren (fremd/unbekannt → 400/404), `subtree=_tag_subtree_ids(tag,user)`, `.filter(Card.tags.any(Tag.id.in_(subtree)))`. `total_count` reflektiert dann den **Scope** (Karten im Teilbaum, nicht nur fällige); ohne Filter bleibt's wie bisher (alle User-Karten).
7. **Tests**: parent setzen/lösen · Zyklus → 400 · fremdes Parent → 400 · Subtree-BFS (mehrstufig) · Tag-Delete reparentet Kinder · `?tag=`-Filter liefert nur Teilbaum-Karten + korrekte Counts · fremder `tag` → 400/404. `pytest` grün ≥ 443.

**Stop + Bericht.**

## Phase 2 — Sammlungen-Backend (Achse B)

1. **Modelle** ([models.py](models.py)): `Collection` (id, `user_id` FK indexed non-null, `name` String(120), `created_at`, optional `description` Text; `UniqueConstraint(user_id, name)`); `card_collections` = `db.Table(card_id FK card ondelete CASCADE PK, collection_id FK collection ondelete CASCADE PK)`; `Card.collections = relationship('Collection', secondary=card_collections, backref='cards')`. Via `db.create_all` (kein ALTER). Lösch-Mechanik ORM (Card-Delete + Collection-Delete nehmen `card_collections` via Relationship mit).
2. **Neues Modul `app_pkg/collections.py`** mit `register(app)` (in [app.py](app.py) registrieren), alle Session/owner-scoped:
   - `GET /api/collections` → Liste (id, name, `card_count`, created_at).
   - `POST /api/collections` → anlegen (`name` Pflicht-non-blank, trim; Duplikat pro User → 409).
   - `PATCH /api/collections/<id>` → umbenennen.
   - `DELETE /api/collections/<id>` → löschen (card_collections cascade ORM).
   - `POST /api/collections/<id>/cards` `{card_id}` → Karte hinzufügen (beide owner-scoped, idempotent).
   - `DELETE /api/collections/<id>/cards/<card_id>` → entfernen.
3. **`/api/review-state?collection=<id>`-Filter**: optional; owner-validieren (fremd → 400/404), `.filter(Card.collections.any(Collection.id == cid))`. **Mit `?tag=` kombinierbar** → AND (Karten in Sammlung **und** unter Topic). `total_count` reflektiert den Scope.
4. **Tests**: Collection CRUD (create/dup-409/rename/delete) · Karte add/remove (idempotent, owner-404) · `?collection=`-Filter · `?tag=`+`?collection=`-AND · fremde Collection → 400/404 · Card-Delete entfernt die card_collections-Zeilen. `pytest` grün.

**Stop + Bericht.**

## Phase 3 — Review-Payoff-UI („Lernen"-Selektor + Karte→Sammlung)

Der Nutzen, früh sichtbar. Dateien: [templates/review.html](templates/review.html) + [static/js/review.js](static/js/review.js) (+ CSS, DS-konform token-driven).

1. **Scope-Selektor** oben in „Lernen": **„Alles fällig"** (Default) / **Taxonomie-Ast** (Tag-Auswahl, der Wald aus `/api/tags` mit Einrückung) / **Sammlung** (Dropdown aus `/api/collections`). Auswahl → re-fetch `/api/review-state?tag=`/`?collection=` → die gefilterte Queue ablaufen; Counter „X von Y" zeigt den Scope.
2. **Karte→Sammlung**-Aktion im Karten-Footer (neben Vertiefen/Notiz/Löschen): „→ Sammlung" → bestehende wählen **oder** neu anlegen → `POST /api/collections/<id>/cards`. So kuratierst du Sammlungen **beim Wiederholen**. Toast bei Erfolg.
3. **Live-Smoke** (lokale Instanz, MacChrome **dark+light**, **0 Console-Errors**): Selektor „Alles fällig"→Taxonomie-Ast→Sammlung schaltet die Queue um (nur passende Karten, Counter stimmt); Karte einer Sammlung hinzufügen → taucht danach im Sammlungs-Scope auf; Default-Verhalten unverändert. `node --check`.

**Stop + Bericht.**

## Phase 4 — Verwaltungs-UI + Wrap

**Lean halten** (funktional, nicht vergoldet — Liste + Dropdown statt Drag-Tree).

1. **Tag-Hierarchie** auf `/tags` ([templates/tags.html](templates/tags.html) + JS): die Tags als **eingerückten Baum** (nach `parent_id`) zeigen, pro Tag ein **„Eltern"-Dropdown** (andere Tags + „— (Wurzel)") → `PATCH /api/tags/<id>`. Zyklus-Fehler vom Server als Toast. `card_count` pro Tag anzeigen.
2. **Sammlungs-Verwaltung**: eine schlichte Fläche (Abschnitt auf `/tags` **oder** kleine `/collections`-Seite) — Sammlungen anlegen/umbenennen/löschen + `card_count`. (Karten-Hinzufügen passiert im Review, Phase 3.)
3. **Live-Smoke** (dark+light, 0 Console-Errors): Tag einem Eltern zuordnen → erscheint eingerückt + wirkt im Review-Taxonomie-Filter; Zyklus-Versuch → Fehler-Toast, kein Schaden; Sammlung anlegen/umbenennen/löschen.
4. **Wrap**: STATUS/BACKLOG (LERN-GROUP ☑ done, „Aktiv offen" → Web-Article-Save P2). **Memory** (`reference_*`): das Zwei-Achsen-Muster — hierarchische Tags via `parent_id` (Reuse des geteilten Vokabulars, Subtree-BFS, Zyklus-Guard, SQLite-no-FK-Reparent-on-Delete) + flache `Collection`-Entität + der Review-Scope-Filter (`?tag=` Teilbaum / `?collection=` / AND). **Bullet-Guard**, MEMORY.md-Pointer, finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. **Olis Deploy-Schritt**: Mintbox `git pull` + `docker compose up -d --build` — **Schema-Touch, aber Auto-Migration** (`parent_id` via `_run_pending_migrations`, neue Tabellen via `db.create_all` beim Boot; **keine** manuelle Migration). Browser-Hard-Reload. **Backup-Disziplin** vor dem ersten Boot (DB im `app_data`-Volume).

## Bewusst NICHT (Scope-Grenze)

- **Nur Karten** — keine Gruppierung von Conversions/Highlights (die teilen das Tag-Vokabular und erben die Hierarchie automatisch, aber Sammlungen sind karten-only v1).
- **Kein** converter-mcp-/Agent-Touch (der Agent setzt weiter nur Tags; Hierarchie/Sammlungen sind User-UI).
- **Kein** Notion-Sync.
- **Kein** Horizont-vs-Kurs-Typ-Unterschied (eine `Collection`-Entität; ggf. später `kind`).
- **Kein** Drag-and-Drop-Baum-Editor (Dropdown reicht v1).

## Akzeptanz

- [ ] `Tag.parent_id` (Auto-Migration) → Tags bilden einen Wald; Zyklus-Guard; Tag-Delete reparentet Kinder
- [ ] `Collection` + `card_collections` (db.create_all); CRUD + Karte add/remove, owner-scoped, idempotent
- [ ] `/api/review-state` filtert nach `?tag=` (Teilbaum) und/oder `?collection=` (AND), Counts reflektieren den Scope; Default unverändert
- [ ] „Lernen"-Selektor (Alles fällig / Taxonomie-Ast / Sammlung) schaltet die Queue um; Karte→Sammlung beim Review
- [ ] `/tags` ordnet Tag-Hierarchie (Eltern-Dropdown) + Sammlungs-Verwaltung; lean, dark+light gesmoked
- [ ] `pytest` grün ≥ 443 + neue Tests; `node --check`; **converter-mcp unberührt**
