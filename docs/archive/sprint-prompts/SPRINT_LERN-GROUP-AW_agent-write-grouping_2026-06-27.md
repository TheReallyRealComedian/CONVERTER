# Sprint LERN-GROUP-AW — Agent schreibt die Gruppierung (Token: Sammlungen + Tag-Baum) (M, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **481**). Du committest jede Phase selbst (Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Backend + Docs**, kein UI, **kein Schema-Touch** (Tabellen/Spalten stehen aus LERN-GROUP). Backend pytest-getestet (Mail/UI-frei).

## Warum & Entscheidungen (gesetzt, Oli 2026-06-27)

**Der Kern-Use-Case: der Agent schreibt + organisiert die Karten, nicht der User.** LERN-GROUP baute die Gruppierung als User-UI (Session+CSRF) — der Agent erreicht das über seinen Bearer-Token **nicht**. Dieser Sprint macht die Gruppierung **agent-schreibbar** über den bestehenden **`CARD_TOKEN`-Gate** (`_authorize_card_write`, CSRF-exempt) — die nächste Agent-Write-Surface nach Card/Highlight-Annotate/Doc.

- **Sammlungen (Achse B)** → über die **bestehende Card-Write-API**: `create_card` **und** `update_card` (beide token, existieren) kriegen ein Feld **`collections: [namen]`** → **get_or_create by-name** (owner-scoped, wie `tags`) + Karte zuordnen (Voll-Ersetzung). Der Agent erzeugt+taggt+sammelt in **einem** Call. **Frei anlegen** (Oli: maximale Agent-Autonomie; Aufräumen = User-UI).
- **Taxonomie (Achse A)** → der Agent **baut den Baum**: ein **token-authed** Endpoint setzt `parent_id` **by-name** (get_or_create beide Tags). Der Agent ordnet neue Themen selbst ein.
- **Löschen/Umbenennen** von Sammlungen + die Verwaltungs-UI bleiben **Session** (User-Kuratierung, LERN-GROUP P4) — **nicht** anfassen.
- **converter-mcp**: wrappt danach die neuen Token-Writes (Koordinator) — der Brief wird in Phase 3 aktualisiert.

## Verifizierte Code-Fakten (Master-gegroundet)

- **Card-Write-Gate**: `_authorize_card_write()` (CARD_TOKEN, 503/401, INGEST_USER-Target) — schon von create/patch_card + Highlight-Annotate + Doc-Write genutzt. **Reuse.**
- **Tag-Pattern**: `_replace_card_tags(card, names, user_id)` ([app_pkg/cards.py:188](app_pkg/cards.py)) — `card.tags=[]`, `Tag.get_or_create` je Name. **Vorbild für Collections.** Gewired in `api_create_card` (Z.307 `_replace_card_tags(card, data.get('tags'), target.id)`) + `api_patch_card` (Z.347–350 `if 'tags' in data: …`).
- **CSRF-exempt-Registry**: cards.py-Ende (Z.581–583, create/patch_card + annotate_highlight); Doc-Writes in docwrite.py. Neue Token-Views ergänzen.
- **`Collection`** ([models.py:375](models.py)): `UniqueConstraint(user_id, name)`, `MAX_NAME_LEN=120`, **`name` case-sensitiv gespeichert**, **kein `get_or_create`/`normalize_name`** (existiert nur für `Tag`, das **lowercased** — für Sammlungen ungeeignet, es sind Eigennamen wie „Boehringer-Pipeline").
- **`Tag.parent_id`** + `Tag.subtree_ids(root,user)` (BFS) + die Session-`PATCH /api/tags/<id>`-Zyklus-Logik (`parent_id in subtree_ids(tag)` → ablehnen) aus LERN-GROUP P1 — der Token-Endpoint **spiegelt** den Zyklus-Guard.
- **`Card.collections`** = `relationship('Collection', secondary=card_collections, backref='cards')` (owning side auf Card). `Card.to_dict()` — **prüfen, ob `collections` drin ist; falls nicht, ergänzen** (`[{id,name}]`), damit der Agent sein Write verifiziert (wie `tags`).

## Phase 1 — Sammlungen über die Card-Write-API (`collections:[namen]`)

1. **`Collection.normalize_name(name)`** ([models.py](models.py)): **trim + interne Whitespace-Runs kollabieren, Case ERHALTEN** (Eigennamen — **nicht** lowercasen). Leer/oversize-Handling beim Caller.
2. **`Collection.get_or_create(user_id, name)`** (Muster `Tag.get_or_create`, aber mit `Collection.normalize_name`): non-str/blank-nach-Normalisierung/`>MAX_NAME_LEN` → `None`; sonst find-by-(user_id, normalized-name) oder anlegen+flush. Case-sensitiv (Sprawl bewusst akzeptiert — UI räumt auf).
3. **`_replace_card_collections(card, names, user_id)`** ([app_pkg/cards.py](app_pkg/cards.py), neben `_replace_card_tags`): `card.collections=[]`; non-list → return; je Name `Collection.get_or_create` + anhängen (Dedup wie bei Tags). **Caveat**: vor dem Collection-Touch muss die Karte in der Session sein (wie der `_replace_card_tags`-Call-Site-Kommentar bei create_card — `db.session.add(card)` zuerst).
4. **Wiring**: `api_create_card` — nach `_replace_card_tags` ein `_replace_card_collections(card, data.get('collections'), target.id)`. `api_patch_card` — `if 'collections' in data: if not isinstance(list) → 400; _replace_card_collections(...)`.
5. **`Card.to_dict`** um `collections: [{id,name}]` ergänzen (falls noch nicht drin).
6. **Tests** (`tests/test_card_collections_write.py` o.ä., Token-Scaffolding aus test_cards.py): create_card mit `collections:[A,B]` → get_or_create + zugeordnet, Response trägt sie; bestehende Sammlung wiederverwendet (gleiche id); `update_card {collections:[…]}` ersetzt; `[]` leert; non-list → 400; **case/whitespace-Normalisierung** („ Boehringer-Pipeline "/„Boehringer-Pipeline" → eine Row, Case erhalten); owner-scoped (Sammlung am Token-Target-User); Token fehlt/falsch → 503/401. `pytest` grün ≥ 481.

**Stop + Bericht.**

## Phase 2 — Token-Endpoint: Tag-Baum bauen (by-name)

1. **`POST /api/tags/parent`** (token, **distinct** vom Session-`PATCH /api/tags/<id>` — kein Path-Clash), in [app_pkg/tags.py](app_pkg/tags.py) (Gate aliased importieren: `from .cards import _authorize_card_write as _authorize_agent_write`): Body `{tag: str, parent: str|null}`.
   - `target, err = _authorize_agent_write(); if err: return err`.
   - `tag = Tag.get_or_create(target.id, data['tag'])` (non-blank → sonst 400). `parent`:
     - `null` → `tag.parent_id = None` (entwurzeln).
     - str → `parent = Tag.get_or_create(target.id, data['parent'])`; **Zyklus-Guard** `parent.id in Tag.subtree_ids(tag.id, target.id)` → 400 (fängt Selbst-Referenz + Eltern-im-Teilbaum); sonst `tag.parent_id = parent.id`.
   - commit; Response = `tag.to_dict()` (mit `parent_id`).
2. **CSRF-exempt** registrieren (in tags.py, am Ende von `register`: `app.extensions['csrf'].exempt(<view>)`).
3. **Tests** (`tests/test_tag_parent_write.py`): Parent by-name setzen (beide Tags neu → get_or_create); unparent (`null`); Zyklus → 400; Token fehlt/falsch → 503/401; owner-scoped (Tags am Target-User); danach wirkt der Baum im `?tag=`-Review-Filter (Teilbaum). `pytest` grün.

**Stop + Bericht.**

## Phase 3 — Docs + Wrap

1. **`docs/card_api_contract.md`** — in „Writes (Bearer `CARD_TOKEN`)": `create_card`/`update_card` um das **`collections: [namen]`**-Feld (get_or_create by-name, Voll-Ersetzung) ergänzen; **neuer Eintrag `POST /api/tags/parent`** (`{tag, parent|null}`, by-name, Zyklus-Guard, Tool z.B. `set_tag_parent`).
2. **`docs/converter_mcp_lern_group_brief.md`** — den „Writes NICHT erreichbar"-Abschnitt **auflösen**: jetzt **gibt** es Token-Writes (Sammlungen via card-write `collections`, Tag-Baum via `POST /api/tags/parent`) → für den MCP zu wrappen. Die Reads bleiben wie beschrieben. Die „offene Entscheidung" auf ☑ umstellen (Agent-Write = ja, gebaut).
3. **`docs/card_agent_guide.md`** — der Agent **organisiert jetzt auch**: `collections`-Feld bei create/update_card (Karten in Horizonte/Kurse legen), `set_tag_parent` (Themen in den Baum einordnen), + Hinweis „Tag-Hierarchie via `list_tags`/`parent_id` lesen, um konsistent einzuordnen". Grenze: Sammlungen löschen/umbenennen bleibt User-UI.
4. **STATUS.md** + **BACKLOG.md**: LERN-GROUP-AW ☑ done (Hashes); „Aktiv offen" → Web-Article-Save (P2). **Bullet-Guard.**
5. **Memory** (`reference_*`): Agent-Write-Grouping — Sammlungen über die Card-Write-API (get_or_create **by-name, case-erhaltend** im Gegensatz zu Tags) + token-`POST /api/tags/parent` (by-name, Zyklus-Guard-Spiegel); die **fünfte** Agent-Write-Surface über `_authorize_card_write`; Lösch/Umbenennen bewusst session-only. MEMORY.md-Pointer.
6. Finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Deploy-Kette: Mac push → Mintbox `git pull` + `docker compose up -d --build` (**keine Migration, kein neuer Token** — `CARD_TOKEN` steht) → dann **converter-mcp** die neuen Writes wrappen (`collections`-Param an create/update_card + `set_tag_parent`) — Koordinator-Scope; End-to-End-Beweis = Koordinator.

## Bewusst NICHT (Scope-Grenze)

- **Kein** Session-/UI-Touch (LERN-GROUP P3/P4 bleiben; Löschen/Umbenennen von Sammlungen + der Tag-Baum-Editor sind User-UI).
- **Kein** token-Delete/Rename für Sammlungen (Agent legt an + ordnet zu; Aufräumen = User).
- **Kein** Schema-Touch, kein neuer Token, kein neuer Dep.
- **Kein** `/api/cards`-Tag/Collection-Filter (separates optionales Item, falls der Agent Karten nach Gruppe *abfragen* will — hier out).

## Akzeptanz

- [ ] `create_card`/`update_card` akzeptieren `collections: [namen]` → get_or_create **by-name, case-erhaltend**, owner-scoped, Voll-Ersetzung; Response trägt `collections`
- [ ] `POST /api/tags/parent` (token, by-name) setzt/löst `parent_id`, get_or_create beide Tags, **Zyklus-Guard**, 503/401
- [ ] beide über `_authorize_card_write` (CARD_TOKEN) + CSRF-exempt; Session-Endpoints + UI unberührt
- [ ] `card_api_contract.md` + converter-mcp-Brief (Writes jetzt da) + `card_agent_guide.md` aktualisiert
- [ ] `pytest` grün ≥ 481 + neue Tests; kein Schema-Touch
