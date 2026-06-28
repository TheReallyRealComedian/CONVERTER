# Sprint TAG-CLEANUP — merge_tags + delete_tag (destruktive Tag-Tools, token-authed, dry-run-default) (M, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **502**). Du committest jede Phase selbst (Hash + push), **fokussiert** (nur diese Tools + Tests + Docs, **keine** unrelated Changes — SAFETY #4). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Backend + Docs**, kein UI, **kein Schema-Touch** (nutzt bestehende Junctions + `parent_id`), kein neuer Token, kein neuer Dep. Backend pytest-getestet.
>
> ⚠️ **Destruktive Tools** — lies den **SAFETY-Block am Ende ZUERST**. Tests **nur** gegen die leere Test-DB (conftest) auf synthetischen Wegwerf-Tags. **Kein Live-Cleanup**, nie gegen Prod-Daten.

## Warum & Entscheidungen (gesetzt, Oli 2026-06-28)

Das Tag-Vokabular ist auf ~150 flache Tags mit vielen Dubletten gewachsen (Großteil vom Newsletter-Auto-Tagger, der pro Topic einen Tag im **geteilten** Namespace mintet). Es gibt `set_tag_parent` (umhängen) + Tag-Erzeugung, aber **kein Merge/Delete** — Konsolidierung geht nur manuell in der UI. Dieser Sprint baut die zwei fehlenden destruktiven Tools, exponiert sie als CONVERTER-Endpoints (Token, dry-run-default). **Die tatsächliche Sprawl-Bereinigung (Source→Target-Map über ~140 Tags) fährt DANACH der Lern-Agent — NICHT dieser Sprint.**

- **Token-authed über den `CARD_TOKEN`-Gate** (`_authorize_card_write`, CSRF-exempt) — die Cleanup macht der Agent, nicht die Session-UI. Sechste/siebte Agent-Write-Surface über denselben Gate.
- **Endpoint-Shapes** (Präzedenz `POST /api/tags/parent` — token, by-name): **`POST /api/tags/merge`** + **`POST /api/tags/delete`**. POST (nicht DELETE-Verb): tragen Body (`dry_run` etc.) + sind by-name, **distinct** vom Session-`DELETE /api/tags/<id>` (by-id) → kein Path-Clash.
- **by-name, Lookup-only** (nicht get_or_create): `Tag.normalize_name(x)` → `Tag.query.filter_by(user_id=target.id, name=normalized).first()`. Source **und** Ziel müssen **existieren** (fehlt → 404). Grund: ein destruktiver Merge soll auf ein **bekanntes** Kanon-Tag konsolidieren, kein Phantom-Ziel aus einem Tippfehler anlegen — und Lookup-only hält den **dry-run garantiert schreibfrei** (kein get_or_create-create-dann-rollback). Existiert das Kanon-Ziel noch nicht, legt der Agent es vorher über die normalen Pfade an.
- **dry_run=true = DEFAULT** (beide Tools). Schreibender Lauf nur bei explizitem `dry_run=false`. **Mechanik: same-path-rollback** (Projekt-Präzedenz R2-E Backfill, Memory `reference_tag_vocab_central_gate_plus_backfill_script`): dieselbe Mutations-Logik laufen lassen, Counts aus den echten Statements (`rowcount`) **vor** dem Rollback abgreifen, dann bei dry_run `db.session.rollback()` statt `commit()` → die Vorschau ist **apply-treu**. ⚠️ Tag-`id`/`name`/Kinder-Namen **vor** dem Rollback in Python-Werte kopieren (nach Rollback sind die ORM-Objekte expired).
- **`delete_tag`-Guard-Rail = `force`-Flag** (Oli-Entscheid 2026-06-28): Signatur **`delete_tag(tag, reassign_to=None, dry_run=true, force=false)`**. Tag hat Objekte **und** kein `reassign_to` → `force=false` wird im echten Lauf **abgewiesen** (409 + Counts); erst **`force=true`** löst von allen Objekten und löscht. Honoriert beide Spec-Klauseln (dry-run-Preview + „Bestätigung verlangen"). (Im dry-run wird **nie** geblockt — dry-run ist immer 200-Preview + `requires_force`-Flag.)
- **Session-`DELETE /api/tags/<id>` bleibt UNANGETASTET** ([app_pkg/tags.py:188](app_pkg/tags.py) — getestet, by-id, session). Die neuen Token-Endpoints kriegen **eigene** Funktionen + einen geteilten Helper. (Kein DRY-Refactor der Session-View — das wäre eine unrelated Change gegen SAFETY #4.)
- **Tools sind GENERISCH — KEINE hardcodierte Protected-Tag-Liste.** Der Schutz der echten Studien-Tags ist Sache des **Lern-Agenten** beim späteren Live-Lauf (er kennt die Caveats), nicht des Tools.
- **converter-mcp**: wrappt danach die zwei neuen Token-Writes (Koordinator-Scope) — Brief in Phase 3. **End-to-end-Beweis (MCP-aufrufbar, `list_tags` spiegelt das Ergebnis) = Koordinator-Seite** nach dem Wrap, wie bei LERN-GROUP-AW.

## Verifizierte Code-Fakten (Master-gegroundet)

- **Drei Junctions**, alle teilen die `Tag`-Rows, importiert in [app_pkg/tags.py:14](app_pkg/tags.py): `card_tags` (Spalten `.c.tag_id`, `.c.card_id`), `highlight_tags` (`.c.tag_id`, `.c.highlight_id`), `conversion_tags` (`.c.tag_id`, `.c.conversion_id`). **Ein Merge muss alle drei umhängen.**
- **`Tag.normalize_name(name)`** ([models.py:155](models.py)) — lowercase+trim+Markdown-Artefakte-strip, `''` wenn nichts übrig bleibt. **Der Lookup-Normalisierer.**
- **`Tag.get_or_create(user_id, name)`** ([models.py:170](models.py)) — normalize + find-or-insert, `None` bei non-str/blank/oversize. Hier **nicht** für Lookup nutzen (es legt an) — nur als Referenz fürs `None`-Handling-Muster.
- **`Tag.subtree_ids(root_id, user_id)`** ([models.py:189](models.py)) — BFS-Set inkl. `root_id`. **Der Zyklus-Guard-Baustein.**
- **`Tag.children`** ([models.py:143](models.py)) Relationship (`backref='parent'`, `remote_side=[id]`) + `Tag.parent_id`. Reparenten via direktem `UPDATE` auf `parent_id` (wie die Session-Delete-View, Z.198–202) — konsistent.
- **Session-`api_delete_tag`** ([app_pkg/tags.py:188](app_pkg/tags.py)): reparentet Kinder→NULL (direkter `Tag.__table__.update().where(parent_id==tag.id).values(parent_id=None)`) + drainet alle drei Junctions (`<junction>.delete().where(.c.tag_id==tag.id)`) + `db.session.delete(tag)`. **Die SQL-Mechanik, die du spiegelst** (aber zu *target* statt NULL beim Merge / mit reassign).
- **`_authorize_card_write`** aliased als `_authorize_agent_write` ([app_pkg/tags.py:18](app_pkg/tags.py)) → `(target, err)`; `if err: return err`. **Reuse.**
- **`api_set_tag_parent`** ([app_pkg/tags.py:110](app_pkg/tags.py)) = die by-name/token/Zyklus-Guard-Präzedenz; **CSRF-exempt-Registry** am Ende von `register` ([app_pkg/tags.py:223](app_pkg/tags.py): `app.extensions['csrf'].exempt(api_set_tag_parent)`) — die zwei neuen Views dort ergänzen.
- **`_tag_name_error()`** ([app_pkg/tags.py:29](app_pkg/tags.py)) = 400-Helper für ungültige Namen.
- **SQLite ohne `PRAGMA foreign_keys`** (Memory `reference_sqlite_no_fk_pragma_orm_delete`) → Junctions + `parent_id` **explizit** behandeln, nie auf DB-FK-Actions verlassen.
- **Test-Scaffolding**: `tests/test_tag_parent_write.py` (Token-Client-Fixtures, owner-scoping, 503/401) — **spiegeln**. conftest fährt `WTF_CSRF_ENABLED=False`; CSRF-exempt-Beweis ggf. via Re-Enable in einem Test (Caveat wie NL1).

## Phase 1 — `merge_tags` (`POST /api/tags/merge`) + geteilter Reassign-Helper

**Endpoint** in [app_pkg/tags.py](app_pkg/tags.py), token, **CSRF-exempt**. Body `{source: str, target: str, dry_run: bool=true}`.

1. **Geteilter Helper** `_reassign_tag_refs(source_id, target_id)` (für Merge **und** delete-mit-reassign): pro Junction (`card_tags`/`highlight_tags`/`conversion_tags`) **dedup-dann-repoint**:
   - **dedup**: `DELETE FROM <j> WHERE tag_id=source_id AND <obj>_id IN (SELECT <obj>_id FROM <j> WHERE tag_id=target_id)` — Objekte, die **beide** Tags haben, würden sonst eine Dublette/Constraint-Verletzung erzeugen. `rowcount` = `deduped`.
   - **repoint**: `UPDATE <j> SET tag_id=target_id WHERE tag_id=source_id`. `rowcount` = `moved`.
   - Rückgabe: `{'cards': {'moved': …, 'deduped': …}, 'highlights': {…}, 'conversions': {…}}`.
2. **Endpoint-Logik** `api_merge_tags`:
   - `target_user, err = _authorize_agent_write(); if err: return err`.
   - `source_norm = Tag.normalize_name(<source>)`, `target_norm = Tag.normalize_name(<target>)`; leer → `_tag_name_error()`.
   - **`source_norm == target_norm` → No-op**: 200, Counts alle 0, `source_deleted: false` (nichts zu mergen).
   - **Lookup-only** beide: `Tag.query.filter_by(user_id=target_user.id, name=source_norm).first()` / `…=target_norm…`. `source` fehlt → 404 „Quell-Tag nicht gefunden."; `target` fehlt → 404 „Ziel-Tag nicht gefunden."
   - **Zyklus-Guard**: `if target.id in Tag.subtree_ids(source.id, target_user.id) and target.id != source.id: return 400` „Merge würde einen Zyklus erzeugen (Ziel liegt im Teilbaum der Quelle) — erst via set_tag_parent entwirren." (Ziel ist echter Nachfahre der Quelle → die Kinder-Umhängung würde eine Schleife bauen.)
   - **Reassign**: `counts = _reassign_tag_refs(source.id, target.id)`.
   - **Kinder reparenten** → target: `UPDATE tag SET parent_id=target.id WHERE parent_id=source.id` (sicher, weil der Guard target ∉ source-Teilbaum garantiert → kein Zyklus). Namen/IDs der betroffenen Kinder **vorher** per SELECT abgreifen für die Response.
   - **Source löschen**: `db.session.delete(source)`.
   - **dry-run-Schalter**: Counts/Kinder/IDs jetzt in Python-Dicts kopieren; `if dry_run: db.session.rollback()` else `db.session.commit()`.
   - Response 200: `{dry_run, applied: not dry_run, source:{id,name}, target:{id,name}, reassigned: counts, children_reparented: [{id,name}], source_deleted: not dry_run}`.
3. **CSRF-exempt** registrieren (am Ende von `register`, neben `api_set_tag_parent`).
4. **Tests** (`tests/test_tag_merge.py`, Token-Scaffolding aus test_tag_parent_write.py — **synthetische Tags only**):
   - merge: source mit Cards+Highlights+Conversions → alle auf target umgehängt, source gelöscht, Counts korrekt; `list_tags` (oder direkter Query) zeigt source weg, target trägt alles.
   - **dedup**: ein Objekt an **beiden** Tags → nach Merge **ein** Link (target), kein Duplikat; `deduped`-Count stimmt.
   - **Kinder**: source hat Kind-Tags → `parent_id` jetzt = target.
   - **Zyklus-Guard**: target ist Nachfahre von source → 400, **nichts** geändert.
   - **source==target** (gleicher Name nach Normalisierung) → No-op, source bleibt, Counts 0.
   - **dry_run=true (Default)**: Counts kommen, aber **nichts geschrieben** (source existiert noch, Links intakt — per zweitem Read verifizieren).
   - **idempotent**: echter Lauf, dann zweiter echter Lauf (source weg) → 404.
   - source/target fehlt → 404; Token fehlt/falsch → 503/401; owner-scoped (Tags am Token-Target-User).

**Stop + Bericht.**

## Phase 2 — `delete_tag` (`POST /api/tags/delete`) + reassign_to + force-Guard-Rail

**Endpoint** in [app_pkg/tags.py](app_pkg/tags.py), token, **CSRF-exempt**. Body `{tag: str, reassign_to: str|null=null, dry_run: bool=true, force: bool=false}`.

1. **Endpoint-Logik** `api_delete_tag_token` (Name distinct von der Session-`api_delete_tag`!):
   - `target_user, err = _authorize_agent_write(); if err: return err`.
   - `tag_norm = Tag.normalize_name(<tag>)`; leer → `_tag_name_error()`. Lookup-only → fehlt → 404 „Tag nicht gefunden."
   - **Objekt-Counts** des Tags (cards/highlights/conversions) per `SELECT count` über die drei Junctions. `has_objects = any > 0`.
   - **Mit `reassign_to`**:
     - `re_norm = normalize`; leer → 400. Lookup → fehlt → 404 „Ziel-Tag (reassign_to) nicht gefunden." `reassign_to == tag` → 400 „reassign_to == tag."
     - **Zyklus-Guard** (wie Merge): `reassign_to.id in subtree_ids(tag.id)` und `!= tag.id` → 400.
     - `counts = _reassign_tag_refs(tag.id, reassign_to.id)`; Kinder → reassign_to (`UPDATE parent_id`); `delete(tag)`.
   - **Ohne `reassign_to`**:
     - **`has_objects and not force`** → **GUARD-RAIL**:
       - `dry_run=true` (Default): **200**-Preview, Counts + `requires_force: true`, `tag_deleted: false` (nichts geschrieben). **Kein** Fehler im dry-run.
       - `dry_run=false`: **409** + Counts + `requires_force: true` „Tag hat N angehängte Objekte und kein reassign_to — reassign_to setzen oder force=true." **Nichts** geschrieben.
     - sonst (**keine Objekte**, ODER `force=true`): **detach-all** (drei Junctions `delete().where(tag_id==tag.id)`) + Kinder → **NULL** (Wurzel, wie die Session-Delete-Semantik) + `delete(tag)`.
   - **dry-run-Schalter**: Counts/Kinder vorher kopieren; `if dry_run: rollback()` else `commit()` (außer der 409-Refuse-Pfad oben, der vor jeder Mutation returnt).
   - Response 200: `{dry_run, applied, tag:{id,name}, reassign_to:{id,name}|null, affected: counts, children_reparented:[…], requires_force: bool, tag_deleted: bool}`.
2. **CSRF-exempt** registrieren.
3. **Tests** (`tests/test_tag_delete.py` — synthetische Tags):
   - **mit reassign_to**: Objekte umgehängt, Kinder → reassign_to, tag gelöscht; dedup wie Merge.
   - **ohne reassign_to, hat Objekte, force=false, dry_run=false** → **409** + Counts + `requires_force`, tag **NICHT** gelöscht, Objekte unberührt.
   - **ohne reassign_to, hat Objekte, force=true** → detach-all + Kinder→NULL + gelöscht.
   - **ohne reassign_to, KEINE Objekte** → direkt gelöscht (force egal).
   - **dry_run=true** in jedem Pfad → Counts/Flags, **nichts** geschrieben; der Guard-Rail-Pfad liefert `requires_force: true` ohne 409.
   - reassign_to-Zyklus → 400; reassign_to==tag → 400; tag/reassign_to fehlt → 404; idempotent (2. Lauf → 404); Token 503/401; owner-scoped.

**Stop + Bericht.**

## Phase 3 — Docs + Wrap-Brief + Wrap-up

1. **`docs/card_api_contract.md`** — „Zu wrappen" von 8 → **10 Tools**; in „Writes (Bearer `CARD_TOKEN`)" zwei Einträge:
   - **`POST /api/tags/merge`** (`{source, target, dry_run}`, by-name Lookup-only, dedup über 3 Junctions, Kinder→target, Zyklus-Guard, dry-run-default; Tool z.B. `merge_tags`).
   - **`POST /api/tags/delete`** (`{tag, reassign_to, dry_run, force}`, force-Guard-Rail 409, dry-run-default; Tool z.B. `delete_tag`).
2. **`docs/converter_mcp_tag_cleanup_brief.md`** (neuer Brief, Muster `converter_mcp_lern_group_brief.md`) — an den mcp-developer: die zwei neuen Token-Writes wrappen (`merge_tags`/`delete_tag`, default dry-run), Signaturen, die force-Semantik, der Hinweis dass `list_tags` das Ergebnis spiegelt; **End-to-end-Beweis (MCP-aufrufbar, auf synthetischen Tags) = Koordinator-Seite**.
3. **`docs/card_agent_guide.md`** + **`docs/card_agent_intro.md`** — knapper Zusatz unter „Karten organisieren"/„Gruppieren mit Disziplin": der Agent kann jetzt auch **konsolidieren** — `merge_tags` (Dubletten zusammenführen) + `delete_tag` (Junk entfernen), **immer erst `dry_run` lesen**, dann anwenden; Grenze bleibt: Löschen/Umbenennen-via-UI ist weiterhin auch User-Sache, aber Cleanup ist jetzt agent-fähig. (Intro: 1–2 Sätze, Doktrin-Schicht.)
4. **STATUS.md** + **BACKLOG.md**: TAG-CLEANUP ☑ done (Hashes); **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md`).
5. **Memory** (`reference_*`): destruktive Tag-Tools — merge/delete by-name Lookup-only über den `CARD_TOKEN`-Gate; dedup-dann-repoint über **alle drei** Junctions; Kinder→target/NULL; Zyklus-Guard via `subtree_ids` (Merge in eigenen Teilbaum abgelehnt, nicht aufgelöst — konservativ wie set_tag_parent); **dry-run = same-path-rollback** (apply-treu, IDs vor Rollback kopieren); `delete_tag`-**force**-Guard-Rail; Session-Delete unangetastet. MEMORY.md-Pointer. (Geschwister: `reference_sqlite_no_fk_pragma_orm_delete`, `reference_tag_vocab_central_gate_plus_backfill_script`, `reference_two_axis_card_grouping`.)
6. Finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Deploy-Kette: Mac push → Mintbox `git pull` + `docker compose up -d --build` (**kein Schema, keine Migration, kein neuer Token** — `CARD_TOKEN` steht; reines Backend) → dann **converter-mcp** die zwei Writes wrappen (Koordinator-Scope); End-to-end auf synthetischen Tags = Koordinator.

## SAFETY (hart — vom Koordinator, wörtlich)

1. **Beide Tools default `dry_run=true`** — destruktiver Lauf nur bei explizitem Opt-out (`dry_run=false`).
2. **Tests laufen NUR auf synthetischen Wegwerf-Tags** (die leere conftest-Test-DB — alle Test-Tags sind dort synthetisch). **NIE** gegen das echte Vokabular / die Prod-DB. **Kein** Live-Cleanup als „Acceptance-Test" auf Produktivdaten.
3. **Diese echten Tags NIE als Testobjekt** (echtes Studienmaterial — sie liegen in Prod, nicht in der Test-DB; relevant erst beim späteren Live-Lauf des Agenten, nicht hier): `bi-pipeline`, `zongertinib`, `nerandomilast`, `plattform-strategie`, `mechanistisches-verständnis`, `fibrose`, `onkologie` + die Baum-Mid-Buckets (`pharma`, `asset`, `indikation`, `strategie`).
4. **Fokussierter Commit** — nur diese Tools + Tests + Docs, **keine** unrelated Changes mitbündeln.

## Bewusst NICHT (Scope-Grenze)

- **Die Merge-Map ausführen** (den echten Sprawl bereinigen) — macht der **Lern-Agent** danach. Dieser Sprint endet, wenn die Tools stehen, getestet + committet sind.
- **Den Newsletter-Tagger** anfassen (anderes Repo, fertig).
- **Geschützte Tags** ändern; **keine** hardcodierte Protected-Liste in den Tools (Schutz = Agent-Sache beim Live-Lauf).
- **Session-`DELETE /api/tags/<id>`** refaktorieren/anfassen (getestet, bleibt; kein DRY-Merge gegen SAFETY #4).
- **MCP-Code** selbst (Koordinator wrappt; hier nur der Brief).
- **Kein** Schema-Touch, kein neuer Token, kein neuer Dep, kein UI.

## Akzeptanz

- [ ] `POST /api/tags/merge` (token, by-name Lookup-only): hängt Cards+Highlights+Conversions von source auf target um (dedup), reparentet Kinder→target, löscht source; **Zyklus-Guard**; `source==target`→No-op; **dry-run-default** liefert akkurate Counts **ohne** Schreibwirkung; idempotent; 404/503/401; owner-scoped.
- [ ] `POST /api/tags/delete` (token, by-name): mit `reassign_to` umhängen-dann-löschen; ohne + Objekte + `force=false` → **409**+Counts (dry-run: `requires_force` ohne 409); `force=true` → detach-all+löschen; keine Objekte → direkt löschen; **dry-run-default**; idempotent; 404/503/401; owner-scoped.
- [ ] geteilter `_reassign_tag_refs`-Helper (dedup-dann-repoint, 3 Junctions); beide Endpoints CSRF-exempt; **Session-Delete unberührt**.
- [ ] `card_api_contract.md` (10 Tools) + neuer converter-mcp-Brief + agent-guide/intro-Zusatz.
- [ ] `pytest` grün ≥ 502 + neue Tests (test_tag_merge.py, test_tag_delete.py); **kein** Schema-Touch; fokussierte Commits.
