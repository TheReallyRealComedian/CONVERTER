# Developer-Brief an das converter-mcp-Team — TAG-CLEANUP (destruktive Tag-Tools)

> **An**: converter-mcp-Entwickler (Koordinator-Repo).
> **Von**: CONVERTER-Master, 2026-06-28.
> **Worum**: Das Tag-Vokabular ist auf ~150 flache Tags mit vielen Dubletten gewachsen (Großteil vom Newsletter-Auto-Tagger). Bisher gab es `set_tag_parent` (umhängen) + Tag-Erzeugung, aber **kein Merge/Delete** — Konsolidierung ging nur manuell in der UI. CONVERTER hat jetzt die zwei fehlenden **destruktiven** Token-Writes: **`merge_tags`** + **`delete_tag`**. Dieser Brief sagt, was der converter-mcp davon wrappt — und mit welcher **Sicherheits-Doktrin**.

## TL;DR (bitte zuerst lesen)

- **Zwei neue token-authed Writes** (`Authorization: Bearer <CARD_TOKEN>`, CSRF-exempt — **derselbe** Token wie Card-/Highlight-/Doc-/Gruppierungs-Writes, **kein neuer Token**, **kein Schema/Dep**). Für den MCP zu wrappen.
- **Beide default `dry_run=true`.** Ein destruktiver Lauf passiert **nur** bei explizitem `dry_run=false`. Die dry-run-Antwort ist **apply-treu** (echte Counts, aber nichts geschrieben — same-path-rollback CONVERTER-seitig).
- **Doktrin fürs Tool**: der Agent soll **immer erst `dry_run` lesen**, die Counts prüfen, **dann** `dry_run:false` schicken. Bitte im Tool-Doc/-Default so verankern (siehe unten).
- **`merge_tags`** und **`delete_tag`** sind **by-name Lookup-only** — beide referenzierten Tags müssen **existieren** (fehlt → 404). Ein Merge konsolidiert auf ein **bekanntes** Kanon-Tag; kein Phantom-Ziel aus einem Tippfehler.
- **`list_tags` spiegelt das Ergebnis** — nach einem echten Lauf ist das gemergte/gelöschte Tag weg, das Ziel trägt alles. Das ist der **End-to-end-Beweis** (siehe unten, **Koordinator-Scope**).
- **Löschen/Umbenennen von Tags via UI** bleibt **auch** User-Sache (Session) — der Agent kann jetzt konsolidieren, aber die kuratierende Hand bleibt beim User. Kein Widerspruch: zwei Pfade, gleiche Tag-Rows.

## Die zwei Writes — Signaturen

Voller Kontrakt mit den exakten Response-Shapes in [docs/card_api_contract.md](docs/card_api_contract.md) (→ 10 Tools). Hier das Wrap-Wesentliche:

### `merge_tags` → `POST /api/tags/merge`
Body `{source, target, dry_run}`. Hängt alle Refs (Cards+Highlights+Conversions) von `source` auf `target` um (**Dedup**: trägt ein Objekt beide Tags, wird der Dublett-Link entfernt statt umgehängt), reparentet `source`-Kinder → `target`, **löscht `source`**.
- `source`/`target` — Pflicht-non-blank, **by-name Lookup-only**, beide müssen existieren (fehlt → **404**).
- `dry_run` — Default **`true`**; nur echtes `false` löst aus.
- `source == target` → **No-op** (200, Counts 0).
- **Zyklus-Guard**: `target` echter Nachfahre von `source` → **400** (erst via `set_tag_parent` entwirren).
- **Response** (200): `{dry_run, applied, source:{id,name}, target:{id,name}, reassigned:{cards/highlights/conversions:{moved,deduped}}, children_reparented:[{id,name}], source_deleted}`.

### `delete_tag` → `POST /api/tags/delete`
Body `{tag, reassign_to=null, dry_run=true, force=false}`. **by-name**, distinct vom Session-`DELETE /api/tags/<id>` (by-id, bleibt UI-only).
- `tag` — Pflicht, Lookup-only (fehlt → **404**).
- `reassign_to` — String **oder** `null`. Gesetzt → Refs auf `reassign_to` umhängen (Dedup wie Merge), `tag`-Kinder → `reassign_to`, `tag` löschen. Fehlt → **404**; `== tag` → **400**; Zyklus → **400**.
- **`force`-Guard-Rail** (ohne `reassign_to`): hat `tag` Objekte **und** `force=false`:
  - **dry-run** (Default): **200**-Preview mit `requires_force:true`, `tag_deleted:false` (**kein** 409, nichts geschrieben).
  - **echter Lauf**: **409** + Counts + `requires_force:true` (nichts geschrieben).
  - `force=true` → von allen Objekten lösen + Kinder → NULL + löschen. Keine Objekte → direkt löschen (force egal).
- **Response** (200): `{dry_run, applied, tag:{id,name}, reassign_to:{id,name}|null, reassigned:{…}|null, affected:{cards,highlights,conversions}, children_reparented, requires_force, tag_deleted}`. `affected` = Objekt-Counts (immer da, treibt die Guard-Rail); `reassigned` = das `moved`/`deduped`-Detail (nur bei `reassign_to`, sonst `null`).

## Strikte Truthiness (wichtig für den Wrapper)

CONVERTER liest `dry_run`/`force` **strikt nach Identität**, nicht nach Truthiness:
- `dry_run`: der Write feuert **nur** bei echtem JSON-`false`. Ein falsy Non-Bool (`0`, `""`) **oder** ein `"false"`-**String** bleibt sicher dry-run.
- `force`: die Guard-Rail lüftet **nur** bei echtem JSON-`true`.

→ **Der MCP-Wrapper muss `dry_run`/`force` als echte JSON-Booleans durchreichen** (nicht als Strings). Schickt der Wrapper `"false"` statt `false`, bleibt es — by design — ein harmloser dry-run, und der Agent wundert sich, warum „nichts passiert". Bitte als bool typisieren.

## Empfehlung fürs converter-mcp

1. **`merge_tags`** wrappen (`POST /api/tags/merge`) — Param `source`, `target`, `dry_run: bool = true`.
2. **`delete_tag`** wrappen (`POST /api/tags/delete`) — Param `tag`, `reassign_to: str|null = null`, `dry_run: bool = true`, `force: bool = false`.
3. **Tool-Doc-Doktrin**: in beiden Tool-Beschreibungen verankern — *„Destruktiv. Immer erst mit `dry_run=true` (Default) aufrufen, die Counts im Ergebnis prüfen, dann mit `dry_run=false` anwenden. Bei `delete_tag` mit angehängten Objekten ohne `reassign_to` braucht der echte Lauf `force=true`."* So bleibt die Doktrin am Tool, nicht im Gedächtnis des Agenten.
4. **Kein neues Read nötig** — `list_tags` (bereits gewrappt, mit `parent_id` + `card_count`) ist der Spiegel: nach einem echten Merge/Delete ist das Quell-/gelöschte Tag weg.

## End-to-end-Beweis = Koordinator-Scope

Nach dem Wrap: **MCP-aufrufbar auf synthetischen Wegwerf-Tags** beweisen (wie bei LERN-GROUP-AW) —
1. zwei Wegwerf-Tags anlegen (z.B. via `set_tag_parent`/`create_card`),
2. `merge_tags` dry-run → Counts prüfen, dann `dry_run:false` → `list_tags` zeigt die Quelle weg, das Ziel trägt alles,
3. `delete_tag` analog (ohne Objekte → direkt; mit Objekten → `force=true` oder `reassign_to`).
**Nie gegen das echte Vokabular** — die reale ~140-Tag-Sprawl-Bereinigung fährt **danach** der Lern-/Projekt-Agent mit Handprüfung (er kennt die geschützten Studien-Tags), **nicht** dieser Wrap-Test.

---

*CONVERTER-Seite: TAG-CLEANUP fertig + getestet (+32 Tests), committet. **Kein Schema-Touch, keine Migration, kein neuer Token, kein neuer Dep** — `CARD_TOKEN` steht. Deploy: Mac push → Mintbox `git pull` + `docker compose up -d --build`. Geschwister-Brief: [docs/converter_mcp_lern_group_brief.md](docs/converter_mcp_lern_group_brief.md), Kontrakt-Muster: [docs/card_api_contract.md](docs/card_api_contract.md).*
