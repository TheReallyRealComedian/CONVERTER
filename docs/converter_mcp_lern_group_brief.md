# Developer-Brief an das converter-mcp-Team — LERN-GROUP (Karten-Gruppierung)

> **An**: converter-mcp-Entwickler (Koordinator-Repo).
> **Von**: CONVERTER-Master, 2026-06-27.
> **Worum**: CONVERTER kann jetzt **Lernkarten gruppieren** (Sprint LERN-GROUP) über zwei Achsen — **Taxonomie-Baum** (hierarchische Tags) + **Sammlungen** (kuratierte Bündel), beide als Filter auf die Wiederhol-Queue. **Update 2026-06-27 (Sprint LERN-GROUP-AW)**: der Agent kann die Gruppierung jetzt **auch schreiben** — über token-authed Pfade. Dieser Brief sagt, **was der converter-mcp davon weiterreichen kann**.

## TL;DR (bitte zuerst lesen)

- **READS** (`GET`): `@login_required` (Session). Der converter-mcp wrappt sie **sofort** mit seiner **bestehenden Session** (wie `list_audio_transcripts`/`list_cards`). **Keine CONVERTER-Änderung nötig.**
- **GRUPPIERUNGS-WRITES** (LERN-GROUP-AW, **token-authed + CSRF-exempt**, `Authorization: Bearer <CARD_TOKEN>` — derselbe Token wie Card-/Highlight-/Doc-Writes, **kein neuer Token**): **jetzt da, für den MCP zu wrappen.**
  - **Sammlungen** (Achse B) → **kein eigener Endpoint**, sondern ein **`collections: [namen]`-Feld an `create_card`/`update_card`** (get_or_create by-name, Voll-Ersetzung). Der Agent erzeugt+taggt+sammelt in **einem** Call.
  - **Tag-Baum** (Achse A) → **`POST /api/tags/parent`** `{tag, parent|null}` (by-name, Zyklus-Guard).
- **KURATIER-WRITES** (Sammlungen löschen/umbenennen, der Tag-Baum-Editor): bleiben **Session+CSRF** (User-UI) — bewusst **nicht** token-exposed (Aufräumen ist User-Sache).

## Was LERN-GROUP gebaut hat (Kontext)

- **Achse A — Taxonomie**: das bestehende, **geteilte** `Tag`-Vokabular ist jetzt ein **Wald** (`Tag.parent_id`, NULL = Wurzel). Karten hängen wie bisher an Tags (`card_tags`, vom Agent via `create_card` gesetzt) — die Hierarchie ordnet diese Tags, **keine Karte wird neu getaggt**. Filter = „Karten, deren Tag im **Teilbaum** eines Knotens liegt".
- **Achse B — Sammlungen**: neue `Collection`-Entität (flach, kuratiert, user-scoped) + `card_collections`-M2M. Eine Karte in beliebig vielen Sammlungen. Cross-cutting zur Taxonomie.

## READS — jetzt wrappbar (Session, keine CONVERTER-Änderung)

### `GET /api/tags` — **angereichert**
Liefert pro Tag jetzt **`parent_id`** (Hierarchie) + **`card_count`**:
```json
{"id": 7, "name": "transformer-modelle", "parent_id": 3, "created_at": "…",
 "highlight_count": 0, "conversion_count": 0, "card_count": 12}
```
→ Der Agent kann den **Themen-Wald** lesen (Wurzeln = `parent_id: null`, Kinder über `parent_id`) und seine `create_card`-`tags` **konsistent unter bestehende Themen** legen. Ein `list_tags`-Tool (bzw. das bestehende erweitern) genügt.

### `GET /api/collections` — neu
```json
[{"id": 2, "name": "Boehringer-Pipeline", "description": null,
  "created_at": "…", "card_count": 8}]
```
→ `list_collections`-Tool: der Agent sieht, welche Bündel der User kuratiert hat.

### `GET /api/review-state?tag=<id>&collection=<id>` — Filter ergänzt
Die **fällige** Queue, optional gefiltert: `?tag=` = Teilbaum von `<id>`, `?collection=` = Sammlung, **kombinierbar → AND**. Owner-scoped (fremd/unbekannt → 404), `total_count` reflektiert den Scope. *Das ist die User-Lern-Queue — für den Agent meist weniger relevant, aber verfügbar.*

## GRUPPIERUNGS-WRITES — jetzt token-authed, für den MCP zu wrappen (LERN-GROUP-AW)

Bearer-`CARD_TOKEN`, CSRF-exempt, Ziel-User server-seitig (`INGEST_USER`/first()). **Voller Kontrakt** in [docs/card_api_contract.md](docs/card_api_contract.md).

| Pfad | Write | Zweck |
|---|---|---|
| `POST`/`PATCH /api/cards` | `collections: [namen]` (neues Feld) | Karte in N Sammlungen legen (get_or_create **by-name, case-erhaltend**, Voll-Ersetzung) |
| `POST /api/tags/parent` | `{tag, parent\|null}` | Thema in den Baum einordnen (by-name, **lowercased** Vokabular, Zyklus-Guard) |

→ MCP-Arbeit: den bestehenden `create_card`/`update_card`-Wrappern den **`collections`-Param** zugeben + ein neues Tool **`set_tag_parent`** für `POST /api/tags/parent`.

## KURATIER-WRITES — bleiben Session+CSRF (NICHT token-exposen)

Diese sind bewusst **User-Kuratierung** (UI, nicht Agent): Sammlungen löschen/umbenennen, der Tag-Baum-Verwaltungs-Editor. Der MCP wrappt sie **nicht**.

| Endpoint | Zweck |
|---|---|
| `PATCH /api/tags/<id>` `{parent_id}` | Session-Variante (Tag-Manager-UI; by-id) |
| `POST /api/collections` `{name, description?}` | Sammlung explizit anlegen (UI) |
| `PATCH /api/collections/<id>` `{name?, description?}` | umbenennen |
| `DELETE /api/collections/<id>` | löschen |
| `POST`/`DELETE /api/collections/<id>/cards[/<card_id>]` | Karte einzeln zu/aus Sammlung (UI) |

## Empfehlung fürs converter-mcp (jetzt)

1. **`list_tags` wrappen/anreichern** → `parent_id` + `card_count` durchreichen (Agent liest den Themen-Wald, ordnet konsistent ein).
2. **`list_collections` wrappen** (Read).
3. **`collections`-Param** an `create_card`/`update_card` durchreichen (Sammlungs-Write).
4. **`set_tag_parent`** neu wrappen (`POST /api/tags/parent`, Tag-Baum-Write).

## ☑ Entschieden + gebaut (LERN-GROUP-AW, 2026-06-27)

Die frühere „offene Entscheidung" ist **erledigt**: Oli hat **volle Agent-Autonomie** beim Gruppieren gewählt. Beide Schreibpfade sind gebaut + getestet:

- **Agent ordnet Karten in Sammlungen** → ✅ über das `collections`-Feld an `create_card`/`update_card` (frei anlegbar; Aufräumen = User-UI).
- **Agent baut die Taxonomie-Hierarchie** → ✅ über `POST /api/tags/parent` (by-name, Zyklus-Guard).
- **Agent fragt Karten nach Gruppe ab** (nicht nur *fällige*): **weiterhin offen, optional**. `GET /api/cards` filtert nur nach `state`/`highlight_id`; ein `?tag=&collection=`-Filter (Teilbaum/AND wie review-state) ist ein kleiner CONVERTER-Add, falls real gebraucht. Bitte beim CONVERTER-Master anfordern.

---

*CONVERTER-Seite: LERN-GROUP + LERN-GROUP-AW fertig + deployed. Reads **und** die Gruppierungs-Writes sind MCP-wrappbar; nur das Kuratieren (Löschen/Umbenennen) bleibt User-UI. Geschwister-Brief: [docs/converter_mcp_list_conversions_brief.md](docs/converter_mcp_list_conversions_brief.md), Kontrakt-Muster: [docs/card_api_contract.md](docs/card_api_contract.md).*
