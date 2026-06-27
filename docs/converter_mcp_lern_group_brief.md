# Developer-Brief an das converter-mcp-Team — LERN-GROUP (Karten-Gruppierung)

> **An**: converter-mcp-Entwickler (Koordinator-Repo).
> **Von**: CONVERTER-Master, 2026-06-27.
> **Worum**: CONVERTER kann jetzt **Lernkarten gruppieren** (Sprint LERN-GROUP) über zwei Achsen — **Taxonomie-Baum** (hierarchische Tags) + **Sammlungen** (kuratierte Bündel), beide als Filter auf die Wiederhol-Queue. Dieser Brief sagt, **was der converter-mcp davon weiterreichen kann — und was (noch) nicht**.

## TL;DR (bitte zuerst lesen — Auth-Falle)

Alle LERN-GROUP-Endpoints sind **`@login_required` (Session) + unter dem globalen CSRF-Schutz** — bewusst **User-Seite**, **nicht** token-authed wie die Card-/Highlight-/Doc-Writes.

- **READS** (`GET`): der converter-mcp wrappt sie **sofort** mit seiner **bestehenden Session** (wie `list_audio_transcripts`/`list_cards`). **Keine CONVERTER-Änderung nötig.**
- **WRITES** (`PATCH`/`POST`/`DELETE`): **CSRF-geschützt** → der Bearer-`CARD_TOKEN`-Pfad des MCP erreicht sie **nicht**, und ein Session-POST ohne CSRF-Token gibt **400**. **Der Agent kann die Gruppierung lesen, aber mit den heutigen Endpoints nicht schreiben.** Agent-Schreiben bräuchte token-authed, CSRF-exempt Varianten in CONVERTER (das Card-Write-Muster) → **CONVERTER-Folge-Sprint, kein MCP-only-Thema** (s.u.).

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

## WRITES — dokumentiert, aber **NICHT** über den MCP-Token erreichbar

Diese Endpoints existieren, sind aber **Session+CSRF** (User-UI). Der MCP kann sie **nicht** über den Bearer-Token aufrufen.

| Endpoint | Zweck |
|---|---|
| `PATCH /api/tags/<id>` `{parent_id}` | Tag in den Baum einordnen (Hierarchie bauen) |
| `POST /api/collections` `{name, description?}` | Sammlung anlegen |
| `PATCH /api/collections/<id>` `{name?, description?}` | umbenennen |
| `DELETE /api/collections/<id>` | löschen |
| `POST /api/collections/<id>/cards` `{card_id}` | Karte zu Sammlung |
| `DELETE /api/collections/<id>/cards/<card_id>` | Karte aus Sammlung |

**Warum nicht einfach token-exposen?** LERN-GROUP hat die Gruppierung bewusst als **User-Kuratierung** gebaut: der User arrangiert den Tag-Baum + schnürt Sammlungen in der UI. Der Agent **partizipiert an Achse A schon vollständig** ohne neue Endpoints — er setzt Leaf-Tags via `create_card`, die in den vom User gebauten Baum einsortiert werden.

## Empfehlung fürs converter-mcp (jetzt)

1. **`list_tags` wrappen/anreichern** → `parent_id` + `card_count` durchreichen (Agent liest den Themen-Wald, taggt konsistent).
2. **`list_collections` wrappen** (Read).
3. **Sonst nichts** — der Agent gruppiert Achse A bereits über die `create_card`-`tags`.

## Offene Entscheidung (Koordinator ↔ CONVERTER-Master)

Falls der Agent **schreiben** soll, sind das **CONVERTER-Folge-Sprints** (nicht MCP-only):

- **Agent ordnet Karten in Sammlungen** (z.B. neue Karten aus einer Quelle automatisch in den passenden Horizont) → CONVERTER braucht einen **token-authed, CSRF-exempt** `POST /api/collections/<id>/cards`-Pfad (Muster wie `update_highlight`/`update_document`). Bitte beim CONVERTER-Master anfordern, wenn real gebraucht.
- **Agent baut die Taxonomie-Hierarchie** → analog token-`PATCH /api/tags/<id>`. (Vermutlich **nicht** nötig — Hierarchie ist User-Kuratierung; der Agent liest sie nur.)
- **Agent fragt Karten nach Gruppe ab** (nicht nur *fällige*): `GET /api/cards` filtert heute nur nach `state`/`highlight_id` — **kein** `tag`/`collection`-Filter (der liegt nur auf `/api/review-state`, due-only). Wenn der Agent „alle Karten in Thema X / Sammlung Y" braucht, ist das ein kleiner CONVERTER-Add (`/api/cards?tag=&collection=`, Teilbaum/AND wie review-state).

---

*CONVERTER-Seite: LERN-GROUP fertig + deployed. Reads sind sofort MCP-wrappbar; jeder Agent-Write-Bedarf ist ein separater, token-authed CONVERTER-Endpoint — sagt dem CONVERTER-Master, was ihr braucht, dann liefern wir den passenden Schreibpfad nach. Geschwister-Brief: [docs/converter_mcp_list_conversions_brief.md](docs/converter_mcp_list_conversions_brief.md), Kontrakt-Muster: [docs/card_api_contract.md](docs/card_api_contract.md).*
