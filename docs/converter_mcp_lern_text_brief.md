# Developer-Brief an das converter-mcp-Team — LERN-TEXT (Lerntexte zu Karten und Sammlungen)

> **An**: converter-mcp-Entwickler (Koordinator-Repo).
> **Von**: CONVERTER (Sprint LERN-TEXT, Sub-Thread; Master-Abnahme), 2026-10-05.
> **Worum**: CONVERTER kann jetzt **Lerntexte** mit Karten und Sammlungen **verknüpfen**. Ein Lerntext ist ein gewöhnliches Library-Dokument (kein neuer Typ, kein neues Tool dafür); neu sind **zwei Verknüpfungen** — **Karte → genau eine Textstelle** (`context` an den Karten-Writes) und **Sammlung → Texte** (ein neuer `PUT`). Dieser Brief sagt, was am converter-mcp dazukommt und was in den Tool-Docs stehen muss, damit der Agent die Verknüpfung richtig setzt.

**Tool-Liste am lebenden Connector gelesen (2026-10-05):** `create_card`, `update_card`, `get_card`, `list_cards`, `list_collections`, `review_state`, `create_conversion`, `update_document`, `replace_section`, `list_conversions`, `get_transcript`, `list_highlights`, `list_recent_highlights`, `update_highlight`, `list_tags`, `set_tag_parent`, `merge_tags`, `delete_tag`, `create_narration`, `get_narration_status`, `list_audio_transcripts`. **Kein Tool schreibt an Sammlungen** (sie entstehen by-name über `create_card`/`update_card`), **kein Tool kennt `context`**, `list_collections` liefert `{items}` mit `id, name, description, card_count, due_count, created_at` — ohne `documents`.

## TL;DR (bitte zuerst lesen)

- **`create_card` / `update_card` bekommen ein Feld `context`** (optional): `{"document_id": <int>, "heading": "<Text>" | null}` oder `null` zum Lösen. Dieselben Routen, derselbe `CARD_TOKEN`, dieselbe CSRF-Freistellung wie heute. Fail-closed mit drei Fehlern, die der Agent **sofort** sieht: **404** fremdes/fehlendes Dokument, **400** Überschrift nicht im Text, **409** Überschrift mehrdeutig (Sätze unten).
- **Neues Tool `set_collection_documents(collection_id, document_ids)`** → `PUT /api/collections/<id>/documents` `{"documents": [<id>, …]}`. Semantik **„ersetzt"**: die Liste ist danach genau die gesendete (Reihenfolge = Leseposition), `[]` löst alles. **Auth: `@login_required`**, also der **per-User-Bearer des Connectors** (die `api_token`-Zeile `converter-mcp`, mit der schon `list_collections` liest) — **nicht** `CARD_TOKEN`. Mit Bearer-Header prüft CONVERTER **kein CSRF** (MOBILE-AUTH-Inversion); mit einer Cookie-Session bräuchte der Aufruf ein `X-CSRFToken` von `GET /api/csrf-token`.
- **`list_collections` trägt je Eintrag additiv `documents: [{id, title, url}]`** in Positionsreihenfolge (leer = `[]`). Bitte durchreichen.
- **`get_card`, `review_state` (jede Karte in `due_cards`) und die Antworten von `create_card`/`update_card` tragen additiv `context`**: `{document_id, document_title, heading, url} | null`. **`list_cards` bleibt schlank** — kein `context` in der Listenform (wie bei den Figuren): wer es braucht, ruft `get_card`.
- **`url` ist relativ** (`/library/<id>` bzw. `/library/<id>#h=<encodeURIComponent(Überschrift)>`). Für einen Menschen `https://converter.smallpieces.de` davorsetzen. Der Sprung geht im Reader über den **Überschriften-Text**, nicht über `id`-Anker — deshalb muss die Überschrift so im Dokument stehen, wie sie gerendert wird (s. *Was der Agent prüfen sollte*).

## Karte → Textstelle: `context`

### Form

```json
{"type": "atomic", "front": "…", "back": "…",
 "context": {"document_id": 262, "heading": "Säuren und Basen"}}
```

| `context` | Bedeutung |
|---|---|
| fehlt (`create_card`) / `None` (`update_card`-Semantik „nicht angefasst") | keine Änderung |
| `null` (explizit) | Stelle lösen — beide Spalten `NULL` |
| `{"document_id": 262}` oder `heading: null` oder `heading: ""` | **der Text als Ganzes** (für kurze Texte): `url` ohne `#h=` |
| `{"document_id": 262, "heading": "Säuren und Basen"}` | genau diese Überschrift: `url = /library/262#h=S%C3%A4uren%20und%20Basen` |

Das gespeicherte `heading` ist die **kanonische Form**: führende `#` und Rand-Whitespace gestrippt (`"## Redox"` → `"Redox"`), level-agnostisch — **dieselbe Erkennung und Vergleichsregel wie `replace_section`** (ATX-Überschriften, fenced Code wird übersprungen, kein Setext). Eine Überschrift, die `replace_section` als Ziel annimmt, ist genau eine, die hier zählt.

### Fehler (Body `{"error": "<Satz>"}`, nichts geschrieben)

| Status | Satz | Wann |
|---|---|---|
| 400 | `Feld 'context' muss ein Objekt oder null sein.` / `Feld 'context.document_id' muss eine Zahl sein.` / `Feld 'context.heading' muss Text oder null sein.` | Form |
| 404 | `Dokument nicht gefunden.` | `document_id` gehört nicht dem Ziel-User oder existiert nicht (nie 403 — keine Existenz-Lecks) |
| 400 | `Überschrift nicht gefunden: ‚<Text>‘.` | die Überschrift kommt im Dokument nicht vor (auch: nur in einem Code-Fence) |
| 409 | `Überschrift kommt N-mal vor.` | mehrdeutig — der Agent wählt eine eindeutige oder schreibt den Text um |

Ein `update_card` mit gültigen Feldern **und** einem abgelehnten `context` schreibt **nichts** (die Karte bleibt, wie sie war). Bitte die Sätze unverändert an den Agenten geben — sie sind so geschrieben, dass er daraus handeln kann.

### Was CONVERTER daraus macht

- **Review**: nach dem Aufdecken ein Link **„Lerntext"** (neuer Tab, Titel „‹Dokument› › ‹Überschrift›"); vor dem Aufdecken nicht im DOM (Spickzettel-Schutz).
- **Done-Panel „Nacharbeiten"**: die Textstellen der Karten, die in der Session **Nochmal/Schwer** bekamen oder per **Vertiefen** „wackelt" wurden — gruppiert nach Dokument, je Überschrift ein Link. Rein Session-Zustand.
- **Reader**: `/library/<id>#h=…` springt zur Überschrift und hebt sie 2 s hervor. Steht sie nicht mehr im Text (umbenannt), zeigt der Reader den Text von oben mit dem Hinweis *„Abschnitt nicht gefunden. Der Text wurde seit der Verknüpfung geändert."* — **die Karte verliert ihre Stelle nicht**, nur der Sprung geht ins Leere, bis jemand `context` neu setzt.
- **Löschen**: wird das Dokument gelöscht, verlieren alle Karten ihre Stelle (`context: null`) und die Sammlungen den Eintrag — im selben Commit. Die Karten bleiben.

## Sammlung → Texte: `set_collection_documents`

```
PUT /api/collections/<id>/documents
{"documents": [262, 263]}
→ 200  {"id": 5, "name": "Chemie-Basics", "description": null, "created_at": "…",
        "documents": [{"id": 262, "title": "Säuren und Basen — Einführung", "url": "/library/262"},
                      {"id": 263, "title": "Redox — Einführung", "url": "/library/263"}]}
```

| Status | Satz | Wann |
|---|---|---|
| 400 | `Feld 'documents' muss eine Liste von Zahlen sein.` | Form (auch `true`, `1.5`, Strings) |
| 404 | `Nicht gefunden.` | die Sammlung gehört nicht dem User / existiert nicht |
| 404 | `Dokument nicht gefunden.` | **ein** fremdes oder fehlendes Dokument in der Liste → **nichts** geschrieben, die alte Liste bleibt |

Wiederholte ids behalten ihre erste Position. Die Sammlungs-id kommt aus `list_collections`; das Tool sollte sie **nicht** by-name raten (zwei Sammlungen können sich nur in Groß-/Kleinschreibung unterscheiden).

**Vorschlag für die Signatur:** `set_collection_documents(collection_id: int, document_ids: list[int]) -> {id, name, documents}`. Docstring-Kern: *„Ersetzt die Lerntext-Liste der Sammlung (Reihenfolge = Leseposition, `[]` löst alles). Nur eigene Dokumente, sonst 404 und nichts geschrieben. Lesen über `list_collections().items[].documents`."*

## Wie ein Lerntext entsteht (Agent-Ablauf)

1. **Text schreiben**: `create_conversion(content=<Markdown>, title=…, conversion_type=…, source_id=…)` → die Antwort trägt die **id** (`list_conversions` findet sie später). Überschriften als ATX (`## Säuren und Basen`), **eindeutig** im Dokument und **ohne Inline-Auszeichnung** (s. u.).
2. **An die Sammlung hängen**: `list_collections` → id der Sammlung → `set_collection_documents(id, [doc_id, …])` in Lesereihenfolge. Im Lern-Launcher erscheint die Zeile „Lesen: ‹Titel› · ‹Titel›", sobald die Sammlung angehakt ist.
3. **Karten verknüpfen**: `update_card(card_id, context={"document_id": doc_id, "heading": "…"})` je Karte — oder gleich beim `create_card`. Die Antwort trägt `context` mit `document_title` und `url` = der **Rücklese-Beweis**; `get_card` liefert dasselbe.

## Was der Agent prüfen sollte (gehört in die Tool-Docs)

- **Überschrift exakt wie im Text**, Groß-/Kleinschreibung inklusive; `#` darf mit (wird gestrippt). Ein Tippfehler ist ein **400** mit dem Namen — kein stiller Fehlanker.
- **Eindeutig**: zweimal „Einleitung" (auch auf verschiedenen Ebenen) ist ein **409**. Der Text braucht dann sprechendere Überschriften.
- **Ohne Inline-Auszeichnung in der Überschrift** (`## **Säuren** und Basen` vermeiden): gespeichert wird der Markdown-Quelltext der Überschrift, der Reader vergleicht mit dem **gerenderten** Text — eine Überschrift mit `**…**`, Backticks oder Links wird gespeichert, aber der Sprung findet sie nicht (Hinweis „Abschnitt nicht gefunden"). Reine Textüberschriften sind der sichere Weg.
- **Nach einem `replace_section`/`update_document`, das Überschriften umbenennt**, die `context`-Werte der betroffenen Karten neu setzen — CONVERTER prüft die Überschrift nur beim Schreiben der Karte, nicht beim Ändern des Texts.
- **Eine Stelle je Karte.** Wer zwei Stellen braucht, wählt die erklärendste; die zweite kann in `note` stehen.

## Nicht gebaut (bewusst)

Kein Lerntext-Tool (Texte sind `create_conversion`), kein Highlight-Schreib-Tool, kein LLM-Pfad in CONVERTER, keine `id`-Anker im gerenderten HTML, keine Nacharbeiten-Seite über Sessions hinweg (benannt: LERN-NACHARBEITEN-SEITE), iOS zeigt Knopf und Zeile noch nicht (Felder liegen an; LERN-TEXT-IOS). Sammlungen anlegen/umbenennen/löschen bleibt User-UI.

## End-to-end-Beweis = Koordinator-Scope

Auf **Wegwerf-Elementen** (eigene `source_id`, eine Wegwerf-Karte; danach löscht Oli in der UI — der Agent löscht nicht):

1. `create_conversion` mit zwei Überschriften → id. `set_collection_documents(<Sammlung>, [id])` → `list_collections` zeigt den Eintrag unter `documents`.
2. `create_card(…, context={"document_id": id, "heading": "<zweite Überschrift>"})` → **201**, `context.url` endet auf `#h=…`; `get_card` liefert dasselbe.
3. `update_card(card, context={"document_id": id, "heading": "Gibt es nicht"})` → **400** mit dem Satz; `get_card` zeigt den alten `context` unverändert.
4. `update_card(card, context=None)` → `context: null`.
5. `set_collection_documents(<Sammlung>, [])` → `documents: []`.

Rückkanal: `docs/converter_mcp_lern_text_rueckmeldung.md` neben diesem Brief.

---

*CONVERTER-Seite: LERN-TEXT fertig, getestet (+58 Tests, 1639 + 1 Skip, Mac und Container-Pin), Browser-Smoke A+B in der Wegwerf-Instanz und gegen Prod gefahren ([scripts/smoke_lern_text.py](../scripts/smoke_lern_text.py)), deployt 2026-10-05 (Image `f9cd92dd81ac`). **Zwei Spalten an `card`, eine Junction `collection_documents`, ein neuer `PUT`, additive Felder — kein neuer Token, kein neuer Dep, kein neuer Dokument-Typ.** Geschwister-Brief: [docs/converter_mcp_lern_group_brief.md](converter_mcp_lern_group_brief.md) (Sammlungen by-name an den Karten-Writes).*
