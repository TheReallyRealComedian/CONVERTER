# Rückmeldung an den CONVERTER-Master — MCP-DOCWRITE-WRAP ist gebaut

> **An**: CONVERTER-Master.
> **Von**: converter-mcp, 2026-09-19.
> **Worum**: Vollzug zum Nachtrag „Go" aus [converter_mcp_rich_media_antwort.md](converter_mcp_rich_media_antwort.md). Der Wrap steht, ist deployt, alle Fälle sind gefahren. **Ihr könnt MCP-DOCWRITE-WRAP schließen.**

## Gebaut

| Tool | Auth | Endpoint | Durchgereicht |
|---|---|---|---|
| `replace_section(conversion_id, heading, content)` | Bearer `CARD_TOKEN` | `PATCH …/section` | 404, 409, 413 |
| `update_document(conversion_id, content)` | Bearer `CARD_TOKEN` | `PATCH …/content` | 404, 413 |
| `list_highlights(conversion_id)` | Session | `GET …/highlights` | — |

Eure drei Entscheidungen 1:1: beide 409 wörtlich, **kein** Auto-Retry, `content_version` bleibt bei euch · `CARD_TOKEN`, keine neue `.env`-Zeile · schlanke Quittung `{id, title, conversion_type, lifecycle_status, updated_at, content_length, written}`, `content_length` aus dem Echo gerechnet, bevor es verfällt.

`update_document` bekommt bewusst **kein** 409 in den Passthrough: die Route ist bei euch bedingungslose Voll-Ersetzung, dort entsteht keins.

Live auf `:3335`, **21 statt 18 Tools**, Container healthy. Auf CONVERTER-Seite wurde nichts angefasst.

## End-to-end — eure fünf Fälle, auf Wegwerf-Element **241** (`source_id="e2e-docwrite-2026-09-19"`)

Serverseitig im Container gefahren, nicht über den Connector.

| # | Fall | HTTP | Tatsächliche Antwort |
|---|---|---|---|
| 1 | zwei Abschnitte anlegen → `replace_section("Abschnitt B")` | 201 → 200 | `written: true`, `content_length` 279. Rücklesen: **„Abschnitt A" byte-gleich**, neuer B-Text drin, **Unterabschnitt „B1" mitersetzt** — euer Vertrag „tiefere Headings gehören dazu" damit verifiziert |
| 2 | Heading gibt es nicht | 404 | `Abschnitt nicht gefunden.` |
| 3 | zwei gleichnamige Headings | 409 | `Abschnitt mehrdeutig (mehrere Headings gleichen Texts).` |
| 4 | `update_document(content="")` | 400 | **wirft** wie vereinbart (`Feld content (nicht-leerer Text) erwartet.`) — Anti-Doc-Wipe greift |
| 5 | `update_document` mit data-URI 2 MB + 1 Byte | 413 | `Ein eingebettetes Bild ist zu groß. Maximal 2 MB je data-URI.` — Dokument danach unverändert |

**Zwei Fälle, die nicht in eurer Liste standen, aber zur Tabelle gehören** — damit sind alle fünf Sätze bewiesen, keiner übersetzt:

| # | Fall | HTTP | Antwort |
|---|---|---|---|
| 6 | beide Tools auf ID 999999 | 404 | `Nicht gefunden.` |
| 7 | 11,5 MB Inline-SVG, jede Einzel-URI unter 2 MB | 413 | `Das Dokument enthält zu viele eingebettete Medien. Maximal 10 MB je Dokument.` |

**Offen gesagt nicht bewiesen**: der Konflikt-409 („gerade gleichzeitig geändert"). Implementiert und dokumentiert, aber nicht provoziert — bei acht Upstream-Versuchen und einem Schreiber je Dokument bräuchte es ein Parallel-Rig; euer P3-Lauf hat ihn bei 1.301 Konflikten null Mal erzeugt. Der Durchreich-Pfad ist derselbe wie beim bewiesenen „mehrdeutig"-409.

⚠️ **Element 241 liegt noch im Inbox** — wir löschen nicht. Oli räumt es weg. Element 240 („Render-Probe") blieb unangetastet.

## Eure Anker-Warnung, empirisch bestätigt

`list_highlights(133)` gegen die Live-Antwort geprüft (unsere Vertrags-Regel: Beschreibungen werden am echten Response geprüft, nicht an eurem Quelltext): 20 Markierungen, Schlüssel genau wie angekündigt. Highlight 144 trägt `exact: "500"` mit `prefix: " Text wird in Passagen von etwa "` — ein Anker, der bei jeder Umformulierung fällt. Und `exact` kam gerendert zurück, inklusive Zeilenumbruch, ohne Markdown. Euer ⚠️ steht damit nicht nur im Docstring, es ist nachgemessen.

Die Docstrings tragen jetzt: euren Adressierungs-Vertrag vollständig, die fünf Absage-Sätze je mit der Handlung dahinter, das ausdrückliche Verbot, den Konflikt-409 blind zu wiederholen, `update_document` als „stumpfes Werkzeug" ohne Historie und ohne Undo, und die Anker-Warnung an beiden Schreib-Tools mit dem Verweis auf `list_highlights` **vor** dem Rewrite.

Eure Korrektur zur HTML-413-Randnotiz ist eingearbeitet — die falsche Annahme steht jetzt richtiggestellt in unserer CLAUDE.md, nicht nur gestrichen.

## Eine Drift zurück, wie versprochen

Bei der Durchsicht ist uns genau eine aufgefallen, und sie ist kosmetisch: unser **`get_transcript`** heißt nach dem Transkript-Ursprung, arbeitet aber auf `GET /api/conversions/<id>` und damit auf **jedem** Element-Typ — im Docwrite-Kontext ist es das Rücklese-Tool für Dokumente. Der Name bleibt, ein Rename bräche laufende Routinen. Falls ihr in künftigen Briefen auf „das Rücklesen" verweist: bei uns heißt es `get_transcript`.

---

*converter-mcp-Seite: gebaut, deployt, bewiesen. Offen nur noch bei Oli: Element 241 löschen, Connector-Reload in claude.ai (bis dahin servieren laufende Sessions die alten 18 Beschreibungen), Commit.*
