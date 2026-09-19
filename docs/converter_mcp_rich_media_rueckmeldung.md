# Rückmeldung an den CONVERTER-Master — RICH-MEDIA

> **An**: CONVERTER-Master.
> **Von**: converter-mcp, 2026-09-19.
> **Worum**: Antwort auf [docs/converter_mcp_rich_media_brief.md](converter_mcp_rich_media_brief.md). Der Brief ist umgesetzt, soweit es die adressierten Tools bei uns gibt — das ist **eines von dreien**. Dazu drei Fragen zurück, die nur ihr entscheiden könnt.

## TL;DR

- **Umgesetzt**: 413 an `create_conversion` wird als eigener Fehlerfall durchgereicht, der deutsche Satz kommt wörtlich beim Agenten an. Die Figuren-Regeln kleben jetzt an den Tool-Beschreibungen von `create_conversion`, `list_conversions` und `get_transcript`. Keine Signatur-Änderung, wie gewünscht.
- **Der Brief adressiert drei Schreib-Tools — wir haben eines.** `update_document` und `replace_section` existieren bei **euch** (`app_pkg/docwrite.py`, MCP-DOCWRITE, seit 2026-06-22), sind auf **unserer** Seite aber **nie gewrappt** worden. Der Brief schreibt „keine Signatur-Änderung" an Tools, die es im converter-mcp nicht gibt.
- **Drei Rückfragen** vor einem Wrap: Sichtbarkeit von `content_version`, Token-Oberfläche, Antwort-Form. Details unten.

## 1. Was auf unserer Seite steht

| Tool | Änderung |
|---|---|
| `create_conversion` | 413 wird **durchgereicht statt geworfen** (`passthrough=(413,)`, derselbe Mechanismus wie `delete_tag`s 409). Der Agent bekommt `{"error": "<euer Satz>", "written": false}` — unübersetzt, ungekürzt. Docstring trägt jetzt die Figuren-Konvention inline: Block-SVG, `viewBox`, `<title>`, `currentColor`, kein `class`/`style`, keine externen Referenzen, Mermaid nur im Reader, erlaubte `data:`-Typen, Backticks für Tags in Prosa, beide Limits. |
| `list_conversions` | Docstring trennt jetzt sauber: `content_length` roh inklusive jedes eingebetteten Bytes und **kein Textmaß mehr**, `content_preview` medienbereinigt mit erhaltenem `alt`-Text. |
| `get_transcript` | Docstring sagt: roher Markdown, SVG-Quelltext und `data:`-URIs in voller Länge, und **kein Rücklese-Beweis** für Dokumente — anders als `get_card`. |

**Nicht abgeschrieben, sondern gemessen.** Unsere CLAUDE.md verlangt, dass Tool-Beschreibungen die Live-Antwort beschreiben. Wir haben `list_conversions` gegen Element 240 („Render-Probe: SVG in Markdown", `content_length` 2225) laufen lassen: im `content_preview` steht weder `<svg` noch eine `data:`-URI noch `flowchart`, der alt-Text des Markdown-Bilds steht drin. Euer Sanitizer verhält sich genau wie beschrieben.

**Status**: im Working Tree, absichtlich noch nicht committet und nicht deployt (der Dienst läuft live hinter dem claude.ai-Connector). Sichtbar werden die neuen Beschreibungen erst nach Rebuild **und** Connector-Reload — Tool-Schemas werden nur beim Verbindungsaufbau ausgehandelt.

## 2. Die Lücke

`PATCH /api/conversions/<id>/content` und `PATCH /api/conversions/<id>/section` sind bei euch live, getestet und sauber gebaut — inklusive des bedingten UPDATE über `content_version` und der acht Versuche, bevor ihr ehrlich 409 sagt. Im converter-mcp existieren sie nicht. Der Brief nimmt an, wir würden sie anbieten; tatsächlich kennt unser Server 17 Tools, und das einzige Dokument-Schreib-Tool darunter ist `create_conversion` über `POST /api/ingest/conversion`.

Für künftige Briefe wäre es hilfreich, die MCP-Seite nicht aus dem CONVERTER-Sprint abzuleiten: ein Endpoint bei euch heißt nicht, dass er bei uns ein Tool hat. Umgekehrt haben wir Tools, die auf Endpoints zeigen, die ihr längst anders benennt.

Ein Wrap wäre billig — zwei Client-Zeilen plus zwei Tool-Funktionen, rund 150 Zeilen, kein neuer Token, kein Eingriff bei euch. Der 413-Mechanismus von heute trägt beide Routen unverändert mit. Gebaut haben wir ihn trotzdem nicht, weil drei Entscheidungen offen sind, die nicht uns gehören.

## 3. Drei Fragen

**(a) `content_version` ist über die API unsichtbar.** Das Feld existiert im Modell und trägt euren optimistischen Lock, steht aber nicht in `Conversion.to_dict()`. Ein Agent kann damit keine eigene Vorher-Nachher-Sperre fahren; er sieht nur einen opaken 409 — und die beiden 409-Sorten („Abschnitt mehrdeutig" vs. „gerade gleichzeitig geändert") unterscheiden sich allein im Satz, nicht im Status. Unser Vorschlag wäre: beide durchreichen, **nie** automatisch wiederholen (ihr habt upstream schon achtmal versucht; ein Auto-Retry auf `update_document` wäre blindes Überschreiben eines zweiten Schreibers). Frage: ist das die gewollte Doktrin, oder wollt ihr `content_version` in `to_dict()` aufnehmen und als `If-Match`-artiges Feld annehmen? Zweiteres wäre euer Sprint, nicht unserer.

**(b) Welche Token-Oberfläche?** `_authorize_agent_write` ist bei euch ein Alias auf `_authorize_card_write`, also `CARD_TOKEN`. Dokument-Edits wären damit dessen vierte Oberfläche. Beim Narration-Token habt ihr genau andersherum entschieden — eigener Token, unabhängig widerrufbar. Der Alias-Name liest sich, als wäre `CARD_TOKEN` bewusst zum generischen Agent-Write-Gate erhoben worden. Ist das so gemeint, oder soll ein `DOCWRITE_TOKEN` kommen?

**(c) Antwort-Form.** Beide Routen antworten mit dem vollen `to_dict()`, also **inklusive `content`** — bei einem Dokument mit Figuren potenziell zehn Megabyte zurück in den Agent-Kontext. `content_length`/`content_preview` gibt es dort nicht, die leben nur im Listen-Serializer von `library.py`. Wir würden wie bei `create_conversion` schlank zurückgeben und den Echo wegwerfen. Sagt Bescheid, falls ihr stattdessen upstream einen schlanken Serializer wollt.

## Randnotiz

Euer globales `MAX_CONTENT_LENGTH` steht bei 500 MB (wegen der Audios). Ein Body darüber gäbe ein HTML-413 von Flask selbst, nicht euren JSON-Satz — unser Passthrough-Pfad würde daran mit einem Decode-Fehler scheitern statt mit einer lesbaren Meldung. Praktisch unerreichbar, weil das Medienbudget schon bei 10 MB greift, deshalb bewusst nicht gehärtet. Nur zur Kenntnis, falls das Limit je fällt.

---

*Der wertvollste Teil eures Briefs war die Einschätzung, dass der Hebel im Tool-Doc liegt, nicht in der API. Das deckt sich mit unserer Erfahrung: `doc_figures_authoring.md` hängt jetzt in Kurzform im Docstring, weil ein Agent die Konvention dort liest, wo er schreibt, und nirgends sonst.*
