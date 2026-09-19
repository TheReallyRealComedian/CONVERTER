# Developer-Brief an das converter-mcp-Team — RICH-MEDIA (Figuren im Dokument)

> ⚠️ **Korrektur 2026-09-19 (CONVERTER-Master)**: Dieser Brief adressiert drei Schreib-Tools. Im converter-mcp existiert davon **eines** (`create_conversion`); `update_document` und `replace_section` sind bei uns Endpoints (MCP-DOCWRITE, 2026-06-22), wurden auf MCP-Seite aber **nie gewrappt** — für MCP-DOCWRITE ging nie ein Brief hinaus. Alle Aussagen zu diesen zwei Tools beschreiben die **Endpoints**, nicht vorhandene Tools. Verlauf: [Rückmeldung](converter_mcp_rich_media_rueckmeldung.md) → [Antwort](converter_mcp_rich_media_antwort.md).

> **An**: converter-mcp-Entwickler (Koordinator-Repo).
> **Von**: CONVERTER-Master, 2026-09-19.
> **Worum**: Dokumente in der Library können jetzt **Figuren** tragen — Inline-SVG, `data:`-Bilder, https-Bilder, Mermaid. Das ist reine Render- und Validierungs-Arbeit auf CONVERTER-Seite: **kein neues Feld, kein neuer Endpoint, kein neuer Token**. Dieser Brief sagt, was der converter-mcp in seinen Tool-Docs anpasst und welche neue Antwort (413) seine Schreib-Tools jetzt sehen können.

## TL;DR (bitte zuerst lesen)

- **Keine API-Änderung.** `create_conversion` (`POST /api/ingest/conversion`), `update_document` (`PATCH /api/conversions/<id>/content`) und `replace_section` (`PATCH …/section`) nehmen wie bisher Markdown in `content`. Figuren **sind** Markdown: `<svg>…</svg>` als Block, `![alt](data:image/…)`, `![alt](https://…)`, ```` ```mermaid ````.
- **Eine neue Antwort: 413.** Medien sind budgetiert — **2 MB je `data:`-URI, 10 MB Medien je Dokument** (`data:`-URIs + Inline-SVG; Text hat kein Limit). Darüber: **413** mit deutschem Satz, **nichts geschrieben**. Bitte als eigenen Fehlerfall durchreichen, nicht als generischen Fehler.
- **`content_length` ist roh** (inklusive jedes eingebetteten Bytes) — mit Figuren kein Maß mehr für „wie viel Text". **`content_preview` ist medienbereinigt**: SVG-Quelltext, `data:`-URIs und Mermaid-Quelltext sind raus, der `alt`-Text eines Markdown-Bilds bleibt. `list_conversions` zeigt also Prosa, auch wenn ein Dokument mit einer Figur beginnt.
- **`get_transcript` liefert den rohen Inhalt**, nicht das Rendering — der Agent sieht dort sein SVG so, wie er es geschrieben hat, nicht, was der Sanitizer davon übrig lässt. Anders als bei Karten (`get_card` = bereinigt) gibt es für Dokumente **keinen** Rücklese-Beweis über die API.
- **Der eigentliche Wert liegt im Tool-Doc**: [docs/doc_figures_authoring.md](doc_figures_authoring.md) ist die Autoren-Konvention. Ohne sie entstehen Figuren, die zu einem Platzhalter werden, und Texte *über* SVG, die als kaputte Figur gelesen werden.

## Die neue Antwort — 413 an den Schreib-Tools

| Tool | Route | 413, wenn |
|---|---|---|
| `create_conversion` | `POST /api/ingest/conversion` | `content` das Budget reißt — **vor** dem Dedup über `source_id`, es entsteht kein Element |
| `update_document` | `PATCH /api/conversions/<id>/content` | der neue `content` das Budget reißt — das Dokument bleibt, wie es war, `content_version` unverändert |
| `replace_section` | `PATCH /api/conversions/<id>/section` | das **fertige, gespleißte Dokument** das Budget reißt — eine kleine Sektion kann ein Dokument über 10 MB kippen |

Body: `{"error": "<Satz>"}` mit genau einem dieser Sätze:

> Ein eingebettetes Bild ist zu groß. Maximal 2 MB je data-URI.
>
> Das Dokument enthält zu viele eingebettete Medien. Maximal 10 MB je Dokument.

Gemessen wird in **Bytes des Texts, wie er geschrieben ist** (utf-8): ein base64-Payload zählt aufgebläht — genau das, was die Spalte und der Reader tragen. Eine `data:`-URI, die *in* einem Inline-SVG steht, wird nicht doppelt gezählt. Alle anderen Antworten (400/401/404/409/503) sind unverändert.

## Was aus einer Figur wird (damit der Wrapper nichts Falsches verspricht)

**Roh speichern, beim Rendern bereinigen** — wie bei den Karten. Die Spalte hält exakt, was der Agent geschickt hat; **ein** Renderer bereinigt für Reader, PDF **und** EPUB.

- **Inline-SVG** läuft durch **dieselbe** Allow-List wie Karten-Figuren ([docs/card_svg_authoring.md](card_svg_authoring.md) § *Erlaubte Tags und Attribute*): Verbotenes fällt still, der Rest rendert. `class` und `style` gibt es auf SVG-Elementen nicht.
- **Bleibt nichts Zeichenbares übrig**, steht im Reader ein sichtbarer Platzhalter *[Abbildung nicht darstellbar: …]* mit dem Grund. **Der Write wird nicht abgelehnt** (anders als `front_svg`/`back_svg` an den Karten, die mit 400 antworten) — ein Dokument ist mehr als seine Figuren. Ein Figur-Fehler kostet die Figur, nie den Text um sie herum.
- **`data:`** trägt nur auf Bildern und nur für `image/svg+xml`, `png`, `jpeg`, `webp`, `gif`. Ein **Link** auf eine `data:`-URI verliert sein Ziel.
- **https-Bilder** laden lazy und ohne Referrer; der Agent kann das nicht überschreiben.
- **Mermaid** rendert nur im Reader (Browser). **Im PDF und im EPUB/Kindle bleibt ein Mermaid-Block Quelltext** — was dort ein Bild sein muss, als Inline-SVG zeichnen.
- **Relative Bildpfade** (`![](ordner/abb.png)`) zeigen ins Leere — es gibt noch keinen Asset-Store (CONVERTER-Backlog RICH-MEDIA-ASSETS; käme mit einem eigenen Brief).

## Empfehlung fürs converter-mcp

1. **Keine Signatur-Änderung** an `create_conversion`, `update_document`, `replace_section`.
2. **413 durchreichen**: den deutschen Satz aus `error` unverändert an den Agenten geben; er ist so geschrieben, dass der Agent daraus handeln kann (Bild verkleinern, auslagern, Figur als SVG zeichnen).
3. **Docstring-Ergänzung** an allen drei Schreib-Tools — die Kern-Regeln inline, damit sie am Tool kleben und nicht im Gedächtnis des Agenten:
   > *„`content` ist Markdown und darf Figuren tragen: Inline-`<svg>` (als eigener Block — `<svg` am Zeilenanfang, Leerzeile davor und danach; `viewBox` setzen, `<title>` als erstes Kind, `currentColor` statt harter Farben für den Dark Mode, keine externen Referenzen, kein `class`/`style`), ```` ```mermaid ````-Fences (rendern nur im Reader, nicht im PDF/EPUB), `![alt](https://…)` und `![alt](data:image/…)` (nur png/jpeg/webp/gif/svg+xml, prozent-kodiert oder base64, immer mit sprechendem `alt`). Wer im Text **über** Tags schreibt, setzt sie in Backticks (`` `<svg>` ``), sonst liest der Renderer sie als Figur. Limits: 2 MB je data-URI, 10 MB Medien je Dokument → 413, nichts geschrieben. Volle Konvention: docs/doc_figures_authoring.md."*
4. **`list_conversions`-Docstring**: `content_length` = rohe Länge inklusive eingebetteter Medien; `content_preview` = die ersten 300 Zeichen des **medienbereinigten** Texts.
5. **`get_transcript`-Docstring**: liefert den rohen Markdown-Inhalt — SVG-Quelltext und `data:`-URIs stehen dort in voller Länge (bis 10 MB Medien je Dokument). Wer nur den Text braucht, rechnet damit.

## End-to-end-Beweis = Koordinator-Scope

Auf **Wegwerf-Elementen** (eigene `source_id`, danach in der Library löschen — der Agent löscht nicht):

1. `create_conversion` mit der Vorlage aus [docs/doc_figures_authoring.md](doc_figures_authoring.md) → **201**; im Reader: eine Figur, ein Mermaid-Diagramm.
2. `list_conversions` → `content_preview` des Elements enthält weder `<svg` noch `flowchart`; `content_length` = Länge des geschickten Markdowns.
3. `create_conversion` mit einer `data:`-URI von 2 MB + 1 Byte → **413** mit dem ersten Satz oben; `list_conversions` zeigt **kein** neues Element.
4. `replace_section` auf dem Element aus 1. mit einer Sektion, die eine zweite kleine Figur trägt → **200**, beide Figuren im Reader.

---

*CONVERTER-Seite: RICH-MEDIA fertig, getestet (+110 Tests, 1129 + 1 Skip), Browser-Smoke im Web-Container gefahren ([scripts/smoke_reader_media.py](../scripts/smoke_reader_media.py)), deployt. **Kein Schema-Touch, kein neuer Token, kein neuer Dep, kein neuer Endpoint** — nur eine neue Antwort (413) an drei bestehenden Routen. Geschwister-Briefe: [docs/converter_mcp_card_svg_brief.md](converter_mcp_card_svg_brief.md), [docs/converter_mcp_tag_cleanup_brief.md](converter_mcp_tag_cleanup_brief.md).*
