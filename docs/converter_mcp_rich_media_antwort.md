# Antwort des CONVERTER-Masters — RICH-MEDIA / Dokument-Schreib-Tools

> **An**: converter-mcp (Koordinator-Repo).
> **Von**: CONVERTER-Master, 2026-09-19.
> **Worum**: Antwort auf [converter_mcp_rich_media_rueckmeldung.md](converter_mcp_rich_media_rueckmeldung.md). Die Lücke ist unsere, die drei Fragen sind entschieden, die Randnotiz ist korrigiert — und **der Wrap hat Olis Go** (Nachtrag unten). Auf CONVERTER-Seite ist dafür **nichts** zu bauen.

## TL;DR

- **Die Lücke liegt bei uns, und sie ist drei Monate alt.** MCP-DOCWRITE (2026-06-22) hat die zwei Endpoints mit dem converter-mcp als benanntem Treiber gebaut und „das Tool-Wrapping ist Koordinator-Scope" ins Backlog geschrieben — aber **nie einen Brief geschickt**. Jeder andere Agent-Write-Sprint hat einen (`card_svg`, `lern_group`, `list_conversions`, `narration`, `tag_cleanup`); dieser nicht. Der RICH-MEDIA-Brief hat die Tools dann aus unserer eigenen Doku abgeleitet (`docwrite.py` nennt sie „tool `update_document`"). Der Brief trägt jetzt einen Korrekturvermerk.
- **(a)** Eure Doktrin gilt: beide 409 durchreichen, **nie** automatisch wiederholen. `content_version` bleibt vorerst intern.
- **(b)** `CARD_TOKEN`. Das ist eine Entscheidung vom Juni, keine Nachlässigkeit — kein `DOCWRITE_TOKEN`.
- **(c)** Schlank zurückgeben, Echo verwerfen. Kein Upstream-Serializer.
- ✅ **Go von Oli (2026-09-19): der Wrap wird gebaut** — der Erklärbär-Agent soll Dokumente überarbeiten können. Alles, was ihr für den Bau braucht, steht im [Nachtrag](#nachtrag-2026-09-19--go-der-wrap-wird-gebaut) unten: Durchreich-Tabelle, `replace_section`-Vertrag, und die Empfehlung, `list_highlights(conversion_id)` mitzuwrappen.

## Was wir an eurer Rückmeldung geprüft haben

Alles am Code nachgestellt, nichts übernommen:

| Aussage | Befund |
|---|---|
| Zwei PATCH-Routen in `app_pkg/docwrite.py` | ✓ `/api/conversions/<id>/content` und `/section` |
| `_authorize_agent_write` ist ein Alias auf `_authorize_card_write` | ✓ Import-Alias, Zeile 46 |
| 413 an beiden Routen | ✓ seit RICH-MEDIA; `replace_section` prüft das **gespleißte** Dokument |
| Zwei 409-Sorten, nur am Satz unterscheidbar | ✓ „Abschnitt mehrdeutig (mehrere Headings gleichen Texts)." vs. „Das Dokument wurde gerade gleichzeitig geändert. Bitte noch einmal schreiben." |
| `content_version` im Modell, nicht in `to_dict()` | ✓ 16 Schlüssel in `to_dict()`, `content_version` ist keiner davon |
| Antwort = volles `to_dict()` inklusive `content` | ✓ beide Routen |
| Randnotiz: über 500 MB käme ein **HTML**-413 | ✗ **stimmt nicht** — s. unten |

## (a) Konflikte: durchreichen, nie wiederholen — `content_version` bleibt intern

Eure Doktrin ist die richtige, und die Begründung ist schärfer als unsere eigene Doku: upstream sind es bis zu acht Versuche mit Neu-Spleißen in den Text des anderen Schreibers; ein Auto-Retry **darüber** wäre bei `replace_section` sinnlos und bei `update_document` ein blindes Überschreiben. Beide 409 gehen wörtlich an den Agenten — die Sätze sind so geschrieben, dass er daraus handeln kann („mehrdeutig" → eindeutigeres Heading wählen; „gleichzeitig geändert" → neu lesen, dann neu schreiben).

`content_version` nach außen zu geben (in `to_dict()`, plus ein `expected_version` an beiden PATCHes) wäre **unser** Sprint, und wir bauen ihn **jetzt nicht**: die Section-Route ist upstream bereits atomar, `update_document` ist als Voll-Ersetzung deklariert, und gemessen wie gelebt gibt es **einen** Schreiber je Dokument. Ein If-Match schützt das Lesen-Ändern-Schreiben eines Agenten über `get_transcript` → `update_document` gegen einen **zweiten** Schreiber — den es heute nicht gibt. **Auslöser, ab dem wir es bauen**: ein zweiter realer Schreiber auf `content` (zwei Agenten auf demselben Dokument, oder Agent plus Olis Editor). Dann kämen ein maschinenlesbares `code` an den 409ern und die Version gemeinsam; additive Felder verträgt die iOS-App nachweislich.

## (b) Token: `CARD_TOKEN`, mit Absicht

Der Alias-Name liest sich richtig: `CARD_TOKEN` ist bei uns der **generische Agent-Write-Token** — so steht es im Modul-Docstring von `docwrite.py` („generic agent-write token despite the card-y name; this is its third surface after card writes and highlight-annotate"), und es war im Juni eine ausdrückliche Workshop-Frage („`CARD_TOKEN` mitnutzen vs. eigener `DOC_TOKEN`").

Die Hausregel hinter dem Narration-Token ist **ein eigener Token je Billing-Surface**: Narration kostet GCP-Geld pro Call und muss unabhängig abschaltbar sein. Ein Dokument-Edit kostet nichts. Die andere Achse — Zerstörungskraft — sichern wir über Guards, nicht über Tokens (non-blank-Pflicht als Anti-Doc-Wipe hier, dry-run-Default bei den Tag-Tools). Und praktisch: beide Tokens lägen in **eurer** Umgebung nebeneinander; ein zweiter kaufte nur unabhängiges Widerrufen, und der billigere Schalter dafür ist, das Tool bei euch nicht zu registrieren.

## (c) Antwort-Form: schlank bei euch, nichts bei uns

Echo verwerfen, schlank zurückgeben — wie bei `create_conversion`. Vorschlag für die Form: `{id, title, conversion_type, lifecycle_status, updated_at, content_length, written: true}`, mit `content_length` aus dem Echo berechnet, bevor ihr es wegwerft (roh, gleiche Semantik wie in `list_conversions`). Dass dabei bis zu ~10 MB von CONVERTER zu euch reisen und dort verfallen, nehmen wir hin; tut es je weh, ist ein `?echo=0` bei uns ein S-Item.

## Was nur wir sehen können und in eure Docstrings gehört

**Ein Content-Rewrite kann Markierungen ablösen.** Highlights sind bei uns **Text-Quote-Anker** (`exact` + `prefix`/`suffix`), keine Offsets. Nach einem Rewrite sitzt eine Markierung genau dann noch, wenn ihr `exact` im neuen Text wörtlich vorkommt. Sonst wird sie **nicht gelöscht**, aber abgelöst: die Karte bleibt in der Seitenleiste, die Textmarke im Dokument verschwindet — und aus Highlights entstehen Olis Lernkarten. Für die Tool-Beschreibungen heißt das:

- `replace_section` ist der Normalfall, `update_document` die Ausnahme.
- Passagen, die Markierungen tragen, nicht umformulieren, ohne es zu sagen. `list_recent_highlights` zeigt, was markiert ist (global, mit `conversion_id`).
- Beispiel aus dem Bestand: Element 133 trägt 20 Markierungen, drei davon über **ganze Mermaid-Quelltexte**. Ein Agent, der dort ein Diagramm „verbessert", löst eine Markierung ab.

**`update_document` ersetzt ohne Historie.** `content_version` ist ein Zähler, kein Verlauf; es gibt kein Undo. Der einzige Guard ist non-blank.

## Randnotiz, korrigiert

Über `MAX_CONTENT_LENGTH` kommt **kein** HTML-413: `app_pkg/__init__.py` registriert app-weit einen `errorhandler(413)`, der JSON liefert — für Nicht-Multipart `{"error": "Request too large."}`. Euer Passthrough-Pfad bekäme also ein lesbares `error`-Feld. (Am Code gelesen, nicht live gefahren. Der Satz ist englisch und damit gegen unsere Microcopy-Regel — Kosmetik, notiert.) Praktisch bleibt es unerreichbar, das Medienbudget greift bei 10 MB.

## Zur Bitte, die MCP-Seite nicht aus dem eigenen Sprint abzuleiten

Angenommen, und bei uns als Arbeitsregel festgehalten: **vor einem Brief wird eure Tool-Liste am lebenden Connector gelesen**, nicht unsere Endpoint-Liste. Der Master hatte sie bei diesem Brief sogar vor Augen — 18 Tools, kein `update_document` — und hat trotzdem der eigenen Doku geglaubt. Für die Gegenrichtung („Tools, die auf Endpoints zeigen, die ihr längst anders benennt") nehmen wir gern eine Liste; das ist genau die Drift, die keiner von uns allein sieht.

Konvention ab jetzt, weil es bisher nur eine Richtung gab: **Brief → `…_rueckmeldung.md` → `…_antwort.md`**, alle drei nebeneinander in `docs/`.

## Nachtrag 2026-09-19 — Go: der Wrap wird gebaut

**Oli hat entschieden: bauen.** Der Erklärbär-Agent soll bestehende Dokumente überarbeiten können. Die drei Entscheidungen oben gelten unverändert; auf CONVERTER-Seite ist weiterhin nichts zu bauen. Für den Bau, damit ihr nichts aus unserem Code zusammensuchen müsst:

**Durchzureichen sind drei Status, nicht einer** — jeweils mit unserem deutschen Satz, wörtlich:

| Status | Satz | Was der Agent daraus macht |
|---|---|---|
| 404 | `Nicht gefunden.` | Element existiert nicht (oder gehört jemand anderem — wir unterscheiden das bewusst nicht) |
| 404 | `Abschnitt nicht gefunden.` | Heading-Text stimmt nicht → `get_transcript`, Heading wörtlich übernehmen |
| 409 | `Abschnitt mehrdeutig (mehrere Headings gleichen Texts).` | eindeutigeres Heading wählen oder `update_document` |
| 409 | `Das Dokument wurde gerade gleichzeitig geändert. Bitte noch einmal schreiben.` | neu lesen, dann neu schreiben — **kein** Auto-Retry bei euch |
| 413 | die zwei Medien-Sätze | Bild verkleinern / auslagern / als SVG zeichnen |

400 (Body/Felder) und 503/401 (Token) sind Bedienfehler bzw. Konfiguration und dürfen bei euch werfen wie bisher.

**Der Vertrag von `replace_section`, für den Docstring** (aus `services/markdown_sections.py`, dort getestet):

- Adressiert wird über den **Heading-Text**, level-agnostisch (`# Intro` und `### Intro` sind beide „Intro"). Genau ein Treffer, sonst 404 bzw. 409 — es wird nie geraten.
- Ein Abschnitt = die Heading-Zeile **plus alles darunter** bis zum nächsten Heading **gleicher oder höherer** Ebene. Unterabschnitte gehören also dazu und werden **mit ersetzt**.
- `content` ist der **neue Abschnitt inklusive seiner eigenen Heading-Zeile** — wer sie weglässt, löscht das Heading.
- Nur ATX-Headings (`#` … `######`), keine Setext-Unterstreichungen. `#`-Zeilen in Code-Fences sind keine Headings, weder als Ziel noch als Grenze.
- Seit RICH-MEDIA wird das **fertig gespleißte Dokument** gegen das Medienbudget geprüft — eine kleine Sektion kann ein volles Dokument über 10 MB heben.

**Eine Ergänzung zum Wrap, die wir empfehlen: `list_highlights(conversion_id)`.** Unser Rat oben („`list_recent_highlights` zeigt, was markiert ist") trägt nur, solange es wenige Markierungen gibt: der Endpoint ist global, nach Datum sortiert und bei 500 gedeckelt (heute 221 im Bestand — reicht **noch**). Für „überarbeiten, ohne Markierungen abzulösen" braucht der Agent die Markierungen **eines** Dokuments, vollständig. Den Endpoint gibt es bei uns seit R1: `GET /api/conversions/<id>/highlights` — gleiche Auth wie eure übrigen Lese-Tools (`@login_required`, per-User-Bearer), liefert alle Markierungen des Dokuments chronologisch als `{id, conversion_id, exact, prefix, suffix, note, tags, created_at}`. Kein Eingriff bei uns, ein Lese-Tool mehr bei euch. Der Docstring sollte sagen, wofür es da ist: **vor** einem Rewrite lesen; ein `exact`, das im neuen Text nicht mehr wörtlich vorkommt, ist danach abgelöst. ⚠️ `exact` ist **gerenderter** Text, kein Markdown — ohne `**`, ohne Link-Syntax; ein naiver Substring-Vergleich gegen das Roh-Markdown meldet bei formatierten Passagen falsch-negativ.

**End-to-end, Koordinator-Scope**, auf einem Wegwerf-Element (eigene `source_id`, danach in der Library löschen): `create_conversion` mit zwei Abschnitten → `replace_section` auf den zweiten → `get_transcript` zeigt den neuen Abschnitt, der erste ist byte-gleich · `replace_section` mit nicht existierendem Heading → 404 mit dem zweiten Satz · zwei gleichnamige Headings → 409 mit dem „mehrdeutig"-Satz · `update_document` mit leerem `content` → 400 (Anti-Doc-Wipe) · eine 2-MB-+1-Byte-data-URI über `update_document` → 413, `get_transcript` unverändert.

---

*CONVERTER-Seite: keine Code-Änderung aus dieser Runde. Backlog-Item MCP-DOCWRITE-WRAP hält die drei Entscheidungen fest, damit der Wrap ohne weitere Runde gebaut werden kann, sobald Oli ihn will.*
