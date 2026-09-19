# SPRINT RICH-MEDIA — Figuren im Dokument: Inline-SVG, Bilder, Mermaid

**Größe**: L (3 Phasen) · **Datum**: 2026-09-19 · **Vorhaben**: Reader / Wissensdarstellung

## Warum

Oli, 2026-09-19: CONVERTER wird für die Darstellung von Wissen immer wichtiger, und Erklärbär-Dokumente brauchen Schaubilder — Strukturformeln, Elektronenpaar-Pfeile, Flowcharts. Die Render-Probe **#240** („Render-Probe: SVG in Markdown", Inbox) zeigt: **keine der vier Varianten** wird als Bild angezeigt — inline `<svg>`, `<img src="data:…">`, `![](data:…)`, ```` ```mermaid ````.

Der Anlass ist ein externes Dev-Item (Rich Media in Conversions). Dieser Sprint-Prompt ist dessen **gegroundete Fassung**: die Analysefragen aus dessen Schritt 1 sind hier beantwortet, sein Umfang ist auf das geschnitten, was ein Sprint trägt, und zwei Teile sind als eigene Items abgetrennt (CSP, Assets — s. Backlog). Wo dieser Prompt vom Dev-Item abweicht, ist das begründet und gilt.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten, alles am Code und an der Prod-DB belegt 2026-09-19)

### Der Renderer und warum jede Variante scheitert

**Ein** Renderer für Reader, PDF **und** EPUB: [app_pkg/markdown_render.py](../../../app_pkg/markdown_render.py) `render_markdown_to_html` — markdown-it-py 3.0.0 mit **`html: True`** (HTML-Passthrough ist **an**), `breaks: True`, dollarmath, pygments (`noclasses=True` → Inline-`style`); danach **ein** `nh3.clean` (Pin `nh3==0.2.18`) mit der Markdown-Allow-List. `img` mit `src` ist erlaubt; `'*': {class, id, style}` ist ein **Wildcard** über alle Tags.

Gemessen je Variante (Python, gegen den heutigen Renderer):

| Variante | Ergebnis heute | Mechanik |
|---|---|---|
| V1 inline `<svg>` | **verschwindet komplett — samt Text** (`<p>vor <svg><text>Lbl</text></svg> nach</p>` → `<p>vor  nach</p>`) | `svg` nicht in der Allow-List; nh3 wirft den Teilbaum inklusive `<text>`-Inhalt weg. markdown-it reicht das Block-SVG aus #240 als `html_block` (CommonMark Typ 7: Open-Tag allein auf der Zeile) unverändert durch. |
| V2 `<img src="data:…">` | `<img alt="…">` — `src` fällt | nh3s Default-`url_schemes` enthält **kein `data`**. |
| V3 `![](data:image/svg+xml…)` | bleibt **Literal-Text** | ⚠️ **Zwei Tore**: markdown-its eigener `validateLink` lässt `data:image/(gif|png|jpeg|webp)` durch, **nicht** `svg+xml` → wird gar nicht erst zum `<img>`. Ein `data:image/png` käme als `<img>` heraus und fiele dann am nh3-Tor (V2). |
| V3b `![](https://…)` | **rendert heute schon** | https-Rasterbilder funktionieren bereits. |
| V3c `![](Kapitel-8/abb.png)` | `<img src="Kapitel-8/abb.png">` ins Leere | relative Pfade → eigenes Item RICH-MEDIA-ASSETS. |
| V4 ```` ```mermaid ```` | `<pre><code class="language-mermaid">` mit pygments-Text-Lexer | Auf der Reader-Seite gibt es kein mermaid.js. Die Mermaid-**Konverter**-Seite lädt `mermaid@10` (Major **floating**) von jsDelivr mit `securityLevel: 'loose'` ([static/js/mermaid_converter.js](../../../static/js/mermaid_converter.js)). |

⚠️ **CommonMark-Falle für V1**: ein `html_block` Typ 7 endet an der **ersten Leerzeile**. Ein SVG mit Leerzeilen im Markup zerfällt in Absätze — die inneren Zeilen landen in `<p>`. #240 hat keine Leerzeilen im SVG; ein Agent, der „schön formatiert", wird welche setzen. Das gehört in die Autoren-Konvention und in einen Test.

### Der Sanitizer, der schon da ist

[services/svg_sanitize.py](../../../services/svg_sanitize.py) `sanitize_card_svg(raw) -> str`: pure Funktion, nh3-Allow-List als Modul-Konstanten (`_ALLOWED_TAGS`, `_ALLOWED_ATTRIBUTES`, `_filter_attribute` für `url(…)` nur als lokales Fragment), Deckel `MAX_CARD_SVG_BYTES = 100_000`. Bewusst gebannt (Doktrin im Modul-Docstring, jedes Element fällt **durch Abwesenheit**): `script`, `style` (Tag **und** Attribut), `foreignObject`, `use`, `image`, `a`, `animate`/`set`, `iframe`/`audio`/`video`. **Kein `class`** — begründet mit der Kollision zu App-Utilities (`class="hidden"`, Memory `feedback_css_class_collision_in_markdown_views`). Sentinel-Test pinnt, dass nh3 camelCase (`viewBox`) erhält.

⚠️ **Die Wildcard-Falle**: nimmt man die SVG-Tags in die Markdown-Allow-List auf, gewährt deren `'*': {class, id, style}` jedem SVG-Element **`style`** — genau das, was CARD-SVG verbietet (`url()` in CSS). Ein `attribute_filter`, der `style`/`class` auf der SVG-Familie verwirft, oder eine Liste ohne Wildcard für diese Tags — eins von beiden ist Pflicht, sonst ist die Doktrin durch die Hintertür weg.

⚠️ **Die `data:`-Falle**: `url_schemes` in nh3 gilt **global**. Wer `data` freischaltet, damit `<img src="data:…">` trägt, schaltet es auch für `<a href="data:…">` frei. Der `attribute_filter` muss `data:` auf `img@src` beschränken und auf die Bildtypen (`image/svg+xml`, `png`, `jpeg`, `webp`, `gif`) begrenzen.

### Wie Highlights verankert sind — die harte Bedingung

[static/js/library_detail.js](../../../static/js/library_detail.js): der Anker ist ein **Text-Quote-Selector** (`exact` / `prefix` / `suffix`, [models.py](../../../models.py) `Highlight`), **keine Offsets**. Koordinatensystem ist `readerRawText(reader)` = Konkatenation **aller Text-Knoten** unter dem Reader (`TreeWalker`, `SHOW_TEXT`) — ⚠️ **auch versteckter**, der Walker kennt keine Sichtbarkeit. Speichern schneidet aus diesem Text; Wiederfinden ist `indexOf(exact)` + Prefix/Suffix-Scoring bei Mehrfachtreffern + Whitespace-toleranter Rückfall (`locateHighlightOffset`, ab Zeile 858). Server speichert nur die drei Strings; `list_recent_highlights` liefert `exact` + Notiz + Tags.

Folgen, aus der Mechanik gelesen: (1) `<img>` bringt keinen Text-Knoten → für den Anker unsichtbar. (2) Inline-SVG mit `<text>` bringt **neue** Text-Knoten (heute: null, siehe V1) → ein bestehender Anker **direkt hinter** einer Figur bekäme einen anderen `prefix`; ein eindeutiges `exact` findet sich trotzdem. (3) Mermaid: ersetzt man den `<pre>` durch das SVG, **ändert sich** `readerRawText` — behält man den `<pre>` (versteckt) im DOM und rendert das SVG daneben, bleibt er **byte-gleich**.

### Der Bestand, nachgezählt (Prod-DB 2026-09-19, read-only)

223 Conversions, 221 Highlights. `<svg` in **2** Dokumenten (#135, #240) · `data:image` in **1** (#240) · Mermaid-Fence in **4** (#133, #135, #138, #240) · `![…](…)` in 5 · `<img` in 2. **Highlights auf betroffenen Dokumenten: nur #133 mit 20** (DENKZEUG Stufe 1, Mermaid-Fences). #240, #239, #165, #121 tragen **null** Highlights. Die Migrationsfrage des Dev-Items ist damit beantwortet: **kein Bestands-Highlight grenzt an ein inline-SVG**; die 20 auf #133 hängen an Mermaid-Blöcken — und sind per Konstruktion sicher, wenn der Quelltext im DOM bleibt (oben, Folge 3).

⚠️ **Nebenbefund zu „escape statt strip"**: 10 von 223 Dokumenten enthalten Tags, die nh3 heute **still verschluckt** — überwiegend Platzhalter in Prosa (`<uuid>` 16×, `<api-pod>`, `<id>`, `<pid>`, `<name>`, `<example>`). Der Text zwischen den Tags bleibt, der Platzhalter selbst verschwindet. Escapen statt strippen machte diese Stellen sichtbar — in 10 Dokumenten, Reader **und** PDF **und** EPUB. Das ist eine Verhaltensänderung mit Diff, keine Sicherheitsfrage (beides ist sicher).

### Preview, Suche, Limits, Schreibwege

- `content_preview` = `content[:300]` **roh** ([app_pkg/library.py:105](../../../app_pkg/library.py) `_conversion_summary`); der MCP `list_conversions` speist sich daraus. #240 beginnt mit einer Überschrift, ein Dokument, das mit `<svg` beginnt, zeigt heute Quelltext in der Inbox.
- **Es gibt keinen Suchindex**: Suche ist `Conversion.content.contains(search, autoescape=True)` = `LIKE` über den Rohtext ([library.py:272](../../../app_pkg/library.py)). „Suchindex strippen" hat kein Objekt.
- **Kein Inhalts-Limit für Markdown**: nur das globale `MAX_CONTENT_LENGTH = 500 MB` ([app_pkg/__init__.py:50](../../../app_pkg/__init__.py), für Audio-Uploads) und der 413-Handler darunter. **Drei** Schreibwege für `content`: `POST /api/conversions` ([app_pkg/ingest.py:149](../../../app_pkg/ingest.py)), Docwrite full + section ([app_pkg/docwrite.py](../../../app_pkg/docwrite.py)) und der Editor-PUT ([library.py:552](../../../app_pkg/library.py)); alle laufen seit LOST-UPDATE über `Conversion.set_content`.
- **Keine CSP.** Inline-`onclick` in Templates, `window.PageData`-Inline-Blöcke, **Tailwind von CDN in `base.html`**, markdown-it und mermaid von jsDelivr. Eine CSP ohne `'unsafe-inline'` bräche die App an mehreren Stellen → eigenes Item **CSP-BASELINE**, nicht dieser Sprint.
- **Blast-Radius des Renderers**: PDF ([app_pkg/markdown.py:193](../../../app_pkg/markdown.py), Playwright) und EPUB ([services/epub_service.py](../../../services/epub_service.py), ebooklib → Kindle) rendern durch dieselbe Funktion. Playwright rendert inline-SVG und data-URIs nativ — das PDF **gewinnt** die Figuren mit. EPUB3 erlaubt inline-SVG, Kindles Unterstützung ist unzuverlässig — der Bau darf nicht brechen, die Darstellung auf dem Gerät ist **nicht** Gegenstand.
- Der MCP-Server ist **kein Teil dieses Repos**; Änderungen an Tool-Docstrings (`create_conversion`) laufen über einen Brief nach dem Muster [docs/converter_mcp_card_svg_brief.md](../../../docs/converter_mcp_card_svg_brief.md).

## Gesperrte Entscheidungen

1. **Eine SVG-Policy, ein Ort.** Die Allow-List, die Bann-Liste und der `url()`-Filter aus `services/svg_sanitize.py` sind die einzige Definition dessen, was ein SVG darf — für Karten **und** Dokumente. Das Modul exportiert sie; `markdown_render` konsumiert sie. Eine zweite Liste ist kein Zustand, der abgenommen wird. Ob das über **einen** nh3-Pass mit zusammengeführter Liste plus `attribute_filter` läuft oder über Extraktion der SVG-Blöcke und einen zweiten Pass, ist deine Wahl **mit Begründung** — das Wildcard-Problem (oben) muss in beiden Fällen benannt und gelöst sein.
2. **Kein `class` auf SVG in Dokumenten.** Das Dev-Item erlaubt „Klassen"; die CARD-SVG-Doktrin verbietet sie aus einem Grund, der im Reader **genauso** gilt (Tailwind-Utilities sind global). `currentColor` ist ein Wert, kein Attribut, und bleibt erlaubt — Dark Mode funktioniert darüber, nicht über Klassen.
3. **`data:` nur auf `img@src`, nur Bildtypen.** Nie global über `url_schemes` allein.
4. **Mermaid-Quelltext bleibt im DOM.** Der `<pre><code class="language-mermaid">` wird versteckt, nicht ersetzt; das gerenderte SVG kommt daneben. Damit ist `readerRawText` vor und nach dem Sprint byte-gleich, die 20 Highlights auf #133 sind per Konstruktion sicher, und ein Parse-Fehler heißt schlicht: `<pre>` wieder zeigen plus Hinweis. `securityLevel: 'strict'` im Reader (die Konverter-Seite fährt `'loose'` — dort ist der Nutzer der Autor, im Reader ist es der Agent). Mermaid-Quelle und Version wie auf der Konverter-Seite, aber **auf eine exakte Version gepinnt**, nicht `@10`.
5. **Keine CSP in diesem Sprint** (eigenes Item). **Keine Assets / relativen Pfade** (eigenes Item). **Kein Anfassen der Karten** (`front_svg`/`back_svg`, Review-UI). **Kein serverseitiges Mermaid.**
6. **Bestand byte-gleich, gemessen.** Für jedes Prod-Dokument, das **weder** `<svg` **noch** `data:image` **noch** einen Mermaid-Fence enthält, muss `render_markdown_to_html` vorher und nachher **denselben String** liefern. Das ist der Regressionsgate für 219 von 223 Dokumenten und kostet einen read-only-Lauf im Container. ⚠️ Wer „escape statt strip" einführt, verletzt diesen Gate in 10 Dokumenten — deshalb ist das eine **eigene Entscheidung** mit Diff in der Hand (Phase 1.4), nicht Teil des Defaults.
7. **Testdaten gehören einem Wegwerf-User**; Aufräumen strikt nach `user_id` (`api_token` trägt Olis iOS-Tokens). Das Malicious-Fixture und die Probe-Dokumente entstehen unter diesem User — **nicht** über `POST /api/conversions` mit `INGEST_TOKEN` (das schreibt auf Olis Konto), sondern per ORM oder über die Session des Wegwerf-Users.

---

# Phase 1 — Der Renderer: SVG und Bilder, sicher (reines Python, `pytest`-gedeckt)

## 1.1 Sanitizer-Refactor

`services/svg_sanitize.py` exportiert seine Policy (Tags, Attribute je Tag, `attribute_filter`, Bann-Doktrin) so, dass `markdown_render` sie konsumieren kann; `sanitize_card_svg` bleibt in Verhalten und Signatur unverändert (Sentinel-Tests grün, `tests/test_svg_sanitize.py`). ⚠️ Zusätzlich verlangt das Dev-Item `clip-path`; prüfe, ob `clipPath` in die Policy gehört (Referenz per `url(#…)` wie Marker — derselbe Filter greift) und entscheide **mit Begründung**. `mask`/`filter` ebenso benennen, nicht stillschweigend aufnehmen.

## 1.2 Inline-SVG und data-URI-Bilder im Renderer

- `<svg>` im Markdown wird gerendert, durch die eine Policy (gesperrte Entscheidung 1); Wildcard-Falle gelöst und im Kommentar benannt.
- Ein SVG, das zu nichts Renderbarem sanitisiert, wird **nicht** still verschluckt (heutiges Verhalten), sondern hinterlässt einen sichtbaren, markierbaren Platzhalter-Text — die Dokument-Fassung der CARD-SVG-Regel „400 statt stumm leer".
- `data:`-URIs auf `img@src` für die fünf Bildtypen (gesperrte Entscheidung 3); **beide Tore** öffnen — markdown-its `validateLink` für `image/svg+xml` **und** nh3.
- https-Bilder tragen `loading="lazy"` und `referrerpolicy="no-referrer"`; dafür müssen die beiden Attribute in die `img`-Allow-List. Ein Feature-Flag ist **nicht** nötig — https-Bilder rendern heute schon, der Sprint macht sie nur zurückhaltender.
- SVG ohne `viewBox` mit `width`/`height`: `viewBox` ableiten. Wo (Server-Pass vs. Reader-JS) ist deine Wahl; **einmal**, nicht an beiden Enden.
- ⚠️ Die Leerzeilen-Falle (CommonMark Typ 7) als Test festnageln: ein SVG mit Leerzeile im Markup — was passiert, und ist das Ergebnis erklärbar statt still kaputt? Ein Platzhalter mit Grund ist ein legitimes Ergebnis, ein `<p>` voller `<path>`-Text nicht.

## 1.3 Limits und Preview

- **Limits** an allen **drei** Schreibwegen (Ist-Zustand oben), aus einem gemeinsamen, puren Validator: ≤ 2 MB je data-URI, ≤ 10 MB Medien je Dokument → **413** mit deutschem Satz, nichts halb geschrieben. ⚠️ Der Audio-Upload (500 MB, `MAX_CONTENT_LENGTH`) darf davon nichts merken.
- **Preview**: `_conversion_summary` liefert `content_preview` aus einem **medienbereinigten** Text (SVG-Quelltext, data-URIs, Mermaid-Quelltext entfernt); `content_length` bleibt roh und wird als roh **dokumentiert**. Die Suche bleibt, was sie ist (LIKE über roh) — im Bericht als bewusst benennen.

## 1.4 Die Bestands-Entscheidung „escape statt strip"

Miss zuerst: die 10 Dokumente, ihre Tags, und wie der Reader sie **nach** einem Escape zeigen würde (ein Diff der gerenderten HTML reicht). Dann entscheide mit Begründung — oder leg es Oli mit dem Diff vor. Default ohne Entscheidung: **strip bleibt**, Gate 6 gilt für alle 219.

## 1.5 Der Beleg

- Tests für jede Variante aus #240 (der Probe-Text steht in der Prod-DB, hol ihn per `get_transcript`-Äquivalent oder read-only-Query), für das Malicious-Fixture (`<script>`, `onload`, `<image href="https://…">`, `<foreignObject>`, `<a href="javascript:…">`, `style="background:url(https://…)"`, `<a href="data:text/html,…">`) — jedes gebannte Teil fällt, der Rest rendert —, für die Limits (2 MB + 1 Byte → 413, Element nicht angelegt), für Preview-Strip, für die Leerzeilen-Falle, für https-Attribute.
- **Gate 6**: read-only-Lauf im Container über alle Prod-Dokumente, alter gegen neuer Renderer; Ausgabe: Liste der abweichenden IDs. Erlaubt sind **genau** die Medien-Dokumente (und, falls in 1.4 so entschieden, die 10 Escape-Dokumente).
- `pytest tests/` grün, Baseline **1019 + 1 Skip**.

## Stop
**Commit + Push** (Refactor, Renderer, Limits/Preview gern getrennt). Dann warten.

---

# Phase 2 — Der Reader: Mermaid, Darstellung, Anker-Stabilität (Browser-gedeckt)

## 2.1 Mermaid im Reader

Nach gesperrter Entscheidung 4. Lazy beim Sichtbarwerden (`IntersectionObserver`), Theme an `data-global-theme` gekoppelt (Muster aus `mermaid_converter.js`), Parse-Fehler → `<pre>` sichtbar + ein Satz Hinweis, der Rest des Dokuments rendert. ⚠️ Der Hinweis ist UI-Text, kein Agenten-Text → `textContent`; das Mermaid-SVG ist die einzige neue `innerHTML`-Senke und kommt aus mermaid.js im `strict`-Modus — im Kommentar so benennen (CARD-MD-Doktrin: DOM-Knoten statt `innerHTML`, hier die begründete Ausnahme).

## 2.2 Darstellung

`.reader-view svg`, `.reader-view img`: `max-width: 100%; height: auto`; `color` erbt vom Reader-Text (damit `currentColor` in Dark greift). **Kein** erzwungener Hintergrund — ein SVG mit hartem `fill="#fff"` (wie #240) ist dann in Dark eine helle Fläche, wie eine Karten-Figur; ein SVG mit `currentColor` passt sich an. Scoping nach Hausregel (Tag innerhalb `.reader-view`, Präzedenz READER-TABLE). `figcaption` bleibt Text.

## 2.3 Anker-Stabilität — messen, nicht annehmen

Browser-Smoke nach Hausmuster ([scripts/smoke_review_skip.py](../../../scripts/smoke_review_skip.py): Playwright im Web-Container, Wegwerf-User, Dokumente per ORM, `exit 1` bei Fehlschlag), als `scripts/smoke_reader_media.py`:

- Dokument mit allen vier Figur-Arten und Text davor/dazwischen/danach. Highlight **unmittelbar vor** und **unmittelbar nach** jeder Figur setzen (echter Maus-Drag, Memory `feedback_selection_anchor_coordinate_system`) → Reload → beide sitzen an derselben Stelle; `/api/highlights/recent` liefert den korrekten `exact`.
- `readerRawText` vor/nach dem Mermaid-Rendern **byte-gleich** (das ist Entscheidung 4 als Messung).
- Malicious-Fixture als echtes Dokument: **null** ausgehende Requests beim Rendern (Route-Interception über alles), kein Script-Effekt.
- Mermaid-Syntaxfehler-Dokument: Quelltext sichtbar, Hinweis, Rest gerendert.
- Dark und hell, 375 px und Desktop: kein horizontaler Überlauf der **Karte** (die 17-px-Scrollbar-Kante der Seiten-Shell aus LEARN-SKIP ist bekannt und nicht Gegenstand).

## 2.4 PDF und EPUB

Dasselbe Testdokument durch `POST /generate-pdf` (Figuren sichtbar — Playwright rendert sie) und durch den EPUB-Bau (`services/epub_service.py`, ohne Versand: **baut ohne Exception**). Mehr nicht.

## Stop
Smoke grün auf dem deployten Stand, `pytest` grün. **Commit + Push.** Dann warten.

---

# Phase 3 — Bestand, Fixture, Wrap

## 3.1 Live-Abnahme

- **#240** im echten Reader: alle vier Varianten sichtbar (Screenshot).
- **#133** (20 Highlights, Mermaid-Fences): vor dem Deploy die 20 Anker-Positionen (raw offsets) sichern, nach dem Deploy nachmessen — **identisch**. Das ist der eine Bestands-Beleg, der zählt.
- **#165, #121, #239**: rendern wie vorher (Gate 6 deckt sie, hier nur der Blick).
- Die Wegwerf-Dokumente und der Wegwerf-User strikt nach `user_id` weg; #240 bleibt (Olis Probe).

## 3.2 Wrap

- **CLAUDE.md**: ein Bullet *Figuren im Dokument (RICH-MEDIA)* — eine Policy für Karten und Dokumente, die Wildcard- und `data:`-Falle, „Mermaid-Quelle bleibt im DOM" mit dem Anker-Grund, die CommonMark-Leerzeilen-Regel, Limits, was PDF/EPUB tun; Smoke in beide Listen; Testzahl.
- **Autoren-Konvention** für den Agenten als Doc (Muster [docs/card_svg_authoring.md](../../../docs/card_svg_authoring.md)): erlaubte Elemente, keine Leerzeilen im SVG, alt-Text/Caption **immer**, `currentColor` für Dark, Limits. Dazu der **MCP-Brief** `docs/converter_mcp_rich_media_brief.md` (Muster CARD-SVG-Brief): keine API-Änderung an `create_conversion`, Docstring-Ergänzung, Limits, Konvention, `content_length` roh.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): RICH-MEDIA schließen; **RICH-MEDIA-ASSETS** und **CSP-BASELINE** bleiben als Items stehen (Master hat sie angelegt); die **Mesomerie-v2-Fixture** (#239 mit ~10 SVGs + Flowchart) ist **Agenten-Autorenarbeit**, kein Code — als Aufgabe für den Karten-/Erklärbär-Agenten notieren, nicht als Sprint.
- **Memory** bei übertragbarer Lehre; nach dem Schreiben `ls` gegen die Index-Zeile. Kandidat: *ein Sanitizer mit Wildcard-Attributen vergibt an neu aufgenommene Tag-Familien still Rechte, die die Familie ausschließt — Wildcards vor jeder Listen-Erweiterung gegen die neue Familie prüfen.*
- **Im Bericht benennen**: welcher Mechanismus für „eine Policy" gewählt wurde und warum · Gate-6-Ergebnis (welche IDs weichen ab) · die 1.4-Entscheidung · die gemessenen Anker-Positionen von #133 vorher/nachher · Request-Zählung am Malicious-Fixture · was **nicht** gebaut wurde.

## Nicht-Ziele

- **Keine** CSP, **keine** Assets/relativen Pfade (beides eigene Items).
- **Kein** Anfassen der Karten-Felder, der Review-UI, des Karten-Sanitizer-**Verhaltens** (nur seine Struktur).
- **Kein** serverseitiges Mermaid, **kein** Mermaid in Karten, **keine** Highlights auf Figuren, **keine** Bildoptimierung.
- **Kein** JS-Test-Harness (CARD-MD-Doktrin); der Browser-Smoke ist der Gate für alles, was `pytest` nicht sieht — und Phase 1 ist bewusst so geschnitten, dass `pytest` sie **ganz** sieht.
- **Kein** Umbau des Anker-Mechanismus (`exact`/`prefix`/`suffix`, `readerRawText`) — er wird gemessen, nicht geändert.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen; Prod-DB-Lesungen read-only (`mode=ro`).
