# SPRINT CARD-SVG — Abbildungen auf Lernkarten

**Größe**: M/L (4 Phasen) · **Datum**: 2026-07-25 · **Vorgänger**: LEARN-UP (`3fbd385`)

## Warum

Oli: *„ich habe festgestellt, dass abbildungen extrem wertvoll sind um lernstoff anzueignen — folglich möchte ich dass die lernkarten auch svgs enthalten können."*

Seine Karten sind Biotech-Stoff (Antikörper-Formate, Chromatographie-Flows, DSP-Pipelines) — genau der Stoff, der als Schema *klickt* und als Fließtext zäh bleibt. Karten schreibt ohnehin der externe Agent via `CARD_TOKEN`; SVG ist Text, passt also ohne neue Infrastruktur in die bestehenden `Text`-Spalten und die bestehende Write-API. **Kein Asset-Store, kein Upload, keine Datei-Verwaltung.**

## Gegroundeter Ist-Zustand (vom Master verifiziert — nicht neu herleiten)

- Kartentext wird **ausschließlich** über `textContent`/DOM-Knoten gerendert; [static/js/review.js:6-7](static/js/review.js) dokumentiert das als XSS-Invariante. Kein Markdown-, kein HTML-Pfad auf Karten.
- **Kein Asset-Store**: `OUTPUT_DIR` ([app_pkg/config.py](app_pkg/config.py)) ist das Podcast-WAV-Volume, sonst nichts. Kein Upload-Pfad.
- [app_pkg/markdown_render.py](app_pkg/markdown_render.py) hat die geteilte nh3-Allow-List (Markdown→PDF, Library-Reader, EPUB). Sie kennt **kein** SVG und wird in diesem Sprint **nicht angefasst**.
- Mermaid existiert nur als eigenständige Konverter-Seite (CDN-geladen, client-only, nichts persistiert) — **irrelevant für diesen Sprint**, nicht anfassen.
- `Card` ([models.py:285](models.py)) hat `front`/`back`/`cloze_text`/`prompt`/`note` als `db.Text`. `to_dict()` serialisiert alle; `_card_summary` ([app_pkg/cards.py:232](app_pkg/cards.py)) ist bewusst schlank („no answer/snapshot bulk").
- Der MCP-Server ist **extern** (`thereallyrealcomedian/converter`), koordiniert über `docs/card_api_contract.md` + `docs/converter_mcp_*_brief.md`.

### Der Sanitizer-Probe (Master, gegen den Repo-Pin `nh3==0.2.18`)

Die eine Frage, die das Design gekippt hätte: **überlebt `viewBox` seine Groß-/Kleinschreibung?** (Ein lowercased `viewbox` und nichts skaliert mehr.) Antwort: **ja** — ammonia/html5ever fährt den SVG-Foreign-Content-Attribut-Adjust.

| Fall | Ergebnis |
|---|---|
| `viewBox`, `preserveAspectRatio`, `markerWidth`, `refX`, `gradientUnits`, Tag `linearGradient` | ✅ camelCase bleibt exakt erhalten |
| `<script>`, `onload`/`onclick`, `<foreignObject>`, `<a xlink:href="javascript:">`, `<use>` extern, `<animate attributeName=href>`, `<image href=extern>`, `<style>` | ✅ alle acht fallen |

⚠️ **Sie fallen durch Weglassen in der Allow-List**, nicht durch Sonderlogik. Die Doktrin ist also: **Allow-List eng halten**, und die acht nie aufnehmen.

## Gesperrte Entscheidungen (Oli, 2026-07-25 — nicht neu aufmachen)

1. **Autoren-Pfad: nur Agent.** `front_svg`/`back_svg` sind zwei neue Felder an `POST /api/cards` + `PATCH /api/cards/<id>`. **Kein** Paste-/Upload-UI, keine Edit-Fläche im Review.
2. **Flächen: Front und Rückseite.** Zwei Spalten. Bild-als-Frage („was passiert in Schritt 3?") **und** Erklär-Figur zur Lösung.
3. **Dark-Mode: feste helle Figuren-Fläche.** Die Figur sitzt wie eine Lehrbuch-Abbildung auf Papier — heller Hintergrund unabhängig vom Theme. Der Agent darf Farben hart setzen, es bricht nie. **Kein** `currentColor`-Vertrag.

## Architektur-Doktrin (bindend)

- **Eigener Sanitizer, eigene Allow-List.** `services/svg_sanitize.py` — pures Modul (kein Flask, kein SDK), Vorbild [services/epub_math.py](services/epub_math.py) / [services/markdown_sections.py](services/markdown_sections.py). Es **importiert nichts** aus `app_pkg/markdown_render.py` und ändert dort nichts → **null Blast-Radius** auf PDF/EPUB/Library/Reader.
- **Roh speichern, beim Lesen sanitisieren** — Haus-Stil (Markdown wird auch roh gespeichert und beim Rendern bereinigt). Zusätzlich **beim Schreiben validieren**, damit der Agent eine 400 mit Grund bekommt statt still eine leere Figur zu produzieren.
- **Der Kartentext bleibt `textContent`.** Nur die zwei neuen Figuren-Container bekommen bereinigtes HTML. Cloze (`renderCloze`) wird **nicht angefasst**.

---

# Phase 1 — Sanitizer (pur, testbar, ohne Flask)

Höchstes Risiko, bestes Test-Verhältnis → zuerst und alleinstehend.

## 1.1 `services/svg_sanitize.py` anlegen

Öffentliche API:

```python
MAX_CARD_SVG_BYTES = 100_000

def sanitize_card_svg(raw: str) -> str:
    """Bereinigt agent-geschriebenes SVG auf eine enge Allow-List.
    Gibt '' zurück, wenn nichts Renderbares übrig bleibt."""
```

Regeln:
- `None`/leer/nicht-`str` → `''` (**`isinstance` vor allem anderen** — truthy Non-String darf keine 500 werfen; Präzedenz `reference_strict_bool_isinstance_destructive_writes`).
- Über `MAX_CARD_SVG_BYTES` (utf-8-kodiert gemessen) → `''`.
- `nh3.clean(raw, tags=..., attributes=...)` mit der **eigenen** Liste.
- Nach dem Clean: enthält das Ergebnis kein `<svg` → `''` (kein Rumpf-Fragment durchlassen).

**Allow-List (Startpunkt — bewusst eng, erweitern nur mit Begründung im Kommentar):**

- Tags: `svg g defs title desc path rect circle ellipse line polyline polygon text tspan marker linearGradient radialGradient stop`
- Attribute pro Tag, u.a. `svg`: `viewBox width height xmlns preserveAspectRatio class` · Geometrie-Attribute je Form · Präsentations-Attribute `fill stroke stroke-width stroke-linecap stroke-dasharray opacity transform font-size font-family text-anchor dominant-baseline`
- **Kein `'*'`-Wildcard-Eintrag.** Insbesondere **kein `style`-Attribut** (der geteilte Markdown-Allow-List hat `'*': {'class','id','style'}` — hier **nicht** nachbauen).

**Nie-aufnehmen-Liste als Kommentar ins Modul**, mit Grund je Eintrag:
`script` · `style` · `foreignObject` (schmuggelt beliebiges HTML) · `use` (externe Refs) · `image` (externe Ladung = LAN-Egress/Tracking) · `a` (`javascript:`) · `animate`/`set` (`attributeName="href"`) · `iframe`/`audio`/`video`.

## 1.2 `tests/test_svg_sanitize.py`

Die Probe-Fälle des Masters als Regressions-Tests — **alle zwölf**:

1. `viewBox` überlebt camelCase · 2. `preserveAspectRatio` + `markerWidth`/`refX` · 3. Tag `linearGradient` + `gradientUnits` · 4. `<script>` weg · 5. `onload`/`onclick` weg · 6. `<foreignObject>` weg (inkl. `onerror` im Rumpf) · 7. `<a xlink:href="javascript:">` weg · 8. `<use>` extern weg · 9. `<animate attributeName="href">` weg · 10. `<image href="https://…">` weg · 11. `<style>` weg · 12. realistisches Diagramm (rect+text+path+g/transform) kommt intakt durch.

Plus: leerer/`None`/Nicht-String-Input → `''` · Über-Cap → `''` · Nicht-SVG-Input (`<p>hi</p>`) → `''`.

⚠️ **Sentinel-Test explizit so benennen und kommentieren**, dass er bei einem `nh3`-Bump laut wird: Test 1 ist die Versicherung, dass die camelCase-Erhaltung nicht still verloren geht (gleiche Doktrin wie die Flask-WTF-Sentinels in `tests/test_csrf_inversion.py`).

## Stop
`pytest tests/` grün. Bericht: Allow-List-Umfang, Test-Zahl vorher/nachher, alle zwölf Vektoren belegt. **Commit + Push** `feat(CARD-SVG): SVG-Sanitizer mit enger Allow-List (P1)`. Dann warten.

---

# Phase 2 — Schema + API

## 2.1 Modell + Migration

- [models.py](models.py): `front_svg` und `back_svg` als `db.Column(db.Text, nullable=True)` an `Card`, mit Kommentar (agent-geschrieben, roh gespeichert, beim Lesen sanitisiert).
- `_run_pending_migrations` ([app_pkg/__init__.py:165](app_pkg/__init__.py)): idempotenter Block für Tabelle `card`, `get_columns`-Check + `ALTER TABLE card ADD COLUMN …`, exakt im Stil der bestehenden Einträge.

## 2.2 Lesen — **eine** autoritative Sanitize-Stelle

- `Card.to_dict()`: `'front_svg': sanitize_card_svg(self.front_svg) or None` (analog `back_svg`). So bekommt **jeder** Konsument — Web-API, MCP, iOS-App — garantiert bereinigtes SVG, ohne dass eine Fläche es vergessen kann.
- **`_card_summary` bleibt SVG-frei.** Es ist der Listen-Endpoint und bewusst schlank; ein 30-KB-SVG pro Zeile hätte dort nichts verloren. Als Kommentar festhalten.

## 2.3 Schreiben — validieren, roh speichern

- `POST /api/cards`: `front_svg`/`back_svg` aus dem Body. Ist ein Feld gesetzt und non-blank, aber `sanitize_card_svg(...)` liefert `''` → **400** mit deutschem Grund (max 2 Sätze, Microcopy-Regel), der den wahrscheinlichen Fall benennt: zu groß, kein `<svg>`-Wurzelelement, oder nur nicht-erlaubte Elemente. Sonst **roh** in die Spalte.
- `PATCH /api/cards/<id>`: beide Felder in die updatable-fields-Tuple ([app_pkg/cards.py:365](app_pkg/cards.py)); dieselbe Validierung; **Leeren muss gehen** (`null`/`""` → Spalte auf `None`).
- Die Typ-Validierung (`_validate_card_type_payload`) bleibt **unverändert**: eine Abbildung ersetzt **kein** Pflichtfeld. Eine Karte braucht weiterhin front+back / cloze / prompt. Das ist bewusst — eine reine Bild-Karte ohne Text wäre nicht abfragbar.

## 2.4 Tests

`tests/test_cards.py` erweitern: POST mit gültigem SVG → 201 + Feld in der Response · POST mit `<script>`-SVG → gespeichert-aber-bereinigt-beim-Lesen · POST mit Müll-SVG → 400 · PATCH setzt/leert · `to_dict()` liefert bereinigt, `_card_summary` enthält die Felder **nicht** · Migration idempotent (zweiter Aufruf no-op).

## Stop
`pytest tests/` grün. Bericht: Test-Zahl, Migrations-Beleg, die exakte 400-Microcopy. **Commit + Push** `feat(CARD-SVG): front_svg/back_svg — Schema, Write-Validierung, Read-Sanitize (P2)`. Dann warten.

---

# Phase 3 — Review-UI

⚠️ **Die Test-Suite rendert keine Templates** (dokumentiertes Limit in CLAUDE.md) — diese Phase braucht einen **Live-Smoke**.

## 3.1 `templates/review.html`

Zwei Figuren-Container:
- einen über `#review-question` (Front-Figur),
- einen im `#review-answer-wrap` (Back-Figur, erscheint mit der Lösung).

Als `<figure class="review-figure hidden">` mit eigener Id. Semantik beachten: die Figur ist Inhalt, kein Dekor.

## 3.2 `static/js/review.js`

- Ein Helper `renderFigure(container, svg)`: leer/falsy → Container leeren **und** verstecken; sonst `innerHTML = svg` + zeigen. Das ist die **einzige** `innerHTML`-Stelle für Kartendaten.
- `renderCard(card)`: Front-Figur setzen **und die Back-Figur zurücksetzen**. ⚠️ **Stale-Figuren-Falle** — ohne explizites Reset zeigt Karte N+1 kurz die Figur von Karte N. Genauso `hide()` bei Kartenwechsel.
- `revealAnswer()`: Back-Figur setzen.
- **Der Kommentar im Datei-Kopf (Zeile 6-7) muss aktualisiert werden.** Er behauptet aktuell pauschal „User content is rendered via textContent / DOM nodes (XSS-safe)". Neu: Text weiterhin ausschließlich `textContent`; **Ausnahme** sind die beiden Figuren-Container, deren SVG **server-seitig** von `services/svg_sanitize.py` bereinigt kommt. Ein falsch stehengelassener Kommentar ist hier schlimmer als kein Kommentar.

## 3.3 `static/css/style.css`

Im passenden Abschnitt (TOC pflegen):
- `.review-figure`: **feste helle Fläche** (gesperrte Entscheidung 3) — heller Hintergrund in **beiden** Themes, dezenter Rahmen/Radius passend zum Neomorphism-DS, Padding, `margin` konsistent zum Karten-Rhythmus.
- `.review-figure svg { width: 100%; height: auto; display: block; }` — `viewBox` trägt die Skalierung.
- **`max-height` setzen.** Sonst schiebt eine hohe Figur die Bewertungs-Buttons unter die Falz — Oli lernt auf dem iPhone. Deckel + `object-fit`-artiges Verhalten über die Höhe.
- Hardcodes vermeiden, `--nm-*`-Token nutzen wo vorhanden (Präzedenz `reference_design_system_realignment_is_budget_audit`).

## 3.4 Smoke (Pflicht)

Karte mit Front- **und** Back-SVG via `POST /api/cards` anlegen (lokal, `CARD_TOKEN`), dann in `/review` durchspielen: Front-Figur sichtbar → aufdecken → Back-Figur erscheint → **nächste Karte ohne SVG zeigt keine Rest-Figur** (der Stale-Test) → Cloze-Karte unverändert → Dark **und** Light. Danach die Test-Karte wieder entfernen.

## Stop
`pytest tests/` grün + Smoke-Protokoll (welche Fälle, welches Theme, Screenshot-Beschreibung oder Beobachtung je Schritt). **Commit + Push** `feat(CARD-SVG): Figuren im Review-UI (P3)`. Dann warten.

---

# Phase 4 — Doku, Agent-Kontrakt, Wrap

## 4.1 `docs/card_svg_authoring.md` (neu) — **das Stück, das über die Qualität entscheidet**

Zeichen-Konvention für den karten-schreibenden Agenten. Ohne sie produziert er hübsches, unlesbares oder stumm zerstrippertes SVG. Inhalt:
- **Erlaubte Tags/Attribute** (die Liste aus P1, als Tabelle).
- **`viewBox` ist Pflicht**, keine absoluten `width`/`height` in px — die Fläche skaliert.
- **Seitenverhältnis** für's Handy: eher breit-flach (~16:10 bis 3:2), nichts Hohes — sonst frisst die Figur den Bildschirm.
- **Mindest-Schriftgröße** relativ zur viewBox, damit Labels auf dem iPhone lesbar bleiben.
- **Feste helle Fläche** → dunkle Farben sind sicher und erwünscht; keine hellen Töne auf Weiß.
- **Größen-Deckel** (`MAX_CARD_SVG_BYTES`).
- **Externe Referenzen werden still entfernt** — keine `<image>`, keine `<use>`, keine Web-Fonts. Alles muss selbsttragend sein.
- **Was zeichnen**: schematische Flows, beschriftete Boxen, Achsen, Verläufe, Zustands-Übergänge. Nicht: Fotorealismus, Molekülstrukturen mit Anspruch auf chemische Korrektheit, dichte Tabellen (dafür ist der Kartentext da).
- **Wann überhaupt**: eine Abbildung nur, wenn sie die Karte *trägt* — Räumliches, Sequenz, Vergleich. Nicht als Deko auf jede Karte.

## 4.2 `docs/converter_mcp_card_svg_brief.md` (neu)

Brief ans converter-mcp-Team, im Format der bestehenden Briefs (Vorbild [docs/converter_mcp_tag_cleanup_brief.md](docs/converter_mcp_tag_cleanup_brief.md)): TL;DR, die zwei neuen optionalen Felder an `create_card`/`update_card`, Typ `string|null`, Validierungs-Verhalten (400 mit Grund), Verweis auf `docs/card_svg_authoring.md` als **Tool-Doc-Inhalt**, und der Hinweis dass `list_cards` die Felder bewusst **nicht** führt (nur `get_card`).

## 4.3 `docs/card_api_contract.md`

Die zwei Felder an `create_card`/`update_card` ergänzen; die Read-Seite (`get_card` führt sie, `list_cards` nicht) festhalten.

## 4.4 `CLAUDE.md`

Ein Bullet in *Architecture Notes*, im Ton der Nachbarn: `front_svg`/`back_svg`, eigener Sanitizer mit **eigener** Allow-List (null Blast-Radius auf den Markdown-Pfad), roh gespeichert / beim Lesen sanitisiert / beim Schreiben validiert, Text bleibt `textContent`, feste helle Figuren-Fläche, `MAX_CARD_SVG_BYTES`. **Plus die Warnung**: bei einem `nh3`-Bump die camelCase-Erhaltung (`viewBox`) re-verifizieren — der Sentinel-Test in `tests/test_svg_sanitize.py` schlägt an.

## 4.5 STATUS.md + BACKLOG.md

Wrap im üblichen Format. **Bullet-Guard vor dem Commit**: `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein.

Zwei Follow-ups **als Backlog-Item anlegen, nicht bauen** (bedarfsgetrieben — Oli entscheidet nach Gebrauch, ob er sie braucht):
- **Figur-Zoom**: Tap auf die Figur → Vollbild/Lightbox. Auf dem iPhone bei detaillierten Schemata plausibel wertvoll.
- **iOS-Rendering**: SwiftUI rendert SVG **nicht** nativ (braucht `WKWebView` oder eine Lib). Die Felder liegen ab P2 in `get_card` bereit; der iOS-Lern-Port (v2) muss das einplanen. Als Notiz für den iOS-Agenten festhalten.

## 4.6 Memory

`reference_card_svg_sanitize.md` in der Memory-Zone + Index-Zeile in `MEMORY.md`. Kern: **eigene enge Allow-List statt geteiltem Markdown-Sanitizer** (Blast-Radius), nh3/ammonia **erhält SVG-camelCase** (`viewBox` überlebt — der Befund, der das Design trug), Vektoren fallen **durch Weglassen** → die Nie-aufnehmen-Liste ist die eigentliche Sicherheitsgrenze, roh-speichern/lesend-sanitisieren als Haus-Stil, Stale-Figuren-Reset beim Kartenwechsel. Verlinken: `[[reference_math_protect_then_render]]` (Schwester-Fall: fremdes Markup sicher durch den Render-Pfad).

## Stop
`pytest tests/` grün, Bullet-Guard leer. **Commit + Push** `docs(CARD-SVG): Zeichen-Konvention, MCP-Brief, Wrap (P4)`. Schlussbericht mit Deploy-Schritten für die Mintbox.

---

## Nicht-Ziele (explizit)

- **Kein** Upload/Paste-UI, kein Asset-Store, keine Bild-Dateien (PNG/JPG). SVG-Text-in-Spalte, sonst nichts.
- **Kein** Markdown-/HTML-Rendering für Kartentext. Text bleibt `textContent`.
- **Kein** Anfassen von `app_pkg/markdown_render.py`, der geteilten Allow-List, des Mermaid-Konverters, des Cloze-Renderers oder des Schedulers.
- **Kein** Bild-Extrahieren aus Quell-PDFs. Verwandt, aber ein anderes Feature — nicht hier.
- **Kein** Figur-Zoom, kein iOS-Port (beide → Backlog).

## Deploy (Schlussbericht)

Schema-Touch → **Prod-DB vor dem Deploy sichern**. Auf der Mintbox: `git pull` + `docker compose up -d --build` (Templates/CSS sind ins Image gebacken — `--build`, nicht `restart`).
