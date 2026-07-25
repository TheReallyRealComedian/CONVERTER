# Developer-Brief an das converter-mcp-Team — CARD-SVG (Abbildungen auf Lernkarten)

> **An**: converter-mcp-Entwickler (Koordinator-Repo).
> **Von**: CONVERTER-Master, 2026-07-25.
> **Worum**: Lernkarten können jetzt **Abbildungen** tragen — SVG-Markup in zwei neuen optionalen Feldern an den bestehenden Card-Writes. Kein neuer Endpoint, kein neuer Token, kein Asset-Store, kein Upload. Dieser Brief sagt, was der converter-mcp anpasst.

## TL;DR (bitte zuerst lesen)

- **Zwei neue optionale Felder** — `front_svg` und `back_svg` — an den **bestehenden** Tools `create_card` (`POST /api/cards`) und `update_card` (`PATCH /api/cards/<id>`). Typ **`string|null`**. **Kein neues Tool, kein neuer Endpoint, kein neuer Token** (`CARD_TOKEN` steht), kein neuer Dep MCP-seitig.
- **Schema-Touch CONVERTER-seitig** (zwei `Text`-Spalten an `card`, Auto-Migration beim Boot) — für den MCP unsichtbar.
- **Validierung**: ist ein Feld gesetzt und non-blank, sanitisiert aber zu nichts Renderbarem → **400** mit deutschem Grund. Sonst wird roh gespeichert und **beim Lesen** bereinigt.
- **Read-Seite asymmetrisch**: `get_card` (und die volle Karten-Response von create/update) führen die Felder; **`list_cards` bewusst nicht** — ein 30-kB-SVG pro Zeile hat in einer Listen-Response nichts verloren. Bitte nicht „der Vollständigkeit halber" nachrüsten.
- **Der eigentliche Wert liegt im Tool-Doc**: [docs/card_svg_authoring.md](card_svg_authoring.md) ist die Zeichen-Konvention. Ohne sie produziert der Agent hübsches, aber auf dem Handy unlesbares SVG. Bitte als Tool-Doc-Inhalt verankern (siehe unten).

## Die zwei Felder — Kontrakt

Voller Kontrakt in [docs/card_api_contract.md](card_api_contract.md). Hier das Wrap-Wesentliche:

### An `create_card` → `POST /api/cards`
- `front_svg` — String oder `null`, optional. SVG-Markup der Figur zur **Frage**.
- `back_svg` — String oder `null`, optional. SVG-Markup der Erklär-Figur zur **Lösung** (erscheint beim Aufdecken).
- **Eine Abbildung ersetzt kein Pflichtfeld**: die Typ-Validierung ist unverändert (`atomic` braucht front+back oder cloze_text, `generative` braucht prompt). Eine reine Bild-Karte ohne Text wäre nicht abfragbar — Absicht.
- → **201** + volle Karte, die beiden Felder **bereinigt** in der Response.

### An `update_card` → `PATCH /api/cards/<id>`
- Dieselben zwei Felder, dieselbe Validierung.
- **Leeren geht**: `null` **oder** `""` setzt die Spalte auf NULL.
- Key weglassen = unberührt (wie bei allen anderen Feldern).

### Fehler-Verhalten (400)
Ist ein Feld gesetzt und non-blank, liefert die Bereinigung aber nichts Renderbares:

> Feld 'front_svg' enthält kein renderbares SVG. Wahrscheinlich: über 100 kB, kein `<svg>`-Wurzelelement oder nur nicht-erlaubte Elemente.

Non-String (Zahl, Objekt, Liste) → **400** „Feld 'front_svg' muss Text oder null sein." Auth-Fehler unverändert **503**/**401**.

**Wichtig für die Fehler-Erwartung des Agenten**: ein SVG mit *teilweise* verbotenen Elementen wird **angenommen** — die verbotenen Teile fallen beim Lesen weg, der Rest rendert. Die 400 kommt nur, wenn **nichts** übrig bleibt.

## Sanitize-Modell (damit der Wrapper nichts Falsches verspricht)

**Roh speichern, beim Lesen sanitisieren.** Die Spalte hält exakt das, was der Agent geschickt hat; **jede** Lese-Fläche (Web-API, MCP, künftige iOS-App) bekommt die bereinigte Fassung — es gibt genau eine Sanitize-Stelle (`Card.to_dict()`), keine Fläche kann sie vergessen.

Für den MCP heißt das: **was `get_card` zurückgibt, ist exakt das, was die Lern-Oberfläche zeigt.** Der Agent kann sein Write direkt verifizieren, indem er die Karte zurückliest — verbotene Elemente sind dort schon weg.

Die Allow-List ist eng und **arbeitet über Weglassen**: `<script>`, `<style>`, `<foreignObject>`, `<use>`, `<image>`, `<a>`, `<animate>`, alle `on*`-Handler und `class` sind nicht drin und werden es nie. Vollständige Liste in der Zeichen-Konvention.

## Empfehlung fürs converter-mcp

1. **`create_card`** um `front_svg: str|None = None` und `back_svg: str|None = None` erweitern (durchreichen, nicht validieren — CONVERTER validiert).
2. **`update_card`** identisch erweitern. Wichtig: der Wrapper muss zwischen „Key nicht gesetzt" (unberührt) und „`None` gesetzt" (leeren) unterscheiden können — wie bei den anderen optionalen Feldern.
3. **`list_cards` NICHT anfassen.** Die Felder fehlen dort by design.
4. **Tool-Doc-Doktrin**: in beide Tool-Beschreibungen den Verweis auf [docs/card_svg_authoring.md](card_svg_authoring.md) aufnehmen, plus die drei Kern-Regeln inline, damit sie am Tool kleben und nicht im Gedächtnis des Agenten:
   > *„Optionale Abbildung als SVG-Markup. `viewBox` ist Pflicht (keine px-Maße), `<title>` als erstes Kind, **breit** zeichnen (~16:10 bis 3:2 — hoch gezeichnete Figuren werden auf dem Handy auf Briefmarkengröße skaliert), Schrift mindestens viewBox-Breite ÷ 30, dunkle Farben (die Fläche ist immer weiß, auch im Dark Mode), max. 100 kB, keine externen Referenzen (`<image>`/`<use>`/Web-Fonts werden still entfernt). Eine Figur nur, wenn sie die Karte trägt — Räumliches, Sequenz, Vergleich — nicht als Deko. Volle Konvention: docs/card_svg_authoring.md."*

## End-to-end-Beweis = Koordinator-Scope

Nach dem Wrap auf einer **Wegwerf-Karte** beweisen:
1. `create_card` mit `front_svg` + `back_svg` (Beispiel-SVG aus der Zeichen-Konvention) → **201**.
2. `get_card` → beide Felder da und bereinigt; `list_cards` → beide Felder **nicht** da.
3. `update_card` mit `front_svg: null` → Feld geleert.
4. `create_card` mit `front_svg: "<p>kein svg</p>"` → **400** mit dem obigen Grund.
5. Wegwerf-Karte danach in der „Lernen"-UI löschen (Agent löscht nicht — unverändert).

---

*CONVERTER-Seite: CARD-SVG fertig + getestet (+28 Tests: 20 Sanitizer, 8 API/Schema), Live-Smoke in `/review` (Dark + Light + Mobile) gefahren, committet. **Schema-Touch** (2 Spalten, Auto-Migration), **kein neuer Token, kein neuer Dep, kein neuer Endpoint**. Deploy: Mac push → Mintbox `git pull` + `docker compose up -d --build`. Geschwister-Briefe: [docs/converter_mcp_tag_cleanup_brief.md](converter_mcp_tag_cleanup_brief.md), [docs/converter_mcp_lern_group_brief.md](converter_mcp_lern_group_brief.md).*
