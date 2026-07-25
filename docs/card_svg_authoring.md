# Abbildungen auf Lernkarten — Zeichen-Konvention für den Agenten

> **Für wen**: den karten-schreibenden Agenten (`create_card`/`update_card` über `CARD_TOKEN`). Dieses Dokument ist als **Tool-Doc-Inhalt** gedacht — der Agent soll es kennen, bevor er die erste Figur zeichnet.
> **Warum**: Abbildungen tragen Biotech-Stoff (Antikörper-Formate, Chromatographie-Flows, DSP-Pipelines) besser als Fließtext. Ohne Konvention entsteht hübsches, aber auf dem Handy unlesbares — oder still zerstrippertes — SVG.

## Die zwei Felder

`front_svg` und `back_svg` an `POST /api/cards` und `PATCH /api/cards/<id>`. Typ `string|null`, beide optional. SVG-Markup als Text; es gibt **keinen** Asset-Store, keinen Upload, keine Bild-Dateien.

- **`front_svg`** — die Figur zur Frage („Was passiert in Schritt 3?" mit dem Flow daneben).
- **`back_svg`** — die Erklär-Figur zur Lösung. Erscheint erst beim Aufdecken.
- Leeren: `null` oder `""` schicken.

**Eine Abbildung ersetzt kein Pflichtfeld.** Die Karte braucht weiterhin front+back / cloze_text / prompt — eine reine Bild-Karte ohne Text wäre nicht abfragbar. Das ist Absicht.

## Wann überhaupt eine Figur

Nur, wenn sie die Karte **trägt**: Räumliches, Sequenz, Vergleich, Zustands-Übergänge. Nicht als Deko auf jede Karte. Wenn der Satz die Sache genauso gut sagt, ist der Satz besser.

**Zeichnen**: schematische Flows, beschriftete Boxen, Achsen, Verläufe, Zustands-Übergänge.
**Nicht zeichnen**: Fotorealismus, Molekülstrukturen mit Anspruch auf chemische Korrektheit, dichte Tabellen (dafür ist der Kartentext da).

## Pflicht-Regeln

### 1. `viewBox` ist Pflicht, absolute px-Maße sind verboten

```xml
<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg">   <!-- richtig -->
<svg width="640px" height="300px">                                <!-- falsch -->
```

Die Fläche skaliert mit dem Gerät; `viewBox` trägt die Skalierung. Das CSS setzt `width: 100%; height: auto`. Ohne `viewBox` ist die Figur auf jeder Bildschirmbreite falsch groß.

### 2. `<title>` als **erstes Kind** des `<svg>`

```xml
<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg">
  <title>DSP-Flow: Bioreaktor → Capture → Eluat</title>
  …
```

Zwei Gründe: der Browser zeigt ihn als nativen Tooltip beim Hovern, und eine Figur ohne Titel ist für Screenreader stumm. Kostet eine Zeile. Der Tag ist erlaubt — er wird nur nicht automatisch erzeugt.

### 3. **Breit zeichnen**, nicht hoch

Seitenverhältnis **~16:10 bis 3:2** (z.B. `viewBox="0 0 320 200"` oder `0 0 320 150`). Nichts Hochformatiges.

Der Grund ist gemessen (Live-Smoke 2026-07-25), nicht geraten — **es bindet an beiden Enden eine andere Grenze**:

| Fläche | Was begrenzt | Effekt bei einer hohen Figur |
|---|---|---|
| **iPhone, 375 px breit** | die **Breite** (Figur wird ~117 px hoch bei 3:2) | Die Figur wird auf **Briefmarkengröße** herunterskaliert — Beschriftungen unlesbar. Sie wird **nicht** abgeschnitten. |
| **Desktop, 1280×720** | der CSS-Deckel `max-height: 32vh` (≈230 px) | Die Figur wird letterboxed (zentriert, leere Seitenränder). Nichts wird abgeschnitten. |

Der Deckel existiert, weil nach dem Aufdecken **beide** Figuren gleichzeitig sichtbar sind: bei 40vh war die Karte 1040 px hoch (Viewport 720) und die Bewertungs-Buttons lagen weit unter der Falz; mit 32vh sind es 563 px. Praktisch heißt das: **je höher die Figur, desto kleiner erscheint sie** — auf dem Handy über die Breite, am Desktop über den Deckel. Breit gezeichnet nutzt sie die verfügbare Fläche aus.

### 4. Mindest-Schriftgröße relativ zur viewBox

Bei einer viewBox-Breite von 320 sind **`font-size="11"` das Minimum**, 13–14 für Box-Beschriftungen, 12 mit `font-weight="700"` für die Überschrift. Faustregel: **mindestens viewBox-Breite ÷ 30**. Kleiner wird auf dem iPhone unleserlich.

### 5. Feste helle Fläche → dunkle Farben setzen

Die Figur sitzt wie eine Lehrbuch-Abbildung auf Papier: **weißer Hintergrund, unabhängig vom Theme** (auch im Dark Mode). Es gibt **keinen `currentColor`-Vertrag** — Farben werden hart gesetzt und es bricht nie.

- **Erwünscht**: dunkle Linien und Schrift (`#0f172a`, `#334155`), gesättigte Flächenfüllungen mit dunklem Rahmen (`fill="#dbeafe" stroke="#1d4ed8"`).
- **Vermeiden**: helle Töne auf Weiß (hellgrau auf weiß, Gelb auf weiß) — unsichtbar.

### 6. Größen-Deckel: 100 000 Bytes

`MAX_CARD_SVG_BYTES = 100_000`, gemessen an der **utf-8-kodierten** Länge. Darüber wird das Feld verworfen und der Write mit **400** abgelehnt. Ein sauberes Schema liegt bei 1–3 kB — der Deckel fängt nur Ausreißer (eingebettete Base64-Daten, tausende Pfad-Punkte).

### 7. Alles muss selbsttragend sein

**Externe Referenzen werden still entfernt.** Keine `<image>`, keine `<use>`, keine Web-Fonts, keine externen Paint-Server (`fill="url(https://…)"`). Für Schrift nur generische Familien: `font-family="sans-serif"`.

Lokale Referenzen innerhalb derselben Figur funktionieren: `fill="url(#meinGradient)"`, `marker-end="url(#pfeil)"` — sofern das Ziel im selben SVG als `<marker>`/`<linearGradient>`/`<radialGradient>` mit `id` definiert ist.

## Erlaubte Tags und Attribute

Die Allow-List ist bewusst eng. Was nicht drinsteht, wird **still entfernt** (der Sanitizer arbeitet über Weglassen, nicht über Sonderregeln).

| Tag | Erlaubte Attribute |
|---|---|
| `svg` | `viewBox`, `width`, `height`, `xmlns`, `preserveAspectRatio` |
| `g` | Präsentation* |
| `defs`, `title`, `desc` | — |
| `path` | `d` + Präsentation* |
| `rect` | `x`, `y`, `width`, `height`, `rx`, `ry` + Präsentation* |
| `circle` | `cx`, `cy`, `r` + Präsentation* |
| `ellipse` | `cx`, `cy`, `rx`, `ry` + Präsentation* |
| `line` | `x1`, `y1`, `x2`, `y2` + Präsentation* |
| `polyline`, `polygon` | `points` + Präsentation* |
| `text`, `tspan` | `x`, `y`, `dx`, `dy` + Präsentation* |
| `marker` | `id`, `viewBox`, `markerWidth`, `markerHeight`, `refX`, `refY`, `orient`, `markerUnits` + Präsentation* |
| `linearGradient` | `id`, `x1`, `y1`, `x2`, `y2`, `gradientUnits`, `gradientTransform` |
| `radialGradient` | `id`, `cx`, `cy`, `r`, `fx`, `fy`, `gradientUnits`, `gradientTransform` |
| `stop` | `offset`, `stop-color`, `stop-opacity` |

\* **Präsentations-Attribute**: `fill`, `stroke`, `stroke-width`, `stroke-linecap`, `stroke-dasharray`, `opacity`, `transform`, `font-size`, `font-family`, `font-weight`, `text-anchor`, `dominant-baseline`, `marker-start`, `marker-mid`, `marker-end`.

**Nicht erlaubt** (und nie erlaubt werdend): `<script>`, `<style>` (Tag **und** `style`-Attribut), `<foreignObject>`, `<use>`, `<image>`, `<a>`, `<animate>`/`<set>`, `<iframe>`/`<audio>`/`<video>`, alle `on*`-Handler.

**Auch nicht erlaubt: `class`.** Es gewährt nichts (CSS lässt sich ohnehin nicht mitliefern), kollidiert aber mit den App-Klassen — ein `class="hidden"` träfe genau die Klasse, mit der die Review-Oberfläche Figuren versteckt: unsichtbare Abbildung ohne auffindbare Ursache.

## Wenn der Write mit 400 abgelehnt wird

> Feld 'front_svg' enthält kein renderbares SVG. Wahrscheinlich: über 100 kB, kein `<svg>`-Wurzelelement oder nur nicht-erlaubte Elemente.

Die drei Ursachen in dieser Reihenfolge prüfen: Größe → fehlendes `<svg>`-Wurzelelement (z.B. nur ein Fragment geschickt) → alles Gezeichnete lag außerhalb der Allow-List.

**Wichtig**: Ein SVG, das *teilweise* verbotene Elemente enthält, wird **angenommen** — die verbotenen Teile fallen beim Lesen weg, der Rest rendert. Die 400 kommt nur, wenn **nichts** Renderbares übrig bleibt. Wer sichergehen will, liest die Karte nach dem Write mit `get_card` zurück: dort steht das Feld genau so, wie die Oberfläche es zeigen wird.

`list_cards` führt die Felder bewusst **nicht** (Listen-Response bleibt schlank) — nur `get_card` und die volle Karten-Response.

## Vorlage

```xml
<svg viewBox="0 0 320 160" xmlns="http://www.w3.org/2000/svg">
  <title>Protein-A-Capture: Bindung, Wäsche, Elution</title>
  <defs>
    <marker id="pfeil" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="#334155"/>
    </marker>
  </defs>
  <text x="160" y="20" text-anchor="middle" font-size="12" font-weight="700"
        font-family="sans-serif" fill="#0f172a">Capture-Schritt</text>
  <rect x="8" y="45" width="86" height="46" rx="8"
        fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>
  <text x="51" y="66" text-anchor="middle" font-size="13"
        font-family="sans-serif" fill="#0f172a">Beladen</text>
  <text x="51" y="82" text-anchor="middle" font-size="11"
        font-family="sans-serif" fill="#475569">Fc bindet</text>
  <path d="M96 68 H 128" stroke="#334155" stroke-width="2" marker-end="url(#pfeil)"/>
  <rect x="130" y="45" width="86" height="46" rx="8"
        fill="#f1f5f9" stroke="#64748b" stroke-width="2"/>
  <text x="173" y="72" text-anchor="middle" font-size="13"
        font-family="sans-serif" fill="#0f172a">Wäsche</text>
  <path d="M218 68 H 250" stroke="#334155" stroke-width="2" marker-end="url(#pfeil)"/>
  <rect x="252" y="45" width="60" height="46" rx="8"
        fill="#dcfce7" stroke="#15803d" stroke-width="2"/>
  <text x="282" y="72" text-anchor="middle" font-size="13"
        font-family="sans-serif" fill="#0f172a">Eluat</text>
  <text x="160" y="120" text-anchor="middle" font-size="11"
        font-family="sans-serif" fill="#64748b">pH-Shift löst die Bindung</text>
</svg>
```

2:1-Verhältnis, `<title>` zuerst, Schrift ab 11 bei viewBox-Breite 320, dunkle Farben auf hellem Grund, lokaler Marker, keine externe Referenz. ~1,4 kB.

---

*Server-Seite: [services/svg_sanitize.py](../services/svg_sanitize.py) (Allow-List + Sanitizer), Kontrakt: [docs/card_api_contract.md](card_api_contract.md), MCP-Brief: [docs/converter_mcp_card_svg_brief.md](converter_mcp_card_svg_brief.md).*
