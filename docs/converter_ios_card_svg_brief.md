# Developer-Brief an den CONVERTER-iOS-Agenten — CARD-SVG (Abbildungen auf Lernkarten)

> **An**: CONVERTER_iOS-Entwickler (`~/CODE/CONVERTER_iOS`).
> **Von**: CONVERTER-Master, 2026-07-26.
> **Worum**: Lernkarten tragen jetzt **Abbildungen** — SVG-Markup in zwei neuen Feldern. Das Web zeigt sie, die App noch nicht. Oli lernt überwiegend mobil, und Abbildungen waren der Grund für das ganze Feature: **ohne diesen Port sieht er sie dort nie.**
> **Umfang**: **nur die Figuren.** „Mehr lernen" (LEARN-MORE) ist bewusst nicht Teil dieses Briefs.

## TL;DR

- **Nichts an der App ist kaputt.** Die Backend-Änderung war rein additiv; `LearnCard`/`ReviewState` nutzen explizite `CodingKeys`, Swifts `Codable` ignoriert unbekannte Schlüssel. Die Felder kommen bereits mit — sie werden nur verworfen.
- **Zwei neue Felder** auf der vollen Karte: `front_svg` und `back_svg`, Typ **`String?`**.
- **Das SVG ist bereits bereinigt.** CONVERTER sanitisiert an **einer** autoritativen Stelle (`Card.to_dict()`) gegen eine enge Allow-List. Die App **filtert nicht nach** und baut **keine** eigene Prüfung.
- **SwiftUI rendert SVG nicht nativ** → `WKWebView` mit dünnem HTML-Wrapper (Details unten). Das ist die einzige echte Bauarbeit.
- **Fläche fest weiß in beiden Color Schemes** — nicht verhandelbar, sonst verschwinden Figuren im Dark Mode.
- **Kein Schreibpfad.** Figuren schreibt ausschließlich der Karten-Agent. Die App zeigt nur.

## 1. Die zwei Felder

| Feld | Typ | Bedeutung |
|---|---|---|
| `front_svg` | `String?` | Abbildung zur **Frage** — erscheint mit der Vorderseite |
| `back_svg` | `String?` | Abbildung zur **Lösung** — erscheint beim Aufdecken |

`null`, wenn keine Figur hinterlegt ist **oder** wenn nach der Bereinigung nichts Renderbares übrig blieb. Beide Karten-Typen können sie tragen (atomar, Cloze, generativ) — sie sind unabhängig von `front`/`back`/`cloze_text`/`prompt`.

**Wo sie kommen:**

- ✅ `GET /api/review-state` → `due_cards[]` (volle Karten — **hier braucht ihr sie**)
- ✅ `GET /api/cards/<id>` (volle Karte)
- ❌ `GET /api/cards` (Listen-Endpoint) — führt sie **bewusst nicht**: eine 30-kB-Figur pro Zeile hat in einer Listen-Response nichts verloren. Falls eine Listenansicht Figuren zeigen soll, muss sie die volle Karte nachladen.

## 2. Die Swift-Änderung

`LearnCard` in [Sources/LearnModels.swift:9](../../CONVERTER_iOS/Sources/LearnModels.swift) bekommt zwei optionale Properties plus die zwei `CodingKeys`-Zeilen:

```swift
let frontSvg: String?
let backSvg: String?
// in CodingKeys:
case frontSvg = "front_svg"
case backSvg  = "back_svg"
```

Mehr nicht. `ReviewState` bleibt unverändert — `due_cards` dekodiert die neuen Properties automatisch mit. Ein bequemes `var hasFigure: Bool` o. ä. ist Geschmackssache.

## 3. Sicherheits-Kontrakt — bitte genau so lesen

Das SVG kommt **server-seitig bereinigt** an, gegen eine enge Allow-List (`services/svg_sanitize.py`). Entfernt sind unter anderem `script`, alle `on*`-Handler, `foreignObject`, `use`, `image`, `a`, `animate`/`set`, `style`, und jede `url(...)`-Referenz, die nicht auf ein lokales Fragment zeigt.

Zwei Konsequenzen:

1. **Nicht nachfiltern.** Es gibt genau eine Sanitize-Stelle; eine zweite in der App würde nur divergieren und irgendwann korrekte Figuren zerstören.
2. **Trotzdem nicht als beliebiges HTML behandeln.** Es ist SVG-Markup aus einer Allow-List — es lädt nichts nach und führt nichts aus. Konfiguriert das `WKWebView` entsprechend defensiv (unten), aber baut keine eigene Parser-Logik.

Es gibt **keine externen Referenzen** — keine Bilder, keine Web-Fonts, keine Netzwerk-Ladungen. Jede Figur ist selbsttragend. Offline-Rendering funktioniert damit ohne Sonderbehandlung.

## 4. Rendering

SwiftUI kann SVG nicht. Empfohlen: **`WKWebView`** mit einem minimalen HTML-Wrapper um das gelieferte Markup. (Eine SVG-Library wäre die Alternative — dann liegt die Entscheidung bei euch; der Rest dieses Abschnitts gilt sinngemäß trotzdem.)

**Anforderungen an den Wrapper:**

- **Hintergrund fest weiß**, unabhängig vom Color Scheme. Gesperrte Produkt-Entscheidung: die Figur sitzt wie eine Lehrbuch-Abbildung auf Papier, der Agent setzt dunkle Farben **hart**. Auf dunklem Grund wären sie unsichtbar. Das Web macht es genauso (`.review-figure { background: #ffffff }`) — es gibt **keinen** `currentColor`-Vertrag.
- **`viewBox` trägt die Skalierung.** CSS im Wrapper: `svg { width: 100%; height: auto; display: block; }`. Nichts an der Größe hart setzen.
- **Höhe an den Inhalt binden.** Die intrinsische Höhe ergibt sich aus `viewBox`-Verhältnis × verfügbarer Breite; die View-Höhe muss dem folgen (Content-Size beobachten oder das Seitenverhältnis aus der `viewBox` lesen und selbst rechnen).
- **Nicht interaktiv**: Scrollen aus, Bounce aus, Textauswahl aus, Zoom aus, keine Navigation zulassen. Die Figur ist Inhalt, kein Browser.
- **Kein Rand-Overhead**: `margin: 0`, kein Body-Padding — das Padding gehört in die SwiftUI-Fläche, nicht ins HTML.

**Größen-Verhalten — gemessen im CARD-SVG-Smoke, nicht geraten:**

| Kontext | Was begrenzt | Folge |
|---|---|---|
| **iPhone, 375 pt breit** | die **Breite** | Eine 3:2-Figur wird ~117 pt hoch. Sie wird **nie abgeschnitten**, nur kleiner. |
| Desktop-Web, 1280×720 | ein CSS-Deckel `max-height: 32vh` | letterboxed, ebenfalls nichts abgeschnitten |

Praktische Folge für euch: **gebt der Figur die volle verfügbare Breite.** Jeder zusätzliche horizontale Inset verkleinert sie doppelt — direkt und über das Seitenverhältnis. Die Zeichen-Konvention verpflichtet den Agenten deshalb auf **breite, flache** Figuren.

**Höhen-Deckel auch auf iOS setzen.** Nach dem Aufdecken sind **beide** Figuren gleichzeitig sichtbar. Im Web lag die Karte bei 40vh und zwei Figuren bei 1040 px Höhe (Viewport 720) — die Bewertungs-Buttons lagen unter der Falz; mit 32vh sind es 563 px. Sorgt dafür, dass die Bewertungs-Buttons ohne Scrollen erreichbar bleiben, oder dass der Scroll offensichtlich ist.

## 5. Zwei Fallen aus dem Web-Port

1. **Stale-Figur beim Kartenwechsel.** Beim Wechsel auf die nächste Karte muss die **Rückseiten-Figur explizit zurückgesetzt** werden. Sonst blitzt beim Aufdecken kurz die Figur der *vorherigen* Karte auf. Das Web hat genau diesen Bug gehabt; im SwiftUI-Port ist die identische Falle ein nicht zurückgesetzter `@State` oder ein wiederverwendetes `WKWebView`.
2. **Nur die Figuren-Fläche rendert Markup.** Der Kartentext bleibt normaler SwiftUI-Text. Zieht die Figuren nicht in den Text-Rendering-Pfad.

## 6. Barrierefreiheit — geschenkt

Die Zeichen-Konvention verpflichtet den Agenten, als **erstes Kind** des `<svg>` ein `<title>` zu setzen, z. B.:

```xml
<title>DSP-Flow: Bioreaktor → Capture → Eluat</title>
```

Zieht es heraus und nutzt es als `accessibilityLabel` der Figuren-View. Ohne das ist die Figur für VoiceOver stumm. Falls ihr eine sichtbare Bildunterschrift wollt, ist es auch dafür der richtige Text.

## 7. Payload

`GET /api/review-state` liefert volle Karten. Figuren blähen die Antwort also auf — pro Figur bis `MAX_CARD_SVG_BYTES = 100 000` Bytes, realistisch 1–30 kB. **Figurenlose Karten kosten null** (das Feld ist dann `null`). Dokumentierte Eigenschaft, heute unkritisch; wenn Oli irgendwann sehr viele Figuren hat, ist das die Stelle, die zuerst spürbar wird.

## 8. Referenzen

- **Zeichen-Konvention + vollständige Allow-List**: [docs/card_svg_authoring.md](card_svg_authoring.md) — lest mindestens die Regeln 1–5, sie erklären, warum die Figuren so aussehen, wie sie aussehen.
- **API-Kontrakt**: [docs/card_api_contract.md](card_api_contract.md)
- **Architektur-Notiz**: `CLAUDE.md`, Bullet *Abbildungen auf Lernkarten (CARD-SVG)*

## 9. Fertig ist es, wenn

1. Eine Karte mit `front_svg` zeigt die Figur über der Frage; nach dem Aufdecken erscheint `back_svg`.
2. Die nächste Karte **ohne** Figur zeigt **keine** Reste (der Stale-Test).
3. Cloze- und generative Karten verhalten sich unverändert.
4. Dark **und** Light geprüft — die Fläche ist in beiden weiß, die Figur lesbar.
5. Bewertungs-Buttons bleiben mit zwei Figuren erreichbar.
6. VoiceOver liest den `<title>`-Text.

## Nicht in diesem Brief

- **Kein Schreibpfad** — die App legt keine Figuren an und bearbeitet keine. Autor ist ausschließlich der Karten-Agent.
- **Kein Zoom/Lightbox** — steht im CONVERTER-Backlog als bedarfsgetriebenes Item. Wenn sich beim echten Lernen herausstellt, dass Figuren auf dem Handy zu klein sind, ist das die nächste Frage — bitte dann melden statt vorbauen.
- **Kein LEARN-MORE** — das gestufte Nachladen (`remaining_today`, `next_ahead`, `?uncapped=1`, `?ahead=<n>`) ist bewusst ein eigener, späterer Brief.
