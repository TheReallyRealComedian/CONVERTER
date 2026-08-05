# Developer-Brief an den CONVERTER-iOS-Agenten — CARD-MD (Markdown im Kartentext)

> **An**: CONVERTER_iOS-Entwickler (`~/CODE/CONVERTER_iOS`).
> **Von**: CONVERTER-Master, 2026-08-05.
> **Worum**: Der Karten-Agent schreibt Markdown, die App zeigt es als Literal — `**Drill mit Molekül-Ionen:**`, `- Ga³⁺ + SO₃²⁻ → …`. Das Web hat das mit CARD-MD gelöst; die App zeigt weiter Sternchen.
> **Vorgänger**: [converter_ios_card_svg_brief.md](converter_ios_card_svg_brief.md) · [converter_ios_learn_more_brief.md](converter_ios_learn_more_brief.md) · [converter_ios_learn_settings_brief.md](converter_ios_learn_settings_brief.md) — alle drei portiert.

## TL;DR

- **Rein clientseitig.** Kein API-Change, kein Feld, kein Endpunkt. Der Text kommt heute schon so, wie er gebraucht wird.
- **Der Präzedenzfall steht in eurem Code**: `ClozeRenderer.attributed(_:reveal:)` ([ReviewSessionView.swift:403](../../CONVERTER_iOS/Sources/ReviewSessionView.swift)) scannt `{{…}}` und baut eine `AttributedString`. Genau diese Funktion wächst zum Ein-Pass-Renderer, der Cloze **und** Auszeichnung kennt — das Web hat es exakt so gemacht.
- **Vier Aufrufstellen**: `questionText` (prompt · cloze · front) und `answerSection` (cloze · back).
- ⚠️ **Die Grammatik ist absichtlich schmaler als CommonMark.** Sie muss zeichengenau der Web-Fassung entsprechen, sonst zeigen zwei Oberflächen dieselbe Karte verschieden.

---

## Eine Annahme, die ich selbst hatte und die falsch war

Im Web-Sprint stand als Begründung, `AttributedString(markdown:)` bzw. eine Markdown-Bibliothek würde `_text_` als Kursiv lesen und damit Formeln wie `μ_max · S/(K_S + S)` zerreißen.

**Das habe ich nachgemessen, und es stimmt nicht.** Apples Parser über alle **324 Textfelder** von Olis 206 Prod-Karten, mit `interpretedSyntax: .inlineOnlyPreservingWhitespace`:

```
Felder: 324 | Unterstrich-Verluste: 0 | Parser-Fehler: 0
```

Grund: CommonMark verbietet Emphase mit `_` **innerhalb** eines Wortes — genau, um `snake_case` zu schützen. `μ_max`, `K_S`, `Δ_Sub E`, `f_mech(x)` kommen unverändert durch. Die Sorge war unbegründet.

**Trotzdem ist Apples Parser hier die falsche Wahl** — aus einem anderen, gemessenen Grund.

## Warum die Web-Grammatik portiert wird und nicht CommonMark

**1. Apples Parser weicht auf echten Karten vom Web ab.** Gemessen: **6 Felder** enthalten literale `\"`, die CommonMark als Escape auflöst. Das Web lässt sie stehen. Dieselbe Karte, zwei Oberflächen, zwei Texte. (Diese sechs Felder sind ein *Datenfehler*, siehe unten — aber sie belegen die Divergenz.)

**2. CommonMark bringt Muster mit, die hier schaden können.** Links, Code-Spans, Roh-HTML, Emphase an Wortgrenzen mit `_`. Der Korpus nutzt heute keins davon (Backticks: **0** Vorkommen). Was heute nicht vorkommt, kann morgen vorkommen — und dann entscheidet auf iOS ein fremder Parser, was mit einer Formel passiert, während das Web seiner engen Liste folgt.

**3. Die Web-Grammatik hat zwei bewusste Abweichungen von CommonMark**, die ein Standardparser nicht kennt:
   - **Whitespace-Regel am Marker**: ein Opener darf rechts von sich keinen Whitespace haben, ein Closer links keinen. Deshalb bleibt `Fläche = 2 * 3 * 4 cm` Text und wird nicht kursiv.
   - **Eingerückte Fortsetzungszeilen** gehören zum vorherigen Listenpunkt. Der Karten-Agent schreibt Rechnungen eingerückt unter die Regel; ohne diese Regel zerfällt die Aufzählung in Ein-Punkt-Listen. Gemessen: **15 solche Zeilen in 6 Feldern**.

**4. Lists löst Apples Parser ohnehin nicht.** `inlineOnlyPreservingWhitespace` macht keine Listen. Der schwierigere Teil bliebe also Handarbeit, während der leichte Teil Divergenz einkauft.

**Fazit**: portiere die Grammatik aus [static/js/card_markup.js](../static/js/card_markup.js). Sie ist ~200 Zeilen, vollständig kommentiert, und jede Regel trägt ihre Begründung.

## Die Grammatik, verbindlich

| Muster | Wird zu | Vorkommen im Korpus |
|---|---|---|
| `**fett**` | fette Schrift | 68 Felder |
| `*kursiv*` | kursive Schrift | 25 Felder |
| Zeile beginnt mit `- ` oder `* ` | Aufzählungspunkt | 31 Felder |
| eingerückte Folgezeile | gehört in den **vorherigen** Punkt | 15 Zeilen in 6 Feldern |
| `{{…}}` | Cloze — heutiges Verhalten unverändert | 88 Felder |

**Nicht unterstützt, und das ist Absicht**: `_…_` (bleibt Literal) · Backticks · Überschriften · Links · Tabellen · Bilder.

**Weitere Regeln**, alle im Web verifiziert:
- Verschachtelung `**… *so* …**` funktioniert und kommt real vor.
- Unvollständige Marker bleiben Literal. `****` bleibt `****`, ein einzelnes `*` bleibt Text.
- Der Cloze-Modus ist **feldgebunden**: nur `cloze_text` schaltet ihn ein. In `front`/`back`/`prompt` bleibt `{{…}}` Literal — im Korpus kommt es dort ohnehin nicht vor.

## Der iOS-eigene Teil: Aufzählungen

Das Web hat gemessen, dass der **hängende Einzug** die Entscheidung trägt: bei schmaler Breite brechen Listenpunkte um, und die Fortsetzung muss unter dem *Text* stehen, nicht unter dem Aufzählungspunkt. Andernfalls klebt sie am linken Rand und ist nicht mehr als Teil ihres Punkts lesbar (gemessen 51 px gegen 28 px bei 311 px Textbreite).

Eine einzelne `Text`-View mit `AttributedString` kann das nicht — `AttributedString` trägt keinen hängenden Einzug. Naheliegend wäre, den Kartentext in Blöcke zu zerlegen und Listen als eigene Zeilen-Views zu setzen (Aufzählungspunkt und Text als getrennte Views, an der ersten Textgrundlinie ausgerichtet). **Deine Entscheidung** — du kennst SwiftUI; die Anforderung ist der hängende Einzug bei Umbruch, nicht eine bestimmte View-Struktur.

⚠️ Beim Web hat genau hier der Live-Smoke etwas gefangen, das die isolierte Messung nicht zeigte: eine CSS-Reset-Regel hatte die Aufzählungspunkte unsichtbar gemacht, obwohl die Einrückung stimmte. Sieh dir die Liste **im laufenden App-Kontext** an, nicht nur in einer Preview.

## Abnahme

Ein Durchstich an echten Karten, hell **und** dunkel, iPhone-Breite **und** iPad:

- **Karte 191** (Verhältnisformeln): Fett gesetzt statt sichtbar · die Aufzählung ist eine Aufzählung mit Punkten · die Fortsetzungszeilen (`Ga³⁺ + SO₃²⁻ → Ga₂(SO₃)₃`) sitzen eingerückt **in** ihrem Punkt.
- **Karten 22, 32, 208** (Formeln): zeichengleich. `μ = μ_max · S/(K_S + S)` · `dm/dt = (D·A/h)·(C_s − C)` · `Δ_r E = Δ_Sub E + Δ_Diss E + Δ_I E + Δ_Ea E + Δ_G E`. Kein Unterstrich verloren, nichts kursiv geworden.
- **Karte 195**: Cloze **und** `**Kation**` im selben Feld — der Ein-Pass-Fall.
- Mehrere Karten **durchbewerten**, nicht nur ansehen: das prüft den Kartenwechsel mit.

## Ausdrücklich nicht in diesem Brief

- **Mathematik-Rendering.** Karten zeigen kein KaTeX/LaTeX — das gilt auf beiden Oberflächen gleich und ist als eigenes Backlog-Item notiert.
- **Die fünf Karten mit literalen `\n` und `\"`** (3, 4, 7 und zwei weitere; 15 bzw. 18 Vorkommen). Das ist ein **Datenfehler aus dem Schreibpfad**, kein Renderproblem: die Felder enthalten die zwei Zeichen Backslash-n statt eines Zeilenumbruchs und haben **null** echte Umbrüche. Weder Web noch App sollen das beim Rendern reparieren — sonst kaschiert man einen Ingest-Fehler, der weiter Karten produziert. Eigenes Item.
