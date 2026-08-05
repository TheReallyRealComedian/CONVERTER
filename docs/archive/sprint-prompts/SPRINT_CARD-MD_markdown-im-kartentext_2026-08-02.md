# SPRINT CARD-MD — Markdown im Kartentext rendern statt anzeigen

**Größe**: S (2 Phasen) · **Datum**: 2026-08-02

## Warum

Der Karten-Agent schreibt Markdown, das Review zeigt es als Literal: `**Drill mit Molekül-Ionen:**`, `*(Sulfit-Index 3 → Klammer)*`, `- Ga³⁺ + SO₃²⁻ → …`. Bei Chemie-Karten, wo ohnehin Indizes und Pfeile im Text stehen, sind die Sternchen besonders störend.

Das ist **kein Versehen**, sondern die CARD-SVG-Doktrin: Kartentext läuft über `textContent`, die zwei Figuren-Container sind die **einzige** `innerHTML`-Stelle. Diese Doktrin bleibt — der Fix darf keine zweite `innerHTML`-Senke aufmachen.

## Gegroundeter Ist-Zustand (Master — gemessen, nicht neu herleiten)

**Der Präzedenzfall existiert schon**: `renderCloze` ([static/js/review.js:109](../../../static/js/review.js:109)) erkennt `{{…}}` im Kartentext und baut daraus **DOM-Knoten** (`createTextNode` + `<span>`), nie `innerHTML`. Genau dieses Muster wird erweitert.

**Gemessen an Olis 206 Prod-Karten (324 Textfelder):**

| Muster | Felder | Konsequenz |
|---|---:|---|
| `**fett**` | **68** | rendern |
| `*kursiv*` | **25** | rendern |
| Listenzeile (`- ` / `* ` am Zeilenanfang) | **31** | rendern |
| `{{cloze}}` | 88 | muss weiter funktionieren |
| mehrzeilig | 53 | `white-space: pre-wrap` ist bereits gesetzt |
| Backticks | **0** | **nicht** unterstützen |
| `_unterstrich_` | 5 | ⚠️ **NICHT** als Kursiv lesen — siehe unten |

⚠️ **Der wichtigste Einzelbefund**: Alle fünf `_…_`-Vorkommen sind **Tiefstellungen in Formeln**, keine Auszeichnung — `μ_max · S/(K_S + S)` (Monod-Kinetik), `C_s`, `f_mech(x)`, `Δ_Sub E`, `Δ_r E`. Würde `_text_` als Markdown-Kursiv interpretiert, zerrisse es genau die naturwissenschaftlichen Formeln, für die die Karten da sind. **`_` bleibt Literal.**

Die zwei Felder mit vermeintlich ungepaarten Sternchen sind Fehlalarme einer groben Regex (verschachteltes `**… *einer* …**`), keine Chemie-Notation. Einzel-`*` ist in diesem Korpus also unbedenklich.

## Gesperrte Entscheidungen

- **Keine zweite `innerHTML`-Senke.** Der Renderer baut DOM-Knoten, wie `renderCloze` es tut. Das ist nicht Vorsicht, das ist der Grund, warum der Kartentext heute sicher ist: er kommt roh vom Agenten.
- **Unterstützt wird genau**: `**fett**` → `<strong>`, `*kursiv*` → `<em>`, Listenzeilen → Aufzählung.
- **Nicht unterstützt**: `_…_`, Backticks, Überschriften, Links, Tabellen, Bilder. Kartentext ist kurz und kein Dokument; jedes zusätzliche Muster ist eine weitere Chance, eine Formel zu zerreißen.

---

# Phase 1 — Der Renderer

## 1.1 Wo er hingehört

Ein eigener, **purer** Renderer, der einen Zielknoten und einen Text nimmt und DOM-Knoten anhängt — dieselbe Signatur-Idee wie `renderCloze`. `renderCloze` geht darin auf: **ein** Durchlauf, der Cloze **und** Auszeichnung kennt, statt zwei Pässe, die sich gegenseitig die Offsets verschieben.

Er ersetzt die `textContent`-Zuweisungen für die Kartenfelder (`front`, `back`, `prompt`, `cloze_text`) in `renderCard`. **Alle anderen `textContent`-Stellen bleiben unangetastet** — Badges, Zähler, Fortschritt, Fehlermeldungen sind keine Agenten-Eingabe.

## 1.2 Was er können muss

- `**fett**` und `*kursiv*`, auch **verschachtelt** (`**… *so* …**` kommt real vor).
- Zeilen, die mit `- ` oder `* ` beginnen, werden zu einer Aufzählung. ⚠️ `white-space: pre-wrap` ist auf den Textflächen gesetzt ([style.css:2071](../../../static/css/style.css:2071) und :2126) — prüf, ob eine echte `<ul>` damit doppelte Abstände erzeugt, und entscheide begründet zwischen `<ul>` und einer leichteren Lösung. Miss es, statt zu raten.
- **Zusammenspiel mit Cloze**: `{{…}}` behält sein heutiges Verhalten (Front: `…`-Kasten, Back: hervorgehobene Antwort). Prüf am echten Korpus, ob Kombinationen wie `**{{Antwort}}**` oder `{{**Antwort**}}` vorkommen, und leg das Verhalten für beide fest — keines darf etwas zerreißen.
- **Unvollständige Marker bleiben Literal.** Ein einzelnes `*` ohne Partner ist Text, kein Fehler.

## 1.3 Tests

Die Suite rendert kein JS — trotzdem ist der Renderer **pur** und damit testbar, wenn du ihn so schneidest. Falls das im aktuellen Aufbau nicht ohne Umbau geht, sag es im Bericht und begründe, warum der Live-Smoke der Gate ist; bau **keinen** Test-Harness für JS in diesem Sprint.

Fälle, die abgedeckt gehören: fett · kursiv · verschachtelt · Liste · Cloze allein · Cloze mit Auszeichnung · **`μ_max · S/(K_S + S)` bleibt zeichengleich** · einzelnes `*` bleibt Literal · leerer Text.

## Stop
`pytest tests/` grün (Baseline **861**). **Commit + Push** `feat(CARD-MD): Markdown-Renderer für Kartentext (P1)`. Dann warten.

---

# Phase 2 — Verdrahtung, Smoke, Wrap

## 2.1 Live-Smoke — Pflicht, und zwar an echten Karten

Der Gate dieses Sprints. Nimm die Karte aus dem Screenshot (Chemie-Sammlung, „Drill mit Molekül-Ionen", Verhältnisformeln) **und** eine Formel-Karte mit `_`-Tiefstellungen (Monod-Kinetik, `μ_max · S/(K_S + S)`).

Zu belegen: Fett und Kursiv sind gesetzt statt sichtbar · die Aufzählung ist eine Aufzählung · **die Formel ist zeichengleich zu vorher** · Cloze funktioniert vorwärts und rückwärts · Dark **und** Light · 375 px Breite.

⚠️ Wegwerf-Instanz mit eigener DB. Olis Prod-Karten werden **nicht** angefasst.

## 2.2 Wrap

- **CLAUDE.md**: der CARD-SVG-Bullet sagt heute „Kartentext bleibt `textContent`". Das stimmt nach diesem Sprint nur noch dem Geist nach — zieh es nach: DOM-Knoten statt `innerHTML` ist weiterhin die Regel, aber der Text läuft jetzt durch einen Renderer. Nenn die Musterliste **und** die Begründung für `_`.
- **STATUS.md** + **BACKLOG.md** (Bullet-Guard).
- **Im Bericht benennen**: wie du `<ul>` gegen `white-space: pre-wrap` gelöst hast, und was der Korpus zu `**{{…}}**` gesagt hat.

## Nicht-Ziele

- **Kein iOS.** Die App hat dasselbe Problem (`Text(card.front ?? "")`) und mit `ClozeRenderer.attributed` denselben Präzedenzfall — das wird ein eigener Brief, sobald das Web steht.
- **Kein** Anfassen der Figuren-Container, des Sanitizers oder der Karten-API.
- **Keine** Markdown-Bibliothek einbinden. Vier Muster rechtfertigen keine Dependency, und eine Allzweck-Bibliothek brächte genau die Muster mit, die hier schaden.
