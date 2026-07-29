# SPRINT LEARN-RATE — die Bewertungsknöpfe sagen, was sie meinen

**Größe**: S (2 Phasen) · **Datum**: 2026-07-29 · **Parallel dispatchbar** mit LEARN-TUNE (disjunkte Dateien: dieser Sprint fasst `templates/` + `static/`, LEARN-TUNE fasst `services/scheduler/` + `tests/`)

## Warum — das ist kein Kosmetik-Sprint

In 122 Bewertungen über drei Sessions ist **kein einziges Mal** „Wieder" gefallen: 69× Schwer, 53× Gut, 0× Wieder, 0× Leicht. Auf die Frage, was „Schwer" für ihn bedeutet, hat Oli geantwortet: **„kaum bis gar nicht gewusst."**

Das heißt: der Scheduler bekommt seit Wochen systematisch die falsche Nachricht. Wo FSRS ein `again` erwartet (= Vergessen, Lapse, Relearning-Schritt, Stabilität bricht ein), bekommt er ein `hard` (= bestanden, Stabilität wächst weiter). Gemessen gegen `fsrs==6.3.1`, Olis echte Rating-Abstände nachgestellt:

| erste Runde | Ergebnis nach 3 Runden |
|---|---|
| `again` / hard / hard | Stabilität 7,0 → **7 Tage** |
| `hard` / hard / hard | Stabilität 15,2 → **15 Tage** |

Der Beschriftungsfehler allein verdoppelt das Intervall. Und die „Ist-Retention" in der Statistik zählt `again` als Fehlschlag — bei null `again` meldet die App also **100 % Behaltensrate**, während Oli tatsächlich bei rund der Hälfte der Karten geblankt hat.

**Die Ursache ist das Label, nicht der Nutzer.** Die Reihe lautet heute `Wieder · Schwer · Gut · Leicht`. „Wieder" steht dort als Adverb ohne Bezugswort und liest sich nicht als „ich wusste es nicht". Wer die vier Knöpfe als *Schwierigkeitsskala* liest — und genau so sehen sie aus — wählt für „kaum gewusst" folgerichtig **Schwer**. Jeder hätte das so gemacht.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

- Die vier Knöpfe: [templates/review.html:99-102](templates/review.html). `good` trägt `c-btn--primary`.
- CSS: eine einzige Zeile, [static/css/style.css:2131](static/css/style.css) — `.review-rate-btn { flex: 1; min-width: 0; }`, die Knöpfe liegen in `.c-btn-row`.
- **Tastaturkürzel 1–4 existieren bereits** ([static/js/review.js:804-805](static/js/review.js)) und sind im UI **nirgends sichtbar**.
- Es gibt heute **keine** Erklärzeile, keinen Tooltip, keine Legende.
- Die Test-Suite rendert keine Templates → **Live-Smoke ist Pflicht**, `pytest` fängt hier nichts.

## Gesperrte Entscheidungen (Master, aus Olis Antwort)

1. **`again` heißt künftig „Nochmal".** Eindeutig temporal statt adverbial, und es ist die etablierte deutsche Anki-Konvention.
2. **Dauerhaft sichtbare Bedeutungszeile pro Knopf** — kein Tooltip, kein Hover, kein Aufklapper. Touch kennt kein Hover, und ausgerechnet hier darf niemand raten müssen.
3. **Das Kriterium wird als Frage vorangestellt, nicht als Gefühl.** Die Entscheidung ist zweistufig und muss auch so dastehen: erst *ob*, dann *wie mühsam*.
4. **Die Ziffern 1–4 werden sichtbar.** Sie existieren und sind unauffindbar.
5. `good` behält die Primary-Hervorhebung — es ist die erwartete Normalantwort.

---

# Phase 1 — Beschriftung, Legende, Smoke

## 1.1 Die Microcopy (Master gibt sie vor, wörtlich übernehmen)

Über der Knopfreihe **eine** Zeile:

> Wusstest du die Antwort? Nur wenn ja: wie mühsam?

Die vier Knöpfe, jeweils Label + Bedeutung + Ziffer:

| Ziffer | Label | Bedeutung |
|---|---|---|
| 1 | **Nochmal** | nicht gewusst |
| 2 | **Schwer** | gewusst, mühsam |
| 3 | **Gut** | gewusst |
| 4 | **Leicht** | sofort da |

Die Bedeutungen sind bewusst asymmetrisch: nur Knopf 1 beschreibt einen **Ausgang**, 2–4 beschreiben **Aufwand**. Genau diese Bruchkante ist die Botschaft — nicht glattziehen.

Keine Emojis. Labels bleiben einwortig (`c-btn`-Konvention, max 3 Wörter).

## 1.2 Umsetzung

- Label `Wieder` → `Nochmal` in [templates/review.html:99](templates/review.html).
- Bedeutungs- und Ziffernzeile als eigene Elemente **im** Knopf (nicht als Geschwister daneben — der ganze Knopf muss die Trefferfläche bleiben).
- Die Frage-Zeile über der Reihe, visuell ruhig (gedämpfte Textfarbe, kleiner als der Kartentext) — sie soll führen, nicht schreien.
- Styling über bestehende `--nm-*`-Tokens, keine neuen Hardcodes.
- `aria-label` pro Knopf, das Label **und** Bedeutung trägt, damit VoiceOver nicht nur „Nochmal" vorliest.
- `static/js/review.js` nur anfassen, falls die neue Struktur es erzwingt — die `data-rating`-Attribute und das Kürzel-Mapping bleiben **unverändert**.

## 1.3 Smoke (Pflicht — die Suite rendert kein JS und keine Templates)

Wegwerf-Instanz wie in den Vorgänger-Sprints (eigene DB im Scratchpad, **nicht** Prod).

| Fall | Erwartung |
|---|---|
| Karte aufdecken, Desktop | vier Knöpfe, jeder mit Label + Bedeutung + Ziffer, Frage-Zeile darüber |
| **Breite 375 px** | nichts bricht um, nichts wird abgeschnitten, alle vier bleiben nebeneinander und tippbar |
| Dark **und** Light | Bedeutungszeile in beiden lesbar (sie ist gedämpft — genau da reißt Kontrast zuerst) |
| Tasten 1–4 | bewerten unverändert wie vorher |
| Klick auf jeden der vier | sendet unverändert `again`/`hard`/`good`/`easy` |

⚠️ Die 375-px-Zeile ist die riskanteste: vier Knöpfe mit je zwei Textzeilen in einer `flex`-Reihe. Wenn es dort nicht trägt, ist die Umbruch-Entscheidung deine — berichte, was du gewählt hast und warum.

## Stop
`pytest tests/` grün (Baseline **762**), Smoke-Protokoll mit allen fünf Zeilen. **Commit + Push** `fix(LEARN-RATE): Bewertungsknöpfe benennen ihre Bedeutung (P1)`. Dann warten.

---

# Phase 2 — Wrap

- **CLAUDE.md**, Learning-Abschnitt: neuer Bullet **Rating-Semantik (LEARN-RATE)** — der Befund (0 von 122 `again`, „Schwer" als „nicht gewusst" benutzt), die gemessene Verdopplung des Intervalls, die zweistufige Frage als Doktrin, und dass die Suite das nicht abdeckt.
- **STATUS.md** + **BACKLOG.md** (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein).
- **Memory** zur übertragbaren Einsicht: *Rating-Skalen mit gemischter Semantik (ein Ausgang + drei Aufwandsstufen) müssen den Ausgang explizit benennen — ein einzelnes Adverb als Label kippt die Skala und vergiftet still die nachgelagerte Statistik.* Verlinken auf `[[reference_fsrs_learning_steps_default_trap]]`.
- **Zwei Folgen ausdrücklich in den Schlussbericht** (beides ist keine Arbeit in diesem Sprint, aber Oli muss es wissen):
  1. **Die Historie bleibt vergiftet.** 122 Bewertungen mit falscher Semantik werden nicht umgeschrieben. Die „Ist-Retention" korrigiert sich von selbst, aber erst über das 30-Tage-Fenster.
  2. **Die iOS-App trägt dieselben Labels** (`CardRating.label` in `Sources/LearnModels.swift`: `Wieder/Schwer/Gut/Leicht`). Sie braucht denselben Fix — das wird ein eigener Brief vom Master, **nicht** Teil dieses Sprints.

## Nicht-Ziele

- **Kein** Anfassen des Schedulers, der Ratings-Werte oder der Statistik-Berechnung.
- **Keine** Rückwirkung auf bestehende Karten, kein Umschreiben von `rating_history`.
- **Kein** iOS-Code.
- **Keine** neue Einstellung — die Beschriftung ist keine Option, sondern die Korrektur.
