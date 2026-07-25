# SPRINT LEARN-COUNT — Live-Zähler im Review

**Größe**: S (1 Phase) · **Datum**: 2026-07-25 · **Vorgänger**: CARD-SVG (`0247f6b`)

## Warum

Oli: *„beim lernen updaten sich die zahlen nicht — erst wenn 10 durch sind sehe ich dass sie runtergegangen ist. das betrifft die anzeige oben bei den lernpfaden aber auch die x von y fällig — beide sind starr."*

Während einer Session steht der Stapel scheinbar still. Das nimmt dem Lernen das Wichtigste, was eine Fortschrittsanzeige leisten kann: das Gefühl, dass der Berg kleiner wird.

## Gegroundete Diagnose (Master, nicht neu herleiten)

Drei starre Zahlen — **eine mehr, als Oli benannt hat**:

| Zahl | Ort | Ursache |
|---|---|---|
| Pill-Badges (`34`) | Lernpfad-Pills | `loadCollections()` läuft nur beim Seitenladen und in `finishSession()` ([review.js:194](static/js/review.js)) — daher „erst wenn alle durch sind". |
| `N Reviews fällig · M neu verfügbar` | Zeile darunter | Wird in `load()` **einmal** gesetzt ([review.js:330-333](static/js/review.js)), danach nie wieder angefasst. |
| `Karte X von Y fällig` | über der Karte | `totalDue` wird **nur beim Löschen** dekrementiert ([review.js:279](static/js/review.js)), nie beim Bewerten. |

**Kein neuer Endpoint nötig.** `POST /api/cards/<id>/review` antwortet mit der vollen Karte (`Card.to_dict()`) — inklusive des **neuen** `review.due` und der `collections`-Liste. Der Client weiß nach jeder Bewertung also selbst, ob die Karte wirklich aus dem Fällig-Pool ist und welche Pills betroffen sind. Kein Extra-Request pro Karte, kein Backend-Touch.

## Gesperrte Entscheidung (Oli, 2026-07-25)

Die Fortschrittszeile wird **`Karte 4 von 12`** — Nenner bleibt die Sessiongröße, das Wort **„fällig" fällt weg**. Sie ist ein Fortschrittsbalken, kein Fällig-Zähler; das Schrumpfen des Stapels zeigen die Pills und die Cap-Zeile, die beide live mitlaufen. **Kein** mitlaufender Nenner (sonst wandern X und Y aufeinander zu).

---

# Phase 1 — Live-Zähler (nur `static/js/review.js`)

## 1.1 Zustand hochziehen

`review_count`/`new_count` werden heute nur inline in `load()` in den Text geschrieben. Beide in Modul-State heben (neben `totalDue`), damit sie fortgeschrieben werden können, plus ein Helper, der die Cap-Zeile aus dem State neu rendert (die Formulierung selbst bleibt **unverändert**).

## 1.2 Die Abgangs-Regel

Nach **erfolgreicher** Bewertung (`resp.ok`, in `rate()`): die Antwort ist die aktualisierte Karte. Eine Karte verlässt den Fällig-Pool nur, wenn ihr **neues** `review.due` in der Zukunft liegt:

```js
const stillDue = new Date(updated.review.due) <= new Date();
```

⚠️ **Das ist der load-bearing Teil.** Bei „Nochmal" plant FSRS die Karte Minuten später wieder ein — sie ist **weiterhin fällig**, und dann darf **nichts** dekrementiert werden. Ein blindes `-1` pro Bewertung wäre falsch und würde die Zahlen gegen die Realität laufen lassen.

## 1.3 Was dekrementiert wird (nur wenn `!stillDue`)

- **Pill-Badges**: für **jede** Collection-Id in der bewerteten Karte den `due_count` im gecachten `collections`-Array senken, dann `renderScopePills()`. Eine Karte in mehreren Sammlungen senkt **mehrere** Badges — korrekt, sie zählt in jeder einzeln. Eine Karte ohne Sammlung senkt keins.
- **Cap-Zeile**: je nachdem, ob die Karte **neu** oder ein **Review** war. Die Klassifikation **vor** der Bewertung aus dem Queue-Objekt lesen: neu = `card.review.stability === null` (Konvention seit LEARN-UP). Nach der Bewertung ist `stability` gesetzt — dann ist die Unterscheidung weg.
- Alle Zähler mit `Math.max(0, …)` gegen negative Werte sichern.

## 1.4 Fortschrittszeile

`updateProgress()` → `Karte ${Math.min(index + 1, totalDue)} von ${totalDue}` (ohne „fällig"). `totalDue` bleibt beim Bewerten **unverändert**. Der Done-Text (`Alle N fälligen Karten wiederholt.`) bleibt wie er ist — er ist korrekt und beschreibt die Session.

## 1.5 Löschen zieht mit

`deleteCard()` senkt heute nur `totalDue`. Eine gelöschte fällige Karte ist ebenfalls aus dem Pool → dieselben Pill-/Cap-Dekremente anwenden (die Karte liegt vor dem Löschen noch im Queue-Objekt vor, inklusive `collections` und `review.stability`). Der bestehende `load()`-Pfad beim Leeren des Tails bleibt.

## 1.6 Selbstheilung nicht anfassen

`finishSession()` ruft weiterhin `loadCollections()`. Das ist ab jetzt der **autoritative Resync**: sollte die lokale Buchführung je driften, ist sie am Session-Ende wieder korrekt. Nicht entfernen — und im Kommentar als genau diese Sicherung benennen.

## 1.7 Smoke (Pflicht — die Suite rendert kein JS)

Wegwerf-Instanz wie beim CARD-SVG-Smoke (eigene DB im Scratchpad, **nicht** Prod). Sammlung mit mehreren fälligen Karten anlegen, dann:

| Fall | Erwartung |
|---|---|
| Karte mit „Gut" bewerten | Pill-Badge **−1**, Cap-Zeile **−1** (im richtigen der beiden Werte), Nenner der Fortschrittszeile **unverändert** |
| Karte mit **„Nochmal"** bewerten | **nichts** dekrementiert (Karte ist gleich wieder fällig) |
| Karte in **zwei** Sammlungen | **beide** Badges −1 |
| Karte ohne Sammlung | kein Badge ändert sich, Cap-Zeile trotzdem −1 |
| Fällige Karte löschen | Badge + Cap-Zeile −1 |
| Session zu Ende spielen | Nach dem `loadCollections()`-Resync stehen **dieselben** Zahlen wie die lokale Buchführung → **kein Drift** |
| Neu vs. Review | Eine neue Karte (stability NULL) senkt `neu verfügbar`, eine Wiederholung senkt `Reviews fällig` |

Danach Wegwerf-Daten restlos entfernen.

## Stop
`pytest tests/` grün (unverändert erwartet — reiner Frontend-Sprint, kein Backend-Touch) + Smoke-Protokoll mit allen sieben Zeilen. **Commit + Push** `fix(LEARN-COUNT): Zähler laufen während der Session live mit`. Danach STATUS.md/BACKLOG.md-Wrap (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein) und Schlussbericht mit Deploy-Schritten.

## Nicht-Ziele

- **Kein** Backend-Touch, **kein** neuer Endpoint, **kein** Refetch pro Karte (die Rating-Antwort trägt alles Nötige).
- **Kein** mitlaufender Nenner in der Fortschrittszeile (gesperrte Entscheidung).
- **Kein** Anfassen der Ordering-/Cap-Logik in `app_pkg/learn.py` — die Zahlen sind korrekt, nur ihre Anzeige war eingefroren.
- **Kein** Umformulieren der Cap-Zeile selbst; nur ihre Werte werden fortgeschrieben.
