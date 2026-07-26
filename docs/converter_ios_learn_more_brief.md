# Developer-Brief an den CONVERTER-iOS-Agenten — LEARN-MORE (Mehr lernen, gestuft)

> **An**: CONVERTER_iOS-Entwickler (`~/CODE/CONVERTER_iOS`).
> **Von**: CONVERTER-Master, 2026-07-26.
> **Worum**: Nach dem Tagespensum bietet das Web jetzt genau **einen** nächsten Schritt an — erst den heutigen Überhang, dann auf ausdrücklichen Wunsch das Vorziehen künftiger Tage. Die App endet stattdessen stumm am Limit. Oli lernt in Lücken; auf dem Handy ist „ich hätte jetzt noch Zeit" eher der Normalfall als am Schreibtisch.
> **Vorgänger**: [converter_ios_card_svg_brief.md](converter_ios_card_svg_brief.md) (Figuren-Port, erledigt).

## TL;DR

- **Nichts ist kaputt.** Rein additiv: zwei **optionale** Query-Parameter, drei neue Antwort-Felder. Ohne Parameter verhält sich `/api/review-state` byte-identisch zu vorher.
- **Zwei Parameter**: `?uncapped=1` (Tages-Cap für diesen Fetch aufheben) und `?ahead=<1..7>` (Fälligkeitsgrenze auf einen künftigen Tag schieben, impliziert `uncapped`).
- **Drei Felder**: `remaining_today`, `next_ahead`, `day_end`. **`day_end` braucht ihr nicht** — Begründung unten, ignoriert es bewusst.
- **Wiederholungsfrei per Konstruktion.** Uncapping kann keine heute schon erledigte Karte zurückbringen. Die App muss **nicht** deduplizieren.
- **Die Progression kommt komplett vom Server.** Der Client rechnet **nie** selbst `days + 1` aus.
- **Genau EIN Angebot** am Sessionende, nie zwei nebeneinander.

## 1. Die zwei Parameter an `GET /api/review-state`

| Parameter | Wert | Wirkung |
|---|---|---|
| `uncapped` | exakt **`1`** | Hebt die Tageslimits für **diesen Fetch** auf → alles, was jetzt fällig ist. |
| `ahead` | ganze Zahl **1–7** | Schiebt die Fälligkeitsgrenze auf das **Ende des Berliner Tages heute+n**. Impliziert `uncapped`. |

⚠️ **`uncapped` wird strikt gelesen** — nur die exakte Zeichenkette `1` schaltet. `true`, `yes`, `0` bewirken **nichts** (bewusst fail-safe: im Zweifel bleibt der Cap stehen). Schickt exakt `uncapped=1`.

`ahead` außerhalb 1–7 oder nicht-ganzzahlig → **400** mit deutschem Grund. Schickt **nicht** beide Parameter gleichzeitig; `ahead` genügt, es impliziert das Uncapping.

Alle bestehenden Scope-Parameter (`?collection=…`) gelten unverändert weiter und **kombinieren** sich damit.

## 2. Die drei neuen Antwort-Felder

```jsonc
{
  "remaining_today": 12,                    // Int
  "next_ahead": { "days": 2, "count": 7 },  // Objekt ODER null
  "day_end": "2026-07-26T22:00:00+00:00",   // ISO, zeitzonen-behaftet
  // … due_count, review_count, new_count, total_count, due_cards wie bisher
}
```

- **`remaining_today`** — wie viele **jetzt fällige** Karten der Tages-Cap zurückgehalten hat. Bei einem `uncapped`/`ahead`-Fetch immer `0` (es wurde ja nichts zurückgehalten). **Treibt Stufe 1.**
- **`next_ahead`** — `{days, count}` oder `null`. Der **nächste Tag, der tatsächlich Karten hat**: leere Tage werden übersprungen (liegt morgen nichts, übermorgen aber schon, kommt `days: 2`). `count` = wie viele Karten der Sprung zusätzlich hereinholt. `null` = nichts mehr in Reichweite oder Deckel (7 Tage) erreicht. **Treibt Stufe 2.**
- **`day_end`** — Ende des heutigen Berliner Tages. **Für diesen Port irrelevant**: das Feld existiert, weil das Web eine lokale Zähler-Buchführung hat, die einen zeitzonen-festen Anker braucht. Eure `rate(_:)` verwirft die Antwort und blättert nur weiter; die Badges kommen beim `reload()` frisch vom Server. **Ignoriert es** — es wäre nur dann relevant, wenn ihr später Live-Zähler nachrüstet, und dann ist es der richtige Anker (niemals JS-/Swift-seitige Zeitzonen-Arithmetik).

## 3. Die Swift-Änderungen

**`ReviewState`** ([Sources/LearnModels.swift:61](../../CONVERTER_iOS/Sources/LearnModels.swift)) bekommt die Felder — beide optional dekodieren, damit ein älterer Server die App nicht bricht:

```swift
let remainingToday: Int?
let nextAhead: NextAhead?
// CodingKeys: case remainingToday = "remaining_today", case nextAhead = "next_ahead"

struct NextAhead: Codable, Hashable {
    let days: Int
    let count: Int
}
```

**`APIClient.reviewState(collections:)`** bekommt zwei optionale Argumente (`uncapped: Bool = false`, `ahead: Int? = nil`) und hängt sie an die Query.

**`LearnStore`** bekommt die Session-Geste als reinen Speicher-Zustand — analog zu `sessionUncapped`/`sessionAhead` im Web:

- `startSession()` startet **immer gecappt** (Geste zurückgesetzt).
- Ein „Mehr"-Aufruf ist ein **normaler** Session-Fetch mit den Parametern: `queue` ersetzen, `sessionTotal = state.dueCount`, `index = 0`, `revealed = false`, Session bleibt aktiv.
- **Scope-Wechsel setzt die Geste zurück** (`toggleScope` — eine andere Auswahl ist eine andere Session).
- Die Geste überlebt **keinen App-Neustart**. Das ist Absicht, kein Mangel: wer neu startet, startet gecappt, und die Tagesgrenze bleibt die Regel statt die Ausnahme.

## 4. Das gestufte Angebot — die Regeln

Am Sessionende (`isDone`, dort wo heute „Alle N fälligen Karten wiederholt." steht) erscheint **genau ein** Angebot:

| Bedingung | Angebot |
|---|---|
| `remaining_today > 0` | **Stufe 1** — „Mehr lernen". Nennt die Restzahl. Klick → Fetch mit `uncapped=1`. |
| sonst, `next_ahead != nil` | **Stufe 2** — sichtbar **anderes** Angebot. Nennt `count` und den Tag. Klick → Fetch mit `ahead = next_ahead.days`. |
| sonst | nichts — Done-Ansicht bleibt wie heute. |

**Stufe 1 hat immer Vorrang.** Nie beide gleichzeitig zeigen: das Angebot soll eine Entscheidung sein, keine Auswahl.

**Stufe 2 braucht immer einen eigenen Klick** und ihre Microcopy muss das **Borgen benennen** — es hat einen echten Preis: zu früh wiederholte Karten bringen weniger Lerneffekt und blähen die künftige Tageslast. Das Web formuliert sinngemäß „Vorziehen wiederholt sie früher als geplant." Formuliert es in eurer Tonalität, aber verschweigt den Preis nicht.

⚠️ **Die Progression ist server-getrieben.** Nach einem `ahead=1`-Durchlauf liefert die Antwort ein **neues** `next_ahead` — nehmt dessen `days`. Rechnet **niemals** selbst weiter: der Server überspringt leere Tage, eine clientseitige Erhöhung um 1 würde auf einem Lückentag ins Leere greifen.

## 5. Die Sackgasse, die das Web hatte

Wenn das Tagesbudget verbraucht ist, kommt die Queue **leer** zurück — obwohl noch fällige Karten existieren (`remaining_today > 0`). Euer `startSession()` behandelt das heute als „nichts fällig" und bricht ab. Damit wäre der Überhang **nach jedem App-Start unerreichbar**, denn die Geste startet ja gecappt.

**Also**: leere Queue **mit** `remaining_today > 0` ist kein Leerlauf, sondern „Tagespensum erreicht" — und muss Stufe 1 anbieten. Ob das in der Done-Ansicht, im Launcher oder als eigener Zustand erscheint, entscheidet ihr; euer Launcher-Aufbau unterscheidet sich vom Web. **Die Invariante ist: der Überhang muss ohne Umweg erreichbar sein.**

## 6. Zwei bewusste Nicht-Entscheidungen

Bitte **nicht** „verbessern" — beides ist so entschieden:

1. **Am komplett leeren Tag gibt es kein Vorzieh-Angebot.** Keine fällige Karte heißt: der Scheduler sagt Stopp. Ein Vorzieh-Knopf ausgerechnet dort arbeitet gegen ihn — Anki versteckt „review ahead" aus demselben Grund hinter Custom Study. Die Done-Ansicht ist der legitime Ort, weil dort gerade gelernt wurde.
2. **Die Geste überlebt keinen Neustart.** Siehe oben — die Tagesgrenze soll die Regel bleiben.

## 7. Eine Eigenschaft, die euch Arbeit spart

**Die Angebots-Zahlen veralten einseitig — und das ist eine Garantie, kein Makel.** `remaining_today` und `next_ahead.count` stammen vom Fetch zu Sessionbeginn. Bewertungen während der Session können beide nur **wachsen** (eine „Nochmal"-Karte wird wieder fällig), nie schrumpfen. Das Angebot verspricht also **nie** Karten, die es nicht gibt, und der Klick lädt ohnehin Server-Wahrheit.

Konsequenz: **kein Nachladen der Zahlen während der Session nötig.** Ein „1 Karte"-Angebot, das dann 2 Karten lädt, ist korrektes Verhalten und kein Bug.

Ebenso: **kein Dedupe nötig.** Die Queue ist `due <= Horizont`, und was heute erledigt wurde, trägt ein Fälligkeitsdatum in der Zukunft. Eine gerade bewertete Karte **kann** strukturell nicht zurückkommen — außer sie wurde „Nochmal"/„Schwer" bewertet, und dann **soll** sie es.

## 8. Referenzen

- **Architektur-Notiz + volle Mechanik**: `CLAUDE.md`, Bullet *„Mehr lernen" gestuft (LEARN-MORE)*
- **Web-Referenzimplementierung**: `static/js/review.js` (`renderMoreOffer`, `scopeUrl`, die Session-Geste) — als Vorbild lesbar, aber **nicht** 1:1 portieren; euer Launcher/Session-Aufbau ist ein anderer.
- **Backend**: `app_pkg/cards.py::api_review_state`, `app_pkg/learn.py::local_day_end`

## 9. Fertig ist es, wenn

1. Session am Tageslimit zu Ende → Stufe-1-Angebot mit korrekter Restzahl.
2. Stufe 1 geklickt → Queue läuft weiter, **keine** gerade bewertete Karte kommt wieder, Fortschritt zählt gegen die neue Sessiongröße.
3. Danach wieder fertig → Stufe-1-Angebot weg, stattdessen sichtbar **anderes** Stufe-2-Angebot (nur wenn `next_ahead != nil`).
4. Stufe 2 geklickt → künftige Karten erscheinen; das Folge-Angebot nutzt das **neue** `next_ahead.days` (Lückentag-Sprung greift).
5. Nichts offen → Done-Ansicht unverändert.
6. App-Neustart bei verbrauchtem Budget → **kein** Leerlauf, sondern „Tagespensum erreicht" + Stufe 1.
7. Scope-Wechsel setzt die Geste zurück (nächste Session startet gecappt).

## Nicht in diesem Brief

- **Keine Live-Zähler** (das Web-Pendant LEARN-COUNT). Euer `reload()`-Modell ist ein anderer, legitimer Ansatz — `day_end` liegt bereit, falls ihr es je nachrüstet.
- **Keine Einstellungen für die Tageslimits** — die gibt es serverseitig über `GET/PUT /api/learn/settings`, sie sind aber nicht Teil dieses Ports.
- **Kein Figuren-Thema** — erledigt im Vorgänger-Brief.
