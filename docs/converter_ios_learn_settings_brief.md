# Developer-Brief an den CONVERTER-iOS-Agenten — Lern-Einstellungen

> **An**: CONVERTER_iOS-Entwickler (`~/CODE/CONVERTER_iOS`).
> **Von**: CONVERTER-Master, 2026-07-31.
> **Worum**: Die drei Lern-Einstellungen — Reihenfolge und die zwei Tageslimits — existieren nur im Web. Die App liest `/api/learn/stats`, kennt aber `/api/learn/settings` überhaupt nicht.
> **Vorgänger**: [converter_ios_card_svg_brief.md](converter_ios_card_svg_brief.md) · [converter_ios_learn_more_brief.md](converter_ios_learn_more_brief.md) — beide portiert.

## Warum das kein Komfort-Feature ist

Am 30.07. hat sich herausgestellt, dass Olis Account auf `ordering_mode: "random"` stand. In diesem Modus ist die Queue ein **Voll-Shuffle** — damit waren zwei fertig gebaute Features bei ihm **wirkungslos**: die Erstellungsreihenfolge neuer Karten (LEARN-QUEUE) und die Retrievability-Sortierung „Wackligste zuerst" (LEARN-UP, seit 18.07.).

Gemessen an dem Tag: 113 fällige Karten, davon **96 neue**. Der Review-Cap band überhaupt nicht; die 87 zurückgehaltenen Karten waren ausnahmslos neue. Welche 9 er zu sehen bekam, entschied allein die Neu-Sortierung — und die würfelte bei `random` **bei jedem Abruf neu**.

Aufgefallen ist das nur, weil jemand die Live-Ausgabe nachgemessen hat. **Der Schalter, der das verursacht, ist auf dem Gerät, auf dem Oli tatsächlich lernt, nicht erreichbar.** Das ist der Grund für diesen Port.

## TL;DR

- **Nichts ist kaputt.** Rein additiv: ein Endpunkt, den die App bisher nicht kennt. Es gibt nichts zu migrieren.
- **Drei Einstellungen**, alle in einem JSON-Blob: `ordering_mode`, `daily_new_limit`, `daily_review_limit`.
- **Auth wie überall**: `@login_required`, der Bearer-Token greift über den `request_loader`. Kein CSRF, kein neuer Token.
- `GET` liefert immer den **vollständigen effektiven** Satz, `PUT` akzeptiert **Teilmengen** und antwortet mit dem vollständigen gemergten Satz.

## Der Kontrakt

### `GET /api/learn/settings`

```json
{ "ordering_mode": "smart", "daily_new_limit": 10, "daily_review_limit": 200 }
```

Immer alle drei Schlüssel, immer die *effektiven* Werte (Defaults überlagert mit dem Gespeicherten). Nie `null`, nie fehlend. Ein Decodable-Struct mit drei nicht-optionalen Feldern ist korrekt.

### `PUT /api/learn/settings`

Body ist ein JSON-Objekt mit **beliebiger Teilmenge** der drei Schlüssel:

```json
{ "ordering_mode": "smart" }
```

Antwort ist der **vollständige gemergte** Satz — dieselbe Form wie `GET`. Übernimm die Antwort als neuen Zustand, statt lokal zu raten.

### Werte und Grenzen

| Schlüssel | Typ | Erlaubt | Default | Bedeutung |
|---|---|---|---|---|
| `ordering_mode` | String | `"smart"` \| `"random"` | `"smart"` | `smart` = bereits bewertete Karten nach Retrievability aufsteigend (die wackligsten zuerst), neue Karten in **Erstellungsreihenfolge** gleichmäßig eingestreut. `random` = Voll-Shuffle beider Töpfe. |
| `daily_new_limit` | Int | `0…10000` | `10` | Neue Karten pro Berliner Tag |
| `daily_review_limit` | Int | `0…10000` | `200` | Wiederholungen pro Berliner Tag |

## Fünf Fallen, alle am Server verifiziert

1. **`0` ist gültig, nicht „aus".** Ein Limit von 0 heißt „heute nichts davon" und ist eine legitime Einstellung. Die UI muss 0 zulassen und darf es nicht als Fehler behandeln.
2. **`bool` wird explizit abgewiesen.** Der Validator prüft `isinstance(value, bool)` **vor** dem Int-Check, weil `bool` in Python eine Int-Unterklasse ist. Aus Swift kommt das nicht vor — aber schick keine `true`/`false` in die Limit-Felder.
3. **Strikte Validierung, nichts wird teilweise geschrieben.** Ein unbekannter Schlüssel oder ein ungültiger Wert → **400**, und es wird **gar nichts** gespeichert, auch nicht die gültigen Schlüssel derselben Anfrage. Optimistische UI-Updates müssen bei 400 also vollständig zurückrollen.
4. **Die Antwort ist die Wahrheit.** Nach jedem erfolgreichen `PUT` den zurückgegebenen Satz übernehmen. Er kann von dem abweichen, was du geschickt hast (Merge mit Defaults).
5. **`desired_retention` aus `/api/learn/stats` ist read-only.** Es kommt aus einer Server-Env-Variable, nicht aus den Nutzereinstellungen. **Nicht** editierbar machen — der Scheduler liest keine Per-User-Retention, ein Regler dort wäre eine Lüge.

## Zur Oberfläche

Die Platzierung ist deine Entscheidung — du kennst die App. Eine Anforderung gibt es aber: **erreichbar von dort, wo gelernt wird.** Der Wert dieses Ports besteht darin, dass Oli den Reihenfolge-Schalter sieht, ohne an den Schreibtisch zu gehen; in einem App-weiten Einstellungsbildschirm drei Ebenen tief wäre er wieder unsichtbar.

Die Web-Beschriftungen lauten **„Wackligste zuerst"** (`smart`) und **„Zufällig"** (`random`), das Label darüber ist **„Reihenfolge"**. Übernimm sie, damit über beide Oberflächen dasselbe Wort dieselbe Sache meint.

**Eine Beobachtung, kein Auftrag**: Das Web sagt nirgends, dass „Zufällig" die Erstellungsreihenfolge neuer Karten mit abschaltet. Genau diese Stille hat den Befund vom 30.07. so lange verborgen. Wenn dir dafür eine knappe, ehrliche Formulierung einfällt (ein Hinweis unter dem Schalter, keine Warnung), nimm sie mit und sag im Bericht Bescheid — dann ziehe ich sie im Web nach. Wenn nicht, ist das Weglassen auch in Ordnung; erfinde keine Angstmache.

Bei den Limits: sie greifen **nach** der Sortierung und pro **Berliner** Tag; neue Karten füllen nur das Headroom, das die Wiederholungen übriglassen. Das muss die UI nicht erklären, aber es erklärt, warum ein hoher Neu-Wert an einem vollen Wiederholungstag nichts bewirkt.

## Ausdrücklich nicht in diesem Brief

- **`GET /api/learn/simulate`** (Workload-Prognose mit What-if-Regler). Existiert, ist aber ein Analyse-Werkzeug für den Schreibtisch. Eigener Brief, falls je gewünscht.
- **Die Waisen-Pille** („Ohne Sammlung", `?uncollected=1`). Bewusst web-only. Oli hat aktuell **null** Karten ohne Sammlung, die Pille wäre also ohnehin unsichtbar. Falls sie später gebraucht wird: die Zahl reist als `uncollected_count` in der `review-state`-Antwort mit, `/api/collections` bleibt dabei unangetastet — es liefert ein blankes Array, das ihr als `[LearnCollection]` dekodiert, und ein Wrapper dort bräche die App.
- **Server-seitige Scheduler-Parameter** (`FSRS_DESIRED_RETENTION`, `FSRS_MAXIMUM_INTERVAL`). Env-only, betreffen die App nicht.

## Abnahme

Ein Live-Durchstich gegen Prod reicht — die Suite deckt UI nicht ab:

1. Einstellungen laden, drei Werte erscheinen (Ist-Zustand: `random` / 24 / 200).
2. Reihenfolge auf „Wackligste zuerst" stellen, App neu starten, Wert ist noch da.
3. Im Web unter `/review` gegenprüfen, dass der Schalter dort dasselbe zeigt — es ist derselbe Blob.
4. Ein Limit auf 0 setzen: wird angenommen, die Queue ist danach für diese Sorte leer.
5. Danach auf sinnvolle Werte zurückstellen — es ist Olis Produktivkonto.
