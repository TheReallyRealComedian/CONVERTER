# SPRINT LEARN-MORE — „Mehr lernen", ohne den Tag neu aufzumachen

**Größe**: M (3 Phasen) · **Datum**: 2026-07-25 · **Vorgänger**: LEARN-COUNT (`ba2e793`), LEARN-STEP (`5730c3a`) — beide gelandet, `static/js/review.js` ist frei

## Warum

Oli: *„wenn ich mal in einer session mehr lernen möchte, hieße das einfach so weiter machen als ob die tägliche menge halt mehr ist; nicht eine neue tagessession machen."*

Heute gibt es dafür **nur** das dauerhafte Tageslimit in den Einstellungen — das ändert jeden künftigen Tag mit. Wer heute Lust auf mehr hat, muss also eine Dauereinstellung verbiegen und hinterher zurückdrehen.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

- `/api/review-state` ([app_pkg/cards.py:548-566](app_pkg/cards.py)) holt **erst alle** fälligen Karten (`Review.due <= now`), übergibt sie an `order_due_cards` und **cappt dort**. Die Vor-Cap-Liste liegt also bereits als `due_cards` vor — `remaining_today` ist eine Subtraktion, keine zweite Query.
- Budgets sind `Limit − heute_erledigt` aus `count_done_today` (distinkte Karten, Berliner Kalendertag). **Uncappen ist per Konstruktion wiederholungsfrei**: die Queue ist `due <= now`, und was heute erledigt wurde, trägt ein Fälligkeitsdatum in der Zukunft und kann gar nicht zurückkommen. Genau das ist Olis Bedingung — sie hält, ohne dass wir etwas dafür tun müssen.
- `local_day_bounds` (in [app_pkg/learn.py](app_pkg/learn.py)) liefert die Berliner Tagesgrenzen und wird schon von `count_done_today` benutzt.

## Gesperrte Entscheidungen (Oli, 2026-07-25)

1. **Gestuft.** Stufe 1 = heutiger Überhang. Ist der leer, fragt Stufe 2 **explizit**, ob aus der Zukunft vorgezogen wird. Nie implizit vorziehen.
2. **Stufe 1 nimmt alles.** Ein Klick hebt den Tages-Cap für diese Session auf und zieht **alles** nach, was heute noch fällig ist — keine Häppchen. (Oli hat das bewusst gegen die Master-Empfehlung gewählt; die Queue darf entsprechend lang werden.)

## Master-Entscheidung, die Oli noch kippen kann

**Stufe 2 borgt tageweise.** „Alles was noch fällig ist" lässt sich auf die Zukunft nicht übertragen — das wäre der gesamte Stapel. Stattdessen zieht Stufe 2 **einen Tag** vor (die Karten, die morgen fällig wären); nochmal drücken borgt den nächsten Tag. Begründung: es ist dieselbe Tages-Metapher wie der Rest des Features, es ist von Natur aus begrenzt, und es ist wiederholbar. Im Schlussbericht als bewusste Entscheidung ausweisen, damit Oli sie nach dem ersten Gebrauch verwerfen kann.

---

# Phase 1 — Backend

## 1.1 Zwei Parameter an `/api/review-state`

- **`?uncapped=1`** — überspringt die Tages-Caps (`review_budget=None, new_budget=None` an `order_due_cards`). Ordering bleibt **unverändert**.
- **`?ahead=<n>`** — verschiebt die Fälligkeitsgrenze auf das **Ende des Berliner Tages `heute + n`** (statt `now`). Impliziert `uncapped` (mit Cap vorziehen ergibt keinen Sinn). `n` auf **1..7** begrenzen, alles andere → **400** mit deutschem Grund.

Beide strikt lesen (kein Truthiness-Zufall): nur der explizite Wert schaltet. Bestehendes Verhalten ohne Parameter bleibt **byte-identisch**.

## 1.2 Drei Felder in der Antwort

- **`remaining_today`** — wie viele **jetzt fällige** Karten der Cap zurückgehalten hat: `len(due_cards_vor_cap) − len(gecappte_queue)`. Bei `uncapped` immer `0`. Treibt Stufe 1 im UI.
- **`ahead_available`** — wie viele Karten **nach jetzt, aber bis Ende des morgigen Berliner Tages** fällig würden. Eine `count()`-Query. Treibt Stufe 2.
- **`day_end`** — Ende des heutigen Berliner Tages als **zeitzonen-behafteter** ISO-String.

⚠️ **`day_end` ist kein Beiwerk, sondern behebt eine bekannte Zufalls-Korrektheit.** LEARN-COUNTs `stillDue`-Regel vergleicht heute `new Date(due) <= new Date()`; die API liefert `due` als **naives** UTC-Isoformat, das JS als **Lokalzeit** liest. Der Berliner Zwei-Stunden-Versatz lässt kurze Lernschritte zufällig als „noch fällig" durchgehen — richtig, aber aus dem falschen Grund, und es bricht in dem Moment, in dem jemand ein Intervall zwischen 15 min und einem Tag einführt. Mit `day_end` rechnet das JS **gar keine** Zeitzonen mehr (siehe 2.3).

## 1.3 Tests

Ohne Parameter unverändert · `uncapped=1` liefert mehr Karten und `remaining_today == 0` · `ahead=1` zieht eine morgen fällige Karte herein, die vorher fehlte · `ahead=0`/`8`/`abc` → 400 · `remaining_today` stimmt gegen einen bekannten Cap · **eine heute bereits erledigte Karte taucht bei `uncapped=1` NICHT auf** (die Kern-Eigenschaft, die Olis Bedingung trägt — explizit festnageln) · `day_end` ist zeitzonen-behaftet und liegt hinter `now`.

## Stop
`pytest tests/` grün, Testzahl vorher/nachher. **Commit + Push** `feat(LEARN-MORE): review-state kennt uncapped + ahead (P1)`. Dann warten.

---

# Phase 2 — UI

## 2.1 Das gestufte Angebot im Done-Panel

Das Done-Panel (`Alle N fälligen Karten wiederholt.`) bekommt genau **einen** Handlungsvorschlag, je nach Zustand:

| Zustand | Angebot |
|---|---|
| `remaining_today > 0` | **Stufe 1** — Button „Mehr lernen", darunter ein Satz, wie viele heute noch liegen. Klick → neu laden mit `uncapped=1`. |
| `remaining_today == 0` **und** `ahead_available > 0` | **Stufe 2** — sichtbar **anderer** Vorschlag: Button „Morgen vorziehen" plus ein Satz, dass damit Karten von morgen vorgezogen werden. Klick → `ahead=1`, nächster Klick → `ahead=2`, usw. |
| beides 0 | unverändert wie heute. |

Stufe 2 darf **nie** ohne eigenen Klick passieren, und ihre Microcopy muss das Borgen benennen — sie hat einen echten Preis (zu früh wiederholte Karten bringen weniger und blähen die künftige Tageslast). Microcopy deutsch: Buttons ≤3 Wörter, Sätze knapp.

## 2.2 Zähler-Kohärenz nach dem Nachladen

Ein Nachlade-Klick ist ein normaler `load()`-Durchlauf: `queue`/`totalDue`/`index` werden neu gesetzt, die Fortschrittszeile zählt wieder ab 1 gegen die **neue** Sessiongröße, die Cap-Zeile kommt frisch vom Server. Die LEARN-COUNT-Buchführung (`decrementPoolCounts`, `renderCapInfo`, `finishSession`-Resync) muss danach **unverändert weiterfunktionieren** — nicht umbauen, nur sicherstellen, dass der Reload-Pfad sie korrekt neu initialisiert.

## 2.3 Die `stillDue`-Regel explizit machen

Heute: „Karte verlässt den Pool, wenn `due` in der Zukunft liegt" — mit der oben beschriebenen Zeitzonen-Zufälligkeit.

Neu: **„Karte verlässt den Pool, wenn `due` hinter dem heutigen Tagesende liegt"**, verglichen gegen das vom Server gelieferte `day_end`. Das ist zugleich die **richtigere Bedeutung**: die Zähler sagen dann „was heute noch dran ist" statt „was in dieser Sekunde fällig ist" — und genau so liest Oli sie. Eine Karte, die in 10 Minuten wiederkommt, senkt die Zahl also weiterhin nicht, aber jetzt **absichtlich**.

Im Kommentar festhalten, dass die JS-Seite bewusst **keine** Zeitzonen-Arithmetik macht: die Grenze kommt vom Server, der sie mit `local_day_bounds` ohnehin schon berechnet.

## 2.4 Smoke (Pflicht — die Suite rendert kein JS)

Wegwerf-Instanz wie in den Vorgänger-Sprints (eigene DB im Scratchpad, **nicht** Prod). Tageslimit klein setzen, mehr fällige Karten anlegen als das Limit hergibt:

| Fall | Erwartung |
|---|---|
| Session zu Ende, Überhang vorhanden | Stufe-1-Button erscheint mit korrekter Restzahl |
| Stufe 1 klicken | Queue läuft weiter, **keine** der eben bewerteten Karten kommt wieder, Fortschrittszeile zählt gegen die neue Sessiongröße |
| Nach Stufe 1 nochmal fertig | Kein Stufe-1-Button mehr; Stufe-2-Angebot **nur** wenn morgen etwas fällig wäre |
| Stufe 2 klicken | Morgen fällige Karten erscheinen; erneuter Klick holt den übernächsten Tag |
| Gar nichts offen | Done-Panel unverändert wie heute |
| „Nochmal"-Karte | senkt die Zähler weiterhin nicht (jetzt gegen `day_end`, nicht gegen `now`) |

Danach Wegwerf-Daten restlos entfernen.

## Stop
`pytest tests/` grün + Smoke-Protokoll mit allen sechs Zeilen. **Commit + Push** `feat(LEARN-MORE): gestuftes Nachladen im Review-UI (P2)`. Dann warten.

---

# Phase 3 — Wrap

CLAUDE.md-Notiz im Learning-Bullet (die zwei Parameter, die gestufte Mechanik, `day_end` als Zeitzonen-Anker) · STATUS.md + BACKLOG.md (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein) · Memory zur tragenden Einsicht: **Tagesgrenze + `due <= now` machen Uncapping per Konstruktion wiederholungsfrei** — und Zeitzonen-Grenzen gehören auf die Server-Seite, nicht in JS-Arithmetik; verlinken auf `[[reference_naive_utc_due_js_local_parse]]`, dessen Zufalls-Korrektheit dieser Sprint auflöst.

Schlussbericht mit Deploy-Schritten **und** einer expliziten Notiz zur Master-Entscheidung „Stufe 2 borgt tageweise", damit Oli sie nach dem ersten Gebrauch bewerten kann.

## Nicht-Ziele

- **Kein** Anfassen der Ordering-Logik, der Cap-Arithmetik in `capped_session_counts`, von `count_done_today` oder des Schedulers.
- **Keine** neue Einstellung, kein Env-Regler — „mehr" ist eine Session-Geste, keine Konfiguration.
- **Kein** Persistieren des Nachlade-Zustands über ein Neuladen der Seite hinweg (wer neu lädt, startet wieder gecappt — das ist korrekt und schützt die Tagesgrenze).
- **Kein** Umbau der LEARN-COUNT-Buchführung über 2.3 hinaus.
