# SPRINT LEARN-QUEUE — neue Karten in Erstellungsreihenfolge + Waisen-Pille

**Größe**: S/M (3 Phasen) · **Datum**: 2026-07-30 · **Parallel dispatchbar** mit LEARN-MCP (anderes Repo, andere Maschine)

## Warum

Zwei kleine Befunde aus der Lern-Layer-Diagnose vom 29.07., beide von Oli entschieden.

**(1) Die Reihenfolge neuer Karten wird pro Abruf neu gewürfelt.** `order_due_cards` macht `rng.shuffle(fresh)` — zwei Aufrufe 15 Minuten auseinander liefern andere neue Karten, obwohl sich nichts geändert hat. Das war der Auslöser für den „die Queue würfelt"-Befund. Kein Datenverlust (eine nie gesehene Karte hat keine Vergessenskurve), aber es zerstört die **Didaktik**: der Karten-Agent schreibt Material in einer Reihenfolge — Chemie-Kapitel 4 erst die Hauptgruppen-Tabelle, dann die PSE-Tendenzen — und ein Shuffle wirft das weg.

**(2) Karten ohne Sammlung sind unauffindbar.** Sie sind **nicht** unerreichbar (der Bericht lag hier falsch: der Launcher hat „Alles fällig", ohne Auswahl greift kein `collection`-Filter). Aber wer ausschließlich über die Pillen einsteigt, bekommt kein Signal, dass es sie gibt. Oli hat aktuell **null** Waisen — der Sprint ist Vorsorge, und das prägt das Design.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

- `order_due_cards` ([app_pkg/learn.py](app_pkg/learn.py)): teilt in `reviewed` (stability gesetzt) und `fresh` (stability NULL); `rng.shuffle(reviewed)` **vor** dem stabilen Sort ist der Zufalls-Tiebreak (load-bearing, py-fsrs rechnet R tages-granular); `rng.shuffle(fresh)` ist der Teil, der hier fällt. Caps greifen **nach** dem Ordering, `_interleave_evenly` streut `fresh` gleichmäßig ein.
- Scope-Filter in `api_review_state` ([app_pkg/cards.py](app_pkg/cards.py)): `?collection=<id>[,<id>…]` als **Union** über `Card.collections.any(Collection.id.in_(...))`; jede id muss owned sein, sonst 404.
- Launcher ([static/js/review.js](static/js/review.js)): Checkbox-Pillen aus `/api/collections`, leere Auswahl = „Alles fällig" (kein Filter); `scopeUrl()` hängt `collection=` nur an, wenn etwas gewählt ist.
- ⚠️ **`/api/collections` liefert ein blankes Array**, und die iOS-App dekodiert genau das (`[LearnCollection]` in `Sources/LearnModels.swift`). **Die Antwortform dieses Endpoints darf sich nicht ändern** — ein Objekt-Wrapper bräche die App.

## Gesperrte Entscheidungen (Oli, 2026-07-30)

1. **Neue Karten kommen in Erstellungsreihenfolge** (Anki „order added"). Das revidiert bewusst einen Teil der LEARN-UP-Aussage „die versteckte Erstellungsreihenfolge ist strukturell weg" — die zielte auf **Wiederholungen** (Root Cause war `ORDER BY due ASC` über alles) und bleibt dort gültig. Für neue Karten ist Reihenfolge kein Bug, sondern die Didaktik.
2. **Waisen bekommen eine Pille, die nur bei Bedarf erscheint.** Gibt es null Waisen, ist sie unsichtbar und kostet nichts.

---

# Phase 1 — Neue Karten in Erstellungsreihenfolge

## 1.1 Die Änderung

In `order_due_cards`, **nur** im `smart`-Pfad: `fresh` nach Erstellung sortieren statt shufflen. Sortierschlüssel `created_at` aufsteigend mit `id` als Tiebreak — der Agent legt Karten im Batch an, Zeitstempel können kollidieren, und die Ordnung muss **total** sein, sonst ist sie wieder nicht reproduzierbar.

**`random` bleibt ein Voll-Shuffle.** Die Einstellung heißt so, sie muss das tun. Der Zufalls-Tiebreak auf `reviewed` bleibt ebenfalls unangetastet — er ist aus einem anderen Grund da.

Folge, die ausdrücklich erwünscht ist: der Tages-Cap schneidet jetzt die **ältesten N** neuen Karten heraus statt zufälliger N. Kapitel 4 wird fertig, bevor Kapitel 5 anfängt.

`_interleave_evenly` bleibt unberührt — neue Karten werden weiter gleichmäßig eingestreut, nur eben in stabiler Reihenfolge.

## 1.2 Bestehende Tests

⚠️ LEARN-UP hat Tests, die **den Shuffle von `fresh` behaupten**. Die sind jetzt falsch — aber **nicht löschen**: sie ziehen um auf `random`-Modus, wo die Aussage weiter gilt. Ein gelöschter Test ist eine verlorene Zusicherung; ein umgezogener ist eine präzisierte.

## 1.3 Neue Tests

- Zwei aufeinanderfolgende `order_due_cards`-Aufrufe auf denselben Daten liefern **dieselbe** Reihenfolge der neuen Karten (die Kern-Eigenschaft).
- Die Reihenfolge entspricht `created_at`; bei identischem `created_at` entscheidet `id`.
- Der Cap nimmt die **ältesten** N, nicht zufällige N.
- `random` shuffelt weiterhin.
- Reviews sind unverändert R-sortiert, Tiebreak intakt.

## Stop
`pytest tests/` grün, Testzahl vorher/nachher (Baseline **787**). **Commit + Push** `feat(LEARN-QUEUE): neue Karten in Erstellungsreihenfolge (P1)`. Dann warten.

---

# Phase 2 — Die Waisen-Pille

## 2.1 Backend

Neuer Scope-Parameter an `/api/review-state`: **`?uncollected=1`**, strikt gelesen (nur der exakte Wert schaltet — Hauspattern seit LEARN-MORE).

**Er kombiniert sich mit `collection=` als Union**, nicht als Alternative: die Pillen sind Mehrfach-Auswahl, und „Sammlung X **oder** ohne Sammlung" ist die einzige Semantik, die zum bestehenden Modell passt. In SQLAlchemy also `or_(Card.collections.any(...), ~Card.collections.any())`; steht `uncollected=1` allein, ist die Bedingung nur der zweite Zweig.

**Die Zahl** für Badge und Sichtbarkeit kommt als neues Feld **`uncollected_count`** in der `review-state`-Antwort (roh, `due <= now`, wie die Collection-Badges — nicht gedeckelt).

⚠️ **`/api/collections` bleibt unangetastet.** Es liefert ein blankes Array und die iOS-App dekodiert genau das; ein Objekt-Wrapper oder ein synthetischer Eintrag mit `id: null` bräche sie. Deshalb reist die Zahl über `review-state` mit, wo additive Felder nachweislich unschädlich sind.

## 2.2 UI

Pille „Ohne Sammlung" im Launcher, **nur gerendert wenn `uncollected_count > 0`**, mit Fällig-Badge wie die Sammlungs-Pillen, ankreuzbar wie sie. Angekreuzt hängt `scopeUrl()` `uncollected=1` an.

Bewusst: verschwindet die letzte Waise, verschwindet die Pille. Ist sie angekreuzt und wird leer, muss der Scope sauber auf „Alles fällig" zurückfallen statt in einen Zustand zu laufen, den man nicht mehr abwählen kann.

## 2.3 Tests + Smoke

Backend-Tests: `uncollected=1` liefert genau die Karten ohne Sammlung · Union mit `collection=` liefert beide Mengen ohne Dubletten · ohne Waisen ist `uncollected_count == 0` · der Parameter ist strikt (`uncollected=true`/`0` schalten nicht) · Karten anderer User bleiben draußen.

**Live-Smoke ist Pflicht** (die Suite rendert kein JS): Wegwerf-Instanz mit einer Waise → Pille erscheint mit korrektem Badge · anklicken filtert · die Waise einer Sammlung zuordnen → Pille verschwindet · null Waisen von Anfang an → Pille war nie da. Dark und Light. Wegwerf-Daten restlos entfernen.

## Stop
`pytest tests/` grün + Smoke-Protokoll. **Commit + Push** `feat(LEARN-QUEUE): Waisen-Pille im Launcher (P2)`. Dann warten.

---

# Phase 3 — Wrap

- **CLAUDE.md**, Learning-Abschnitt: die Erstellungsreihenfolge für neue Karten **mit dem ausdrücklichen Hinweis**, dass sie die LEARN-UP-Aussage nur für neue Karten revidiert und für Wiederholungen unberührt lässt; die Waisen-Pille samt der Notiz, warum die Zahl über `review-state` reist und **nicht** über `/api/collections` (iOS-Dekodierung).
- **STATUS.md** + **BACKLOG.md** (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein).
- **Memory**: *Ein Shuffle, der Reproduzierbarkeit kosten soll, muss begründen, wogegen er schützt — bei Wiederholungen ist es ein Reihenfolge-Effekt, bei neuen Karten zerstört er die Didaktik des Autors.* Verlinken auf `[[reference_two_axis_card_grouping]]`. **Nach dem Schreiben mit `ls` prüfen**, dass die Datei liegt und MEMORY.md genauso viele Index-Zeilen hat wie Dateien existieren — bei LEARN-RATE ist ein Memory-Write stillschweigend nicht angekommen.
- **Im Bericht benennen**: die iOS-App bekommt die Waisen-Pille **nicht** (ihr Scope-UI baut auf `/api/collections`, das bewusst unverändert bleibt). Das ist kein Versehen — falls Oli sie dort will, ist es ein eigener Brief.

## Nicht-Ziele

- **Kein** Anfassen der Review-Ordnung (R-Sortierung + Tiebreak bleiben), der Cap-Arithmetik, von `count_done_today` oder des Schedulers.
- **Keine** Änderung an der Antwortform von `/api/collections`.
- **Keine** Pflicht-Sammlung, **keine** NOT-NULL-Zusicherung, **kein** Default-Lernpfad — Waisen sind ein Anzeige-Thema, kein Datenmodell-Thema.
- **Kein** iOS-Code.
