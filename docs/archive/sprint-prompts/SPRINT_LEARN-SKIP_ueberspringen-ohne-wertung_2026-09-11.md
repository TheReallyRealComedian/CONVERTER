# SPRINT LEARN-SKIP — Überspringen ohne Wertung

**Größe**: S (2 Phasen) · **Datum**: 2026-09-11 · **Vorhaben**: Lern-Layer

## Warum

Oli, 2026-09-11: *„ich bräuchte bei learning noch einen überspringenbutton — wird nicht gewertet, aber ans ende der session gelegt"*.

Heute gibt es im Review nur Bewerten, Löschen oder die Seite verlassen. Wer eine Karte gerade nicht angehen will, muss sie bewerten — und jede Bewertung ist ein Eingang ins Scheduling-Modell. Ein Knopf, der die Karte **zurückstellt, ohne etwas zu melden**, fehlt.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten, alles am Code belegt 2026-09-11)

**Die Session ist ein Array im Modul-State** ([static/js/review.js](../../../static/js/review.js)): `let queue = []` + `let index = 0`, gefüllt einmal je `load()` aus `/api/review-state` (`queue = data.due_cards`, `index = 0`). `advance()` zählt nur `index` hoch; am Ende `finishSession()`.

⚠️ **Es gibt kein Wiedereinreihen in der Session.** Nichts wird je an `queue` angehängt — auch eine „Nochmal"-Karte (10-min-Schritt) kommt erst beim nächsten `load()` wieder, nicht im laufenden Array. Überspringen ist damit die **erste** Operation, die das Array umsortiert; es kollidiert mit keinem bestehenden Mechanismus. Die einzige andere Array-Mutation ist `deleteCard` (`queue.splice(index, 1)`, `index` bleibt, die Folgekarte rückt nach) — **das ist das Vorbild**.

**`renderCard(card)`** setzt den Kartenzustand vollständig zurück: `revealed = false`, Antwort/Bewertung/Notiz/Sammlungs-Panel versteckt, Back-Figur explizit resettet (CARD-SVG-Stale-Falle). Eine übersprungene Karte, die später wiederkommt, startet also sauber — **über `renderCard`, nicht an ihm vorbei**.

**Fortschritt**: `updateProgress()` zeigt `Karte ${index + 1} von ${totalDue}`. ⚠️ `totalDue` kommt aus `data.due_count`, **nicht** zwingend aus `queue.length`. Alles, was „letzte Karte" oder „wie viele noch" entscheidet, muss aus `queue.length − index` gelesen werden, nie aus `totalDue`.

**Pool-Zähler**: `decrementPoolCounts(card)` senkt Sammlungs-Badges und Cap-Zeile, wenn eine Karte den heutigen Pool verlässt (Bewertung ins Morgen, Löschen). Überspringen verlässt den Pool **nicht**.

**Tastatur** (`keydown`-Handler am Dateiende): unaufgedeckt Space/Enter → aufdecken; aufgedeckt 1–4 → bewerten; in TEXTAREA/INPUT ignoriert.

**Die sichtbare Fläche vor dem Aufdecken** ist genau die `review-reveal-row` mit dem Knopf „Aufdecken" ([templates/review.html:96](../../../templates/review.html)); die Fußzeile trägt Karten-Aktionen (Vertiefen · Notiz · Sammlung · Löschen).

⚠️ **Testkarten dürfen nicht über die API entstehen.** `POST /api/cards` ist CARD_TOKEN-authed und schreibt **immer** auf `_resolve_target_user()` = `INGEST_USER` bzw. den ersten User — das ist **Olis Konto** ([app_pkg/cards.py:197](../../../app_pkg/cards.py) `_authorize_card_write`). Ein Smoke, der so Karten anlegt und dann bewertet, schreibt in Olis `rating_history`.

## Gesperrte Entscheidungen

1. **Überspringen schreibt nichts.** Kein Request, keine Bewertung, kein `rating_history`-Eintrag, kein `decrementPoolCounts`. Die Karte bleibt fällig. Das ist der Kern von „wird nicht gewertet" und wird **am Netzwerk** belegt, nicht angenommen.
2. **Nur vor dem Aufdecken.** Master-Entscheidung, die Oli kippen kann: wer die Antwort gesehen hat, hat eine Information bekommen, die das Modell nicht kennt. Kommt die Karte Minuten später wieder, wird sie auf ein frisch geprimtes Gedächtnis hin bewertet — genau die Klasse **erfundener Stabilität**, die LEARN-RATE und LEARN-BACK beseitigt haben. Nach dem Aufdecken bleiben die vier Bewertungen plus Vertiefen/Notiz/Löschen.
3. **Die Session ist die Seitenlebenszeit.** Die Umordnung lebt nur im Modul-State (wie die LEARN-MORE-Session-Geste). Ein `load()` — Reload, Scope-Wechsel, „Mehr lernen", „Neu laden" — stellt die Server-Ordnung wieder her; die übersprungene Karte ist dann an ihrem regulären Platz. **Kein** Server-Zustand, **keine** neue Spalte, **kein** Endpoint.
4. **Die letzte verbleibende Karte ist nicht überspringbar.** Ans Ende legen hieße, sie an dieselbe Stelle zu legen — der Knopf täte sichtbar nichts, oder schlimmer: ein naives `splice`+`push` endet nie. ⚠️ Nicht über `finishSession()` lösen: dessen Text („Alle N fälligen Karten wiederholt") wäre dann gelogen.
5. **Testdaten gehören einem Wegwerf-User — mit eigenen Karten.** Angelegt per ORM im Container unter dessen `user_id` (Ist-Zustand oben: die API ist dafür der falsche Weg). Null Schreibvorgänge auf User 1. Abräumen **strikt nach `user_id`** — `api_token` trägt Olis iOS-Tokens.

---

# Phase 1 — Der Knopf, belegt

## 1.1 Bauen

Ein Knopf **„Überspringen"** in der `review-reveal-row`, sichtbar genau solange die Karte unaufgedeckt ist. ⚠️ **Sekundäres Gewicht**: „Aufdecken" ist die vorgesehene Handlung und bleibt der Primärknopf; Überspringen darf nicht mit ihm konkurrieren.

Die Mechanik ist die von `deleteCard`, nur ohne Server: die aktuelle Karte aus `queue` nehmen, hinten anhängen, `index` stehen lassen, `renderCard(currentCard())`. Den `busy`-Guard respektieren (keine Umordnung, während eine Bewertung unterwegs ist).

**Tastenkürzel**: deine Wahl — kollisionsfrei mit Space/Enter und 1–4, im TEXTAREA/INPUT wirkungslos wie die anderen, und **sichtbar im Knopf** wie die Ziffern der Bewertungsknöpfe (LEARN-RATE: kein Tooltip, Touch kennt kein Hover).

**Die letzte Karte** (gesperrte Entscheidung 4): wie du „nicht verfügbar" zeigst — deaktiviert oder weggelassen —, entscheidest du mit Begründung; die Bedingung wird aus `queue.length − index` gelesen. Ob ein Toast das Zurückstellen bestätigt, ebenso; Microcopy-Hausregeln gelten.

## 1.2 Der Beleg

`pytest` rendert kein JS und fängt nichts davon (CLAUDE.md *Test-Suite-Limit*); **kein** JS-Harness bauen (CARD-MD-Doktrin). Der Gate ist ein Browser-Smoke nach dem Muster von [scripts/smoke_markdown_reader.py](../../../scripts/smoke_markdown_reader.py): Playwright **im** Web-Container, Wegwerf-User, Docstring = Laufanleitung, gemessene Werte in der Ausgabe, `exit 1` bei Fehlschlag. Als `scripts/smoke_review_skip.py` committen.

Er belegt mindestens:

- **Reihenfolge**: bei vier fälligen Karten A B C D führt Überspringen auf A zur Folge B C D A. ⚠️ Neue Karten laufen im `smart`-Modus in **Erstellungsreihenfolge** (LEARN-QUEUE) — der Smoke kennt die Ausgangsfolge also, wenn er die Karten in bekannter Folge anlegt. Unter dem Neu-Limit bleiben (Default `daily_new_limit` = 10), sonst schneidet der Cap.
- **Null Schreibvorgänge**: während des Überspringens verlässt **kein** Request die Seite (Route-Interception, gezählt), und `rating_history` der übersprungenen Karte ist vorher/nachher byte-gleich (DB-Lesung).
- **Pool-Zähler** und „Karte X von N" sind nach dem Überspringen unverändert.
- **Die zurückgekehrte Karte** ist bewertbar, und ihre Bewertung landet normal in `rating_history` (das ist der eine Schreibvorgang des Smokes — deshalb Wegwerf-User).
- **Nach dem Aufdecken** ist Überspringen nicht erreichbar, weder per Knopf noch per Taste.
- **Letzte Karte**: nicht überspringbar, kein Endlos-Kreisel, kein falsches Done-Panel.
- **Im Notizfeld** löst die Taste nichts aus.
- **Dark und hell, 375 px und Desktop** — der Knopf sitzt in einer Zeile, die bei 375 px heute einen `w-full`-Knopf trägt.

## Stop
Smoke grün auf dem deployten Stand, `pytest tests/` grün (Baseline **1019 + 1 Skip**). **Commit + Push** `feat(LEARN-SKIP): Karte ohne Wertung ans Session-Ende legen` (Knopf und Smoke gern getrennt). Dann warten.

---

# Phase 2 — Wrap

- **CLAUDE.md**: im Learning/Review-Bullet ein kurzer Satz — Überspringen ist session-lokal, schreibt nichts, nur vor dem Aufdecken, letzte Karte ausgenommen, und **warum** nur vor dem Aufdecken. Dazu den Smoke in die Smoke-Liste unter *Key Files* und *Test-Suite-Limit*.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): LEARN-SKIP schließen, mit Ergebnis. **LEARN-SKIP-IOS** als benannte Möglichkeit hinterlassen (die iOS-App hat einen eigenen Review in `CONVERTER_iOS`; Präzedenz CARD-MD, Waisen-Pille).
- **Memory** nur bei übertragbarer Lehre — „keine" ist bei einem S-Sprint ein legitimes Ergebnis.
- **Im Bericht benennen**: welches Kürzel und warum · wie die letzte Karte behandelt wird und warum · die gemessenen Request-Zahlen beim Überspringen · dass der Wegwerf-User samt Karten strikt nach `user_id` entfernt ist.

## Nicht-Ziele

- **Kein** Server-Zustand für übersprungene Karten, **keine** Persistenz über ein `load()` hinaus.
- **Kein** Überspringen nach dem Aufdecken (gesperrte Entscheidung 2 — Oli kann sie kippen, dann eigener Zuschnitt).
- **Kein** Umbau von Ordering, Caps, Scheduler oder `rating_history`.
- **Kein** iOS.
- **Keine** Testkarten über `POST /api/cards`.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen.
