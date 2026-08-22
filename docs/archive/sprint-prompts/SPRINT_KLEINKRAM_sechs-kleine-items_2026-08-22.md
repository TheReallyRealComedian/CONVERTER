# SPRINT KLEINKRAM — sechs kleine Items, nach Risiko sortiert

**Größe**: M (3 Phasen) · **Datum**: 2026-08-22 · **Vorhaben**: Betrieb

## Warum

Sechs Items, die einzeln zu klein für einen Sprint sind und zusammen einen ergeben. Sie sind **nicht** thematisch verwandt — die Reihenfolge sortiert nach **Risiko**, nicht nach Thema: zuerst das, was echte Daten anfasst, zuletzt das, was nur Text ändert.

⚠️ **Ein siebtes Item ist beim Grounding erledigt worden**: `DOC-NOTES` beschreibt, dass `partition(strategy="fast")` ohne `include_slide_notes` die Sprechernotizen verliert. Seit DOC-ENGINE läuft PPTX aber über **markitdown** ([services/document_router.py:77](../../../services/document_router.py)), das genau wegen der Sprechernotizen gewonnen hat; `partition` sieht nur noch EML/TXT/MD. **Im Wrap als erledigt schließen, mit dieser Begründung — nicht kommentarlos streichen.**

## Gesperrte Entscheidungen

1. **Reproduzieren vor reparieren.** Zwei der sechs stehen auf **unverifizierten** Vermutungen (TXT-BINDESTRICH, KARTEN-ESCAPE-Ursache). Wer den Befund nicht erst herstellt, repariert das Falsche und zerstört dabei den Beleg.
2. **Ein Item, ein Commit.** Sie hängen nicht zusammen; wenn eins zurückgerollt werden muss, soll das nicht die anderen mitnehmen.
3. ⚠️ **Engine-Generation**: TXT-BINDESTRICH ändert, wenn behoben, die **Ausgabe** des TXT/MD/EML-Zweigs. Dann **muss** `DOC_CONVERT_ENGINE_GENERATION` gebumpt werden — die Bump-Regel steht am Konstanten-Wert. Für die übrigen fünf Items gilt das nicht.

---

# Phase 1 — Die zwei, die echte Daten anfassen

## 1.1 KARTEN-ESCAPE — fünf Karten tragen literale `\n` und `\"`

Verifiziert an der Prod-DB: **5 Karten, 15× literales Backslash-n, 18× literales `\"`**, und die betroffenen Felder haben **null echte Zeilenumbrüche**. Sichtbar z.B. auf Karte 3: `…trotz Systemtherapie)\n- Viele TKIs penetrieren die BHS…` — die Aufzählung dahinter wird nie eine, weil sie nicht am Zeilenanfang steht.

⚠️ **Erst den Schreibweg finden, dann die Daten reparieren.** Repariert man zuerst, ist der Beleg weg und der Fehler kommt beim nächsten Kartenschwung wieder. Das Muster (literales `\n` **und** literales `\"`, beides gleichzeitig) riecht nach **doppelter JSON-Kodierung** irgendwo zwischen Agent und `POST /api/cards` / `PATCH /api/cards/<id>` bzw. dem MCP-Weg. Such dort, und **belege**, welcher Weg es erzeugt — mit einem Aufruf, der es reproduziert.

Findest du den Weg nicht: **sag das**, repariere die Daten trotzdem, und leg ein Item an. Ein bekannter, benannter Datenfehler ist besser als ein unsichtbarer — aber eine unbelegte Ursachen-Behauptung ist schlechter als keine.

⚠️ **Vor dem Datenschreiben ein Prod-DB-Backup** nach dem WAL-Rezept in CLAUDE.md (`sqlite3.backup()` im Container, **nicht** nacktes `docker cp` der `.db`). Es sind Olis echte Lernkarten.

## 1.2 AUDIO-TMP-LEAK — die Upload-Kopie bleibt liegen

Jede Transkription **ohne** Chunking (≤ 90 min, also der Normalfall) hinterlässt `/tmp/tmp*.audio`: `needs_splitting` behält die Datei für `split_audio`, das dann nie läuft. Gemessen: acht Dateien à 22–91 MB nach acht Läufen. Seit SYNC-FREEZE P3 liegt das im **Worker**-Container, nicht mehr im Web.

Die Aufräumung gehört an die Stelle, die die Datei **anlegt**. ⚠️ **Der gechunkte Pfad darf nicht brechen** — dort wird die Datei gebraucht. Ein Test, der beide Fälle unterscheidet, ist der Beleg.

Selbstbegrenzend ist es nur, weil ein Container-Neustart `/tmp` leert; zwischen zwei Deploys wächst es ungebremst.

## Stop
Beide belegt, je ein Commit. **Push.** Dann warten.

---

# Phase 2 — Die zwei, die erst reproduziert werden müssen

## 2.1 TXT-BINDESTRICH — der TXT-Zweig zerlegt an Bindestrichen

Eine TXT mit `SYNC-FREEZE multi-process RQ check` kam durch den Dienst als `SYNC\n\nFREEZE multi\n\nprocess RQ check` zurück — der Bindestrich wird zur Absatzgrenze.

⚠️ **Die Ursache ist Vermutung, nicht Befund.** Vermutet: `partition(strategy="fast")` zerlegt die Zeile in mehrere Elemente, und der Serializer fügt sie mit `\n\n` zusammen. **Prüf es an echten Element-Objekten** — das ist die Hausregel von [services/unstructured_markdown.py](../../../services/unstructured_markdown.py), dessen sämtliche Regeln an gemessenen Objekten entstanden sind und nicht aus der Doku.

⚠️ **Blast-Radius**: derselbe Zweig trägt **EML**. Olis Mail-Zitatketten sind voller Bindestriche. Ein Belegexemplar je Format (TXT **und** EML) gehört in den Beleg.

Wenn du es behebst, ändert sich die Ausgabe → **Engine-Generation bumpen** (gesperrte Entscheidung 3).

## 2.2 TEST-HANG-KEEPALIVE — ein Test, der hängt statt zu fallen

`tests/test_asgi_adapter.py::test_stock_adapter_leaks_its_context_into_the_follow_up_request` reproduziert den asgiref-Keep-Alive-Leak mit dem **Stock**-Adapter und erwartet einen `RuntimeError`. Der Leak hat aber **zwei** Erscheinungsformen (SYNC-FREEZE: 500 **oder** stilles Hängen), und die zweite kennt der Test nicht: im Container hängt er in 3 von 4 Läufen ewig — auf Python 3.10 **und** 3.12, also kein Umzugs-Effekt.

⚠️ **Ein hängender Test ist schlimmer als ein fallender**: er blockiert den Lauf, statt ihn zu beenden. Maßnahme: ein Timeout um den Aufruf, und **Ablauf zählt als Nachweis des Leaks** — beide Erscheinungsformen bestätigen dieselbe Sache.

Der Produktiv-Pfad (`test_keep_alive_follow_up_request_survives_on_the_pool`) ist grün und bleibt unangetastet.

## Stop
Beide belegt, je ein Commit. **Push.** Dann warten.

---

# Phase 3 — Die zwei Textänderungen, dann Wrap

## 3.1 LEARN-HINT-WEB — der Hinweis, den es auf iOS schon gibt

Die Reihenfolge-Auswahl im Review hat im Web nur ein `title`-Attribut ([templates/review.html:29](../../../templates/review.html)) — ein Tooltip, der **auf Touch unsichtbar** ist. Genau diese Stille hat Olis `ordering_mode: random` monatelang verborgen und zwei fertige Features wirkungslos gelassen, bis es jemand nachgemessen hat.

Der iOS-Port hat pro Modus einen Text formuliert; bei „Zufällig": *„Auch neue Karten kommen dann in zufälliger Reihenfolge statt in der, in der sie angelegt wurden."* **Hol ihn ins Web**, sichtbar, nicht als Tooltip. Microcopy-Hausregeln gelten.

## 3.2 DOC-KORPUS-14 — die Klassen-README steht auf einer falschen Prämisse

„Jeden Umlaut verloren / mit englischem Modell erzeugt" ist nachgemessen falsch: die Ebene trägt **549 korrekte Umlaute** und ist nur sporadisch kaputt (2× „Asthetik" S. 2/14, ~13 Einzelbrüche, darunter „8.167" statt „S. 157").

⚠️ Die korrigierte Fassung muss die **Bruchstellen-Anker** tragen, nicht nur die Korrektur — sonst misst der Nächste wieder gegen eine Prämisse statt gegen Fundstellen. Betrifft [corpus/14_ocr-ebene-kaputt/README.md](../../../corpus/14_ocr-ebene-kaputt/README.md) und den Nachtrag in [corpus/00_KORPUS-UEBERSICHT.md](../../../corpus/00_KORPUS-UEBERSICHT.md).

## 3.3 Wrap

- **DOC-NOTES als erledigt schließen**, mit der Begründung aus dem Kopf dieses Prompts.
- **CLAUDE.md** nur, wo eines der Items eine dort dokumentierte Aussage berührt (TXT-BINDESTRICH betrifft den Serializer-Bullet).
- **STATUS.md**, **BACKLOG.md** (Bullet-Guard): die sechs Items schließen, je mit Ergebnis. Wo etwas **nicht** behoben wurde, steht das als Ergebnis da.
- **Memory** nur, falls sich eine übertragbare Lehre zeigt — bei sechs kleinen Items ist „keine" ein legitimes Ergebnis.
- **Im Bericht benennen**: den gefundenen (oder nicht gefundenen) Schreibweg von KARTEN-ESCAPE · die gemessene Ursache von TXT-BINDESTRICH · ob die Engine-Generation gebumpt wurde und warum · was du **nicht** repariert hast.

## Nicht-Ziele

- **Kein** Umbau des Serializers über die eine Ursache hinaus, **kein** Engine-Wechsel.
- **Kein** Anfassen des Produktiv-Adapter-Tests, **kein** asgiref-Bump (in SYNC-FREEZE gemessen: hilft nicht).
- **Keine** Feature-Arbeit am Lern-Layer über den einen Hinweistext hinaus.
- **Kein** Aufräumen weiterer Backlog-Items „bei der Gelegenheit" — sechs sind sechs.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen; Wegwerf-User strikt nach `user_id`.
