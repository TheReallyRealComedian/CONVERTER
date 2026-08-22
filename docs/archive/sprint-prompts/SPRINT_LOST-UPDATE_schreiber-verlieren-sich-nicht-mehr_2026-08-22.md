# SPRINT LOST-UPDATE — gleichzeitige Schreiber verlieren sich nicht mehr

**Größe**: M (3 Phasen) · **Datum**: 2026-08-22 · **Vorhaben**: Betrieb

## Warum

In SYNC-FREEZE P2 wurde gemessen: `api_review_card` ist ein **ungeschützter Read-Modify-Write**. Zwei Schreiber auf derselben Karte im selben Moment überschreiben sich — 3.200 Bewertungen ergaben **66–76 statt 80** Einträgen je Karte, bei **3.200 × HTTP 200**.

⚠️ **Das Fenster ist von uns eingebaut, nicht vorgefunden.** Die Attribution ist gemessen: **1 Prozess mit Stock-Adapter (Zustand vor SYNC-FREEZE) → sauber** · **4 Prozesse (nach P1) → 75–77 von 80** · **disjunkte Karten → 400/400 exakt**. Heute läuft die Instanz mit **2 Prozessen × 8 Threads**, das Fenster besteht also prozess- **und** thread-seitig.

**Der Schaden ist nicht abstrakt.** `rating_history` ist die Quelle für zwei Dinge in [app_pkg/learn.py](../../../app_pkg/learn.py): `count_done_today` klassifiziert neu-gegen-Wiederholung am **ersten** Eintrag, und `true_retention` rechnet über ein 30-Tage-Fenster. Verschwundene Einträge verfälschen beide **still**. Ausgerechnet `true_retention` hat schon einmal gelogen — vor LEARN-RATE meldete es 100 %, weil die Knopfbeschriftung falsch war. Eine Kennzahl, die aus zwei verschiedenen Gründen falsch sein kann, ist schlimmer als keine.

**Praktisch trifft es Oli heute kaum** — ein Mensch bewertet eine Karte zur Zeit. Das ist ein Grund für Sorgfalt statt Eile, kein Grund, es liegen zu lassen.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

**Die Struktur** ([models.py:368](../../../models.py)): `Review` ist 1:1 zur Karte — Skalare (`due`, `stability`, `difficulty`, `last_reviewed`, `reps`, `lapses`) **plus** `rating_history` als **JSON-Liste in einer Textspalte**.

**Der Schreibweg** ([app_pkg/cards.py:671](../../../app_pkg/cards.py) `api_review_card`): Zeile lesen → `scheduler.apply_rating(current_state, rating)` → Skalare setzen → JSON-Liste lesen, anhängen, zurückschreiben. **Ein** Read-Modify-Write, ohne jeden Schutz. Beide Verlustarten hängen an derselben Zeile: die Skalare verlieren nach Last-Writer-Wins, die Historie verliert kumulativ und nachzählbar.

**Das Messwerkzeug ist committed**: [scripts/verify_concurrency.py](../../../scripts/verify_concurrency.py) — es zählt `rating_history` **nach**, statt Statuscodes zu glauben. **Benutzen, nicht neu bauen.**

**Der Migrations-Hausweg** ([app_pkg/__init__.py](../../../app_pkg/__init__.py) `_run_pending_migrations`): inspector-geprüftes `ALTER TABLE … ADD COLUMN` + commit; neue Tabellen kommen aus `db.create_all()`. Kein Alembic (Memory `reference_inline_sqlite_migration`).

**Vermutete Geschwister — ungemessen, das ist der Punkt**: `learn.write_settings_keys` (⚠️ der **geteilte** Settings-Blob: ein verlorener Schreibvorgang löscht einen ganzen fremden Namensraum — genau die Fehlerklasse, vor der der LEARN-UP-Bullet warnt), `last_read_percent` ([app_pkg/library.py](../../../app_pkg/library.py), Furthest-Read-Semantik — ein verlorener Schreibvorgang lässt den Fortschritt nur nachhinken), und der Docwrite-Section-Replace.

## Gesperrte Entscheidungen

1. **Erst messen, dann anfassen.** Die drei Geschwister sind Verdacht, kein Befund. Ein Pfad, dessen Rennen nicht nachgewiesen ist, wird in diesem Sprint nicht umgebaut.
2. ⚠️ **Kein pauschales `BEGIN IMMEDIATE`.** Das serialisierte alle Reads und nähme zurück, was SYNC-FREEZE gerade gewonnen hat (Sonden von 3–9 ms zurück auf Warteschlange).
3. **Verifiziert wird an Ergebnissen, nicht an Statuscodes.** Der Befund entstand bei 3.200 × 200.
4. **Eine Versionsspalte schützt die ganze Zeile** — Skalare und Historie zusammen. Eine eigene `rating_event`-Tabelle wäre strukturell sauberer und der `Review`-Docstring sieht sie ausdrücklich vor, aber sie bräuchte eine Datenmigration für 206 Karten **und** den Umbau beider Konsumenten. **Nicht in diesem Sprint** — als Möglichkeit benennen, nicht bauen.
5. **Der Retry ist die richtige Antwort, kein Notbehelf.** FSRS ist deterministisch für (Zustand, Bewertung); nach erneutem Lesen wendet die zweite Bewertung auf das **Ergebnis** der ersten an — genau das soll passieren.

---

# Phase 1 — Wie weit reicht es?

## 1.1 Grundlinie bestätigen

Fahr [scripts/verify_concurrency.py](../../../scripts/verify_concurrency.py) gegen den **heutigen** Stand (2 Prozesse × 8 Threads) und halte fest, wie groß der Verlust jetzt ist. Die P2-Zahlen stammen aus dem 4-Prozess-Zustand.

## 1.2 Die drei Verdächtigen messen

Für `write_settings_keys`, `last_read_percent` und den Docwrite-Section-Replace: **jeweils ein Lauf, der das Ergebnis nachzählt.** Beim Settings-Blob ist die Frage nicht „gehen Schreibvorgänge verloren", sondern **„verschwindet ein fremder Namensraum"** — genau das ist der Schaden, und genau das muss der Test prüfen.

⚠️ Denk daran, dass nicht jeder Verlust gleich viel wiegt. Furthest-Read ist monoton: ein verlorener Schreibvorgang lässt den Balken nachhinken und korrigiert sich beim nächsten Scrollen. Das ist ein anderer Befund als ein gelöschter Settings-Namensraum. **Sortier die Ergebnisse nach Schaden, nicht nach Häufigkeit.**

## 1.3 Was der Bericht liefern muss

Je Pfad: Rennen nachgewiesen ja/nein, mit Zahlen · welcher Schaden entsteht · ob er sich selbst korrigiert. Daraus folgt der Umfang von Phase 3.

## Stop
Zahlen im Bericht. **Commit + Push** falls Messskripte entstanden (sie gehören ins Repo, wie `verify_concurrency.py`). Dann warten.

---

# Phase 2 — Der Review-Pfad wird sicher

## 2.1 Optimistisches Sperren

Eine Versionsspalte auf `Review`, SQLAlchemys `version_id_col` ist der idiomatische Weg — der UPDATE trägt dann die Bedingung, und ein Konflikt kommt als `StaleDataError` an, statt still zu gewinnen. Migration nach dem Hausweg (inspector-geprüftes `ALTER TABLE`).

⚠️ **Der Retry braucht eine frische Transaktion**: nach dem Konflikt neu lesen, FSRS erneut anwenden, erneut schreiben. Begrenz die Versuche und entscheide bewusst, was nach dem letzten fehlgeschlagenen passiert — ⚠️ **ein stiller Verlust ist genau das, was der Sprint abschafft**; eine ehrliche Fehlermeldung ist besser als ein leises Vergessen.

## 2.2 Der Beleg

Derselbe Lauf wie in 1.1, mit demselben Skript: **80 von 80** Einträgen je Karte. Und die Gegenprobe, dass der Gewinn aus SYNC-FREEZE steht: die Sonden aus [scripts/measure_sync_blocking.py](../../../scripts/measure_sync_blocking.py) bleiben im Millisekundenbereich. ⚠️ Wenn die Sperre die Nebenläufigkeit zurücknimmt, ist sie falsch gebaut.

## Stop
80/80 belegt, Sonden unverändert, `pytest tests/` grün (Baseline **963**). **Commit + Push** `fix(LOST-UPDATE): optimistisches Sperren auf dem Review-Schreibweg (P2)`. Dann warten.

---

# Phase 3 — Die bestätigten Geschwister, dann Wrap

## 3.1 Nur, was Phase 1 nachgewiesen hat

Dieselbe Mechanik auf die Pfade, deren Rennen **gemessen** ist. Für jeden davon derselbe Beleg: nachgezählt, nicht zugesichert.

Für Pfade, bei denen das Rennen nachweisbar ist, der Schaden sich aber selbst korrigiert: **benennen und stehen lassen** ist eine legitime Entscheidung — sie gehört dann in den Bericht und ins BACKLOG, nicht in den Code.

## 3.2 Wrap

- **CLAUDE.md**: die Nebenläufigkeits-Notiz nennt LOST-UPDATE heute als offene Folge von SYNC-FREEZE — nachziehen, mit dem, was jetzt geschützt ist und was bewusst nicht.
- **STATUS.md**, **BACKLOG.md** (Bullet-Guard): Item schließen, die `rating_event`-Tabelle als benannte Möglichkeit hinterlassen.
- **Memory**, falls übertragbar; nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.
- **Im Bericht benennen**: die Zahlen je Pfad aus Phase 1 · was geschützt wurde und was nicht, mit Begründung · dass die Sonden nach dem Umbau unverändert sind.

## Nicht-Ziele

- **Keine** `rating_event`-Tabelle, **keine** Datenmigration der Historie.
- **Kein** pauschales `BEGIN IMMEDIATE`, **kein** Zurückdrehen von Prozessen oder Threads.
- **Kein** Umbau des Schedulers oder der Lern-Logik — der Schreibweg wird geschützt, die Mathematik nicht angefasst.
- **Kein** Anfassen von Pfaden, deren Rennen nicht gemessen wurde.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen; Wegwerf-User am Ende **strikt nach `user_id`** abräumen.
