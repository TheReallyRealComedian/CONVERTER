# SPRINT SYNC-FREEZE — die Instanz hört auf, bei jeder langen Anfrage stehenzubleiben

**Größe**: L (3 Phasen) · **Datum**: 2026-08-22 · **Vorhaben**: Betrieb

## Warum

In DOC-WEB-ASYNC P1 wurde gemessen, dass der Web-Prozess **eine** Anfrage zur Zeit bedient. Der Dokument-Knopf wurde daraufhin in die Queue verlegt — die **Ursache** blieb. Sie trifft weiter alles andere:

**`POST /transcribe-audio-file`** ruft Deepgram **synchron im Web-Prozess** ([app_pkg/audio.py:51](../../../app_pkg/audio.py)), mit `TIMEOUT_DEEPGRAM_SECONDS = 1200`. Eine lange Transkription friert damit die **gesamte Instanz** ein — Library, Review, iOS-App, bis zu zwanzig Minuten. Das ist kein Randfall: Oli transkribiert regelmäßig Diktate.

Dazu die zweite Ausprägung: bei zwei Anfragen im Millisekundenabstand auf derselben Keep-Alive-Verbindung entsteht sporadisch ein **500** (`RuntimeError: Single thread executor already being used, would deadlock`) — im Review also genau dort, wo am schnellsten geklickt wird.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten, alles verifiziert)

**Die Ursache, an der installierten Quelle belegt**: asgiref 3.8.1, [`asgiref/wsgi.py:134`](https://github.com/django/asgiref) — `run_wsgi_app` trägt ein **nacktes** `@sync_to_async`, also `thread_sensitive=True`, also `executor = self.single_thread_executor` (`ThreadPoolExecutor(max_workers=1)`). Der Docstring sagt „Called in **a** subthread". ⚠️ **`--threads` hilft nicht** — die Serialisierung sitzt in asgiref, je Prozess. Nur mehr **Prozesse** helfen.

**Die Messung liegt vor und das Werkzeug auch**: [scripts/measure_sync_blocking.py](../../../scripts/measure_sync_blocking.py). Zahlen: Leerlauf 5–7 ms · während einer 78-s-Konvertierung Median 28,3 s / Max 73,3 s (**jede** Sonde, auch `/login` ohne Session) · zwei gleichzeitige Läufe strikt seriell (77,3 s / 154,4 s, zweiter Handler 5 ms nach dem ersten). **Benutze das Skript, bau es nicht neu.**

**RAM ist keine Schranke** (gemessen 2026-08-22): Web-Container **121 MB** RSS, Worker 19 MB, Host 62 GB mit 49 GB verfügbar. Vier Prozesse kosten ~0,5 GB. Die RAM-Warnung im Backlog-Item ist damit gegenstandslos.

⚠️ **SQLite ist die eigentliche Vorbedingung.** `PRAGMA journal_mode` liefert **`delete`** — der Rollback-Journal-Modus, in dem ein Schreiber die ganze Datei exklusiv sperrt und Leser blockiert. Es gibt **keine** `SQLALCHEMY_ENGINE_OPTIONS`, **keine** `connect_args`, **kein** gesetztes `busy_timeout` (nur pysqlites Default von 5 s). Solange **ein** Prozess läuft, fällt das nie auf. Mit mehreren Prozessen, die gleichzeitig schreiben (Bewertungen, Markierungen, Einstellungen), wird daraus Sperrkonkurrenz. **WAL vor Workern, nicht danach.**

**Prozess-lokaler Zustand ist unkritisch**: einziger Fund ist ein `functools.lru_cache` für die KaTeX-Assets ([app_pkg/markdown.py:32](../../../app_pkg/markdown.py)) — jeder Prozess baut ihn einmal selbst, kein geteilter Zustand.

**Der Worker trägt `DEEPGRAM_API_KEY` bereits** ([docker-compose.yml](../../../docker-compose.yml), `environment:`) — die Verlagerung braucht **kein** neues Geheimnis.

**Die Vorlage ist einen Commit alt**: der Dokument-Konverter hat genau diese Migration in DOC-WEB-ASYNC P2 durchlaufen — einreichen, pollen, rendern, mit Zähler und Dedup-Hinweis; dazu [scripts/smoke_document_converter.py](../../../scripts/smoke_document_converter.py) als reproduzierbarer UI-Beleg. **Kopieren, nicht neu erfinden.**

**Job-Mechanik Option B**, zweimal getragen (NARR-3, DOC-API): der Worker mountet **kein** `app_data`, hat also **keine DB**. Er arbeitet auf `podcast_data`, schreibt ein strukturiertes Ergebnis, die Web-Seite rekonziliert file-first.

**Die übrigen langen synchronen Pfade**: der Playwright-Markdown→PDF-Weg ([app_pkg/markdown.py](../../../app_pkg/markdown.py)) und Kindle/EPUB. Ihre Dauer ist **ungemessen** — Phase 3 misst, bevor irgendwer sie anfasst.

## Gesperrte Entscheidungen

1. **WAL vor Workern.** Mehrere Prozesse auf einer Rollback-Journal-Datenbank tauschen ein Einfrieren gegen `database is locked` — das wäre kein Fortschritt.
2. **Verifiziert wird mit dem bestehenden Messskript**, mit denselben Sonden, damit die Zahlen vergleichbar sind.
3. **Die Transkription folgt Option B**; der Worker bleibt **DB-frei**.
4. **Das Frontend kopiert die Dokument-Konverter-Migration.** Kein zweiter Polling-Mechanismus.
5. **Keine zweite Queue** (Oli, 2026-08-21): ein Nutzer, er wartet höchstens auf sich selbst.

---

# Phase 1 — Mehrere Prozesse, sicher gemacht

## 1.1 WAL und Sperrverhalten zuerst

`journal_mode=WAL` und ein ausdrückliches `busy_timeout` beim Verbindungsaufbau setzen (SQLAlchemy-`connect`-Event ist der Hausweg; SQLite merkt sich WAL **in der Datei**, der Modus überlebt also Neustarts).

⚠️ **Vorher ein Backup der Prod-DB ziehen** — sie liegt im Volume `converter_app_data`, nicht im Home; `docker cp` ist der Weg (Memory `reference_mintbox_prod_db_backup`). Ein Journal-Moduswechsel ist reversibel, aber ein Backup vor dem ersten Schreibpfad-Eingriff seit Monaten ist billig.

## 1.2 Dann mehr Prozesse

`--workers N` in [Dockerfile:99](../../../Dockerfile). N ist deine Wahl **mit Begründung** — RAM ist keine Schranke (121 MB je Prozess), also entscheidet, wie viele lange Anfragen gleichzeitig laufen können sollen, ohne dass die Instanz zäh wird.

⚠️ Prüf, was mit mehreren Prozessen bricht: Sessions und CSRF (beide cookie-/secret-basiert, sollten tragen — **belegen**, nicht annehmen), die RQ-Anbindung, und ob irgendein Start-Code doppelt läuft, der das nicht verträgt (`_run_pending_migrations`, `db.create_all`).

## 1.3 Der Beleg

Denselben Messlauf wie in DOC-WEB-ASYNC P1 fahren — vorher/nachher, dieselben Sonden. Erwartung: eine lange Anfrage blockiert die anderen nicht mehr; N gleichzeitige lange Anfragen laufen parallel, die N+1-te wartet.

⚠️ **Prüf auch, ob der Keep-Alive-`would deadlock` noch reproduzierbar ist.** Mehr Prozesse senken die Wahrscheinlichkeit, beseitigen die Ursache aber nicht. Falls er bleibt: sieh nach, ob ein `asgiref`-Bump ihn adressiert — und wenn ja, ob der Bump verhaltensneutral ist (die CSRF-Inversion repliziert Flask-WTF-Interna, asgiref ist davon unabhängig, aber `WsgiToAsgi` ist Kernpfad). **Ergebnis berichten, nicht stillschweigend bumpen.**

## Stop
Zahlen vorher/nachher, `pytest tests/` grün (Baseline **935**). **Commit + Push** `feat(SYNC-FREEZE): WAL und mehrere Worker-Prozesse (P1)`. Dann warten.

---

# Phase 2 — Die Transkription verlässt den Web-Prozess

Das ist der eigentliche Fix: mehr Prozesse machen das Einfrieren erträglich, die Queue macht es weg.

## 2.1 Der Auftrag

Nach Option B, Vorbild `tasks.convert_document_task`: die Web-Seite legt die Audiodatei auf `podcast_data`, erzeugt eine `pending`-Zeile und enqueued; der Worker transkribiert **DB-frei** und schreibt ein strukturiertes Ergebnis; die Web-Seite rekonziliert beim Pollen.

⚠️ Die Diarisierung und der Chunk-Pfad (`MAX_AUDIO_DURATION_SECONDS = 5400`) bleiben **unverändert** in `services/deepgram_service.py` — sie werden verlagert, nicht umgebaut.

## 2.2 Der Zeit-Umschlag

Der RQ-Umschlag muss den schlimmsten Fall tragen: `TIMEOUT_DEEPGRAM_SECONDS = 1200` je Request, plus Chunking bei über 90 Minuten. Rechne ihn aus der Arbeitsmenge wie `doc_convert_job_timeout_for`, setz ihn, benenne ihn.

## 2.3 Das Frontend

Kopier die Migration des Dokument-Konverters: einreichen, pollen, rendern, Zähler. Die Microcopy-Hausregeln gelten (Fehler max 2 Sätze, keine Emojis).

## 2.4 Der Beleg

Eine echte, ausreichend lange Audiodatei über den **Browser**, und **währenddessen** der Messlauf: die App bleibt bedienbar. Das ist der Nachweis, den der Sprint schuldet.

⚠️ Wegwerf-User wie gehabt, Abbau am Sprint-Ende **strikt nach `user_id`** — Olis iOS-Tokens liegen in derselben Tabelle.

## Stop
Transkription läuft über die Queue, App bleibt bedienbar, belegt. **Commit + Push** `feat(SYNC-FREEZE): Transkription als Hintergrund-Auftrag (P2)`. Dann warten.

---

# Phase 3 — Die restlichen langen Pfade, dann Wrap

## 3.1 Erst messen, dann entscheiden

Der Playwright-Markdown→PDF-Weg und Kindle/EPUB sind die letzten synchronen Kandidaten. **Ihre Dauer ist ungemessen.** Miss sie an realistischen Eingaben (ein langes Dokument aus Olis Library) und entscheide **mit Zahlen**, ob sie in die Queue gehören oder ob N Prozesse reichen.

⚠️ **Nicht auf Verdacht migrieren.** Jede Verlagerung kostet eine UI-Umstellung; wenn ein Pfad in Sekunden fertig ist, ist er in einem Sprint gut aufgehoben, der ihn nicht anfasst.

## 3.2 Wrap

- **CLAUDE.md**: die Architektur-Notiz kennt heute keinen Hinweis auf die Serialisierung — der neue Zustand (WAL, N Prozesse, welche Pfade asynchron sind) gehört dorthin, damit ihn niemand neu herleiten muss.
- **STATUS.md**, **BACKLOG.md** (Bullet-Guard), Kontrakt-Doc **nur falls betroffen**.
- **Engine-Generation NICHT anfassen** — an keinem Konvertierungs-Ergebnis ändert sich etwas.
- **Memory**, falls übertragbar; nach dem Schreiben mit `ls` prüfen, dass Datei und Index-Zeile zusammenpassen.
- **Im Bericht benennen**: die Vorher/Nachher-Zahlen · welches N du gesetzt hast und warum · ob der Keep-Alive-Deadlock überlebt hat · die gemessenen Dauern aus 3.1 und die Entscheidung daraus.

## Nicht-Ziele

- **Kein** Umbau von `services/deepgram_service.py` (Diarisierung, Chunking bleiben, wie sie sind).
- **Kein** zweiter Queue-/Worker-Aufbau.
- **Kein** Wechsel der Datenbank — WAL ist ein Modus, kein Umzug.
- **Kein** Anfassen der Dokument-Konvertierung, der Narration oder des Lern-Layers.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein, keine unversionierten Dateien zurücklassen.
