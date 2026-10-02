# SPRINT JOB-ID-REUSE — jede Job-Datei trägt die Auftragsmarke im Namen

**Größe**: M (3 Phasen: Bau mit roten Tests zuerst · Deploy + Belege · Wrap) · **Datum**: 2026-10-02 · **Herkunft**: Befund W-4 aus [ARCH-AUDIT](../audit-outputs/AUDIT_ARCHITEKTUR_2026-10-01.md) (Schweregrad 3, Rang 1 der Priorisierung), BACKLOG-Item JOB-ID-REUSE

## Warum

`Conversion.id` ist `Integer primary_key` ohne `sqlite_autoincrement` ([models.py](../../../models.py) Z. 95; in der Prod-DB gibt es keine `sqlite_sequence`, gemessen 2026-10-02). SQLite vergibt nach dem Löschen der höchsten Zeile **dieselbe id erneut**. Alle Job-Dateien auf dem geteilten Volume heißen nur nach dieser id — `source_<id>.<ext>`, `result_<id>.json`, `narration_<id>.wav` —, und alle drei Reconciles ordnen ein Ergebnis **file-first allein über `conversion.id`** zu ([narration.py](../../../app_pkg/narration.py) Z. 142, [document_api.py](../../../app_pkg/document_api.py) Z. 320, [audio.py](../../../app_pkg/audio.py) Z. 149). `api_delete_conversion` ([library.py](../../../app_pkg/library.py) Z. 585–598) löscht die Zeile, räumt nur die Narrations-WAV und bricht keinen Job ab.

Der Bedienweg, der es auslöst, ist der natürlichste Korrekturweg der App: **falsche Datei hochgeladen → `pending`-Zeile gelöscht → richtige Datei hochgeladen.** Die neue Zeile bekommt dieselbe id, und der alte Auftrag liefert ihr sein Ergebnis — `ready`, richtiger Titel, **fremder Inhalt**, kein Fehler. Der ARCH-AUDIT-Sub-Thread hat das Ende-zu-Ende reproduziert (Dokument und Transkription), 0 Tests decken es.

## Warum nicht der Zuschnitt aus dem Befund (Master, 2026-10-02 — am Code hergeleitet)

Befund und BACKLOG-Item schlagen vor: *der Worker schreibt seine `job_id` ins Ergebnis-JSON, der Reconcile verwirft ein fremdes.* Das schließt die stille Fehlzuordnung, **aber es erwischt den eigenen Fall nicht**, sobald B eingereicht wird, während A noch läuft — und genau das ist der Normalfall („falsche Datei, sofort gelöscht, richtige hinterher"):

1. Submit A (id 1, `m4a`) → `source_1.m4a`, Job A läuft (liest die Datei, wartet auf Deepgram).
2. Zeile 1 gelöscht. Submit B → wieder id 1 → `source_1.m4a` ist jetzt **B's Datei**, Job B wartet hinter A.
3. Job A endet: schreibt `result_1.json` und läuft in sein `finally` — `os.remove(source_path)` ([tasks.py](../../../tasks.py), beide Tasks) über den **nur id-abhängigen** Pfad → **löscht B's Quelle**.
4. Job B startet: `FileNotFoundError` → B endet `failed`.

Aus dem stillen Fremdinhalt würde ein lauter Fehlschlag der richtigen Datei — besser, aber der Korrekturweg bleibt kaputt. Bei der Narration dasselbe Muster ohne Lärm: rendert A noch, wenn B (gleiche id) eingereicht wird, landet A's `narration_<id>.wav`, B's Reconcile sieht „Datei da" → `ready` mit A's Audio und A's Dauer, bis B's Render sie überschreibt.

Die Ursache ist nicht das fehlende Etikett im Ergebnis, sondern **dass zwei Aufträge sich einen Dateinamen teilen können**. Der Zuschnitt dieses Sprints bindet deshalb die **Namen** an den Auftrag. Größe M statt S.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-10-02)

- **Drei Job-Typen, strukturgleich (Option B):** Submit legt Zeile an → erster Commit → `task_queue.enqueue(…)` → zweiter Commit schreibt `metadata['job_id']` (RQ-eigene id). Stellen: [document_api.py](../../../app_pkg/document_api.py) Z. 495–545, [audio.py](../../../app_pkg/audio.py) Z. 318–362, [narration.py](../../../app_pkg/narration.py) Z. 287–322 (Submit) und Z. 401–448 (Retry, ein Commit nach enqueue). `enqueue` steht in den drei Submits ungeschützt **nach** dem ersten Commit: ein Redis-Fehler ergibt 500, eine `pending`-Zeile ohne `job_id` und eine liegengebliebene Quelle (bis 100 bzw. 500 MB).
- **Pfad-Helfer, alle nur über die id:** [services/document_conversions.py](../../../services/document_conversions.py) (`doc_source_path`, `doc_result_path`, `write_result_file` atomar über `.tmp` + `os.replace`, `read_result_file`, `discard_job_files`), [services/transcription_jobs.py](../../../services/transcription_jobs.py) (dieselben fünf), [services/narration_library.py](../../../services/narration_library.py) (`narration_audio_path`, `delete_narration_audio`).
- **Tasks** ([tasks.py](../../../tasks.py)): `convert_document_task(conversion_id, source_ext, mode, budget_eur, page_count)`, `transcribe_audio_task(conversion_id, source_ext, language)` — beide mit `finally: os.remove(source_path)`; `generate_narration_task(conversion_id, turns, voices, style_prompt, mode, language_code, model_name)` legt die WAV per **`shutil.move`** aus `/tmp` auf das Volume (Z. 230 — über Dateisystemgrenzen ein Kopieren, nicht atomar: ein Reconcile im Kopierfenster sieht eine halbe WAV und setzt `ready`). `get_current_job` ist importiert, die Tasks laufen in den Tests aber in-process ohne RQ-Kontext.
- **Reconciles:** Dokument und Transkription lesen `result_<id>.json`, setzen den Inhalt über `Conversion.set_content` und rufen `discard_job_files`; Narration prüft `os.path.exists(narration_<id>.wav)`. Kein Ergebnis → `fetch_job(job_id)`: `NoSuchJobError`/kein `job_id` → `failed` „Job nicht mehr auffindbar.", `is_failed` → `failed` mit Traceback-**Tail**, sonst bleibt `pending` — **auch bei einem beendeten Job ohne Datei** (heute nicht erreichbar, nach diesem Sprint denkbar).
- **RQ:** `Queue.enqueue(…, job_id=…)` wird auf dem Mac (rq 1.16.0) **und** auf dem Pin (rq 2.8.0, im Worker-Container nachgesehen) akzeptiert. `DEFAULT_RESULT_TTL` 500 s — ein fertiger Job ist nach gut acht Minuten aus Redis weg, deshalb ist „Datei zuerst" richtig und bleibt.
- **Prod-Volume (lesend, 2026-10-02):** 2 `narration_*.wav` zu 2 `ready`-Narrationen, **keine** Waisen; `doc_conversions/` und `transcriptions/` leer; **0 `pending`-Zeilen** (Narration 2 ready · Transkription 34 ready + 31 Legacy ohne Job-Schlüssel · Dokument 4 ready); `max(id)` 259. Der Fehler hat in Prod keine Spur hinterlassen, das Deploy-Fenster ist frei.
- **Namen in Doku und Tests:** `result_<id>.json`/`source_<id>.<ext>`/`narration_<id>.wav` stehen in CLAUDE.md (4 Stellen) und [docs/narration_reframe.md](../../narration_reframe.md) (3); die Pfad-Helfer oder Literal-Namen in `tests/test_document_api.py` (11 Zeilen), `test_transcriptions.py` (13), `test_narration_library.py` (12), `test_narration_serve.py` (5), `test_narration_task.py` (5) und vereinzelt weiteren.
- **Baseline:** 1341 passed + 1 skipped, Mac und Container-Pin. Prod-Image `converter-app:latest` = `8bf21a3e2f35`, Rollback-Tag `pre-notion-tz` (alt, darf mit seinem Image weg).

## Gesperrte Entscheidungen

1. **Die Invariante.** *Der Worker liest und schreibt nur unter Namen, die die Auftragsmarke tragen; der Reconcile sucht nur nach dem Namen seines eigenen Auftrags.* Kein Job-Artefakt hängt mehr allein an der Zeilen-id. Damit sind Quelle **und** Ergebnis zweier Aufträge per Konstruktion verschieden, und das `finally` eines Tasks kann nur seine eigene Quelle löschen.
2. **Auftragsmarke = die RQ-Job-id, vom Web erzeugt.** `job_id = str(uuid.uuid4())` im Submit, **bevor** die Quelle abgelegt wird; `task_queue.enqueue(task, …, job_id=job_id)`; derselbe Wert steht in `metadata['job_id']` (Schlüssel und Antwortfeld `job_id` unverändert, das Format bleibt eine uuid). Die Marke reist als **explizites Task-Argument**, nicht über `get_current_job()` — die Tasks müssen in-process testbar bleiben.
3. **Namen:** Quelle und Ergebnis von Dokument und Transkription heißen nur nach der Marke (`source_<job>.<ext>`, `result_<job>.json`, in den bestehenden Unterverzeichnissen). Die Narration: der Worker schreibt seine WAV unter einem **Auftragsnamen in einem eigenen Unterverzeichnis** desselben Volumes (kein Name, der je mit `narration_<id>.wav` verwechselbar ist), über `<name>.part` + `os.replace`; der **Web-Reconcile adoptiert** die fertige Datei per `os.replace` auf den Artefaktnamen `narration_<id>.wav` und setzt dann `ready`. Der Artefaktname, `audio_filename` in den Metadaten und der Serve-Pfad bleiben, bestehende Zeilen und ihre WAVs sind nicht betroffen. Die genauen Namen wählst du; die Invariante und „Adoption nur der eigenen Marke" sind fest.
4. **Submit-Reihenfolge: erst einreihen, dann die Zeile.** Marke erzeugen → Quelle unter dem Auftragsnamen ablegen → `enqueue` → Zeile mit `job_id` in **einem** Commit. Kein offener DB-Schreibvorgang während des Redis-Aufrufs (kein `flush` vor `enqueue` — die id wird für keinen Dateinamen mehr gebraucht; die Narration braucht sie nur für `audio_filename`, das sich nach dem Einfügen setzen lässt). Folge: beim Einreihen steht `conversion_id` noch nicht fest — die Tasks brauchen sie nicht mehr (ihre Logs nennen die Marke), und `job.meta['conversion_id']` entfällt oder wird nach dem Commit nachgetragen; vorher nachsehen, ob irgendetwas `job.meta` liest (der Befund fand im Repo keinen Leser). Scheitert `enqueue`: Quelle entfernen, **503** `{"error": "Auftrag konnte nicht eingereiht werden. Bitte erneut versuchen."}`, **keine Zeile**. Scheitert der Commit danach: der Job läuft, schreibt unter einer Marke, die niemand sucht; die Quelle räumt der Task. Der Retry der Narration bekommt dieselbe Mechanik (neue Marke, Fehler → 503, Zeile bleibt `failed`).
5. **Delete räumt den eigenen Auftrag.** `api_delete_conversion` liest **vor** dem Löschen Typ, `job_id` und Quell-Endung und räumt **nach** dem Commit die Dateien dieser Marke (Quelle, Ergebnis, unfertige Job-WAV) und wie bisher das Narrations-Artefakt. **Kein Abbruch laufender RQ-Jobs** (bleibt draußen). Benannte Eigenschaft: ein Auftrag, der beim Löschen schon läuft, schreibt danach eine Ergebnisdatei, die niemand mehr liest — eine Waise je solchem Fall, per Marke nie zuordenbar; der Wrap nennt den Operator-Befehl, der sie zählt.
6. **Reconcile:** unverändert „Datei zuerst", nur unter dem Namen der eigenen Marke. Neu und erlaubt, weil sonst eine Zeile ewig `pending` bliebe: **eigener Job beendet, eigene Datei fehlt → `failed`** („Ergebnis nicht auffindbar.") — nur wenn es ohne Wettlauf geht (der Worker schreibt die Datei vor dem Job-Ende); wenn nicht belegbar, weglassen und im Bericht benennen. Zwei Reconciles hintereinander oder gleichzeitig dürfen eine adoptierte Narration nicht auf `failed` kippen. Transienter Redis-Fehler bleibt `pending`, Traceback-Tail, `set_content` — alles wie heute.
7. **Altbestand:** `ready`- und `failed`-Zeilen bleiben unberührt (terminal). Eine `pending`-Zeile im alten Namensschema endet nach dem Deploy `failed` — deshalb gehört „0 `pending`, Queue leer" zum Deploy-Fenster (heute erfüllt).
8. **Rot zuerst.** Die Abläufe unten stehen als Tests, **bevor** der Fix entsteht, und sind gegen HEAD rot gezeigt (Ausgabe in den Bericht). Kein roter Stand wird gepusht.
9. **Kontrakte:** Metadaten-Schlüssel, Antwortformen und Env-Namen bleiben. Neu ist nur die 503-Ursache „nicht eingereiht" (vorher ein nackter 500) — eine Zeile in [docs/document_api_contract.md](../../document_api_contract.md) §9; für die Narration im Wrap prüfen, wo Fehlercodes dokumentiert sind.
10. **Nicht in diesem Sprint:** das gemeinsame Job-Gerüst (W-3), Abbruch laufender Jobs, `sqlite_autoincrement`/Tabellen-Umbau, Status-Vokabular, ein Orphan-Sweeper, Dedup-Änderungen.

**Arbeitsweise:** inline, **kein Workflow, keine Subagenten** ohne Olis ausdrückliches Wort. Nach jeder Phase Commit + Push, dann Stop + Bericht. Nichts Tragendes im Session-Scratch — Belege gehören in den Bericht oder ins Repo. Editiert wird nur auf dem Mac.

---

# Phase 1 — Rote Tests, dann der Fix

## 1.1 Die Abläufe als Tests (zuerst, gegen HEAD rot)

Mit echter SQLite-id-Wiederverwendung (höchste Zeile löschen, neu anlegen — **prüfen, dass die id wirklich gleich ist**, sonst testet der Fall nichts) und den Tasks in-process:

- **(a) Altes Ergebnis kommt nach dem Löschen:** Submit A → Zeile löschen, solange `pending` → Task A läuft durch → Submit B (gleiche id) → Poll B bleibt `pending`; nach Task B ist B `ready` **mit B's Inhalt**. Je für Dokument und Transkription.
- **(b) Ergebnis lag schon, nie gepollt:** Submit A → Task A fertig, nie gepollt → Zeile löschen → Submit B (gleiche id) → Poll B: nicht `ready` mit A's Inhalt.
- **(c) B kommt, während A läuft (der Fall, den der Befund-Zuschnitt nicht fängt):** Submit A → Zeile löschen → Submit B (gleiche id, **gleiche Endung**) → Task A läuft durch (samt `finally`) → Task B läuft durch und findet **seine** Quelle → B `ready` mit B's Inhalt. Je für Dokument und Transkription.
- **(d) Narration:** A rendert noch, B wird mit gleicher id eingereicht, A's WAV landet → B bleibt `pending`; nach B's Render ist B `ready`, und `narration_<id>.wav` trägt **B's** Bytes und Dauer. Dazu: `pending`, solange nur `.part` liegt; ein zweiter Reconcile nach der Adoption ist ein No-op.
- **(e) `enqueue` wirft:** 503, keine Zeile, keine Quelle auf dem Volume — für alle drei Submits; der Narrations-Retry lässt die Zeile `failed`.
- **(f) Delete räumt:** nach dem Löschen einer `pending`-Zeile liegen weder Quelle noch Ergebnis ihrer Marke; das Narrations-Artefakt einer `ready`-Zeile ist weg (Bestand).
- **Sentinels:** `enqueue` wird mit genau der `job_id` gerufen, die in den Metadaten steht; kein Pfad-Helfer der drei Job-Module nimmt noch eine `conversion_id` für einen Job-Dateinamen (das Narrations-**Artefakt** ausgenommen).

## 1.2 Der Fix

Nach den gesperrten Entscheidungen 1–7, in den bestehenden Modulen (keine neuen gemeinsamen Module — das Gerüst ist ein eigenes Item): Pfad-Helfer und `discard_job_files` in den drei Service-Modulen, die drei Tasks, die drei Submits plus Narrations-Retry, die drei Reconciles, `api_delete_conversion`. Bestehende Tests an die neuen Signaturen anpassen, ohne ihre Aussage zu verdünnen.

**Beifang in denselben Dateien (klein, sonst nichts):** im Narrations-Player die **letzte** statt der ersten Fehlerzeile ([static/js/library_detail.js](../../../static/js/library_detail.js) Z. 1693–1696 — die Logik steht 80 Zeilen tiefer schon, Z. 1775–1776); das nackte `'ready'` in [app_pkg/narration.py](../../../app_pkg/narration.py) Z. 362 auf `NARRATION_STATUS_READY`; im Kopf-Docstring von [app_pkg/audio.py](../../../app_pkg/audio.py) Z. 8–9 die Behauptung „its only caller was this page's JS" richtigstellen (die iOS-App ruft die entfernte Route weiterhin, BACKLOG IOS-TRANSCRIBE-ROUTE); die Modul-Docstrings mit dem Datei-Layout auf den neuen Stand.

## 1.3 Gates

- `python3 -m pytest tests/ -q -p no:cacheprovider` grün am Mac; Zahl gegen die Baseline 1341 + 1 Skip nennen.
- **Container-Suite auf dem Pin** (rq 2.8.0 — `enqueue(job_id=…)` und `fetch_job` gehören dorthin): das stdin-Rezept aus CLAUDE.md mit `COPYFILE_DISABLE=1 tar --no-xattrs`, keine Datei auf der Mintbox.
- Die Tests aus 1.1 einmal gegen HEAD **rot** gezeigt (welche, mit welcher Meldung), nach dem Fix grün.
- Commit(s) + Push. Kein Deploy in dieser Phase.

## Stop
Bericht: die roten Läufe vor dem Fix (je Ablauf eine Zeile) · die gewählten Namen und wo die Marke entsteht · Diff-Umfang je Datei · Suite Mac und Container · was du anders gelöst hast als hier beschrieben und warum · was nicht ging. Dann warten.

---

# Phase 2 — Deploy + Belege

1. **Fenster:** Queue leer, Worker idle, kein `mineru_*`-Container, **0 `pending`-Zeilen** (DB nur `mode=ro`), `doc_conversions/` und `transcriptions/` leer. Vorher-Inventar des Volumes in den Bericht.
2. **DB-Backup** nach dem Rezept in CLAUDE.md (`sqlite3.backup()` im Container, `chmod 600`, `setfacl -b`), Kopie nur mit `mode=ro&immutable=1` prüfen. Rollback-Tag `docker tag converter-app:latest converter-app:pre-job-id-reuse`; `pre-notion-tz` darf samt Image weg. **Nie `docker image prune -a`.**
3. `git pull --ff-only`, `docker compose up -d --build` aus dem Projektverzeichnis; mit `docker inspect` belegen, was neu angelegt wurde; Login 200.
4. **Smokes:** [scripts/smoke_document_converter.py](../../../scripts/smoke_document_converter.py) und [scripts/smoke_audio_converter.py](../../../scripts/smoke_audio_converter.py) (Wegwerf-User, Aufräumen strikt nach `user_id`).
5. **Der eigene Fall, live:** ein kleines Probe-Skript `scripts/probe_job_id_reuse.py` im Stil der Smokes (im Web-Container, Wegwerf-User): PDF A einreichen (Modus `lokal`, läuft rund eine Minute) → Zeile löschen, solange `pending` → **anderes** PDF B einreichen → **ausgeben, ob B dieselbe id bekam** (wenn nicht, ist der Fall nicht geprüft — so berichten) → warten, bis B `ready` ist → Inhalt ist B's, nicht A's. Danach das Volume inventarisieren: was von A liegen blieb, mit Namen. Aufräumen strikt nach `user_id`.
6. **Eine Narration Ende-zu-Ende** (der Schreibweg der WAV ändert sich, die Tests mocken TTS): ein kurzer Satz, ein Sprecher, über `POST /api/narrations` mit dem Token aus der Container-Env (**nie ausgeben**) → pollen bis `ready` → Dauer > 0, `narration_<id>.wav` existiert, das Job-Unterverzeichnis ist leer → die Zeile **strikt nach ihrer id** samt WAV wieder entfernen (sie liegt auf Olis Konto). Kostet einen TTS-Aufruf.
7. Nachher-Inventar: keine Job-Datei ohne Zeile außer der in 5 benannten Waise (falls eine entstand — dann entfernen und sagen).

Sicherheits-Regeln wie immer: `docker exec` nie mit `-u 0`; DB-Lesen `mode=ro`; keine Token, Schlüssel oder Cookie-Werte in Ausgaben; keine unversionierte Datei auf der Mintbox.

## Stop
Bericht: Fenster · neu angelegte Container und Image-ID · beide Smokes · Probe mit der id-Zeile und dem Inhaltsvergleich · Narration E2E · Vorher-/Nachher-Inventar · dass Wegwerf-User und Probe-Zeilen weg sind.

---

# Phase 3 — Wrap

- **CLAUDE.md:** die Stellen, die `result_<id>.json`, `source_<id>.<ext>` und den Schreibweg der Narrations-WAV nennen (Faithful-Narration, DIARIZE/SYNC-FREEZE P3, DOC-API), auf die Invariante bringen — ein Satz je Stelle, kein neuer Roman; dazu die benannte Waisen-Eigenschaft mit dem Operator-Befehl und die Test-Baseline. Die Sätze über `GeminiService`, Singletons und Factory **nicht** anfassen (ARCH-NARR5, ARCH-FACTORY).
- **[docs/document_api_contract.md](../../document_api_contract.md)** §9: die 503-Zeile. [docs/narration_reframe.md](../../narration_reframe.md): die drei Namens-Stellen.
- **STATUS.md**, **BACKLOG.md** (JOB-ID-REUSE schließen; ⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber). Im geschlossenen Item festhalten, dass der Zuschnitt vom Befund abweicht und warum (Abschnitt „Warum nicht …" oben).
- **Brief ans converter-mcp:** die Agent-Fläche `POST /api/narrations` ändert sich nur im Fehlerfall (503 statt 500 bei nicht erreichbarer Queue). Vor dem Urteil die Tool-Liste am lebenden Connector lesen; wenn kein Tool den Statuscode auswertet, im Wrap sagen, warum kein Brief.
- **Memory** nur bei übertragbarer Lehre. Kandidat: *ein Etikett im Ergebnis schützt die Zuordnung, nicht die Datei — teilen sich zwei Aufträge einen Namen, löscht der eine dem anderen die Quelle; die Marke gehört in den Namen.* Vorher prüfen, was `reference_narration_async_db_free_reconcile` und `feedback_guard_rail_must_catch_its_own_case` schon tragen.
- Commit + Push, Stop + Bericht.

## Nicht-Ziele

Kein Job-Gerüst, kein RQ-Abbruch, kein Schema-Touch, kein Sweeper, keine Dedup-Änderung, kein Eingriff in die Engines, kein Frontend außer der einen Fehlerzeile. Nichts an `app.py`-Singletons oder am Schlüssel-Bind (ARCH-NARR5).
