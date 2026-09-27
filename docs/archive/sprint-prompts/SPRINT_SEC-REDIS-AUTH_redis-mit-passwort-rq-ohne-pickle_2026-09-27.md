# SPRINT SEC-REDIS-AUTH — Redis mit Passwort, RQ ohne Pickle

**Größe**: S (2 Phasen) · **Datum**: 2026-09-27 · **Vorhaben**: SEC-AUDIT-Folge, Finding F-7 ([Befund-Doc](../audit-outputs/AUDIT_SECURITY_2026-09-25.md))

## Warum

F-7 des Security-Audits: Redis läuft ohne Passwort, und RQ serialisiert Jobs mit Pickle. Wer einen Fuß ins Docker-Netz `converter_default` bekommt, schiebt einen präparierten Pickle-Job in die Queue, der Worker de-pickelt ihn — Code-Ausführung im Worker, und der hält den root-äquivalenten Docker-Socket (F-13). Redis hat keinen veröffentlichten Port, das ist heute die einzige Schicht. Dieser Sprint zieht zwei weitere ein: Redis verlangt ein Passwort, und Jobs reisen als JSON, das nichts ausführt.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-09-27 — nicht neu herleiten, Abweichungen benennen)

**Redis live** (`file-transformer-redis`): `redis:alpine` **schwebend**, gemessen **8.4.0**; `requirepass` leer, `protected-mode no`, `bind * -::*`; RDB-Snapshots aktiv (`save 3600 1 300 100 60 10000`), `appendonly no`; ⚠️ **anonymes Volume unter `/data`** — ein Recreate behält es, die Alt-Daten überleben also. `DBSIZE 15`: 5 `rq:job:*` + 5 `rq:results:*` (**gepickelt**), Worker-Schlüssel, `rq:failed:default`; alle Queues leer. Verbunden: Web (`172.21.0.3`) und Worker (`172.21.0.4`), sonst niemand — das Netz `converter_default` trägt nur Web, Worker, Redis. Kein weiterer Redis-Nutzer im Code (`grep`: nur `app.py`, `worker.py`).

**Verdrahtung** ([docker-compose.yml](../../../docker-compose.yml)): Redis-Service Zeile 3–5 ohne `command`, ohne Healthcheck; `REDIS_URL=redis://redis:6379` **fest im `environment:`** von Web (Z. 42) und Worker (Z. 84); `depends_on: - redis` ohne `condition`. `.env` trägt **keinen** `REDIS_*`-Schlüssel; sie gehört seit 26.09. `oliver`, Modus `600`, per ssh **in place** beschreibbar (`>>` hält den Inode; ⚠️ `~/CODE` trägt eine Default-ACL, jede *neu* angelegte Datei erbt sie — nie `.env` neu anlegen, immer anhängen).

**RQ hängt an drei Stellen, heute alle mit Default-Serializer (Pickle):**
1. [app.py:63–65](../../../app.py): `REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')` · `redis_conn = Redis.from_url(REDIS_URL)` · `task_queue = Queue(connection=redis_conn)`.
2. [worker.py](../../../worker.py): `Queue(name, connection=conn)` + `Worker(queues, connection=conn)` — unter `if __name__ == '__main__'`, also **nicht testbar importierbar**.
3. `Job.fetch(job_id, connection=_app_module.redis_conn)` in den beiden Reconciles [app_pkg/audio.py:173](../../../app_pkg/audio.py) und [app_pkg/document_api.py:357](../../../app_pkg/document_api.py). ⚠️ Beide fangen `NoSuchJobError` → `failed` und **jede andere Exception → Warnung + `return`** (bleibt `pending`). RQ liest beim `refresh()` das `meta`-Feld **mit dem Serializer** — ein Serializer, der nur an Queue und Worker gesetzt ist, lässt `Job.fetch` an genau dieser Stelle scheitern, und der Reconcile schweigt: der Auftrag bleibt **für immer `pending`**, kein Fehler, kein Log außer einer Warnung. Das ist der Grund, warum es **einen** Serializer-Ort geben muss (Entscheidung 3). Tests patchen `app.Job.fetch` und `app.task_queue` ([tests/conftest.py:215–240](../../../tests/conftest.py)).

**Was über die Queue reist** (vier `enqueue`-Stellen): `generate_narration_task(conversion.id, turns, voices, style_prompt, mode, language_code, tts_model)` ([narration.py:305](../../../app_pkg/narration.py), Retry :431), `transcribe_audio_task(conversion_id, source_ext, language)` ([audio.py:337](../../../app_pkg/audio.py)), `convert_document_task(conversion_id, source_ext, mode, budget_eur, …)` ([document_api.py:521](../../../app_pkg/document_api.py)); überall `meta={'user_id', 'conversion_id'}` und `job_timeout`. Worker-seitig schreibt `update_job_stage` ([tasks.py:20](../../../tasks.py)) `job.meta['stage']` + Extras. Rückgaben sind Payload-Dicts (`build_result_payload`) bzw. `None`. Pins: `rq==2.8.0`, `redis==7.4.0`. `rq.serializers.JSONSerializer` existiert im Pin.

**Ergebnisse liegen nicht in Redis**: Worker schreiben `result_<id>.json`/WAV aufs geteilte Volume, die Web-Seite rekonziliert **file-first** (Option B). Redis trägt nur Job-Buchführung.

## Gesperrte Entscheidungen

1. **Passwort aus `.env`** (`REDIS_PASSWORD`), erzeugt mit `secrets.token_urlsafe(32)`, **auf der Mintbox in place angehängt**, nie ausgegeben, nie im Repo. Compose interpoliert es **fail-closed**: `${REDIS_PASSWORD:?REDIS_PASSWORD fehlt in .env}` an **jeder** Stelle — ohne Schlüssel startet nichts, statt dass Web/Worker passwortlos gegen einen gesperrten Redis laufen.
2. **Redis bekommt es per Env**, nicht als Klartext-Argument im Compose: `environment: REDIS_PASSWORD=${…}` + `command: ["sh", "-c", "exec redis-server --requirepass \"$$REDIS_PASSWORD\""]`. Folge, benannt: im Container zeigt `ps` das Passwort in den Argumenten (root-only), `docker inspect` zeigt es in `Env` wie jedes andere Secret der Box. Eine Conf-Datei mit `requirepass` wäre die Alternative (Secret auf Platte statt in argv) — deine Wahl, im Bericht begründen. `save`-Verhalten unverändert lassen.
3. **Ein Serializer, ein Ort, drei Stellen.** `RQ_SERIALIZER = JSONSerializer` als **eine** Konstante (Vorschlag [app_pkg/config.py](../../../app_pkg/config.py)); `Queue(…, serializer=RQ_SERIALIZER)` in `app.py`, `Queue`+`Worker` in `worker.py`, und ein **geteilter Helper** `fetch_job(job_id)` (in `app.py` neben `redis_conn`), den **beide** Reconciles statt des rohen `Job.fetch` rufen — mit `serializer=RQ_SERIALIZER`. Kein Aufrufer darf den Serializer einzeln setzen können. ⚠️ Der Test-Patch-Punkt `app.Job.fetch` bleibt für die Tests erhalten (der Helper ruft ihn).
4. **`worker.py` wird testbar**: Aufbau in eine Funktion (`build_worker(conn)`), `__main__` ruft sie. Nur dieser Umbau, kein weiterer.
5. **JSON muss alles tragen, was heute reist**: die Argumente der vier `enqueue`-Stellen, `meta`, `update_job_stage`-Extras, Rückgabe-Payloads. Bekannte Änderung: Tupel kommen als Listen zurück. Prüfen, ob irgendein Task-Argument oder -Rückgabewert `bytes`, `datetime`, `set` oder ein Tupel trägt, das als Tupel gebraucht wird — dann ist das der Sprint-Befund, nicht ein Umbau des Tasks (benennen, Master entscheidet).
6. **Deploy nur in einem leeren Fenster**: Queues leer **und** keine `pending`-Conversion mit `job_id` in der Prod-DB (`mode=ro`). Die gepickelten Alt-Schlüssel im anonymen Volume räumt ein `FLUSHDB` **vor** dem Start des JSON-Workers — Buchführung, keine Ergebnisse (die liegen als Dateien); im Bericht sagen, wie viele Schlüssel weg sind.
7. **Image pinnen**: `redis:8.4-alpine` (= gemessene 8.4.0), weil der Service ohnehin angefasst wird. Healthcheck mit `REDISCLI_AUTH=$$REDIS_PASSWORD redis-cli ping` und `depends_on: condition: service_healthy` für Web und Worker sind erlaubt und erwünscht.
8. **Beleg am Ende ist ein echter Job**, nicht ein `PING`: siehe Phase 2.
9. ⚠️ **Editiert wird nur auf dem Mac.** Mintbox = Runtime; die einzige Runtime-Schreibung ist die `.env`-Zeile. Wegwerf-User strikt nach `user_id` abräumen (`api_token` trägt Olis und den MCP-Token).

---

# Phase 1 — Bauen und lokal belegen

## 1.1 Bauen

Compose (Redis `command`/`environment`/`healthcheck`/Pin, `REDIS_URL=redis://:${REDIS_PASSWORD:?…}@redis:6379/0` an Web **und** Worker, `depends_on` mit `condition`), die Serializer-Konstante, `app.py` (Queue + `fetch_job`-Helper), `worker.py` (Funktion), beide Reconciles auf den Helper. Der Default `redis://localhost:6379` in `app.py` bleibt (Tests, Mac-Dev ohne Compose).

## 1.2 Beleg

- **Sentinels**: `app.task_queue.serializer is RQ_SERIALIZER` · `build_worker(conn)` liefert Worker und Queues mit `RQ_SERIALIZER` · beide Reconciles gehen durch `fetch_job` (z. B. Mock auf `fetch_job` statt `Job.fetch`, oder Assertion auf die `serializer=`-kwarg am gepatchten `Job.fetch`) · Compose-Sentinel nach dem Muster in [tests/test_proxy_fix.py](../../../tests/test_proxy_fix.py): Redis-Service pinnt `8.4-alpine`, trägt `requirepass`, beide `REDIS_URL` tragen `:${REDIS_PASSWORD` mit `:?`.
- **JSON-Roundtrip**: für jede der vier `enqueue`-Stellen ein Test, der repräsentative Argumente (aus den bestehenden Test-Fixtures — `turns`/`voices` der Narration sind die reichsten) durch `RQ_SERIALIZER.dumps/loads` schickt und die Gleichheit prüft; dazu `meta` und ein `build_result_payload`-Ergebnis.
- **Der stille Hänger als Test**: ein Job mit Pickle-Serializer geschrieben, mit `fetch_job` (JSON) gelesen → muss laut werden (Exception), damit klar ist, was ein vergessener Ort kostet. Braucht Redis: `fakeredis` ist **nicht** installiert — wenn ohne echten Redis nicht machbar, den Fall im Container gegen den echten Redis als Skript-Beleg fahren und im Bericht zeigen.
- `pytest tests/` grün, Baseline **1170 + 1 Skip**.

## Stop
Commit + Push (Code + Tests). Bericht: der Serializer-Ort, was JSON nicht trug (falls etwas), und **der Deploy-Plan für Phase 2** mit den Prüfungen aus Entscheidung 6. Dann warten — die Runtime-Schreibung und der Flush brauchen den Master-Blick.

---

# Phase 2 — Deploy-Fenster, Messung, Wrap

## 2.1 Fenster

Vorher messen und im Bericht zeigen: `LLEN` aller Queues = 0, keine `pending`-Conversion mit `job_id` (DB `mode=ro`), kein laufender Worker-Job (`rq:workers`-Status). Dann: `REDIS_PASSWORD` in place an die `.env` anhängen (nie ausgeben; danach `ls -l .env` zeigt weiter `-rw------- oliver`), `git pull`, `FLUSHDB` mit Zählung, `docker compose up -d --build`. Reihenfolge Redis → Worker → Web ergibt sich aus `depends_on`.

## 2.2 Messung an der laufenden Instanz

- `redis-cli PING` **ohne** Auth → `NOAUTH`; mit `REDISCLI_AUTH` → `PONG`. Healthcheck `healthy`.
- Aus dem Web-Container: `redis.from_url(os.environ['REDIS_URL']).ping()` → `True`.
- **Ein echter Job durch den Worker**: Wegwerf-User, kleine TXT-Datei per `POST /api/document-conversions` (Session + CSRF oder per-User-Bearer des Wegwerf-Users) → 202 → pollen bis `ready`, `provenance` sichtbar. Damit sind Enqueue-Serialisierung, Worker-Deserialisierung, `meta`-Schreiben und Reconcile über `fetch_job` in **einem** Lauf belegt. Zusätzlich einen Fehlerpfad: den Job eines `pending`-Wegwerf-Auftrags per `redis-cli DEL rq:job:<id>` entfernen → nächster `GET` liefert `failed` „Job nicht mehr auffindbar." (der `NoSuchJobError`-Zweig funktioniert mit JSON).
- Wegwerf-User samt Conversions strikt nach `user_id` löschen; Container-Reste räumen.
- Gegenprobe Blast-Radius: ein Container **außerhalb** von `converter_default` erreicht Redis weiter nicht (unverändert), und **innerhalb** ohne Passwort nur noch `NOAUTH` (z. B. `docker run --rm --network converter_default redis:8.4-alpine redis-cli -h redis PING`).

## 2.3 Wrap

- **CLAUDE.md**: im Bullet *Die Web-Instanz hinter nginx* (Sicherheit) ein Satz zu F-7: Redis mit Passwort aus `.env`, RQ mit **einem** JSON-Serializer an drei Stellen, warum ein vergessener Ort ein stiller Hänger ist; unter *Running*: `REDIS_PASSWORD` ist Pflicht, Compose startet ohne ihn nicht; Hinweis für Operatoren, dass `rq info`/`rq`-CLI jetzt `--serializer rq.serializers.JSONSerializer` brauchen. Tests-Baseline aktualisieren.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): SEC-REDIS-AUTH schließen; im Befund-Doc die Zeile F-7 in „Stand der Findings" auf **geschlossen** setzen (ein Satz, Commit-Hash).
- **Kein Brief ans converter-mcp** (keine Agent-Fläche berührt) — im Wrap so sagen.
- **Memory** nur bei übertragbarer Lehre. Kandidat: *Ein Serializer-Wechsel in RQ ist ein Drei-Stellen-Problem (Queue, Worker, `Job.fetch`), und die vergessene Stelle hängt still, weil der Reconcile Exceptions schluckt — eine Konstante, ein Helper, drei Sentinels.* Nur schreiben, wenn der Test aus 1.2 das wirklich gezeigt hat.
- **Im Bericht benennen**: gewählte Passwort-Übergabe (Env/argv oder Conf) mit Begründung · Zahl der geflushten Schlüssel · die Messwerte aus 2.2 · dass `.env` weiter `600 oliver` ist · dass der Wegwerf-User weg ist.

## Stop
Commit + Push (Wrap eigener Commit). Dann warten.

## Nicht-Ziele

- **Kein** Umbau der Tasks oder ihrer Argumente (ein JSON-Befund wird benannt, nicht stillschweigend gelöst).
- **Kein** Redis-Volume-Umbau, **keine** Persistenz-Änderung, **kein** veröffentlichter Port, **keine** ACL-User jenseits des Default-Users.
- **Kein** Touch an SEC-SOCKET/SEC-NONROOT (eigene Items), **kein** nginx.
- **Keine** Passwörter, Tokens oder URLs mit Credentials im Repo oder im Bericht.
