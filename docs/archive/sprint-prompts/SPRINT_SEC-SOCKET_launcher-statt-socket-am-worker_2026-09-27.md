# SPRINT SEC-SOCKET — der Socket wandert vom Worker in einen Launcher, der nur eines kann

**Größe**: L (3 Phasen) · **Datum**: 2026-09-27 · **Vorhaben**: SEC-AUDIT-Folge, Finding F-13 ([Befund-Doc](../audit-outputs/AUDIT_SECURITY_2026-09-25.md)), Top-3-Risiko

## Warum

Der Worker mountet `/var/run/docker.sock`, um für den lokalen PDF-Pfad (DOC-LOCAL) den Geschwister-Container `mineru:3.4.4` zu starten. Der Socket ist **root-äquivalent auf dem Host**: wer im Worker Code ausführt — und der Worker parst fremde Dokumente mit pandoc, unstructured, PyMuPDF, mineru —, ist mit einem `docker run -v /:/host` Root über die ~30 Container der Box. SEC-REDIS-AUTH hat den bequemsten Zünder (Pickle über Redis) entfernt; der Verstärker steht noch. Die DOC-LOCAL-Entscheidung „GPU nur während des Auftrags, kein Dauer-Sidecar" bleibt richtig; was sich ändert, ist **wer** den Socket hält.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-09-27 — nicht neu herleiten, Abweichungen benennen)

**Worker heute** (`docker inspect markdown-converter-worker`): Mounts `/var/run/docker.sock`, `./doclocal_exchange → /app/doclocal_exchange`, `google-credentials.json`, Volume `podcast_data`; Netz nur `converter_default`; Env `MINERU_IMAGE=mineru:3.4.4`, `MINERU_MODELS_DIR=/home/oliver/bakeoff-models`, `DOC_LOCAL_EXCHANGE_DIR=/app/doclocal_exchange`, `DOC_LOCAL_EXCHANGE_HOST_DIR=/home/oliver/CODE/CONVERTER/doclocal_exchange`. Läuft als root (SEC-NONROOT, eigenes Item). Das Image trägt die docker-**CLI** 27.5.1 als statisches Binary ([Dockerfile:93–101](../../../Dockerfile)), keinen Daemon.

**Host**: Docker 29.8.1, API 1.56, Runtimes `runc` + `nvidia`, cgroup v2/systemd, **nicht** rootless. nvidia-container-toolkit 1.20.1. GPU-Nutzer heute: `accounting_celery_worker`, `muncher-worker`, dazu Olis ComfyUI zeitweise. `mineru:3.4.4` = `6cc9e57ff5bd`, 29,7 GB, lokal geladen (kein Registry-Digest; `mineru:latest` zeigt auf dieselbe ID).

**Die zwei `docker run` des Workers** ([services/pdf_local.py:221–246](../../../services/pdf_local.py), `_run_mineru_container`):
1. `docker run --rm --gpus all --shm-size 16g -v <in_host>:/in:ro -v <out_host>:/out [-v <models_host>:/models] -e HF_HOME=/models -e MINERU_MODEL_SOURCE=huggingface mineru:3.4.4 mineru -p /in/doc.pdf -o /out -b vlm-engine` — der Bake-off-Vektor, **wörtlich gepinnt** in [tests/test_pdf_local.py:174–190](../../../tests/test_pdf_local.py) (`test_invocation_vector_is_the_measured_one`, prüft benachbarte Paare + `:/in:ro`).
2. Danach `docker run --rm -v <out_host>:/out busybox chown -R <uid>:<gid> /out` — ⚠️ `busybox` **schwebend**, und `<uid>:<gid>` ist `os.getuid()` des **Workers**, heute `0:0` (root) — der Pass ist auf der Mintbox faktisch ein No-op und wird erst mit SEC-NONROOT wieder tragend.

**Aufrufer** `_ensure_run` ([pdf_local.py:325–371](../../../services/pdf_local.py)): legt `<exchange>/mineru_<hex12>/{in,out}` an, schneidet das Teil-PDF als `doc.pdf`, ruft `_run_mineru_container(in_host, out_host, 'doc.pdf', mineru_run_timeout_for(n))`, liest `out/*_content_list.json`. **Jede** Exception → `_failed=True`, Degradation `backend_fallback` „Lokale Engine fehlgeschlagen. Textebene übernommen. (<Grund>)", bei `subprocess.TimeoutExpired` lautet der Grund „Zeitlimit <n> s überschritten."; `finally: rmtree(job_dir)`. Timeouts: `mineru_run_timeout_for = 300 + 10·n` (pdf_local), RQ-Umschlag `doc_convert_job_timeout_for(n,'lokal') = 600 + 10·n` ([app_pkg/config.py:121–132](../../../app_pkg/config.py)), Invariante Umschlag > Deadline ist testgenagelt.

⚠️ **Vorbestehender Defekt, den der Umbau mitnimmt**: `subprocess.run(timeout=)` tötet den **docker-CLI-Client**, nicht den Container. Ein mineru-Lauf, der die Deadline reißt, läuft auf dem Host **weiter** (kein `--name`, kein `docker rm -f`), hält die GPU und schreibt später in ein Verzeichnis, das `rmtree` schon entfernt hat.

**Tests**: `tests/test_pdf_local.py` fährt `_run_mineru_container` über eine `fake_docker`-Fixture (fängt `subprocess.run` ab, liefert `content_list`); `tests/test_document_api.py` prüft den Umschlag. Suite-Baseline **1186 + 1 Skip** (Mac: rq 1.16, Pin 2.8.0 — Suite auf dem Pin im Container fahren, s. Phase 1).

**Warum die alte Stufe 1 nicht trägt**: ein Socket-Proxy à la `tecnativa/docker-socket-proxy` filtert **Pfad und Methode**, nie den Body. `POST /containers/create` mit `HostConfig.Binds=["/:/host"]` oder `Privileged:true` geht durch, sobald `create` erlaubt ist — und `create` **muss** erlaubt sein. Der Proxy nimmt `exec`, `images`, fremde `inspect` weg (Randhärtung), nicht die Root-Äquivalenz. Eine Body-Policy gäbe es nur als Daemon-AuthZ-Plugin (host-weit, trifft Oli und alle anderen Projekte) oder rootless (GPU-Risiko neben ComfyUI). Deshalb:

## Gesperrte Entscheidungen

1. **Ein Launcher-Sidecar hält den Socket, sonst niemand.** Neuer Compose-Service `mineru-launcher` aus **demselben Image** `converter-app:latest` (hat die docker-CLI schon), `command: python -m services.mineru_launcher`. Mounts: **nur** `/var/run/docker.sock`. Kein Exchange-Mount — er reicht Host-Pfade an `docker run` durch, er muss die Dateien nicht sehen. Env: `MINERU_IMAGE`, `MINERU_MODELS_DIR`, `DOC_LOCAL_EXCHANGE_HOST_DIR`, `EXCHANGE_OWNER=<uid>:<gid>` (heute `0:0`; SEC-NONROOT setzt es später) — sie wandern vom Worker zum Launcher.
2. **Der Worker verliert den Socket und die Host-Pfad-Envs.** `docker-compose.yml`: Socket-Mount und `DOC_LOCAL_EXCHANGE_HOST_DIR` raus aus `worker`. Die docker-CLI bleibt im Image (der Launcher braucht sie; Dockerfile unverändert).
3. **Der Launcher kann genau eine Sache**: `POST /run` mit `{"job": "<mineru_hex12>", "pdf_name": "doc.pdf", "page_count": n}`. Er **baut die Kommandozeile selbst** aus seiner Env — der Worker liefert **Daten, nie Argumente**. Validierung fail-closed: `job` matcht `^mineru_[0-9a-f]{12}$`, `pdf_name` ist genau ein Pfadsegment ohne `/`, `..`, Steuerzeichen; `page_count` ist eine kleine positive Ganzzahl; unbekannte Felder → 400. Der Bake-off-Vektor zieht **wörtlich** in ein pures Modul (`services/mineru_invocation.py`: Argv-Bau + Timeout-Konstanten, keine Flask-/Docker-Imports), aus dem Launcher **und** `pdf_local` (für die Timeout-Rechnung) lesen; der Sentinel-Test zieht mit und prüft dort weiter dieselben Paare.
4. **Der Launcher besitzt den Timeout — und tötet.** `docker run --name <job> …` mit `subprocess.run(timeout=mineru_run_timeout_for(n))`; bei `TimeoutExpired` **`docker rm -f <job>`**, bevor die Antwort geht. Kein Lauf darf den Launcher als Waise überleben. Der chown-Pass bleibt (mit `EXCHANGE_OWNER`), `busybox` wird **gepinnt** (Tag oder Digest, im Bericht begründen).
5. **Einer nach dem anderen.** Der Launcher fährt höchstens **einen** mineru-Lauf gleichzeitig (Lock); ein zweiter `POST /run` bekommt **409** und der Worker behandelt das wie jeden Fehler (Fallback auf die Textebene, Grund benannt). Das ist heute schon die Realität (ein Worker, ein Job) und begrenzt, was ein kompromittierter Worker anrichten kann: mineru auf Dateien im Exchange-Verzeichnis laufen lassen — nichts, was er nicht ohnehin dürfte.
6. **Antwort ist JSON**: `{returncode, timed_out, stdout_tail, stderr_tail}` (Tails ≤ 800 Zeichen wie heute). Der Client in `pdf_local` mappt `timed_out` auf eine eigene Exception, damit `_ensure_run` weiter „Zeitlimit <n> s überschritten." meldet; Launcher nicht erreichbar / 409 / 5xx → derselbe Fallback-Pfad wie heute jeder Fehler. **Kein** neues Verhalten für den Nutzer, nur eine andere Fehlerquelle im Grund-Text.
7. **Netz**: eigenes Compose-Netz `converter_launch` mit **nur** `worker` und `mineru-launcher`; der Launcher hängt in **keinem** anderen Netz (kein Redis, kein Web, kein Internet nötig — `docker run` geht über den Socket). Kein Token zwischen Worker und Launcher: gegen einen kompromittierten Worker hilft er nicht (der ist der legitime Client), gegen andere Container hilft das Netz. Healthcheck `GET /health` → 200; `worker` mit `depends_on: mineru-launcher: condition: service_healthy`.
8. **Minimale Oberfläche im Launcher**: Stdlib `http.server` (ein Thread reicht bei Entscheidung 5), zwei Routen, striktes JSON, `subprocess.run` mit **Listen**-Argumenten, kein Shell, keine weiteren Imports aus `app_pkg`. Der Launcher ist der neue Ort der Root-Äquivalenz — er darf nichts parsen, was aus einem Dokument stammt.
9. **Compose-Sentinel** wie in [tests/test_proxy_fix.py](../../../tests/test_proxy_fix.py): `docker.sock` kommt in der Compose-Datei **genau einmal** vor und nur unter `mineru-launcher`; `worker` hat keinen Socket-Mount und kein `DOC_LOCAL_EXCHANGE_HOST_DIR`; `mineru-launcher` hängt nur in `converter_launch`.
10. ⚠️ **Editiert wird nur auf dem Mac.** Mintbox = Runtime; Deploy nur in einem Fenster ohne laufende Lokal-Konvertierung; Wegwerf-User strikt nach `user_id` (`api_token` trägt Olis und den MCP-Token); DB-Snapshot vorher nach dem WAL-Rezept.

---

# Phase 1 — Bauen und lokal belegen

## 1.1 Bauen

`services/mineru_invocation.py` (pur: `build_run_argv(...)`, `build_chown_argv(...)`, `mineru_run_timeout_for`, Konstanten; `pdf_local` importiert die Timeout-Funktion von dort, `app_pkg/config.py` ebenso, damit die Umschlag-Invariante an **einer** Quelle hängt) · `services/mineru_launcher.py` (Server; Entscheidungen 3–8) · in `pdf_local.py` ersetzt ein HTTP-Client `_run_mineru_container` (Ziel aus Env `MINERU_LAUNCHER_URL`, Standard `http://mineru-launcher:8765`; Timeout des HTTP-Calls = Deadline + Puffer, damit der Launcher den Kill immer vor dem Client-Timeout schafft) · Compose (neuer Service, Netz, Worker-Änderungen, Healthcheck, `depends_on`) · Dockerfile **unverändert**.

## 1.2 Beleg

- **Launcher-Tests** (ohne Docker): Validierung fail-closed (Traversal `../x`, Slash, leerer Name, falsches Job-Muster, Extra-Feld, `page_count` 0/negativ/String → 400) · Argv exakt aus der Env, Client-Felder tauchen **nur** als `/in/<pdf_name>` und `--name <job>` auf · 409 bei laufendem Lauf · Timeout → `docker rm -f <job>` wird gerufen und `timed_out:true` geliefert (`subprocess.run` gefakt) · chown-Pass mit `EXCHANGE_OWNER`.
- **Sentinel zieht um**: `test_invocation_vector_is_the_measured_one` prüft dieselben Paare an `build_run_argv`; dazu `:/in:ro` und der gepinnte `busybox`.
- **Client-Tests in `test_pdf_local.py`**: die `fake_docker`-Fixture wird zu einem gefakten Launcher (gepatchter HTTP-Client **oder** ein Stdlib-Server im Thread — deine Wahl, im Bericht begründen); alle bestehenden Fälle bleiben grün, plus: Launcher nicht erreichbar → Fallback mit Grund; 409 → Fallback; `timed_out` → „Zeitlimit … überschritten.".
- **Compose-Sentinel** (Entscheidung 9) und die Umschlag-Invariante gegen die neue Quelle.
- `pytest tests/` grün auf dem Mac (Baseline **1186 + 1**) **und** im Container auf dem Pin (Wegwerf-Container aus dem frisch gebauten Image, `tests/` + `Dockerfile` + `docker-compose.yml` gemountet, ohne Netz).
- **Lokaler Funktionsbeleg ohne GPU**: auf dem Mac den Launcher mit einem Fake-`MINERU_IMAGE` (z. B. `busybox`, Kommando läuft durch, schreibt nichts) einmal real über den Socket fahren — zeigt, dass `docker run --name` + `rm -f` + chown im echten Docker funktionieren. Wenn der Mac-Docker das nicht hergibt, im Bericht sagen; der echte Beleg ist Phase 2.

## Stop
Commit + Push (Code + Tests; Compose gern eigener Commit). Bericht: Modul-Schnitt, welche Werte der Worker noch sendet, wie der Kill belegt ist, Deploy-Plan mit Fenster-Prüfung. Dann warten.

---

# Phase 2 — Deploy-Fenster und Beleg an der laufenden Instanz

## 2.1 Fenster

Vorher lesend: keine laufende oder wartende Lokal-Konvertierung (Queues, `rq:wip`, Worker `idle`, keine `pending`-Conversion mit `job_id`), kein `mineru`-Container aktiv (`docker ps`), in den letzten zehn Minuten kein `Claude-User` in den nginx-Logs von `converter-mcp`/`mail-mcp`. DB-Snapshot nach dem WAL-Rezept. `git pull --ff-only`, `docker compose up -d --build` (Launcher startet, wird `healthy`, dann Worker). Auszeit im Bericht in Sekunden.

## 2.2 Messung

- **Der Socket ist weg**: `docker exec markdown-converter-worker ls /var/run/docker.sock` → *No such file*; `docker inspect mineru-launcher` zeigt den Socket-Mount und **keinen** Exchange-Mount; Netze: Worker `converter_default` + `converter_launch`, Launcher nur `converter_launch`; `curl http://mineru-launcher:8765/health` aus dem **Web**-Container scheitert (kein Netz), aus dem Worker → 200.
- **Ein echter Lokal-Lauf**: Wegwerf-User, ein **Scan-PDF** (das 15-Seiten-Exemplar aus DOC-LOCAL oder ein kleines aus `corpus/`) per `POST /api/document-conversions` mit `mode=lokal` → `ready`, `provenance` `modell`, Text vorhanden; Laufzeit im Bericht (Referenz: 61 s Start + 2,5 s/Seite). Während des Laufs `docker ps` zeigt den Container **mit dem Job-Namen**; danach ist er weg, das Exchange-Verzeichnis geräumt, kein `root`-Rest darin.
- **Der Kill**: einen Lauf mit künstlich kurzer Deadline erzwingen (Env-Override der Basis-Konstante **nur** am Launcher, für die Probe) → Antwort `timed_out:true`, `docker ps -a` zeigt **keinen** `mineru_*`-Container mehr, die Conversion endet `ready` mit `backend_fallback` „Zeitlimit … überschritten." (Textebene). Override danach zurück, Launcher neu gestartet, im Bericht.
- **Blast-Radius-Gegenprobe**: aus dem Worker `POST /run` mit `pdf_name: "../../etc/passwd"` und mit einem zusätzlichen Feld `argv` → 400, nichts gestartet (`docker ps -a` unverändert).
- Aufräumen strikt nach `user_id`; Prod-Zahlen vorher/nachher gleich.

## Stop
Bericht mit Messwerten. Dann warten.

---

# Phase 3 — Wrap

- **CLAUDE.md**: DOC-LOCAL-Bullet — „Geschwister-Container über den Host-Docker-Socket" wird „über den Launcher; der Socket liegt **nur** dort"; die gesperrte Entscheidung (GPU nur während des Auftrags) bleibt, ihr Preis (Socket am Worker) ist weg; Kill-bei-Timeout als neue Eigenschaft; `mineru_invocation.py` als der eine Ort des Vektors. Sicherheits-Bullet: F-13 geschlossen, Root-Äquivalenz jetzt im Launcher mit zwei Routen, warum ein pfadbasierter Proxy nicht gereicht hätte. Kopfsatz („offen als Items SEC-SOCKET …") anpassen. Test-Baseline.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): SEC-SOCKET schließen; SEC-NONROOT um den Hinweis ergänzen, dass `EXCHANGE_OWNER` dann den Worker-uid bekommt; im Befund-Doc F-13 auf **geschlossen**.
- **Kein Brief ans converter-mcp** (keine Agent-Fläche) — so sagen.
- **Memory** bei übertragbarer Lehre; Kandidat: *Ein Socket-Proxy filtert Pfade, nicht Bodies — `containers/create` mit `Binds=["/:/host"]` bleibt Root; die Allow-List muss ein Dienst sein, der die Kommandozeile selbst baut und vom Client nur Daten nimmt.* Nur mit den Messwerten aus Phase 2.
- **Im Bericht**: Auszeit · Laufzeit des Scan-Laufs vorher (DOC-LOCAL-Referenz) und nachher · dass kein `mineru_*`-Container überlebt hat · was der Worker noch an Envs trägt · Wegwerf-User weg.

## Nicht-Ziele

- **Kein** rootless Docker, **kein** Daemon-AuthZ-Plugin, **kein** Dauer-Sidecar mit geladener GPU.
- **Kein** Non-Root-Umbau der Container (SEC-NONROOT), **keine** Änderung an mineru-Version, Modellen oder Invokations-Paaren (der Sentinel bleibt wörtlich).
- **Kein** Umbau der Cloud-Engine, des Routers oder der Degradations-Form.
- **Keine** Passwörter/Tokens im Repo; keine unversionierten Dateien auf der Mintbox.
