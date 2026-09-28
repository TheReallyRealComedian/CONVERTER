# SPRINT SEC-NONROOT — Web und Worker laufen als uid 1000, nicht als root

**Größe**: M (3 Phasen) · **Datum**: 2026-09-27 · **Vorhaben**: SEC-AUDIT-Folge, Finding F-8 ([Befund-Doc](../audit-outputs/AUDIT_SECURITY_2026-09-25.md))

## Warum

Web und Worker laufen als root im Container (`id -u` → 0, kein `USER` im [Dockerfile](../../../Dockerfile)). Beide parsen fremde Dokumente (pandoc, unstructured, PyMuPDF, mineru-Ausgabe, Chromium für PDF). Seit SEC-SOCKET führt ein Code-Exec dort nicht mehr zu Host-Root, aber immer noch zu root **im** Container: jede Datei im Image beschreibbar, jeder Prozess killbar, jeder Kernel-Fehler mit voller Wirkung. Der Sprint nimmt die letzte Stufe: die Prozesse laufen als unprivilegierter Nutzer, der Code im Image ist für sie nur lesbar.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-09-27 — nicht neu herleiten, Abweichungen benennen)

**Image** `mcr.microsoft.com/playwright/python:v1.62.0-noble`: kennt **uid 1000 = `ubuntu`** (`/home/ubuntu`, 750) und `pwuser` = 1001. `HOME=/root` heute. `/ms-playwright` ist `drwxrwxrwx`, die Browser sind für jeden lesbar. Das Dockerfile hat **kein `USER`, kein `mkdir`, kein `chown`**; `COPY . .` nach `/app` (root, 755; die Pakete darunter `drwxrwxr-x+`), `ENV PYTHONDONTWRITEBYTECODE 1` (kein `__pycache__`-Schreiben), `CMD gunicorn --bind 0.0.0.0:5000` (Port > 1024). ⚠️ **`/app/data`, `/app/output_podcasts`, `/app/doclocal_exchange` existieren im Image nicht** — sie entstehen erst als Mount-Punkte. Docker legt sie beim ersten Mount als `root:root` an, und ein **frisches** benanntes Volume übernimmt Inhalt **und Besitz** nur von einem Verzeichnis, das im Image existiert. Ohne `mkdir`+`chown` im Dockerfile könnte ein neuer Stack (Mac-Dev, Wiederaufbau) als uid 1000 nicht in seine eigenen Volumes schreiben. Die Prod-Volumes sind davon nicht betroffen (sie existieren), aber die Regel muss ins Image.

**Host = Mintbox**: `oliver` ist **uid 1000**. Der Exchange-Bind `~/CODE/CONVERTER/doclocal_exchange` ist `drwxr-xr-x 1000:1000`, `google-credentials.json` ist `-rw------- 1000:1000` (600). **Ein Container-Nutzer mit uid 1000 liest und schreibt beides ohne jede Host-Änderung.** Die docker-Gruppe hat gid 996.

**Volumes** (root-owned, komplett): `converter_app_data` 18,85 MB (`converter.db` + `-wal` + `-shm`, `converter.db.startup.lock`, ein Alt-`.bak`), `converter_podcast_data` 91,48 MB (zwei Narration-WAVs `-rw-------`, `doc_conversions/` 755, `transcriptions/` 755). Alle 11 Einträge gehören `root`.

**Schreibpfade zur Laufzeit** (Code): `/app/data` (`os.makedirs(..., exist_ok=True)` beim Start, SQLite + WAL + `flock`-Lock daneben, [app_pkg/__init__.py:170–252](../../../app_pkg/__init__.py)), `OUTPUT_DIR=/app/output_podcasts` ([app_pkg/config.py:20](../../../app_pkg/config.py)) mit `doc_conversions/` und `transcriptions/` (Uploads per `NamedTemporaryFile(dir=job_dir)`), `/app/doclocal_exchange` (Worker: `os.makedirs(in/out)`, `rmtree`), `/tmp` (`NamedTemporaryFile` für WAV-Konkatenation, Backup-Rezept `/tmp/backup.db`, Smokes per `docker cp … :/tmp/`). Chromium: [app_pkg/markdown.py:217](../../../app_pkg/markdown.py) `p.chromium.launch()` ohne Argumente — Playwrights Python-API hat `chromium_sandbox=False` als Default (laut API-Doku; im Sprint an der installierten Version prüfen), also `--no-sandbox` heute wie nach dem Wechsel; Chromium legt sein Profil unter `/tmp` an, will aber ein **beschreibbares `HOME`**.

**Launcher** (SEC-SOCKET): läuft als root mit dem Socket (`srw-rw---- root:996`). **Er bleibt root** — ein Prozess mit Socket-Zugriff ist root-äquivalent, egal welche uid er trägt; ein `USER` dort wäre Theater und brächte eine host-spezifische gid (996) in die Compose-Datei. Er kopiert die mineru-Ausgabe als `EXCHANGE_OWNER` (heute `0:0`, [docker-compose.yml](../../../docker-compose.yml)) in das Job-Verzeichnis zurück; mineru selbst schreibt als root in sein Volume. ⚠️ **Ungemessen**: ob mineru seine Ausgabe so anlegt, dass ein Kopierer mit uid 1000 sie **lesen** kann (Dateien 644 / Verzeichnisse 755 → ja; 600/700 → nein). Der Sentinel in [tests/test_compose_socket.py:44](../../../tests/test_compose_socket.py) pinnt `EXCHANGE_OWNER=0:0`.

**Bedienung heute**: `docker exec markdown-converter-web flask create-user|set-password|reset-collection`, Smokes per `docker cp … :/tmp/` + `docker exec … python /tmp/smoke.py`, Backup per `docker exec … python3 -c … /tmp/backup.db` + `docker cp`. Alles läuft heute als root und würde als uid 1000 weiter laufen — **außer** jemand hängt `-u 0` an: dann entstehen root-Dateien in den Datenverzeichnissen, die der uid-1000-Prozess nicht mehr anfassen kann.

**Tests**: Dockerfile lesen [tests/test_db_runtime.py:78](../../../tests/test_db_runtime.py) (Worker-Prozesse ohne `--preload`) sowie die Compose-Sentinels in `test_proxy_fix.py`, `test_rq_serializer.py`, `test_compose_socket.py`. Baseline **1272 + 1 Skip**. ⚠️ Der Mac baut das Image nicht (PEP 668 auf arm64, seit SEC-SOCKET bekannt); Bau und Container-Suite laufen auf der Mintbox **vor** dem Fenster, ohne die laufenden Container anzurühren (`docker compose build` taggt nur; das Muster aus SEC-REDIS-AUTH P2). pytest im Container braucht `-p no:cacheprovider` oder ein beschreibbares Cache-Ziel, weil `/app/tests` als uid 1000 nicht beschreibbar ist.

## Gesperrte Entscheidungen

1. **uid:gid 1000:1000 für Web und Worker.** Nicht weil es bequem ist, sondern weil es die Grenze richtig zieht: was ein Container-Prozess auf dem Host anfassen kann, entscheiden die **Mounts**, nicht die uid — und die einzigen Host-Pfade, die Web/Worker sehen (Exchange-Bind, Credentials read-only), gehören ohnehin uid 1000 und **müssen** erreichbar sein. Eine fremde uid (z. B. 10001) brächte dieselbe Reichweite plus `chown` auf Olis Dateien. Der Nutzer bekommt einen Namen (`converter`; ob per `usermod -l` aus `ubuntu` oder als eigener Eintrag mit uid 1000 — deine Wahl) **und ein beschreibbares `HOME`** (`ENV HOME=/home/converter`), weil Chromium sonst warnt oder fällt.
2. **Dockerfile-Reihenfolge ist tragend**: erst `RUN mkdir -p /app/data /app/output_podcasts /app/doclocal_exchange && chown 1000:1000 …` (damit frische Volumes den Besitz erben), dann `COPY . .` (Code bleibt **root:root, nur lesbar** — der Prozess kann sein eigenes Programm nicht umschreiben, das ist Absicht), dann `USER 1000:1000`, dann `CMD`. Ein Sentinel pinnt: genau ein `USER`, nach dem letzten `COPY`, vor `CMD`; die drei Verzeichnisse im `mkdir`/`chown`.
3. **Launcher bleibt root** (Begründung oben), `EXCHANGE_OWNER=1000:1000`. Der Sentinel in `test_compose_socket.py` wird angepasst. Ob der Copy-out als 1000 die mineru-Ausgabe lesen kann, wird **gemessen** (Phase 2, echter Lauf). Falls nicht: ein dritter Helfer-Schritt als root (`chown -R <owner> /x/<job>/out` über die Austausch-**Wurzel**, symlink-sicher wie die anderen), nicht ein Zurück auf `0:0`.
4. **Migration der Prod-Volumes in einem Zug**: Web + Worker gestoppt, DB-Snapshot nach dem WAL-Rezept vorher, dann `chown -R 1000:1000` über beide Volumes mit dem **gepinnten** busybox (`HELPER_IMAGE` aus `services/mineru_invocation.py`, kein schwebendes Image), dann `up -d`. Rückweg: `USER` raus, `EXCHANGE_OWNER=0:0`, neu bauen — root liest Dateien von 1000 problemlos, die Migration ist also rückwärtskompatibel.
5. **Die Bedienregel wird Doku**: `docker exec` **nie** mit `-u 0`/`--user root` in Web oder Worker. Das Backup-Rezept in CLAUDE.md bleibt gültig (schreibt `/tmp`).
6. **Chromium-Sandbox bleibt aus** wie heute (`chromium_sandbox=False`-Default → `--no-sandbox`). Sie einzuschalten bräuchte User-Namespaces oder `SYS_ADMIN` — eigenes Item, falls je gewünscht, hier **nicht**.
7. ⚠️ **Editiert wird nur auf dem Mac.** Mintbox = Runtime; Bau + Container-Suite dort vor dem Fenster; Wegwerf-User strikt nach `user_id` (`api_token` trägt Olis und den MCP-Token); keine unversionierten Dateien zurücklassen.

---

# Phase 1 — Bauen und auf dem Pin belegen

## 1.1 Bauen

Dockerfile (Nutzer, `HOME`, `mkdir`/`chown`, `USER`; Kommentar sagt, warum die Reihenfolge zählt und warum uid 1000), Compose (`EXCHANGE_OWNER=1000:1000`; sonst nichts), `docs/mac-dev-setup.md` (ein Satz: Volumes eines frischen Stacks gehören 1000; `docker exec` ohne `-u 0`).

## 1.2 Beleg

- **Sentinels**: Dockerfile-Test (Entscheidung 2) · Compose-Sentinel auf `1000:1000` · Launcher-Test `EXCHANGE_OWNER` weiter env-getrieben (existiert, [tests/test_mineru_launcher.py:164](../../../tests/test_mineru_launcher.py)).
- Mac-Suite grün (Baseline **1272 + 1**).
- **Auf der Mintbox, ohne die laufenden Container anzurühren**: `docker compose build` (taggt `converter-app:latest` neu; laufende Container behalten ihre Image-ID — im Bericht mit `docker inspect -f {{.Image}}` belegen, dass sie unverändert sind). Dann im **neuen** Image: `id -u` → 1000, `HOME` beschreibbar, `ls -ldn /app/data /app/output_podcasts /app/doclocal_exchange` → 1000:1000, `/app` root:755, `python -c "import app"` als 1000 läuft (Import ohne DB? sonst mit `DATABASE_URL` auf `/tmp`), Container-Suite grün (`tests/` + `Dockerfile` + `docker-compose.yml` gemountet, `--network none`, `-p no:cacheprovider`).
- **Frisches-Volume-Probe** im neuen Image: `docker run --rm -v probe_data:/app/data <image> sh -c 'id -u; touch /app/data/x && echo schreibbar'` → schreibbar, danach `docker volume rm probe_data`. Das ist der Beleg für Entscheidung 2.
- **Chromium als 1000**: im neuen Image (Wegwerf-Container, `--network none`) ein Playwright-Start mit `chromium.launch()` und `page.pdf()` einer Inline-Seite → PDF-Bytes > 0, keine Sandbox-Fehlermeldung. Damit ist der PDF-Pfad vor dem Fenster belegt.

## Stop
Commit + Push (Dockerfile + Compose + Tests + Doku-Zeile). Bericht: gewählter Nutzername und Weg, die Frisches-Volume-Probe, die Chromium-Probe, Container-Suite auf dem Pin, Deploy-Plan mit Fenster-Prüfung. Dann warten.

---

# Phase 2 — Fenster, Migration, Beleg

## 2.1 Fenster

Wie zuletzt: keine laufende/wartende Konvertierung, `rq:wip` 0, Worker `idle`, keine `pending` mit `job_id`, kein `mineru_*`, kein `Claude-User` in den letzten zehn Minuten auf `converter-mcp`/`mail-mcp`. DB-Snapshot (WAL-Rezept, danach `chmod 600` + `setfacl -b`). Dann: `docker compose stop markdown-converter worker` → `chown -R 1000:1000` über `converter_app_data` und `converter_podcast_data` mit `HELPER_IMAGE` (Zahl der Einträge vorher/nachher; `find … ! -user 1000 | wc -l` → 0 — ⚠️ `-user`, nicht `-uid`, s. Nachtrag) → `git pull --ff-only` (Image ist gebaut) → `docker tag converter-app:sec-nonroot converter-app:latest` → `docker compose up -d`. Auszeit in Sekunden.

> **Nachtrag Master nach dem P1-Bericht (2026-09-27, gegengemessen):** (1) **`! -user 1000`, nicht `! -uid 1000`** — der Sprint-Prompt stand hier falsch. busybox 1.38 `find` kennt `-uid` nicht: Usage auf stderr, exit 1, stdout leer — und `| wc -l` macht daraus eine **0, die wie Erfolg aussieht**. Gemessen an `converter_app_data`: `! -uid 1000` → „0", `! -user 1000` → 6 von 6. Der Zähler ist erst glaubwürdig, wenn er **vor** dem chown die volle Zahl liefert (Ist heute: app_data 6 von 6 nicht-1000, podcast_data 5 von 5, 0 Symlinks; 18,0 + 87,3 MB — der chown dauert Sekundenbruchteile, die Auszeit ist das Neuanlegen der Container). (2) Das Image liegt als **`converter-app:sec-nonroot`** (`ed2260ef3619`), `latest` zeigt weiter auf `0179903fe815` — gewollte Abweichung von 1.1, damit ein versehentliches `up -d` vor dem Fenster nicht als 1000 auf root-eigene Volumes startet; deshalb der `docker tag`-Schritt oben. (3) **Rückweg = `docker tag 0179903fe815 converter-app:latest` + `docker compose up -d`, sonst nichts.** Kein `git revert` auf der Mintbox — sie ist Runtime, Edits und Reverts passieren am Mac und werden gepullt; die neue Compose-Datei läuft auch mit dem alten Image (der Launcher war schon root, eine 1000-eigene mineru-Ausgabe liest ein root-Worker), und root liest die auf 1000 umgehängten Volumes ohnehin. (4) `docker cp` in `/tmp` der Container legt root-eigene Dateien an, die uid 1000 im Sticky-`/tmp` nicht löschen kann — Smokes per `docker exec -i … sh -c 'cat > /tmp/…'` hineinstreamen; die Docstrings der fünf Smokes und von `probe_configured_models.py` bekommen in P3 eine Zeile dazu. ⚠️ *Korrektur P3 (Sub-Thread, gemessen): `docker cp` übernimmt den Eigentümer der **Quelle** aus dem Tar-Header, nicht root — eine Datei aus Olis 1000-Checkout landet als 1000 und ist löschbar; nur eine root- oder fremd-eigene Quelle bleibt im Sticky-`/tmp` stehen. Die Aussage oben stammte aus der Docker-Doku, nicht aus einer Messung; das Streamen bleibt die robuste Empfehlung.*

## 2.2 Messung

- `docker exec … id -u` → 1000 in Web **und** Worker; Launcher weiter 0 (gewollt); `ls -ln /app/data` zeigt 1000.
- **Web-Pfade**: Login 200, Reader-Smoke ([scripts/smoke_markdown_reader.py](../../../scripts/smoke_markdown_reader.py), erzeugt ein PDF über Chromium als 1000) grün, `flask set-password --help` und `create-user` als Default-Nutzer (Wegwerf-User) → DB-Datei bleibt 1000.
- **Worker-Pfade**: eine TXT-Konvertierung über `POST /api/document-conversions` (Worker schreibt `result_<id>.json` unter `podcast_data`, Web liest) → `ready` · **ein echter Lokal-Lauf** (Scan-PDF, `mode=lokal`) → `ready`, `modell`; damit ist der Copy-out als `1000:1000` gemessen — im Job-Verzeichnis dürfen keine root-Dateien liegen, `rmtree` muss geräumt haben. Scheitert der Copy-out an Leserechten der mineru-Ausgabe → Entscheidung 3, Rückfall bauen, Lauf wiederholen.
- **Die Bedienregel**: `docker exec -u 0 … touch /app/data/root-test` → Datei gehört root; dann `docker exec … rm /app/data/root-test` als 1000 → scheitert (`Permission denied`) → mit `-u 0` löschen. Das ist der Beleg, warum die Regel in CLAUDE.md steht.
- **Backup-Rezept** einmal als Default-Nutzer fahren → funktioniert (schreibt `/tmp`), Kopie danach löschen.
- **Negativprobe**: als 1000 `touch /app/app_pkg/x` → `Permission denied` (Code nicht beschreibbar).
- Aufräumen strikt nach `user_id`, Prod-Zahlen vorher = nachher, keine Reste im Volume, kein Snapshot außer dem geplanten.

## Stop
Bericht mit Messwerten. Dann warten.

---

# Phase 3 — Wrap

- **CLAUDE.md**: Kopfsatz (offen nur noch SEC-SSRF); Sicherheits-Bullet F-8 geschlossen — welche Container als wer laufen (Web/Worker 1000 = `oliver` auf der Mintbox, warum; Launcher root, warum; mineru root im Vektor; Redis als `redis`), Code im Image read-only; *Running*: `docker exec` nie als root, frische Volumes erben 1000 aus dem Image, `pytest` im Container mit `-p no:cacheprovider`; DOC-LOCAL-Bullet: `EXCHANGE_OWNER=1000:1000`. Test-Baseline.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): SEC-NONROOT schließen; im Befund-Doc F-8 auf **geschlossen**. Falls die Chromium-Sandbox als P3-Möglichkeit benannt werden soll: eine Zeile, nicht mehr.
- **Kein Brief ans converter-mcp** (keine Agent-Fläche) — so sagen.
- **Memory** nur bei übertragbarer Lehre. Kandidat: *Ein frisches Named Volume erbt Inhalt und Besitz nur von einem Verzeichnis, das im Image existiert — `mkdir`+`chown` vor `USER`, sonst schreibt der Nutzer nicht in sein eigenes Volume.* Nur, wenn die Frisches-Volume-Probe das gezeigt hat.
- **Im Bericht**: Auszeit · Zahl der umgehängten Einträge je Volume · Ergebnis des Copy-out-Tests · welche Container als welche uid laufen · Wegwerf-User weg.

## Nicht-Ziele

- **Kein** `USER` am Launcher, **keine** Änderung an mineru oder dem Vektor.
- **Keine** Chromium-Sandbox, **keine** Capabilities-/seccomp-Arbeit (eigenes Item bei Bedarf).
- **Kein** Umbau von Pfaden, Volumes oder Compose jenseits `EXCHANGE_OWNER`.
- **Keine** Änderung an Olis Host-Dateien oder -Rechten — die uid-Wahl macht sie unnötig.
