# SPRINT CREDS-RO — der Schlüssel-Bind an Web und Worker wird nur lesbar

**Größe**: XS (2 Phasen) · **Datum**: 2026-09-29 · **Vorhaben**: Härtung aus IMG-CONTEXT P2, BACKLOG CREDS-RO — **unmittelbar vor SEC-KEY-ROTATION**, damit der frische Schlüssel von der ersten Minute an nur lesbar im Container liegt

## Warum

[docker-compose.yml](../../../docker-compose.yml) bindet `./google-credentials.json:/app/google-credentials.json` an Web (Z. 45) und Worker (Z. 87) **ohne `:ro`**. Seit SEC-NONROOT laufen beide als uid 1000, die Host-Datei gehört `1000 600` — der Prozess, der fremde Dokumente parst und fremdes Markdown rendert, darf den GCP-Service-Account-Schlüssel auf dem Host **überschreiben**. Lesen muss er ihn (Cloud-TTS im Worker, `GoogleTTSService` im Web); schreiben muss er ihn nie.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-09-29 — nicht neu herleiten, Abweichungen benennen)

- `docker inspect` an beiden Containern: der Mount `/app/google-credentials.json` trägt **`RW=true`**. Im Worker als uid 1000 gelingt `os.open(path, O_WRONLY|O_APPEND)` — **die Datei ist für den Prozess schreibbar** (nur geöffnet, nichts geschrieben).
- Leser: [services/google_tts_service.py:15](../../../services/google_tts_service.py) (verlangt `GOOGLE_APPLICATION_CREDENTIALS`), [tasks.py:214](../../../tasks.py) (Worker), [app.py:50](../../../app.py); die Env kommt aus `.env` über `GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}` (Compose Z. 59, 100). Kein Schreiber im Code.
- Der Launcher hat seit IMG-CONTEXT keinen Schlüssel mehr, weder als Bind noch im Image.
- Compose-Sentinels lesen die Datei per `yaml.safe_load` und prüfen `volumes`-Listen wörtlich ([tests/test_compose_socket.py:26–57](../../../tests/test_compose_socket.py) — `'./doclocal_exchange:/app/doclocal_exchange' in worker['volumes']` ist das Muster).
- Prod: `converter-app:latest` = `8bf21a3e2f35`, Rollback-Tag `pre-notion-tz` = `b008b52231b5`. Baseline **1340 + 1 Skip**. Mac `main` auf `5849ade`+, Mintbox-Clone auf `7c461bd`.

## Gesperrte Entscheidungen

1. **`:ro` an genau den zwei Bind-Zeilen** (Z. 45 und 87), sonst nichts in Compose. Der Exchange-Bind am Worker bleibt schreibbar (der Worker schreibt dort seine Aufträge). Kein Image-Bau — Compose-Änderung, `docker compose up -d` legt Web und Worker neu an.
2. **Sentinel** in [tests/test_compose_socket.py](../../../tests/test_compose_socket.py) (Nachbar der Socket-/Owner-Prüfungen, gleiche Parsing-Art): Web **und** Worker tragen exakt `./google-credentials.json:/app/google-credentials.json:ro`; kein Dienst trägt die Bind-Zeile ohne `:ro`; der Launcher trägt sie gar nicht.
3. **Beleg am laufenden System, vorher/nachher**: `docker inspect` → `RW=false` an beiden; im Worker als uid 1000 scheitert derselbe `os.open(…, O_WRONLY|O_APPEND)` mit `EROFS` (Read-only file system); `scripts/probe_configured_models.py` im Worker → exit 0 (Cloud-TTS liest den Schlüssel weiterhin); Login 200. Die Negativprobe **öffnet nur**, sie schreibt nie — auch vorher nicht.
4. ⚠️ **Editiert wird nur auf dem Mac.** Mintbox = Runtime; `docker exec` nie `-u 0`; keine Reste. Rückweg: Commit am Mac zurücknehmen, pullen, `up -d` — kein Image beteiligt.

---

# Phase 1 — Compose + Sentinel

`:ro` an beiden Zeilen, Sentinel nach Entscheidung 2 (gegen den alten Stand rot, gegen den neuen grün — kurz belegen). Mac-Suite, Container-Suite auf dem Pin (stdin-Rezept, `--no-xattrs`). Commit + Push.

## Stop
Bericht: Diff, Testzahl. Dann warten.

---

# Phase 2 — Deploy, Beleg, Wrap

- Fenster wie zuletzt (keine laufende Konvertierung, `rq:wip` 0, Worker `idle`, kein `mineru_*`). Mintbox: `git pull --ff-only`, `docker compose up -d` aus dem Projektverzeichnis (kein `--build`; Web und Worker werden neu angelegt, Launcher und Redis bleiben — mit `docker inspect` belegen). Login 200. Dann Entscheidung 3 komplett.
- **Wrap** im selben Zug: CLAUDE.md — im *Build-Kontext*-Absatz den Halbsatz „noch schreibbar, BACKLOG CREDS-RO" durch die Tatsache ersetzen (Bind `:ro`, gemessen `RW=false`, `EROFS` für den Prozess); STATUS.md (Eintrag mit den drei Belegen); BACKLOG.md (CREDS-RO schließen, ⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber; **SEC-KEY-ROTATION ist jetzt dran** — nur so sagen). Test-Baseline. **Keine Memory** (keine übertragbare Lehre über „Bind-Mounts von Geheimnissen `:ro`" hinaus — falls doch, eine Zeile in `reference_nonroot_image_ownership_traps`, keine neue Datei). **Kein Brief ans converter-mcp** (keine Agent-Fläche) — so sagen.

## Stop
Bericht: `RW` vorher/nachher, `EROFS`-Beleg, Probe exit 0, Testzahl. Dann warten.

## Nicht-Ziele

- **Kein** Image-Bau, **keine** Dockerfile-Änderung.
- **Keine** Rotation — das ist SEC-KEY-ROTATION, danach, Olis Hand.
- **Kein** `:ro` am Exchange-Bind.
