# SPRINT IMG-CONTEXT — der Build-Kontext trägt keine Geheimnisse und keinen Werkzeug-Müll mehr ins Image

**Größe**: S (3 kurze Phasen) · **Datum**: 2026-09-29 · **Vorhaben**: Hygiene-Folge aus SEC-NONROOT P1 (kein Audit-Finding), BACKLOG IMG-CONTEXT

## Warum

`COPY . .` kopiert das **Verzeichnis**, nicht das Repo: alles, was `.dockerignore` nicht ausschließt, landet im Image — auch, was `.gitignore` versteckt. Gemessen am deployten Image `converter-app:latest` (`2a97cb4ed290`, 6,44 GB) liegen unter `/app`: `google-credentials.json` (der GCP-Service-Account-Schlüssel, root `600` — zur Laufzeit von Compose per Bind überdeckt, im **Artefakt** aber drin, und der `mineru-launcher` läuft als root auf demselben Image), `.codebuddy/db` (28 MB Werkzeug-Datenbank), `.claude/` (`settings.local.json` + eine **versionierte** Streu-Kopie `mermaid_converter.html`), `.pytest_cache/`, `.env.example`, dazu `BACKLOG.md`, `STATUS.md`, `CLAUDE.md`, `MASTER_BACKLOG_HANDOFF_2026-05-26.md`, `pytest.ini`, `test_redis_connection.py`, `test_worker_libraries.py` — alles ohne Laufzeit-Leser. Die SEC-SOCKET-Aussage „der Launcher trägt keine App-Geheimnisse" stimmt für die Env, nicht für das Image. Und SEC-KEY-ROTATION folgt direkt: erst den Kontext bereinigen, dann rotieren — sonst wandert der neue Schlüssel ins nächste Image.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-09-29 — nicht neu herleiten, Abweichungen benennen)

**Image `2a97cb4ed290`, `ls -A /app`:** `.claude .codebuddy .env.example .pytest_cache BACKLOG.md CLAUDE.md MASTER_BACKLOG_HANDOFF_2026-05-26.md STATUS.md app.py app_pkg constraints.txt data doclocal_exchange google-credentials.json keyterms.json models.py output_podcasts pytest.ini requirements.txt scripts services static tasks.py templates test_redis_connection.py test_worker_libraries.py worker.py`. Größen: `.codebuddy` 28 MB (`db/`), `.claude` 12 K, `.pytest_cache` 32 K, `scripts` 204 K, `static` 1,1 MB. `find /app` nach `*.db`, `*credentials*`, `.env*`, `*.pem`, `*.key` (ohne `static/`): **`/app/.env.example`, `/app/google-credentials.json`**. Die zwei `app_data_bak-*`-Ordner sind seit MINTBOX-BAK (2026-09-28) weg und **nicht** mehr im Image; `.claude/worktrees` ebenso.

**Build-Kontext auf der Mintbox** (`git status --ignored` im Clone): ignoriert, aber vorhanden — `.claude/settings.local.json` (ignoriert über Olis **globale** Git-Ignore `~/.config/git/ignore`, nicht über die Repo-`.gitignore`), `.codebuddy/`, `.env`, `.pytest_cache/`, `corpus/…` (in `.dockerignore`), `google-credentials.json`. Auf dem Mac zusätzlich `docker-compose.override.yml` (Mac-Dev-Override, gitignored) — heute **nicht** in `.dockerignore` (dort steht nur `docker-compose.yml`).

**`.dockerignore` heute:** `.dockerignore Dockerfile docker-compose.yml .git .gitignore __pycache__/ *.pyc *.pyo *.pyd .env venv/ env/ .DS_Store ._* output_pdfs/ corpus/ tests/ docs/ .vscode/ .idea/ doclocal_exchange/ data/ output_podcasts/`. Sentinel [tests/test_nonroot.py::test_dockerignore_keeps_the_mount_points_out_of_the_context](../../../tests/test_nonroot.py) pinnt die drei Mount-Punkte (Muster sind **wurzel-verankert**, Go `filepath.Match`, kein Gitignore-Rekursiv: `data/` trifft nur `./data`).

**Was bleiben MUSS:** `keyterms.json` — [services/deepgram_service.py:98](../../../services/deepgram_service.py) `load_keyterms` liest es zur Laufzeit; `static/`, `templates/`, `app_pkg/`, `services/`, `app.py`, `tasks.py`, `worker.py`, `models.py`, `requirements.txt`, `constraints.txt`. `scripts/` bleibt (204 K, Smokes werden per `docker exec -i` gestreamt, nicht aus `/app` gestartet — kein Grund zur Änderung in diesem Sprint).

**`.claude/mermaid_converter.html`** ist seit `2c3446b` (Mai, „add Mermaid diagram renderer") versioniert; alle acht Repo-Referenzen auf den Dateinamen meinen `templates/mermaid_converter.html` (die Route in [app_pkg/mermaid.py:10](../../../app_pkg/mermaid.py)) oder sind Doku/BACKLOG-Erwähnungen. Die `.claude/`-Kopie hat **null** Leser.

**Laufzeit-Beleg für den Schlüssel:** Compose bindet `./google-credentials.json:/app/google-credentials.json` an Web **und** Worker ([docker-compose.yml:45,87](../../../docker-compose.yml)); die Datei gehört auf der Mintbox `1000:1000 600`. Ohne die Kopie im Image ändert sich zur Laufzeit nichts — das ist zu **messen**, nicht anzunehmen: `docker exec markdown-converter-web ls -ln /app/google-credentials.json` → `1000 1000 600` (der Bind), und `scripts/probe_configured_models.py` im Worker spricht Cloud-TTS und Gemini real an (exit 0).

**Tags/Images:** `converter-app:latest` = `2a97cb4ed290`, `converter-app:pre-sec-nonroot` = `0179903fe815` (Rückweg aus SEC-NONROOT, darf nach diesem Deploy weg). ⚠️ `docker image prune -a` bleibt verboten (`mineru:3.4.4` ohne Registry-Digest). Builder-Cache gesamt 51,85 GB — **nicht** Gegenstand, nur notieren. Baseline **1320 + 1 Skip** (Mac und Container-Pin). Mac `main` auf `2a31c47`, Mintbox-Clone auf `86c4742` (nur Docs dahinter).

## Gesperrte Entscheidungen

1. **`.dockerignore` bekommt genau diese Einträge dazu** (Kommentar je Gruppe, warum): Geheimnisse `google-credentials.json`, `.env*` (ersetzt `.env`; trifft `.env.example` mit — kein Laufzeit-Leser); Werkzeug-Zustand `.claude/`, `.codebuddy/`, `.pytest_cache/`; Backup-Muster `app_data_bak-*`; Compose `docker-compose*.yml` (ersetzt `docker-compose.yml`, nimmt den Mac-Override mit); Master-Docs und Root-Skripte ohne Laufzeit-Leser `BACKLOG.md`, `STATUS.md`, `CLAUDE.md`, `MASTER_BACKLOG_HANDOFF_*.md`, `pytest.ini`, `test_redis_connection.py`, `test_worker_libraries.py`. **Nichts sonst** — `scripts/`, `keyterms.json`, `static/`, `templates/` bleiben.
2. **`git rm .claude/mermaid_converter.html`** — Streu-Kopie ohne Leser; `templates/mermaid_converter.html` bleibt unberührt. (Damit ist auch der MINTBOX-BAK-Rest erledigt: die Datei war als Mount-Rest gelistet, ist aber versioniert.)
3. **Sentinel `tests/test_dockerignore.py`**: liest `.dockerignore` (Kommentare/Leerzeilen weg), prüft die Muss-Liste aus Entscheidung 1 **exakt** (jedes Muster als eigene Zeile) und eine **Bleib-Liste** (`keyterms.json`, `static`, `templates`, `app_pkg`, `services`, `scripts`, `app.py`, `tasks.py`, `worker.py`, `models.py`, `requirements.txt`, `constraints.txt`): kein Muster darf einen dieser Namen treffen (`fnmatch` auf den Namen und auf den ersten Pfadbestandteil, Muster ohne abschließenden `/`). `pytest.importorskip`-frei; skippt nur, wenn `.dockerignore` nicht neben den Tests liegt (Container-Lauf mountet sie, s. CLAUDE.md *Key Files*).
4. **Gemessen wird das Image, nicht die Datei.** Nach dem Bau: `ls -A /app` ohne `.claude .codebuddy .pytest_cache .env.example google-credentials.json BACKLOG.md STATUS.md CLAUDE.md MASTER_BACKLOG_HANDOFF_* pytest.ini test_*.py`; der `find` aus dem Ist-Zustand leer; `keyterms.json` vorhanden; Image-Größe vorher/nachher. Bleibt Müll trotz Eintrag im Image → zuerst `docker builder prune --filter type=source.local` (der Kontext-Cache hielt in SEC-NONROOT alte Zustände fest), neu bauen, so sagen.
5. **Rückweg vorbereiten, alten Rückweg räumen**: vor dem Bau `docker tag 2a97cb4ed290 converter-app:pre-img-context`; nach der Abnahme `docker rmi converter-app:pre-sec-nonroot` (nur das Tag). `pre-img-context` bleibt bis zur nächsten Deploy-Runde, in STATUS notieren.
6. ⚠️ **Editiert wird nur auf dem Mac.** Mintbox = Runtime; `docker exec` nie `-u 0`; keine Reste; kein Wegwerf-User nötig (keine Web-Pfade ändern sich), außer die Probe braucht einen — dann strikt nach `user_id`.

---

# Phase 1 — Kontext bereinigen, Sentinel, Suiten

- `.dockerignore` nach Entscheidung 1, `git rm` nach Entscheidung 2, `tests/test_dockerignore.py` nach Entscheidung 3. Mac-Suite grün, dann Container-Suite auf dem Pin (stdin-Rezept aus CLAUDE.md *Key Files* — `.dockerignore` reist mit, weil `git ls-files` sie trägt). Commit + Push.
- Kurz belegen, dass Docker das Muster so liest wie erwartet: `docker build` ist am Mac nicht möglich — deshalb Beleg in Phase 2 am Image.

## Stop
Bericht: Diff, Testzahl. Dann warten.

---

# Phase 2 — Bau, Beleg, Deploy

## 2.1 Bau ohne Deploy
Mintbox: `git pull --ff-only`, `docker tag 2a97cb4ed290 converter-app:pre-img-context`, `docker compose build` (taggt `latest` neu; laufende Container behalten `2a97cb4ed290` — mit `docker inspect -f '{{.Image}}'` belegen). Messung nach Entscheidung 4 am neuen Image (`docker run --rm --network none --entrypoint sh converter-app:latest -c 'ls -A /app; find …'`), Größe vorher/nachher (`docker images`).

## 2.2 Deploy
Fenster wie zuletzt (keine laufende Konvertierung, `rq:wip` 0, Worker `idle`, kein `mineru_*`). `docker compose up -d` aus dem Projektverzeichnis. Login 200. Dann die zwei Laufzeit-Belege: `docker exec markdown-converter-web ls -ln /app/google-credentials.json` → `1000 1000 600`; `scripts/probe_configured_models.py` per `docker exec -i markdown-converter-worker python - < scripts/probe_configured_models.py` → beide Modelle OK, exit 0. Dazu `docker exec markdown-converter-worker python3 -c "import json; print(len(json.load(open('/app/keyterms.json'))))"` oder gleichwertig — `keyterms.json` liegt und ist lesbar. `docker rmi converter-app:pre-sec-nonroot`.

## Stop
Bericht: `ls -A /app` alt/neu, `find`-Ergebnis, Größen, Laufzeit-Belege, Tags. Dann warten.

---

# Phase 3 — Wrap

- **CLAUDE.md**: im SEC-NONROOT-/Running-Kontext oder beim Image-Slim-Bullet ein kurzer Absatz *Build-Kontext*: `.dockerignore` ist die Grenze, nicht `.gitignore` — gitignorierte Dateien (Schlüssel, Werkzeug-Caches) liegen im Kontext und kämen mit; Muss-Liste per Sentinel gepinnt; `keyterms.json` bleibt bewusst; die SEC-SOCKET-Aussage zum Launcher gilt jetzt auch für das Image. Test-Baseline.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): IMG-CONTEXT schließen; im MINTBOX-BAK-Abschluss den Satz zur `git rm`-Entscheidung nachziehen; SEC-KEY-ROTATION ist jetzt dran (nur nennen, nicht anfassen).
- **Memory**: **keine neue Datei** — eine Zeile als Punkt 6 in `reference_nonroot_image_ownership_traps`: *`COPY . .` kopiert das Verzeichnis, nicht das Repo — `.gitignore` versteckt vor Git, `.dockerignore` vor dem Image; Schlüssel und Werkzeug-Caches sind genau die Dateien, die im ersten stehen und im zweiten fehlen.* Index-Zeile der Datei um „docker-ignore ≠ git-ignore" ergänzen.
- **Kein Brief ans converter-mcp** (keine Agent-Fläche) — so sagen.
- **Im Bericht**: `ls -A /app` alt/neu · Image-Größe alt/neu · Laufzeit-Belege · Tags · Testzahl.

## Nicht-Ziele

- **Kein** Umbau des Dockerfiles, **keine** Layer-Reihenfolge, **kein** Multi-Stage.
- **Kein** `docker builder prune` ohne Not (nur der `source.local`-Filter, falls Entscheidung 4 ihn braucht); **nie** `docker image prune -a`.
- **Keine** Änderung an `scripts/`, `keyterms.json`, Templates, Static.
- **Keine** Rotation von Schlüsseln — das ist SEC-KEY-ROTATION, danach, in Olis Hand.
