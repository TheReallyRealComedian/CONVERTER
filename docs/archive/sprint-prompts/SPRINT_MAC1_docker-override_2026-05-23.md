# Sprint MAC1 — Docker-Override für Mac-Dev

**Datum**: 2026-05-23

**Ziel**: Den Stack auf einem Apple-Silicon-Mac (arm64) lokal buildbar und startbar machen, ohne das Server-Setup auf Mintbox zu verändern. Mechanik: ein gitignored `docker-compose.override.yml`, das Compose automatisch zusätzlich zu `docker-compose.yml` lädt. Server-Stack bleibt bit-identisch.

**Vorbedingung**:
- Repo-Clone auf Mac unter `/Users/olivergluth/CODE/CONVERTER`. Branch `main`, working tree clean.
- Architektur: Apple M4 (arm64), Docker Desktop 28.5.2 (OSType linux, Architecture aarch64).
- Bekannte Showstopper auf Mac:
  1. `notion-mcp-net` external network existiert nicht → Compose-Abbruch beim Start.
  2. `.env` und `google-credentials.json` fehlen lokal (beide gitignored, müssen aus Server-Setup übernommen werden — **das ist Olivers Aufgabe vor Phase 2, nicht Sub-Thread**).
  3. Playwright-Base-Image (`mcr.microsoft.com/playwright/python:v1.44.0-jammy`) ist nur `linux/amd64` → läuft auf M4 über Rosetta-Emulation, langsam aber funktional.
  4. `notion-mcp-server` Container läuft nicht lokal mit; `NOTION_MCP_URL` zeigt im Stack auf einen Hostnamen, der lokal nicht auflöst — Notion-Integration ist auf Mac nicht funktional, aber muss den App-Start nicht blockieren.
- Live-Stack: `localhost:5656` auf Mintbox plus `converter.smallpieces.de`. **Bleibt unangetastet.**

**Out-of-scope**:
- Native Mac-Dev ohne Docker (war Option B in der Master-Analyse — separater Sprint, falls Option A nicht reicht).
- Multi-Arch-Dockerfile (Option C) — kein eigener Sprint geplant, nur falls Option A in der Praxis zu langsam wird.
- Code-Refactor der hardgecoded `/app/...`-Pfade in [config.py:10](app_pkg/config.py:10), [__init__.py:48,73](app_pkg/__init__.py:48), [markdown.py:19](app_pkg/markdown.py:19), [deepgram_service.py:48](services/deepgram_service.py:48), [podcasts.py:339](app_pkg/podcasts.py:339) — im Container egal, nur für Option B relevant.
- `requirements.txt`-Touches, Service-Code-Touches, Test-Touches — Mac-Dev darf das nicht erzwingen.
- Notion-Integration auf Mac lauffähig machen.

---

## Phase 1 — Implementation

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. `docker network ls | grep notion-mcp-net` → bestätigen dass das Network lokal **nicht** existiert (das ist der Grund warum wir es im Override neu definieren).
3. Existing-Repo-State: `docker-compose.override.yml` darf nicht existieren (sonst STOP, Master fragen — könnte aus früherem Versuch sein).

**Files**:

```
docker-compose.override.yml    # NEU — gitignored, Mac-spezifische Overrides
.gitignore                     # EDIT — Eintrag für docker-compose.override.yml ergänzen
docs/mac-dev-setup.md          # NEU — knappes Setup-Doc für Mac-Erst-Bootstrap
```

**`docker-compose.override.yml` — Mechanik**:

Compose lädt das File automatisch zusätzlich zu `docker-compose.yml`, kein extra `-f`-Flag nötig. Es muss exakt die Service-Namen aus dem Base-File matchen (`markdown-converter`, `worker`, `redis`).

Inhaltliche Anforderungen:

1. **`notion-mcp-net` von external auf lokal-bridge umstellen.** Im Override unter dem Top-Level-`networks`-Key das Network neu deklarieren ohne `external: true`. Compose mergt die Netzwerk-Definitionen; der Container kann dem Namen joinen, Compose meckert nicht über fehlendes external Network.
2. **`platform: linux/amd64`** explizit auf `markdown-converter` UND `worker` setzen. Sonst versucht Docker Desktop auf M4 das Image als arm64 zu bauen / zu ziehen, was beim Playwright-Image fehlschlägt.
3. **`google-credentials.json`-Volume-Mount nicht verändern** — Oliver bereitet die Datei in Phase 2 vor, sonst legt Docker ein leeres Verzeichnis an und der `if GOOGLE_CREDENTIALS_PATH else None`-Guard in [app.py:51](app.py:51) lässt die App trotzdem starten (TTS halt aus). Akzeptiert.
4. **`MCP_AUTH_TOKEN` und `NOTION_TOKEN`** sind im Base-Compose als Env-Vars deklariert; wenn sie in `.env` fehlen, setzt Compose sie auf leeren String. Kein Override nötig.
5. **Restart-Policy auf `no` setzen** für `markdown-converter` und `worker` — auf Server ist `unless-stopped` richtig (langlebige Prod-Instanz), lokal will Oliver nicht dass ein crashender Container in Endlosschleife läuft.

**`.gitignore`-Patch**:

Eintrag `docker-compose.override.yml` direkt unter dem `.env`-Block ergänzen, mit Inline-Kommentar dass das Mac-Dev-File ist. Server-Setup soll **kein** Override-File haben.

**`docs/mac-dev-setup.md`**:

Knapp (max ~30 Zeilen). Inhalt:

1. Vorbedingungen (Docker Desktop, Apple Silicon).
2. Was kopiert werden muss vom Server: `.env`, `google-credentials.json`. Pfade dazu (Mintbox-Pfad falls bekannt, sonst Hinweis dass Oliver weiß wo).
3. Befehlssequenz: `docker compose up --build` → erster Build dauert wegen Rosetta-Emulation ~5–10 min. Hinweis dass das Override-File automatisch geladen wird.
4. Was funktioniert / was nicht: Markdown-PDF + Audio + Library laufen erwartet; Notion-Integration nicht (kein lokaler `notion-mcp-server`).
5. Smoke-URL: `http://localhost:5656`.

**Was Sub-Thread NICHT tun darf**:

- `.env` schreiben oder kopieren — Oliver macht das in Phase 2 selbst (Secrets sind nicht im Repo, nicht im Memory, nicht in dieser Sprint-Doc).
- `google-credentials.json` schreiben oder kopieren — gleich.
- An `docker-compose.yml` (Base-File) touchen — alles muss als Override gehen.
- An `Dockerfile`, `requirements.txt`, `app.py`, irgendeinem Service-Code touchen — wenn das Setup das erzwingt, ist Option A gescheitert und es geht in einen Folge-Sprint (Master entscheidet).
- `docker compose up --build` selbst starten — der Build dauert auf M4 mehrere Minuten via Emulation und ich will, dass Oliver vorher Sign-off gibt + die Secrets bereitstellt.

**Nach Phase 1: STOP — Bericht.** Was geschrieben wurde, ob `docker compose config` ohne Fehler durchläuft (das ist der Compose-Syntax-Check, kein Build — Sub-Thread darf das laufen lassen).

---

## Phase 2 — Verify

**Vor Phase 2 macht Oliver manuell**:

- `.env` aus Mintbox kopieren (`scp` oder via 1Password).
- `google-credentials.json` aus Mintbox kopieren in `/Users/olivergluth/CODE/CONVERTER/`.
- Sign-off im Sub-Thread dass beide Files da sind.

**Sub-Thread-Schritte**:

1. `ls -la /Users/olivergluth/CODE/CONVERTER/.env /Users/olivergluth/CODE/CONVERTER/google-credentials.json` → beide existieren?
2. `docker compose config` → keine Syntax-Fehler, `notion-mcp-net` wird als lokales Bridge-Network gemerged, `platform: linux/amd64` auf den richtigen Services.
3. `docker compose up --build` (Foreground, mit `--progress=plain` damit Build-Log lesbar bleibt). Build dauert lange wegen Emulation — Geduld. Wenn Build crasht: STOP, Bericht mit Log-Tail.
4. Nach erfolgreichem Start: `curl -sI http://localhost:5656/` → erwarte 200 oder 302 (Redirect zu Login).
5. Im Browser `http://localhost:5656` öffnen, Login-Seite muss rendern. Login mit Olivers User (Oliver weiß die Credentials).
6. Smoke-Test der drei Hauptpfade (alle ohne Notion-Touch):
   - **Markdown-Converter**: kleine Test-Markdown rein, PDF kommt raus. Das ist der Playwright-Pfad — der kritische Test, weil Playwright unter Emulation am ehesten Probleme macht.
   - **Document-Converter**: kleine Test-Datei (z.B. eine kurze `.docx`) → Markdown raus. Testet `unstructured`-Pfad.
   - **Library**: aufrufen, History muss leer aber Page muss rendern.
7. Pytest im Container: `docker compose exec markdown-converter pytest tests/` — muss grün sein (Erwartung: gleiche Zahl wie auf Server, siehe STATUS.md aktueller Stand).

**Nach Phase 2: STOP — Bericht.** Was funktioniert, was nicht, ob Playwright-PDF-Generierung unter Emulation realistisch nutzbar ist (Zeitmessung wäre wertvoll: wie lange braucht eine simple MD→PDF?).

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit reicht (Override + .gitignore-Patch + Setup-Doc gehören zusammen). Subject z.B. „MAC1: docker-compose.override.yml + mac-dev-setup.md für Apple-Silicon-Dev".
- Branch: direkt auf `main`.
- `git push origin main` ist Teil des Sprints.
- **Wichtig**: `docker-compose.override.yml` ist gitignored und darf **nicht** mit committet werden. `git status` vor Commit kontrollieren — nur `.gitignore` und `docs/mac-dev-setup.md` sollten im Diff stehen.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Spezifisch für MAC1: in Phase 1 nach dem `docker compose config`-Check **stoppen, bevor `docker compose up --build` läuft** — der Build verbrennt unter Emulation 5–10 min Compute, und ich will Oliver erst Sign-off geben lassen plus die Secrets-Vorbereitung machen.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — ein Override-File, ein .gitignore-Patch, ein knappes Setup-Doc. Kein Code-Touch erwartet. Wenn Phase 2 ergibt, dass Code-Patches nötig sind (z.B. weil ein hardgecoded Pfad in einem unerwarteten Kontext bricht), eskaliert der Sprint auf M und Master entscheidet ob splitten.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Smoke-Test in Phase 2 ein konkreter `/app/...`-Pfad als Mac-Problem auffällt: **nicht fixen**, nur im Bericht aufzählen — gehört in Option-B-Sprint.
- Wenn Playwright unter Emulation für ein konkretes Pattern (z.B. wide tables, web fonts) bricht, wo es auf Server grün ist: ebenfalls nur im Bericht aufzählen.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „MAC1 ☑ done 2026-05-23 → commit `<hash>` (docker-compose.override.yml + mac-dev-setup.md). Mac-Stack startet lokal via `docker compose up --build`, Markdown-Converter + Document-Converter + Library getestet. Build-Zeit erster Run ~X min via Rosetta. Notion-Integration auf Mac off (kein lokaler `notion-mcp-server`). Server-Setup unangetastet."
- **BACKLOG.md**: neuen Eintrag oben („Mac-Dev-Infra") falls Follow-ups aufgetaucht sind (z.B. „Native Mac-Dev ohne Docker [Option B]", „/app/-Pfade konfigurierbar machen", „Notion-MCP lokal hochziehen") — sonst kein BACKLOG-Touch.
- **Memory**: wenn ein nicht-offensichtlicher Mac-spezifischer Workaround auftaucht (z.B. Docker-Desktop-Einstellung die Rosetta-Emulation merklich beschleunigt): `feedback_mac_docker.md` oder `reference_mac_dev.md`. Nichts erzwingen — nur wenn überraschend.

---

## Phase-0-Entscheidungen

_(Phase 0 nicht aktiviert — Mechanik ist klar: docker-compose.override.yml, gitignored, vier Override-Punkte oben im Doc spezifiziert. Kein Workshop nötig.)_
