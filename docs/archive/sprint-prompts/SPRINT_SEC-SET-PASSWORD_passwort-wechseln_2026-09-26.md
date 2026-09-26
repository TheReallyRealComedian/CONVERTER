# SPRINT SEC-SET-PASSWORD — ein Passwort lässt sich wechseln

**Größe**: S (eine Phase, Wrap inklusive) · **Datum**: 2026-09-26 · **Vorhaben**: SEC-AUDIT-Folge (Befund 1 der MCP-Rückmeldung)

## Warum

Das converter-mcp-Team, 2026-09-26 ([Rückmeldung](../../converter_mcp_sec_audit_rueckmeldung.md), Befund 1): *„CONVERTER hat keinen Weg, ein Passwort zu ändern."* Olis Passwort lag Monate im Env des MCP-Containers (per `docker inspect` lesbar) und in einer bis 26.09. world-readable `.env`. SEC-AUDIT hat `SECRET_KEY` und drei Tokens rotiert — das Passwort selbst lässt sich heute nur per Handarbeit im Container ändern. Ein Audit, das Credentials als exponiert einstuft, braucht den Hebel.

## Gegroundeter Ist-Zustand (Master, 2026-09-26)

- `User.set_password` ([models.py](../../../models.py)) hat genau **einen** Aufrufer: `flask create-user` in [app_pkg/__init__.py](../../../app_pkg/__init__.py) `_register_cli_commands` — `click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True)`, Mindestlänge 8, bricht bei bestehendem Benutzer mit `click.echo` ab (Exit-Code 0 — eine Kante, die nicht zu übernehmen ist). Daneben liegt `flask reset-collection` (LEARN-BACK) als zweites Vorbild für ein Einmal-Werkzeug mit Dry-run-Denke.
- `User.authenticate` (SEC-AUDIT) ist der einzige Login-Pfad; `check_password` liest `password_hash` (scrypt, Werkzeug 3.1.8).
- `ApiToken` ([models.py](../../../models.py)): per-User-Bearer, sha256-Hash, `label` (80), `expires_at` NULL erlaubt; Widerruf = Row-Delete (`POST /api/auth/logout` mit genau diesem Token). Prod heute: vier Zeilen, drei `ios-app`, eine `converter-mcp` (id 4).
- Browser-Sessions sind signierte Cookies (`SECRET_KEY`), kein Server-Zustand — ein Passwortwechsel kann sie **nicht** beenden; das tut nur die `SECRET_KEY`-Rotation.
- Prod: **ein** User (`oliver`, id 1). Das Kommando läuft im Web-Container: `docker exec -it markdown-converter-web flask set-password oliver`.

## Gesperrte Entscheidungen

1. **CLI, kein Endpoint, keine UI, keine Token-Fläche** — wie `create-user` und `reset-collection`: ein Operator-Werkzeug, das nur mit Container-Zugriff erreichbar ist.
2. **Kein impliziter Token-Widerruf.** Tokens sind eigene Credentials mit eigenem Widerruf; ein stiller Massen-Widerruf träfe iOS-App und converter-mcp, ohne dass der Operator es sieht. Stattdessen: nach dem Wechsel die aktiven Tokens des Benutzers **auflisten** (id, label, erstellt, expires_at) und `--revoke-tokens` anbieten, das **alle** Tokens dieses Benutzers löscht — mit Hinweis, was danach neu ausgestellt werden muss.
3. **Dieselbe Passwort-Regel wie `create-user`** (Mindestlänge 8) aus **einer** Stelle — den Check in einen Helper ziehen, den beide nutzen; `create-user` sonst nicht anfassen.
4. **Fehler enden mit Exit-Code ≠ 0** (`click.ClickException` / `sys.exit(1)`): unbekannter Benutzer, zu kurzes Passwort, Bestätigung ungleich.
5. **Das Kommando sagt, was es nicht tut**: Sessions bleiben gültig (Hebel: `SECRET_KEY`), Tokens bleiben ohne Flag gültig.
6. ⚠️ **Editiert wird nur auf dem Mac.** Mintbox = Runtime. **Das Passwort selbst setzt Oli** — der Sub-Thread führt das Kommando in Prod **nicht** aus.

---

# Phase 1 — Bauen, belegen, wrappen

## 1.1 Bauen

`flask set-password <username> [--revoke-tokens]` in `_register_cli_commands`: Passwort per Prompt (hide_input + confirmation_prompt, wie `create-user`), Helper für die Längenregel, Benutzer muss existieren, `set_password` + Commit, danach Token-Liste; mit `--revoke-tokens` alle `ApiToken`-Zeilen des Benutzers löschen und die Zahl ausgeben. Ausgabe deutsch, knapp.

## 1.2 Beleg

Tests mit `app.test_cli_runner()` ([tests/test_reset_collection.py](../../../tests/test_reset_collection.py) als Vorbild): Wechsel → altes Passwort scheitert, neues greift (`User.authenticate`) · unbekannter Benutzer → Exit ≠ 0, Hash unverändert · zu kurz → Exit ≠ 0 · ohne Flag bleiben Tokens (Zahl und Liste in der Ausgabe) · mit Flag: nur die Tokens **dieses** Benutzers weg (zweiter Benutzer im Test behält seine). `pytest tests/` grün (Baseline **1159 + 1 Skip**).

## 1.3 Deploy + Gegenprobe

Deploy (`git pull` + `docker compose up -d --build`), dann `docker exec markdown-converter-web flask set-password --help` als Existenzbeleg. **Nicht** ausführen. Prod-User und Tokens vorher/nachher zählen (`mode=ro`): unverändert.

## 1.4 Wrap

- **CLAUDE.md**, Sektion *Auth*: ein Satz — `flask set-password` als einziger Wechselpfad, Tokens bleiben ohne `--revoke-tokens`, Sessions bleiben (Hebel `SECRET_KEY`).
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): SEC-SET-PASSWORD schließen. Kein Brief ans converter-mcp (keine Agent-Fläche berührt) — im Wrap so sagen. Memory: keine erwartet.
- **Im Bericht**: das genaue Kommando für Oli (mit und ohne Flag) und die Folge je Variante (iOS-App / converter-mcp müssen bei `--revoke-tokens` neue Tokens holen).

## Stop
Commit + Push (Code + Tests, Wrap eigener Commit), deployt, dann warten.

## Nicht-Ziele

- **Kein** Endpoint, **keine** UI, **kein** Token-Pfad; **kein** Umbau von `create-user` über den geteilten Helper hinaus.
- **Kein** Setzen des Passworts in Prod durch den Sub-Thread.
- **Keine** Session-Invalidierung (bräuchte Server-Zustand — benannte Möglichkeit, nicht gebaut).
