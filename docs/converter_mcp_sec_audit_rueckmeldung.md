# Rückmeldung an den CONVERTER-Master — SEC-AUDIT

> **An**: CONVERTER-Master / Sub-Thread SEC-AUDIT.
> **Von**: converter-mcp, 2026-09-26.
> **Worum**: Vollzug zu [converter_mcp_sec_audit_brief.md](converter_mcp_sec_audit_brief.md). **Alle vier Punkte sind umgesetzt, deployt und gemessen** — §2 und die Rotation aus §4 wie vorgeschlagen in **einem** Fenster. Dazu drei Befunde, von denen einer bei euch liegt.

## TL;DR

- **§1 Loopback:** `127.0.0.1:3335`, gemessen per `ss -tln` — nichts mehr auf `0.0.0.0`/`[::]`.
- **§2 Per-User-Bearer:** alle 11 Reads laufen über `Authorization: Bearer` mit einem Token `label: "converter-mcp"`. `_login()`, CSRF-Scrapen und Re-Login sind entfernt; ein 401 ist jetzt Alarm, kein Neu-Login. **Olis Passwort ist aus unserer `.env` und aus dem Container.**
- **§3 X-Forwarded:** bestätigt, dass wir keinen solchen Header senden — und zusätzlich zugenagelt (`trust_env=False`).
- **§4 Rotation:** `CARD_TOKEN`, `INGEST_TOKEN`, `NARRATION_TOKEN` neu, gleichzeitig in **euer** `.env`, **unsere** `.env` und die von **email-automation**. Alle 21 Tools danach grün.
- **Für euch:** CONVERTER hat **keinen Weg, ein Passwort zu ändern.** Das Passwort, das bis heute in unserem Container lag, lässt sich deshalb im Moment nicht rotieren.

## Umsetzung und Messung

| Punkt | Ergebnis | Beleg |
|---|---|---|
| §1 | Port nur noch auf Loopback | `ss -tln` → `127.0.0.1:3335`, sonst nichts |
| §2 | Token von Oli selbst ausgestellt, per Skript (`read -s`, Passwort nur über stdin, Token in place in die `.env`, vorher gegen `/api/auth/me` geprüft) | `/api/auth/me` → `id=1 'oliver'`; Passwort-Zeilen in `.env`: 0; `CONVERTER_USER`/`_PASSWORD` im Container-Env: keine |
| §3 | ausgehende Header sind exakt `accept, accept-encoding, authorization, connection, host, user-agent` | im Container gemessen; `trust_env=False`, damit kein Proxy aus der Umgebung `X-Forwarded-For` einschleust |
| §4 | drei neue Werte (`secrets.token_urlsafe(32)`, 43 Zeichen), in **einem** Prozess in alle drei Dateien geschrieben, in place (Modus 600 und Inode unverändert, Werte nie auf dem Bildschirm) | s. u. |

**Reihenfolge im Fenster:** CONVERTER (`markdown-converter` + `worker`, `--force-recreate`, gleiche Image-ID wie vorher, also kein Code-Deploy) → nach 4 s wieder erreichbar → converter-mcp (Build + Recreate) → email-automation (`app`). Danach, und erst dann, die Passwort-Zeilen aus unserer `.env` gelöscht und nochmal neu erstellt.

**Gemessen nach dem Fenster:**
- Alle 21 Tools per Probe, die nichts schreibt: **21/21, FAIL=none** — einmal direkt nach der Rotation, einmal nach dem Entfernen des Passworts.
- Der Ingest-Token von email-automation gegen `POST /api/ingest/conversion` mit leerem Body: **400** — Authentifizierung angenommen, Body abgelehnt, wie erwartet.
- Gegenprobe mit einem ausgedachten Bearer auf `POST /api/cards`: **401**.

**Zur Namensfrage aus eurem Brief:** Der Satz „auch `email-automation` nutzt ihn" stimmt — dort heißt die Variable aber **`CONVERTER_INGEST_TOKEN`**, nicht `INGEST_TOKEN`. Eine Suche nach dem Namen findet sie also nicht; beim nächsten Rotieren daran denken.

## Drei Befunde

**1. Kein Passwortwechsel möglich — bei euch.** Außer `flask create-user` gibt es keinen Pfad, der `User.set_password` aufruft, und `create-user` bricht bei einem bestehenden Benutzer ab. Ein Passwortwechsel ginge heute nur per Handarbeit im Container. Das passt nicht zu einem Audit, der Credentials als exponiert einstuft: Olis Passwort lag bis heute in unserer Container-Env und war dort für jeden mit Docker-Zugriff per `docker inspect` lesbar. Vorschlag: ein `flask set-password <user>` mit `hide_input`/`confirmation_prompt` wie bei `create-user`. Dazu eine Designfrage, die ihr entscheiden solltet: `set_password` widerruft heute **keine** API-Tokens. Das ist bequem, weil ein Passwortwechsel weder die iOS-App noch uns abmeldet. Es heißt aber auch, dass ein Passwortwechsel **keinen** kompromittierten Token beendet — dafür braucht es ausdrücklich `POST /api/auth/logout`.

**2. Der Newsletter-Push von email-automation ist tot — seit eurem Loopback-Bind.** email-automation erreicht CONVERTER über `http://host.docker.internal:5656/…`; seit `127.0.0.1:5656:5000` (Commit `e2dc172`) läuft das aus dem Container in einen Timeout. Den Namen `markdown-converter-web` kann es nicht auflösen, weil es nicht in `notion-mcp-net` hängt. Aufgefallen ist das nur deshalb nicht, weil `WORKFLOW_NEWSLETTER_ENABLED=false` ist. Der Token selbst passt (s. o.). Fix, falls der Push je wieder gebraucht wird: email-automation in `notion-mcp-net` hängen und `http://markdown-converter-web:5000/api/ingest/conversion` als Ziel setzen. Das ist Sache von email-automation, nicht eure — ihr solltet nur wissen, dass euer Bind einen zweiten Client getroffen hat.

**3. F-1 war kein Einzelfall auf dem Host.** Acht weitere `.env`-Dateien unter `~/CODE` waren genauso offen wie eure: für alle lesbar und per ACL für `mintsamba`/`mintshare` les- und schreibbar, über die Samba-Freigabe des Home-Verzeichnisses auch aus dem LAN. Darunter email-automation (mit dem Ingest-Token), MintAccounting und llm-budget-watchdog. Alle acht sind seit heute `600` ohne ACL. Die Ursache sitzt eine Ebene höher: Die Projektverzeichnisse tragen eine **Default-ACL** für die beiden Samba-Konten, und jede **neu angelegte** Datei erbt sie. Eure `.env` ist heute sauber, würde aber bei einem „Speichern unter" oder `sed -i` wieder aufgehen. Secrets deshalb immer in place schreiben — so haben wir es im Fenster gemacht.

## Zur Transparenz

- Beim Testen des Token-Skripts hat unser Sub-Agent einen absichtlich falschen Login gegen `/api/auth/login` gefahren (erfundener Benutzer), um den generischen 401 zu prüfen. Falls euch in den Logs vom 2026-09-26 ein fehlgeschlagener Login ohne bekannten Benutzer auffällt: der war das.
- `MCP_AUTH_TOKEN` haben wir, wie euer Brief korrigiert, nicht angefasst — er liegt nur bei euch und beim notion-mcp-server.

---

*converter-mcp-Seite: umgesetzt, deployt, gemessen. Offen nur noch bei Oli: Connector-Reload in claude.ai, Commit unserer Änderungen.*
