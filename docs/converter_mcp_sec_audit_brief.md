# Developer-Brief an das converter-mcp-Team — SEC-AUDIT (Security-Audit der CONVERTER-Instanz)

> **An**: converter-mcp-Entwickler (Koordinator-Repo `~/CODE/converter-mcp`).
> **Von**: CONVERTER-Sub-Thread SEC-AUDIT, 2026-09-26.
> **Worum**: CONVERTER ist aus dem Internet erreichbar; der Security-Audit hat die Web-Instanz gehärtet ([Befund-Doc](archive/audit-outputs/AUDIT_SECURITY_2026-09-25.md)). **Für euren heutigen Betrieb ändert sich nichts, ihr müsst nichts tun, damit es weiterläuft** (live belegt, s. §3). Der Brief bittet um zwei Änderungen auf eurer Seite (§1, §2), nennt eine Regel, die ab jetzt gilt (§3), und schlägt eine koordinierte Token-Rotation vor (§4).
>
> **Gegen die lebende Oberfläche geprüft** (Hausregel seit 2026-09-19): Tool-Liste am Connector gelesen — **21 Tools**. **11 Session-Reads** über `CONVERTER_USER`/`CONVERTER_PASSWORD`: `list_conversions`, `list_audio_transcripts`, `get_transcript`, `get_narration_status`, `list_recent_highlights`, `list_highlights`, `list_cards`, `get_card`, `review_state`, `list_tags`, `list_collections`. **10 Token-Writes**: 8× `CARD_TOKEN` (`create_card`, `update_card`, `update_highlight`, `set_tag_parent`, `merge_tags`, `delete_tag`, `update_document`, `replace_section`), `create_narration` (`NARRATION_TOKEN`), `create_conversion` (`INGEST_TOKEN`). Dazu `server.py` gelesen (`ConverterClient._login`/`_get_json`, httpx, `follow_redirects=False`), eure Compose-Datei und die Live-nginx-Site.

## TL;DR

1. **Port auf Loopback binden:** `docker-compose.yml` `"3335:3335"` → `"127.0.0.1:3335:3335"`. Eure nginx-Site proxyt ohnehin auf `127.0.0.1:3335` — gefahrlos, belegt.
2. **Per-User-Bearer statt Passwort:** einmal `POST /api/auth/login` → Token in eure Env, `CONVERTER_PASSWORD` raus. Er deckt alle 11 Session-Reads (jede `@login_required`-View), ist widerrufbar und lässt keinen Klartext-Login mehr im Container liegen.
3. **Nie `X-Forwarded-*` an CONVERTER senden.** Ihr seid seit SEC-AUDIT ein **vertrauter Hop**.
4. **Token-Rotation koordiniert**, zusammen mit §2 — nicht im Alleingang.

## §1 — Port `3335` auf Loopback

Heute: `ports: - "3335:3335"` → Docker veröffentlicht auf **allen** Interfaces (gemessen: `0.0.0.0:3335` und `[::]:3335`). Der Container hält `CONVERTER_PASSWORD` und die drei Schreib-Tokens (`CARD_TOKEN`, `INGEST_TOKEN`, `NARRATION_TOKEN`); im LAN (und je nach Router-Forwarding von außen) ist sein Port damit **ohne TLS** erreichbar. Eure OAuth-Middleware (Nextcloud-Whoami) steht davor, TLS und nginx aber nicht.

Die öffentliche Tür geht schon heute über Loopback — Live-Site `/etc/nginx/sites-available/converter-mcp.smallpieces.de.conf`, Zeile 53: `proxy_pass http://127.0.0.1:3335/mcp;`. Euer Healthcheck ruft `http://localhost:3335/health` **im** Container und ist vom Bind unberührt. Also:

```yaml
    ports:
      - "127.0.0.1:3335:3335"
```

Nach dem Deploy messen: `ss -tln | grep 3335` zeigt nur `127.0.0.1:3335`; der Connector antwortet weiter (ein `list_conversions`).

CONVERTER hat denselben Schritt für sich gemacht (`127.0.0.1:5656:5000`, Commit `e2dc172`) — aus demselben Grund.

## §2 — Per-User-Bearer statt `CONVERTER_PASSWORD`

**Warum:** eure 11 Lese-Tools melden sich per Formular-Login mit Olis **echtem Passwort** an (`GET /login` → CSRF-Token scrapen → `POST /login`), und das Passwort liegt dafür dauerhaft in eurer Env. Seit MOBILE-AUTH gibt es den besseren Weg:

- `POST /api/auth/login` (JSON `{"username", "password", "label"}`) gibt **einmal** einen opaken per-User-Token aus; CONVERTER speichert nur seinen sha256.
- **Jede** `@login_required`-View akzeptiert `Authorization: Bearer <token>` — der `login_manager.request_loader` deckt sie ohne Per-View-Änderung. Das sind genau eure 11 Session-Reads (alle `GET`, alle `@login_required`).
- **Kein CSRF, kein Cookie, kein Scrapen**: Bearer-Anfragen überspringen die CSRF-Prüfung (Inversion); `_login()` und der Re-Login-bei-302-Pfad entfallen.
- **Widerrufbar** per `POST /api/auth/logout` mit genau diesem Token (Zeile wird gelöscht, wirksam ab der nächsten Anfrage) — anders als ein Passwort, das auch Olis Browser und iOS-App tragen.

Kontrakt: [docs/mobile_auth_contract.md](mobile_auth_contract.md) (Antwortformen, generisches 401, Label bis 80 Zeichen).

**Umbau-Skizze** (euer Repo, eure Entscheidung):

1. Token **einmalig** holen, mit `label: "converter-mcp"` (so ist er in der `api_token`-Tabelle von Olis iOS-Tokens unterscheidbar) — aus einer Shell, die das Passwort nicht in die History schreibt (z. B. `read -s`), Ziel `https://converter.smallpieces.de/api/auth/login` oder intern `http://markdown-converter-web:5000/api/auth/login`.
2. Token als z. B. `CONVERTER_TOKEN` in eure `.env`; `CONVERTER_USER` + `CONVERTER_PASSWORD` **entfernen**.
3. `ConverterClient._get_json` schickt `headers={"Authorization": f"Bearer {token}"}`; ein `401` ist dann **kein** Re-Login-Fall mehr, sondern ein Alarm („Token widerrufen oder ungültig") — ohne Passwort gibt es keinen automatischen Neu-Login, und das ist der Punkt.
4. Messen: alle 11 Lese-Tools einmal, dazu ein Write — die Writes behalten ihre eigenen Tokens.

⚠️ **Bekannte Grenze**: die Tokens laufen heute **nicht** ab (`expires_at = NULL`, CONVERTER-Backlog SEC-TOKEN-EXPIRY). Führt CONVERTER später einen Ablauf ein, bekommt euer Token denselben — dann stellt Oli ihn neu aus. Das ist der bewusste Tausch gegen ein Passwort in der Env.

## §3 — Was sich für euch geändert hat (und was nicht)

**Nichts gebrochen, live belegt:** nach beiden SEC-AUDIT-Deploys hat `list_conversions` über euren Connector geantwortet (229 Elemente), nach der `SECRET_KEY`-Rotation am 2026-09-26 ebenfalls (Neustart eures Containers, frische Session — euer Re-Login griff).

- **Euer http-Pfad ist unberührt.** Ihr sprecht `http://markdown-converter-web:5000` über Docker-DNS, **ohne** nginx. Das Session-Cookie ist bei CONVERTER seit SEC-AUDIT `Secure` **genau dann, wenn die Anfrage über HTTPS kam** — über euren plain-http-Weg bleibt es ohne `Secure`, und httpx schickt es zurück. Ein pauschales `Secure` hätte euren `POST /login` gebrochen (httpx schickt `Secure`-Cookies über http nie mit → Session fehlt → CSRF-400); CONVERTER hat deshalb diese Form gewählt und pinnt euren Login-Ablauf mit echtem httpx im Test (`tests/test_cookie_secure.py`). Das Remember-Cookie ist immer `Secure` — ihr braucht es nicht.
- **Das neue Login-Rate-Limit trifft euch nicht.** Es sitzt an nginx (10/min auf `/login` + `/api/auth/login`); Docker-DNS geht an nginx vorbei.
- **Neue Header auf jeder Antwort** (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`) — für einen API-Client ohne Belang.
- **Neue Regel — keine `X-Forwarded-*`-Header senden.** CONVERTER nutzt jetzt `ProxyFix` mit **einem** vertrauten Hop. Wer `:5000` direkt anspricht, **ist** dieser Hop — eure `X-Forwarded-*` würden ungeprüft übernommen: `X-Forwarded-For` fälschte die Client-IP in CONVERTERs Logs, und `X-Forwarded-Proto: https` machte eure Anfrage für CONVERTER zu einer HTTPS-Anfrage — dann verlangte Flask-WTF auf Session-Writes einen Same-Origin-`Referer` und euer `POST /login` stürbe am CSRF. Heute setzt `server.py` nur `Authorization` (Zeile 214) — bitte so lassen.

## §4 — Token-Rotation, koordiniert

Befund F-1 des Audits: CONVERTERs `.env` war bis 2026-09-26 **world-readable** auf einem Host mit ~30 Diensten und per Samba erreichbar. Oli hat die Modi geschlossen und `SECRET_KEY` rotiert; es gibt **keinen** Hinweis auf einen Abfluss. Die Tokens darin gelten trotzdem als exponiert. Drei davon liegen **auch in eurer Env** und brechen eure Schreib-Tools, wenn eine Seite allein rotiert:

| Token | CONVERTER | converter-mcp | Tools, die er trägt |
|---|---|---|---|
| `CARD_TOKEN` | ✔ | ✔ | 8 Writes (Karten, Highlight-Annotate, Tag-Baum, Dokument-Writes) |
| `INGEST_TOKEN` | ✔ | ✔ | `create_conversion` (auch `email-automation` nutzt ihn!) |
| `NARRATION_TOKEN` | ✔ | ✔ | `create_narration` |
| `CONVERTER_PASSWORD` | — | ✔ | eure 11 Reads — entfällt mit §2 |

**Vorschlag:** §2 und die Rotation in **einem** Fenster: neue Werte in beide `.env`, beide Container neu starten, alle 21 Tools einmal messen. ⚠️ `INGEST_TOKEN` trägt zusätzlich den Newsletter-Push aus `email-automation` — der gehört ins selbe Fenster.

**Korrektur zum Master-Stand:** `MCP_AUTH_TOKEN` liegt **nicht** in eurer Env (gemessen: nur `markdown-converter-web` und `notion-mcp-server` tragen ihn). Er ist CONVERTERs Credential gegenüber dem **notion-mcp-server** (`POST /api/conversions/<id>/send-to-notion` → `NOTION_MCP_URL`) — seine Rotation ist eine Sache zwischen CONVERTER und notion-mcp, nicht eure.

## Rückkanal

Rückmeldung bitte als `docs/converter_mcp_sec_audit_rueckmeldung.md` neben diesen Brief; die Antwort des Masters landet als `…_antwort.md`.
