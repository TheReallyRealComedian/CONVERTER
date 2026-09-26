# AUDIT — Security (OWASP Top 10 + Stack-Hot-Spots)

**Reihe**: Code-Check 1 von 5 (ARCH → CONSIST → TEST → DOC folgen) · **Sprint**: SEC-AUDIT · **Datum**: 2026-09-25 · **Prämisse**: Instanz ist **aus dem Internet erreichbar** (`converter.smallpieces.de`), nicht LAN-only.

Alle messbaren Befunde tragen Kommando **und** Ausgabe (redigiert wo nötig). Messungen gegen die deployte Instanz waren read-only (`curl` ohne Cookie, `docker exec … id`/`ls -l`/`getfacl`, DB nur `mode=ro`). Aktives Ausnutzen wurde **nicht** durchgeführt; wo ein Befund nur aktiv zu belegen wäre, steht ein `VERIFY:`-Prefix mit einem Messvorschlag für Oli. Der SSRF- und der unstructured-CVE-Pfad sind **per Code-Trace** eindeutig aufgelöst (kein präpariertes Fixture nötig).

---

## Risiko-Übersicht

| OWASP-Kategorie | Funde | Höchster Schweregrad | Quick-Win verfügbar |
|---|---|---|---|
| A01 Broken Access Control | 0 | — (geprüft, kein Finding) | — |
| A02 Cryptographic Failures | 3 (F-1 Secrets-Modi, F-5 Cookie-Secure, F-11 Backups) | **High** | ja (F-5), teils (F-1/F-11 = Mintbox) |
| A03 Injection | 0 | — (geprüft, kein Finding) | — |
| A04 Insecure Design | 3 (F-3 Login-Throttling, F-6 Remember-Cookie, F-14 Logout per GET) | **Medium** | ja (nginx für F-3) |
| A05 Security Misconfiguration | 4 (F-2 Header, F-8 root-Container, F-10 :5656, F-12 MCP-Port) | **Medium** | ja (F-2, F-10) |
| A06 Vulnerable Components | 0 laufzeit-erreichbar (CVE-Tabelle unten aufgelöst) | Low (DEPS-FLOAT) | — |
| A07 Authentication Failures | 3 (F-3 Brute-Force, F-4 Enumeration, F-15 Bearer ohne Ablauf) | **Medium** | ja (F-4) |
| A08 Data Integrity Failures | 1 (F-7 Redis-Auth + RQ-Pickle) | **Medium** | teils (Redis-Pw) |
| A09 Logging & Monitoring | 1 (F-9 remote_addr = Proxy) | Low | ja (ProxyFix) |
| A10 SSRF | 1 (F-6-SSRF Playwright-PDF) | **Medium** | nein (eigenes Item) |
| Hot-Spot: Blast-Radius | 1 (F-13 Worker-docker.sock → Host-Root) | **High** | nein (strukturell) |

**Nicht anwendbar** (Notion-Vorlage nennt sie, für diesen Stack irrelevant — nicht stillschweigend weggelassen): RLS/Postgres-Row-Security (SQLite, keine DB-Rollen), JWT-Handling (kein JWT — opake Bearer-Tokens per sha256-Lookup), Multi-Tenant-Isolation (Single-User; Owner-Scope ist die einzige Grenze und greift, s. A01).

---

## Findings

Schweregrad-Konvention: **Critical** = Datendiebstahl/RCE ohne Vorbedingung · **High** = defeat-auth oder Host-Kompromittierung mit realistischer Vorbedingung · **Medium** = auth-gated oder adjazenter Angreifer nötig · **Low** = geringer Hebel / Single-User-Selbstbezug.

| # | OWASP | Fundstelle | Vulnerability | Exploit-Szenario (1 Satz) | Schwere | Fix | Aufwand |
|---|---|---|---|---|---|---|---|
| F-1 | A02/A05 | Mintbox `~/CODE/CONVERTER/.env`, `google-credentials.json` (`-rwxrwxr-x+`, other `r-x`) | Geheimnisse (SECRET_KEY, sechs Tokens, GCP-Creds, SMTP-Pw) world-readable auf einem Host mit ~32 Diensten und Samba-Export von `/home/oliver` | Jeder lokale Prozess/Nutzer oder Samba-Nutzer (die `MintHome`-Freigabe listet `www-data` = jede Web-App der Box) liest `SECRET_KEY` und fälscht damit ein gültiges Session-Cookie für Oli — **ohne Passwort**, der Login-Zaun ist umgangen. | **High** | `chmod 600`, ACL entfernen (`setfacl -b`); Backups aus der Samba-Freigabe nehmen | XS (Mintbox, Olis Go) |
| F-2 | A05 | nginx-Site-Config (kein `add_header`) + App (kein `after_request`) | Keine Security-Header: kein HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy | Das Login-Formular ist per iframe einbettbar (fehlendes X-Frame-Options) → Clickjacking-Overlay fischt Zugangsdaten; ohne HSTS ist der Erst-Besuch SSL-strip-bar. | **Medium** | App-Header per `after_request` + HSTS an nginx (Block unten) | XS/S |
| F-3 | A04/A07 | `app_pkg/auth.py:17` + `app_pkg/mobile_auth.py:92` | Kein Rate-Limit / Lockout an `/login` **und** `/api/auth/login` | Ein Angreifer rät aus dem Internet unbegrenzt Passwörter gegen das eine Konto (scrypt bremst auf ~47 ms/Versuch, stoppt aber nichts — Credential-Stuffing läuft durch). | **Medium** | `limit_req` an nginx für beide Pfade (Block unten); Oli setzt es (sudo) | S (nginx) |
| F-4 | A07 | `app_pkg/auth.py:24-25` | Username-Enumeration per Timing-Orakel am **Web**-Login (der Mobile-Login ist korrekt konstant) | Antwortzeit trennt „User existiert nicht" (0,4 ms) von „User existiert, falsches Passwort" (47,6 ms) → der gültige Username ist vor dem Raten bestätigt. | **Low** | Dummy-Hash brennen wie in `mobile_auth._DUMMY_PASSWORD_HASH` | XS |
| F-5 | A02 | `app_pkg/__init__.py:51-54` | Session- und Remember-Cookie **ohne `Secure`-Flag** (HttpOnly + SameSite=Lax sind gesetzt) | Ein einziger Klartext-`http://`-Request vor dem HSTS-Griff (Mixed-Content-Subressource, LAN-Zugriff an nginx vorbei über `:5656`, MITM-Downgrade) sendet das Session-Cookie im Klartext → Session-Hijack auf vertrauliche Meeting-Transkripte. **DSGVO-relevant** (Dritt-PII in Transkripten). | **Medium** | `SESSION_COOKIE_SECURE`/`REMEMBER_COOKIE_SECURE` env-gegated (Mintbox an, Mac aus) | XS |
| F-6 | A04/A07 | `app_pkg/auth.py:26` + Flask-Login-Default | `login_user(remember=True)` **immer** + kein `REMEMBER_COOKIE_DURATION` (Default 365 Tage); Logout invalidiert nur clientseitig | Ein gestohlenes Remember-Cookie (mit F-5 im Klartext abgreifbar) ist ein Jahr gültig und lässt sich serverseitig nicht widerrufen (statisch signiert; nur SECRET_KEY-Rotation zieht es zurück). | **Low→Medium** (Kette mit F-5) | `REMEMBER_COOKIE_DURATION` auf begründeten Wert (z. B. 30 Tage) | XS |
| F-6-SSRF | A10 | `app_pkg/markdown.py:193,219` + `markdown_render._URL_SCHEMES` (http/https) | Server-seitiger Fetch beliebiger http(s)-URLs im Markdown→PDF-Renderer (Playwright `set_content(wait_until='networkidle')` im Container) | Ein authentifizierter Aufrufer (Session **oder** ein gestohlener, nie ablaufender per-User-Bearer) reicht Markdown mit `<img src="http://redis:6379/…">` / `http://converter-mcp-server:3335/…` ein — Playwright lädt es serverseitig aus dem Container, der auf `converter_default` **und** `notion-mcp-net` sitzt, und wird zur internen Netz-Sonde. | **Medium** | Private/Loopback-Ziele im Renderer blocken (IP-Pinning/Allow-List) — eigenes Item, s. Strukturelle Bewegungen | M |
| F-7 | A08 | `docker-compose.yml:3-6` (Redis) + RQ-Default-Serializer (`pickle`) | Redis **ohne** `requirepass`, `protected-mode no`; RQ serialisiert Jobs mit `pickle`, der Worker de-pickled sie | Wer einen Fuß ins `converter_default`-Netz bekommt (via F-6-SSRF-Write oder einen anderen Container), schiebt einen präparierten Pickle-Job in die Queue → Code-Ausführung im Worker → der hält den docker.sock (F-13) = **Host-Root**. | **Medium** (adjazent; Redis hat keinen veröffentlichten Port) | `requirepass` setzen; Redis bleibt netz-intern | S |
| F-8 | A05 | `Dockerfile` (kein `USER`) | Web **und** Worker laufen als **root** (`uid=0`) | Jeder Code-Exec-Bug (Parser der Dokument-Engines auf angreifer-beeinflussten Dateien, F-7) läuft als root im Container statt als unprivilegierter Nutzer — größere Wirkung, leichteres Ausbrechen. | **Medium** (Defense-in-Depth) | Non-root `USER` im Dockerfile (Volume-Perms prüfen) | S/M (strukturell) |
| F-9 | A09 | `app_pkg` (kein `ProxyFix`) | `request.remote_addr` ist für **jede** Anfrage `172.21.0.1` (Docker-Gateway), nie die Client-IP | Der einzige Auth-Fehler-Log (`Mobile login failed from 172.21.0.1`) ist blind für die Herkunft, und **jeder künftige Rate-Limit-Schlüssel auf `remote_addr` träfe alle Clients als einen** — Throttling wäre wirkungslos. | **Low** | `ProxyFix(x_for=1, x_proto=1, x_host=1, x_port=1)` (genau ein vertrauter Hop) | XS |
| F-10 | A05 | `docker-compose.yml:19-20` | Web-Container hört auf `0.0.0.0:5656` (4 `docker-proxy`-Prozesse) | Im LAN ist die App **ohne TLS** an nginx vorbei erreichbar → Klartext-Session-Cookies (F-5) und der ganze Login umgeht HSTS/Header. iOS-App und MCP nutzen beide `https://…` bzw. internes DNS — niemand braucht `:5656` direkt. | **Low→Medium** | Compose-Bind `127.0.0.1:5656:5000` | XS |
| F-11 | A02 | Mintbox `~/app_data_bak-2026-06-22/` (root, `r--r--r--`) + neun `~/converter.db.pre-*` (oliver, `r--r--`) | Vollständige DB-Kopien (Meeting-Transkripte, Passwort-Hash) world-readable in der Samba-exportierten Home | Ein Samba-/Local-Nutzer liest eine DB-Kopie → alle Transkripte + der scrypt-Hash zum Offline-Cracken. **DSGVO-relevant**. | **Low→Medium** | `chmod 600` / aus der Freigabe nehmen (Backlog MINTBOX-BAK) | XS (Mintbox) |
| F-12 | A05 | Mintbox `converter-mcp-server` Port `0.0.0.0:3335` | Der MCP-Server (hält `INGEST/CARD/NARRATION_TOKEN` + `CONVERTER_PASSWORD` in env, proxyt zur App) ist auf **allen** Interfaces veröffentlicht | Wer `:3335` erreicht (LAN sicher; extern = `VERIFY:` Router-Forwarding), spricht die Agent-Schreibfläche mit den dort hinterlegten Tokens an. | **Medium** | Bind `127.0.0.1:3335` (Fremd-Repo `notion-mcp-server`/MCP-Compose — Hinweis an Oli) | XS (Fremd-Config) |
| F-13 | Hot-Spot (Blast-Radius) | `docker-compose.yml:69` | Worker mountet den Host-`/var/run/docker.sock` (**root-äquivalent**) und läuft als root | Nicht direkt aus dem Internet, aber **ein** Worker-Code-Exec (F-7, oder ein Engine-Parsing-Bug) wird zu voller Host-Root-Kontrolle über ~32 Container. Unter LAN-only (DOC-LOCAL) bewusst akzeptiert — unter Internet-Exposition neu zu bewerten. | **High** (Amplifier) | Stufenleiter s. u. (Socket-Proxy / rootless / eigener Nutzer) | L (strukturell) |
| F-14 | A04 | `app_pkg/auth.py:36` (`/logout` ist `GET`) | Zustandsändernde Nutzer-Aktion per GET — die einzige der App (die Reconcile-GETs schreiben nur Job-Status fort; CSRF-Token schützen nur POST/PUT/PATCH/DELETE) | Eine Seite auf **irgendeiner** `*.smallpieces.de`-Subdomain (same-site: `SameSite=Lax` schickt dort das Cookie auch an Subressourcen) meldet Oli per `<img src="…/logout">` ab; von fremden Sites nur per Top-Level-Navigation. Nur Ärgernis, kein Datenzugriff. *(Nachtrag Phase 3)* | **Low** | `/logout` auf `POST` mit CSRF-Token umstellen | XS |
| F-15 | A02/A07 | `models.ApiToken.expires_at` (bei allen 3 Tokens `NULL`) | Per-User-Bearer der iOS-App laufen nie ab; Widerruf nur per `POST /api/auth/logout` mit genau diesem Token oder Zeilen-Delete | Ein abgeflossener iOS-Token (Geräte-Backup, verlorenes Telefon) liest und schreibt alles, bis jemand merkt, welche Zeile zu löschen ist — es gibt keinen Ablauf und keine Übersicht. *(Nachtrag Phase 3)* | **Low** | `expires_at` beim Ausstellen setzen (App meldet sich bei 401 neu an) + Token-Übersicht mit Widerruf; Kontrakt MOBILE-AUTH | S |

---

## Top-3-Risiken

1. **F-1 — Geheimnisse world-readable auf einem geteilten Host (High).** Warum zuerst: `SECRET_KEY` im Klartext lesbar **hebelt die gesamte Auth aus** — mit ihm wird ein Session-Cookie gefälscht, kein Passwort nötig. Die Hürde ist niedrig, weil `/home/oliver` world-readable und via Samba (`www-data`!) exportiert ist und ~30 weitere Dienste auf der Box denselben FS teilen. Reversibel und billig zu schließen (`chmod 600`), aber solange offen macht es jeden anderen Befund zweitrangig.
2. **F-3 — Kein Login-Throttling auf der Internet-Fläche (Medium, aber die exponierteste Fläche).** Warum: `/login` und `/api/auth/login` sind die einzigen aus dem Internet erreichbaren Schreibpunkte vor der Auth, und gegen **ein** Konto läuft Online-Rate/Credential-Stuffing ungebremst. scrypt (gemessen ~47 ms) ist die einzige Bremse — das ist Verzögerung, kein Stopp. `limit_req` an nginx ist der wirksame Ort (F-9 macht app-seitiges Throttling ohne ProxyFix wertlos).
3. **F-13 + F-7 — Kette Worker-Code-Exec → Host-Root (High als Amplifier).** Warum: der docker.sock am Worker verwandelt jeden Code-Exec-Bug in Host-Root über die ganze Box. F-7 (Redis ohne Auth + Pickle) ist der konkrete, heute existierende Zünder aus dem internen Netz; F-6-SSRF ist ein möglicher Weg **in** dieses Netz. Die Kette ist real, jedes Glied belegt — sie rechtfertigt die strukturelle Socket-Härtung.

---

## Quick-Wins (XS/S, hohe Wirkung — Kandidaten für Phase 2)

Jeder mit Ort, Aufwand und geplanter Messung an der Kante.

- **QW-1 (F-2): App-Header per `after_request`** in `app_pkg/__init__.py`: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, minimale `Permissions-Policy`. **Aufwand XS.** Test: Test-Client prüft die Header; Messung: `curl -I https://converter.smallpieces.de/login` vorher/nachher. ⚠️ `X-Frame-Options: DENY` gegen den Vorschau-iframe belegen — der ist `srcdoc` (same-origin, kein Cross-Origin-Frame), sollte weiter rendern; im Smoke bestätigen.
- **QW-2 (F-5): Cookie-`Secure` env-gegated** (`SESSION_COOKIE_SECURE`/`REMEMBER_COOKIE_SECURE`), Mintbox an, Mac ohne TLS aus (sonst ist der Dev-Login tot). **Aufwand XS.** Test: Config-Assertion im Test-Client; Messung: `curl -D-` zeigt `Secure` am `set-cookie`.
- **QW-3 (F-6): `REMEMBER_COOKIE_DURATION`** auf einen begründeten Wert (Vorschlag 30 Tage). **Aufwand XS.** Test: Config-Assertion.
- **QW-4 (F-4): Dummy-Hash am Web-Login** — `auth.py` brennt bei unbekanntem User einen `check_password_hash` wie `mobile_auth._DUMMY_PASSWORD_HASH`. **Aufwand XS.** Test: Timing-Angleichung im Test-Client (das Skript aus 1.3 wiederverwenden).
- **QW-5 (F-9): `ProxyFix(x_for=1, x_proto=1, x_host=1, x_port=1)`** um die WSGI-App (genau **ein** vertrauter Hop). **Aufwand XS.** Test: Header-Injektion im Test-Client (`X-Forwarded-For` → `remote_addr`); Messung: eine fehlgeschlagene Login-Zeile im Container zeigt die echte Client-IP.
- **QW-6 (F-10): Compose-Bind `127.0.0.1:5656:5000`.** **Aufwand XS.** Messung: `ss -tln | grep 5656` auf dem Host zeigt `127.0.0.1:5656` statt `0.0.0.0`; App bleibt über nginx erreichbar.
- **QW-7 (F-1/F-11): Datei-Modi auf der Mintbox** — `chmod 600 .env google-credentials.json`, ACL prüfen (`setfacl -b`), Backups aus der Samba-Freigabe. **Aufwand XS (Mintbox-Runtime, Olis Go).** Messung: `ls -l` + `getfacl` vorher/nachher im Bericht.
- **QW-8 (F-7): Redis `requirepass`** (env, in beiden Containern als `REDIS_URL=redis://:pw@redis:6379`). **Aufwand S.** Messung: `redis-cli PING` ohne Auth → `NOAUTH`, mit → `PONG`.

**Nur-Vorschlag an Oli (nginx = Mintbox-Systemconfig, sudo — kein Edit ohne Go):**

```nginx
# in den server{} 443-Block, vor location /:
add_header Strict-Transport-Security "max-age=86400" always;   # klein anfangen, KEIN preload
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;

# oben im http{} (nginx.conf) oder server{}:
limit_req_zone $binary_remote_addr zone=login:10m rate=10r/m;
# im location-Match für /login und /api/auth/login:
location = /login            { limit_req zone=login burst=5 nodelay; proxy_pass http://127.0.0.1:5656; }
location = /api/auth/login   { limit_req zone=login burst=5 nodelay; proxy_pass http://127.0.0.1:5656; }
```
(⚠️ Wenn App-Header per `after_request` **und** nginx `add_header` gleichzeitig gesetzt werden, doppeln sie sich — Header entweder nur in der App **oder** nur an nginx. Empfehlung: HSTS an nginx, der Rest in der App, damit `:5656`-Direktzugriffe die Header auch tragen.)

---

## Strukturelle Bewegungen (Backlog-Items mit Stufenleiter)

- **SEC-SOCKET — Worker-docker.sock härten (F-13, L).** Stufenleiter, je mit Kosten:
  1. **Socket-Proxy mit Allow-List** (`tecnativa/docker-socket-proxy` o. ä.): der Worker spricht einen Proxy, der nur `container create/start/remove` des `mineru`-Images zulässt — kein `exec`, kein `inspect` fremder Container, kein Volume-Mount außerhalb des Exchange. Kosten: ein Sidecar, eine Allow-List. **Empfohlen** (kleinster Umbau, größter Gewinn).
  2. **Rootless Docker** für den mineru-Lauf: kein root-äquivalenter Socket. Kosten: GPU-Passthrough unter rootless verifizieren (Olis ComfyUI teilt die Karte) — riskant.
  3. **Dediziertes mineru-Sidecar statt on-demand-Container**: kein Socket am Worker. Kosten: hält 6,5 GB VRAM dauerhaft (die DOC-LOCAL bewusst vermied) — steht Olis ComfyUI im Weg. Nicht empfohlen.
- **SEC-NONROOT — Non-Root-Container (F-8, S/M).** `USER` im Dockerfile; die Volume-Perms (`app_data`, `podcast_data`, `doclocal_exchange`) auf den neuen uid anpassen und den Startup-`flock`/`create_all`-Pfad prüfen.
- **SEC-SSRF — Renderer-Egress einschränken (F-6-SSRF, M).** Im Markdown→PDF-Pfad (und im EPUB-Pfad, falls ebooklib fetcht — s. VERIFY) private/loopback/link-local-Ziele blocken (IP-Pinning nach DNS-Auflösung, nicht nur URL-Regex). Alternativ: Playwright den Netzzugang komplett nehmen (nur `data:`/vendored Assets im PDF) — RICH-MEDIA-https-Bilder rendern dann nicht mehr im PDF (Fähigkeits-Abwägung).
- **SEC-REDIS-AUTH — Redis absichern (F-7, S).** `requirepass` + optional den RQ-Serializer von Pickle auf JSON (RQ unterstützt `serializer=`) — Pickle-Jobs sind der eigentliche RCE-Hebel; JSON-Serializer nimmt ihn weg, kostet aber, dass Job-Argumente JSON-fähig sein müssen (heute: ints, strings, dict/list — passt).

**Backlog-Bezug (nicht neu erfinden):** CSP-BASELINE (M) — Voraussetzung ist der Play-CDN-Ausbau (s. u.); DEPS-FLOAT (M) — trägt filelock/Werkzeug-Pinning; MINTBOX-BAK (XS) — deckt F-11.

---

## Defense-in-Depth-Lücken (separat ausgewiesen)

- **Login-Fence ist die einzige Schicht.** Kein zweiter Faktor, kein Throttling (F-3), keine IP-Allow-List, keine Fail2ban-Anbindung. Fällt das Passwort oder ein Token, ist alles offen. MFA ist für einen Single-User bewusst kein Muss, aber Throttling (F-3) ist die minimale zweite Schicht und fehlt.
- **Redis ohne Auth (F-7)** ist die fehlende Schicht zwischen „im Netz" und „Code im Worker". Der fehlende veröffentlichte Port ist heute die einzige Schicht davor.
- **Kein CSP (Backlog CSP-BASELINE).** nh3 + Autoescape + die DOM-Knoten-Doktrin sind die vorhandenen XSS-Schichten und greifen; CSP wäre die fehlende Netz-/Skript-Schicht darüber. ⚠️ Voraussetzung: der **Tailwind-Play-CDN** (`base.html:10`, `login.html:9`) injiziert zur Laufzeit `<style>` in jede Seite — eine CSP ohne `unsafe-inline` für `style-src` ist mit ihm **unmöglich**. CSP-BASELINE muss neu geschnitten werden als *„erst Play-CDN raus (Tailwind CLI zur Build-Zeit, statisches CSS im Image), dann CSP"*.
- **Supply-Chain ohne SRI.** `cdn.tailwindcss.com` (ungepinnt, JIT-Compiler in jeder Seite inkl. Login), `markdown-it@14.1.0` und `mermaid@10` (schwebend) laden **ohne** SRI. Kompromittierung eines CDN = Skript im Nutzerkontext jeder Seite (Session-Cookie ist `HttpOnly`, aber jeder authentifizierte Request läuft). SRI-Pin für `markdown-it`/`mermaid` ist Template-only und ein sauberer Phase-2-Kandidat (der Reader lädt `mermaid@10.9.8` **mit** SRI seit RICH-MEDIA — den Hash übernehmen); der Play-CDN ist das eigene Item (CSP-BASELINE-Voraussetzung).
- **`mermaid_converter.js` mit `securityLevel: 'loose'`** (mermaid_converter.js:7). Rendert nur die selbst eingegebene Mermaid-Quelle des angemeldeten Nutzers, Output wird nicht persistiert/geteilt → geringer Hebel, aber `'strict'` wie im Reader wäre die konsistente Härtung. (Low, kein eigenes Finding — hier als DiD-Notiz.)

---

## Sektion: „Unter LAN-only entschieden" — Neubewertung unter Internet-Exposition

Urteil je Punkt aus dem Angreifermodell (Internet-erreichbar, ein Login-Zaun, sechs Env-Tokens, Worker→Host-Root über den Socket).

| Entscheidung | Urteil | Begründung |
|---|---|---|
| **Docker-Socket am Worker** (DOC-LOCAL) | **hält mit Auflage** | Der Nutzen (GPU nur während des Auftrags) bleibt gültig, aber der root-äquivalente Socket ist unter Internet-Exposition der Blast-Radius-Verstärker (F-13). Auflage: Socket-Proxy mit Allow-List (SEC-SOCKET Stufe 1). Der Socket wurde mit DOC-WEB-ASYNC korrekt vom **Web**-Container entfernt — dieser Teil hält voll. |
| **root in beiden Containern** | **hält nicht** | Unter LAN-only ein Schönheitsfehler, unter Internet-Exposition + Socket eine Eskalationsstufe (F-8). → SEC-NONROOT. |
| **Single-User-Env-Tokens inkl. `CARD_TOKEN` als generischer Schreib-Token** | **hält** | Fail-closed, constant-time, nie geloggt — die Token-Mechanik ist sauber. Dass `CARD_TOKEN` drei Schreibflächen bedient (Cards, Highlight-Annotate, Docwrite, Tag-Baum) ist eine bewusste, dokumentierte Bündelung; ein Leak trifft alle drei, aber alle drei sind derselbe Vertrauens-Level (Agent-Schreibzugriff). Das echte Token-Risiko liegt nicht im Modell, sondern in F-1 (Tokens world-readable in `.env`) und F-12 (Tokens im exponierten MCP-Container). |
| **Kein Login-Throttling** | **hält nicht** | Unter LAN-only vertretbar, unter Internet-Exposition die offene Brute-Force-Fläche (F-3). → nginx `limit_req`. |
| **Traceback-Tail an den Client** | **hält** | Geprüft: ein unbehandelter 500 liefert mit `debug=False` (gunicorn) die nackte „Internal Server Error"-Seite, **keinen** Traceback an den Client (nur `@errorhandler(413)` + `CSRFError` sind registriert, kein 500-Handler der etwas leakt). Der gespeicherte Traceback-**Tail** (`exc_info[-2000:]`) landet in `metadata_json` und in der UI — aber nur für den **Owner** sichtbar (owner-scoped Reads) und er trägt einen Python-Traceback, keine Env/Token (die Tokens stehen in `os.environ`, nicht in den Exceptions dieser Pfade). Kein Finding. |
| **`0.0.0.0:5656`** | **hält nicht** | Umgeht TLS+Header+HSTS im LAN (F-10, verstärkt F-5). → Compose-Bind `127.0.0.1`. |

---

## VERIFY-Oli-Liste (nur du kannst das prüfen/entscheiden)

- **VERIFY-1 — Router-Port-Forwarding:** Reicht der Router extern `5656` (F-10) oder `3335` (F-12) durch? Von hier nicht messbar (ufw/Router braucht sudo/Zugang). Prüfen: `sudo ufw status` und die Portweiterleitungs-Regeln des Routers. Sind beide **nicht** geforwardet, ist F-10 ein reiner LAN-Befund (bleibt XS-Quick-Win) und F-12 ein LAN/Host-Befund.
- **VERIFY-2 — nginx-Header + Rate-Limit:** Die Header (HSTS/X-Frame-Options/…) und `limit_req` gehören an nginx (Systemconfig, sudo). Die Blöcke oben sind Vorschläge — setzen und mit `curl -I` gegenprüfen liegt bei dir.
- **VERIFY-3 — EPUB-Egress:** Ob `ebooklib` beim EPUB-Bau Remote-Bilder (`<img src="https://…">`) serverseitig nachlädt (zweite SSRF-Fläche neben F-6-SSRF), ist per Code plausibel *nein* (ebooklib schreibt HTML in den Container, holt keine externen Ressourcen), aber nicht empirisch belegt — ein Kindle-Versand eines Dokuments mit einem `<img>` auf einen von dir kontrollierten Zeugen-Endpoint würde es zeigen. Niedrige Priorität.
- **VERIFY-4 — Samba-Reichweite:** Die `MintHome`-Freigabe (`/home/oliver`) listet `www-data` als valid user. Prüfen, ob das gewollt ist — es macht `.env`/Backups (F-1/F-11) für jede Web-App der Box lesbar, selbst nach `chmod 600` bliebe der Samba-Pfad, wenn die Freigabe mit einem privilegierten Nutzer läuft.

---

## CVE-Tabelle aus dem Master-Ist-Zustand — je Zeile aufgelöst

Reproduktion der Master-Methode: `docker exec markdown-converter-web pip freeze` → Erreichbarkeit per Aufrufer-Grep im Code.

| Paket | Version | CVE | Erreichbarkeit (belegt) | Urteil |
|---|---|---|---|---|
| unstructured | 0.18.32 | CVE-2026-71428 (SSRF via `partition(url=…)`) | **Nicht erreichbar.** Einziger Aufruf: `document_router.py:62` `partition(filename=source_path, strategy="fast", paragraph_grouper=…)` — nie `url=`. EML/HTML werden aus der **lokalen Datei** partitioniert, ohne Remote-Fetch-Flag. | **Kein Finding** |
| filelock | 3.14.0 | CVE-2026-22701, CVE-2025-68146 (TOCTOU-Symlink, lokal) | Braucht lokalen Schreibzugriff auf das Lock-Verzeichnis im Container; als root voll wirksam, aber der Angreifer ist dann schon im Container. Eingefroren durch `constraints.txt` (DEPS-FLOAT). | **Low** (Defense-in-Depth; via DEPS-FLOAT lösen) |
| torch | 2.12.1+cpu | CVE-2025-3000 (Modell-Loading) | **Nicht erreichbar** — nur transitiv, nie Inferenz (IMG-SLIM: `partition(strategy="fast")`, kein Modell-Load). | **Kein Finding** |
| accelerate | 1.14.0 | CVE-2026-69112 (Path-Traversal `load_checkpoint`) | **Nicht erreichbar** — `load_checkpoint` wird nirgends aufgerufen (Grep: 0 Treffer). | **Kein Finding** |
| nltk | 3.10.3 | CVE-2026-81726 (Model-Path-Handling) | Transitiv über unstructured; die `fast`-Sentence-Tokenisierung nutzt gebündelte NLTK-Daten (im Image, Dockerfile), **kein** angreifer-kontrollierter Model-Path. | **Kein Finding** (via DEPS-FLOAT mitheben) |
| setuptools | 81.0.0 | CVE-2026-59890 (sdist-Build) | Build-Zeit, nicht Laufzeit — kein sdist-Build im laufenden Container. | **Kein Finding** (Laufzeit) |

**Zusätzlich gemessen (nicht in der Master-Tabelle):** Werkzeug **3.1.8** im Container (Mac-Repo installiert 3.1.6 — der Split ist harmlos, aber benannt), Flask-Login 0.6.3, nh3 0.2.18, lxml 6.1.2, RQ 2.8.0. Werkzeug 3.1.8 ist der aktuelle 3.1.x-Stand; kein offener CVE bekannt zum Auditdatum. **Passwort-Hash: `scrypt:32768:8:1`** (Werkzeug-Default, stark) — A02-Hashing ist sauber, kein md5/sha1.

---

## Validierungs-Checkliste (beantwortet)

- [x] **Hat jedes Finding ein konkretes Exploit-Szenario?** Ja — jede Zeile der Findings-Tabelle trägt genau einen Satz.
- [x] **Wurden Top-3-Risiken explizit ausgewiesen?** Ja (F-1, F-3, F-13+F-7).
- [x] **Schweregrad-Verteilung plausibel?** Ja: 2× High (F-1, F-13), 7× Medium, 5× Low/Low→Medium über **14** Einträge (F-1…F-13 plus F-6-SSRF — der Phase-1-Bericht sprach von „13 Findings", weil F-6-SSRF keine eigene Nummer trägt; korrigiert im Wrap), Phase 3 ergänzt F-14/F-15 (beide Low) — kein Critical (keine unauthentifizierte RCE / kein unauth. Datendiebstahl im Code; die schwersten Punkte brauchen einen lokalen/adjazenten Fuß oder eine Auth), nichts pauschal Low.
- [x] **Defense-in-Depth-Lücken separat ausgewiesen?** Ja (eigene Sektion: Login-Fence, Redis, CSP/Play-CDN, SRI, mermaid loose).
- [x] **Compliance-relevante Findings markiert?** Ja — F-5 und F-11 als **DSGVO-relevant** (Dritt-PII in Meeting-Transkripten).
- [x] **VERIFY:-Prefix korrekt verwendet?** Ja — für Router-Forwarding, nginx-Config, EPUB-Egress, Samba-Reichweite (alles, was sudo/Router/aktives Zeugen-Testen braucht).

---

## Anhang B — Routen-Tabelle (A01, alle 75 Routen, kein Sampling)

Methodik: `flask_app.url_map` + Decorator-Scan + CSRF-Exempt-Set (`csrf._exempt_views`), dazu ein dynamischer Probe mit dem Test-Client **ohne** Cookie/Bearer und mit **falschem** Bearer. Legende Auth: `LOGIN` = `@login_required` (Session oder per-User-Bearer), `CARD` = `CARD_TOKEN`-Gate, `NARR` = `NARRATION_TOKEN`, `INGEST` = `INGEST_TOKEN`, `DUAL` = Session/Bearer **oder** `DOC_CONVERT_TOKEN`, `PUBLIC` = kein Gate. „unauth" = Statuscode ohne Auth, „badbearer" = mit falschem Bearer.

**Ergebnis A01: kein Finding.** Jede der 63 `/api`-Routen ist gated; jede ID-tragende Route prüft den Owner (`get_owned_conversion` / `user_id == current_user.id` / `_get_owned_*` / `_parse_owned`) und gibt für fremde/fehlende IDs **404** (nie 403, kein Existenz-Leak). Ohne Auth: 302→/login (GET-Seiten), 400 (CSRF vor Auth bei Session-Mutationen) oder 401 (Token-Flächen). Mit falschem Bearer: durchweg 401 (fail-closed — die CSRF-Inversion überspringt bei Bearer-Präsenz, die Auth killt den ungültigen Token). Die einzigen zwei `PUBLIC`-Routen sind die beiden Login-Endpunkte (per Design öffentlich).

| Methoden | Pfad | Auth | CSRF-exempt | unauth | badbearer | Owner-Check | Datei |
|---|---|---|---|---|---|---|---|
| GET | `/` | LOGIN | – | 302→/login | 401 | — | markdown.py:142 |
| POST | `/api/auth/login` | PUBLIC | ✔ | 401 | 401 | — | mobile_auth.py:92 |
| POST | `/api/auth/logout` | LOGIN | – | 400 | 401 | präsentierter Token | mobile_auth.py:131 |
| GET | `/api/auth/me` | LOGIN | – | 401 | 401 | eigene Identität | mobile_auth.py:123 |
| GET | `/api/cards` | LOGIN | – | 302→/login | 401 | `user_id==current_user` | cards.py:535 |
| POST | `/api/cards` | CARD | ✔ | 401 | 401 | `user_id=target` | cards.py:378 |
| DELETE | `/api/cards/<id>` | LOGIN | – | 400 | 401 | `user_id==current_user` | cards.py:803 |
| GET | `/api/cards/<id>` | LOGIN | – | 302→/login | 401 | `user_id==current_user` | cards.py:559 |
| PATCH | `/api/cards/<id>` | CARD | ✔ | 401 | 401 | `user_id=target` | cards.py:441 |
| POST | `/api/cards/<id>/annotate` | LOGIN | – | 400 | 401 | `user_id==current_user` | cards.py:772 |
| POST | `/api/cards/<id>/review` | LOGIN | – | 400 | 401 | `user_id==current_user` | cards.py:730 |
| GET | `/api/collections` | LOGIN | – | 302→/login | 401 | `Collection.user_id==current_user` | collections.py:33 |
| POST | `/api/collections` | LOGIN | – | 400 | 401 | `user_id=current_user` | collections.py:65 |
| DELETE | `/api/collections/<id>` | LOGIN | – | 400 | 401 | `_get_owned_collection` | collections.py:130 |
| PATCH | `/api/collections/<id>` | LOGIN | – | 400 | 401 | `_get_owned_collection` | collections.py:93 |
| POST | `/api/collections/<id>/cards` | LOGIN | – | 400 | 401 | `_get_owned_collection`+`user_id` | collections.py:142 |
| DELETE | `/api/collections/<id>/cards/<cid>` | LOGIN | – | 400 | 401 | `_get_owned_collection`+`user_id` | collections.py:162 |
| GET | `/api/conversions` | LOGIN | – | 302→/login | 401 | `user_id==current_user` | library.py:389 |
| POST | `/api/conversions` | LOGIN | – | 400 | 401 | `user_id=current_user` | library.py:470 |
| DELETE | `/api/conversions/<id>` | LOGIN | – | 400 | 401 | `get_owned_conversion` | library.py:585 |
| GET | `/api/conversions/<id>` | LOGIN | – | 302→/login | 401 | `get_owned_conversion` | library.py:541 |
| PUT | `/api/conversions/<id>` | LOGIN | – | 400 | 401 | `get_owned_conversion` | library.py:550 |
| PATCH | `/api/conversions/<id>/content` | CARD | ✔ | 401 | 401 | `user_id=target` | docwrite.py:64 |
| GET | `/api/conversions/<id>/highlights` | LOGIN | – | 302→/login | 401 | `get_owned_conversion` | highlights.py:53 |
| POST | `/api/conversions/<id>/highlights` | LOGIN | – | 400 | 401 | `get_owned_conversion` | highlights.py:22 |
| POST | `/api/conversions/<id>/place` | LOGIN | – | 400 | 401 | `get_owned_conversion` | library.py:647 |
| PATCH | `/api/conversions/<id>/progress` | LOGIN | – | 400 | 401 | `get_owned_conversion` | library.py:600 |
| POST | `/api/conversions/<id>/queue` | LOGIN | – | 400 | 401 | `get_owned_conversion` | library.py:699 |
| PATCH | `/api/conversions/<id>/section` | CARD | ✔ | 401 | 401 | `user_id=target` | docwrite.py:96 |
| POST | `/api/conversions/<id>/send-to-kindle` | LOGIN | – | 400 | 401 | `get_owned_conversion` | kindle.py:21 |
| POST | `/api/conversions/<id>/send-to-notion` | LOGIN | – | 400 | 401 | `get_owned_conversion` | notion.py:98 |
| POST | `/api/conversions/<id>/tags` | LOGIN | – | 400 | 401 | `get_owned_conversion` | library.py:740 |
| DELETE | `/api/conversions/<id>/tags/<tid>` | LOGIN | – | 400 | 401 | `get_owned_conversion`+`tag.user_id` | library.py:763 |
| GET | `/api/csrf-token` | LOGIN | – | 302→/login | 401 | — | __init__.py:389 |
| POST | `/api/document-conversions` | DUAL | – | 400 | 401 | `user_id=target` | document_api.py:414 |
| GET | `/api/document-conversions/<id>` | DUAL | – | 401 | 401 | `user_id=target` | document_api.py:544 |
| GET | `/api/document-conversions/settings` | LOGIN | – | 302→/login | 401 | current_user-Blob | document_api.py:567 |
| PUT | `/api/document-conversions/settings` | LOGIN | – | 400 | 401 | current_user-Blob | document_api.py:578 |
| GET | `/api/get-deepgram-token` | LOGIN | – | 302→/login | 401 | — | audio.py:234 |
| DELETE | `/api/highlights/<id>` | LOGIN | – | 400 | 401 | `.user_id!=current_user`→404 | highlights.py:63 |
| PATCH | `/api/highlights/<id>` | LOGIN | – | 400 | 401 | `.user_id!=current_user`→404 | highlights.py:74 |
| PATCH | `/api/highlights/<id>/annotate` | CARD | ✔ | 401 | 401 | `.user_id!=target`→404 | cards.py:493 |
| POST | `/api/highlights/<id>/tags` | LOGIN | – | 400 | 401 | `_get_owned_highlight` | tags.py:425 |
| DELETE | `/api/highlights/<id>/tags/<tid>` | LOGIN | – | 400 | 401 | `_get_owned_highlight`+`tag.user_id` | tags.py:449 |
| GET | `/api/highlights/recent` | LOGIN | – | 302→/login | 401 | `Conversion.user_id==current_user` | cards.py:349 |
| POST | `/api/ingest/conversion` | INGEST | ✔ | 401 | 401 | `user_id=target` (resolver) | ingest.py:119 |
| GET | `/api/learn/settings` | LOGIN | – | 302→/login | 401 | current_user-Blob | learn.py:437 |
| PUT | `/api/learn/settings` | LOGIN | – | 400 | 401 | current_user-Blob | learn.py:442 |
| GET | `/api/learn/simulate` | LOGIN | – | 302→/login | 401 | — (nur eigene Settings) | learn.py:507 |
| GET | `/api/learn/stats` | LOGIN | – | 302→/login | 401 | `Card.user_id==current_user` | learn.py:468 |
| POST | `/api/narrations` | NARR | ✔ | 401 | 401 | `user_id=target` | narration.py:234 |
| GET | `/api/narrations/<id>` | LOGIN | – | 302→/login | 401 | `get_owned_conversion` | narration.py:325 |
| GET | `/api/narrations/<id>/audio` | LOGIN | – | 302→/login | 401 | `get_owned_conversion` + Traversal-Guard | narration.py:343 |
| POST | `/api/narrations/<id>/retry` | LOGIN | – | 400 | 401 | `get_owned_conversion` | narration.py:385 |
| GET | `/api/notion/suggestions` | LOGIN | – | 302→/login | 401 | — | notion.py:80 |
| GET | `/api/review-state` | LOGIN | – | 302→/login | 401 | `user_id==current_user`+`_parse_owned` | cards.py:565 |
| GET | `/api/tags` | LOGIN | – | 302→/login | 401 | `Tag.user_id==current_user` | tags.py:122 |
| DELETE | `/api/tags/<id>` | LOGIN | – | 400 | 401 | `.user_id!=current_user`→404 | tags.py:463 |
| PATCH | `/api/tags/<id>` | LOGIN | – | 400 | 401 | `.user_id!=current_user`→404 | tags.py:157 |
| POST | `/api/tags/delete` | CARD | ✔ | 401 | 401 | `user_id=target_user` | tags.py:316 |
| POST | `/api/tags/merge` | CARD | ✔ | 401 | 401 | `user_id=target_user` | tags.py:231 |
| POST | `/api/tags/parent` | CARD | ✔ | 401 | 401 | `user_id=target_user` (by-name) | tags.py:191 |
| POST | `/api/transcriptions` | LOGIN | – | 400 | 401 | `user_id=current_user` | audio.py:245 |
| GET | `/api/transcriptions/<id>` | LOGIN | – | 302→/login | 401 | `user_id==current_user` | audio.py:361 |
| GET | `/audio-converter` | LOGIN | – | 302→/login | 401 | — | audio.py:223 |
| POST | `/convert-markdown` | LOGIN | – | 400 | 401 | — (Formular-Eingabe; **F-6-SSRF**) | markdown.py:156 |
| GET | `/document-converter` | LOGIN | – | 302→/login | 401 | — | documents.py:43 |
| GET | `/library` | LOGIN | – | 302→/login | 401 | `user_id==current_user` | library.py:224 |
| GET | `/library/<id>` | LOGIN | – | 302→/login | 401 | `get_owned_conversion` | library.py:375 |
| GET,POST | `/login` | PUBLIC | – | 200/400 | 200/200 | — (**F-4** Timing) | auth.py:17 |
| GET | `/logout` | LOGIN | – | 302→/login | 401 | — | auth.py:36 |
| GET | `/mermaid-converter` | LOGIN | – | 302→/login | 401 | — | mermaid.py:7 |
| GET | `/review` | LOGIN | – | 302→/login | 401 | — | cards.py:342 |
| GET | `/tags` | LOGIN | – | 302→/login | 401 | — | tags.py:117 |
| POST | `/transform-document` | LOGIN | – | 400 | 401 | — (Nicht-PDF, secure_filename) | documents.py:52 |

**CSRF-Inversion — die 11 Exempts gegen die Token-Gates geprüft:** Alle 11 CSRF-exempten Views sind token-gated (INGEST/CARD/NARRATION), keine ist session-erreichbar ohne Token → korrekt (ein session-loser Token-Caller trägt kein CSRF-Cookie). Umgekehrt: `/api/document-conversions` ist token-gated (DOC_CONVERT_TOKEN) und **bewusst nicht** exempt — die Bearer-Präsenz überspringt die Inversion, ein Cookie-Session-POST behält CSRF (steht im Kontrakt §… / document_api.py-Docstring). Kein weiterer token-gated-aber-nicht-exempt-Fall gefunden. **A03/A04-Konsistenz: sauber.**

**Injection (A03) — geprüft, kein Finding:** kein `shell=True`, kein `os.system`/`os.popen`/`eval`/`exec`/`pickle.loads`/`yaml.load` im App-Code (Grep: 0). Alle fünf `subprocess.run` sind Listen-Argumente mit id-abgeleiteten Pfaden (nie roher Dateiname). Raw-SQL: `learn.py:124` (`json_patch`) ist parameter-gebunden (`:updates`/`:uid`); alle `tags.py`-`execute`-Stellen sind SQLAlchemy-Core-Konstrukte (`junction.delete()/update()`, `select()`), kein f-String. SSTI: Jinja-Autoescape an, eine `|safe`-Stelle (nh3-gereinigter Renderer-Output, `library_detail.html:104`) + `script_safe` (`Markup`, `</script`-escaped) — keine Nutzereingabe in Template-**Strings**. XSS: 30 echte `innerHTML`-Stellen (39 inkl. Kommentaren) — alle leeren oder setzen statischen/`escHtml`-maskierten Text; `library_detail.js:380` maskiert selbst; die zwei fremd-String-Senken (Karten-SVG `review.js:134`, Mermaid-SVG im Shadow-Root `reader_figures.js:131`) sind server-sanitisiert bzw. `securityLevel:'strict'`. Mass-Assignment: alle Create/PUT-Routen nehmen **explizite** Felder, kein `**data` ins Model.

**Datei-Upload-Hot-Spot — geprüft:** Vier Upload-Stellen; alle leiten die Endung aus `secure_filename(...)` ab (auch `documents.py:62-63` — die Master-Notiz „rohe Endung" ist **überholt**, `secure_filename` läuft davor). Speicher-Pfade sind id-abgeleitet (`source_<id>.<ext>`, `narration_<id>.wav`), kein Traversal aus dem Namen. Größen-Limits: global 500 MB (Audio), Dokument-Service 100 MB (Content-Length-Preflight + On-Disk-Backstop). MIME wird nicht getraut (Endung + Inhalt entscheiden). `narration_audio` hat zusätzlich einen `is_relative_to(OUTPUT_DIR)`-Traversal-Guard. **Kein Finding.**

---

## Anhang C — Mess-Belege (Kommando + Ausgabe, redigiert)

**Kante — Security-Header + TLS (F-2, F-5):**
```
$ curl -sS -D - -o /dev/null https://converter.smallpieces.de/login
HTTP/2 200
server: nginx
content-type: text/html; charset=utf-8
vary: Cookie
set-cookie: session=<redacted>; HttpOnly; Path=/; SameSite=Lax     # kein Secure
# kein strict-transport-security / content-security-policy / x-frame-options /
#   x-content-type-options / referrer-policy / permissions-policy
$ curl -sS -o /dev/null -w '%{http_code} -> %{redirect_url}\n' http://converter.smallpieces.de/login
301 -> https://converter.smallpieces.de/login
$ curl … /api/conversions   → 302 -> …/login?next=%2Fapi%2Fconversions   # Auth-Zaun steht
```

**Container-Nutzer + Socket (F-8, F-13):**
```
$ docker exec markdown-converter-web  id   → uid=0(root) gid=0(root) groups=0(root)
$ docker exec markdown-converter-worker id → uid=0(root) gid=0(root) groups=0(root)
$ ls -l /var/run/docker.sock → srw-rw---- 1 root docker …    ($ getent group 996 → docker:x:996:oliver)
```

**Bind + Proxy (F-9, F-10):**
```
$ ss -tln | grep 5656 → LISTEN 0.0.0.0:5656  und  [::]:5656   ($ pgrep -af docker-proxy.*5656 → 4)
$ docker network inspect converter_default → gateway 172.21.0.1   # == das gemessene remote_addr
$ docker logs …web | grep 'login failed' → [WARNING] Mobile login failed from 172.21.0.1
```

**Geheimnis-Modi + ACL (F-1):**
```
$ ls -l .env google-credentials.json
-rwxrwxr-x+ 1 mintshare mintshare 1742 … .env
-rwxrwxr-x+ 1 mintshare mintshare 2361 … google-credentials.json
$ getfacl .env → user:mintsamba:rwx  user:mintshare:rwx  group::rwx  other::r-x
$ ls -ld /home/oliver → drwxrwxr-x+ …   # Home world-r-x
# smb.conf: [MintHome] path=/home/oliver/ browseable=yes valid users=…,www-data,oliver
```

**Backups (F-11):**
```
$ ls -ld ~/app_data_bak-2026-06-22 → drwxr-xr-x+ root root …
$   … / → -rw-r--r--+ root root converter.db  +  converter.db.bak-mcp1fix
$ ls -l ~/converter.db.pre-* → 9 Kopien, oliver, -rw-r--r--  (bis 10 MB, RICH-MEDIA)
```

**MCP-Port + Netz (F-12):**
```
$ docker port converter-mcp-server → 3335/tcp -> 0.0.0.0:3335 (+ [::])
$ docker port notion-mcp-server    → 3333/tcp -> 127.0.0.1:3333   # lokal-only, korrekt
# converter-mcp env: CONVERTER_BASE_URL=http://markdown-converter-web:5000 ; hält NARRATION/INGEST/CARD_TOKEN + CONVERTER_PASSWORD
# markdown-converter-web sitzt auf converter_default UND notion-mcp-net
```

**Redis (F-7):**
```
$ docker exec file-transformer-redis redis-server --version → v=8.4.0
$ redis-cli CONFIG GET requirepass → (leer, Länge 0)     $ … protected-mode → no     $ docker port … → (kein Port)
# RQ DefaultSerializer: pickle.dumps(protocol=HIGHEST) / pickle.loads
```

**Passwort-Hash + Tokens (A02, F-6):**
```
$ sqlite3 file:/app/data/converter.db?mode=ro  "SELECT id, password_hash FROM user"
  user 1 -> scrypt:32768:8:1           # stark, Werkzeug-Default
  api_token: 3 rows | with expires_at: 0 | oldest 2026-07-13   # Bearer laufen nie ab (F-6-Kette)
```

**Login-Timing (F-4) — lokaler Test-Client (Test-DB, nie Prod), Median über 20:**
```
web /login              unknown user:   0.4 ms (200)  |  known user + wrong pw:  47.6 ms (200)   ← Orakel
mobile /api/auth/login  unknown user:  47.7 ms (401)  |  known user + wrong pw:  47.5 ms (401)   ← konstant (Dummy-Hash)
```

**SSRF-Code-Trace (F-6-SSRF) — kein Fixture, reiner Pfad:**
```
markdown.py:193  html = render_markdown_to_html(markdown_text)   # _URL_SCHEMES enthält 'http','https'; img@src erlaubt http/https
markdown.py:219  await page.set_content(full_html, wait_until='networkidle')   # Playwright lädt Subressourcen SERVERSEITIG im Container
→ <img src="http://<internes-ziel>"> im Nutzer-Markdown wird vom Container geladen. Auth-gated (@login_required).
```

**unstructured CVE-2026-71428 (nicht erreichbar):**
```
$ grep -rn 'partition(' services app_pkg
services/document_router.py:62  partition(filename=source_path, strategy="fast", paragraph_grouper=…)   # nie url=
```

---

*Ende des Befunds. Nichts wurde auf der Mintbox oder im Repo verändert; alle Messungen waren read-only. Kein Wegwerf-User angelegt (die Befunde ließen sich statisch/read-only belegen — der im Sprint vorgesehene SSRF-Zeugentest wurde durch den eindeutigen Code-Trace ersetzt und steht als VERIFY-3 offen). Kein Token, kein Hash-Wert, kein Cookie-Wert im Dokument.*

---

## Phase 2 — umgesetzt (2026-09-26)

Sieben freigegebene Quick-Wins, je ein Commit, Suite **1129 + 1 Skip → 1159 + 1 Skip**, zwei Deploys (nach Punkt 3 und nach Punkt 7). Jede Messung an der deployten Instanz; Wegwerf-User `zz_smoke` nach jedem Gate strikt per `user_id` gelöscht (besaß nichts), keine Reste in Container oder Host.

| # | Punkt | Commit | Test (Gegenprobe) | Messung vorher → nachher |
|---|---|---|---|---|
| 1 | ProxyFix (`x_for/x_proto/x_host/x_port=1`) | `f250bcb` | `test_proxy_fix.py` 8 (ohne ProxyFix: 4 rot) | Log `Mobile login failed from 172.21.0.1` → `… from 94.x.x.3` (echte Client-IP, redigiert) |
| 2 | Security-Header per `after_request` | `7ba617c` | `test_security_headers.py` 6 (ohne Hook: 6 rot) | keine → `nosniff`, `DENY`, `strict-origin-when-cross-origin`, `Permissions-Policy` auf HTML, 401-JSON, statischer Datei; Gate Reader-Smoke 155/155 |
| 3 | Cookie-`Secure` — **Form geändert** (s. u.) | `b019f0f` | `test_cookie_secure.py` 5 inkl. MCP-Login mit echtem httpx (ohne Interface: 2 rot) | `session=…; HttpOnly; Path=/; SameSite=Lax` → `session=…; Secure; HttpOnly; Path=/; SameSite=Lax` |
| 4 | Dummy-Hash am Web-Login, eine Definition (`User.authenticate`) | `dc27112` | `test_login_enumeration.py` 6, mock-basiert (alter Web-Login: 2 rot) | Timing lokal, Median/20: Web unbekannt vs. bekannt+falsch **0,4 vs. 47,6 ms → 47,7 vs. 47,3 ms** (Mobile unverändert 47,5/47,5) |
| 5 | `REMEMBER_COOKIE_DURATION` 30 Tage | `fc9396b` | `test_remember_cookie.py` 2 (ohne: 2 rot) | Expires 365 → 30 Tage (Test liest das `Set-Cookie` eines echten Logins) |
| 6 | Compose-Bind `127.0.0.1:5656:5000` | `e2dc172` | Sentinel in `test_proxy_fix.py` (alte Compose: rot) | `ss -tln`: `0.0.0.0:5656` + `[::]:5656` → nur `127.0.0.1:5656`; vom Mac im LAN: `Couldn't connect to server`, über nginx `200` |
| 7 | SRI `markdown-it@14.1.0` + `mermaid@10.9.8` | `e26c699` | `test_cdn_sri.py` 2 (alte Templates: 2 rot) | Headless im Container: beide Bibliotheken laden, Diagramm rendert, **kein** Integrity-Fehler; Reader-Smoke 155/155 |

**Benannte Abweichungen und Verhaltensänderungen:**

- **Punkt 3 in anderer Form als freigegeben** (Olis Entscheidung in P2): Secure-by-default mit `ALLOW_INSECURE_COOKIES`-Opt-out hätte den `converter-mcp` gebrochen. Er meldet sich per Cookie-Session über plain http im Docker-Netz an (`http://markdown-converter-web:5000`), und httpx schickt `Secure`-Cookies über http nie zurück — schon sein Login-POST verlöre die Session und stürbe am CSRF (im Test mit echtem httpx belegt). Umgesetzt: `HttpsOnlySecureSessionInterface` — Session-Cookie `Secure` genau dann, wenn die Anfrage über HTTPS kam (hinter nginx jede Browser-Anfrage), Remember-Cookie immer. Kein Schalter, kein `.env`-Eintrag.
- **ProxyFix schaltet Flask-WTFs SSL-strict-Zweig ein:** hinter nginx ist `is_secure` jetzt wahr, jede Cookie-Session-Mutation braucht einen Same-Origin-`Referer` (Browser senden ihn; die neue `Referrer-Policy` behält ihn). An der Kante belegt mit einer absichtlich fehlschlagenden Anmeldung (Fantasie-Username, gültiges CSRF-Token): mit Referer `200` (View erreicht), ohne `400`. Bearer-Writes und der plain-http-Form-Login des MCP sind unberührt (getestet).
- **Grenze von Punkt 5:** der Remember-Cookie-Wert ist `user_id|digest` ohne Zeitstempel (an der installierten Flask-Login-Quelle belegt). Die 30 Tage begrenzen den ehrlichen Browser, nicht einen gestohlenen Wert — der Widerrufshebel ist die `SECRET_KEY`-Rotation. Weil `SECRET_KEY` bis heute world-readable in `.env` lag (F-1), gehört die Rotation zu Olis `chown`/`chmod`-Schritt.

**Zwischen Phase 1 und 2 von Oli an nginx gesetzt** (Site-Config gelesen): `Strict-Transport-Security: max-age=86400` auf Server-Ebene, `limit_req` 10 r/min mit `burst=10` → `429` auf `location = /login` und `location = /api/auth/login`, die `proxy_set_header`-Zeilen auf Server-Ebene (von den Login-Locations geerbt — belegt durch die echte Client-IP im `/api/auth/login`-Log). Keine Header-Dubletten zur App. **Gemessen von Oli** (von der Mintbox, 2026-09-26 15:24): HSTS `max-age=86400`, `/api/conversions` `302`, `POST /api/auth/login` zwölfmal → zehnmal `401`, dann zweimal `429`; vom Mac: HSTS da, Statics `200`, Login `401`.

**Stand der Findings nach Phase 2:**

| Finding | Stand |
|---|---|
| F-1 Geheimnisse world-readable | **geschlossen** — Oli 2026-09-26: `chown oliver:oliver` + `setfacl -b` + `chmod 600` (gemessen `-rw------- oliver oliver`, ACL leer, Deploy intakt) **und** `SECRET_KEY` rotiert (Nachtrag unten) |
| F-2 Security-Header | **geschlossen** (App-Header + HSTS an nginx) |
| F-3 Login-Throttling | **geschlossen** an der Kante (nginx `limit_req`, Oli) — gemessen: 12× Mobile-Login → 10× `401`, 2× `429` |
| F-4 Enumeration am Web-Login | **geschlossen** |
| F-5 Cookie ohne `Secure` | **geschlossen** (Secure hinter HTTPS) |
| F-6 Remember-Cookie 365 Tage | **teils** — 30 Tage; Widerruf nur per `SECRET_KEY`-Rotation |
| F-6-SSRF Playwright-PDF | offen → Item SEC-SSRF |
| F-7 Redis ohne Auth + Pickle | offen → Item SEC-REDIS-AUTH |
| F-8 root-Container | offen → Item SEC-NONROOT |
| F-9 `remote_addr` = Proxy | **geschlossen** (ProxyFix, gemessen) |
| F-10 `0.0.0.0:5656` | **geschlossen** (Loopback-Bind, gemessen) |
| F-11 Backups world-readable | **Modi geschlossen** (Oli: die drei Backup-Verzeichnisse — eins im Home, zwei im Clone — `drwx------ oliver`, die neun `converter.db.pre-*` `-rw-------`); Löschen offen → MINTBOX-BAK |
| F-12 MCP-Port `0.0.0.0:3335` | offen — Brief an converter-mcp (Phase 3) |
| F-13 Worker-docker.sock | offen → Item SEC-SOCKET |
| F-14 Logout per GET | offen, Low → P3-Reminder |
| F-15 Bearer ohne Ablauf | offen, Low → P3-Reminder (MOBILE-AUTH) |
| DiD Supply-Chain SRI | **teils** — markdown-it + mermaid gepinnt; Tailwind-Play-CDN bleibt (CSP-BASELINE-Voraussetzung) |

**Nachtrag Master 2026-09-26 16:15:** `SECRET_KEY` rotiert (Oli), Sicherung gelöscht, beide Container + converter-mcp neu gestartet, MCP-Aufruf und Kante danach gemessen — F-1 vollständig geschlossen.
