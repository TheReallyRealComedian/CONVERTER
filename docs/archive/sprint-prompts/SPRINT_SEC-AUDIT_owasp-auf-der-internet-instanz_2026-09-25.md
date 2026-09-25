# SPRINT SEC-AUDIT — Security-Audit der Internet-Instanz (OWASP Top 10 + Stack-Hot-Spots)

**Größe**: M (3 Phasen; Phase 1 ist lesend) · **Datum**: 2026-09-25 · **Vorhaben**: Code-Check-Reihe, Audit 1 von 5 (Notion-Katalog „Audit — Security", Projekt MINTBOX, 2026-05-16)

## Warum

Oli, 2026-09-25: *„ich würde gerne mal wieder einen code-check über die applikation laufen lassen — dazu haben wir anregungen in notion"*, dann: *„wir beginnen mit security"*. Der letzte Code-Check war die Cleanup-Welle im Mai (SEC + CVE-Sprints, Stage-4-Sweep). Seitdem: 384 Commits, 100 Sprints, 71 → 1129 Tests — und **eine geänderte Prämisse**: am 2026-08-21 stellte sich heraus, dass die Instanz nicht LAN-only ist, sondern unter `converter.smallpieces.de` aus dem Internet erreichbar. CLAUDE.md sagt seither selbst, dass mehrere Sicherheitsabwägungen unter der falschen Annahme getroffen wurden und neu zu bewerten sind. Genau das ist dieser Sprint: **nicht** „ist der Code sicher", sondern „welche unter LAN-only getroffenen Entscheidungen halten unter Internet-Exposition, und was fehlt".

## Gegroundeter Ist-Zustand (Master — gemessen 2026-09-25, nicht neu herleiten; Abweichungen im Bericht benennen)

**Kante.** Host-nginx auf der Mintbox terminiert TLS (Let's Encrypt), `http://` → 301 auf `https://`, `proxy_pass http://127.0.0.1:5656`, setzt `X-Forwarded-For/Proto/Host/Port`, `client_max_body_size 500M`, alle Timeouts 1800 s. **Kein `add_header`** in der Site-Config (`/etc/nginx/conf.d/converter.smallpieces.de.conf`, nur lesbar per ssh). Gemessen am Login ohne Cookie: `server: nginx`, `set-cookie: session=…; HttpOnly; Path=/; SameSite=Lax` — **kein HSTS, kein CSP, kein X-Frame-Options, kein X-Content-Type-Options, kein Referrer-Policy, kein `Secure`-Flag**. `GET /api/conversions` ohne Auth → 302 `/login?next=…`; `POST /api/auth/login` mit leerem JSON → 401.

**Der Web-Container hört auf `0.0.0.0:5656`** (`docker-compose.yml` Ports `5656:5000`; `ss` auf dem Host zeigt `0.0.0.0:5656` und `[::]:5656`). Damit ist die App im LAN **ohne TLS an nginx vorbei** erreichbar. Ob der Router 5656 nach außen weiterreicht, ist **nicht messbar von hier** (ufw braucht sudo) → **VERIFY-Oli**.

**Kein `ProxyFix`** in der App (grep `ProxyFix|X-Forwarded|is_secure` → 0 Treffer). Folge: `request.remote_addr` ist für jede Anfrage `127.0.0.1`, `request.is_secure` ist `False`. Das Login-Log `Mobile login failed from %s` ([app_pkg/mobile_auth.py:112](../../../app_pkg/mobile_auth.py)) protokolliert damit die Proxy-Adresse, und **jeder künftige Rate-Limit-Schlüssel auf `remote_addr` träfe alle Clients als einen**.

**Prozesse.** Web **und** Worker laufen als **root** (`docker exec … id` → `uid=0`); das [Dockerfile](../../../Dockerfile) hat keinen `USER`. Der Worker hält `/var/run/docker.sock` (`srw-rw---- root 996`) — **gesperrte Entscheidung aus DOC-LOCAL** (mineru als Geschwister-Container, GPU nur während des Auftrags), damals unter LAN-only. `app.py:92` trägt `app.run(debug=True)` — nur unter `__main__`, das Image startet gunicorn.

**Auth-Flächen.** (1) Flask-Login-Session-Cookie, `login_user(user, remember=True)` **immer** ([app_pkg/auth.py:26](../../../app_pkg/auth.py)), keine `REMEMBER_COOKIE_DURATION` gesetzt (Flask-Login-Default 365 Tage), `HttpOnly` + `SameSite=Lax` gesetzt, `Secure` nicht. (2) Per-User-Bearer (`ApiToken`, sha256-Hash, Revoke = Row-Delete, MOBILE-AUTH). (3) **Sechs Env-Tokens**: `INGEST_TOKEN`, `CARD_TOKEN` (seit MCP-DOCWRITE-WRAP bewusst der **generische Agent-Schreib-Token**, auch für Docwrite), `NARRATION_TOKEN`, `DOC_CONVERT_TOKEN` (eingehend, fail-closed, `hmac.compare_digest`), `MCP_AUTH_TOKEN` und `NOTION_TOKEN` (**ausgehend**, [app_pkg/integrations/notion.py](../../../app_pkg/integrations/notion.py)). CSRF läuft über die **Inversion** ([app_pkg/__init__.py](../../../app_pkg/__init__.py) `_register_csrf_inversion`, repliziert Flask-WTF 1.2.1); **elf** CSRF-exempte Views (tags ×3, ingest, cards ×3, mobile-login, docwrite ×2, narration) — alle token-authed außer dem öffentlichen `POST /api/auth/login`. Anti-Enumeration am Mobile-Login per Dummy-Hash (`_DUMMY_PASSWORD_HASH`); **kein Rate-Limit, kein Lockout** an keinem Login. Passwort-Hash: `generate_password_hash` Default bei Werkzeug **3.1.8** (= scrypt) — der Prefix in der Prod-DB ist ungemessen.

**Eingaben.** Vier Upload-Stellen (`markdown.py:160`, `audio.py:260`, `document_api.py:435`, `documents.py:58`); `secure_filename` an drei davon, [app_pkg/documents.py:63](../../../app_pkg/documents.py) leitet die Endung aus dem **rohen** Namen ab. Fünf `subprocess.run`-Stellen (pandoc in `office_backends.py:64`, ffprobe/ffmpeg in `audio_chunker.py:57/218` und `transcription_jobs.py:144`, `docker run` in `pdf_local.py:231/236`). Raw-SQL: Core-Konstrukte in `tags.py`, `text()` in `learn.py:124` (`json_patch`) und in den statischen Migrationen. Templates: Jinja-Autoescape, **eine** `|safe`-Stelle (`library_detail.html:104`, der nh3-gereinigte Renderer-Output), ein `Markup`-Filter für `</script`-Escaping (`__init__.py:434`). **39 `innerHTML`-Stellen** in `static/js` — die Doktrin seit CARD-MD/RICH-MEDIA erlaubt genau zwei Senken für fremde Strings (Karten-Figuren, Mermaid-SVG im Shadow-Root); die anderen 37 sind zu klassifizieren, nicht zu vermuten. Markdown-Rendering: markdown-it-py + **ein** `nh3.clean` (Pin 0.2.18), SVG-Policy `services/svg_sanitize.py`, 27 + 15 Angriffs-Batterien in RICH-MEDIA ohne Befund — **nicht wiederholen**, nur die Ränder prüfen (siehe VERIFY-Liste).

**Externe Ressourcen (Supply Chain).** [templates/base.html:10](../../../templates/base.html) und [templates/login.html:9](../../../templates/login.html) laden **`https://cdn.tailwindcss.com`** — den Tailwind **Play-CDN**: ungepinnt, ohne SRI, ein JIT-Compiler, der zur Laufzeit `<style>` in **jede Seite inklusive Login** injiziert. Dazu `markdown-it@14.1.0` ohne SRI (`markdown_converter.html:153`), `mermaid@10` **schwebend** ohne SRI (`mermaid_converter.html:77`; der Reader lädt seit RICH-MEDIA `10.9.8` **mit** SRI), Google Fonts. **Keine CSP** (BACKLOG CSP-BASELINE, M). ⚠️ Der Play-CDN ist die stille Voraussetzung von CSP-BASELINE: eine CSP ohne `unsafe-inline` für `style-src` ist mit ihm nicht möglich.

**Ausgehend.** Notion-API (`api.notion.com`, feste Basis-URL, Pfad aus Code), MCP-Server (`MCP_AUTH_TOKEN` als Bearer, Ziel aus Env), Gemini/Deepgram/Google-TTS-SDKs, SMTP (Kindle, server-fester Empfänger). **Server-seitige Fetches durch Playwright**: `generate_pdf` rendert das Dokument-HTML in headless Chromium **im Web-/Worker-Container** — `<img src="https://…">` in einem Dokument wird dort vom Container aus geladen, nicht vom Browser des Nutzers. Das ist der SSRF-Kandidat dieses Stacks (unten in der VERIFY-Liste).

**Geheimnisse auf der Mintbox.** `~/CODE/CONVERTER/.env` und `google-credentials.json` sind **`-rwxrwxr-x+`** (world-readable, group-writable, ausführbar, mit ACL) auf einem Host, der ~30 weitere Container und Dienste trägt. `app_data_bak-2026-06-27/` und `-06-28/` (BACKLOG MINTBOX-BAK) sind root-owned, world-readable, je ~2,7 MB DB-Kopie im Home.

**Abhängigkeiten — CVE-Stand des laufenden Web-Containers** (Master, 2026-09-25, `pip freeze` im Container → PyPI-JSON-API je Paket, 183 Pakete, 0 Fehler; ⚠️ `pip-audit -r` scheitert zweimal: am `+cpu`-Pin von torch und danach an einem **Resolver-Konflikt** `filelock==3.14.0` vs. `huggingface_hub==1.28.0`/`python-discovery==1.5.0` — die per `constraints.txt` eingefrorene Umgebung ist für pips Resolver nicht installierbar, ein DEPS-FLOAT-Datenpunkt):

| Paket | Version | CVE | Fix in | Klasse | Vorab-Einschätzung (zu verifizieren) |
|---|---|---|---|---|---|
| unstructured | 0.18.32 | CVE-2026-71428 | 0.24.0 | SSRF über `partition(url=…)` | wir übergeben Dateipfade — **prüfen, ob irgendein Pfad `url=` oder Auto-Fetch (HTML/EML mit Remote-Ressourcen) erreicht** |
| filelock | 3.14.0 | CVE-2026-22701, CVE-2025-68146 | 3.20.3 | TOCTOU-Symlink, lokaler Angreifer | eines der zehn `constraints.txt`-Pakete (DEPS-FLOAT); im Container als root → Symlink-Race hätte volle Wirkung, braucht aber lokalen Schreibzugriff |
| torch | 2.12.1+cpu | CVE-2025-3000 | 2.13.0 | Modell-Loading | nur transitiv, nie Inferenz (IMG-SLIM) |
| accelerate | 1.14.0 | CVE-2026-69112 | – | Path-Traversal `load_checkpoint` | nie aufgerufen |
| nltk | 3.10.3 | CVE-2026-81726 | – | Model-Path-Handling | transitiv über unstructured; prüfen, ob `partition(strategy="fast")` nltk-Modelle lädt |
| setuptools | 81.0.0 | CVE-2026-59890 | 83.0.0 | sdist-Build | Build-Zeit, nicht Laufzeit |

**Schon im Backlog, nicht neu erfinden:** CSP-BASELINE (M), DEPS-FLOAT (M, trägt filelock), MINTBOX-BAK (XS).

**Angreifermodell, das die Schweregrade trägt:** Ein Nutzer (Oli), eigene Daten — aber darunter **Transkripte von Arbeitsmeetings** (vertraulich). Erreichbar aus dem Internet: Login-Formular, `POST /api/auth/login`, jeder Token-Endpoint (fail-closed, aber Bearer-Geheimnis = einziger Zaun). Wer einen der sechs Env-Tokens oder das Passwort hält, ist drin. Wer im Worker Code ausführt (die Dokument-Engines lesen angreiferkontrollierte Dateien), ist mit dem Socket **root auf dem Host** mit ~30 Diensten. Das ist die Kette, gegen die bewertet wird: **ein Auth-Fehler entfernt von Host-Root** — nicht „Single-User, egal".

## Gesperrte Entscheidungen

1. **Phase 1 ist lesend.** Kein Fix, kein Config-Touch (auch nicht auf der Mintbox), keine Dependency-Änderung, solange der Befund nicht steht. Ein Fix mitten im Audit verändert den gemessenen Zustand und macht den Bericht unehrlich.
2. **Messen, nicht behaupten.** Jedes Finding, das messbar ist (Header, Statuscode, Dateimodus, uid, Log-Zeile), trägt Kommando **und** Ausgabe im Befund-Doc. Messungen gegen die deployte Instanz sind read-only: `curl` ohne Cookie oder mit Wegwerf-User, `docker exec … id`, `ls -l`, DB nur `mode=ro`. Wegwerf-User strikt nach `user_id` abräumen (`api_token` trägt Olis iOS-Tokens).
3. **Befund-Format = Notion-Vorlage wörtlich** (Anhang A): Risiko-Übersicht je OWASP-Kategorie · Findings-Tabelle mit **Exploit-Szenario in einem Satz**, C/H/M/L, Fix, Aufwand · Top-3-Risiken · Quick-Wins · Strukturelle Bewegungen · Defense-in-Depth-Lücken · Validierungs-Checkliste beantwortet. `VERIFY:`-Prefix für Unbestätigtes. Die Reihe (fünf Audits) soll vergleichbar bleiben.
4. **Eigene Sektion „Unter LAN-only entschieden"** mit Urteil je Punkt — *hält* / *hält nicht* / *hält mit Auflage* — und Begründung aus dem Angreifermodell. Mindestens: Docker-Socket am Worker · root in beiden Containern · Single-User-Env-Tokens inkl. `CARD_TOKEN` als generischer Schreib-Token · kein Login-Throttling · Traceback-Tail an den Client · `0.0.0.0:5656`. Was unter der neuen Prämisse nicht hält, wird **nicht re-litigiert, sondern mit einer Stufenleiter versehen** (z. B. Socket: rootless Docker / Socket-Proxy mit Allow-List / eigener Nutzer im Container — je mit Aufwand und was es kostet).
5. **Keine FUD.** Ein Finding ohne Exploit-Pfad aus dem Internet oder aus einem kompromittierten Token ist kein High. Schweregrade ehrlich verteilen. Das Wort „könnte" braucht ein Szenario.
6. **Ablageort der Reihe**: `docs/archive/audit-outputs/AUDIT_SECURITY_2026-09-25.md` (Verzeichnis neu anlegen; Konvention für alle fünf Audits). Ins BACKLOG wandern **nur** Top-3 + strukturelle Bewegungen als Items mit Code; Quick-Wins werden in Phase 2 gebaut, nicht gelistet.
7. **Phase 2 baut nur, was das Sign-off freigibt** — XS/S, jedes mit Test (`pytest` kann Header und `ProxyFix` über den Test-Client prüfen) **und** Messung an der Kante nach dem Deploy. **HSTS gehört an den TLS-Terminator** (nginx = Mintbox-Systemkonfig außerhalb des Repos, braucht sudo) → **Vorschlag als Config-Block im Bericht, kein Edit ohne Olis Go**; alles, was mit der App reist (`after_request`-Header, `ProxyFix`, Cookie-Flags, SRI-Pins, Compose-Bind), gehört in die App.
8. **CSP wird hier nicht gebaut.** CSP-BASELINE existiert; der Bericht benennt seine Voraussetzung (Play-CDN raus, Tailwind zur Build-Zeit) und schneidet das Item neu. Keine Dependency-Bumps (DEPS-FLOAT / eigener CVE-Sprint); SRI-Pins sind Template-only und erlaubt.
9. ⚠️ **Editiert wird nur auf dem Mac.** Mintbox = Runtime. **Keine Token, keine Hashes, keine Cookie-Werte im Befund-Doc** — Prefixe reichen (`scrypt:`), Werte redigieren.

---

# Phase 1 — Befund

## 1.1 Input zusammenstellen

Lies vollständig: [app_pkg/__init__.py](../../../app_pkg/__init__.py) (Config, Cookies, request_loader, CSRF-Inversion, unauthorized_handler, errorhandler, Migrationen, `create-user`), [app_pkg/auth.py](../../../app_pkg/auth.py), [app_pkg/mobile_auth.py](../../../app_pkg/mobile_auth.py), [app_pkg/ingest.py](../../../app_pkg/ingest.py) (`_bearer_token`, `_resolve_target_user`), [app_pkg/cards.py](../../../app_pkg/cards.py) (`_authorize_card_write`), [app_pkg/narration.py](../../../app_pkg/narration.py), [app_pkg/document_api.py](../../../app_pkg/document_api.py), [app_pkg/docwrite.py](../../../app_pkg/docwrite.py), [app_pkg/library.py](../../../app_pkg/library.py), [app_pkg/highlights.py](../../../app_pkg/highlights.py), [app_pkg/tags.py](../../../app_pkg/tags.py), [app_pkg/collections.py](../../../app_pkg/collections.py), [app_pkg/learn.py](../../../app_pkg/learn.py), [app_pkg/audio.py](../../../app_pkg/audio.py), [app_pkg/documents.py](../../../app_pkg/documents.py), [app_pkg/markdown.py](../../../app_pkg/markdown.py), [app_pkg/kindle.py](../../../app_pkg/kindle.py), [app_pkg/mermaid.py](../../../app_pkg/mermaid.py), [app_pkg/integrations/notion.py](../../../app_pkg/integrations/notion.py), [app_pkg/markdown_render.py](../../../app_pkg/markdown_render.py), [services/svg_sanitize.py](../../../services/svg_sanitize.py), [services/doc_media.py](../../../services/doc_media.py), [services/office_backends.py](../../../services/office_backends.py), [services/audio_chunker.py](../../../services/audio_chunker.py), [services/transcription_jobs.py](../../../services/transcription_jobs.py), [services/pdf_local.py](../../../services/pdf_local.py), [services/pdf_cloud.py](../../../services/pdf_cloud.py), [services/document_router.py](../../../services/document_router.py), [services/kindle_service.py](../../../services/kindle_service.py), [tasks.py](../../../tasks.py), [worker.py](../../../worker.py), [models.py](../../../models.py), [Dockerfile](../../../Dockerfile), [docker-compose.yml](../../../docker-compose.yml), [requirements.txt](../../../requirements.txt), [constraints.txt](../../../constraints.txt), [templates/base.html](../../../templates/base.html), [templates/login.html](../../../templates/login.html), alle `innerHTML`-Stellen in `static/js`, und per ssh **lesend** die nginx-Site-Config. Dazu die Kontrakte [docs/mobile_auth_contract.md](../../mobile_auth_contract.md), [docs/ingest_contract.md](../../ingest_contract.md), [docs/card_api_contract.md](../../card_api_contract.md), [docs/document_api_contract.md](../../document_api_contract.md) — was dort **versprochen** ist, ist der Maßstab für Object-Level-Authorization.

Erzeuge als Grundlage eine **Routen-Tabelle** (75 Routen, 63 unter `/api`): Methode · Pfad · Auth-Mechanismus (Session / Bearer / welcher Env-Token / öffentlich) · CSRF (Inversion / exempt) · Owner-Check (welche Zeile) · Anmerkung. Die Tabelle ist die A01-Prüfung — **jede** Route, kein Sampling. Sie gehört als Anhang ins Befund-Doc.

## 1.2 Den Katalog abarbeiten

Alle elf Kategorien aus Anhang A, in dieser Reihenfolge, je Kategorie: was geprüft, was gefunden, was messbar war und gemessen wurde. Wo eine Kategorie leer ist, steht „geprüft, kein Finding" mit einem Satz, **was** geprüft wurde — eine leere Zeile ist kein Beleg.

## 1.3 VERIFY-Liste (Master-vorab; jedes Item mit Beleg auflösen, nicht mit Meinung)

- **SSRF über Playwright-PDF**: Wegwerf-User, Dokument mit `<img src="http://127.0.0.1:<port>/probe.png">` und einem zweiten Ziel im Docker-Netz (`http://redis:6379/`), im Web-Container ein `python3 -m http.server <port>` als Zeuge → PDF erzeugen → Zugriffs-Log zeigen. Dasselbe für EPUB (ebooklib holt vermutlich nichts — belegen) und für den Reader (dort lädt der **Browser**, nicht der Server — belegen). Wer kann Dokumente anlegen: Session, `INGEST_TOKEN`, `CARD_TOKEN`-Docwrite, `DOC_CONVERT_TOKEN` (Konvertierung schreibt Markdown). Das Ergebnis bestimmt, ob RICH-MEDIA-https-Bilder im PDF ein Finding sind.
- **unstructured CVE-2026-71428**: jeden `partition*`-Aufruf listen; erreicht irgendeiner `url=`? Holt `partition_html`/`partition_email` Remote-Ressourcen aus dem Dokument nach? Am Code **und** mit einer EML/HTML-Probe mit `<img src="http://127.0.0.1:<port>/…">` gegen denselben Zeugen belegen.
- **`remote_addr` hinter nginx**: eine fehlgeschlagene Mobile-Login-Anfrage von außen → Log-Zeile im Container zeigen (`127.0.0.1` erwartet). Konsequenz für Throttling und Logging benennen.
- **`0.0.0.0:5656`**: Wer spricht `:5656` direkt an? `docker inspect converter-mcp-server` (Env, read-only) und die iOS-App-Basis-URL — wenn beide über nginx gehen, ist `127.0.0.1:5656:5000` im Compose ein XS-Quick-Win; wenn nicht, ist es eine Entscheidung. Ob der Router 5656 forwarded: **VERIFY-Oli** im Bericht.
- **Dateimodi der Geheimnisse** auf der Mintbox: `getfacl .env google-credentials.json` (die `+`-ACL lesen), welche Nutzer/Gruppen existieren auf dem Host, welche Container Home-Verzeichnisse mounten. Dazu `app_data_bak-*` (MINTBOX-BAK): root-owned, world-readable, DB-Kopien.
- **Passwort-Hash-Prefix** in der Prod-DB (`mode=ro`, nur der Teil vor dem ersten `$`).
- **Remember-Cookie**: Laufzeit (Default 365 Tage) und ob Logout die Session-Cookies clientseitig-nur invalidiert (Flask-Eigenschaft, kein Bug — aber benennen, was ein gestohlenes Cookie wert ist und wie lange).
- **`learn.py:124` `text()`-Statement**: Parameter gebunden? `tags.py`-Core-Statements: nichts per f-String?
- **`documents.py:63` rohe Endung**: wohin fließt sie (Dateiname auf Platte? Router-Dispatch?) und was passiert mit `../`, Nullbytes, `.PDF`?
- **Die fünf `subprocess.run`**: Listen-Args ohne `shell=True`? Welche Argumente stammen aus Nutzereingaben (Dateinamen!)? `docker run` in `pdf_local.py:236` mit `-v {out_dir_host}:/out` — woher kommt `out_dir_host`?
- **`innerHTML` ×39**: Tabelle Datei:Zeile · Quelle des Strings (statisch / Server-JSON / Nutzertext / Agent-Text) · Sanitizing davor · Urteil. Die zwei erlaubten Senken benennen, den Rest klassifizieren.
- **`Markup`-Filter `__init__.py:434`** und die `window.PageData`-Inline-Blöcke: welche Werte reisen dort hinein, ist `</script`-Escaping die einzige nötige Kante (auch `<!--`, U+2028)?
- **`api_send_to_notion(conversion_id)`**: Owner-Check? Was reist an den MCP-Server, über welches Netz (Host-Docker-Netz, Klartext, Token als Bearer)?
- **Fehlerantworten**: was liefert ein unbehandelter 500 an den Client (gunicorn, `debug` aus)? Der gespeicherte **Traceback-Tail** (Transkription/Dokument/Narration) landet als letzte Zeile in der UI und komplett in `metadata_json` — enthält er Pfade, Env, Token? Einmal provozieren und zeigen.
- **Logging**: `grep -rn 'logger\.' app_pkg services tasks.py` gegen Token/Authorization/Passwort/Cookie — was wird geloggt, was nie.
- **CSRF-Inversion**: die elf Exempts gegen die Liste der token-authed Views — ist jede exempte View token-gated, und gibt es token-gated Views, die **nicht** exempt sind und darum vom Agenten nur mit Bearer erreichbar sind (DOC-API ist bewusst so — steht im Kontrakt; prüfen, ob es sonst noch eine gibt)?
- **Anti-Enumeration am Web-Login** (`auth.py`): Dummy-Hash wie am Mobile-Login, oder unterscheidet die Antwortzeit „User existiert nicht" von „falsches Passwort"? Messen (20 Anfragen je Fall, Median), nicht schätzen.
- **Supply Chain**: Tailwind-Play-CDN auf jeder Seite inklusive Login: Exploit-Szenario (Kompromittierung des CDN = Skript in jeder Seite, Session-Cookie ist `HttpOnly`, aber jeder Request läuft im Nutzerkontext) und der Weg raus (Tailwind CLI zur Build-Zeit im Dockerfile, statisches CSS, dann SRI-frei). `markdown-it`/`mermaid@10` ohne SRI: Hash berechnen und pinnen ist Phase-2-Kandidat.
- **CVE-Tabelle** oben: je Zeile Erreichbarkeit belegen (Aufrufer-Grep), dann Urteil. Reproduktion: `docker exec markdown-converter-web pip freeze` → PyPI-JSON je Paket (`/pypi/<name>/<version>/json` → `vulnerabilities`); `pip-audit -r` mit `+cpu` und dem filelock-Konflikt ist der dokumentierte Fehlweg.

## 1.4 Bericht

Das Befund-Doc unter `docs/archive/audit-outputs/AUDIT_SECURITY_2026-09-25.md` im Format aus Anhang A, plus Sektion „Unter LAN-only entschieden" (Entscheidung 4), Routen-Tabelle als Anhang, VERIFY-Oli-Liste (was nur Oli prüfen/entscheiden kann: Router-Forwarding, nginx-Header, sudo-Modi), beantwortete Validierungs-Checkliste. **Commit + Push** (`docs(SEC-AUDIT): Befund — …`).

## Stop
Bericht an den Master: Top-3 mit Exploit-Pfad in je einem Satz · die Quick-Win-Liste **mit Ort, Aufwand und geplanter Messung** je Item · die strukturellen Bewegungen mit Stufenleiter · was nicht messbar war und warum. Dann warten. Der Master wählt aus der Quick-Win-Liste für Phase 2 aus.

---

# Phase 2 — Quick-Wins (nur die im Sign-off freigegebenen)

Kandidaten, die der Master **vorab** sieht — der Bericht kann sie bestätigen, streichen oder ergänzen; gebaut wird nur Freigegebenes:

- **App-seitige Header** per `after_request` in `app_pkg/__init__.py`: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` (der Vorschau-iframe ist `srcdoc`, kein Cross-Origin-Frame — belegen, dass er weiter geht), `Referrer-Policy: strict-origin-when-cross-origin`, minimale `Permissions-Policy`. Test im Test-Client, Messung per `curl -I` an der Kante.
- **`ProxyFix(x_for=1, x_proto=1, x_host=1, x_port=1)`** um die WSGI-App — danach ist `remote_addr` die Client-Adresse und `is_secure` wahr. ⚠️ Nur mit genau **einem** vertrauten Proxy-Hop; lokal ohne Proxy bleibt es harmlos (keine Header → keine Änderung). Test: Header-Injektion im Test-Client.
- **Cookie-`Secure`-Flags** env-gegated (`SESSION_COOKIE_SECURE`/`REMEMBER_COOKIE_SECURE`), auf der Mintbox an, lokal ohne TLS aus — sonst ist der Mac-Dev-Login tot. Dazu `REMEMBER_COOKIE_DURATION` auf einen begründeten Wert, falls der Bericht 365 Tage als Finding führt.
- **SRI + Versions-Pin** für `markdown-it@14.1.0` und `mermaid` auf `mermaid_converter.html` (dieselbe `10.9.8` + derselbe Hash wie im Reader — RICH-MEDIA hat ihn nachgerechnet).
- **Compose-Bind `127.0.0.1:5656:5000`**, falls 1.3 belegt, dass niemand `:5656` direkt braucht.
- **Dateimodi** auf der Mintbox (`chmod 600 .env google-credentials.json`, ACL prüfen) — Runtime-Hygiene, kein Repo-Edit; erlaubt, wenn `getfacl` keine Überraschung zeigt; im Bericht vorher/nachher.
- **nginx**: `add_header Strict-Transport-Security` (mit Bedacht: `max-age` klein anfangen, kein `preload`) und `limit_req` für `/login` + `/api/auth/login` — **Vorschlag als Block**, Oli setzt ihn (sudo).

Je Item: eigener Commit, `pytest tests/` grün (Baseline **1129 + 1 Skip**), Deploy (`git pull` + `docker compose up -d --build` auf der Mintbox), **Messung an der Kante** im Bericht (Header vorher/nachher, Log-Zeile mit echter Client-IP). Kein Item ohne Messung.

## Stop
Bericht: je Item Commit, Test, Kanten-Messung. Dann warten.

---

# Phase 3 — Wrap

- **CLAUDE.md**: im Kopf-Absatz („Nicht LAN-only") den Satz ersetzen, der die Neubewertung fordert — sie ist jetzt gemacht: ein Satz mit Verweis auf das Befund-Doc, was hält, was gebaut wurde, was als Item offen ist. Kein zweiter Sicherheits-Roman; das Doc ist der Ort.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): SEC-AUDIT schließen; je strukturelle Bewegung **ein** Item mit Code, Größe, Stufenleiter (Kandidaten: Socket-Härtung, Non-Root-Container, Login-Throttling, Play-CDN-Ablösung); CSP-BASELINE mit der Voraussetzung neu schneiden; MINTBOX-BAK und DEPS-FLOAT um die Befunde ergänzen statt zu duplizieren. Die vier Folge-Audits der Reihe (ARCH → CONSIST → TEST → DOC) stehen als benannte Reihe im Item SEC-AUDIT — nicht neu eröffnen.
- **Brief ans converter-mcp** nur, wenn sich an der Agent-Fläche etwas geändert hat (Token-Modell, Endpoint-Verhalten); sonst im Wrap sagen, warum nicht (Hausregel seit 2026-09-19; **vorher** die Tool-Liste am lebenden Connector lesen).
- **Memory** nur bei übertragbarer Lehre. Kandidat: *hinter einem Reverse-Proxy ohne `ProxyFix` ist `remote_addr` immer die Proxy-Adresse — jedes Log und jeder Rate-Limit-Schlüssel darauf ist wertlos; erst messen, dann throttlen.* Prüfen, ob das schon irgendwo liegt; sonst `reference_*`.
- **Im Bericht benennen**: die Top-3 mit ihrem Status (gebaut / Item / VERIFY-Oli) · welche LAN-only-Entscheidungen gekippt sind · die nginx-Blöcke für Oli · dass der Wegwerf-User samt Daten nach `user_id` entfernt ist · die Mintbox ohne unversionierte Dateien.

## Nicht-Ziele

- **Keine CSP** (CSP-BASELINE), **kein** Play-CDN-Umbau (eigenes Item), **keine** Dependency-Bumps (DEPS-FLOAT / CVE-Sprint), **kein** Rate-Limit mit neuer Dependency, **kein** rootless Docker, **keine** Auth-Neuarchitektur.
- **Kein** nginx-Edit ohne Olis Go. **Kein** Mintbox-Touch in Phase 1.
- **Keine** Wiederholung der RICH-MEDIA-Angriffs-Batterien am Sanitizer — die Ränder (VERIFY-Liste) reichen.
- **Keine** Token, Hashes oder Cookie-Werte im Repo.

---

## Anhang A — Der Prüfkatalog (Notion „Audit — Security", MINTBOX, 2026-05-16, wörtlich)

Quelle: https://app.notion.com/p/Audit-Security-3623f5db30d28107b46fd6c176a9e727

```
KONTEXT
- Produkt: CONVERTER — Flask-Multimedia-Konverter (Markdown↔PDF, Dokument→Markdown, Transkription, Narration, Lernkarten, Reader), Single-User
- Daten-Sensitivität: vertraulich (Meeting-Transkripte, eigene Notizen)
- Auth-Modell: Session-Cookie (Flask-Login) + per-User-Bearer + sechs Env-Tokens; CSRF per Inversion
- Compliance-Anforderungen: keine formalen; DSGVO-relevant, weil Dritte in Transkripten vorkommen
- Stack: Flask 3.1.3 / SQLAlchemy+SQLite (WAL) / Flask-Login 0.6.3 / Flask-WTF 1.2.1 / gunicorn+uvicorn / Redis+RQ / Playwright-Chromium / nh3 0.2.18 / Docker (root) hinter Host-nginx

AUFGABE
Prüfe systematisch gegen die OWASP-Top-10 (2021) plus Stack-spezifische Hot-Spots:

1. BROKEN ACCESS CONTROL (OWASP A01)
   - Permission-Check pro Route systematisch (kein Vergessen)
   - Object-Level-Authorization (User darf nur eigene Daten lesen/ändern)
   - Tenant-Isolation
   - Direct Object References ohne Owner-Check
   - Admin-Endpoints klar abgegrenzt

2. CRYPTOGRAPHIC FAILURES (OWASP A02)
   - Secrets in Code / Repo / Logs / URLs
   - Passwort-Hashing-Algorithmus (bcrypt/argon2/scrypt vs md5/sha1)
   - TLS-Erzwingung
   - Token-Expiry und Rotation
   - Sensitive Daten in Cookies ohne HttpOnly+Secure+SameSite

3. INJECTION (OWASP A03)
   - SQL-Injection: Raw-SQL mit String-Concat vs Parametrized
   - Command-Injection: subprocess/exec mit User-Input
   - SSTI (Server-Side-Template-Injection)
   - XSS in Responses (HTML-Encoding fehlt)

4. INSECURE DESIGN (OWASP A04)
   - Fehlende Rate-Limits auf Login / sensitive Endpoints
   - Predictable Resource-IDs (sequentielle IDs erlauben Enumeration)
   - Fehlende Anti-CSRF bei State-changing Operations (wenn Cookie-Auth)
   - Business-Logic-Bypässe

5. SECURITY MISCONFIGURATION (OWASP A05)
   - Debug-Mode in Production
   - Default-Credentials
   - Übermäßig offene CORS-Policy
   - Stack-Traces in Error-Responses an Client
   - Unnötige HTTP-Header
   - Security-Header fehlend (CSP, HSTS, X-Frame-Options, X-Content-Type-Options)

6. VULNERABLE COMPONENTS (OWASP A06)
   - Outdated Dependencies mit bekannten CVEs
   - Verlassene/unmaintained Libraries an kritischen Stellen

7. AUTHENTICATION FAILURES (OWASP A07)
   - Brute-Force-Schutz fehlt
   - Session-Fixation möglich
   - Logout invalidiert Token wirklich
   - Passwort-Reset-Token mit zu langer Gültigkeit
   - MFA-Optional vs Required für sensitive Accounts

8. DATA INTEGRITY FAILURES (OWASP A08)
   - Deserialization unsicherer Formate (pickle, java serialization)
   - Unsignierte Updates / Code-Loads aus External Sources
   - Webhook-Signatur-Validierung

9. LOGGING & MONITORING FAILURES (OWASP A09)
   - Security-relevante Events nicht geloggt
   - Sensitive Daten in Logs (Passwörter, Tokens, PII)
   - Audit-Trail-Lücken

10. SSRF (OWASP A10)
    - User-Input fließt in interne HTTP-Calls
    - Fehlende Allow-Listen für externe Targets

11. STACK-SPEZIFISCHE HOT-SPOTS
    - File-Upload: Type-Check (MIME nicht trauen), Größen-Limit, Storage-Pfad-Traversal
    - Background-Job-Auth-Context (Jobs laufen als wer?)
    - Mass-Assignment (ORM nimmt alle Felder aus Request entgegen)

OUTPUT-FORMAT

### Risiko-Übersicht
| OWASP-Kategorie | Funde | Höchster Schweregrad | Quick-Win verfügbar |
|---|---|---|---|

### Findings
| # | OWASP # | Fundstelle | Vulnerability | Exploit-Szenario | Schweregrad C/H/M/L | Fix | Aufwand |
|---|---|---|---|---|---|---|---|

Schweregrad: Critical (Datendiebstahl/RCE) / High / Medium / Low. Pro Finding ein konkretes Exploit-Szenario in 1 Satz.

### Top-3-Risiken
Die drei dringendsten Findings mit Begründung warum sie zuerst kommen.

### Quick-Wins (XS/S, hohe Wirkung)
z.B. Security-Header ergänzen, Rate-Limit aktivieren, Permission-Check ergänzen.

### Strukturelle Bewegungen
Größere Themen: Auth-Refactor, Secret-Management-Einführung.

### Defense-in-Depth-Lücken
Stellen wo eine Schicht fehlt obwohl andere greifen.

REGELN
- Keine FUD (Fear-Uncertainty-Doubt) ohne konkretes Exploit-Szenario.
- Schweregrad ehrlich: nicht alles ist Critical.
- Bei vermutetem Issue ohne klare Bestätigung im Code: "VERIFY:" Prefix.
- Compliance-relevante Findings (DSGVO) klar markieren.
- Keine Empfehlung "verwendet Library X" ohne Begründung warum die vorhandene unzureichend ist.

VALIDIERUNGS-CHECKLISTE (am Ende des Befund-Docs beantworten)
- [ ] Hat jedes Finding ein konkretes Exploit-Szenario?
- [ ] Wurden Top-3-Risiken explizit ausgewiesen?
- [ ] Schweregrad-Verteilung plausibel (nicht alles Critical, nicht alles Low)?
- [ ] Wurden Defense-in-Depth-Lücken separat ausgewiesen?
- [ ] Sind Compliance-relevante Findings markiert wenn relevant?
- [ ] Wurde VERIFY:-Prefix korrekt verwendet bei unklarer Bestätigung?
```

Die Notion-Vorlage nennt zusätzlich RLS/Postgres/JWT-Punkte — für SQLite ohne Rollen und ohne JWT **nicht anwendbar**, im Befund als solche markieren, nicht stillschweigend weglassen.
