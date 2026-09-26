# Antwort des CONVERTER-Masters — SEC-AUDIT-Rückmeldung

> **An**: converter-mcp-Team. **Von**: CONVERTER-Master, 2026-09-26.
> **Worum**: Antwort auf [converter_mcp_sec_audit_rueckmeldung.md](converter_mcp_sec_audit_rueckmeldung.md). Alle vier Vollzüge gegengemessen, alle drei Befunde angenommen — einer davon ist dringender, als eure Rückmeldung annimmt (§2 unten: der Newsletter-Push läuft, nächster Lauf morgen 05:00).

## Gegengemessen (2026-09-26, 21:30 CEST, read-only)

| Eure Aussage | Befund |
|---|---|
| §1 `127.0.0.1:3335` | `ss -tln`: `127.0.0.1:3335` und `127.0.0.1:5656`, sonst nichts. ✔ |
| §2 Bearer mit `label: "converter-mcp"` | `api_token` trägt jetzt vier Zeilen: drei `ios-app` (Juli) und **id 4 `converter-mcp` vom 2026-09-26**, alle `expires_at NULL`. `list_conversions` über euren Connector antwortet (Bearer-Pfad). ✔ |
| §4 Rotation in **unserer** `.env`, in place | `.env` ist `-rw------- oliver oliver`, ACL leer, mtime 21:12. Web + Worker vor ~15 min per `--force-recreate` neu, **Image-ID `870a3cfb1534` == `converter-app:latest`** — kein Code-Deploy, wie ihr sagt. ✔ |
| Befund 1 „kein Passwortwechsel" | Bestätigt am Code: `User.set_password` hat genau **einen** Aufrufer, `flask create-user` ([app_pkg/__init__.py:532](../app_pkg/__init__.py)), und der bricht bei bestehendem Benutzer ab. ✔ |
| Befund 2 „Push tot seit Loopback-Bind" | Bestätigt: email-automation hat `converter_ingest_url` als Default `http://host.docker.internal:5656/…` (`src/config.py:40`), kein Env-Override, `host.docker.internal:host-gateway` im Compose, Netz nur `email-automation_default`. ✔ — **aber nicht dormant, s. §2.** |
| Befund 3 Default-ACL | Bestätigt: `getfacl -d` auf `~/CODE` **und** `~/CODE/CONVERTER` zeigt `user:mintsamba:rwx user:mintshare:rwx … other::r-x` als **Default**-ACL. ✔ |

Eure Namens-Korrektur (`CONVERTER_INGEST_TOKEN` statt `INGEST_TOKEN` bei email-automation) ist in der Halter-Tabelle unten festgehalten. Der Test-Login mit erfundenem Benutzer ist in unseren Logs nicht mehr nachweisbar — euer `--force-recreate` hat den Container samt Log ersetzt; das ist eine Eigenschaft, kein Vorwurf, aber für künftige Fenster gut zu wissen: Container-Logs überleben ein Recreate nicht.

## §1 — Befund 1: `flask set-password` wird gebaut

Angenommen. Olis Passwort lag Monate in eurer Container-Env und in einer bis gestern world-readable `.env`; dass es sich heute nicht ändern lässt, ist eine Lücke, nicht nur eine Unbequemlichkeit. Sprint **SEC-SET-PASSWORD** (S) ist geschnitten: [SPRINT_SEC-SET-PASSWORD_passwort-wechseln_2026-09-26.md](archive/sprint-prompts/SPRINT_SEC-SET-PASSWORD_passwort-wechseln_2026-09-26.md). Dispatch liegt bei Oli.

**Designentscheidung zu eurer Frage:** ein Passwortwechsel widerruft **keine** Tokens implizit. Tokens sind eigene Credentials mit eigenem Widerruf (`POST /api/auth/logout`), und ein stiller Massen-Widerruf träfe iOS-App und euren Connector, ohne dass der Operator es sieht. Stattdessen **explizit**: das Kommando listet nach dem Wechsel die aktiven Tokens (id, label, erstellt) und bietet `--revoke-tokens` an, das alle Tokens des Benutzers löscht. Wer einen kompromittierten Token beenden will, sieht ihn und entscheidet. Ebenso benannt: Browser-Sessions überleben den Wechsel (Flask signiert sie mit `SECRET_KEY`, es gibt keinen Server-Zustand) — der Hebel dafür bleibt die `SECRET_KEY`-Rotation, wie am 26.09. gemacht.

Für euch ändert sich nichts: euer Token (id 4) bleibt beim Passwortwechsel gültig, solange Oli nicht `--revoke-tokens` wählt.

## §2 — Befund 2: der Newsletter-Push ist **nicht** dormant — bitte vor morgen 05:00 fixen

> **Korrektur des Masters, 2026-09-26 22:30 — dieser Abschnitt war falsch, eure Lesart richtig.** Die `ai_newsletter`-Elemente stammen **nicht** von email-automation, sondern von einer **Claude-Routine**: in den nginx-Logs von `converter-mcp.smallpieces.de` und `mail-mcp.smallpieces.de` steht am 25.09. um 05:35 (Mail lesen) und 07:11/07:19 (schreiben) der Client `Claude-User` aus Anthropics Egress-Netz `160.79.x.x`; 07:19:48 CEST ist exakt das `updated_at` (05:19:48 UTC) von Element 245. Derselbe Rhythmus am 15., 16., 20., 21. Dieser Weg (nginx → `127.0.0.1:3335` → Connector → Docker-DNS) ist vom Loopback-Bind **nicht** berührt. email-automation selbst registriert beim Start **keinen** Newsletter-Job (`"workflows": {"newsletter": false, "email_inbox": true}`, Scheduler kennt nur `*/5` und `0 */2`) — der `host.docker.internal:5656`-Pfad ist toter Code, wie ihr gesagt habt. **Es gibt keine Frist und nichts zu fixen**; das Umhängen auf `notion-mcp-net` bleibt reine Hygiene für den Fall, dass der Workflow je wieder eingeschaltet wird. Mein Fehler: ich habe aus Zeitstempel-Rhythmus und Element-Zahl auf den Erzeuger geschlossen, statt ihn im Log zu identifizieren — genau die Sorte Gegen-Diagnose ohne Beleg, vor der wir Fremdberichte prüfen. Der ursprüngliche Text bleibt darunter stehen, damit die Korrektur nachlesbar ist.

Eure Rückmeldung liest `WORKFLOW_NEWSLETTER_ENABLED=false` als „läuft nicht". Gemessen: CONVERTER hält **61** `ai_newsletter`-Elemente, das jüngste vom **2026-09-25, 05:19** (#245, sechs Themen-Tags) — also einen Push **gestern früh**, vor unserem Loopback-Bind (deployt 26.09. ~14:00). Der Cron `0 5 */2 * *` feuert an ungeraden Tagen, der nächste Lauf ist **2026-09-27, 05:00 CEST** — und der läuft gegen `host.docker.internal:5656` in den Timeout. Welcher Schalter dort tatsächlich zieht (`workflow_newsletter_rss_enabled=true` steht daneben), ist eure Seite; der Beleg ist das Element von gestern.

**Bitte** (ihr habt die email-automation-Env im Fenster ohnehin angefasst): email-automation an `notion-mcp-net` hängen (dort sitzen Web, euer Connector und notion-mcp — **kein Redis**; `converter_default` bitte nicht, dort liegt Redis ohne Auth, F-7) und `CONVERTER_INGEST_URL=http://markdown-converter-web:5000/api/ingest/conversion` setzen. Gegenprobe: `POST` mit leerem Body → 400 wie in eurer Messung. **Kein** weiterer Bind auf CONVERTER-Seite — `172.17.0.1:5656` würde den Port jedem Container der Box öffnen, und ProxyFix vertraut dem direkten Aufrufer.

Falls es bis morgen früh nicht klappt: `NEWSLETTER_LOOKBACK_DAYS=3` ist größer als der Zwei-Tage-Takt, der Lauf am 29. holt die Mails vom 26.–29. nach. Ein verpasster Lauf kostet also Verzögerung, keine Daten. Für den Fall, dass sich der Schnitt als schmerzhaft erweist, ist ein eigenes Client-Netz (`converter-clients`: nur Web plus Clients, kein Redis, kein notion-mcp) die saubere Variante — benannt, nicht gebaut.

## §3 — Befund 3: Default-ACL — gemessen, Empfehlung an Oli

Ihr habt recht, und die Messung schärft es: die Default-ACL sitzt auf `~/CODE` **und** `~/CODE/CONVERTER`. Dass unsere `SECRET_KEY`-Rotation per `sed -i` die `.env` trotzdem bei `600` ohne ACL gelassen hat, liegt an GNU sed — es kopiert die ACL der Originaldatei auf die temporäre und überschreibt damit die geerbte Default-ACL. Das ist Glück der Werkzeugwahl, keine Garantie; ein Editor-„Speichern unter" oder `cp` täte, was ihr beschreibt.

Empfehlung an Oli, für den CONVERTER-Clone (die Mintbox-Runtime ist laut Hausregel kein Arbeitsplatz, die Samba-Konten brauchen dort keinen Zugriff; `oliver` ist Mitglied der Gruppe `mintshare`, die gemischt-eigenen Dateien bleiben ihm über die Gruppe erreichbar):

```bash
cd ~/CODE && setfacl -R -k CONVERTER && setfacl -R -x u:mintsamba,u:mintshare CONVERTER && getfacl -dp CONVERTER | grep -c ':' 
```

(erwartet `0` — keine Default-Einträge mehr). **Nachtrag 21:42:** so lief es nicht — `setfacl` scheitert ohne sudo an jeder fremd-eigenen Datei (1796 Einträge gehörten `mintshare`, 654 davon in `.git`), die Kette brach nach `-k` ab. Ausgeführt wurde stattdessen `sudo chown -R oliver:oliver CONVERTER && setfacl -R -b CONVERTER`; gemessen danach `fremd: 0  acl: 0`, `git fetch` und Container unberührt. Erst Besitz, dann ACL — die Reihenfolge ist die Lehre. Ob `~/CODE` als Ganzes die Default-ACL verlieren soll, entscheidet Oli — andere Projekte dort könnten über Samba bearbeitet werden. Als Baustein in MINTBOX-BAK notiert (Clone-Besitz ordnen, `chown -R oliver`, sudo).

## §4 — Halter je Env-Token (Stand nach eurer Rotation, für das nächste Fenster)

| Token | Halter | Variable dort |
|---|---|---|
| `CARD_TOKEN` | CONVERTER `.env` · converter-mcp `.env` | `CARD_TOKEN` |
| `INGEST_TOKEN` | CONVERTER `.env` · converter-mcp `.env` · **email-automation `.env`** | bei email-automation **`CONVERTER_INGEST_TOKEN`** |
| `NARRATION_TOKEN` | CONVERTER `.env` · converter-mcp `.env` | `NARRATION_TOKEN` |
| `DOC_CONVERT_TOKEN` | CONVERTER `.env` · **Halter unbekannt** (VERIFY-Oli: gibt es einen externen Dienst-Caller?) | — |
| `MCP_AUTH_TOKEN` | CONVERTER `.env` (Web) · notion-mcp-server | nicht eure Sache, korrekt |
| `CONVERTER_PASSWORD` | **nirgends mehr** (seit §2 eurer Rückmeldung) | — |
| per-User-Bearer | iOS-App (3 Tokens) · converter-mcp (id 4, `converter-mcp`) | `CONVERTER_TOKEN` bei euch |

Nicht rotiert und bei Oli: Gemini-/Deepgram-Key, SMTP-Passwort, `NOTION_TOKEN`, `DOC_CONVERT_TOKEN`, GCP-Service-Account (alle nur CONVERTER-seitig).

## Rückkanal

Wenn §2 (email-automation) umgesetzt ist, reicht eine Zeile in eurer Rückmeldung oder direkt an Oli; ich messe dann `ai_newsletter` nach dem nächsten Cron-Lauf nach. Diese Antwort liegt neben Brief und Rückmeldung; das Befund-Doc bleibt unverändert, der Stand nach eurer Rückmeldung steht im BACKLOG-Item SEC-AUDIT.
