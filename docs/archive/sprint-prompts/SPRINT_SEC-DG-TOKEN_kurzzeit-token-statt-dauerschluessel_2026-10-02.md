# SPRINT SEC-DG-TOKEN — Kurzzeit-Token für die Live-Transkription statt des Deepgram-Dauer-Schlüssels im Browser

**Größe**: S (3 Phasen: Bau · Deploy + Live-Beleg · Wrap) · **Datum**: 2026-10-02 · **Herkunft**: Nebenbefund aus [ARCH-AUDIT](../audit-outputs/AUDIT_ARCHITEKTUR_2026-10-01.md) (Sektion „Nebenbefund außerhalb des Audit-Rahmens"), BACKLOG-Item SEC-DG-TOKEN; von SEC-AUDIT nur in der Routen-Tabelle geführt, nie bewertet

## Warum

`GET /api/get-deepgram-token` ([app_pkg/audio.py](../../../app_pkg/audio.py) Z. 273–282, `@login_required` + `require_service('deepgram')`) ruft `create_temporary_key(ttl_seconds=60)` — und das gibt `self.api_key` zurück ([services/deepgram_service.py](../../../services/deepgram_service.py) Z. 293–300, `ttl_seconds` wird nicht gelesen). Der Browser bekommt für die Live-Transkription also den **Dauer-Schlüssel**. Der Docstring begründet es mit „LAN-only and login-protected"; die Prämisse ist seit 2026-08-21 falsch (die Instanz ist aus dem Internet erreichbar).

Was der Schlüssel wert ist, ist gemessen (unten): er trägt den Scope `account:write` und darf die Projekt-Verwaltung lesen. Jedes XSS und jede gestohlene Session bekäme damit einen nicht ablaufenden Schlüssel mit Konto-Rechten, und er liegt in DevTools und Speicher jedes Geräts, das die Live-Transkription öffnet. Der Schlüssel wurde am 2026-10-01 frisch rotiert — und geht seitdem bei jeder Live-Aufnahme wieder hinaus.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-10-02 aus dem Worker-Container; kein Wert hat den Container verlassen)

- **Deepgrams Token-Grant geht mit dem aktuellen Schlüssel:** `POST https://api.deepgram.com/v1/auth/grant` mit `Authorization: Token <key>` und `{"ttl_seconds": 60}` → **HTTP 200**, Felder `access_token` (JWT, drei Teile, 485 Zeichen) und `expires_in: 60`.
- **Das Kurzzeit-Token ist eng:** mit `Authorization: Bearer <jwt>` → `/v1/projects` **403**, `/v1/auth/grant` (sich selbst verlängern) **403**.
- **Der Dauer-Schlüssel ist weit:** `GET /v1/auth/token` → Scope **`account:write`**; `/v1/projects` → 200 (bei der Rotation am 2026-10-01 gemessen).
- **SDK am Pin** `deepgram-sdk==7.1.0`: `client.auth.v1.tokens.grant` existiert.
- **Einziger Konsument** ist [static/js/audio_converter.js](../../../static/js/audio_converter.js): `getDeepgramToken()` (Z. 170–178) liest `data.deepgram_token`, `new WebSocket(deepgramUrl, ['token', deepgramToken])` (Z. 259). Die iOS-App nutzt weder den Endpunkt noch Deepgram direkt (im Clone nachgesehen), der converter-mcp auch nicht.
- **Reihenfolge im Browser heute:** erst Token holen (Z. 199), **dann** `getUserMedia` (Z. 213) mit dem Berechtigungs-Dialog des Browsers, dann der WebSocket. Kein automatischer Wiederaufbau: `onclose` beendet die Aufnahme, eine neue Aufnahme holt ein neues Token.
- **Tests:** keiner trifft die View oder `create_temporary_key`. [scripts/smoke_audio_converter.py](../../../scripts/smoke_audio_converter.py) deckt nur die Datei-Transkription, nicht das Mikrofon.
- **Baseline:** 1401 passed + 1 skipped, Mac und Pin. Prod-Image `67a4ccf3dfd7`, Rollback-Tag `pre-arch-narr5`.

## Gesperrte Entscheidungen

1. **Echte Kurzzeit-Tokens, fail-closed.** Die View liefert ausschließlich ein per Grant erzeugtes Token. Scheitert der Grant (Netz, 4xx/5xx, Zeitüberschreitung), antwortet sie mit einem Fehler (**502**, deutscher Satz) — **nie** mit dem Dauer-Schlüssel, in keinem Zweig. Kein Fallback, kein Flag.
2. **Die Lebensdauer setzt der Server**, nicht der Client: eine Konstante neben den anderen Deepgram-Werten, **30 s** (Deepgrams eigener Default; der Browser holt das Token unmittelbar vor dem Verbinden). Der Grant bekommt eine **eigene Deadline je Aufruf** (rund 10 s) — er läuft in einem Web-Thread.
3. **Über das SDK** (`client.auth.v1.tokens.grant`), gemockt an der SDK-Grenze wie die übrigen Deepgram-Tests. Nur wenn der SDK-Weg am Pin nicht trägt, der rohe HTTPS-Aufruf im Dienst (`requests` ist dort schon importiert) — dann im Bericht sagen, warum.
4. **Antwortform:** `deepgram_token` bleibt als Feldname (einziger Konsument ist unser JS), neu dazu `expires_in`. Der Methodenname im Dienst darf nicht länger lügen — umbenennen oder Docstring und Verhalten in Einklang bringen; die „LAN-only"-Begründung fällt.
5. **Browser:** der WebSocket nimmt das Token als `['bearer', token]` (für JWTs; `'token'` ist die Form für API-Schlüssel). **Reihenfolge umdrehen: erst das Mikrofon, dann das Token, dann der Socket** — sonst läuft das Token ab, während der Nutzer den Berechtigungs-Dialog liest. Die bestehende Fehler-Microcopy bleibt; wer das Mikrofon verweigert, löst keinen Grant aus.
6. **Was nicht belegt ist und der Live-Beleg klären muss:** ob eine aufgebaute Verbindung über das Token-Ende hinaus weiterläuft (erwartet: ja, geprüft wird beim Verbindungsaufbau). Die Probe-Aufnahme läuft deshalb **länger als die TTL**. Bricht die Verbindung am Token-Ende ab: stoppen und berichten, nicht die TTL hochdrehen.
7. **Sentinel:** in keinem Antwortzweig der View steht der konfigurierte Schlüssel (mit einem bekannten Fake-Schlüssel geprüft, Erfolg **und** Fehler); der Dienst hat keine Methode mehr, die `self.api_key` nach außen reicht.
8. **Nicht in diesem Sprint:** den Audio-Strom serverseitig proxyen, CSP, der Datei-Transkriptions-Pfad, ein neuer Schlüssel bei Deepgram, das Worker-Logging (ARCH-BUILD).

**Arbeitsweise:** inline, **kein Workflow, keine Subagenten** ohne Olis ausdrückliches Wort. Commit + Push je Phase, dann Stop + Bericht. Nichts Tragendes im Session-Scratch. Editiert wird nur auf dem Mac. **Kein Schlüssel, kein Token, kein JWT in irgendeiner Ausgabe oder Datei** — Längen, Formen und Hash-Vergleiche reichen.

---

# Phase 1 — Bau

**Tests zuerst, gegen HEAD rot gezeigt:**
- View, Erfolg: liefert das Token des (gemockten) Grants und `expires_in`; der Grant wurde mit der Server-TTL und einer Deadline gerufen.
- View, Grant wirft: Fehlerantwort, im Body weder Schlüssel noch Token; Status 502.
- View ohne Login: wie jede `@login_required`-Route; ohne konfigurierten Dienst: das bestehende 503 aus `require_service`.
- Sentinel aus Entscheidung 7.
- Dienst: die Methode reicht TTL und Deadline an das SDK durch und gibt Token und Ablauf zurück; ein leeres oder formloses SDK-Ergebnis ist ein Fehler, kein leeres Token.

**Dann der Fix** in [services/deepgram_service.py](../../../services/deepgram_service.py), [app_pkg/audio.py](../../../app_pkg/audio.py), der Konstante in [app_pkg/config.py](../../../app_pkg/config.py) und [static/js/audio_converter.js](../../../static/js/audio_converter.js) (Subprotokoll, Reihenfolge). Die englische Fehlermeldung der View wird deutsch (Haus-Microcopy).

**Gates:** Suite am Mac (Baseline 1401 + 1 Skip, Abweichung benennen); **Container-Suite auf dem Pin** (stdin-Rezept aus CLAUDE.md, `COPYFILE_DISABLE=1 tar --no-xattrs`); dazu im Pin-Container ein Aufruf der echten SDK-Methode gegen einen Wegwerf-Client **ohne Netz**, der nur belegt, dass Signatur und Parameter-Namen stimmen (kein echter Grant in dieser Phase).

## Stop
Bericht: die roten Läufe vor dem Fix · Diff je Datei · welche SDK-Signatur am Pin gilt (Parameter-Namen, Rückgabeform) · Suite Mac und Container · Abweichungen vom Prompt. Dann warten.

---

# Phase 2 — Deploy + Live-Beleg

1. **Fenster:** Queue leer, Worker idle, 0 `pending`. Rollback-Tag `converter-app:pre-sec-dg-token`; `pre-arch-narr5` bleibt (sein Rückweg braucht die alte Compose-Datei — nicht anfassen). Kein DB-Backup nötig, wenn der Smoke nichts auf Olis Konto schreibt; sonst nach Rezept.
2. `git pull --ff-only`, `docker compose up -d --build` aus dem Projektverzeichnis; `docker inspect`: was neu ist; Login 200.
3. **Server-Beleg im Web-Container** (ohne Werte auszugeben): die View über einen Wegwerf-User aufrufen → 200, `deepgram_token` hat JWT-Form, sein sha256 ist **ungleich** dem sha256 des Env-Schlüssels, `expires_in` ist die Server-TTL; mit dem gelieferten Token `/v1/projects` → 403.
4. **Der Live-Beleg im echten Chromium**, als Skript im Stil der Smokes (im Web-Container, Wegwerf-User, Aufräumen strikt nach `user_id`): Chromium mit falschem Mikrofon starten (`--use-fake-ui-for-media-stream`, `--use-fake-device-for-media-stream`, `--use-file-for-fake-audio-capture=<wav>`; die Sprach-WAV im Container aus einer vorhandenen Aufnahme schneiden und danach löschen), auf der Audio-Seite die Live-Aufnahme starten und belegen:
   - der Token-Abruf geschieht **nach** der Mikrofon-Freigabe;
   - der WebSocket zu `api.deepgram.com` wird mit dem ersten Subprotokoll **`bearer`** geöffnet (ein Init-Skript um `window.WebSocket` zeichnet nur `protocols[0]` auf, nie das Token);
   - im Textfeld erscheint Transkript-Text;
   - die Aufnahme läuft **länger als die TTL**, und nach dem Ablauf kommt weiter Text (Entscheidung 6);
   - Stopp beendet sauber, eine zweite Aufnahme holt ein neues Token (zwei Abrufe, zwei verschiedene Hashes).
5. **Gegenprobe:** im Web-Log steht bei keinem der Abrufe der Schlüssel oder das Token; die Antwort der View enthält den Env-Schlüssel nicht (Hash-Vergleich aus 3).
6. Inventar: Wegwerf-User weg, keine Datei in `/tmp` der Container, Volume und DB wie vorher, Mintbox-Checkout sauber.

Sicherheits-Regeln wie immer: `docker exec` nie mit `-u 0`; DB-Lesen `mode=ro`; keine unversionierte Datei auf der Mintbox. Kosten: rund zwei Minuten Deepgram-Streaming.

## Stop
Bericht: Deploy · Server-Beleg (Form, Hash-Ungleichheit, 403) · Live-Beleg je Punkt · ob die Verbindung das Token-Ende überlebt hat, mit Zeiten · Inventar.

---

# Phase 3 — Wrap

- **CLAUDE.md:** im DIARIZE-Bullet (oder eigener kurzer Bullet) die Live-Transkription: Kurzzeit-Token per Grant, Server-TTL, `bearer`-Subprotokoll, Reihenfolge Mikrofon → Token → Socket, fail-closed, die gemessene Eigenschaft aus Entscheidung 6; im SEC-AUDIT-Bullet ein Satz, dass der Endpunkt nachträglich bewertet und geschlossen ist; das Live-Smoke-Skript in die Skript-Liste; Test-Baseline.
- **[AUDIT_SECURITY_2026-09-25.md](../audit-outputs/AUDIT_SECURITY_2026-09-25.md)** bleibt Archiv — nicht editieren; der Nachtrag steht in STATUS und im geschlossenen BACKLOG-Item.
- **STATUS.md**, **BACKLOG.md** (SEC-DG-TOKEN schließen; ⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber). Im geschlossenen Item als **Empfehlung an Oli, nicht gebaut**: der App-Schlüssel trägt `account:write` — ein Schlüssel mit der kleinsten Rolle, mit der Transkription **und** Grant noch gehen, wäre der nächste Schritt (Olis Hand bei Deepgram; die Proben dieses Sprints sagen in Sekunden, ob er trägt).
- **Kein Brief ans converter-mcp** (kein Tool nutzt den Endpunkt; im Wrap so sagen, nach einem Blick auf die Tool-Liste am lebenden Connector).
- **Memory** nur bei übertragbarer Lehre. Kandidat: *eine Abwägung überlebt den Wegfall ihrer Prämisse nicht von selbst — wo ein Docstring eine Sicherheitsentscheidung mit einer Umgebungsannahme begründet („LAN-only"), gehört die Stelle beim Kippen der Annahme auf die Prüfliste.* Vorher prüfen, was `reference_reverse_proxy_one_hop_cookie_scheme` und die SEC-AUDIT-Memories schon tragen.
- Commit + Push, Stop + Bericht.

## Nicht-Ziele

Kein Proxy für den Audio-Strom, keine CSP, kein Eingriff in `transcribe_file` oder den Job-Pfad, kein neuer Schlüssel, kein Umbau von `require_service`, kein Frontend außer Token-Abruf, Subprotokoll und Reihenfolge.
