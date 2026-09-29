# SPRINT NOTION-TZ — „An Notion senden → Meetings" trägt die Uhrzeit mit Zeitzone

**Größe**: S (3 kurze Phasen, eine davon mit einem Oli-Schritt) · **Datum**: 2026-09-29 · **Vorhaben**: Bug-Fix nach Hinweis des Koordinators ([docs/notion_meetings_zeitzone_hinweis.md](../../notion_meetings_zeitzone_hinweis.md)), BACKLOG NOTION-TZ, **live bestätigt** (Oli, 2026-09-29: Termin steht in Notion zwei Stunden später als im Formular)

## Warum

Das Feld `datum` im Notion-Panel ist ein `<input type="datetime-local">` und liefert Ortszeit **ohne** Zeitzone (`2026-09-29T14:30`). CONVERTER reicht es unverändert an `POST /api/meetings` des notion-mcp-server weiter, der es unverändert an Notion gibt — und Notion liest eine Uhrzeit ohne Offset und ohne `time_zone` als **UTC**. Aus 14:30 Ortszeit werden 16:30 (MESZ), im Winter 15:30. Der Fehler ist so alt wie der Knopf (MCP1), betrifft aber nur Meetings, die über CONVERTER angelegt wurden; die Transkript-Meetings aus `notion-transcripts` tragen kein Zeitfeld.

## Gegroundeter Ist-Zustand (Master, gemessen 2026-09-29 — nicht neu herleiten, Abweichungen benennen)

**Formular** ([static/js/library_detail.js](../../../static/js/library_detail.js) `renderNotionFields`, Z. 342–348): nur das Ziel `meetings` hat `{key: 'datum', type: 'datetime-local', value: formatDatetimeLocalNow()}`; `notes` und `inbox` tragen **kein** Zeitfeld. `formatDatetimeLocalNow` ([static/js/_utils.js:127](../../../static/js/_utils.js)) liefert `toISOString().slice(0, 16)` der lokal verschobenen Zeit → `YYYY-MM-DDTHH:MM`. `sendToNotion` (Z. 401–405): `fields[key] = el.value.trim()`, roh.

**Backend** ([app_pkg/integrations/notion.py](../../../app_pkg/integrations/notion.py) `api_send_to_notion`, Z. 98–125): `target ∈ {meetings, notes, inbox}`, `payload = {k: v for k, v in data['fields'].items() if v}`, `POST {NOTION_MCP_URL}/api/{target}` mit `Authorization: Bearer {MCP_AUTH_TOKEN}`, Timeout 30; 4xx des Servers werden mit dessen `error`/`detail` durchgereicht, `RequestException` → 502. **Keine Umrechnung, kein `time_zone`.** `http_requests` ist das Modul-Alias von `requests` — Tests patchen `app_pkg.integrations.notion.http_requests.post`. **Kein Test deckt die Route** (`grep send-to-notion tests/` leer; `tests/conftest.py` stubbt nur Module).

**Server** (Mintbox-Clone `~/CODE/notion-mcp-server`, `134d1d8`, Container seit 2026-09-29 04:32; `server.py` `_normalize_meeting_datum`, Z. 1657–1708): akzeptiert `datum` als String `YYYY-MM-DD` (ganztägig), `YYYY-MM-DDTHH:MM[:SS][±HH:MM|Z]`, **oder** als Objekt `{start, end?, time_zone?}`; `time_zone` muss ein IANA-Name sein, ist nur mit Uhrzeit erlaubt und **nur ohne** UTC-Offset im `start` (Notion-Regel); `start`/`end` beide gleicher Art. Docstring wörtlich: *„Offset-free datetimes without `time_zone` are still accepted for backward compatibility (CONVERTER sends datetime-local values) — Notion stores them as UTC."* Die Server-Tests (`tests/test_meetings_api.py`, Z. 339–358) fahren genau `{"start": "2026-10-05T09:00:00", "time_zone": "Europe/Berlin"}`. **`POST /api/meetings/query`** (Z. 2080 ff., `MCP_AUTH_TOKEN` genügt): Body `{"date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}` → kompakte Liste, nach `Datum` aufsteigend, **ohne** Transkript-Text; Antwortform vor Gebrauch lesen (ob `time_zone`/Rohwert je Eintrag sichtbar ist, entscheidet, wie der Altbestand erkennbar wird).

**Zeitzone im Code**: `LOCAL_TZ = ZoneInfo('Europe/Berlin')` in [app_pkg/learn.py:36](../../../app_pkg/learn.py) (9 Verwendungen in learn) und `_BERLIN_TZ = ZoneInfo('Europe/Berlin')` in [app_pkg/library.py:119](../../../app_pkg/library.py) (Dateinamen-Zeiten werden „als Europe/Berlin gelesen"). Zwei Konstanten, ein Wert, kein geteilter Ort.

**Tests**: Baseline **1323 + 1 Skip** (Mac und Container-Pin). Fixtures `app`, `client`, `test_user`, `authenticated_client` in [tests/conftest.py](../../../tests/conftest.py); wie Tests eine Conversion des `test_user` anlegen, zeigen die Library-/Docwrite-Tests (ORM im `app.app_context()`). Mac `main` auf `2ce3b2f`+, Mintbox-Clone auf `66da12a`.

## Gesperrte Entscheidungen

1. **Fix im Backend, nicht im Browser** (Empfehlung des Hinweises, vom Master geteilt): die Zone ist **serverfest `Europe/Berlin`** — unabhängig von der Gerätezone, wie schon `learn` und die Dateinamen-Zeiten in `library`. Dokumentierte Eigenschaft: wer aus einer anderen Zone sendet, bekommt Berliner Zeit; das ist Olis Kalender, nicht der des Geräts.
2. **Eine Zone, ein Ort**: `LOCAL_TZ = ZoneInfo('Europe/Berlin')` wandert nach [app_pkg/config.py](../../../app_pkg/config.py); `learn.py` importiert sie unter demselben Namen (die 9 Verwendungen bleiben unberührt), `library.py` setzt `_BERLIN_TZ = LOCAL_TZ` (oder ersetzt es). Sentinel: `learn.LOCAL_TZ is config.LOCAL_TZ` und `library._BERLIN_TZ is config.LOCAL_TZ`. Keine weitere Umbenennung.
3. **Pure Helper `normalize_notion_datum(value)`** im Notion-Modul (kein Flask): `YYYY-MM-DD` → unverändert (ganztägig); String mit Offset (`±HH:MM`, `Z`) → unverändert; **naive Zeit** `YYYY-MM-DDTHH:MM[:SS]` → `{"start": "<Wert mit Sekunden>", "time_zone": "Europe/Berlin"}` (Sekunden ergänzen, wenn sie fehlen — die Server-Tests fahren die Sekunden-Form); alles andere (Objekt, leer, Unerkennbares) → **unverändert durchreichen** — der Server validiert und antwortet 400 mit seinem Text, den die Route schon heute weitergibt. Der Helper **wirft nie**. Anwendung in `api_send_to_notion` auf `payload['datum']`, wenn vorhanden (nur `meetings` trägt es; keine Zielabfrage nötig).
4. **Kein Frontend-Touch**, **kein Umbau anderer Felder**, **keine Korrektur des Altbestands** — der Sprint **listet** ihn (Phase 2) und Oli entscheidet.
5. **Tests**: Helper-Fälle (ganztägig, naiv Minuten, naiv Sekunden, `+02:00`, `Z`, leer, Objekt, Müll — je erwartete Rückgabe); Route-Test mit `authenticated_client`, einer Conversion des `test_user` und gepatchtem `http_requests.post`: `fields.datum = "2026-09-29T14:30"` → das an den Server gesendete `json` trägt `datum == {"start": "2026-09-29T14:30:00", "time_zone": "Europe/Berlin"}`, alle anderen Felder unverändert; Gegenprobe `datum = "2026-09-29"` unverändert; `target: notes` ohne `datum` → kein `datum` im Payload; Server-400 wird weiter durchgereicht (Bestandsverhalten). Baseline danach 1323 + n.
6. **Der Live-Beleg gehört Oli**: die Wahrheit steht in Notion, und eine Probe legt dort eine Seite an. Kein Sub-Thread-Zugriff auf Notion, keine Probe-Seiten aus dem Sprint heraus; Phase 2 stoppt für Olis Klick.
7. ⚠️ **Editiert wird nur auf dem Mac.** Mintbox = Runtime; `docker exec` nie `-u 0`; `MCP_AUTH_TOKEN` nie ausgeben — die Altbestands-Abfrage läuft **im Web-Container** (`docker exec -i markdown-converter-web python3 -`), der den Token in der Env hat; keine Reste.

---

# Phase 1 — Helper, Konstante, Route, Tests

- Entscheidung 2 (Konstante), Entscheidung 3 (Helper + Einbau), Entscheidung 5 (Tests). Mac-Suite grün, Container-Suite auf dem Pin (stdin-Rezept aus CLAUDE.md *Key Files*, mit `--no-xattrs`). Commit + Push.
- Kein Deploy in Phase 1.

## Stop
Bericht: Diff, Testzahl, der gesendete Payload aus dem Route-Test wörtlich. Dann warten.

---

# Phase 2 — Deploy, Altbestand, Olis Probe

## 2.1 Deploy
Fenster wie zuletzt (keine laufende Konvertierung, `rq:wip` 0, Worker `idle`, kein `mineru_*`). Mintbox: `git pull --ff-only`, `docker compose up -d --build` aus dem Projektverzeichnis (Code-Layer). Login 200. Rollback-Tag vorher: `docker tag <aktuelle Image-ID> converter-app:pre-notion-tz`; `pre-img-context` darf danach weg (nur das Tag).

## 2.2 Altbestand listen (nur lesen)
Im Web-Container `POST {NOTION_MCP_URL}/api/meetings/query` mit `date_from` = **2026-06-01** (MCP1, der Knopf entstand im Juni) und `date_to` = heute, Token aus der Env, **nie ausgeben**. Antwortform lesen; Meetings mit Uhrzeit und ohne `time_zone` sind Kandidaten für den Versatz — falls die Liste das nicht hergibt, alle Meetings mit Uhrzeit im Fenster nennen und das so sagen. Ergebnis als Tabelle (Titel, gespeicherter Datum-Wert) in den Bericht, **keine** Änderung.

## 2.3 Olis Probe (Stop-Punkt)
Dem Bericht einen Absatz für Oli voranstellen: ein beliebiges Library-Element öffnen → „An Notion senden → Meetings" → Titel `NOTION-TZ Probe`, Datum heute **14:30** → senden. In Notion prüfen: steht der Termin auf **14:30** (Zone Europe/Berlin)? Danach die Probe-Seite in Notion löschen. Oli meldet das Ergebnis in den Sub-Thread; erst dann Phase 3.

## Stop
Bericht: Deploy, Altbestands-Tabelle, Probe-Anleitung. Warten auf Olis Meldung, dann Stop + Bericht der Meldung.

---

# Phase 3 — Wrap

- **Antwort an den Koordinator**: `docs/notion_meetings_zeitzone_antwort.md` neben dem Hinweis — was CONVERTER sendet (die Objektform mit `time_zone`, Sekunden ergänzt, ganztägig und Offset-Formen unverändert), dass ihre Alternative (Zeiten ohne Offset serverseitig als Berlin werten) **nicht** nötig ist, Olis Probe-Ergebnis, der Altbestand als Liste mit dem Hinweis, dass die Korrektur Olis Entscheidung ist; Dank für den Kompatibilitätstest mit unserem Payload. Sachlich, kurz.
- **CLAUDE.md**: im Architecture-Bullet zur Notion-Integration (oder als eigener kurzer Bullet, falls keiner existiert) drei Sätze: `datetime-local` ist zonenlos, Notion liest zonenlos als UTC, das Backend hängt `Europe/Berlin` an; `LOCAL_TZ` liegt jetzt in `config.py` und ist die eine Zone für learn, library und Notion. Test-Baseline.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): NOTION-TZ schließen; falls Oli den Altbestand korrigieren will, ein Folge-Item **NOTION-TZ-ALT** mit der Liste anlegen, sonst den Verzicht notieren.
- **Kein Brief ans converter-mcp** (Web-Knopf, keine Agent-Fläche; der Koordinator ist die Gegenseite und bekommt die Antwort) — so sagen.
- **Memory**: eine Datei nur, wenn die Lehre übertragbar ist — Kandidat: *`datetime-local` ist zonenlos; jeder Konsument, der eine Zone braucht, muss sie explizit bekommen, sonst wird sie still UTC — und das fällt erst auf, wenn jemand die Uhr vergleicht.* Kurz, mit dem gemessenen Versatz.
- **Im Bericht**: Payload alt/neu, Testzahl, Olis Probe-Ergebnis, Altbestands-Zahl, Tags.

## Nicht-Ziele

- **Keine** Änderung an [static/js/library_detail.js](../../../static/js/library_detail.js) oder `_utils.js`.
- **Keine** Zeitzone aus dem Browser, **kein** Nutzer-Setting dafür.
- **Keine** Korrektur bestehender Notion-Meetings ohne Olis Wort.
- **Keine** Änderung am notion-mcp-server (fremdes Repo, fremder Master).
