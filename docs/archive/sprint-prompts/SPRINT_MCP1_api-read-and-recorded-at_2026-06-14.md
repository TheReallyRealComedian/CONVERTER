# SPRINT MCP1 — JSON-Read-API für Conversions + `recorded_at`-Capture

**Größe:** M · **Datum:** 2026-06-14 · **Rolle:** Master = Dispatch, du = Execute.
**Stop-Regel:** Nach **jeder** Phase anhalten, kurz berichten, auf Sign-off warten. Nicht zusammenfassen, nicht umplanen — ausführen.

---

## Warum dieser Sprint (Kontext, knapp)
CONVERTER soll mittelfristig per eigenem **`converter-mcp`** (separates Projekt, Muster wie `nextcloud-mcp`) an Claude andocken. Erster konkreter Use-Case: unarchivierte **Audio-Transkripte** finden → Meeting identifizieren → zusammenfassen → in Notion ablegen → in CONVERTER archivieren.

Für die **Lese-Seite** fehlt CONVERTER heute alles: `/api/conversions` hat nur **POST/PUT/DELETE**, die Liste kommt ausschließlich als **HTML** über `/library`. Archivieren (`PUT … {lifecycle_status:'archive'}`) und Anlegen (`POST /api/conversions`, Typ `audio_transcription`) existieren bereits und bleiben **unangetastet**.

Dieser Sprint liefert: **(1)** zwei JSON-Leser (Liste + Einzel), **(2)** das Mitschreiben eines **Aufnahme-Zeitstempels** (`recorded_at`) in die Conversion-Metadaten beim Speichern (Datenbasis fürs spätere Meeting-Matching). Alles read-only bzw. additiv — kein bestehendes Verhalten ändern.

## Ausdrücklich NICHT in diesem Sprint (Out of Scope)
- Der `converter-mcp` selbst (eigenes Projekt, kommt separat über den Master).
- Notion-/Kalender-Orchestrierung, Matching-Logik, Zusammenfassen (macht der Agent zur Laufzeit, kein Code hier).
- Auth/Exposure-Fragen, Nginx, Mermaid-Abklemmen.
- Keine Änderung an `POST`/`PUT`/`DELETE`-Verhalten außer der in **Phase 2** beschriebenen additiven `recorded_at`-Anreicherung im POST.

## Guardrails / Konventionen (verbindlich)
- **Routing:** Route-Module exponieren `register(app)`, **keine** Flask-`Blueprint`. Neue Endpoints kommen in `app_pkg/library.py` in das bestehende `register(app)`. Flache Endpoint-Namen.
- **Eigene View-Funktionen pro Methode:** `/api/conversions` (POST existiert) bekommt eine **separate** GET-Funktion `api_list_conversions` (eigener Funktionsname = eigener Endpoint, Flask dispatcht nach Methode). Genauso `api_get_conversion` für GET auf `<id>` (PUT/DELETE existieren dort schon). NICHT in eine Funktion zusammenlegen.
- **Security:** Jeder neue Endpoint `@login_required` **und** owner-scoped auf `current_user.id`. Für den Einzel-Reader `get_owned_conversion(conversion_id)` wiederverwenden (liefert das 404-Verhalten konsistent).
- **Wiederverwenden statt neu bauen:** `ALLOWED_CONVERSION_TYPES`, `LIFECYCLE_STATUSES`, `get_owned_conversion`, `Conversion.to_dict()`.
- **`Conversion.to_dict()` NICHT verändern** (wird von POST/PUT-Responses + Frontend genutzt). Der List-Reader baut sich ein **eigenes, schlankes** Summary-Dict (siehe Phase 1).
- **Fehlertexte deutsch**, im Stil der bestehenden (`'Ungültiger Request-Body. JSON-Objekt erwartet.'`, `'Ungültiger Lifecycle-Status.'`).
- **Tests:** in `tests/test_library.py` ergänzen (oder klar benanntes neues Modul). `pytest tests/` muss **grün bleiben** (Baseline zuerst feststellen, lt. BACKLOG zuletzt 213/213).
- **Code-Stil:** knapp, am umgebenden Code orientiert.

## Pre-Flight (vor Phase 1)
1. `pytest tests/` laufen lassen, **Baseline-Zahl** notieren (muss grün sein; sonst STOP + melden).
2. `app_pkg/library.py` lesen: `register(app)`, `get_owned_conversion`, `api_create_conversion` (POST), `api_update_conversion` (PUT), `pagination`/Query-Logik im `/library`-Handler, `ALLOWED_CONVERSION_TYPES`, `LIFECYCLE_STATUSES`. `models.py` → `Conversion.to_dict()`.

---

## Phase 1 — JSON-Read-Endpoints (read-only, der Kern)

### `GET /api/conversions` → `api_list_conversions`
`@login_required`, nur Conversions von `current_user`. Query-Parameter (alle optional):
- `type` — wenn gesetzt, muss in `ALLOWED_CONVERSION_TYPES` sein (sonst **400**). Filtert `conversion_type`.
- `status` — wenn gesetzt, muss in `LIFECYCLE_STATUSES` sein (sonst **400**). Exakter `lifecycle_status`-Filter.
- `exclude_status` — wenn gesetzt, muss in `LIFECYCLE_STATUSES` sein (sonst **400**). Schließt diesen Status aus. (Damit holt die MCP „unarchiviert" = `exclude_status=archive` sauber serverseitig.)
- `limit` — int, Default **100**, Max **500** (darüber → 400 oder cappen auf 500; entscheide dich für **cappen** und dokumentiere es im Docstring). Ungültig (nicht-int, <1) → **400**.
- `offset` — int, Default **0**, <0 → **400**.

**Response 200:**
```json
{ "items": [ <summary>, ... ], "total": <int matching vor limit>, "limit": <int>, "offset": <int> }
```
`<summary>` = schlankes Dict **ohne vollen `content`**, mit:
`id, conversion_type, title, source_filename, source_mimetype, source_size_bytes, lifecycle_status, is_favorite, last_read_percent, queue_position, created_at, updated_at, tag_refs (Liste {id,name}), metadata (geparster Dict), content_length (len(content)), content_preview (erste 300 Zeichen von content)`.

Sortierung: `created_at` **desc** (neueste zuerst).

### `GET /api/conversions/<int:conversion_id>` → `api_get_conversion`
`@login_required`. `get_owned_conversion(conversion_id)` nutzen → bei fremd/fehlend dessen 404. Response 200 = volles `conversion.to_dict()` (inkl. `content` + `metadata`).

### Tests (Phase 1)
- List: unauth → kein Zugriff (Login-Redirect/401, wie die anderen `/api`-Tests).
- List: Owner-Scoping (User A sieht B's Conversions nicht).
- `type` valide filtert; invalide → 400.
- `status` valide filtert; invalide → 400.
- `exclude_status=archive` blendet Archivierte aus.
- Summary enthält **kein** `content`, aber `content_length` + `content_preview` + `metadata`.
- `limit`/`offset`: Begrenzung + Cap auf 500 + invalide → 400; `total` = Gesamtzahl vor limit.
- Sortierung created_at desc.
- Single: volles `to_dict` inkl. content; fehlende/fremde id → 404.

### Akzeptanz Phase 1
Neue Endpoints liefern korrektes JSON, alle neuen Tests grün, **Gesamt-Suite grün** (≥ Baseline). → **STOP + Bericht** (Baseline-Zahl, neue Testzahl, kurze Endpoint-Demo z. B. via `curl`/Test-Client).

---

## Phase 2 — `recorded_at`-Capture beim Anlegen (backend, additiv)

Ziel: Beim Speichern einer Conversion einen **Aufnahme-Zeitpunkt** in die Metadaten schreiben, ohne Schema-Änderung (der `metadata`-Beutel existiert). Quellen, in dieser **Präzedenz**:
1. **Explizit** vom Client: optionales Feld `recorded_at` im Request-Body (ISO-8601-String **oder** Epoch-ms-Zahl, z. B. aus `file.lastModified`). Hat Vorrang.
2. **Aus dem Dateinamen** abgeleitet: best-effort Parser über `source_filename` (Diktiergeräte kodieren oft Datum/Zeit).

### Parser `parse_recorded_at_from_filename(filename) -> datetime | None` (pure function, unit-testbar)
- Erkennt **klare** Datums(+Zeit)-Muster im Namen, z. B.: `20260612`, `2026-06-12`, `20260612_1430`, `20260612-143005`, `2026-06-12 14.30`, `2026_06_12T14_30`. Optional vorangestellte Recorder-Präfixe (`REC`, `ZOOM`, `VOICE`, `WS`, `MIC` …) ignorieren — einfach das Datums-Substring suchen.
- **Validieren** (Monat 01–12, Tag 01–31, Jahr 2000–2100). Keine Zeit gefunden → 00:00 lokale Zeit.
- **Konservativ:** Bei Mehrdeutigkeit/keinem validen Treffer → `None`. **Falsch ist schlimmer als leer** (ein falscher `recorded_at` vergiftet später das Matching). Negative Beispiele müssen `None` ergeben: `Besprechung.mp3`, `audio (1).mp3`, `12345678.mp3` (kein valides Datum), `New Recording 7.m4a`.
- **Zeitzone:** Dateiname-Zeiten als **Europe/Berlin** interpretieren, als ISO-8601 **mit Offset** zurückgeben/speichern.

### Einbau in `api_create_conversion`
- `metadata`-Dict aus `data.get('metadata', {})` nehmen (defensiv: muss dict sein, sonst {}).
- `recorded_at` ermitteln (Präzedenz oben). Client-`recorded_at`: ISO **oder** Epoch-ms akzeptieren, nach ISO-8601-UTC normalisieren; unparsebar → ignorieren (kein 400, additiv bleiben — aber loggen).
- Wenn ein `recorded_at` ermittelt wurde **und** der Client nicht schon selbst `metadata.recorded_at` gesetzt hat: in `metadata['recorded_at']` (ISO-String) schreiben **und** `metadata['recorded_at_source']` = `'client'` | `'filename'`.
- Dann erst `json.dumps(metadata)` → `metadata_json`. Bestehende Felder im metadata-Dict nicht verlieren.

### Tests (Phase 2)
- Parser: positive Muster (mind. 3 Varianten), negative (die o. g. → None), keine False-Positives bei ungültigen Datumszahlen.
- Create speichert `metadata.recorded_at` + `recorded_at_source='client'` bei explizitem ISO-`recorded_at`.
- … bei Epoch-ms-`recorded_at` (korrekt nach ISO normalisiert).
- … `recorded_at_source='filename'` wenn nur über `source_filename` ableitbar.
- Explizit schlägt Dateiname (Präzedenz).
- Weder noch → kein `recorded_at` im metadata (oder nicht vorhanden), kein Crash.

### Akzeptanz Phase 2
Alle neuen Tests grün, Gesamt-Suite grün, `POST`-Verhalten sonst unverändert (bestehende Create-Tests weiter grün). → **STOP + Bericht.**

---

## Phase 3 — Frontend: `recorded_at` beim Audio-Speichern mitgeben (discovery-gated, live-smoke)

**Erst Discovery, dann entscheiden:** Finde die Stelle, an der ein **Audio-Transkript in die Library gespeichert** wird (vermutlich `static/js/audio_converter.js` → `POST /api/conversions` mit `conversion_type='audio_transcription'`).

- **Falls es diesen Save-Pfad NICHT gibt** (Audio wird heute nur transkribiert/angezeigt, aber nicht gespeichert): **STOP + melden** — das ist eine Scope-Frage für den Master (neuer Save-Flow ist ein eigener Sprint), NICHT hier hineinbauen.
- **Falls es ihn gibt:** beim Speichern zusätzlich mitsenden:
  - `source_filename` = Originalname der hochgeladenen Datei (falls nicht eh schon),
  - `recorded_at` = `file.lastModified` der Upload-Datei (Epoch-ms; Phase-2-Backend nimmt das an).
  - Minimaler, lokaler Eingriff; Helfer aus `_utils.js` wiederverwenden; keine UI-Strings nötig außer evtl. unverändert.

**Live-Smoke (Pflicht, da Charakterisierungstests kein JS/Template abdecken):** echte/Beispiel-Audiodatei hochladen → transkribieren → speichern → prüfen, dass die erzeugte Conversion `source_filename` **und** `metadata.recorded_at` (`recorded_at_source='client'`) trägt (über den neuen `GET /api/conversions/<id>`).

### Akzeptanz Phase 3
Save-Pfad sendet die Felder, Live-Smoke bestätigt das Mitschreiben, Suite grün. → **STOP + Bericht.**

---

## Wrap-up (nach Sign-off der letzten Phase)
- `STATUS.md` (aktueller Stand) + `BACKLOG.md` (Item als erledigt markieren / Folge-Items wie „converter-mcp scaffolden", „recorded_at retroaktiv?" notieren) pflegen — Working Practice Punkt 5.
- Falls etwas Nicht-Triviales gelernt wurde (z. B. reale Diktiergerät-Namensmuster): kurz in die Memory-Zone.
- Neue Endpoints im README/CLAUDE.md erwähnen, falls sinnvoll (optional).

## Definition of Done (Gesamt)
`GET /api/conversions` (mit `type`/`status`/`exclude_status`/`limit`/`offset`) + `GET /api/conversions/<id>` liefern sauberes JSON; `recorded_at` wird beim Anlegen aus Client-Wert bzw. Dateiname additiv in die Metadaten geschrieben; (Phase 3) Audio-Save gibt `recorded_at` mit oder hat sauber an den Master zurückgemeldet, dass der Save-Pfad fehlt; `pytest tests/` grün ≥ Baseline; kein bestehendes Verhalten regressiert.
