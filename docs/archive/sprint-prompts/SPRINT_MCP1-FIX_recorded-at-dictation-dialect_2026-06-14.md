# Sprint MCP1-FIX — `recorded_at` Diktiergerät-Dialekt + Präzedenz-Flip + Backfill (S)

> **Executor-Doc, Folge-Sprint zu MCP1.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **262**). Working Practice: Master dispatcht, du executest; **du committest jede Phase selbst** (eigener Hash + push). Arbeitsverzeichnis: `/Users/olivergluth/CODE/CONVERTER` (Mac-lokal = Source-of-Truth; der Mintbox-Mount `/Volumes/...` ist NUR Runtime, nicht dein Arbeitsplatz — Memory `reference_two_clone_coordination_mac_mintbox`). Zeilennummern = Orientierung, geh über Funktions-/Symbol-Namen.

## Kontext / Warum

Real-Diktat-Test (Top-Level-Koordinator, 2026-06-14): Olis Diktiergerät benennt Files **`YYMMDD_NNNN.MP3`** — 2-stelliges Jahr + laufende Nummer, **keine Uhrzeit**, z.B. `260521_0176.MP3` = 2026-05-21. Der MCP1-Parser `parse_recorded_at_from_filename` ([app_pkg/library.py](../../app_pkg/library.py)) kennt nur **4-stellige** Jahre (`_RECORDED_AT_RE`, YYYY-first) → liefert für **alle** realen Diktate `None` (verifiziert am deployten Parser: `20260612_1430.mp3` → ok, `260521_0176.MP3` → None). `recorded_at` bleibt bei echten Diktaten leer.

Dieser Sprint: (1) Parser um den Diktiergerät-Dialekt erweitern, (2) Präzedenz `filename > client-lastModified` (Master-Entscheidung, s.u.), (3) die 5 Bestands-Rows backfillen. Alles additiv/konservativ — kein Schema-Touch, kein Bruch bestehender Endpoints.

## Master-Entscheidungen (gesetzt, nicht neu diskutieren)
- **Präzedenz-Flip**: `metadata.recorded_at` explizit gesetzt (wins) > **Dateiname-Datum** > Client-`recorded_at`-Feld (lastModified, Fallback). Grund: lastModified kann die Kopier-Zeit sein; der Diktiergerät-Dateiname trägt das geräte-autoritative Aufnahme-Datum. Der Flip betrifft nur Fälle mit *parsebarem* Dateinamen — bei beliebigen Namen bleibt filename→None→client unverändert.
- **Backfill ja**: Script importiert den Runtime-Parser, dry-run-Default; **Mintbox-Apply = Olis Real-Welt-Schritt** (die 5 Diktate liegen auf der Prod-DB).
- **Konservativität bleibt oberstes Prinzip**: „falsch ist schlimmer als leer" (ein falscher `recorded_at` vergiftet das Meeting-Matching).

---

## Phase 1 — Parser-Dialekt + Präzedenz-Flip (backend, additiv) + Tests

### 1a — `parse_recorded_at_from_filename` um `YYMMDD_NNNN` erweitern
- **Nur in klarer Diktiergerät-Form greifen**: der Stem **beginnt** mit `\d{6}_\d+` (eigene, am Stem-Anfang **verankerte** Regex `^(\d{2})(\d{2})(\d{2})_\d+` — **kein** beliebiger 6-Ziffern-Run irgendwo im Namen, kein Recorder-Präfix davor).
- `YY → 20YY`. **Validieren**: `MM` 01–12, `DD` 01–31, plus echte `datetime()`-Konstruktion (fängt z.B. `260230` Feb-30) → ungültig ⇒ kein Kandidat (Konservativität).
- **Keine Uhrzeit** in diesem Format → `00:00` Europe/Berlin (wie die bestehenden date-only-Fälle), tz-aware Rückgabe mit Offset.
- **Sequenznummer (`_NNNN`) NICHT als Datum** interpretieren.
- **Mechanik-Empfehlung**: den Diktat-Kandidaten in dasselbe `found`-Set legen wie die bestehende YYYY-Logik und das vorhandene `len(found) == 1`-Ambiguitäts-Gate wiederverwenden → wenn (kontrivierter) Fall sowohl Diktat- als auch YYYY-Treffer mit **verschiedenen** Daten liefert, bleibt es konservativ `None`. Für reale Diktate feuert nur der Diktat-Zweig (die kurze `_NNNN`-Sequenz triggert die 8-stellige YYYY-Regex nicht).

### 1b — Präzedenz-Flip in `api_create_conversion`
Heutige Reihenfolge (MCP1-P2): explizit-`metadata.recorded_at` > Client-`recorded_at`-Feld > Dateiname. **Neu**: explizit-`metadata.recorded_at` > **Dateiname** > Client-`recorded_at`-Feld.
- Wenn der Client nicht schon `metadata.recorded_at` gesetzt hat: **zuerst** `parse_recorded_at_from_filename(source_filename)`; Treffer → `metadata['recorded_at']` + `recorded_at_source='filename'`. **Nur wenn None** → Client-`recorded_at`-Feld via `_normalize_client_recorded_at`; Treffer → `recorded_at_source='client'`.
- Sonst alles wie gehabt: additiv, **nie 400**, bestehende metadata-Felder erhalten, kein Schema-Touch.

### Tests (Phase 1)
- **Parser-Positiv (5 reale Namen)**: `260521_0176` / `260518_0172` / `260508_0171` / `260506_0170` / `260610_0184` (je `.MP3`) → korrektes `20YY-MM-DD` 00:00 Berlin-Offset.
- **Parser-Negativ/Konservativ bleibt grün**: die bestehenden 4-stelligen Positiv-Fälle unverändert; die MCP1-Negativen (`Besprechung.mp3`, `audio (1).mp3`, `12345678.mp3`, `New Recording 7.m4a`) weiter `None`; **neu**: 6-Ziffern-Run **nicht am Stem-Anfang** (`notes_260521_0176.mp3`) → `None`; **ungültiges** Diktat-Datum (`263421_0001.mp3`, MM=34) → `None`; `260521.mp3` ohne `_NNNN` → `None` (oder bewusst zulassen? **nein** — Form ist `\d{6}_\d+`, ohne Sequenz kein Diktat-Match; halte es strikt).
- **Präzedenz-Flip**: bei parsebarem `source_filename` **und** mitgeschicktem Client-`recorded_at` → `recorded_at_source='filename'` (war vorher `'client'` — **bestehenden P2-Test invertieren**); explizit vorgesetztes `metadata.recorded_at` schlägt weiterhin beides; nicht-parsebarer Dateiname + Client-Wert → `'client'` (Fallback unverändert).
- Gesamt-Suite grün ≥ Baseline.

**Stop + Bericht** (Baseline-Zahl, neue/invertierte Testzahl, kurze Parser-Demo der 5 realen Namen).

## Phase 2 — Backfill-Script + Wrap

### 2a — `scripts/backfill_recorded_at.py`
- **Importiert** `parse_recorded_at_from_filename` aus `app_pkg.library` (nicht reimplementieren — Memory `reference_tag_vocab_central_gate_plus_backfill_script`).
- Scope: `Conversion`-Rows mit `conversion_type='audio_transcription'`, die **kein** `metadata.recorded_at` haben **und** deren `source_filename` einen validen Parser-Treffer liefert. Nur diese befüllen (`metadata['recorded_at']` + `recorded_at_source='filename'`), bestehende metadata-Felder erhalten.
- **`--dry-run` ist Default** (listet pro Row alt→neu); echter Lauf nur mit `--apply`. Idempotent (zweiter `--apply` = no-op, da `recorded_at` dann gesetzt). Pro Row eine structlog-/print-Zeile.
- Lokal verifizieren: ein paar synthetische `audio_transcription`-Rows mit Diktat-Namen + leerem `recorded_at` in der Mac-dev-DB anlegen (direkt, nicht über den gehärteten Pfad), dry-run → apply → zweiter apply (no-op) durchspielen. **Die 5 echten Diktate liegen auf der Mintbox** — der reale `--apply` dort ist **Olis Schritt** (Kommandos in den Bericht, Pfad auf der Mintbox verifizieren statt raten; Backup-Disziplin wie beim R2-E-Cleanup).

### 2b — Wrap
- `STATUS.md` + `BACKLOG.md`: MCP1-FIX ☑ done mit Hashes; das BACKLOG-Item **„MCP1-FOLLOWUP — recorded_at retroaktiv"** als durch dieses Script erledigt markieren (Apply = Olis Real-Welt-Schritt). Das **„MCP1-P3 browser-E2E-Smoke nach Deploy"**-Item bleibt offen (separat).
- **Memory**: deine Einschätzung — das reale Diktiergerät-Namensmuster (`YYMMDD_NNNN`, kein Jahr-4-stellig, keine Zeit) ist ein konkretes, wiederverwendbares Faktum; ein knapper `reference_*`-Eintrag (oder eine Zeile am bestehenden recorded_at-Kontext) kann sich lohnen — nicht erzwungen.
- **Bullet-Guard** vor dem Doc-Commit: `grep -nE '(- \*\*.*){2,}' BACKLOG.md STATUS.md`.
- `pytest tests/` final grün.

**Stop + Schluss-Bericht** — inkl. Olis offener Schritte: (1) Mintbox-Deploy (zieht MCP1-FIX additiv mit, **keine Migration**, `up -d --build`), (2) danach `scripts/backfill_recorded_at.py --apply` auf der Mintbox für die 5 Diktate, (3) der separate browser-E2E-Smoke.

## Out of scope
- Live-Mikro-Save-Pfad (`save-live-btn`) — hat keine Upload-Datei, kein `lastModified`/Dateiname; unverändert.
- `converter-mcp`, Notion/Kalender, Deploy-Ausführung — Top-Level-Koordinator.
- Uhrzeit-Heuristik für das Diktat-Format (es gibt keine im Namen) — 00:00 ist korrekt.
