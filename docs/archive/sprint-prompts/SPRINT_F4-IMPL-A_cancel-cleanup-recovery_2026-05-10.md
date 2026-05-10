# Sprint F4-IMPL-A — F-4 Implementation Cluster I (Cancel + Cleanup-Recovery)

**Datum**: 2026-05-10

**Ziel**: Cluster I aus F-4.3 implementieren — die zwei Patterns die die zentrale F-4-Schmerzpunkte adressieren: **P1 (Cancel-Mechanik mit Worker-Stop + Frontend-Lüge-Fix + Confirmation-Dialog)** und **P2 (orphaned-File-Cleanup)**. **Höchstes Sprint-Risiko der gesamten F-4-Welle**: Sev-4-Findings F1+F2 mit L-Aufwand auf P1, RQ-cooperative-cancel-Recherche-Anteil, Async-Pipeline-Touch, Worker-Code-Pfad. **Bewusst aus F4-IMPL-B isoliert**, damit dieser Sprint volle Konzentration bekommt.

**Vorbedingung**:
- Pytest 51/51 grün auf `main`. Letzter Code-Touch: F4-PATTERNS (commit `d25d4ef`, 2026-05-10).
- **Eingabe**: [docs/ui_patterns_podcast_flow_2026-05.md](docs/ui_patterns_podcast_flow_2026-05.md) (Sub-Thread liest **vor Phase 1** mindestens P1, P2, plus Master-Annotation, plus Architektur-Hebel-Sektion, plus die Helper-Vorschlag-Sektion am Doc-Ende).
- **Patterns dieses Sprints**:
  - **P1 (Cancel-Mechanik)** — Sev 4, L-Aufwand. `✅ live-verifiziert` durch Master-Walkthrough 2026-05-10 (siehe Patterns-Doc Master-Annotation):
    - Job 9bf48e0a-... lief 1:12 Min nach Cancel-Klick weiter und schrieb 11.7 MB WAV.
    - Adressiert F1 (Worker-Stop-Lüge Sev 4), F2 (Frontend-Lüge Sev 4), BT1 (verknüpfter Bug).
    - **Drei verbindliche Sub-Mechaniken** aus Master-Annotation:
      1. Cancel-Button muss **echten Worker-Stop** triggern.
      2. Frontend-Lüge entfernen (Confirmation-Dialog + „wird abgebrochen"-Zwischenstate + Worker-Confirmed-Terminal).
      3. (verzahnt mit P2) Cleanup für mid-cancel-WAV-Files.
  - **P2 (orphaned-File-Cleanup)** — Sev 3, S-Aufwand. `✅ live-verifiziert` durch Master-Walkthrough 2026-05-10:
    - Test-Lauf zeigte `tmp_rgn3y1o.wav` (11.7 MB) verbleibt im Volume nach abgebrochenem Job.
    - Adressiert F4 (File-Cleanup-vs-Re-Download Sev 3), BT2 (verknüpfter Bug).
- **Worker-Code-Pfade**:
  - [worker.py](worker.py) — RQ-Worker-Bootstrap mit `connection=conn`-Idiom aus CVE-RQ.
  - [tasks.py](tasks.py) — `generate_podcast_task` (RQ-Background-Job).
  - [services/gemini/synthesis.py](services/gemini/synthesis.py) — TTS-Pipeline-Schritte (relevant für cooperative cancel checks zwischen Chunks).
  - [app_pkg/podcasts.py](app_pkg/podcasts.py) — Routes inkl. `/podcast-status/<job_id>` (F-001-narrow-except-Pattern), `/podcast-download/<job_id>` (Cleanup-on-Download mit `Path.is_relative_to`-Schutz aus SEC).
- **Frontend-Code-Pfade**:
  - [static/js/audio_converter.js](static/js/audio_converter.js) — Podcast-Tab-Funktionen, Generate-Klick, Status-Polling-Loop, Cancel-Button-Handler (P18 aus F-2 Cluster II — wird hier substantiell überarbeitet).
  - [templates/audio_converter.html](templates/audio_converter.html) — Podcast-Pane, Cancel-Button, Status-Display.
- **Memory-Layer-Pflicht-Lese**:
  - `reference_converter_dep_bump_constraints.md` — rq-2.x-Removals + JobStatus-StrEnum-Verhalten + Container-pytest.
  - `feedback_no_silent_fixes.md` — Bugs während Implementation als separates Bug-Ticket dokumentieren.
- **Methodik-Vorlagen**:
  - F-3-IMPL Cluster A (Foundation + Silent-Failure) als Mechanik-Referenz für Failure-Banner-Patterns (commits `843574b` / `40dd02e` / `b3e666a`).
  - F-2 Cluster I commit `ef78508` für Multi-Pattern-Holistic-Rewrite (auch wenn hier nur 2 Patterns).

**Out-of-scope**:
- **Cluster II + III** (P3–P12) — eigener Folge-Sprint `F4-IMPL-B`.
- **BT4** (pure Bug-Ticket Blob-URL-Revoke) — separat, aber wenn der Cancel-Pattern den Audio-Player-Code touchiert: kurz im Bericht erwähnen ob BT4 nahegelegen mit-fixbar wäre.
- WAVE-CLOSE.

---

## Phase 1 — Implementation

Pre-Flight (vor jedem Code-Touch):

1. `pytest tests/` — muss 51/51 grün sein im Container (`docker exec markdown-converter-web python -m pytest tests/`).
2. `git status -s` → clean tree erwartet.
3. **P1 + P2 + Master-Annotation komplett lesen** in `docs/ui_patterns_podcast_flow_2026-05.md`.
4. **`_utils.js`-Helper-Bestand verifizieren**.

### RQ-Cooperative-Cancel-Vorab-Recherche (Pflicht-Schritt für P1)

**Erwartung gemäß Memory + Patterns-Doc**: rq 2.8.0 hat `Job.cancel()` und `send_stop_job_command(redis, job_id)`. Die Frage ist welche Wirkung das auf einen `started`-Job hat (RQ stoppt nicht mid-execution out-of-the-box — der Task muss kooperieren).

Sub-Thread macht:

1. **rq-2.8.0-Source-Reading** (PyPI-Tarball oder GitHub-Repo): wie funktioniert `send_stop_job_command`? Sendet es `SIGINT` an Worker oder ist es nur ein Status-Flip in Redis?
2. **`generate_podcast_task` in `tasks.py` lesen**: gibt es bereits Stop-Check-Punkte zwischen TTS-Chunks? Falls nein: das ist Teil der P1-Implementierung.
3. **Drei Implementations-Optionen** evaluieren (in Reihenfolge der Bevorzugung):
   - **Option A (Cooperative Cancel via Redis-Key-Poll)**: vor jedem TTS-Chunk-Call `if redis.exists(f"cancel:{job_id}"): raise CancelledError`. Beim Cancel-Backend-Pfad: `redis.set(f"cancel:{job_id}", "1", ex=3600)` plus `Job.cancel()`. Saubere User-Code-Logik, kein RQ-Internals-Eingriff.
   - **Option B (rq native send_stop_job_command)**: `from rq.command import send_stop_job_command; send_stop_job_command(redis_conn, job_id)`. RQ-eigene Mechanik. Vor-Recherche zeigt ob das mid-execution-stop tatsächlich liefert.
   - **Option C (Worker.kill_horse / SIGINT)**: niedrige Granularität (stoppt ganzen Worker, nicht nur den Job). Wahrscheinlich zu aggressiv.

   Default-Empfehlung: **Option A** wenn rq-Source-Reading zeigt dass `send_stop_job_command` nur Status-Flip macht ohne Worker-Interrupt. Option B wenn rq tatsächlich SIGINT/SIGTERM-Mechanik liefert.

Nach Vorab-Recherche: STOP, Master fragen — welche Option, wenn nicht klar Default-A.

### P1 — Cancel-Mechanik (Worker-Stop + Frontend-Lüge-Fix + Confirmation)

**Backend-Pfade**:

1. **Cancel-Endpoint** (neu oder erweitert): z.B. `POST /podcast-cancel/<job_id>`. Setzt Redis-Key `cancel:{job_id}`, ruft `Job.fetch(job_id, connection=conn).cancel()`, returned Bestätigung. Gate mit `@require_service('gemini')` falls existing.
2. **`generate_podcast_task` in `tasks.py`**: Cooperative-Stop-Checks zwischen TTS-Chunks. Bei Erkennung des Cancel-Keys: `CancelledError` raisen, Cleanup laufen lassen (mid-write-WAV löschen, siehe P2-Verzahnung), Job-Status zu `cancelled` setzen.
3. **Job-Meta-Feld `meta['cancelled_at']`** für Frontend-Polling-Sichtbarkeit.

**Frontend-Pfade** (in `audio_converter.js` Podcast-Block):

1. **Cancel-Button-Handler überarbeiten**:
   - Confirmation-Dialog (DE-Microcopy aus Patterns-Doc): „Generierung abbrechen? Bisher verbrauchte TTS-Token können nicht zurückgeholt werden."
   - Bei Bestätigung: `POST /podcast-cancel/<job_id>`, dann **NICHT** Status sofort flippen. Stattdessen Polling-Loop läuft weiter mit Zwischenstate „Wird abgebrochen …".
2. **Status-Polling-Loop**: erkennt `cancelled` (oder `failed` mit cancel-Marker) als Terminal-State, dann erst Frontend-Status auf „abgebrochen" setzen + Cleanup-UI (Reset Generate-Button etc.).
3. **Confirmation-Dialog-Helper**: `confirmInPlace(triggerEl, options)` aus Helper-Vorschlag der Patterns-Doc evaluieren. Wenn Cross-Feature-Wert (z.B. P3 Delete-Confirmation in F-3-IMPL hatte ähnliche Logik): in `_utils.js` anlegen. Wenn nur diese eine Stelle: lokal in audio_converter.js.

### P2 — Orphaned-File-Cleanup

**Backend-Pfade**:

1. **Mid-cancel-Cleanup**: in `generate_podcast_task` `CancelledError`-Handler — wenn WAV teilweise oder ganz geschrieben ist: `os.remove(...)` mit `Path.is_relative_to(OUTPUT_DIR)`-Schutz.
2. **Periodischer Cleanup** (optional, Architektur-Entscheidung): cron-Style „lösche WAVs älter als X Stunden ohne Download-Trigger". Kann RQ-Periodic-Task oder einfacher Cleanup-on-`/podcast-download`-Pfad sein. **Default-Empfehlung**: Cleanup-on-Cancel reicht für jetzt; periodischer Cleanup ist eigene Hygiene-Welle wenn das Problem weiterbesteht.

**Test-Erwartung**:

- 1-2 neue Tests für P1: Cancel-Endpoint-Contract, JobStatus-Übergang.
- 1 Test für P2: WAV-Cleanup-Pfad bei Cancel.
- Erwartete Final-Test-Anzahl: **52-54**.

**Mechanik-Leitplanken**:

- **DE-Microcopy** für Confirmation + Status-Strings (Fehler ≤2 Sätze, Buttons ≤3 Wörter, keine Emojis bei Fehlern).
- **Helper-Reuse**: `showAlert` für Cancel-Failure-Banner, `showToast` für Cancel-Success. `confirmInPlace` neu (siehe oben).
- **F-001-Pattern beachten**: `/podcast-status/<job_id>` darf bei Redis-Down weiter 500 (nicht 404) liefern. Cancel-Endpoint analog narrow-except.

**Erwartete Files**:

```
worker.py                              # eventuell EDIT — Stop-Check-Mechanik falls Worker-Level nötig
tasks.py                               # EDIT — Cooperative-Cancel-Checks + mid-write-WAV-Cleanup (P1+P2 Backend)
app_pkg/podcasts.py                    # EDIT — neuer Cancel-Endpoint (P1)
static/js/audio_converter.js           # EDIT — Podcast-Cancel-Handler-Refactor (P1 Frontend)
static/js/_utils.js                    # eventuell EDIT — confirmInPlace-Helper falls Cross-Feature
templates/audio_converter.html         # eventuell EDIT — Confirmation-Dialog-Mountpoint, Status-Display
tests/test_podcasts.py                 # EDIT — 2-3 neue Tests
```

Nach Phase 1: STOP — Bericht. Welche RQ-Option gewählt, ob `confirmInPlace` in _utils.js oder lokal, neue Test-Anzahl, ob Worker-Stop tatsächlich greift (Code-Evidenz für Cancel-Pfad), Mid-Sprint-Drifts.

---

## Phase 2 — Verify

1. `docker exec markdown-converter-web python -m pytest tests/` final grün (52-54 erwartet).
2. **Live-Smoke Cancel-Disk-Forensik (Wiederholung des Master-Walkthroughs vom 2026-05-10)** — diesmal mit dem Patch:
   - Podcast-Generation starten (kleines Skript, monolog).
   - Sobald Worker `started` zeigt: Cancel-Button klicken.
   - Confirmation-Dialog erscheint → bestätigen.
   - **Frontend-Verhalten**: Status zeigt „Wird abgebrochen …" während Worker stoppt.
   - **Worker-Verhalten**: `docker logs markdown-converter-worker` zeigt Cancel-Erkennung, kein „Successfully completed" für diesen Job, sondern „cancelled" oder Exception-Pfad.
   - **Disk-Verhalten**: keine WAV-Datei in `/var/lib/docker/volumes/converter_podcast_data/_data/` nach Cancel.
   - **Frontend-Final-Status**: „Abgebrochen" sobald Worker bestätigt.
3. **Vergleich-Test (positiver Pfad)**: zweiten Job ohne Cancel laufen lassen → muss weiter funktionieren wie vor dem Sprint (nicht regressed).
4. Server-Logs auf neue Warnings/Errors aus dem Cancel-Pfad checken.

Nach Phase 2: STOP — Bericht. Live-Smoke-Resultate inkl. Cancel-Disk-Forensik-Vergleich „vor Sprint vs. nach Sprint", Test-Stand, etwaige Drift-Befunde.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Default: **ein Commit für F4-IMPL-A** (P1+P2 thematisch eng verzahnt, atomic-Backend-und-Frontend-Refactor). Falls der Sub-Thread Backend und Frontend separat committet: zwei Commits, in derselben Branch.
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints. Wenn Auto-Mode-Classifier blockt: Master pushed von Hand.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

**Zusätzlich für F4-IMPL-A** (höchstes Sprint-Risiko der F-4-Welle):
- Bei rotem Test-Run im Container: STOP, nicht weiter zu nächstem Pattern.
- Bei Worker-Boot-Loop nach Refactor: STOP — `tasks.py`-Änderung könnte Import-Fehler oder Signature-Bruch verursachen.
- Bei Live-Smoke Cancel-Disk-Forensik mit erneutem WAV-Schreib-Verhalten (wie vor Patch): STOP — Cancel-Mechanik wirkt nicht, Master entscheidet ob andere RQ-Option oder Re-Skopung.
- Bei RQ-Vorab-Recherche-Befund dass **keine** der drei Optionen sauber funktioniert: STOP — Master entscheidet ob Sprint-Re-Skopung oder Plan-D (Frontend-only-Lie-Fix ohne echten Worker-Stop, mit explizitem User-Hinweis „Worker läuft noch X Sekunden weiter").

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M-L** — zwei Patterns aber L-Aufwand auf P1 (Worker-Stop-Mechanik + Frontend-Lüge-Fix + RQ-Recherche). P2 ist S-Aufwand. Insgesamt mittelgroßer Sprint mit hohem Risiko-Profil. Bewusst aus F4-IMPL-B isoliert.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Backend-Refactor die F-001-Narrow-Except-Pattern für den neuen Cancel-Endpoint sauber angewandt werden müssen: dokumentieren, ist Pattern-Treue.
- Wenn Worker-Logger-Lines aus `services/gemini/synthesis.py` für die Cooperative-Cancel-Checks „in der Nähe" sind (zwischen TTS-Chunks): notieren ob das später für Cluster 2 (P3 job.meta-Stage-Progress in F4-IMPL-B) wiederverwendbar ist.
- Wenn beim Frontend-Refactor Inline-Status-Display-Code auffällt der besser als Helper wäre (z.B. `setPollingState(state, label)`): kurz im Bericht aufzählen — F4-IMPL-B kann es fold-en.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F4-IMPL-A ☑ done 2026-05-XX → commit `<hash>`. P1+P2 implementiert. Pytest <neue Anzahl>/<neue Anzahl> grün im Container. Live-Smoke Cancel-Disk-Forensik bestätigt: Worker stoppt nach Cancel, kein orphaned WAV-File. RQ-Cancel-Option: <A/B/C>. confirmInPlace-Helper: <_utils.js / lokal>. Verbleibende Sequenz: F4-IMPL-B → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F4-IMPL" raus → ersetzt durch zwei Erledigt-Einträge (F4-IMPL-A done) + neue Position 1 für F4-IMPL-B mit Pattern-IDs P3-P12 als Hint.
- **Memory**: ggf. übertragbare RQ-Cancel-Mechanik-Lehre erweitern in `reference_converter_dep_bump_constraints.md` (z.B. „rq 2.x cooperative-cancel via Redis-Key-Poll funktioniert wenn Worker-Tasks Stop-Checks haben — gilt für künftige Long-Running-Job-Features").

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — RQ-Vorab-Recherche ist Phase-1-Pflicht-Schritt mit explizitem STOP nach Recherche, nicht Master-Workshop. Master-Walkthrough-Annotation hat die drei Sub-Mechaniken vorab spec'd.)_
