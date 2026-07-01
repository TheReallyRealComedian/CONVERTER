# SPRINT NARR-TIMEOUT — Narration-Hang-Fix (#80): per-Call-TTS-Deadline + skalierter RQ-Job-Timeout

**Datum:** 2026-07-01 · **Size:** S/M · **Code:** NARR-TIMEOUT · **Doc-Pfad:** `docs/archive/sprint-prompts/SPRINT_NARR-TIMEOUT_narration-hang_2026-07-01.md`

---

## Problem

Faithful-Narration rendert ein Dokument via Google **Cloud** Text-to-Speech (`google-cloud-texttospeech`, Modell `gemini-2.5-flash-tts`) zu einer WAV. Ablauf: `POST /api/narrations` enqueued `tasks.generate_narration_task` auf RQ (job_timeout = `TIMEOUT_RQ_JOB_SECONDS = 600`), der RQ-Worker (`worker.py`, default `UnixSignalDeathPenalty`) fährt `services/narration_render.render_turns`, das den Transkript per `chunk_turns` (`MAX_TRANSCRIPT_BYTES=3500`) in Chunks zerlegt und pro Chunk `client.synthesize_speech` in `_synthesize_with_retry` aufruft. Transkript **#80** = 107 Turns / 23,8 KB utf-8 → **8 sequenzielle** `synthesize_speech`-Calls. Am 2026-07-01 hing #80 für immer: der Work-Horse-Prozess blieb bei ~0 % CPU am Leben (blockiert in einem Netzwerk-Call), keine WAV, RQ-Job blieb `status=started` mit `last_heartbeat = start+600s`, terminierte aber nie. **Root Cause (verifiziert):** `client.synthesize_speech` (`narration_render.py:214-216`) hat **keinen per-Call-Timeout** → ein gewedgeter gRPC-Call parkt den Horse-Main-Thread im nativen C-Core-Poll von grpc, der `EINTR` absorbiert → RQs `SIGALRM`-Death-Penalty wird ewig aufgeschoben, `JobTimeoutException` fällt nie → DB-Row bleibt `pending`. Zusätzlich ist der flache 600 s job_timeout für 8 langsame sequenzielle TTS-Calls zu knapp.

**Container-verifiziert** (`markdown-converter-worker`): `rq==2.8.0`, `redis==7.4.0`, ein einzelner Worker über `['default']` (keine `deploy.replicas`); `synthesize_speech(self, request, input, voice, audio_config, retry, timeout, metadata)` akzeptiert `timeout=`; `google.api_core.timeout.TimeToDeadlineTimeout` vorhanden; `gax.DeadlineExceeded` ist bereits in `_RETRYABLE` (`narration_render.py:59`). Beide Compose-Services `web` und `worker` laden `env_file: .env` → ein `.env`-Eintrag erreicht **beide** Container (keine Drift-Fläche).

---

## Ziel — der geschichtete Fix

- **Layer 1 (der eigentliche Fix):** absoluter per-Call-gRPC-Deadline (`timeout=` float) in den einzigen unbegrenzten SDK-Call — vom C-Core-Timer erzwungen, signal-unabhängig; auf Ablauf `DeadlineExceeded` → schon in `_RETRYABLE`, **null neuer Error-Handling-Code**.
- **Layer 2:** RQ-`job_timeout` pro Render aus der Chunk-Anzahl skaliert (statt flacher 600 s), die Hüllkurve **trackt `T`** (der per-Chunk-Envelope ist aus `T` abgeleitet mit Floor → Anheben von `T` kann einen echt-fortschreitenden Render nie mid-flight killen).
- **Layer 3 (emergent, kein Code):** weil Layer 1 die Kontrolle zwischen Attempts/Chunks an den Python-Eval-Loop zurückgibt, feuern die RQ-Death-Penalty-Backstops wieder (in-Horse `SIGALRM` am skalierten job_timeout, Parent `SIGKILL` bei `+60`) — aber nur bei einem echt-steckenden **nicht-gRPC**-Horse.

**Nicht ändern:** `google_tts_service.py`- und `tasks.py`-Signaturen bleiben stabil (der Deadline erreicht den Worker über das Modul-Level-Default-Arg, bei Import eingefroren). Kein Schema-Touch. `retry=` **nicht** übergeben (gapic-retry bleibt `None`, der Renderer besitzt das Retry).

---

## Phase 1 — Layer 1: per-Call-TTS-Deadline (`services/narration_render.py` + `app_pkg/config.py`)

### 1a — `app_pkg/config.py`: Konstanten + Formel + F-015-Kommentar-Cleanup

`config.py` ist aktuell import-frei. Füge **`import os`** und **`import math`** oben hinzu.

**Ersetze** den veralteten F-015-Kommentar (`config.py:12-17`) — er begründet die 600 s mit der entfernten „Gemini-podcast pipeline". Neu formulieren: die zentralisierten Timeouts sind (a) der per-Call-Cloud-TTS-gRPC-Deadline und (b) der pro-Render skalierte RQ-Envelope; Deepgram/Gemini-Konstanten bleiben eigenständig.

**`TIMEOUT_GEMINI_SECONDS = 300`** (genutzt von `services/gemini/client.py:33`) und **`TIMEOUT_DEEPGRAM_SECONDS = 600`** (genutzt von `services/deepgram_service.py:157`) **behalten** — nicht anfassen.

Füge folgende Bausteine hinzu (Werte exakt so):

- Guarded env-Parser `_env_positive_float(name, default)`: `os.environ.get(name)` → `None` gibt `float(default)`; `float(raw)` in try/except `(TypeError, ValueError)` → auf Fehler `float(default)`; Rückgabe `val if val > 0 else float(default)` (≤0/Junk → sicherer Default). **Grund (adversarial #7):** ein malformter Wert darf beide Container beim Boot nicht bricken.
- `TIMEOUT_TTS_SYNTH_SECONDS = _env_positive_float('NARRATION_TTS_TIMEOUT_SECONDS', 120.0)` — Layer-1-per-Call-Deadline.
- `_TTS_MAX_RETRIES = 2` (→ 3 Attempts total) — **muss** `narration_render._synthesize_with_retry` spiegeln.
- `_TTS_RETRY_BACKOFF_TOTAL = 3` (`sleep(1)+sleep(2)` zwischen den 3 Attempts).
- `TIMEOUT_RQ_JOB_BASE_SECONDS = 200` (SDK-Init + WAV-Concat + `shutil.move`-Headroom).
- `_RQ_PER_CHUNK_FLOOR = 400` (hält Default `T=120` → `n=1 == 600`, verhaltensneutral).
- `TIMEOUT_RQ_JOB_PER_CHUNK_SECONDS = max(_RQ_PER_CHUNK_FLOOR, math.ceil((_TTS_MAX_RETRIES + 1) * TIMEOUT_TTS_SYNTH_SECONDS + _TTS_RETRY_BACKOFF_TOTAL))`.
- `TIMEOUT_RQ_JOB_HARD_CAP = 4 * 3600` (Backstop, beißt nur bei pathologisch großem `n`, ~`n≳36`).
- `rq_job_timeout_for(n)`: `scaled = TIMEOUT_RQ_JOB_BASE_SECONDS + TIMEOUT_RQ_JOB_PER_CHUNK_SECONDS * max(n, 0)`; `return min(scaled, TIMEOUT_RQ_JOB_HARD_CAP)`.
- `TIMEOUT_RQ_JOB_SECONDS = rq_job_timeout_for(1)` — Back-Compat-Export (**== 600** beim Default, damit bestehende Importe + `test_narration_write.py:16` weiter auflösen).

**Formel:** `job_timeout = min(200 + PER_CHUNK·n, 14400)`, `n = len(chunk_turns(turns))`, `PER_CHUNK = max(400, ceil(3·T + 3))`. Default `T=120` → `PER_CHUNK=400`; `n=1→600` (exakt der alte flache Wert), `n=8` (#80) → 3400 s. Override `T=200` → `PER_CHUNK=603`; `n=8→5024 s`.

### 1b — `services/narration_render.py`: Deadline durchfädeln

- **`_synthesize_with_retry(...)`** (`~199-216`): ein keyword-only `timeout=None` ergänzen; der Call wird `client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config, timeout=timeout)`. Sonst nichts ändern (Retry-Loop, `_RETRYABLE`, Backoff bleiben).
- **`render_turns(...)`** (`~274`): keyword-only `synth_timeout` mit Default `TIMEOUT_TTS_SYNTH_SECONDS` ergänzen. Am `_synthesize_with_retry`-Call (`~343`) `timeout=synth_timeout` forwarden.
- **Keine** Änderung an `services/google_tts_service.py` oder `tasks.py` (Signaturen bewusst stabil — `synth_timeout` wird NICHT durch `tasks.py`/`google_tts_service.py` gefädelt).

**Zirkular-Import-Check (vorab verifiziert, unkritisch):** `app_pkg/__init__.py` registriert beim Package-Import **keine** Route-Module (Routes werden erst in `app.py` verdrahtet), und `services/gemini/client.py`, `services/narration_library.py`, `services/deepgram_service.py` importieren bereits `from app_pkg.config import …`. `services/narration_render.py` darf `from app_pkg.config import TIMEOUT_TTS_SYNTH_SECONDS` also gefahrlos ergänzen — kein Zyklus, etablierter Pattern. (Der Wert lebt damit an **einer** Quelle: config berechnet die job_timeout-Formel daraus, der Renderer erzwingt denselben Wert.)

**Begründung:** `api_core` wandelt den float via `TimeToDeadlineTimeout` in einen **absoluten** gRPC-Deadline, den grpcs C-Core-Timer erzwingt (bindet den ganzen RPC-Lifecycle: DNS+TCP+TLS+Response), signal-unabhängig. Auf Ablauf → `StatusCode.DEADLINE_EXCEEDED` → `google.api_core.exceptions.DeadlineExceeded` → **bereits in `_RETRYABLE`** → gefangen/retried/re-raised, null neuer Code.

**STOP + Bericht.** Zeige das config.py-Diff (Konstanten + Formel + neuer Kommentar) und das narration_render.py-Diff. Rechne kurz vor: `rq_job_timeout_for(1)==600`, `rq_job_timeout_for(8)==3400`, `rq_job_timeout_for(0)==200`. Warte auf Sign-off.

---

## Phase 2 — Layer 2: skalierter RQ-`job_timeout` an beiden Enqueue-Sites (`app_pkg/narration.py`)

- **Import (Zeile 24):** den jetzt toten `TIMEOUT_RQ_JOB_SECONDS` **droppen** (adversarial #5 — nach dieser Änderung nicht mehr referenziert), stattdessen `rq_job_timeout_for` importieren. Also `from app_pkg.config import OUTPUT_DIR, rq_job_timeout_for`.
- **Import (Zeile 42):** `chunk_turns` zum `services.narration_render`-Import ergänzen (neben `DEFAULT_NARRATION_MODEL, validate_turns`).
- **Defensiver Modul-Helper** ergänzen:

  ```python
  def _job_timeout_for_turns(turns):
      try:
          return rq_job_timeout_for(len(chunk_turns(turns)))
      except Exception:
          return rq_job_timeout_for(0)
  ```

  **Grund (adversarial #8):** der Retry-Pfad zieht `turns` direkt aus `metadata` ohne Re-Validierung; ein korruptes Legacy-Transkript muss **innerhalb des Tasks** scheitern, nicht den Request 500en. Fallback `rq_job_timeout_for(0)=200` ist harmlos (Render scheitert eh schnell auf leeren Turns).

- **Beide Enqueue-Sites** (create `~256`, retry `~376`): den bisherigen `job_timeout=TIMEOUT_RQ_JOB_SECONDS` durch `job_timeout=_job_timeout_for_turns(turns)` ersetzen. `turns` ist an beiden Sites web-seitig verfügbar (create: aus dem validierten Body; retry: aus `metadata`).

**STOP + Bericht.** Zeige das narration.py-Diff, bestätige beide Enqueue-Sites getroffen und den toten Import entfernt. Warte auf Sign-off.

---

## Phase 3 — Tests (`pytest tests/` muss grün bleiben) + Verifikations-Hinweis

### `tests/test_narration_render.py`
Der `_call`-Helper (`l.58-61`) liest nur `input`/`voice`/`audio_config`; kein Test macht Full-Kwargs-Equality → `timeout=` bricht nichts. **Neu ergänzen:**
- (a) **Deadline-Forwarding:** `render_turns` forwardet den Deadline — assert `client.synthesize_speech.call_args.kwargs['timeout'] == TIMEOUT_TTS_SYNTH_SECONDS` beim Default, **und** `== 99` wenn `synth_timeout=99` an `render_turns` übergeben wird.
- (b) **DeadlineExceeded retried-dann-re-raised:** `gax.DeadlineExceeded` als `side_effect` (patch `time.sleep`); assert `call_count == 3` und `pytest.raises(gax.DeadlineExceeded)` — spiegelt `test_render_resource_exhausted_retries_then_succeeds` (`~389`). Plus eine `DeadlineExceeded`-dann-Success-Variante (One-off-Resilienz: erste Timeout-Attempt, dann Erfolg).

### `tests/test_narration_write.py:173`
**Bleibt grün ohne Edit** (adversarial #4, bestätigt): `_TURNS` = 2 winzige Turns → 1 Chunk → `rq_job_timeout_for(1) == TIMEOUT_RQ_JOB_SECONDS == 600`; der Import (`l.16`) löst weiter auf. *Optionale* Klarheits-Verschärfung (nicht erforderlich): assert `== rq_job_timeout_for(len(chunk_turns(payload_turns)))` (beide importieren), damit die Invariante explizit statt tautologisch ist.

### `tests/test_narration_retry.py`
Enqueue-Assert prüft `job_timeout` nicht → grün (optional den Write-Assert spiegeln).

### `tests/test_narration_task.py:71-73`
`assert_called_once_with` bleibt grün, **genau weil** `synth_timeout` NICHT durch `tasks.py`/`google_tts_service.py` gefädelt wird. Nicht anfassen.

### Backward-Compat
Kein Schema-Touch. Pre-NARR-5-Retry-Rows: `chunk_turns([]) → [] → len 0 → rq_job_timeout_for(0)=200` (harmlos). Erfolgreiche Renders + bestehende `ResourceExhausted`/`ServiceUnavailable`-Semantik verhaltensneutral — nur `DeadlineExceeded` entsteht jetzt tatsächlich.

### Verifikations-Hinweis (im Bericht festhalten)
Tests mocken `synthesize_speech` und installieren **nie** einen echten gRPC-Deadline → sie fangen einen echten Hang **nicht**. Der reale Beweis ist ein **Container-Live-Smoke** (Phase 4).

**STOP + Bericht.** `pytest tests/` grün zeigen (~37 Tests). Warte auf Sign-off.

---

## Phase 4 — Live-Smoke + Kalibrierung + Abschluss (Pflicht, nicht optional)

Auf der Mintbox: `git pull` von origin + `docker compose up -d --build` (Templates/Code ins Image gebacken → `--build`).

1. **Healthy-Baseline:** einen langen (≥8-Chunk) Transkript rendern, reale per-Chunk-`gemini-2.5-flash-tts`-Latenz messen.
2. **Down-Backend-Beweis:** die TTS- (und Token-)Endpoint bei einem **ersten** Render blackholen/DNS-failen → bestätigen: der Call kehrt innerhalb ~`T` mit `DeadlineExceeded` zurück, der Task raised, `render_turns` bricht **beim ersten scheiternden Chunk ab** (re-raise nach 3 Attempts, `narration_render.py:337-355` räumt partielle WAVs auf), RQ markiert `failed`, `reconcile_narration` flippt die Row beim nächsten Poll `pending→failed` → Library zeigt Fehler + „Erneut versuchen". Erwartung: **#80s ewiger Hang wird ein ~6-Minuten-sauberer, retrybarer Fehler** (~363 s für einen Chunk statt endlos).
   - **Forensik-Notiz (offener Punkt):** die Memory-Note beobachtete den Work-Horse #80 noch ~2 h nach Start am Leben — das widerspricht RQs Parent-`SIGKILL`-bei-`job_timeout+60`-Backstop (der bei ~660 s hätte reapen müssen). Im Down-Backend-Smoke **explizit prüfen**, ob der Horse jetzt sauber reaped wird (`docker exec markdown-converter-worker ps -ef` nach Ablauf). Falls NICHT: der Parent-Monitor ist separat gewedged — als **Folge-Finding** festhalten (Layer 1 umgeht das Problem, aber es wäre ein tieferes RQ-Thema für einen eigenen Sprint). Alte Worker-Logs von #80 sind seit dem Container-Restart (2026-07-01) vermutlich weg.
3. **Kalibrierung:** aus der gemessenen Healthy-Latenz `NARRATION_TTS_TIMEOUT_SECONDS` in der **geteilten `.env`** pinnen (120 s ist ein Kalibrier-Ballpark) — ein Eintrag konfiguriert Worker (Enforcement) **und** Web (Envelope) gemeinsam, kein Drift.
4. **#80 als Live-Long-Render-Smoke:** Job #80 (aktuell `failed`/retrybar) frisch retrien und einen **gesunden** Durchlauf bis `ready` beobachten (skaliertes job_timeout = 3400 s bei n=8, healthy ≈ 240–320 s ≪ 3400).

**Deferred (nur falls #6 im Smoke auftritt):** wenn ein Connect/Auth-Stall den Deadline überschreitet, in einem Folge-Sprint `google-auth`-Request-Timeout/keepalive als zweite Schicht — **jetzt nicht** einbauen.

**Abschluss:**
- `STATUS.md` aktualisieren (NARR-TIMEOUT abgeschlossen, #80-Hang gefixt).
- `BACKLOG.md`: das #80-Hang-Item schließen (Memory `project_narration_80_stuck_2026-07-01.md` als gelöst annotieren).
- Commit + Push (Single-User-Repo, Push ist normal). Falls auf `main`: vorher branchen.

**STOP + Abschlussbericht** mit Smoke-Ergebnissen (Baseline-Latenz, Down-Backend-Fail-Zeit, Horse-Reaping-Befund, #80-Erfolg) + gepinntem `NARRATION_TTS_TIMEOUT_SECONDS`-Wert.

---

## Retry-/Timeout-Mathematik (Referenz)

- **Per Call:** hart auf `T` (Default 120 s) durch den absoluten gRPC-Deadline (deckt connect/DNS/TLS + Response).
- **Per Chunk:** `range(max_retries+1)=3` Attempts, Backoff `sleep(1)+sleep(2)=3 s` → worst `= 3·T+3 = 363 s`. Fixe 3-Attempt-Zahl = **kein Retry-Storm**.
- **Down-Backend (der Bug):** `render_turns` bricht beim ersten scheiternden Chunk ab → persistent-gewedgeter Endpoint failt bei **~363 s** (ein Chunk), nicht endlos.
- **Slow-but-completing n-Chunk:** jeder erfolgreiche Chunk ebenfalls durch `3·T+3` begrenzt → total ≤ `n·(3T+3) + fixed_overhead`; `job_timeout = 200 + PER_CHUNK·n` dominiert es per `BASE=200`-Headroom (`PER_CHUNK ≥ 3T+3`) → ein echt-fortschreitender Render wird **nie** false-gekillt.
- **Envelope-Kohärenz:** `PER_CHUNK ≥ (max_retries+1)·T + backoff` gilt für **alle** `T` (per `max(...)`-Konstruktion). In-Horse-`SIGALRM` am skalierten job_timeout, Parent-`SIGKILL` bei `+60` — beide dominieren den Burn, feuern nur bei echt-steckendem nicht-gRPC-Horse.

## Adversarial-Dispositionen (schon gefaltet, nicht neu diskutieren)
#1 validiert · #2 gefaltet (`PER_CHUNK=max(400,ceil(3T+3))` + shared `.env`) · #3 teils abgelehnt (early-abort existiert schon in `render_turns`), teils gefaltet (`HARD_CAP`) · #4 Test nicht gebrochen · #5 toter Import gedroppt · #6 Live-Smoke-Schritt · #7 `_env_positive_float`-Guard · #8 `_job_timeout_for_turns` try/except.
