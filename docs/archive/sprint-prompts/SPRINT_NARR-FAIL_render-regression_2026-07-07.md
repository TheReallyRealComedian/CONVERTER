# Sprint NARR-FAIL — Narration-Render schlägt seit ~07-07 durchgängig fehl (S/M, 3 Phasen)

> **Executor-Doc.** Nach jeder Phase **Stop + Bericht**, auf Sign-off warten. **Phase 1 ist ein hartes Diagnose-Gate** — der eigentliche Fehler ist im gespeicherten `metadata.error` **abgeschnitten** (`[:500]`), die Fix-Richtung hängt an der echten Exception-Klasse; NICHT vorher raten/fixen. Pre-Flight: `pytest tests/` grün (Baseline aus STATUS). Arbeitsverzeichnis Mac `/Users/olivergluth/CODE/CONVERTER` = Source-of-Truth. **P1 braucht Mintbox-Shell** (Worker-Logs/Container-`pip freeze`) — läuft entweder als Mintbox-dispatchter Lauf oder Oli führt die P1-Kommandos aus und liefert den Output. Code-Edits (P2) passieren auf dem Mac, Deploy dann Mintbox `git pull` + `up -d --build`.
>
> **Symptom (2026-07-07)**: Narrationen 99–102 (`audio_narration`, `two_speaker`, `voices {"Nora":"Kore","Timo":"Puck"}`, `tts_model=gemini-2.5-flash-tts`) gehen alle auf `failed`. Auch ein 6-Turn-Minimalfall (102) scheitert → **nicht** input-größen-abhängig. Der gespeicherte Fehler ist bei allen identisch und **bei ~500 Zeichen abgeschnitten**, endet im Frame `google/api_core/grpc_helpers.py:55 error_remapped_callable` → die echte Exception-Klasse + gRPC-Status liegen **hinter** dem Abschnitt.

## Master-gegroundete Fakten (nicht neu erheben — darauf aufsetzen)

- **Es ist eine REGRESSION, kein „never worked".** Der Cloud-TTS-Pfad hat **end-to-end funktioniert**: Narration **77** wurde am **2026-06-30** `ready` in ~64 s gerendert (NARR-4-Agent-Beweis, BACKLOG), NARR-1B **live-verifiziert 2026-06-29**. Jobs 99–102 sind von **07-07**. Was zwischen 06-29/06-30 und 07-07 rein-deployt wurde: **NARR-TIMEOUT** (07-01), **DIARIZE** (07-02), **IMG-SLIM** (06-30) — jedes davon per `up -d --build` **neu gebaut**.
- **Architektur = Google _Cloud_ TTS, NICHT genai/Vertex.** [services/narration_render.py](services/narration_render.py) fährt `google.cloud.texttospeech` (`client.synthesize_speech`), **nicht** `google.genai`. Der gRPC-Stack im Traceback (`google/api_core/grpc_helpers.py`) ist **auch** der Cloud-TTS-Pfad — „gRPC ⇒ Vertex" ist eine **Fehl-Schlussfolgerung**.
- **Modell-ID `gemini-2.5-flash-tts` ist KORREKT** für den Cloud-Pfad (CLAUDE.md „Gemini Models"; **live-verifiziert 2026-06-29**: der Cloud-Name hat **kein** `-preview-`-Infix, der genai-Name `gemini-2.5-flash-preview-tts` **500t** auf dem Cloud-Endpoint). ⚠️ **NICHT** auf `-preview-tts` „korrigieren" — das bräche die verifiziert-funktionierende Config.
- **PCM→WAV ist bereits gelöst.** `pcm_to_wav_bytes` ist header-agnostisch (RIFF-Passthrough + `wave`-Wrap). Kein WAV-Bug — 77 hat gültiges Audio produziert.
- **Der echte Fehler ist bereits da (ohne Code-Change):** [tasks.py:88](tasks.py) loggt auf dem Fehlerpfad `logger.error(f"Error: {type(e).__name__}: {str(e)}")` → **Worker-Container-Logs enthalten Typ + Message** für 99–102. Zusätzlich steckt der **volle** Traceback in `job.exc_info` in Redis (FailedJobRegistry).
- **Truncation-Stelle (bestätigt):** [app_pkg/narration.py:138](app_pkg/narration.py) `reconcile_narration` → `error = (job.exc_info or '')[:500]`. Das ist die einzige Kappung; sie hat diese Diagnose blind gemacht.
- **`updated_at` ≠ „Dauer bis Fehler".** Es ist der Poll-Zeitpunkt, an dem `reconcile_narration` `pending`→`failed` geflippt hat, nicht der Render-Fehler-Zeitpunkt. Die Timing-Tabelle im Ur-Dossier (99/100/102 ≈ 5 s, 101 ≈ 134 s) ist damit **kein** verlässliches Signal — nicht darauf theoretisieren.
- **SDK-Pinning-Risiko (starker Regressions-Verdacht):** [requirements.txt](requirements.txt) hat `google-cloud-texttospeech>=2.31.0` und `google-genai>=1.0.0` **nur mit Floor**; `grpcio`/`protobuf`/`google-api-core` sind **transitiv, ungepinnt**. Ein `up -d --build` nach 06-29 kann eine **neuere** Version gezogen haben → klassisches „lief letzte Woche, heute kaputt". Das ist der **Lead-Verdacht**, aber die echte Exception aus P1 entscheidet.

## Phase 1 — Diagnose (HART-Gate, kein Code-Change)

**Ziel: die echte Exception-Klasse + gRPC-Status + die live-laufenden google/grpc-Versionen.** Auf der Mintbox (Container laufen dort):

1. **Worker-Log-Fehler** (Typ + Message, schon geloggt):
   ```bash
   docker logs markdown-converter-worker 2>&1 | grep -A3 "NARRATION TASK FAILED" | tail -40
   ```
2. **Voller Traceback aus Redis** (die abgeschnittene Hälfte), für einen der Job-IDs (99=`bc71bf2d-…`, 101=`4320089c-…`, 102=`8f02100d-…`):
   ```bash
   docker exec markdown-converter-worker python -c "import os,redis;from rq.job import Job;c=redis.from_url(os.environ['REDIS_URL']);j=Job.fetch('8f02100d-704c-4c61-8a68-84d1588d6e81',connection=c);print(j.exc_info)"
   ```
   (falls der Job aus Redis evicted ist → Log-Weg aus 1. reicht.)
3. **Live-Versionen des google/grpc-Stacks** (gegen den 06-29-Zustand — Bump = Regressions-Beweis):
   ```bash
   docker exec markdown-converter-worker pip freeze | grep -iE "google|grpc|proto"
   ```
4. **Optional, falls Log+Redis nichts hergeben:** ein Minimal-Repro **im Worker-Container** (umgeht MCP), 6 Turns wie Job 102, `render_turns(...)` direkt, volle Exception ungekürzt printen. Kosten ~1 Cent.

**Verzweigung (im Bericht die Ursache benennen + Go-Richtung):**
- **A — `ResourceExhausted` (429 Quota):** Gemini-TTS-Preview-Quota erschöpft (Burst 99→102 in 10 min). Fix = Backoff/Quota-Handling + Oli prüft GCP-Quota/Billing; ggf. **kein** Code-Change nötig außer robusterem Retry.
- **B — `InvalidArgument`/`FailedPrecondition` (400) mit „Unknown field"/Schema-Meldung:** ein **gebumptes `google-cloud-texttospeech`** hat die Request-Shape (`MultiSpeakerMarkup`/`SynthesisInput.prompt`/Speaker-Typen) geändert. Fix = **Stack pinnen** auf die 06-29-Versionen (aus P1.3 / letzte funktionierende).
- **C — `ImportError`/`TypeError`/`AttributeError` im Stack:** ein gebumptes `grpcio`/`protobuf`/`google-api-core` ist inkompatibel. Fix = **pinnen**.
- **D — `PermissionDenied`(403)/`Unauthenticated`(401):** Creds/SA-Rolle/Projekt-Drift (Vertex-AI-API/`roles/aiplatform.user`, s. CLAUDE.md GCP-Setup). Fix = Oli-seitig GCP, kein App-Code.
- **E — `DeadlineExceeded`(504) durchgängig:** server-seitige Langsamkeit/Hang → NARR-TIMEOUT tut, was es soll (Hang→Fehler). Dann Ursache tiefer (Region/Modell-Routing) — Optionen skizzieren, Master entscheidet.

**Stop + Bericht mit: echter Exception-Klasse, voller Message, gRPC-Status, google/grpc-Versionen, Go-Richtung. Warten auf Sign-off.**

## Phase 2 — Fix (nach Go)

1. **MUST, ursachen-unabhängig — Fehler nie wieder blind kappen:** in [app_pkg/narration.py](app_pkg/narration.py) `reconcile_narration` (Zeile ~138) den `[:500]`-Cut aufheben. Mindestens Exception-Typ + Message + großzügiger Traceback (z.B. `[:5000]`) speichern. Zusätzlich in [tasks.py](tasks.py) den Fehlerpfad so lassen/erweitern, dass `type(e).__name__` + `str(e)` sicher im Log stehen (tun sie schon). **+Test** (Mock: `job.exc_info` lang → gespeicherter `error` enthält Kopf **und** Schwanz, nicht bei 500 gekappt). Das ist der eine unstrittig-richtige Punkt aus dem Ur-Dossier.
2. **Ursachen-Fix je nach P1-Befund:**
   - Pfad **B/C** (Bump brach es): die betroffenen Pakete in [requirements.txt](requirements.txt) **auf die funktionierenden Versionen pinnen** (`==`), Kommentar mit Datum/Grund. **Nicht** nur den Floor anheben — exakt pinnen, damit der nächste Rebuild nicht wieder driftet. Memory `reference_cpu_torch_pin_past_resolver`-Geschwister: SDK-Pins am Resolver festnageln.
   - Pfad **A** (Quota): Retry/Backoff prüfen (429 ist schon in `_RETRYABLE`); ggf. `max_retries`/`base_delay` moderat anheben; **kein** Overengineering — die Quota-Wurzel ist GCP-seitig (Oli).
   - Pfad **D**: **kein App-Code** — Oli-GCP-Schritt dokumentieren.
3. **Verifikation ist ein echter Render, nicht pytest** (Test-Suite mockt die SDK-Boundary → fängt einen Live-Break NICHT; CLAUDE.md Test-Suite-Limit). `pytest` muss trotzdem grün bleiben (Baseline + neuer Truncation-Test).

**Stop + Bericht.**

## Phase 3 — Live-Verify + Wrap

1. **Live-Smoke**: der 6-Turn-Minimalfall (wie Job 102) rendert **`ready`** mit gültiger Dauer; danach ein voller (78-Turn) Job wie 99. Über `create_narration`/`get_narration_status` (MCP) oder direkt. Audio spielt ab.
2. **Retry der Altlasten**: 99–102 via `POST /api/narrations/<id>/retry` (failed-only, re-enqueued aus gespeicherten Inputs) neu rendern → `ready`. Beweist den Fix an genau den Fehl-Jobs.
3. **Wrap**: BACKLOG (NARR-FAIL ☑ + Ursache/Fix) · STATUS (pytest-Zahl) · CLAUDE.md **nur falls** ein Pin/Verhalten dokumentierbar ist (z.B. „google-cloud-texttospeech auf `==x.y.z` gepinnt weil Bump `z+1` die MultiSpeaker-Shape brach"; Truncation-Fix im Narration-Bullet) · **Memory** bei übertragbarer Lehre (Kandidat: „ungepinnte externe SDKs + `up -d --build` = stille Regression; Live-Break wird von SDK-mockenden Tests nicht gefangen → Verifikation = echter Render + Container-`pip freeze`-Diff"). **Bullet-Guard** `grep -nE '(- \*\*.*){2,}' BACKLOG.md` vor jedem Wrap-Commit.
4. **Deploy-Notiz**: Mac committen/pushen → Mintbox `git pull` + `docker compose up -d --build` (Pin/Code ins Image gebacken).

**Stop + Schluss-Bericht.**

## Bewusst NICHT

- **Kein** Wechsel auf den genai/Vertex-Pfad, **kein** Ändern der Modell-ID auf `-preview-tts` (bräche die verifizierte Cloud-Config — der zentrale Fehlgriff des Ur-Dossiers).
- **Kein** WAV-Header-„Fix" (schon header-agnostisch gelöst).
- **Kein** spekulatives Streaming/Chunk-Redesign (Chunking + per-Call-Deadline existieren bereits; NARR-TIMEOUT ist absichtlich).
- **Kein** Blind-Fix vor der echten Exception aus P1.

## Akzeptanz

- [ ] **P1**: echte Exception-Klasse + gRPC-Status + google/grpc-Versionen berichtet; Ursache (A–E) benannt; Sign-off vor P2.
- [ ] **P2**: Truncation-Fix (Fehler ungekürzt gespeichert) **+** ursachen-spezifischer Fix (Pin/Quota/GCP); `pytest` grün inkl. neuem Truncation-Test.
- [ ] **P3**: echter Render `ready` (6-Turn + voller Job); 99–102 per Retry grün; Docs/Memory/Wrap + Bullet-Guard; Deploy-Notiz.
