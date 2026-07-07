# Sprint NARR-FAIL — Narration-Render schlägt fehl: ungültiger Sprach-Code (S, 2 Phasen)

> **Executor-Doc.** Nach jeder Phase **Stop + Bericht**, auf Sign-off warten. **P1 (Diagnose) ist ✅ erledigt** (Mintbox-Agent 2026-07-07, s.u.) — dieser Sprint startet direkt in **Phase 2 (Fix)**. Pre-Flight: `pytest tests/` grün (Baseline aus STATUS). Arbeitsverzeichnis Mac `/Users/olivergluth/CODE/CONVERTER` = Source-of-Truth. Code-Edits auf dem Mac, Deploy Mintbox `git pull` + `up -d --build`. Kein Schema/Dep/Token.
>
> **Root Cause (P1 ✅)**: `InvalidArgument: 400 Requested language code 'de' is not supported for Gemini voices.` Gemini-TTS-Voices auf Cloud TTS akzeptieren **kein nacktes `de`** — sie verlangen den **regionierten BCP-47-Code `de-DE`**. Der POST übernimmt `data.get('language')` **ungeprüft** ([app_pkg/narration.py:235-237](app_pkg/narration.py)); der MCP-`create_narration`-Call hat `'de'` übergeben → 1:1 an die API → 400. **Keine Regression im Code — der Input hat sich geändert**: der funktionierende Render 77 hatte **kein** `language_code`-Feld → lief mit dem Code-Default `de-DE` ([narration_render.py:280](services/narration_render.py), `DEFAULT_LANGUAGE_CODE='de-DE'` [narration.py:49](app_pkg/narration.py)). 99–102 haben `language_code:'de'` gespeichert. **SDK-Verdacht tot**: Stack ist gebumpt (texttospeech 2.37.0 / grpcio 1.81.1 / protobuf 7.35.1) aber unbeteiligt — der 400er ist eine saubere serverseitige Request-Ablehnung, **kein Pin nötig**. `mode`/`tts_model` sind korrekt.

## Master-gegroundete Fakten (nicht neu erheben — darauf aufsetzen)

- **Es ist eine REGRESSION, kein „never worked".** Der Cloud-TTS-Pfad hat **end-to-end funktioniert**: Narration **77** wurde am **2026-06-30** `ready` in ~64 s gerendert (NARR-4-Agent-Beweis, BACKLOG), NARR-1B **live-verifiziert 2026-06-29**. Jobs 99–102 sind von **07-07**. Was zwischen 06-29/06-30 und 07-07 rein-deployt wurde: **NARR-TIMEOUT** (07-01), **DIARIZE** (07-02), **IMG-SLIM** (06-30) — jedes davon per `up -d --build` **neu gebaut**.
- **Architektur = Google _Cloud_ TTS, NICHT genai/Vertex.** [services/narration_render.py](services/narration_render.py) fährt `google.cloud.texttospeech` (`client.synthesize_speech`), **nicht** `google.genai`. Der gRPC-Stack im Traceback (`google/api_core/grpc_helpers.py`) ist **auch** der Cloud-TTS-Pfad — „gRPC ⇒ Vertex" ist eine **Fehl-Schlussfolgerung**.
- **Modell-ID `gemini-2.5-flash-tts` ist KORREKT** für den Cloud-Pfad (CLAUDE.md „Gemini Models"; **live-verifiziert 2026-06-29**: der Cloud-Name hat **kein** `-preview-`-Infix, der genai-Name `gemini-2.5-flash-preview-tts` **500t** auf dem Cloud-Endpoint). ⚠️ **NICHT** auf `-preview-tts` „korrigieren" — das bräche die verifiziert-funktionierende Config.
- **PCM→WAV ist bereits gelöst.** `pcm_to_wav_bytes` ist header-agnostisch (RIFF-Passthrough + `wave`-Wrap). Kein WAV-Bug — 77 hat gültiges Audio produziert.
- **Der echte Fehler ist bereits da (ohne Code-Change):** [tasks.py:88](tasks.py) loggt auf dem Fehlerpfad `logger.error(f"Error: {type(e).__name__}: {str(e)}")` → **Worker-Container-Logs enthalten Typ + Message** für 99–102. Zusätzlich steckt der **volle** Traceback in `job.exc_info` in Redis (FailedJobRegistry).
- **Truncation-Stelle (bestätigt):** [app_pkg/narration.py:138](app_pkg/narration.py) `reconcile_narration` → `error = (job.exc_info or '')[:500]`. Das ist die einzige Kappung; sie hat diese Diagnose blind gemacht.
- **`updated_at` ≠ „Dauer bis Fehler".** Es ist der Poll-Zeitpunkt, an dem `reconcile_narration` `pending`→`failed` geflippt hat, nicht der Render-Fehler-Zeitpunkt. Die Timing-Tabelle im Ur-Dossier (99/100/102 ≈ 5 s, 101 ≈ 134 s) ist damit **kein** verlässliches Signal — nicht darauf theoretisieren.
- **SDK-Pinning-Verdacht — WIDERLEGT (P1):** der google-Stack ist zwar gebumpt (texttospeech 2.37.0 / grpcio 1.81.1 / protobuf 7.35.1, alles Floor-gepinnt), aber **unbeteiligt** — der Fehler ist ein sauberer serverseitiger 400 auf einen ungültigen `language`-Wert, kein Shape/ABI-Break. **Kein Pin nötig.** (Der ursprüngliche Master-Lead-Verdacht lag hier daneben — die „Regression" war ein Input-Wechsel, kein Dependency-Drift.)

## Phase 1 — Diagnose ✅ ERLEDIGT (Mintbox-Agent 2026-07-07)

Root Cause = `InvalidArgument: 400 Requested language code 'de' is not supported for Gemini voices.` (Worker-Log, alle vier Jobs identisch). Beweiskette: DB-Metadaten 99–102 haben `language_code:'de'`; Render 77 (funktionierte) hatte gar kein Feld → Code-Default `de-DE`. Eintrittspforte [narration.py:235-237](app_pkg/narration.py) (`data.get('language')` ungeprüft). Google-Stack gebumpt aber unbeteiligt (sauberer 400). **Kein Diagnose-Rest offen — direkt Phase 2.**

## Phase 2 — Fix (Language-Normalisierung + Truncation-Tail)

1. **Language-Code am Boundary normalisieren (Kern-Fix).** Neuer pure Helper `_normalize_language_code(raw)` in [app_pkg/narration.py](app_pkg/narration.py):
   - bereits regioniert (enthält `-`, z.B. `de-DE`) → **unverändert** durchreichen;
   - nacktes 2-Letter-Kürzel → über eine **kleine Map** regionalisieren (mind. `de→de-DE`, `en→en-US`; erweiterbar, konservativ);
   - leer/kein-String → `DEFAULT_LANGUAGE_CODE` (`de-DE`);
   - **unmappbares** Kürzel → früh **400 mit klarer Message** („Sprach-Code `xx` nicht unterstützt — regionierten BCP-47-Code wie `de-DE` senden.") statt es blind an die API zu geben. Max 2 Sätze, deutsch, keine Emojis (CLAUDE.md-Microcopy).
   - **An BEIDEN Aufrufstellen anwenden** — sonst bleibt eine Lücke:
     - **POST** [narration.py:235-237](app_pkg/narration.py): den rohen `data.get('language')` durch den Helper schicken, bevor er in Metadaten/Enqueue geht.
     - **Retry** [narration.py:382](app_pkg/narration.py): `metadata.get('language_code')` durch **denselben** Helper schicken. ⚠️ **Ohne das failt ein Retry von 99–102 erneut identisch** — `'de'` ist truthy, das bestehende `or DEFAULT` greift nie. Mit der Normalisierung am Retry-Read heilen die vier Altlasten **ohne** DB-Handedit.
2. **Truncation-Fix — den Tail persistieren, nicht den Head.** [narration.py:138](app_pkg/narration.py) `reconcile_narration`: `error = (job.exc_info or '')[:500]` kappt den **Kopf** des Tracebacks — die Exception-Zeile steht am **Ende**, also bliebe der bisherige Cut wirkungslos. Fix: die **letzten** ~2000 Zeichen persistieren (Tail) **oder** die finale Exception-Zeile extrahieren + voranstellen. So steht die `InvalidArgument: …`-Zeile künftig sichtbar im gespeicherten `error`. (`tasks.py:88` loggt Typ+Message schon — dort nichts nötig.)
3. **Tests** (Mock, keine echte SDK-Boundary):
   - POST mit `language:'de'` → gespeicherter/enqueuter `language_code == 'de-DE'`; mit `de-DE` → unverändert; mit `xx` → 400.
   - **Retry** einer Row mit gespeichertem `language_code:'de'` → enqueut `de-DE` (Regressionsschutz gegen die Retry-Falle).
   - Truncation: langer `job.exc_info` mit der Exception-Zeile am Ende → gespeicherter `error` **enthält** diese Zeile.
4. `pytest` grün (Baseline aus STATUS + neue Tests).

**Stop + Bericht.**

## Phase 3 — Live-Verify + Retry-Heilung + Wrap

1. **Frischer Render** (POST-Pfad): 6-Turn-Minimalfall mit `language:'de'` → jetzt `ready` (beweist die POST-Normalisierung). Über `create_narration`/`get_narration_status` (MCP) oder direkt.
2. **Altlasten heilen** (Retry-Pfad): 99–102 via `POST /api/narrations/<id>/retry` → `ready`. Beweist die Retry-Normalisierung **und** räumt die vier Fehl-Jobs auf, ohne die DB anzufassen.
3. **Wrap**: BACKLOG (NARR-FAIL ☑ + Root Cause/Fix) · STATUS (pytest-Zahl) · CLAUDE.md (Narration-Bullet: `language` muss regionierter BCP-47 sein; Server normalisiert nackt `de→de-DE`, unmappbar → 400; `metadata.error` persistiert jetzt den Traceback-**Tail**) · **Memory** — die bestehende Master-Memory `reference_gemini_tts_language_code_bcp47` (schreibt der Master beim Sign-off; Sub-Thread verweist nur) bzw. Wrap-Lehre. **Bullet-Guard** `grep -nE '(- \*\*.*){2,}' BACKLOG.md` vor jedem Commit.
4. **Optionaler Caller-Hinweis** (nicht dieser Sprint, nur notieren): der `create_narration`-MCP-Wrapper / die `erklaerbaer-narration`-Skill sollten idealerweise `de-DE` senden; die Server-Normalisierung ist die belastbare Verteidigung, der Caller-Fix nur Kosmetik.
5. **Deploy-Notiz**: Mac committen/pushen → Mintbox `git pull` + `docker compose up -d --build`.

**Stop + Schluss-Bericht.**

## Bewusst NICHT

- **Kein** SDK-Pin (der google-Stack ist unbeteiligt — sauberer 400; das Ur-Dossier UND der erste Master-Verdacht lagen hier falsch).
- **Kein** Wechsel auf den genai/Vertex-Pfad, **kein** Ändern der Modell-ID auf `-preview-tts` (bräche die verifizierte Cloud-Config — der zentrale Fehlgriff des Ur-Dossiers).
- **Kein** WAV-Header-„Fix" (schon header-agnostisch gelöst).
- **Kein** DB-Handedit der 99–102 — die Retry-Normalisierung heilt sie sauber.

## Akzeptanz

- [ ] **P2**: `_normalize_language_code` an POST **und** Retry; Truncation persistiert den Exception-Tail; `pytest` grün inkl. der drei neuen Tests (POST-Map, Retry-Map, Truncation-Tail).
- [ ] **P3**: frischer Render mit `language:'de'` → `ready`; 99–102 per Retry → `ready` (ohne DB-Handedit); Docs/Memory/Wrap + Bullet-Guard; Deploy-Notiz.
