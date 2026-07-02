# Sprint DIARIZE — Sprecher-Erkennung in der Audio-Transkription (Deepgram Diarization v2) (M, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **616**). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Kein** Schema/neuer Token; kein neuer Vendor.
>
> **Kontext**: Olis Mehr-Sprecher-Audios (Meetings) werden korrekt transkribiert, aber **ohne Sprecher-Zuordnung** → Fehlattribution/Missverständnisse beim Lesen. Das Audio ist **mono** (keine Sprecherkanäle) → Trennung muss an der Stimme passieren = **Speaker Diarization**. Entscheidung (Oli, 2026-07-02): **Deepgram Diarization v2** (kein neuer Vendor; AssemblyAI = Fallback-Option falls Qualität nicht reicht; lokal/GPU verworfen — kollidiert mit IMG-SLIM). Labels **generisch** („Sprecher 1/2/3") — die Zuordnung zu echten Namen macht nachgelagert der notion-transcripts-Agent.

## Verifizierte Fakten (Master-gegroundet 2026-07-02)

**Code:**
- Der API-Call ([services/deepgram_service.py:147](services/deepgram_service.py), `_transcribe_single`): SDK 5.1.0, `self.client.listen.v1.media.transcribe_file(request=…, model="nova-3", smart_format=True, utterances=True, punctuate=True, language=…, numerals=True, paragraphs=True, keyterm=…, request_options={timeout…})`. **`utterances=True` ist schon an** — aber **kein `diarize`** wird angefordert, und die Response-Verarbeitung zieht **nur** `response.results.channels[0].alternatives[0].transcript` (Plain-Text) → Speaker-Info würde heute weggeworfen.
- **Chunking** (`DeepgramService`-Klassenkonstanten): `MAX_AUDIO_DURATION_SECONDS = 600` (>10 min → Split), `CHUNK_DURATION_SECONDS = 1800`, `OVERLAP_SECONDS = 5`, `MAX_FILE_SIZE_MB = 500`. `TranscriptMerger.merge_transcripts` ([services/audio_chunker.py:221](services/audio_chunker.py)) merged **Plain-Strings** per Overlap-Dedup.
- Caller: [app_pkg/audio.py:85](app_pkg/audio.py) `transcribe_file(buffer_data, language) -> str` — der String fließt unverändert in Library/Downstream. gunicorn-Timeout 1800 s (Dockerfile-CMD) — lange Uploads ok. `TIMEOUT_DEEPGRAM_SECONDS` = per-Request-Timeout (app_pkg/config.py).

**API (recherchiert, [Deepgram-Docs Diarization](https://developers.deepgram.com/docs/diarization) + [Batch Diarization V2](https://deepgram.com/learn/introducing-batch-diarization-v2)):**
- **v2 aktivieren via `diarize_model="v2"`** (oder `"latest"`). ⚠️ **NICHT zusätzlich `diarize=true`** — Requests mit **beiden** Params werden **rejected**. `diarize=true` allein = altes v1 (schlechter). v2 ist **batch-only** — passt, wir sind prerecorded.
- v2: bessere Attribution, in Human-Evals 3,3× häufiger bevorzugt als v1.
- Speaker-Indizes stehen an den `utterances` (`utterance.speaker`, 0-basiert, int) und sind **pro Request lokal** — Chunk-übergreifend NICHT stabil.
- Deepgram prerecorded verarbeitet lange Dateien nativ (Limit 2 GB/Request; ~90-min-Audio in typ. <2 min) — unsere 10-min-Chunk-Schwelle ist historisch konservativ.

## Design (gesetzt)

1. **`diarize_model="v2"`** in den Call; alle bestehenden Optionen (inkl. `keyterm`, `smart_format`, `paragraphs`) bleiben.
2. **Pure Formatter-Funktion** `format_diarized_transcript(utterances) -> str` (neues kleines Modul oder in `deepgram_service.py`, aber pur + direkt testbar):
   - Aufeinanderfolgende Utterances **desselben** Sprechers zu einem Block konsolidieren (kein Zeilen-Spam).
   - Ausgabe-Format: `**Sprecher 1:** text` pro Block (Markdown-fett — das Transkript landet als Markdown in der Library; API-Index 0-basiert → Anzeige 1-basiert).
   - **Genau 1 distinct Speaker → Plain-Transkript ohne Labels zurückgeben** (= der `channels[0]`-Text wie heute). ⚠️ Das ist die **Diktat-Regression-Guard**: Olis Einzel-Diktate (Hauptvolumen!) dürfen sich **null** ändern.
   - Defensiv: `utterances` fehlt/leer oder `speaker`-Felder fehlen (None) → Fallback Plain-Transkript wie heute. Diarization-Ausfall darf nie die Transkription brechen.
3. **Chunk-Schwelle anheben**: `MAX_AUDIO_DURATION_SECONDS` 600 → **5400** (90 min) → Meetings laufen als **1 Request** = konsistente Sprecher über die ganze Aufnahme. Dabei prüfen: `TIMEOUT_DEEPGRAM_SECONDS` muss ein 90-min-Audio tragen (Deepgram braucht typ. <2 min; Wert ggf. moderat anheben — im Bericht begründen); `MAX_FILE_SIZE_MB=500` bleibt (unter Deepgrams 2-GB-Limit).
4. **Multi-Chunk-Fall (>90 min) = graziöse Degradation, KEIN Merger-Umbau**: Chunks laufen weiter als Plain-Text (ohne Sprecher-Formatierung) durch den bestehenden Overlap-Merger — Verhalten exakt wie heute, plus **ein** vorangestellter Hinweis im Ergebnis (`> Hinweis: Aufnahme über 90 Minuten — Sprecher-Erkennung für diese Länge deaktiviert.`). Begründung: per-Chunk-Speaker sind inkonsistent + Präfixe würden den Overlap-Dedup brechen; Sprecher-Mapping über Chunks ist ein eigener, größerer Sprint, falls je nötig. (Olis Audio-Profil: „gemischt" — Normalfall ≤90 min, Degradation dokumentiert.)
5. **Kein UI-Toggle** — Diarization immer an (bei 1 Sprecher unsichtbar per Design). Keine Route-/Template-Änderung nötig (der String fließt wie bisher). DE-Microcopy nur für den Degradations-Hinweis.

## Phase 1 — Formatter + v2-Wiring + Tests

1. **SDK-Fähigkeit klären** (zuerst!): akzeptiert das Python-SDK 5.1.0 `diarize_model` als kwarg am `transcribe_file`-Call? (SDK-Source/Signatur im venv prüfen — kwargs-Passthrough als Query-Param vs. typisierte Param-Liste. Falls das SDK den Param nicht kennt: den dokumentierten Escape-Hatch des SDK nutzen (extra/query-params) oder als **letzte** Option den REST-Call. Befund im Bericht.)
2. `format_diarized_transcript(utterances)` als **pure** Funktion + `diarize_model="v2"` in `_transcribe_single` + Response-Pfad: bei ≥2 distinct Speakers → Formatter-Output, sonst Plain wie heute.
3. **Tests** (Mock am SDK-Boundary wie gehabt, `app.deepgram_service`-Pattern): Multi-Speaker → Labels + Konsolidierung aufeinanderfolgender gleicher Sprecher · Single-Speaker → **exakt** der Plain-Text (Regression-Guard) · `utterances` fehlen/`speaker=None` → Plain-Fallback · keyterms/Optionen unverändert im Call (inkl. **kein** `diarize=true` neben `diarize_model`).
4. `pytest` grün (Baseline 616 + neue).

**Stop + Bericht** (inkl. SDK-Befund aus 1.).

## Phase 2 — Chunk-Schwelle + Degradation

1. `MAX_AUDIO_DURATION_SECONDS` → 5400; `TIMEOUT_DEEPGRAM_SECONDS` prüfen/begründet anpassen (config-Kommentar aktualisieren — er erklärt die Timeout-Beziehungen).
2. Multi-Chunk-Pfad: Sprecher-Formatierung **aus** (Plain wie heute) + der eine Degradations-Hinweis vorangestellt. Merger unberührt.
3. Tests: Schwellen-Logik (89 min → single, >90 min → chunked+Hinweis+plain) über die bestehende `needs_splitting`-Mechanik gemockt.
4. `pytest` grün.

**Stop + Bericht.**

## Phase 3 — Live-Smoke + Kalibrierung + Wrap

1. **Live-Smoke (echte Instanz, echtes Geld — Umfang klein halten):**
   - Ein **echtes 2+-Sprecher-Audio** (Oli liefert ein Meeting-Stück, gern nur 5–10 min) → Labels da, Attribution plausibel, Blocks konsolidiert, Markdown rendert sauber im Library-Reader.
   - Ein **Einzel-Diktat** (Regression!) → Ergebnis **identisch** zum heutigen Format, keine Labels, kein Hinweis.
   - Deepgram-Dashboard kurz checken: kostet `diarize_model=v2` sichtbar extra? (Erwartung: nein/marginal — Befund notieren.)
2. **Qualitäts-Urteil an Oli**: Attribution gut genug? (Falls klar unbrauchbar → im Wrap als Befund festhalten; der dokumentierte Fallback-Pfad ist AssemblyAI — **nicht** in diesem Sprint bauen.)
3. **Wrap**: BACKLOG (DIARIZE ☑ + Ergebnis) · STATUS (pytest-Zahl, „kein Schema/Dep/Token") · CLAUDE.md (Deepgram-Bullet: nova-3 + `diarize_model=v2`, Single-Speaker-Guard, 90-min-Schwelle) · ggf. Memory (nur falls nicht-offensichtliche Lehre, z.B. SDK-kwargs-Befund) · **Bullet-Guard** vor BACKLOG/STATUS-Commit.
4. **Deploy-Notiz**: Mintbox `git pull` + `docker compose up -d --build`. Kein Schema/Dep/Token.

**Stop + Schluss-Bericht.**

## Bewusst NICHT (Scope-Grenze)

- **Kein** AssemblyAI/Vendor-Wechsel (dokumentierter Fallback, eigener Sprint falls v2-Qualität nicht reicht).
- **Kein** lokales pyannote/WhisperX, **kein** GPU-Touch (kollidiert mit IMG-SLIM; verworfen).
- **Keine** echten Sprecher-Namen/Voice-Profile (macht nachgelagert der notion-transcripts-Agent aus den generischen Labels).
- **Kein** Sprecher-Mapping über Chunks (>90 min degradiert graziös — eigener Sprint falls je real nötig).
- **Kein** UI-Toggle, keine Template-Änderung.

## Akzeptanz

- [ ] Mehr-Sprecher-Audio → Markdown-Transkript mit `**Sprecher N:**`-Blocks (konsolidiert, 1-basiert), Attribution im Live-Smoke plausibel.
- [ ] **Einzel-Diktat byte-gleich zu heute** (keine Labels, kein Hinweis) — Regression-Guard testet das.
- [ ] `diarize_model="v2"` (ohne `diarize=true` daneben); keyterms + alle bestehenden Optionen unverändert.
- [ ] ≤90 min = 1 Request (konsistente Sprecher); >90 min = heutiges Verhalten + Degradations-Hinweis.
- [ ] Diarization-Ausfall (keine utterances/speaker) → Plain-Fallback, nie ein Transkriptions-Bruch.
- [ ] `pytest` grün; kein Schema/Dep/Token; Docs/Wrap + Bullet-Guard.
