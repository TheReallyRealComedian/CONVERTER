# Sprint DIARIZE-FIX — Lange Mehrsprecher-Transkripte werden am Ende abgeschnitten (S/M, 3 Phasen)

> **Executor-Doc.** Nach jeder Phase **Stop + Bericht**, auf Sign-off warten. **Phase 1 ist ein hartes Diagnose-Gate** — die Fix-Richtung (Phase 2) hängt am Befund; NICHT vorher Code ändern. Pre-Flight: `pytest tests/` grün (Baseline **628**). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. Jede Phase selbst committen + pushen. Kein Schema/Dep/Token erwartet.
>
> **Bug (Oli-Report 2026-07-03, prod conv 91)**: Ein langes Mehrsprecher-Meeting-Transkript endet **mitten im Wort** („**Sprecher 2:** Der") — der Schwanz fehlt. Symptom-Muster: **lange** Transkripte im **Diarize-Pfad** (Mehrsprecher) verlieren das Ende; kurze + Einzelsprecher sind ok.

## Was der Master schon ausgeschlossen hat (nicht neu prüfen)

- **Kein Storage-Cap**: `models.py` `content = db.Column(db.Text)` — unlimitiert.
- **Keine `[:N]`-Truncation** im Transkript-Pfad (`grep` clean; nur `content_preview[:300]` in der List-View + `title[:255]`).
- **Formatter vollständig**: `format_diarized_transcript` ([services/deepgram_service.py:26](services/deepgram_service.py)) hat den Tail-Flush (`if current_texts: blocks.append(...)`) — verliert das Ende nicht selbst.
- **Conv 91 lief Single-Request-diarisiert** (Labels da, **kein** 90-min-Degradations-Hinweis → ≤90 min).
- **Regressions-These**: die DIARIZE-Schwelle 600→5400 hat lange Audios, die früher **gechunkt** (vollständig) waren, in **einen** Request verlegt; der Diarize-Pfad baut aus `response.results.utterances` (statt `channels[0].alternatives[0].transcript`) → Verdacht: `utterances` verliert bei langem Audio den Schwanz.

## Phase 1 — Diagnose (HART-Gate, kein Code-Change)

**Reproduktion**: es braucht ein **langes (~30–90 min) Mehrsprecher**-Audio. Kandidat schon bekannt: `260409_0142.MP3` (~35 min, 2 Sprecher — das DIARIZE-Smoke-File). Falls es den Bug nicht triggert (evtl. zu kurz), Oli nach der conv-91-Quelldatei fragen. Kosten klein halten (ein echter Call ~$0.15).

**Ein Call, dann die Response forensisch vermessen** (echter `_transcribe_single`-Pfad ODER direkter Deepgram-Call mit denselben Options; `.env` laden). Für die **eine** Response messen + berichten:
1. `len(plain)` = `len(response.results.channels[0].alternatives[0].transcript)`.
2. `len(diarized)` = `len(format_diarized_transcript(utterances, plain))`.
3. **Utterance-Coverage**: `utterances[-1].end` (Sekunden) vs. **Audio-Dauer** (aus ffprobe/dem Chunker-Metadata). Endet die letzte Utterance deutlich vor Audio-Ende?
4. **Plain-Coverage**: hört das Plain-Transkript inhaltlich beim selben „Der" auf wie das diarisierte, oder geht es weiter? (Die **entscheidende** Frage.)
5. **Alternativ-Quellen vollständig?** Gibt es `channels[0].alternatives[0].words[]` mit `.speaker` + `.punctuated_word`, und `...paragraphs.paragraphs[]` mit `.speaker`? Decken deren letzte Einträge die volle Audio-Dauer? (= Kandidaten für den Fix-Source.)

**Verzweigung (im Bericht die Go-Richtung nennen):**
- **A — Plain vollständig, `utterances` kurz** (erwartet): Fix = Source-Wechsel auf `paragraphs`/`words` (vollständig-per-Konstruktion) + Coverage-Guard.
- **B — Plain AUCH gekappt** (whole-response/Request-Limit): dann ist Diarization nur der Bote. Fix-Richtung anders (z.B. Chunk-Schwelle für sehr lange Audios differenzierter, oder Request-/Timeout-Ursache) — **im Bericht Optionen skizzieren, Master entscheidet**.
- **C — nichts reproduzierbar** (conv 91 war evtl. Audio-Ende): Befund melden, evtl. echte Quelldatei von Oli.

**Stop + Bericht mit den 5 Messungen + Go-Richtung. Warten auf Sign-off.**

## Phase 2 — Fix (nach Go; unten der erwartete Pfad A)

1. **Coverage-Guard (MUST, ursachen-unabhängig)** in `format_diarized_transcript` bzw. am Aufrufpunkt: den reinen gesprochenen Text der Blocks (ohne Labels) gegen `plain` messen — ist er **materiell kürzer** (z.B. < 90 % der Plain-Länge) → **`plain` zurückgeben + `logger.warning`** (Truncation-Verdacht). Nie wieder still kürzen. +Test.
2. **Source-Wechsel (Pfad A)**: die Sprecher-Blocks aus einem vollständigen Feld bauen — **`paragraphs` bevorzugt** (bei Diarization sprecher-segmentiert, Interpunktion erhalten, an das volle Transkript gekoppelt), **`words[].speaker` als robuster Fallback** (atomar lückenlos, `punctuated_word` joinen). Konsolidierung aufeinanderfolgender gleicher Sprecher + 1-basierte `**Sprecher N:**`-Labels **beibehalten** (Output-Format bit-stabil zum bisherigen). Single-Speaker-Guard + None-Fallbacks unverändert.
3. **Response-Optionen prüfen**: braucht der Source-Wechsel `paragraphs=True` (schon gesetzt) bzw. reichen die `words`? Keine Extra-Deepgram-Option ohne Not; **NIE** `diarize=true` neben `diarize_model=v2`.
4. **Tests** (Mock am SDK-Boundary): langes Multi-Speaker-Mock mit vollständigem `words`/`paragraphs` aber **verkürztem `utterances`** → Output deckt jetzt die volle Länge; Coverage-Guard feuert bei künstlich gekürztem Source → Plain-Fallback; bestehende DIARIZE-Tests (Single-Speaker byte-gleich, None-Fallback, Konsolidierung) **bleiben grün**.
5. `pytest` grün (628 + neue).

**Stop + Bericht.**

## Phase 3 — Live-Verify + Wrap

1. **Live-Smoke**: dasselbe lange Audio aus Phase 1 erneut durch den gefixten Pfad → Transkript **deckt bis zum Audio-Ende** (kein „Der"-Cutoff mehr), Labels + Konsolidierung intakt, Markdown sauber. `len(diarized-text) ≈ len(plain)`.
2. **Wrap**: BACKLOG (DIARIZE-FIX ☑ + Befund/Ursache) · STATUS (pytest-Zahl) · CLAUDE.md (Deepgram-Bullet: Diarize-Source ist jetzt `paragraphs`/`words`, nicht `utterances`; Coverage-Guard) · **Memory** — die bestehende `reference_deepgram_diarize_v2_query_param` um den Truncation-Befund + Coverage-Guard-Lehre ergänzen (wiederverwendbar: „Diarized-Transkript nie aus der separaten utterances-Liste bauen, immer aus dem vollständigen words/paragraphs + Guard gegen stille Truncation"). **Bullet-Guard** vor Commit.
3. **Deploy-Notiz**: Mintbox `git pull` + `up -d --build`. **Oli-Hinweis**: bereits betroffene lange Transkripte (conv 91 & Co.) einmal **neu transkribieren** — die Audios sind unversehrt, nur die alten Transkripte gekappt.

**Stop + Schluss-Bericht.**

## Bewusst NICHT

- **Kein** Zurückdrehen der 90-min-Schwelle als „Fix" (verlöre die Sprecher-Konsistenz, für die DIARIZE da war) — nur als Notfall-Lever falls Phase 1 Pfad B zeigt UND Master es anordnet.
- **Kein** neuer Vendor, **kein** Chunk-übergreifendes Sprecher-Mapping (>90 min bleibt degradiert).
- **Kein** Format-Wechsel der `**Sprecher N:**`-Ausgabe (nur die Quelle dahinter).

## Akzeptanz

- [ ] **P1**: 5 Messungen + Go-Richtung (Plain vs. utterances vs. words/paragraphs vs. Audio-Dauer) — Ursache benannt, Sign-off vor P2.
- [ ] **P2**: Diarized-Source aus vollständigem Feld (paragraphs/words) + **Coverage-Guard** (materiell-kürzer → Plain-Fallback + Log); Output-Format unverändert; Single-Speaker byte-gleich bleibt.
- [ ] **P3**: langes Mehrsprecher-Audio deckt bis Audio-Ende (kein Cutoff), `len(diarized)≈len(plain)`; pytest grün; Docs/Memory/Wrap + Bullet-Guard; Oli-Reprocess-Hinweis.
