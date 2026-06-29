# Sprint NARR-1 — Treue-Synth-Kern (Turn-Liste → Audio, ohne Skript-Gen/Tag-Hack) (M, 2 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **546**). Du committest jede Phase selbst (Hash + push), **fokussiert**. Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Reiner Synth-Kern**: **kein** Endpoint, **keine** Persistenz, **kein** UI, **kein** Schema/Dep/Token. Backend pytest-getestet (mock am SDK-Singleton-Boundary, keine echten Gemini-Calls).
>
> **Kontext**: Teil des NARRATION-Reframes (Podcast → treue Erklärbär-Vertonung). Voller Entwurf: [docs/narration_reframe.md](docs/narration_reframe.md). Dieser Sprint baut **nur** den Render-Kern; Endpoint (NARR-3), Persistenz (NARR-2), Skill (NARR-4), UI (NARR-5) folgen.

## Warum & Entscheidungen (gesetzt, Master-gegroundet)

Der „brutale Tag-Hack" sitzt **upstream** in der Skript-Generierung (`prompts.py::calculate_tag_guidance` + die „Transform, do NOT summarize"-Prompts + `script.py::format_dialogue_with_llm` + `dialogue.py::parse_dialogue`). **Die TTS-Synth-Maschinerie ist bereits sauber**: `tts.py::generate_podcast` nimmt schon eine strukturierte Liste `[{'speaker','style','text'}]`, das Chunking (`_create_dialogue_chunks`) ist **size-only** (keine Tag-Zahl). Der Treue-Pfad **umgeht** die Skript-Gen komplett — der Agent (NARR-4) liefert die Turns direkt — und reicht sie an die bestehende Synth-Maschinerie. NARR-1 baut diesen Treue-Entry + zwei kleine Anpassungen.

- **Label↔Voice entkoppeln.** Heute setzt `generate_podcast` `SpeakerVoiceConfig(speaker=X, voice_name=X)` — **Label == Voice-Name**. Unser Kontrakt trennt Label (`"Anna"`) und Voice (`"Kore"`) über `voices: {label: voice}`. Lösung: `generate_podcast` kriegt einen **optionalen `voices`-Param** (Default `None`) → `voice_name = voices.get(speaker, speaker)`. **`voices=None` = altes Verhalten byte-identisch** (alter Flow unberührt).
- **Treue = Metadaten-Filter überspringen.** `generate_podcast` ruft heute `filter_metadata_lines`, das Zeilen mit URL-/Caption-Mustern (`.de$`, `^http`, `IMAGO`, …) **und** <2-Zeichen-Zeilen **droppt**. Für treue Vertonung ist das ein **Fidelity-Bug** (eine legitime Turn „Mehr auf example.de" würde verschwinden). Lösung: optionaler **`filter_metadata`-Param** (Default `True`) → Treue-Pfad ruft mit `False`. Alter Flow Default `True` = unberührt.
- **`split_long_dialogue_turns` bleibt** auch im Treue-Pfad — es splittet lange Turns an Satzgrenzen (**inhaltserhaltend**, kein Drop) → bessere TTS-Qualität. Kein Param nötig.
- **Tags inline im Text** (v1). Der Kontrakt-Turn ist `{speaker, text}`; optionale Performance-Tags setzt der Agent **inline in `text`** (z.B. `"[ruhig] …"`, Gemini liest sie inline), die Synth reicht `text` **verbatim** durch (`style=''`). **Kein** separates Tag/Style-Feld in v1 (die Cowork-Tag-Recherche speist später NARR-4; ein strukturiertes `tag`→`style`-Mapping ist dann ein trivialer Nachzug). Begründung: maximal treu, minimal Mechanik.
- **`mode`** (`single_speaker`/`two_speaker`) ist eine **Validierungs**-Bedingung, kein Synth-Branch — die Synth erkennt 1-vs-multi-Speaker schon über `len(unique_speakers)`.
- **Der Hack wird NICHT gelöscht** — `calculate_tag_guidance`/`format_dialogue_with_llm`/`parse_dialogue` bleiben (der **alte Flow nutzt sie bis NARR-5**). NARR-1 ruft sie im Treue-Pfad nur **nicht** auf. So bleibt jeder Sprint grün und der alte Flow läuft weiter.
- **SDK-Singleton-Boundary**: der Treue-Entry kommt als Methode auf `GeminiService` (wie `generate_podcast`/`format_dialogue_with_llm`), damit Tests am `app.gemini_service`/Client-Boundary mocken (Projekt-Konvention).

## Verifizierte Code-Fakten (Master-gegroundet)

- **`services/gemini/tts.py::generate_podcast(client, dialogue, language='en', tts_model=None, pydub_available=True)`** ([tts.py:60](services/gemini/tts.py)) — `dialogue` = `[{'speaker','style','text'}]`. Baut `speaker_voice_configs` mit `SpeakerVoiceConfig(speaker=speaker, voice_config=…PrebuiltVoiceConfig(voice_name=speaker))` (Z.96–105 — **`voice_name=speaker`**). Ruft `filter_metadata_lines` (Z.120) + `split_long_dialogue_turns(max_words=50)` (Z.126). Chunk-Entscheidung size-only: `len ≤ MAX_LINES_PER_CHUNK(80)` → `generate_single_chunk`, sonst `_generate_with_chunking`. `MAX_CHARS_PER_CHUNK=3000`.
- **`generate_single_chunk`** ([synthesis.py:24](services/gemini/synthesis.py)) — Transkript `"{speaker}: [{style}] {text}"` (bzw. ohne `[{style}]` wenn leer); 1 Speaker → `voice_config`, >1 → `multi_speaker_voice_config` mit den nach `config.speaker in unique_speakers` gefilterten Configs. Echter `client.models.generate_content(model, contents, config)`-Call → WAV-Tempfile.
- **`filter_metadata_lines`** ([dialogue.py:62](services/gemini/dialogue.py)) — **droppt** Zeilen <2 Zeichen + `_METADATA_PATTERNS` (`.de$`/`.com$`/`^http`/`IMAGO`/…). **Fidelity-Risiko → im Treue-Pfad aus.**
- **`split_long_dialogue_turns`** ([dialogue.py:88](services/gemini/dialogue.py)) — splittet >50-Wort-Turns an Satzgrenzen, **erhält allen Text** (Style nur auf 1. Teil). **Bleibt.**
- **`parse_dialogue`** ([dialogue.py:33](services/gemini/dialogue.py)) — der `:`-Parser; **nur vom alten `script.py` genutzt**. Treue-Pfad braucht ihn nicht (strukturierte Turns rein).
- **`GeminiService`** ([services/gemini/__init__.py:25](services/gemini/__init__.py)) — Facade, delegiert an die Free-Functions; `TTS_MODELS`/`DEFAULT_TTS_MODEL` (`gemini-2.5-flash-preview-tts`/`-pro-preview-tts`). `app.gemini_service`-Singleton = Test-Patch-Punkt.
- **Alter Flow-Caller**: `tasks.py::generate_podcast_task` → `GeminiService.generate_podcast(dialogue, language, tts_model)` (kein voices/filter_metadata → Defaults → byte-identisch). **Nicht anfassen.**

## Phase 1 — Treue-Synth-Entry + Label↔Voice + Fidelity-Filter-Gate + Tests

1. **`generate_podcast` zwei optionale Params** ([services/gemini/tts.py](services/gemini/tts.py)) — Signatur → `generate_podcast(client, dialogue, language='en', tts_model=None, pydub_available=True, voices=None, filter_metadata=True)`:
   - Voice-Config-Bau: `voice_name = voices.get(speaker, speaker) if voices else speaker`. (`voices=None` → exakt heutiges Verhalten.)
   - Filter: `if filter_metadata: dialogue_lines = filter_metadata_lines(dialogue_lines)` (sonst überspringen). `split_long_dialogue_turns` läuft **immer**.
   - Sonst **nichts** ändern (Chunking/Retry/Concat unberührt).
2. **Treue-Entry** `synthesize_turns(client, turns, voices, *, mode='two_speaker', language='de', tts_model=None, pydub_available=True)` (in [services/gemini/tts.py](services/gemini/tts.py), neben `generate_podcast`):
   - **Validierung** (sonst `ValueError`): `turns` nicht-leere Liste; jede Turn `{speaker: non-blank-str, text: non-blank-str}`; `voices` Dict, das **alle** distinct `speaker` abdeckt; `mode='single_speaker'` → genau 1 distinct Speaker, `mode='two_speaker'` → 1–2 distinct Speaker; (`tts_model` defaultet `generate_podcast` selbst).
   - **Map** Turns → `dialogue = [{'speaker': t['speaker'], 'style': '', 'text': t['text']} for t in turns]` (Tags bleiben inline im `text`, verbatim).
   - **Call** `return generate_podcast(client, dialogue, language=language, tts_model=tts_model, pydub_available=pydub_available, voices=voices, filter_metadata=False)`.
3. **Facade** `GeminiService.synthesize_turns(self, turns, voices, *, mode='two_speaker', language='de', tts_model=None)` ([services/gemini/__init__.py](services/gemini/__init__.py)) → delegiert (mit `self.client` + `pydub_available=self.pydub_available`). In `__all__`/Import-Surface wie `generate_podcast`.
4. **Tests** (`tests/test_narration_synth.py` o.ä.; Mock-Scaffolding aus den bestehenden Gemini/TTS-Tests übernehmen — Client-`models.generate_content` mocken, den `config`/`contents`-Arg **capturen**):
   - **Validierung** (pure, kein Client-Call): leere turns → ValueError; Turn ohne text/speaker → ValueError; `two_speaker` mit 3 Speakern → ValueError; `single_speaker` mit 2 → ValueError; Speaker ohne Voice in `voices` → ValueError.
   - **Label↔Voice**: `synthesize_turns` mit `voices={'Anna':'Kore','Ben':'Puck'}` → der gecapturte `multi_speaker_voice_config` trägt `SpeakerVoiceConfig(speaker='Anna', …voice_name='Kore')` etc. (Label bleibt Transkript-Label, Voice = gemappt).
   - **Fidelity (load-bearing)**: eine Turn mit `text='Mehr dazu auf example.de'` landet **im Transkript** (`contents` enthält den Text) — wird **nicht** gedroppt (Treue-Pfad `filter_metadata=False`). Kontrast-Test: `generate_podcast(..., filter_metadata=True)` **droppt** dieselbe Zeile (beweist, dass der Default-Pfad unverändert filtert).
   - **Transkript verbatim**: inline-Tag im Text (`'[ruhig] Hallo'`) bleibt im `contents` erhalten; `style` leer → Format `"Anna: [ruhig] Hallo"` (kein doppeltes Bracketing).
   - **Alter Flow byte-identisch**: `generate_podcast(dialogue)` ohne voices → `voice_name==speaker` (gecapturt); bestehende Podcast-Tests bleiben grün.
   - `pytest tests/` grün ≥ 546.

**Stop + Bericht.**

## Phase 2 — Wrap

1. **STATUS.md** + **BACKLOG.md**: NARR-1 ☑ done (Hash); im NARR-Cluster-Eintrag „NARR-1 bereit für Dispatch" → „NARR-1 ☑". **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md`).
2. **Sprint-Prompt-Doc** (dieses File) mit eincheckbar, falls noch untracked.
3. Finaler `pytest`. (Memory erst, wenn der NARR-Stack steht — hier noch nicht nötig; falls eine nicht-offensichtliche Synth-Lehre auftaucht, kurz festhalten.)

**Stop + Schluss-Bericht.** Kein Deploy nötig (reiner Backend-Code ohne neuen Caller — `synthesize_turns` wird erst ab NARR-3 aufgerufen; nichts läuft live anders). NARR-2 (Persistenz) ist der nächste Sprint.

## Bewusst NICHT (Scope-Grenze)

- **Kein** Endpoint, **keine** Persistenz, **kein** UI, **kein** Token, **kein** RQ-Task (NARR-2/3/5).
- **Kein** Löschen von `calculate_tag_guidance`/`format_dialogue_with_llm`/`parse_dialogue` (alter Flow nutzt sie bis NARR-5).
- **`generate_podcast`-Default-Verhalten unverändert** (voices=None + filter_metadata=True → byte-identisch); `tasks.py` + `script.py` + `prompts.py` unberührt.
- **Kein** separates `tag`/`style`-Feld im Kontrakt (Tags inline, v1).
- **Kein** Voice-Namen-Katalog-Check (Gemini fehlert bei ungültiger Voice; Katalog-Validierung ist NARR-3/4-Sache) — NARR-1 prüft nur, dass `voices` die Speaker abdeckt.

## Akzeptanz

- [ ] `generate_podcast` akzeptiert `voices` (Label→Voice; `None`=Label-als-Voice, byte-identisch) + `filter_metadata` (Default `True`; `False`=kein Drop); sonst unverändert.
- [ ] `synthesize_turns` (+ `GeminiService`-Facade): Validierung (mode/Speaker-Zahl/voices-Abdeckung/non-blank), Turns→dialogue (Tags inline verbatim), ruft `generate_podcast(voices=…, filter_metadata=False)`.
- [ ] Tests: Validierung, Label↔Voice-Mapping, **Fidelity-no-drop** (+ Kontrast filter=True droppt), Transkript-verbatim, alter-Flow-byte-identisch. `pytest` grün ≥ 546.
- [ ] Kein Endpoint/Persistenz/UI/Schema/Dep/Token; Hack-Funktionen + alter Flow unberührt.
