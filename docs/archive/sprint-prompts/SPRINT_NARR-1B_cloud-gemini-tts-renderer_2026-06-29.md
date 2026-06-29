# Sprint NARR-1B — Cloud-Gemini-TTS-Renderer (Pivot: google-cloud-texttospeech ≥2.31.0) (M, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **560**). Du committest jede Phase selbst (Hash + push), **fokussiert**. Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Backend + ein Dep-Bump** (`google-cloud-texttospeech` 2.21.0→≥2.31.0). Tests am **SDK-Singleton-Boundary** (Mock des `texttospeech`-Clients — **keine** echten Cloud-Calls in pytest).
>
> ⚠️ **Live-Verify-Gate jenseits pytest**: einige API-Fakten sind nur am **echten Cloud-TTS-Endpoint + GCP-Projekt** final bestätigbar (exakter Modell-Name, WAV-Header-Präsenz, Byte-Limit). Der Renderer wird **defensiv** gebaut (header-agnostisch, Modell-Name konfigurierbar), sodass er **unabhängig** von der Auflösung korrekt ist; die Live-Bestätigung ist ein **Smoke-Gate** (Phase 3) — teils vom Sub-Thread (falls Creds da), teils Oli. **Kontext**: NARRATION-Reframe, Pivot auf den Cloud-TTS-Pfad. Voller Entwurf: [docs/narration_reframe.md](docs/narration_reframe.md); Tag-Doktrin: [docs/narration_tag_doctrine.md](docs/narration_tag_doctrine.md).

## Warum & Entscheidungen (gesetzt)

NARR-1 P1 (`d419271`) baute den **Treue-Synth-Kontrakt** (`synthesize_turns` Validierung + Mapping + Facade) gegen den **Gemini-API-Pfad** (`google.genai`, `generate_podcast`). **Pivot 2026-06-29 (Oli: „gleich richtig machen")**: die v1-Render-Engine wird der **Cloud-TTS-Pfad** (`google-cloud-texttospeech`), weil er drei Dinge nativ löst, die der API-Pfad nicht kann: **Label↔Voice-Trennung** (`speaker_alias`/`speaker_id` — auch Single-Speaker korrekt, behebt den NARR-1-P1-Gap), **Director's-Notes ohne Leakage** (`SynthesisInput.prompt`, strukturell getrennt vom Transkript), **dokumentierte Tags**.

- **NARR-1-Validierung/Kontrakt wird wiederverwendet** (in ein pures Modul gezogen); der **genai-`generate_podcast` bleibt für den Alt-Flow** unberührt (seine NARR-1-Params sind inert mit Defaults). Die NARR-1-`GeminiService.synthesize_turns`-Facade wird **abgelöst** (zeigte auf die falsche Engine).
- **Renderer ist defensiv + konfigurierbar**: header-agnostisches WAV-Wrapping, Modell-Name als Config (Default `gemini-2.5-flash-tts`), Byte-basiertes Chunking. So überlebt er die Diskrepanzen unten.
- **Bestehender `texttospeech`-Client wird wiederverwendet** (`GoogleTTSService.client`) — kein neuer Client, keine neuen Creds. Der Standard-Neural-Pfad (`synthesize_speech`) bleibt **unberührt** und funktionsfähig.
- **Pydub-Concat** wie gehabt (`services/gemini/audio.py::concatenate_with_pydub` nimmt Chunk-WAV-Pfade).

## Verifizierte API-Fakten (Grounding-Workflow `cloud-gemini-tts-api-grounding`, 3 Quellen cross-verifiziert)

**SDK-Floor**: `google-cloud-texttospeech>=2.31.0` (`multi_speaker_markup` ab 2.20, `prompt` ab 2.29, Gemini-Speaker-Typen ab 2.30/2.31 → ≥2.31.0 ist der von allen Quellen geteilte sichere Floor). Stable-Surface `from google.cloud import texttospeech` (nicht v1beta1).

**Multi-Speaker** (≤2 distinct Voices; `speaker_alias` muss **alphanumerisch, kein Whitespace** sein → menschliche Labels zu `Speaker1`/`Speaker2` sanitizen, Mapping Label→Alias halten):
```python
import io, wave
from google.cloud import texttospeech

speaker_configs = [texttospeech.MultispeakerPrebuiltVoice(speaker_alias=alias, speaker_id=voices[label])
                   for label, alias in alias_for.items()]   # alias='Speaker1'…; speaker_id='Kore'/'Charon'
markup = texttospeech.MultiSpeakerMarkup(turns=[
    texttospeech.MultiSpeakerMarkup.Turn(speaker=alias_for[t['speaker']], text=t['text'])  # speaker == ein deklarierter alias
    for t in chunk_turns])
synthesis_input = texttospeech.SynthesisInput(multi_speaker_markup=markup, prompt=style_prompt or None)
voice = texttospeech.VoiceSelectionParams(
    language_code=language_code, model_name=model_name,        # model_name PFLICHT um Gemini-TTS zu aktivieren
    multi_speaker_voice_config=texttospeech.MultiSpeakerVoiceConfig(speaker_voice_configs=speaker_configs))
    # KEIN name= hier (name + multi_speaker_voice_config schließen sich aus)
audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=24000)
response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
# response.audio_content = bytes (LINEAR16 PCM @24kHz mono 16-bit)
```
**Single-Speaker** (genau 1 distinct Label → **MUSS** über `text=`, nicht `multi_speaker_markup`):
```python
synthesis_input = texttospeech.SynthesisInput(text=turn_text, prompt=style_prompt or None)
voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_id, model_name=model_name)
# rest identisch (LINEAR16/24000, synthesize_speech)
```
- **`prompt`** lebt auf `SynthesisInput` (Feld 6, eigenes oneof) — **neben** dem Transkript-Feld gesetzt, **nie** hineinkonkateniert. Nicht auf AudioConfig/VoiceSelectionParams.
- **`speaker_alias`** = Label-Seite (Join-Key zu `Turn.speaker`); **`speaker_id`** = die Gemini-Voice. Aus dem Agent-`voices`-Map `{label: voice}` → `speaker_alias`=(sanitized) label, `speaker_id`=voice. Jeder `Turn.speaker` MUSS exakt einem deklarierten `speaker_alias` gleichen.
- **WAV** (Quellen widersprechen sich, ob LINEAR16 schon einen RIFF-Header trägt → **header-agnostisch** wrappen): `if audio_content[:4]==b'RIFF'` → schon WAV (für Concat ggf. Frames extrahieren), sonst via stdlib `wave` (1ch, 2 sampwidth, 24000) zu WAV wrappen. Dann pydub-Concat der Chunk-WAVs.
- **Chunking byte-basiert** (deutsche Umlaute/ß sind multi-byte → `len(text.encode('utf-8'))`, nicht `len`): ganze Turns akkumulieren bis ~**3500 Bytes** Transkript (Headroom unter dem von Quelle 1 genannten 4000-Byte-Cap), **nie mitten in einer Turn splitten**; eine einzelne Turn über dem Cap an Satzgrenzen in mehrere Same-Speaker-Turns splitten (wie `split_long_dialogue_turns`). **Derselbe `prompt` + dieselben 2 speaker_configs auf JEDEM Chunk** (Voice/Style stabil).
- **Error-Handling** (Cloud-SDK, **kein** `finishReason`/`candidates` — das war der genai-Pfad): `google.api_core.exceptions.*` — `InvalidArgument` (400: Turn.speaker ohne Alias / >2 Voices / Übergröße / falscher model_name), `NotFound` (404: Modell im Projekt/Region nicht aktiviert), `ResourceExhausted` (429), `DeadlineExceeded`. **Retry mit Backoff** auf 429/503/DeadlineExceeded, **nicht** auf 400. **Empty `audio_content` = harter Chunk-Fehler** (nicht still Stille concaten). Pre-Flight-Input-Validierung (jedes Label hat ein voices-Mapping, ≤2, Alias alphanumerisch, non-empty text).
- **Backward-Compat LOW**: der bestehende `GoogleTTSService.synthesize_speech` (Standard-Neural: `text=`/`name=`/MP3/`speaking_rate`/`pitch` + `list_voices`) nutzt **nur** stabile, lang-vorhandene Felder; 2.21→≥2.31 ist **rein additiv**. Verifizieren via Rebuild + `pytest`.

**Diskrepanzen (Quellen uneinig → defensiv bauen + Phase-3-Live-Verify):** (1) **Modell-Name** `gemini-2.5-flash-tts` (Quelle 1/3, Docs+Notebook) vs `gemini-2.5-flash-preview-tts` (Quelle 2, = genai-Name) → **konfigurierbar**, Default `gemini-2.5-flash-tts`, **live verifizieren**. (2) **WAV-Header** auf LINEAR16 → header-agnostisch (s.o.). (3) **Byte-Limit** 4000 (Quelle 1) vs undokumentiert (Quelle 2) → konservativ ~3500. (4) **Feldname** `multi_speaker_voice_config` auf VoiceSelectionParams (Quelle 2 self-flagged unbestätigt) → **lokal per `hasattr`/Konstruktion prüfen**. (5) **exactly-2 vs up-to-2** speaker_configs → 1 Speaker geht über den `text=`-Pfad (umgeht das).

## Verifizierte Code-Fakten (Master-gegroundet)

- **`services/google_tts_service.py::GoogleTTSService`** — hält `self.client = texttospeech.TextToSpeechClient()`; bestehendes `synthesize_speech(text, voice_name, language_code, speaking_rate, pitch)` (Standard-Neural, MP3) + `list_voices()`. **Client wiederverwenden, alte Methoden unberührt.** Singleton `app.google_tts_service` (Test-Patch-Punkt).
- **`services/gemini/audio.py::concatenate_with_pydub(chunk_files)`** — nimmt eine Liste Chunk-WAV-**Pfade**, gibt einen konkatenierten Pfad. **Reuse** für den Multi-Chunk-Fall.
- **NARR-1 `synthesize_turns`** ([services/gemini/tts.py](services/gemini/tts.py)) + **`GeminiService.synthesize_turns`** ([services/gemini/__init__.py](services/gemini/__init__.py)) + Tests ([tests/test_narration_synth.py](tests/test_narration_synth.py)): die **Validierung** (non-blank speaker/text, mode↔Speaker-Zahl, voices-Abdeckung, voices-ist-Dict) + das Turns-Mapping werden **in das neue pure Modul gezogen**; die genai-Facade wird abgelöst. **`generate_podcast` + seine voices/filter_metadata-Params bleiben** (Alt-Flow byte-identisch; optional: zurückbauen, falls sauberer — aber nicht den Alt-Flow brechen).
- Kontrakt-Feld bleibt **`speaker`** (das Label; konsistent mit NARR-1 + Design-Doc) — der Renderer sanitizet es zum Alias.

## Phase 1 — Dep-Bump + pures Renderer-Modul + Unit-Tests

1. **`requirements.txt:14`** `google-cloud-texttospeech==2.21.0` → **`google-cloud-texttospeech>=2.31.0`** (oder exakt-getesteten Pin). Lokal installieren.
2. **SDK-Shape lokal verifizieren** (kein Netz nötig — Introspektion): in einer REPL/einem Throwaway prüfen, dass in der installierten Version existieren: `texttospeech.MultispeakerPrebuiltVoice(speaker_alias=…, speaker_id=…)`, `texttospeech.MultiSpeakerMarkup` + `.Turn(speaker=…, text=…)`, `texttospeech.MultiSpeakerVoiceConfig(speaker_voice_configs=…)`, `texttospeech.SynthesisInput(prompt=…)`, und der Feldname **`multi_speaker_voice_config`** auf `VoiceSelectionParams` (`hasattr`/Konstruktion). Diskrepanz (4) damit auflösen. Im Bericht festhalten.
3. **`services/narration_render.py`** (pures Modul, `texttospeech`-Client als Arg → mockbar):
   - `validate_turns(turns, voices, mode)` — die **aus NARR-1 gezogene** Validierung (ValueError bei Verstoß).
   - Byte-Chunking-Helper (ganze Turns, ~3500-Byte-utf-8-Cap, Übergröße-Turn an Satzgrenzen splitten).
   - `pcm_to_wav_bytes(audio_content, rate=24000, ch=1, width=2)` — **header-agnostisch** (RIFF erkennen → Frames extrahieren, sonst via `wave` wrappen).
   - `render_turns(client, turns, voices, *, style_prompt=None, mode='two_speaker', language_code='de-DE', model_name='gemini-2.5-flash-tts', pydub_available=True) -> wav_path`: validate → Alias-Sanitizing + Mapping → Chunken → pro Chunk Multi- **oder** Single-Speaker-`SynthesisInput` (s. Skeletons) → `client.synthesize_speech` (api_core-Error-Handling + Retry-Backoff auf 429/503/Deadline; empty-audio = Fehler) → `pcm_to_wav_bytes` → Temp-WAV → bei >1 Chunk `concatenate_with_pydub`.
4. **Unit-Tests** (`tests/test_narration_render.py`; **Mock `client.synthesize_speech`**, das `input`/`voice`/`audio_config`-Arg **capturen** + ein Fake-`audio_content` zurückgeben):
   - Multi-Speaker: 2 Labels → `SpeakerVoiceConfig`s mit `speaker_alias`=Alias + `speaker_id`=gemappte Voice; jeder `Turn.speaker` == ein Alias; `prompt` auf SynthesisInput (nicht im Transkript); `model_name` gesetzt; `name=` NICHT gesetzt; LINEAR16/24000.
   - Single-Speaker: 1 Label → `SynthesisInput(text=…, prompt=…)` + `VoiceSelectionParams(name=voice, model_name=…)` (kein multi_speaker_markup). **Behebt den NARR-1-Single-Speaker-Gap** (Voice korrekt gemappt).
   - Byte-Chunking: viele/lange Turns → mehrere Chunks, **nie mitten in Turn**, jeder Chunk trägt denselben `prompt` + dieselben speaker_configs; utf-8-Byte-Messung (Umlaut-Test). Übergröße-Einzel-Turn → an Satzgrenzen gesplittet.
   - WAV header-agnostisch: Fake-`audio_content` mit **und** ohne RIFF-Präfix → beide ergeben valides WAV.
   - Validierung (aus NARR-1 gespiegelt): >2 Speaker → ValueError; mode-Mismatch; fehlende Voice; non-blank.
   - Error-Handling: `client.synthesize_speech` wirft `InvalidArgument` → propagiert (kein Retry); `ResourceExhausted` → Retry dann Erfolg; empty audio_content → Fehler.
5. Pre-Flight `pytest tests/` grün.

**Stop + Bericht.**

## Phase 2 — Öffentliche Narration-Entry + NARR-1-Facade ablösen + Backward-Compat

1. **Öffentlicher Entry** auf `GoogleTTSService` (besitzt den `texttospeech`-Client): `synthesize_narration(self, turns, voices, *, style_prompt=None, mode='two_speaker', language_code='de-DE', model_name=DEFAULT_NARRATION_MODEL) -> wav_path` → `narration_render.render_turns(self.client, …)`. (`DEFAULT_NARRATION_MODEL='gemini-2.5-flash-tts'` als Modul-Konstante/Env-konfigurierbar.) Singleton `app.google_tts_service` = Test-Patch-Punkt.
2. **NARR-1-`GeminiService.synthesize_turns` ablösen**: entfernen (es zeigte auf den genai-Pfad) — die Kontrakt-Validierungs-Tests aus `test_narration_synth.py` auf `narration_render.validate_turns` **repointen** (Validierung lebt jetzt dort). Die `generate_podcast`-voices/filter_metadata-Params + ihre Tests: **bleiben** (Alt-Flow, inert) **oder** sauber zurückbauen — Alt-Flow darf nicht brechen (die `test_narration_synth.py`-Tests „alter-Flow-byte-identisch" müssen grün bleiben bzw. mitgezogen werden).
3. **Backward-Compat**: nach dem Dep-Bump `pytest tests/` grün — besonders die bestehenden `GoogleTTSService`/Podcast-Tests (Standard-Neural unberührt). Falls Docker-relevant: Image rebuild ist Olis Deploy-Schritt (hier nur pytest).
4. Pre-Flight `pytest tests/` grün.

**Stop + Bericht.**

## Phase 3 — Live-Verify-Gate + Docs + Wrap

1. **Live-Verify** (die Netz-/Projekt-abhängigen Offenpunkte — **falls** valide `google-credentials.json` + Modell im Projekt erreichbar, macht's der Sub-Thread als One-Shot; **sonst dokumentierter Smoke-Gate für Oli**):
   - **Modell-Name**: ein 1-Zeilen-`synthesize_speech` mit `gemini-2.5-flash-tts` — liefert Audio? Sonst `gemini-2.5-flash-preview-tts` probieren. Den funktionierenden String pinnen.
   - **WAV-Header**: `audio_content[:4] == b'RIFF'`? — bestätigt, welcher Wrap-Zweig greift (der Renderer ist eh header-agnostisch).
   - **Byte-Limit** (optional): grob die multi_speaker_markup-Grenze abtasten (oder beim konservativen ~3500 bleiben).
   - **de-DE-Voice-Katalog**: welche `speaker_id`s (Kore/Charon/Puck/…) für Deutsch gültig sind (informiert NARR-4).
   Ergebnisse im Bericht; den Renderer-Default ggf. auf den verifizierten Modell-Namen setzen.
2. **`CLAUDE.md`** — die „Gemini Models Used"-Sektion: der **Cloud-TTS-Pfad** nutzt `gemini-2.5-flash-tts` (**ohne** `-preview-`-Infix, anders als der genai-Name) für die Narration; Notiz zur Pfad-Unterscheidung.
3. **`docs/narration_reframe.md`** — NARR-1B ☑ + den live-verifizierten Modell-Namen/WAV-Befund einpflegen.
4. **STATUS.md** + **BACKLOG.md**: NARR-1B ☑ done (Hashes); Sequenz `1✅→1B✅→2…`. **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md`).
5. **Memory** (`reference_*`): die Cloud-Gemini-TTS-Render-Mechanik (speaker_alias≠speaker_id, prompt-als-eigenes-Feld-kein-Leakage, single-via-text/multi-via-markup, byte-Chunking-utf-8, header-agnostisches WAV, Cloud-Modell-Name ≠ genai-Name, Backward-Compat-additiv) — wiederverwendbar für künftige TTS-Arbeit. MEMORY.md-Pointer.
6. **Sprint-Doc** (dieses File) einchecken. Finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Deploy-Notiz: Mintbox `git pull` + `docker compose up -d --build` (**Dep-Bump `google-cloud-texttospeech` → Image rebuildt**; kein Schema/Token/Migration). Der Renderer wird erst ab NARR-3 (Endpoint) live aufgerufen; bis dahin ändert sich am laufenden Verhalten nichts.

## Bewusst NICHT (Scope-Grenze)

- **Kein** Endpoint/Persistenz/UI/Token (NARR-2/3/5).
- **Kein** Anfassen des Standard-Neural-`GoogleTTSService.synthesize_speech` + `list_voices` (Alt-Pfad bleibt).
- **Kein** Löschen der genai-Hack-Funktionen (`calculate_tag_guidance` etc.) + des Alt-Flows (bleibt bis NARR-5).
- **Kein** Director's-Notes-Authoring (das ist NARR-4-Skill; hier nur das **Durchreichen** von `style_prompt`).
- **Kein** v1beta1-Surface, **kein** `AdvancedVoiceOptions`/Safety-Relax (nur falls je nötig).

## Akzeptanz

- [ ] `google-cloud-texttospeech>=2.31.0`; SDK-Shape lokal verifiziert (inkl. Feldname `multi_speaker_voice_config`).
- [ ] `services/narration_render.py`: `render_turns` (multi via `multi_speaker_markup`, single via `text=`), `speaker_alias`(sanitized)↔`speaker_id`(voice), `prompt` als eigenes Feld, byte-utf-8-Chunking (~3500, nie mitten-in-Turn), header-agnostisches WAV, pydub-Concat, api_core-Error-Handling + Retry; `validate_turns` aus NARR-1 gezogen.
- [ ] `GoogleTTSService.synthesize_narration`-Entry → `render_turns(self.client,…)`; NARR-1-`GeminiService.synthesize_turns` abgelöst, Validierungs-Tests repointet; **Standard-Neural-Pfad + Alt-Flow byte-identisch/grün**.
- [ ] Unit-Tests am `synthesize_speech`-Boundary (Multi/Single/Chunking/WAV/Validierung/Error). `pytest` grün ≥ 560 (± repointete NARR-1-Tests).
- [ ] Live-Verify-Befunde (Modell-Name/WAV-Header) dokumentiert; CLAUDE.md + Design-Doc + STATUS/BACKLOG + Memory aktualisiert; Dep-Bump backward-compat-verifiziert.
