# Podcast → treue Dokument-Vertonung (Erklärbär-Narration) — Design / „großer Wurf"

> **Status**: vorbereitet 2026-06-28 (Workshop Oli + Master, recherche-gestützt). Kohärenter Entwurf, **phasiert** in 5 Sprints (NARR-1…5) geliefert. Dieses Doc ist die **Referenz**, die die Sprint-Prompts zitieren — nicht jeder Sprint leitet die Architektur neu her.
>
> **Recherche-Grundlage**: Workflow `podcast-tts-platform-research` (7 Agenten, Gemini/ElevenLabs/OpenAI/Azure/Deepgram). Kern-Code-Anker unten sind **Master-verifiziert** (existieren real); jedes Sprint-Prompt macht zusätzlich file-level-Grounding beim Schreiben.
>
> **Pivot 2026-06-29**: v1-Render-Engine = **Gemini-TTS über den Cloud-TTS-Pfad** (`google-cloud-texttospeech` ≥2.31.0), **nicht** der Gemini-API-Pfad — s. §Plattform-Pfad + Grounding-Workflow `cloud-gemini-tts-api-grounding`. NARR-1 P1 (Kontrakt/Validierung/Facade) bleibt gültig; der Renderer kommt in **NARR-1B**.

## Worum es geht

Die alte „Podcast"-Funktion erzeugte **unkontrollierte, lockere Paraphrasen** — kaum genutzt. Reframe: **treue Vertonung von Erklärbär-Dokumenten** — der Doc-Inhalt wird, fürs Hören geglättet, aber **inhaltstreu** (nicht paraphrasiert), als 1- oder 2-Speaker-Audio wiedergegeben; **agent-triggerbar** (Claude schreibt das Skript); das Audio wird **erstklassiges Library-Element**.

## Die Kern-Erkenntnis (verifiziert)

Der „brutale Hack" ist **nicht** Gemini, sondern CONVERTERs eigener Code:
- **`services/gemini/prompts.py::calculate_tag_guidance`** — `recommended_tags = max(int(char_count / chars_per_tag), 2)` + `_TAG_DENSITY_MAP`. Die „Rechen-Operation", die Tag-Zahlen rät. **Stirbt.**
- **`services/gemini/prompts.py`**-Prompts „Transform the source … do NOT just summarize or rephrase" + **`services/gemini/script.py::format_dialogue_with_llm`** (ruft `gemini-2.5-flash` zum *Paraphrasieren*). **Stirbt** im Treue-Pfad.
- **`services/gemini/dialogue.py::parse_dialogue`** (`line.split(':', 1)`) — fragiler „Name:"-Parser. **Ersetzt** durch strukturierte Daten.

Gemini-TTS **rezitiert wörtlich** (Default) und kann **nativ ≤2 Speaker**. Der Kern-Fix ist **Löschen** der Skript-Gen/Tag-Arithmetik (oben) — *plus* (Pivot 2026-06-29) der Render-Engine-Wechsel vom Gemini-API-Pfad auf den **Cloud-TTS-Pfad**, der Turns/`prompt`/Voice strukturell sauber trennt (s. §Plattform-Pfad). Die alte `MultiSpeakerVoiceConfig`-Verdrahtung im API-Pfad bleibt für den Alt-Flow.

> **Sequenz-Hinweis (wichtig)**: Der alte chatty-Flow **nutzt** `calculate_tag_guidance`/`format_dialogue_with_llm` bis zu seiner Stilllegung. Darum **baut NARR-1 den Treue-Pfad neu, ohne diese Funktionen aufzurufen** — **gelöscht** werden sie erst mit der Alt-Flow-Abschaltung in **NARR-5**. So bleibt jeder Sprint grün und der alte Flow läuft bis zur bewussten Abschaltung weiter.

## Gelockte Entscheidungen (Oli, 2026-06-28)

- **Treue-Grad**: „treu, fürs Hören geglättet" — die **Glättung passiert beim Authoring** (Claude), Gemini **rezitiert das Skript verbatim**. Zwei Schichten, kein Widerspruch.
- **Alter chatty-Modus**: **ersetzt** (raus). Ein klarer Modus.
- **Scope**: kohärenter Entwurf, **phasierte** Lieferung (5 Sprints), jeder grün.
- **Plattform v1**: **Gemini-TTS über den Cloud-TTS-Pfad** (`google-cloud-texttospeech` ≥2.31.0 — Pivot 2026-06-29, s. §Plattform-Pfad). Native `multi_speaker_markup` + separates `prompt`-Feld + `MultispeakerPrebuiltVoice(speaker_alias, speaker_id)` → Label↔Voice nativ, Director's-Notes ohne Leakage, dokumentierte Tags. Turn-Listen-Architektur bleibt plattform-neutral → ElevenLabs späterer **Render-Schicht-Swap**, falls Stimmqualität (nicht Treue) der Frust wird.
- **Defaults**: Flash als Default (Pro opt-in pro Vertonung); Chunk-Nähte für v1 akzeptiert; Treue = Skill-Kontrakt + leichter Server-Sanity-Check, **kein** harter Verbatim-Gate (würde die gewollte Glue verbieten).

## Architektur (vier Schichten)

1. **Authoring (Claude-Skill, lebt in Claude, nicht CONVERTER)** — liest den Doc, schreibt die treue, geglättete **Turn-Liste**, mappt Speaker → Gemini-Voice, POSTet an CONVERTER. *Die Intelligenz lebt hier.*
2. **Token-Endpoint + async (CONVERTER, NARR-3)** — `POST /api/narrations`, Bearer `NARRATION_TOKEN` (eigener Token, Spiegel von `CARD_TOKEN`; **billing-rationale** — Narration kostet GCP-Geld pro Call → unabhängig revozierbar), CSRF-exempt, fail-closed. Legt sofort eine `pending`-Conversion an + enqueued den RQ-Task. **⚠️ Der Worker-Container hat KEINEN DB-Zugriff** (mountet `podcast_data`, **nicht** `app_data` — verifiziert in docker-compose) → der **Worker bleibt DB-frei** (rendert → schreibt `narration_<id>.wav` → returnt), und die **Web-Seite rekonziliert beim Pollen** (`GET /api/narrations/<id>`: Datei existiert → `ready` + Dauer; Job failed/abgelaufen → `failed`). Spiegelt den DB-freien Alt-Podcast-Flow; **keine Infra-Änderung**.
3. **Render (CONVERTER)** — empfängt die Turn-Liste, rendert über den **Cloud-TTS-Pfad** (`google-cloud-texttospeech` ≥2.31.0: `SynthesisInput.multi_speaker_markup`-Turns + `prompt` + `MultispeakerPrebuiltVoice(speaker_alias=Label, speaker_id=Voice)`, Modell `gemini-2.5-flash-tts`), **Byte-Chunking** (≤4000 Bytes/Markup, nie Tag-Zahl) + Concat → WAV. **Kein** server-seitiges Paraphrasieren, **keine** Tag-Arithmetik.
4. **Persistenz (CONVERTER)** — Audio wird erstklassiges Library-Element.

### Der Turn-Listen-Kontrakt (plattform-neutral)

```json
{
  "title": "Wie funktioniert FSRS?",
  "language": "de",
  "tts_model": "gemini-2.5-flash-tts",
  "mode": "two_speaker",
  "voices": { "Anna": "Kore", "Ben": "Puck" },
  "turns": [
    { "speaker": "Anna", "text": "Spaced Repetition heißt: du wiederholst genau dann, wenn du kurz vorm Vergessen bist." },
    { "speaker": "Ben",  "text": "Und FSRS schätzt diesen Moment über zwei Werte: Stabilität und Schwierigkeit." }
  ]
}
```
Speaker-Zuordnung ist **Daten** (kein `:`-Parsing); Performance-Tags sind **optional**, vom Agenten bewusst gesetzt (Erklärbär-Default: spärlich/keine), nie berechnet.

### Persistenz-Modell (kein Schema-Touch)

- **Conversion wiederverwenden** (wie audio_transcription): `conversion_type='audio_narration'` (in `ALLOWED_CONVERSION_TYPES`, `app_pkg/library.py`), `content` = die Turn-Liste als lesbares Markdown (→ Reader/Suche/Tags/Highlights/`list_conversions` gratis), Audio-Felder in **`metadata_json`** (`narration_status`/`audio_filename`/`audio_mimetype`/`duration_seconds`/`tts_model`/`speakers`/`transcript`/`error`) — **keine neue Tabelle, keine Migration** (`metadata_json` ist die etablierte Escape-Hatch). **✅ NARR-2 gebaut** im puren [services/narration_library.py](narration_library.py): `narration_to_markdown` (multi = `**Label:** …`-Blöcke, single = label-frei, Text verbatim) · `narration_audio_path` · `build_narration_metadata` + dokumentierter Kontrakt · robuste Lese-Helfer (`narration_metadata`/`narration_status`/`narration_audio_filename`, fehlend/kaputt → Defaults).
- **Audio-Datei**: persistenter, deterministischer Pfad `OUTPUT_DIR/narration_<id>.wav` auf dem **`podcast_data`-Volume** (liegt schon auf **beiden** Containern — `docker-compose.yml:24+55` — Worker schreibt ab NARR-3, Web liefert), **nicht** löschen-beim-Serve. **Serve** `GET /api/narrations/<id>/audio` (neu `app_pkg/narration.py`, `@login_required`, owner-404, **type-404** (kein Typ-Leak), **status≠ready-404**, Traversal-Guard `is_relative_to(OUTPUT_DIR)` wie podcast_download, Pfad **id-abgeleitet** → kein Filename-Injection, **kein `os.unlink`**). **Delete-Cleanup** = bewusst die **Route-Variante** (**nicht** ORM-`before_delete`): in `api_delete_conversion` wird der Audio-Pfad **post-commit** — also nur bei erfolgreichem Row-Delete, nie mitten im Flush — best-effort + traversal-guarded via `narration_library.delete_narration_audio` unlinkt. Begründung: Conversions haben **keinen** Cascade-Lösch-Pfad (nur die Route löscht sie), und ein Unlink nach erfolgreichem Commit kann nie „Datei weg, Row aber zurückgerollt" erzeugen.

## Sprint-Sequenz (NARR-1…5)

| Sprint | Größe | Inhalt |
|---|---|---|
| **NARR-1** ✅ | M | **Treue-Synth-Kontrakt** (P1 done `d419271`): `synthesize_turns`-Entry + Validierung (mode/Speaker-Zahl/voices-Abdeckung) + `GeminiService`-Facade + 14 Tests; `voices`/`filter_metadata`-Params an `generate_podcast` (alter Flow byte-identisch). **Kontrakt/Validierung/Facade werden in NARR-1B wiederverwendet**; der API-Pfad-Renderer darunter wird durch den Cloud-Renderer ersetzt. (P2-Wrap entfällt — Master absorbiert wegen Pivot.) |
| **NARR-1B** ✅ | M | **Cloud-Gemini-TTS-Renderer** (Pivot 2026-06-29; **☑ done 2026-06-29** — P1 `fef8c36` · P2 `9c8debc` · P3 dieser Commit): Dep-Bump `google-cloud-texttospeech` 2.21.0→≥2.31.0; pures [services/narration_render.py](narration_render.py) (`render_turns`/`validate_turns`/`chunk_turns`/`pcm_to_wav_bytes`) über `SynthesisInput.multi_speaker_markup` + `prompt` (**eigenes Feld, kein Leak**) + `MultispeakerPrebuiltVoice(speaker_alias=Speaker1/2, speaker_id=Voice)`; **per-Chunk single/multi-Routing** (Monolog-Chunk → `text=`, deklariert nie einen ungenutzten Alias — behebt zugleich den NARR-1-Single-Speaker-Voice-Gap); byte-utf-8-Chunking (~3500, nie mitten-in-Turn), header-agnostisches WAV, pydub/wave-Concat; api_core-Error-Handling + Retry-Backoff (429/503/Deadline). Entry **`GoogleTTSService.synthesize_narration`**; NARR-1-genai-Facade abgelöst (Validierung lebt jetzt in `narration_render.validate_turns`). **Backward-Compat** Standard-Neural-`GoogleTTSService` + Alt-Podcast-Flow grün (pytest **580**). **Live-Verify ✅ 2026-06-29** (`gemini-2.5-flash-tts` greift, WAV=RIFF, multi-speaker ok) — s. §NARR-1B Live-Verify-Befund. |
| **NARR-2** ✅ | M | **Library-Persistenz** (**☑ done 2026-06-29** — P1 `3f7eaf5` · P2 `17310d7` · P3 dieser Commit): `audio_narration` in `ALLOWED_CONVERSION_TYPES`; pures [services/narration_library.py](narration_library.py) (`narration_to_markdown`/`narration_audio_path`/`build_narration_metadata` + Lese-Helfer + `delete_narration_audio`); Serve `GET /api/narrations/<id>/audio` in neuem `app_pkg/narration.py` (`@login_required`, owner-404, type-404, status≠ready-404, Traversal-Guard, **kein** Delete-on-serve); Delete-Cleanup **post-commit** in `api_delete_conversion` (best-effort, traversal-guarded, nur Narration). **Kein** Schema/Token/Dep; pytest **610**. |
| **NARR-3** | M | **Token-Endpoint + async**: `POST /api/narrations` (NARRATION_TOKEN, mirror `_authorize_card_write`, CSRF-exempt), Validierung, `pending`-Row sofort, `tasks.generate_narration_task` enqueue, Worker flippt ready/failed; `GET /api/narrations/<id>` Status. Auth-Matrix-Tests. |
| **NARR-4** | S/M | **Claude-Skill + Agent-Docs**: `erklaerbaer-narration`-Skill (Turn-Kontrakt, Voice-Katalog aus `_GEMINI_VOICES`, Treue-Regel: near-verbatim + Glue, keine Paraphrase, spärliche Tags, 1–2 Speaker, POST+Poll). converter-mcp-Brief/Agent-Guide. |
| **NARR-5** | S | **Library-UI-Player + alten Pfad stilllegen**: `<audio>`-Player für `audio_narration`, pending/failed/ready, Retry-aus-`metadata_json`; alter `/generate-gemini-podcast`-Paraphrase-Flow raus/abgeklemmt **+ die dann toten `calculate_tag_guidance`/`format_dialogue_with_llm` löschen**. Live-Smoke. |

Abhängigkeit: 1 ✅ → **1B ✅** → **2 ✅** → 3 → (4, 5). **NARR-3 (Token-Endpoint + RQ-Worker — verbindet Renderer + Persistenz live) ist der nächste Sprint.**

### NARR-1B Live-Verify-Befund (2026-06-29) — ✅ verifiziert

Der Renderer ist **defensiv** gebaut (header-agnostisches WAV, Modell-Name konfigurierbar via `NARRATION_TTS_MODEL`, byte-Chunking) → **korrekt unabhängig** von den unten offenen API-Fakten; das Live-Verify pinnt nur den finalen Default + bestätigt. SDK-Shape lokal verifiziert (`google-cloud-texttospeech` 2.36.0): alle Skeleton-Typen existieren, **`multi_speaker_voice_config` ist das echte Feld auf `VoiceSelectionParams`** (Diskrepanz #4 aufgelöst), `SynthesisInput.prompt` ist ein eigenes Feld neben dem `text`/`multi_speaker_markup`-oneof (Director's-Notes ohne Leak strukturell bestätigt).

**Live-One-Shot durchgeführt + ✅ verifiziert** (Mac, `google-credentials.json` → SA `podcast-tts@podcasts-476919`): **(1) Modell-Name = `gemini-2.5-flash-tts`** — greift (207 KB Audio); der `-preview-tts`-Name **500t** (`InternalServerError`), ist also **falsch** für den Cloud-Pfad. **(2) WAV-Header: LINEAR16 trägt `RIFF`** → `pcm_to_wav_bytes` macht Passthrough. **(3) Multi-Speaker bestätigt** (192 KB). Der Renderer-Default `gemini-2.5-flash-tts` + das header-agnostische Wrapping waren **schon korrekt** → **kein Code-Change**, reiner Doku-Pin. (de-DE-Voices `Kore`/`Puck` live bestätigt; voller Katalog für NARR-4 separat.)

**GCP-Setup, das den Zugriff freischaltete** (Projekt `podcasts-476919`, einmalig, Owner-/Billing-Schritt — gilt **projektweit**, also auch für die Mintbox-Prod): **Cloud-Text-to-Speech-API** aktiviert · **Agent-Platform/Vertex-AI-API** (`aiplatform.googleapis.com`) aktiviert (die Gemini-TTS-Modelle routen über Vertex-Publisher → ohne diese API 403 „Agent Platform API not used") · SA `podcast-tts@` Rolle **„Agent Platform User"** (= Vertex AI User, `roles/aiplatform.user` → liefert das fehlende `aiplatform.endpoints.predict`). Damit ist das Live-Verify **erledigt**. Der Renderer wird **erst ab NARR-3 (Endpoint) live aufgerufen** → bis dahin ändert sich am laufenden Verhalten nichts.

## Plattform-Pfad: Gemini-API vs Cloud TTS (aus der Tag-Recherche 2026-06-28)

Die Cowork-Tag-Recherche ([docs/narration_tag_doctrine.md](narration_tag_doctrine.md)) deckte einen Fork auf, der vorher nicht explizit war:

- **CONVERTER nutzt heute den Gemini-API-Pfad** (`client.models.generate_content(model='gemini-2.5-flash-preview-tts', contents=…)`). Darauf gibt es **keine offiziellen Inline-Tags**; Stil-Steuerung läuft nur über einen Natural-Language-Prompt **im selben `contents`-String** → Prompt-Leakage-Risiko, wenn man Director's Notes voranstellt.
- Der **Cloud-TTS/Vertex-Pfad** (`texttospeech`) trennt `prompt` strukturell vom `text`/`multiSpeakerMarkup`, hat dokumentierte Markup-Tags (Preview) und mappt sauber auf `[{speaker,text}]` (`MultiSpeakerMarkup.Turn`) — für eine Turn-Liste der *bessere* Pfad, aber eine **neue Integration**.

**Korroboriert den Reframe**: die alten `[excited]`/`[pause]`-Inline-Tags des chatty-Flows lagen auf dem API-Pfad, der sie **nicht offiziell unterstützt** → mitgesprochen/ignoriert. Das war — neben der Tag-Arithmetik — ein **zweiter** Grund für „unkontrolliert".

**Entscheidung (Oli, 2026-06-29) — v1 = Cloud-TTS-Pfad (Pivot, „gleich richtig machen")**: Master-verifiziert, dass `google-cloud-texttospeech` **≥2.31.0** Gemini-TTS nativ kann — `SynthesisInput.multi_speaker_markup` (strukturierte Turns) + separates `prompt`-Feld + `MultispeakerPrebuiltVoice(speaker_alias=Label, speaker_id=Voice)` + Modelle `gemini-2.5-flash-tts`/`pro-tts`. Das löst **drei** Dinge auf einmal, die der API-Pfad nicht konnte: **native Label↔Voice-Trennung** (auch Single-Speaker korrekt — behebt den NARR-1-P1-Single-Speaker-Gap nativ), **Director's-Notes ohne Leakage** (`prompt` strukturell vom Transkript getrennt), **dokumentierte Tags**. Infra größtenteils da (Credentials, SDK installiert, `GoogleTTSService`-Pattern) → Kosten = **Dep-Bump 2.21.0→≥2.31.0** + ein neuer Cloud-Renderer. **NARR-1 P1 bleibt gültig** (Kontrakt/Validierung/Facade wiederverwendet); nur der Renderer darunter wechselt → **NARR-1B**. Exakte API: Grounding-Workflow `cloud-gemini-tts-api-grounding`.

## Bewusst später / nicht v1

- **ElevenLabs** (Text-to-Dialogue) — Render-Schicht-Swap, falls Stimmqualität der bindende Frust wird. Strukturierte Turns nativ; Kosten ~$22/mo + Verbatim-Guardrail (kann ad-libben). Die Turn-Liste macht den Swap billig.
- **Azure Batch + SSML `<voice>`** — die **einzige** Option, die Chunk-Nähte ganz killt (ein Job pro Doc). Neuer Stack + SSML-Authoring → nur, falls die Concat-Nähte real stören.
- **Pro-TTS als Default**, **harter Verbatim-Gate** — bewusst nicht v1.

## Offene Code-Grounding-Notiz

Die Anker oben sind verifiziert (existieren). Jedes NARR-Sprint-Prompt macht beim Schreiben zusätzlich **file-level-Grounding** der konkret berührten Funktionen (Master-Disziplin) — der Workflow-Code-Reader hatte einen Schema-Fehlschlag, daher nicht blind auf relayte Zeilennummern verlassen.
