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
2. **Token-Endpoint (CONVERTER)** — `POST /api/narrations`, Bearer `NARRATION_TOKEN` (Spiegel von `CARD_TOKEN`), CSRF-exempt, fail-closed, async via bestehenden RQ-Worker.
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

- **Conversion wiederverwenden** (wie audio_transcription): `conversion_type='audio_narration'` (zu `ALLOWED_CONVERSION_TYPES` in `app_pkg/library.py`), `content` = die Turn-Liste als lesbares Markdown (→ Reader/Suche/Tags/Highlights/`list_conversions` gratis), Audio-Felder in **`metadata_json`** (`narration_status`/`audio_filename`/`duration`/`transcript`/`speakers`) — **keine neue Tabelle, keine Migration** (`metadata_json` ist die etablierte Escape-Hatch).
- **Audio-Datei**: persistenter, deterministischer Pfad `narration_<id>.wav` auf dem **`podcast_data`-Volume** (liegt schon auf **beiden** Containern — `docker-compose.yml:24+55` — Worker schreibt, Web liefert), **nicht** löschen-beim-Download. Serve via `GET /api/narrations/<id>/audio` (`@login_required`, owner-404, Traversal-Guard wie podcast_download). Delete-Cleanup-Hook (unlink Audio bei Conversion-Delete, ORM-seitig wie `Highlight before_delete`).

## Sprint-Sequenz (NARR-1…5)

| Sprint | Größe | Inhalt |
|---|---|---|
| **NARR-1** ✅ | M | **Treue-Synth-Kontrakt** (P1 done `d419271`): `synthesize_turns`-Entry + Validierung (mode/Speaker-Zahl/voices-Abdeckung) + `GeminiService`-Facade + 14 Tests; `voices`/`filter_metadata`-Params an `generate_podcast` (alter Flow byte-identisch). **Kontrakt/Validierung/Facade werden in NARR-1B wiederverwendet**; der API-Pfad-Renderer darunter wird durch den Cloud-Renderer ersetzt. (P2-Wrap entfällt — Master absorbiert wegen Pivot.) |
| **NARR-1B** ✅ | M | **Cloud-Gemini-TTS-Renderer** (Pivot 2026-06-29; **☑ done 2026-06-29** — P1 `fef8c36` · P2 `9c8debc` · P3 dieser Commit): Dep-Bump `google-cloud-texttospeech` 2.21.0→≥2.31.0; pures [services/narration_render.py](narration_render.py) (`render_turns`/`validate_turns`/`chunk_turns`/`pcm_to_wav_bytes`) über `SynthesisInput.multi_speaker_markup` + `prompt` (**eigenes Feld, kein Leak**) + `MultispeakerPrebuiltVoice(speaker_alias=Speaker1/2, speaker_id=Voice)`; **per-Chunk single/multi-Routing** (Monolog-Chunk → `text=`, deklariert nie einen ungenutzten Alias — behebt zugleich den NARR-1-Single-Speaker-Voice-Gap); byte-utf-8-Chunking (~3500, nie mitten-in-Turn), header-agnostisches WAV, pydub/wave-Concat; api_core-Error-Handling + Retry-Backoff (429/503/Deadline). Entry **`GoogleTTSService.synthesize_narration`**; NARR-1-genai-Facade abgelöst (Validierung lebt jetzt in `narration_render.validate_turns`). **Backward-Compat** Standard-Neural-`GoogleTTSService` + Alt-Podcast-Flow grün (pytest **580**). **Live-Verify offen** — s. §NARR-1B Live-Verify-Befund. |
| **NARR-2** | M | **Library-Persistenz**: `audio_narration`-Conversion + `metadata_json`-Kontrakt + Turn-Liste→Markdown; persistenter Audio-Pfad + Serve-Endpoint (owner-404, Traversal-Guard, **nicht** löschen-beim-Serve) + Delete-Cleanup-Hook. |
| **NARR-3** | M | **Token-Endpoint + async**: `POST /api/narrations` (NARRATION_TOKEN, mirror `_authorize_card_write`, CSRF-exempt), Validierung, `pending`-Row sofort, `tasks.generate_narration_task` enqueue, Worker flippt ready/failed; `GET /api/narrations/<id>` Status. Auth-Matrix-Tests. |
| **NARR-4** | S/M | **Claude-Skill + Agent-Docs**: `erklaerbaer-narration`-Skill (Turn-Kontrakt, Voice-Katalog aus `_GEMINI_VOICES`, Treue-Regel: near-verbatim + Glue, keine Paraphrase, spärliche Tags, 1–2 Speaker, POST+Poll). converter-mcp-Brief/Agent-Guide. |
| **NARR-5** | S | **Library-UI-Player + alten Pfad stilllegen**: `<audio>`-Player für `audio_narration`, pending/failed/ready, Retry-aus-`metadata_json`; alter `/generate-gemini-podcast`-Paraphrase-Flow raus/abgeklemmt **+ die dann toten `calculate_tag_guidance`/`format_dialogue_with_llm` löschen**. Live-Smoke. |

Abhängigkeit: 1 ✅ → **1B ✅** → 2 → 3 → (4, 5). **NARR-2 (Library-Persistenz) ist der nächste Sprint.**

### NARR-1B Live-Verify-Befund (2026-06-29) — Smoke-Gate für Oli

Der Renderer ist **defensiv** gebaut (header-agnostisches WAV, Modell-Name konfigurierbar via `NARRATION_TTS_MODEL`, byte-Chunking) → **korrekt unabhängig** von den unten offenen API-Fakten; das Live-Verify pinnt nur den finalen Default + bestätigt. SDK-Shape lokal verifiziert (`google-cloud-texttospeech` 2.36.0): alle Skeleton-Typen existieren, **`multi_speaker_voice_config` ist das echte Feld auf `VoiceSelectionParams`** (Diskrepanz #4 aufgelöst), `SynthesisInput.prompt` ist ein eigenes Feld neben dem `text`/`multi_speaker_markup`-oneof (Director's-Notes ohne Leak strukturell bestätigt).

**Live-One-Shot versucht** (Mac, `google-credentials.json` → Service-Account `podcast-tts@podcasts-476919`): Credentials + Projekt lösen **korrekt** auf, aber die **Cloud-Text-to-Speech-API ist im GCP-Projekt `podcasts-476919` (Projekt-Nr. `465831190001`) deaktiviert** → `403 PermissionDenied` am API-Enablement-Gate, **vor** jeder Modell-/Voice-Auflösung. Damit **noch nicht final pinbar**: (1) Modell-Name `gemini-2.5-flash-tts` vs `gemini-2.5-flash-preview-tts`, (2) WAV-Header-Präsenz (`audio_content[:4]==b'RIFF'`?), (3) de-DE-Gemini-Voice-Katalog (Kore/Charon/Puck/… für NARR-4).

**Olis Smoke-Gate** (wie KINDLE — der echte Done-Gate für die Live-Fakten): Cloud-Text-to-Speech-API im Projekt `podcasts-476919` aktivieren (Console-Link steckt im 403), dann den One-Shot re-run — 1-Zeilen-`synthesize_speech` mit `gemini-2.5-flash-tts`, single-speaker `text=`, `language_code='de-DE'`, voice `Kore` → liefert Audio? sonst `gemini-2.5-flash-preview-tts` probieren. Den funktionierenden String als `NARRATION_TTS_MODEL`/`DEFAULT_NARRATION_MODEL` pinnen; `audio_content[:4]==b'RIFF'` prüfen (welcher Wrap-Zweig greift — Renderer ist eh header-agnostisch). Der Renderer wird **erst ab NARR-3 (Endpoint) live aufgerufen** → bis dahin ändert sich am laufenden Verhalten nichts.

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
