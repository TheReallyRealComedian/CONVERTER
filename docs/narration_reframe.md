# Podcast → treue Dokument-Vertonung (Erklärbär-Narration) — Design / „großer Wurf"

> **Status**: vorbereitet 2026-06-28 (Workshop Oli + Master, recherche-gestützt). Kohärenter Entwurf, **phasiert** in 5 Sprints (NARR-1…5) geliefert. Dieses Doc ist die **Referenz**, die die Sprint-Prompts zitieren — nicht jeder Sprint leitet die Architektur neu her.
>
> **Recherche-Grundlage**: Workflow `podcast-tts-platform-research` (7 Agenten, Gemini/ElevenLabs/OpenAI/Azure/Deepgram). Kern-Code-Anker unten sind **Master-verifiziert** (existieren real); jedes Sprint-Prompt macht zusätzlich file-level-Grounding beim Schreiben.

## Worum es geht

Die alte „Podcast"-Funktion erzeugte **unkontrollierte, lockere Paraphrasen** — kaum genutzt. Reframe: **treue Vertonung von Erklärbär-Dokumenten** — der Doc-Inhalt wird, fürs Hören geglättet, aber **inhaltstreu** (nicht paraphrasiert), als 1- oder 2-Speaker-Audio wiedergegeben; **agent-triggerbar** (Claude schreibt das Skript); das Audio wird **erstklassiges Library-Element**.

## Die Kern-Erkenntnis (verifiziert)

Der „brutale Hack" ist **nicht** Gemini, sondern CONVERTERs eigener Code:
- **`services/gemini/prompts.py::calculate_tag_guidance`** — `recommended_tags = max(int(char_count / chars_per_tag), 2)` + `_TAG_DENSITY_MAP`. Die „Rechen-Operation", die Tag-Zahlen rät. **Stirbt.**
- **`services/gemini/prompts.py`**-Prompts „Transform the source … do NOT just summarize or rephrase" + **`services/gemini/script.py::format_dialogue_with_llm`** (ruft `gemini-2.5-flash` zum *Paraphrasieren*). **Stirbt** im Treue-Pfad.
- **`services/gemini/dialogue.py::parse_dialogue`** (`line.split(':', 1)`) — fragiler „Name:"-Parser. **Ersetzt** durch strukturierte Daten.

Gemini-TTS **rezitiert wörtlich** (Default) und kann **nativ ≤2 Speaker** — `MultiSpeakerVoiceConfig` ist in `services/gemini/synthesis.py`/`tts.py` **schon verdrahtet**. Der Fix ist primär **Löschen**, kein Plattform-Wechsel.

> **Sequenz-Hinweis (wichtig)**: Der alte chatty-Flow **nutzt** `calculate_tag_guidance`/`format_dialogue_with_llm` bis zu seiner Stilllegung. Darum **baut NARR-1 den Treue-Pfad neu, ohne diese Funktionen aufzurufen** — **gelöscht** werden sie erst mit der Alt-Flow-Abschaltung in **NARR-5**. So bleibt jeder Sprint grün und der alte Flow läuft bis zur bewussten Abschaltung weiter.

## Gelockte Entscheidungen (Oli, 2026-06-28)

- **Treue-Grad**: „treu, fürs Hören geglättet" — die **Glättung passiert beim Authoring** (Claude), Gemini **rezitiert das Skript verbatim**. Zwei Schichten, kein Widerspruch.
- **Alter chatty-Modus**: **ersetzt** (raus). Ein klarer Modus.
- **Scope**: kohärenter Entwurf, **phasierte** Lieferung (5 Sprints), jeder grün.
- **Plattform v1**: **Gemini-reframed** (null Migration, verbatim-by-default, ≤2 Speaker schon verdrahtet, billigste Schiene). Die Turn-Listen-Architektur ist **plattform-neutral** → ElevenLabs ist ein späterer **Render-Schicht-Swap**, falls Stimmqualität (nicht Treue) der echte Frust wird.
- **Defaults**: Flash als Default (Pro opt-in pro Vertonung); Chunk-Nähte für v1 akzeptiert; Treue = Skill-Kontrakt + leichter Server-Sanity-Check, **kein** harter Verbatim-Gate (würde die gewollte Glue verbieten).

## Architektur (vier Schichten)

1. **Authoring (Claude-Skill, lebt in Claude, nicht CONVERTER)** — liest den Doc, schreibt die treue, geglättete **Turn-Liste**, mappt Speaker → Gemini-Voice, POSTet an CONVERTER. *Die Intelligenz lebt hier.*
2. **Token-Endpoint (CONVERTER)** — `POST /api/narrations`, Bearer `NARRATION_TOKEN` (Spiegel von `CARD_TOKEN`), CSRF-exempt, fail-closed, async via bestehenden RQ-Worker.
3. **Render (CONVERTER)** — empfängt die Turn-Liste, baut `MultiSpeakerVoiceConfig` (schon da), **Größen-Chunking** (nie Tag-Zahl), bestehende pydub-Concat. **Kein** server-seitiges Paraphrasieren, **keine** Tag-Arithmetik.
4. **Persistenz (CONVERTER)** — Audio wird erstklassiges Library-Element.

### Der Turn-Listen-Kontrakt (plattform-neutral)

```json
{
  "title": "Wie funktioniert FSRS?",
  "language": "de",
  "tts_model": "gemini-2.5-flash-preview-tts",
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
| **NARR-1** | M | **Treue-Synth-Kern**: **neuer** Treue-Render-Entry (Turn-Liste → Audio über die schon-verdrahtete Multi-Speaker-Config), der `calculate_tag_guidance`/`format_dialogue_with_llm` **nicht** aufruft; Größen-Chunking; pure-function-Tests. Alt-Funktionen + alter Flow **unberührt** (Löschung in NARR-5). Kein Endpoint, keine Persistenz. |
| **NARR-2** | M | **Library-Persistenz**: `audio_narration`-Conversion + `metadata_json`-Kontrakt + Turn-Liste→Markdown; persistenter Audio-Pfad + Serve-Endpoint (owner-404, Traversal-Guard, **nicht** löschen-beim-Serve) + Delete-Cleanup-Hook. |
| **NARR-3** | M | **Token-Endpoint + async**: `POST /api/narrations` (NARRATION_TOKEN, mirror `_authorize_card_write`, CSRF-exempt), Validierung, `pending`-Row sofort, `tasks.generate_narration_task` enqueue, Worker flippt ready/failed; `GET /api/narrations/<id>` Status. Auth-Matrix-Tests. |
| **NARR-4** | S/M | **Claude-Skill + Agent-Docs**: `erklaerbaer-narration`-Skill (Turn-Kontrakt, Voice-Katalog aus `_GEMINI_VOICES`, Treue-Regel: near-verbatim + Glue, keine Paraphrase, spärliche Tags, 1–2 Speaker, POST+Poll). converter-mcp-Brief/Agent-Guide. |
| **NARR-5** | S | **Library-UI-Player + alten Pfad stilllegen**: `<audio>`-Player für `audio_narration`, pending/failed/ready, Retry-aus-`metadata_json`; alter `/generate-gemini-podcast`-Paraphrase-Flow raus/abgeklemmt **+ die dann toten `calculate_tag_guidance`/`format_dialogue_with_llm` löschen**. Live-Smoke. |

Abhängigkeit: 1 → 2 → 3 → (4, 5). NARR-1 ist der Einstieg.

## Bewusst später / nicht v1

- **ElevenLabs** (Text-to-Dialogue) — Render-Schicht-Swap, falls Stimmqualität der bindende Frust wird. Strukturierte Turns nativ; Kosten ~$22/mo + Verbatim-Guardrail (kann ad-libben). Die Turn-Liste macht den Swap billig.
- **Azure Batch + SSML `<voice>`** — die **einzige** Option, die Chunk-Nähte ganz killt (ein Job pro Doc). Neuer Stack + SSML-Authoring → nur, falls die Concat-Nähte real stören.
- **Pro-TTS als Default**, **harter Verbatim-Gate** — bewusst nicht v1.

## Offene Code-Grounding-Notiz

Die Anker oben sind verifiziert (existieren). Jedes NARR-Sprint-Prompt macht beim Schreiben zusätzlich **file-level-Grounding** der konkret berührten Funktionen (Master-Disziplin) — der Workflow-Code-Reader hatte einen Schema-Fehlschlag, daher nicht blind auf relayte Zeilennummern verlassen.
