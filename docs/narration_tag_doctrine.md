# Tag- & Delivery-Doktrin: Sprecher-Gestaltung in Gemini 2.5 TTS

> **Provenienz**: Cowork-Recherche (Oli, 2026-06-28), beauftragt aus dem NARRATION-Reframe-Workshop. **Rolle**: die Referenz, die der **NARR-4 Claude-Narration-Skill** als Tag-/Delivery-Doktrin übernimmt (`erklaerbaer-narration`). Geschwister-Doc: [docs/narration_reframe.md](narration_reframe.md) (Architektur). Marker: `(G)` = Gemini-spezifisch · `(A)` = engine-allgemein (gilt auch bei Wechsel zu ElevenLabs/SSML).

**Zweck:** Direkt in einen Claude-Narration-Skill übernehmbare Doktrin für die *treue* Vertonung von Erklär-Dokumenten (near-verbatim + leichte konversationelle Glue) als 1- oder 2-Sprecher-Narration. Tags sind **bewusst + optional**, nie berechnet.

---

## 0. Kernprinzip (zuerst lesen)

1. **Stil steuert man bei Gemini primär über einen Natural-Language-Prompt ("Director's Notes"), NICHT über Inline-Tags.** Das Modell ist ein LLM, das *weiß, wie* etwas gesagt wird — es folgt Prosa-Anweisungen besser als Markup. `(G)`
2. **Default = NULL Tags.** Erst Stimme + Prompt + sauberer Text. Ein Tag wird nur gesetzt, wenn eine *konkrete, lokale* Wirkung gebraucht wird, die der Prompt nicht trägt. `(A)`
3. **Treue schlägt Performance.** Tags/Glue dürfen den Inhalt nie verändern, kürzen oder doppeln. Im Zweifel: weniger Steuerung, mehr Text. `(A)`
4. **Google wörtlich:** „Define only what's important to the performance, being careful to not overspecify. Too many strict rules will limit the model's creativity and may result in a worse performance." `(G)`

---

## 1. Was Gemini 2.5 TTS WIRKLICH kann (dokumentiert)

**Zwei Pfade — entscheidend, weil Tag-Verfügbarkeit unterschiedlich ist:**

| Pfad | Modell-IDs | Stil-Steuerung | Inline-Tags |
|---|---|---|---|
| **Gemini API / AI Studio** (`generativelanguage…`) | `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts` | Nur Natural-Language-Prompt (alles in einem `contents`-String) | **Nicht dokumentiert** — keine offiziellen Bracket-Tags |
| **Cloud TTS / Vertex AI** (`texttospeech`) | `Gemini 2.5 Pro/Flash TTS` (+ neuere `gemini-3.x-*-tts`) | Getrenntes `prompt`-Feld + `text`/`multiSpeakerMarkup` | **Markup-Tags (Preview)** — siehe §2 |

> **Konsequenz für die App:** Zuerst klären, welcher Pfad genutzt wird. Auf dem **Gemini-API-Pfad gibt es keine offiziellen Tags** → reine Prompt-Steuerung. Bracket-Tags sind dort bestenfalls inoffiziell/instabil. Der **Cloud-Pfad** trennt Direktive (`prompt`) strukturell vom Transkript (`text`) — das senkt das Prompt-Leakage-Risiko (§6) und ist für eine `[{speaker, text}]`-Turn-Liste der sauberere Pfad (entspricht `MultiSpeakerMarkup.Turn`).
>
> **CONVERTER HEUTE = Gemini-API-Pfad** (`client.models.generate_content(model='gemini-2.5-flash-preview-tts', contents=…)`). v1 bleibt darauf (Zero-Migration). Cloud-Pfad = dokumentierte spätere Option, s. [narration_reframe.md](narration_reframe.md) §Plattform-Pfad.

**Steuerhebel (beide Pfade):**

- **Director's-Notes-Prompt-Struktur** `(G)`: *Audio Profile* (Persona/Archetyp) · *Scene* (Umgebung/„Vibe") · *Director's Notes* (Style, **Pacing**, Accent) · *Sample context* · *Transcript*. Director's Notes sind der wichtigste Teil; die übrigen Blöcke sind optional.
- **Voice-as-style-lever** `(G)`: 30 Stimmen mit Charakter-Deskriptoren (z. B. *Kore – Firm*, *Charon – Informative*, *Rasalgethi/Sadaltager – Informative/Knowledgeable* für Erklär-Ton; *Enceladus – Breathy*, *Puck – Upbeat*). Stimme passend zur gewünschten Wirkung wählen — das verstärkt den Prompt.
- **Pacing kontextsensitiv** `(G)`: 2.5 folgt Pace-Anweisungen mit hoher Treue und verlangsamt/beschleunigt kontextabhängig.
- **Multi-Speaker:** **max. 2 Sprecher.** Sprecher-Namen im Prompt müssen den Voice-Configs entsprechen; pro Sprecher eigene Stil-Direktive möglich („Make Speaker1 sound tired, Speaker2 excited"). `(G)`
- **Sprachen:** Deutsch ist voll unterstützt (Cloud: **de-DE = GA**); Gemini API **erkennt die Sprache automatisch** aus dem Transkript. Alle Stimmen sind sprachübergreifend (dieselben 30 Voices sprechen Deutsch). `(G)`
- **Limits:** Gemini API: **32k-Token**-Kontext pro TTS-Session. Cloud: `prompt` ≤ 4000 Bytes, `multiSpeakerMarkup` ≤ 4000 Bytes, kombiniert ≤ 8000 Bytes. Output: 24 kHz PCM, Text-rein → Audio-rein. `(G)`

---

## 2. Inline-Tags: dokumentiert vs. Folklore

**Offiziell dokumentiert sind Bracket-Tags NUR im Cloud-TTS-Pfad, als „Markup Tags (Preview)".** Sie sind für *lokale* Effekte gedacht, **nicht** zum Setzen des Gesamttons (das macht der Style-Prompt). Google beschreibt vier Wirk-Modi:

| Modus | Beispiel-Tags | Wirkung | Für treue Erklär-Vertonung? |
|---|---|---|---|
| **1 – Non-Speech-Sounds** | `[sigh]`, `[laughing]`, `[uhm]` | Tag wird durch hörbaren Laut ersetzt, nicht gesprochen. Hohe Reliability. | **Sehr sparsam.** `[uhm]` kann im Dialog Natürlichkeit geben; sonst meiden. |
| **2 – Style-Modifier** | `[sarcasm]`, `[robotic]`, `[shouting]`, `[whispering]`, `[extremely fast]` | Nicht gesprochen; verändert *folgende* Sprechweise. | Selten nötig; nur für bewussten lokalen Effekt + passenden Prompt. |
| **3 – Vocalized Adjectives** | `[scared]`, `[curious]`, `[bored]` | ⚠️ **Tag-Wort wird MITGESPROCHEN** + färbt den Satz. | **NIE.** Google: „undesired side effect for most use cases — prefer the Style Prompt." Bricht Treue. |
| **4 – Pacing/Pausen** | `[short pause]` (~250 ms), `[medium pause]` (~500 ms), `[long pause]` (~1000 ms+) | Fügt Stille ein. | **Wichtigster nützlicher Tag-Typ.** Nur für *bewusste* Dramatik. „Avoid overuse, as it can sound unnatural." |

**Folklore / Engine-Verwechslung — NICHT auf Gemini übertragen:**

- **`[laughs]`, `[whispers]`, `[sighs]`, `[excited]`, `[sad]`, `[shouts]`, `[sarcastic]`, `[gunshot]` … = ElevenLabs v3 Audio-Tags** `(andere Engine)`. ElevenLabs hat ein reiches Tag-Vokabular (frei platzier- und kombinierbar). Gemini hat das **nicht** in gleicher Form — ElevenLabs-Taglisten 1:1 auf Gemini = Folklore (bestenfalls ignoriert, schlimmstenfalls mitgesprochen → Modus 3).
- **SSML (`<break>`, `<prosody>`, `<emphasis>` …):** gehört zum *klassischen* Cloud-TTS / engine-allgemeinen Standard, **nicht** zur prompt-getriebenen Gemini-TTS-Steuerung.
- **Tags sind in neueren Modellen (Gemini 3.x) erstklassiger** als in 2.5. Auf **2.5** Tags als *experimentell* behandeln: pro Modell/Sprache testen, bevor produktiv („a tag you assume is a style modifier might be vocalized").

---

## 3. Platzierung & Dichte

- **Stufenleiter der Steuerung** (immer von oben beginnen) `(A)`: 1) richtige **Stimme** → 2) **Style-Prompt/Director's Notes** → 3) **Interpunktion & Satzbau** für Rhythmus → 4) *erst dann* ein Tag, und nur lokal.
- **Dichte-Faustregel** `(A)`: In treuer Erklär-Narration **0 Tags als Normalfall.** Wenn überhaupt, dann **Pausen-Tags an wenigen, bewusst dramatischen Stellen** (z. B. vor einer Pointe/Zahl). Richtwert: höchstens ein bewusster Tag pro Absatz/Sinnabschnitt, nicht „pro Satz".
- **Kipp-Punkt** `(A)`: Sobald Tags dichter als die natürliche Betonung werden, klingt es over-acted/künstlich. Mehrere Pausen-Tags in Folge = Stottern.
- **Treue-Schranke** `(A)`: Tags stehen *zwischen* Wörtern, nie *statt* Inhalt. Sie dürfen kein Wort des Originals ersetzen, kürzen oder ergänzen.
- **Konsistenz-Regel (Google, „align all three levers")** `(G)`: Style-Prompt, Textinhalt und ggf. Tag müssen **dasselbe** wollen. Evokativer Text trägt Emotion zuverlässiger als ein Tag auf neutralem Text.

---

## 4. Einzelsprecher-Narration (Erklär-Delivery)

- **Prompt-Rezept (ruhig, klar, kompetent)** `(G)`: kurze Director's Note statt Tag-Streuung, z. B.: *„Lies das im ruhigen, klaren Ton einer Wissens-/Dokumentations-Erzählerin. Sachlich-warm, nicht werblich. Natürliches Tempo mit kleinen Atempausen an Sinngrenzen; betone Schlüsselbegriffe leicht, ohne zu dramatisieren."*
- **Tempo & Pausen** `(A)`: zuerst über **Interpunktion** (Komma ≈ kurze, Punkt ≈ mittlere Pause) und kurze Sätze. Pausen-Tag nur für bewusste Betonung vor Kernaussage/Zahl. Pacing global im Prompt setzen, nicht satzweise.
- **Betonung** `(G)`: über **Textkohärenz**, nicht über Markup (evokativer Text > neutraler Text).
- **Stimmwahl** `(G)`: „Informative/Firm/Clear/Knowledgeable"-Voices (z. B. Charon, Kore, Iapetus, Rasalgethi, Sadaltager) für Erklär-Inhalte; „Excitable/Upbeat" nur, wenn der Inhalt es trägt.

---

## 5. Zwei-Sprecher-Dialog (Treue erhalten)

- **Unterscheidbarkeit** `(G)`: zwei **kontrastierende** Voices (z. B. *Firm* vs. *Upbeat*, oder m/w-Charakter) und **fixe** Sprecher→Stimme-Zuordnung über alle Chunks. Optional pro Sprecher eine knappe, *unterschiedliche* Director's Note.
- **Turn-Struktur** `(A)`: `[{speaker, text}]` direkt nutzen (Cloud: `MultiSpeakerMarkup.Turn`). **Variable Turn-Längen** gegen Ping-Pong-Monotonie — nicht jede Replik gleich lang.
- **Back-Channels** `(A)`: „Mhm.", „Genau.", „Okay—" als **echten Text** in einen kurzen Turn (kein Tag dafür). **Sehr sparsam** — Back-Channels sind die häufigste Quelle für Treue-Drift, weil sie Inhalt *hinzufügen*. Bei strikter Treue: nur als minimale Glue, nicht als Inhalt.
- **Übergaben** `(A)`: natürliche Anschlüsse über Satzbau („Und genau deshalb…", „Heißt konkret…"), nicht über Tags. Frage→Antwort-Wechsel sparsam, sonst Quiz-Effekt.
- **Hesitation** `(G, optional)`: `[uhm]` (Cloud Modus 1) extrem dosiert — im Erklär-Kontext meist weglassen.

---

## 6. Fehlermodi & Gegenmaßnahmen

*(Quellen: Google AI Developers Forum — reale Produktionsberichte.)*

| Fehlermodus | Ursache | Gegenmaßnahme |
|---|---|---|
| **Roboterhaft/monoton** | Zu starre Regeln, neutraler Text, zu viele Tags | Tags raus; ein kohärenter Prosa-Prompt; passende Voice; Text evokativ lassen. `(A)` |
| **Over-acted** | Zu viele Stil-Direktiven/Tags, dramatische Voice | Prompt entschärfen („sachlich-ruhig"); Tags entfernen; ruhigere Voice. `(A)` |
| **Gedroppter / verdoppelter Text, Halluzinationen** | Langer Input; Modell „liest noch mal von vorn"; Filler an Absatzgrenzen | **Kürzere Chunks** an Sinn-/Absatzgrenzen; weit unter Limit bleiben; Output gegen Soll-Transkript diffen; Seam-Dedup. `(G)` |
| **Prompt wird vorgelesen (Leakage)** | Direktive im selben String wie Transkript | **Direktive strukturell vom Text trennen** (Cloud `prompt`-Feld). Auf Gemini-API: Direktive klar als Anweisung markieren („Sag Folgendes …:") + auf Deutsch halten, damit ein Leak nicht fremdsprachig auffällt. `(G)` |
| **Stimm-Drift / Voice-Swapping über Chunks** | Neu „geratene" Stimme pro Chunk; Multi-Speaker-Bug | **Feste Voice-Map als Konstante** über alle Chunks; pro Chunk dieselbe SpeechConfig; Chunks an Sinngrenzen; bei Instabilität **Pro statt Flash**. `(G)` |
| **Truncation mid-sentence / `finishReason OTHER`** | Multi-Speaker-Instabilität, langer Turn | Turns/Chunks verkürzen; Response-Länge prüfen, fehlende Audio-Enden erkennen + neu rendern. `(G)` |
| **Inkonsistenter Akzent/Pacing (v. a. Pro)** | Unter-/widersprüchliche Direktive | Akzent/Pacing **einmal global & spezifisch** setzen, nicht pro Satz wechseln. `(G)` |

---

## 7. Autoren-Checkliste pro Turn (für Claude)

Beim Schreiben **jeder** Turn `{speaker, text}` durchgehen:

1. **Treue zuerst:** Ist der Text near-verbatim am Original? Keine Paraphrase, keine neuen Fakten. Glue (Übergänge/Back-Channels) nur minimal und inhaltsneutral.
2. **Braucht diese Turn überhaupt Steuerung?** Default = **nein**. Stimme + globaler Style-Prompt reichen meist.
3. **Wenn Färbung nötig:** Geht sie über den **Style-Prompt/Director's Note** (global oder pro Sprecher)? → Dann **keinen** Inline-Tag setzen.
4. **Tag nur, wenn** ein *lokaler, bewusster* Effekt nötig ist, den der Prompt nicht trägt — und der Pfad Tags offiziell unterstützt (Cloud-Preview). Erlaubt: `[short/medium/long pause]` für Dramatik; in Maßen `[uhm]`. **Verboten:** Modus-3-Adjektiv-Tags (`[scared]` etc.) und ElevenLabs-Tags (`[laughs]`, `[whispers]` …) auf Gemini.
5. **Dichte-Check:** ≤ 1 bewusster Tag pro Sinnabschnitt; keine Tag-Ketten; keine Tag-Folge ohne Text dazwischen.
6. **Konsistenz-Check:** Wollen Text, Prompt und (etwaiger) Tag dasselbe? Passt die Voice zur Färbung?
7. **Dialog-Check (2 Sprecher):** Turn-Länge variiert ggü. Vorgänger? Fixe Voice-Zuordnung? Back-Channel (falls vorhanden) inhaltsneutral und selten?
8. **Chunk-Check:** Bleibt der kumulierte Text bequem unter dem Limit? Liegt die Chunk-Grenze an einer Sinn-/Absatzgrenze (nicht mitten im Satz)? Voice-Map identisch zum Nachbar-Chunk?
9. **Sprache:** de-DE — Direktiven auf Deutsch formulieren (reduziert Leakage-Artefakte); Modus-3-Tags strikt meiden (englisches Tag-Wort würde im deutschen Audio gesprochen).

---

## 8. Engine-Wechsel (Gemini → ElevenLabs/SSML)

| Prinzip | Gemini 2.5 | ElevenLabs v3 | SSML / klassisches TTS |
|---|---|---|---|
| Primäre Stil-Steuerung | **Natural-Language-Prompt** | **Inline-Audio-Tags** + Prompt | **SSML-Markup** (`<prosody>`, `<emphasis>`) |
| Pausen | Interpunktion, `[…pause]` (Cloud-Preview) | Interpunktion/Tags | `<break time="500ms"/>` |
| Reaktionen/Lacher | sehr begrenzt (`[laughing]`/`[uhm]` Cloud) | reich: `[laughs]`, `[sighs]`, `[whispers]` … | keine (nur Prosodie) |
| Treue-/Dichte-Disziplin | **gilt** | **gilt** | **gilt** |
| Voice-Map fix über Chunks | **gilt** | **gilt** | **gilt** |
| „Don't overspecify", Text-Prompt-Kohärenz | **gilt** | **gilt** (Tags sparsam) | teilweise |

**Portabel (engine-allgemein `(A)`):** Treue-Vorrang, „weniger ist mehr", Steuer-Stufenleiter (Voice→Prompt→Interpunktion→Tag), variable Turn-Längen, feste Voice-Zuordnung, Chunking an Sinngrenzen, Seam-Dedup.
**Gemini-spezifisch `(G)`:** Director's-Notes-Prompt-Struktur, Prompt-statt-Tag-Doktrin, Cloud-Markup-Modi 1–4 inkl. Modus-3-Verbot, 2-Sprecher-Limit, 32k/4000-Byte-Limits, Pfad-Unterscheidung API vs. Cloud.

---

## Quellen

- [Text-to-speech generation (TTS) — Gemini API](https://ai.google.dev/gemini-api/docs/speech-generation) — Prompt-Steuerung, Director's-Notes, Voices, Sprachen, Multi-Speaker, Limits (keine Inline-Tags dokumentiert).
- [Gemini-TTS — Cloud Text-to-Speech](https://docs.cloud.google.com/text-to-speech/docs/gemini-tts) — Markup-Tags (Preview), Modi 1–4, `prompt`/`text`-Trennung, Byte-Limits, de-DE = GA.
- [Gemini 2.5 TTS model updates — blog.google](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-2-5-text-to-speech/) — kontextsensitives Pacing, Expressivität.
- [Gemini TTS Multi-Speaker Mode: bugs in production — AI Dev Forum](https://discuss.ai.google.dev/t/gemini-tts-multi-speaker-mode-7-critical-bugs-after-3-weeks-in-production-finishreason-other-truncation-voice-swapping-hallucinated-lines/132776) — Truncation, `finishReason OTHER`, Voice-Swapping, Halluzinationen.
- [Gemini-2.5-flash-tts repeats text / reads prompt — AI Dev Forum](https://discuss.ai.google.dev/t/gemini-2-5-flash-tts-repeats-text-and-reads-out-a-section-of-the-prompt/113883) — Doppelung & Prompt-Leakage.
- [What are Eleven v3 Audio Tags — ElevenLabs](https://elevenlabs.io/blog/v3-audiotags) — Abgrenzung: ElevenLabs-Tags ≠ Gemini.

> **„Preview"-Caveat**: Tag-Verhalten/Limits der Gemini-TTS-Modelle können sich ändern — vor Produktiveinsatz pro Modell (Flash vs. Pro vs. 3.x) und pro Sprache (de-DE) kurz gegentesten.
