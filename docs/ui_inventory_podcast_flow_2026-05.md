# UI-Inventur: podcast-flow (2026-05-09)

**Methodik:** Stufe 1 der Duan-Kaskade (analog F-1.1 / F-2.1 / F-3.1). **Code-only** Inventur — kein Live-Walkthrough verfügbar in diesem Sub-Thread (Browser-Pfad würde echten Gemini-TTS-Credit verbrennen + ist async + chunked, also schwer mockbar; Master macht Walkthrough-Nachreichung in F4-REVIEW falls notwendig). Async-spezifische States (`queued`, `started`, `stage-progress`, `finished`, `failed`, `cancelled`) sind aus Code abgeleitet, nicht visuell verifiziert. Stale-Job-, Network-Drop-, Browser-Reload-Pfade sind Hypothesen aus Code-Lesen, mit konkreten Test-Anleitungen für F4-REVIEW.

**Feature-Kontext (aus CLAUDE.md):** Single-User-App, LAN-only, login-protected. `podcast-flow` ist der Long-Running-Async-Flow innerhalb `audio_converter`: User pastet Quelltext (oder schreibt direkt ein Skript) → wählt Format/Sprache/Stil/TTS-Modell → optional „Skript aus Quelltext generieren" (synchron, ein API-Call ~5–20s) → „Podcast generieren" (async, RQ-Worker, Minuten-Bereich, chunked bei langen Skripten).

**Quellen:** [templates/audio_converter.html](templates/audio_converter.html) (Zeilen 119–251, `#podcast-pane`), [static/js/audio_converter.js](static/js/audio_converter.js) (Zeilen 619–885, Podcast-Block; plus `setPodcastGenerating` / `cancel-podcast-btn`-Handler), [app_pkg/podcasts.py](app_pkg/podcasts.py) (alle 6 Routes), [tasks.py](tasks.py) (`generate_podcast_task`), [worker.py](worker.py) (RQ-Worker-Bootstrap), [services/gemini/__init__.py](services/gemini/__init__.py), [services/gemini/tts.py](services/gemini/tts.py) (Chunking + Retry-Loop), [services/gemini/synthesis.py](services/gemini/synthesis.py) (Single-Chunk-API-Call), [static/js/_utils.js](static/js/_utils.js) (Helpers), [static/css/style.css](static/css/style.css) (für State-Klassen `.mode-radio-input`/`.mode-radio-label`, `.prompt-editor-toggle`, `.prompt-editor-content[.expanded]`, `#mic-button.mic-loading`, `.c-btn`, `.c-btn--primary`, `.c-btn--danger`, `.c-input`, `.c-alert`, `.toast-notification`, `.hidden`).

**Bereits durch F-2 / F-3-IMPL gemappt — NICHT erneut behandelt:**
- Layout (Sidebar, Theme-Toggle, Brand, Mobile-Sidebar) — F-2.1 Zeilen #1–#9.
- Header (Audio-Werkzeuge-Heading) — F-2.1 Zeile #10.
- Tabs-Switcher (Live / Datei / Text-zu-Podcast — der Tab-Button ist hier nur das Eintritts-Tor, der **Inhalt** der Pane ist Inventur-Gegenstand) — F-2.1 Zeilen #12–#14, hier nur ergänzend für aria-disabled/Service-Gating-State.
- Podcast-Alert-Mountpoint (`#podcast-alert-container`) — F-2-Pattern P15 erfüllt (Zeile 121 im Template).
- Sämtliche DE-Microcopy für Podcast-Tab — F-2 DE-Pass abgeschlossen, hier nur Bestätigung.
- F-001-Critical-Fix (NoSuchJobError vs. transport-error distinguish) — bestätigt im Code, [app_pkg/podcasts.py:172-176](app_pkg/podcasts.py#L172-L176) und [:198-203](app_pkg/podcasts.py#L198-L203).
- F-005 Path-Traversal in `podcast_download` — durch SEC-Sprint mit `Path.is_relative_to` ersetzt, bestätigt [app_pkg/podcasts.py:219-222](app_pkg/podcasts.py#L219-L222).
- F-013 Input-Allowlist für Gemini-Parameter (`narration_style`, `script_length`, `language`, `num_speakers`) — bestätigt [app_pkg/podcasts.py:265-286](app_pkg/podcasts.py#L265-L286). **Aber:** der primäre Podcast-Workflow-Pfad `generate_gemini_podcast` (Zeile 138–164) hat **keine** Allowlist auf `language`/`tts_model` — siehe Befund 12.

---

## Async-Pipeline-Mapping (NEU für podcast-flow)

```
SYNCHRONOUS PRE-FLIGHT (kein RQ involviert):
  [User füllt Step-1-Setup + Step-2 Quelltext]
      │
      ▼
  [Skript aus Quelltext generieren]  →  POST /format-dialogue-with-llm
      │                                       (synchron, blockierend, ~5–20s)
      │                                       Loading-State: "Generiert …", Button disabled
      │                                       services/gemini/script.py (LLM-Call)
      ▼
  [#podcast-script Textarea gefüllt]
      │
      ▼
  [Manuelle Edit-Möglichkeit]

ASYNCHRONOUS PODCAST-GENERATION (RQ-Pipeline):
  [Generate Podcast Click]
      │
      │── JS parst Skript zeilenweise → dialogue[]
      │── voiceMap {Kate→Zephyr, Max→Charon, default→Kore}
      │
      ▼
  POST /generate-gemini-podcast  (Frontend → Backend)
      │
      │── Backend: task_queue.enqueue(generate_podcast_task,
      │              args=(dialogue, language, tts_model),
      │              job_timeout=600,  ← TIMEOUT_RQ_JOB_SECONDS in app_pkg/config.py
      │              meta={'user_id': current_user.id})
      │── Response: {"job_id": "...", "status": "queued"}
      │
      ▼
  setPodcastGenerating(true)
    → generate-Btn .hidden, cancel-Btn sichtbar
    → cancel-Btn-Text "Abbrechen" (initial)
    → Result-Container .hidden
      │
      ▼
  POLLING-LOOP (Frontend, 2s tick):
      │
      │  while (status === 'pending' || 'processing') {
      │    sleep 2s
      │    if (cancelRequested) break
      │    pollCount++
      │    cancel-Btn-Text → "Abbrechen (Ns)"  (Live-Counter)
      │    GET /podcast-status/<job_id>
      │    → Backend: Job.fetch + user_id-Match-Check
      │    → JSON-Response basierend auf RQ-status:
      │       - 'queued'   → {"status": "processing"}     ← ↯ konflatiert
      │       - 'started'  → {"status": "processing"}     ← ↯ konflatiert
      │       - 'finished' → {"status": "completed", "result": "<file_path>"}
      │       - 'failed'   → {"status": "failed", "error": "<traceback>"}
      │       - sonst      → {"status": "<raw>"}          ← ↯ unbehandelt
      │    if (status === 'failed') throw → catch → showAlert
      │  }
      │
      ▼
  WORKER (parallel, isolated process, kein job.meta-Update):
      │
      │  generate_podcast_task(dialogue, language, tts_model)
      │    1. GeminiService(api_key)  ← fresh init in worker
      │    2. tts.generate_podcast()
      │       a. filter_metadata_lines  (dialogue.py)
      │       b. split_long_dialogue_turns (max_words=50)
      │       c. CHUNKING-DECISION:
      │          - if len(lines) ≤ 80 → single chunk
      │          - else → multi-chunk via _generate_with_chunking
      │       d. PER-CHUNK:
      │          - generate_single_chunk → Gemini TTS API call (~5–60s/chunk)
      │          - max_retries=2, progressive backoff (5s, 10s)
      │          - between-chunk-delay 1.0s (rate-limit)
      │          - on retry-exhaustion: cleanup partial chunks + raise
      │       e. concatenate_with_pydub OR concatenate_with_wave
      │    3. shutil.move(temp_path, OUTPUT_DIR/filename) ← shared volume
      │    4. return final_path
      │
      │  Worker-Logging: nur via logger.info — keine job.meta-Updates,
      │  also kein User-sichtbarer Stage-Progress.
      │
      ▼
  status === 'completed' → break out of poll loop
      │
      ▼
  GET /podcast-download/<job_id>
      │── Backend: Job.fetch + user_id-match + is_finished + path-traversal-check
      │── Liest File in BytesIO + os.unlink(real_path)  ← File wird hier gelöscht!
      │── send_file → audio/wav, attachment "gemini_podcast.wav"
      │
      ▼
  Frontend: blob → URL.createObjectURL → audio.src + download.href
      │
      ▼
  Result-Container sichtbar + scrollIntoView
      │
      ▼
  setPodcastGenerating(false) → generate-Btn zurück, cancel-Btn versteckt

CANCEL-PFAD (Frontend-only, Backend läuft weiter):
  [Cancel Click] → podcastCancelRequested = true
                 → break aus Poll-Loop in nächstem Tick
                 → showAlert("Generierung abgebrochen. Backend-Job läuft im Hintergrund weiter.")
                 → setPodcastGenerating(false)
  Worker-Job läuft bis zu job_timeout=600s weiter und schreibt File in OUTPUT_DIR.
  Datei wird nie heruntergeladen → bleibt liegen bis Container-Restart.

DOWNLOAD-NACH-GENERATION (Re-Download):
  Nicht möglich. Die Backend-Route löscht das File nach erstem Download
  (podcasts.py:231 os.unlink). Wenn der Browser zwischendurch neu lädt
  oder der Download fehlschlägt, ist die Datei weg, aber der Audio-Player
  und der Download-Button im UI funktionieren weiterhin (Blob-URL ist
  client-seitig gecached) — bis zum nächsten Tab-Reload.
```

**Zentrale Async-Pipeline-Beobachtungen** (Detail in „Befunde" weiter unten):
1. **Kein Stage-Progress-Mechanismus**: weder die Backend-Route noch der Worker pflegen `job.meta`. Bei Multi-Chunk-Generierung weiß die UI nichts über den Fortschritt (Skript-Filter → Chunking → Chunk-für-Chunk-Generation → Concat). Counter zählt nur Wand-Sekunden.
2. **`queued` und `started` werden konflatiert** zu `"processing"`: Der User sieht keinen Unterschied zwischen „Job in Warteschlange, Worker noch nicht da" und „Worker arbeitet aktiv".
3. **Cancel ist soft**: Der Cancel-Btn beendet nur das Frontend-Polling. Der Worker läuft weiter, generiert das File, schreibt es in `OUTPUT_DIR` — und niemand holt es ab. Die Frontend-Message ist ehrlich („Backend-Job läuft im Hintergrund weiter."), aber die OUTPUT_DIR-Cleanup-Strategie ist „beim nächsten Download löscht der Endpoint das File" — was bei abgebrochenem Job nie passiert.
4. **Browser-Reload während Polling = job_id verloren**: Job läuft im Worker bis zum Ende, generiert File, das nie abgeholt wird. Keine Persistenz von `job_id` (LocalStorage o.ä.).
5. **Network-Drop während Polling**: `safeJSON(statusResponse)` würde bei Network-Fail werfen, fällt in den outer-Catch → showAlert generic, Worker läuft trotzdem weiter.
6. **job_timeout=600s** ist der harte Cutoff: bei sehr langen Podcasts (>>10min Multi-Chunk) wird der Job Worker-seitig abgewürgt → status `failed`, mit RQ-spezifischem Timeout-Error.
7. **`/podcast-status/<job_id>` hat keinen Stale-Check** auf Frontend-Seite: theoretisch könnte der Counter ewig laufen (kein Frontend-Timeout im Poll-Loop). Praktisch begrenzt durch job_timeout=600s, dann wird der Server-side-status `failed`.

---

## Element-Tabelle (podcast-spezifische Sub-Bereiche + Async-States)

Legende States: ✓ vorhanden im Code · ✗ fehlt · ? unklar/teils · n/a nicht zutreffend
Async-Spalten: queued/started/stage/finished/failed/cancelled — die im Code als sichtbarer UI-State belegt sind.
Live-Spalte: ✓ live verifiziert · ✗ live nicht beobachtet · ↯ Differenz Code↔live · n/a nicht prüfbar (rein statisch). **Code-only-Inventur**, daher überall n/a außer bei direkten Static-Diff-Beobachtungen.

Regions: **Pane** = `#podcast-pane`-Header + Service-Gate · **Setup** = Schritt-1-Card (Format / Sprache / Stil / TTS-Modell) · **Script** = Schritt-2-Card (Quelltext / Skript-Editor / Custom-Prompt) · **Action** = Generate / Cancel · **Result** = Audio-Player / Download · **Async-State** = pseudo-Element für Polling-Status-Klassen, nicht im DOM · **Feedback** = Banner / Toasts

| #  | Region   | Element-Typ              | Label/Text                                         | Aktion                                                                                                                                          | default | hover | focus / focus-visible | disabled | loading | queued | started | stage | finished | failed | cancelled | error | success | empty | Live verifiziert |
|----|----------|--------------------------|----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|---------|-------|------------------------|----------|---------|--------|---------|-------|----------|--------|-----------|-------|---------|-------|-------------------|
| 1  | Pane     | Tab-Button (Eintritt)    | "Text zu Podcast" (`data-tab="podcast"`)           | Toggelt Tab-Pane; Lang-Selector wird versteckt; Service-Gate via `aria-disabled="true"` + `title="Service nicht konfiguriert"` wenn `!gemini_api_key_set` | ✓ + `.tab-inactive` | ✓ | ? (kein eigener `:focus-visible`-Ring im CSS, Browser-default-Outline) | ✗ aria-disabled=true wird gesetzt, aber **JS `tabButtons.forEach` ignoriert es** — Click feuert trotzdem (siehe Befund 1) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-String OK** |
| 2  | Pane     | Container                | `#podcast-alert-container` (P15-Mountpoint)        | Empfänger für `showAlert(podcastAlertContainer, level, msg)` aus dem JS                                                                          | ✓ leer initial | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ✓ via showAlert | n/a | ✓ leer | n/a (Code-only) |
| 3  | Pane     | Conditional Banner       | "Podcast-Generierung nicht verfügbar — Gemini-API-Key fehlt." (`.c-alert.c-alert--warning`) | Server-rendered nur wenn `!gemini_api_key_set`. Bleibt im Pane-Header, **rest des Panes wird trotzdem voll gerendert** (anders als Live/File-Tab Configuration-Error in F-2.1 #11, der den ganzen Tab-Block überspringt). Gemini-Buttons sind dann disabled. | ✓ Markup vorhanden | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-String OK** |
| 4  | Setup    | Card-Container           | "Schritt 1: Setup" (`.c-surface--flat`, h5 + Felder) | rein dekorativ                                                                                                                                  | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-Heading OK** |
| 5  | Setup    | Radio-Card (visually-hidden Input) | "Monolog / Einzelne Stimme (Kate)" (`#mode-monolog`, `value="monolog"`, default-checked) | Form-Wert für `mode` — bestimmt `numSpeakers` (1) und `speakerDescriptions` ({Kate Zephyr warm and professional}) im `/format-dialogue-with-llm`-Body sowie das Skript-Parsing in `generate-podcast-btn`-Handler | ✓ + `.mode-radio-input:checked + .mode-radio-label` (Inset-Shadow + Accent-tint bg) | ✓ (`.mode-radio-label:hover` raised shadow) | ✓ (`.mode-radio-input:focus-visible + .mode-radio-label` Ring; `:checked:focus-visible` kombiniert beides — F-2-Pattern P15 erfüllt) | n/a (Service-Gate hat keinen Disable-Pfad auf Radios) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) |
| 6  | Setup    | Radio-Card (visually-hidden Input) | "Dialog / Gespräch (Kate & Max)" (`#mode-dialogue`, `value="dialogue"`) | s.o.; numSpeakers=2, beide Kate+Max                                                                                                              | ✓ | ✓ | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) |
| 7  | Setup    | Select                   | "Sprache" (`#podcast-language`, default `de` selected) | Form-Wert für `language` in beiden Endpoints. Optionen: `en`/`de`/`es`/`fr`. **Kein Service-Allowlist-Check im `/generate-gemini-podcast`** (siehe Befund 12). | ✓ | ✓ | ✓ (`.c-input:focus`) | ? — kein `disabled`-Pfad, auch nicht beim Service-Gate | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-Optionen OK** |
| 8  | Setup    | Select                   | "Sprech-Stil" (`#narration-style`, default `conversational`) | Form-Wert für `narration_style` in `/format-dialogue-with-llm`. Optionen: `documentary` / `conversational` / `academic` / `satirical` / `dramatic` mit DE-Labels. | ✓ | ✓ | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-Labels OK** |
| 9  | Setup    | Select                   | "TTS-Modell" (`#tts-model`, default `gemini-2.5-flash-preview-tts`) | Form-Wert für `tts_model` in `/generate-gemini-podcast`. Optionen: Flash (default) + Pro. **Kein Service-Allowlist-Check im Backend** — Worker validiert in `services/gemini/tts.py:54-55` und fällt sonst auf default zurück (silent). | ✓ | ✓ | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-Labels OK** |
| 10 | Script   | Card-Container           | "Schritt 2: Podcast erstellen" (`.c-surface--flat`, h5) | rein dekorativ                                                                                                                                  | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) |
| 11 | Script   | Textarea                 | "Quelltext" (`#podcast-raw-text`, 6 rows, DE-Placeholder + Hint) | Input für `/format-dialogue-with-llm`-`raw_text`. Leer-erlaubt: in dem Fall arbeitet User direkt im Skript-Editor (#13). **Kein Max-Length-Hint** im UI; Backend hat ebenfalls keinen Max-Check (siehe Befund 13). | ✓ | n/a | ✓ (`.c-input:focus`) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ✓ Placeholder | n/a (Code-only) — **DE-Microcopy OK** |
| 12 | Script   | Button                   | "Skript aus Quelltext generieren" (`#generate-script-btn`, `.c-btn` mit Spinner-SVG-Icon-Span) | `POST /format-dialogue-with-llm` mit raw_text + speakerDescriptions + language + narration_style + custom_prompt. Bei Erfolg: `#podcast-script.value = data.raw_formatted_text`. Loading-State: `disabled=true` + `.generate-script-btn__label`-textContent → "Generiert …" (kein Spinner-Animation, das Icon ist statisch). | ✓ | ✓ | ✓ (`.c-btn:focus-visible`) | ✓ (`.c-btn:disabled` opacity 0.4 cursor not-allowed; auch via Service-Gate `{% if not gemini_api_key_set %}disabled{% endif %}`) | ✓ Text-Swap "Generiert …" — **kein Spinner, das Icon-SVG dreht nicht** (siehe Befund 14) | n/a (synchron, kein RQ) | n/a | n/a | n/a | n/a | n/a | ✓ via showAlert(level=danger): "Skript-Generierung fehlgeschlagen. Bitte erneut versuchen." | ✓ implizit (Skript erscheint im Editor) | ✓ via showAlert(level=warning) "Bitte erst Quelltext im Feld oben eintragen." wenn rawText leer | n/a (Code-only) — **DE-Microcopy OK** |
| 13 | Script   | Textarea                 | "Podcast-Skript" (`#podcast-script`, 12 rows, mono font, DE-Placeholder mit Format-Beispiel + DE-Format-Hint) | Input für Skript-Parsing → `dialogue[]`. **User-editierbar jederzeit** (auch während Podcast-Generierung läuft — siehe Befund 15). Format: `Sprecher [stil]: Text` pro Zeile. | ✓ | n/a | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ✓ Placeholder | n/a (Code-only) — **DE-Microcopy OK** |
| 14 | Script   | Disclosure-Toggle (`<button>`) | "Erweitert: Eigener Prompt" (`#prompt-editor-toggle`, `.prompt-editor-toggle`, mit `aria-expanded` + `aria-controls="prompt-editor-content"` + `▼/▴`-Glyph) | `classList.toggle('expanded')` auf `#prompt-editor-content`; Glyph-Swap; `aria-expanded` wird gepflegt. F-2-Pattern P16 (echtes `<button>` statt `<div>`) erfüllt. | ✓ collapsed | ✓ (`hover:text-neo-text` text-color-shift, kein bg/shadow) | ✓ (`.prompt-editor-toggle:focus-visible` mit inset-shadow Ring) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-Label OK + a11y vollständig** |
| 15 | Script   | Glyph-Span               | `#prompt-toggle-icon` (`▼` collapsed → `▴` expanded, HTML-Entities `&#9660;` / `&#9650;`) | rein dekorativ; `aria-hidden="true"`; via `innerHTML` getauscht                                                                                  | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) |
| 16 | Script   | Container                | `#prompt-editor-content` (`.prompt-editor-content` collapsed via `display:none`, `.expanded` → `display:block`) | Wrapper für Custom-Prompt-Editor + Reset-Btn                                                                                                    | ✓ collapsed | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) |
| 17 | Script   | Textarea                 | "Eigener Prompt" (`#custom-prompt-editor`, `min-h-[200px]`, mono font, DE-Placeholder mit Platzhalter-Liste) | User-bearbeitbarer Override. Leer = Server-Default-Prompt. **Kein Validation für Platzhalter** (z.B. wenn `{raw_text}` fehlt → Server bekommt unsubstituierte Format-Anweisung). | ✓ | n/a | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ✓ Placeholder | n/a (Code-only) — **DE-Placeholder OK** |
| 18 | Script   | Button                   | "Auf Standard zurücksetzen" (`#reset-prompt-btn`, `.c-btn`, text-xs) | `customPromptEditor.value = ''` mit `confirmIfLong`-Guard (Threshold 200 Zeichen, DE-Confirm-Message). Helper-Reuse aus `_utils.js` ✓.            | ✓ | ✓ | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-Confirm OK** |
| 19 | Action   | Button (Primary)         | "Podcast generieren" (`#generate-podcast-btn`, `.c-btn--primary`, w-full) | (1) Skript-Parse → dialogue[] (Befund 16: Parse-Logik fragil); (2) `setPodcastGenerating(true)` → Btn .hidden, Cancel-Btn sichtbar; (3) `POST /generate-gemini-podcast`; (4) Polling-Loop; (5) Download → Blob → audio + download-href. **Kein Text-Swap auf dem Button** während Generierung — der Button verschwindet komplett (`.hidden`) und der Cancel-Btn übernimmt die Position. | ✓ | ✓ | ✓ (`.c-btn:focus-visible`) | ✓ (Service-Gate `{% if not gemini_api_key_set %}disabled{% endif %}`) | ✓ implizit: Btn `.hidden` während Run, Cancel-Btn übernimmt | ✗ **kein eigener queued-Indicator** — die Backend-Response `{"status":"queued"}` aus `/generate-gemini-podcast` wird im JS nicht ausgewertet, der Poll-Loop startet sofort und fragt `/podcast-status/<id>` ab, was `processing` zurückgibt sobald RQ einen Worker pickt | ✗ **kein started-Indicator** — `started` wird zu `processing` konflatiert (Befund 5) | ✗ **kein Stage-Progress** — Worker pflegt `job.meta` nicht; UI weiß nichts über Skript-Filter / Chunking-Decision / Per-Chunk-Progress (Befund 4) | ✓ implizit: `status === 'completed'` → break → Download → Audio-Player | ✓ catch + showAlert(danger): "Podcast-Generierung fehlgeschlagen. Bitte erneut versuchen." (kein Status-Code-spezifischer Hinweis) | ✓ via Cancel-Btn-Flag → showAlert(warning): "Generierung abgebrochen. Backend-Job läuft im Hintergrund weiter." (Befund 9) | ✓ guards: leeres Skript → showAlert(warning); Parse-Failure → showAlert(danger) | ✓ Result-Container `.hidden`-Toggle + scrollIntoView | n/a | n/a (Code-only) — **DE-Microcopy OK** |
| 20 | Action   | Button (Danger)          | "Abbrechen" / "Abbrechen (Ns)" (`#cancel-podcast-btn`, `.c-btn--danger`, initial `.hidden`) | Setzt `podcastCancelRequested = true`. Polling-Loop bricht im nächsten 2s-Tick ab. Counter-Text wird im Loop pro Tick aktualisiert (`Abbrechen (${pollCount * 2} s)`). Im finally-Block wird Text auf "Abbrechen" zurückgesetzt. **Backend-Job wird NICHT gecancelt** — RQ-`job.cancel()` o.ä. wird nicht aufgerufen. | ✓ initial `.hidden` | ✓ | ✓ | n/a | ✓ Live-Counter im Text alle 2s (Tick-genau, nicht Realtime) | ✗ Counter zeigt nur Wand-Sekunden, ohne Unterscheidung queued/started/stage | s.o. | s.o. | n/a | n/a | ✓ ist genau dieser Pfad | n/a | n/a | n/a | n/a (Code-only) — **DE-Label OK; Cancel-Mechanik soft, siehe Befund 9** |
| 21 | Async-State | Polling-Loop (kein DOM, Pseudo-Element) | `while(status === 'pending' \|\| 'processing')` mit `setTimeout(2000)` | Frontend-Heartbeat: alle 2s `GET /podcast-status/<job_id>` + `safeJSON`-Parse + `status`-Update. **Kein Frontend-Timeout** — der Loop läuft, bis status `completed`/`failed` oder `cancelRequested`. Praktisch begrenzt nur durch Backend-Timeout job_timeout=600s. | n/a | n/a | n/a | n/a | n/a | ✓ via `processing` (von RQ `queued` konflatiert) | ✓ via `processing` (von RQ `started` konflatiert) | ✗ kein Stage-Progress aus Worker | ✓ via `completed` | ✓ via `failed` (mit `error`-Feld, das in `throw new Error` landet, aber generischen Toast triggert) | ✓ via cancelRequested-Flag | ✓ HTTP-Fehler im Poll selbst (z.B. 404 NoSuchJobError nach Job-Cleanup von RQ) → safeJSON-Parse normalerweise fine, aber wenn `r.ok===false` und JSON-Body keinen `status` hat, läuft die Loop endlos auf `undefined` (Befund 6) | n/a | n/a | n/a (Code-only) |
| 22 | Async-State | Backend-Endpoint (Pseudo-Element) | `GET /podcast-status/<job_id>` JSON-Response-Schema | `{status: "queued"\|"started"}` → `{"status": "processing"}` (200); `finished` → `{"status": "completed", "result": <path>}` (200); `failed` → `{"status": "failed", "error": <traceback>}` (200, kein 5xx); `NoSuchJobError` → `{"error": "Job not found"}` (404); user_id-Mismatch → `{"error": "Job not found"}` (404, Camouflage); transport-error → `{"error": "Job lookup failed"}` (500); andere RQ-Statuses (`deferred`, `scheduled`, `stopped`, `canceled`) → `{"status": <raw>}` (200) — vom Frontend ungehandelt | n/a | n/a | n/a | n/a | n/a | s.o. | s.o. | s.o. | s.o. | s.o. | n/a (RQ `canceled`-State würde `{status:"canceled"}` liefern, aber JS hat keinen `else if`-Branch dafür) | n/a | n/a | n/a | n/a (Code-only) |
| 23 | Async-State | Worker-Process (Pseudo-Element) | `tasks.generate_podcast_task` | `GeminiService.generate_podcast` → Filter → Chunking-Decision → Per-Chunk-Generation (mit Retry-Loop max_retries=2 + progressive backoff 5s/10s) → Concat. **Kein job.meta-Update**, nur `logger.info`-Output. Bei Worker-Exception: re-raise, RQ markiert job als `failed`, speichert `exc_info` als Traceback-String. Bei Worker-Crash (z.B. OOM): RQ erkennt das via SIGKILL → job auch als `failed` markiert (RQ-Default-Behavior). | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ✗ kein Stage-Progress publiziert | n/a | s.o. | n/a | n/a | n/a | n/a | n/a (Code-only) |
| 24 | Result   | Container                | `#podcast-result-container` (`.hidden` initial) | Enthält Heading + Audio + Audio-Error-Mountpoint + Download-Btn. Wird `.hidden`-getoggelt nach erfolgreichem Download.                            | ✓ initial `.hidden` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ✓ remove `.hidden` | n/a | n/a | n/a | ✓ + scrollIntoView | ✓ initial `.hidden` (kein Empty-Placeholder à la "Bisher kein Podcast generiert") | n/a (Code-only) |
| 25 | Result   | Heading                  | "Dein Podcast:"                                    | rein dekorativ                                                                                                                                  | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) — **DE-Microcopy OK** |
| 26 | Result   | Audio-Element            | `#podcast-audio` (`controls`, `<source id="podcast-audio-source" type="audio/wav">`) | Browser-native `<audio controls>` Player UI. `loadeddata`-Event räumt `#podcast-audio-error` auf; `error`-Event zeigt persistent-Banner via showAlert(info, closable=false, autoDismissMs=null) "Audio nicht mehr verfügbar — bitte erneut generieren." | ✓ Browser-default | n/a | ✓ Browser-default | n/a | ✓ Browser-default Buffering | n/a | n/a | n/a | n/a | ✓ Browser-default Decoding-Error → eigener `error`-Event-Handler | n/a | n/a | n/a | n/a (kein Empty — vor erstem Erfolg ist `#podcast-result-container` `.hidden`, also Audio ist nicht eingebunden) | n/a (Code-only) — **DE-Microcopy für error OK** |
| 27 | Result   | Container (Error-Mountpoint) | `#podcast-audio-error` (initial `.hidden`) | Empfänger für showAlert(info) bei Audio-load-Error (z.B. wenn Blob-URL revoked wurde, was hier nicht passiert, oder wenn Audio-Codec nicht supported). | ✓ initial `.hidden` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | s.o. (#26 error-Pfad) | n/a | n/a | n/a | ✓ initial `.hidden` | n/a (Code-only) |
| 28 | Result   | Anchor (Download)        | "Herunterladen" (`#download-podcast-btn`, `.c-btn`, `download="podcast.wav"`, mit Download-Icon-SVG) | `href` wird beim Erfolg auf Blob-URL gesetzt (`URL.createObjectURL(audioBlob)`). Click-Handler triggert Toast "✓ Podcast heruntergeladen" via `showToast()` (default-level success, 2.5s auto-dismiss). | ✓ | ✓ | ✓ (`.c-btn:focus-visible`) | n/a (kein expliziter `disabled`-Pfad; nur sichtbar nachdem Result-Container nicht mehr `.hidden` ist) | ✗ kein Loading (instant — Browser-native Anchor-Download) | n/a | n/a | n/a | ✓ Toast | ✗ kein Error-Pfad (kein try/catch um den Anchor-Click — wenn Browser den Download blockt, kein Feedback) | n/a | n/a | ✓ via Toast | n/a (kein Empty — Btn ist nur sichtbar nachdem Result-Container `.removed-hidden`) | n/a (Code-only) — **DE-Label + Toast OK** |
| 29 | Feedback | Toast (`.toast-notification`) | dynamisch via `showToast` aus `_utils.js` | Singleton-Toast: alte werden entfernt, neuer mit `.show`-Klasse, auto-dismiss nach 2.5s. **Nur einmal in podcast-flow verwendet**: Download-Click (success). | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | ✓ Download-Confirm | n/a | n/a | n/a | n/a | n/a | n/a (Code-only) |
| 30 | Feedback | Banner (`.c-alert`)      | dynamisch via `showAlert(podcastAlertContainer, level, msg)` aus `_utils.js` | level=`warning` für Validation-Hints (leerer Quelltext, leeres Skript, Cancel); level=`danger` für Failures (Skript-Generation, Podcast-Generation, Parse-Failure); level=`info` für Audio-Reload-Hint. **Persistente Banner für `danger` (kein autoDismiss)**, autoDismiss 6s für andere Levels (Helper-Default). | ✓ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | s.o. | n/a | s.o. | s.o. | n/a | n/a (Code-only) |

---

## Zusammenfassung

- **Gesamtzahl interaktive Elemente (podcast-spezifisch):** 17 (#1, #5, #6, #7, #8, #9, #11, #12, #13, #14, #17, #18, #19, #20, #26, #28 — plus #21/#22/#23 als Pseudo-Elemente für die Async-State-Klasse). Mit Containern + Feedback + Heading: 30 Tabellen-Zeilen.

- **Async-State-Coverage** (queued / started / stage-progress / finished / failed / cancelled):
  - **queued**: ✗ nicht als eigener UI-State erkennbar (konflatiert mit started zu `processing`).
  - **started**: ✗ nicht als eigener UI-State erkennbar (konflatiert mit queued).
  - **stage-progress**: ✗ nicht implementiert — Worker pflegt kein `job.meta`, UI hat nur Wand-Sekunden-Counter.
  - **finished**: ✓ implizit via `status === 'completed'` → Download → Audio-Player.
  - **failed**: ✓ via showAlert(danger) — aber generischer Text, kein server-error-Pass-through.
  - **cancelled**: ✓ Frontend-Soft-Cancel via showAlert(warning); Backend-Job läuft weiter (siehe Befund 9).
  - **Polling-Heartbeat-Visual**: ✓ Counter im Cancel-Btn-Text (`Abbrechen (Ns)`), aber 2s-Tick-Granularität, kein Spinner.

- **Im Code identifizierte fehlende States (✗) — podcast-spezifisch:**
  - **#1 Tab-Button "Text zu Podcast"** hat `aria-disabled="true"` aber JS ignoriert das beim Click — siehe Befund 1.
  - **#1 Tab-Button** hat keinen eigenen `:focus-visible`-Ring (nur Browser-default-Outline).
  - **#3 Service-Gate-Banner** für `gemini_api_key_set=False` lässt das Pane sichtbar (anders als der Live/File-Tab Configuration-Error in F-2.1, der den ganzen Tab-Block überspringt) — siehe Befund 2.
  - **#7 Sprache-Select** und **#9 TTS-Modell-Select** haben keinen `disabled`-Pfad beim Service-Gate.
  - **#11 Quelltext-Textarea** hat keinen Max-Length-Hint im UI; Backend `format_dialogue_with_llm` validiert die Länge ebenfalls nicht (siehe Befund 13).
  - **#12 Skript-Generieren-Btn** hat `Generiert …` als Text-Swap, aber **das Spinner-SVG-Icon ist statisch** — keine Drehung/CSS-Animation (siehe Befund 14).
  - **#13 Skript-Textarea** ist während `generate-podcast-btn`-Run **nicht readonly** — User kann mid-pipeline editieren, ohne dass das Auswirkung hat (Befund 15).
  - **#19 Generate-Podcast-Btn** zeigt **keinen queued/started/stage-Indicator** während des Polling-Loops; nur der Cancel-Btn-Counter zählt Wand-Sekunden.
  - **#19 Generate-Podcast-Btn** wird `.hidden`-getoggelt statt mit Text-Swap-Loading-State (anders als #12) — UI-Inkonsistenz innerhalb derselben Pane.
  - **#20 Cancel-Btn** ist nur ein Frontend-Cancel, kein Backend-Cancel (Befund 9). Disposition-Frage in F4-REVIEW: ist das ein Bug oder bewusst-soft?
  - **#21 Polling-Loop** hat kein Frontend-Timeout (Befund 6) — bei kaputten Backend-Responses (z.B. `r.ok===false` mit JSON-Body ohne `status`-Feld) läuft der Loop endlos auf `undefined`.
  - **#22 `/podcast-status/<job_id>` Backend-Schema** konflatiert `queued` mit `started` zu `processing` und lässt andere RQ-Statuses (`deferred`, `scheduled`, `stopped`, `canceled`) als raw durchpurzeln, was Frontend ungehandelt lässt.
  - **#23 Worker** publiziert keinerlei Progress (`job.meta` ungenutzt; nur Logger-Output landet im Worker-Container-Log, das vom UI nicht erreichbar ist).
  - **#26 Audio-Element** error-Banner ist persistent (`autoDismissMs=null`), aber kein Retry-Btn neben dem Banner.
  - **#28 Download-Anchor** hat keinen Loading-State und keinen Error-Pfad (instant; wenn Browser-Download-Block, kein Feedback).
  - **#28 Download-Anchor** zeigt erfolgreiches Toast „✓ Podcast heruntergeladen" auch wenn der Browser den Download blockiert hat oder `download`-Attribut nicht respektiert (Code zeigt Toast vor jedem Browser-Download-Verhalten).

- **Code↔live-Divergenzen-Verdacht (↯, Code-deduced, **kein Browser-Walkthrough verfügbar**):**
  - **#1 Tab-Button-`aria-disabled` ohne JS-Handling** — siehe Befund 1.
  - **#3 Service-Gate-Banner verbleibt sichtbar trotz Pane-Inhalt** — User sieht warning + alle Felder dahinter; Generate-Buttons sind disabled (CSS opacity 0.4), Selects nicht. Inkonsistenz zu Live/File-Tab-Configuration-Error.
  - **#19 Cancel-Counter beginnt erst nach erstem 2s-Sleep** — wenn der erste Polling-Roundtrip schon `completed` zurückgibt (kurzer Job), sieht der Counter nie etwas anderes als „Abbrechen". Bei 30s+-Jobs ist das fine.
  - **#20 Cancel-Btn-Counter im finally-Block resettet** — auch nach erfolgreichem Download wird der Btn wieder auf "Abbrechen" gestellt, dann aber in `setPodcastGenerating(false)` `.hidden`. Letzter sichtbarer Counter-Wert vor dem Verstecken könnte glitchen — aber wahrscheinlich durch `.hidden` sofort verdeckt.
  - **#26 Audio-Element + Result-Container** werden nach erfolgreicher Generation beim **nächsten Generate-Klick** nicht zurückgesetzt: `podcastResultContainer.classList.add('hidden')` passiert in Generate-Btn-Handler bevor Polling startet ([static/js/audio_converter.js:816](static/js/audio_converter.js#L816)), aber `podcastAudioSource.src` und `downloadPodcastBtn.href` werden erst beim nächsten Erfolgs-Pfad neu gesetzt. Bei Failure oder Cancel bleibt die alte Blob-URL drin (revoked nicht — kein `URL.revokeObjectURL` im Code) → potentielles Memory-Leak (Befund 17).
  - **#28 Download-Btn `href`** wird nicht zurückgesetzt zwischen Generationen — wenn Generate fehlschlägt nach erfolgreicher vorheriger Generation, hat der Download-Btn noch die alte Blob-URL als `href`, ist aber unsichtbar (Container `.hidden`). Theoretisch beim nächsten Erfolg überschrieben.

- **Unverifizierbare States (?):**
  - **`:focus-visible`-Tönung der Selects** (#7, #8, #9) — visuelle Erkennbarkeit nicht aus Code prüfbar.
  - **Mode-Radio-Card-Selected-State** (#5/#6) — F-2.1 Befund 17 hatte das als „sehr subtil" markiert; mit `:focus-visible`-Ring (P15) sollte es jetzt klarer sein. Live-Walkthrough wäre wertvoll.
  - **Browser-Download-Block-Verhalten** (#28) — wenn ein Pop-up-Blocker oder Download-Block aktiv ist, ist der Toast trotzdem grün. Nicht aus Code prüfbar.
  - **Audio-Codec-Unterstützung im Browser** (#26) — bei Browser ohne WAV-Decoder-Support würde der `error`-Event greifen.
  - **Long-Job-Verhalten >300s** — Worker macht Multi-Chunk + 2s-Tick. Counter wächst sichtbar; UI bleibt funktionsfähig (kein Stale-Indicator). Sehr lange Skripte (z.B. 200+ Zeilen → 3+ Chunks) sind aus Code möglich, aber praktisch nicht testbar in F4-REVIEW ohne Live-Walkthrough mit echtem Skript.
  - **Worker-Crash (OOM, SIGKILL)** während laufender Generierung — RQ markiert Job als failed, Frontend bekommt `failed`-Status mit RQ-Worker-Death-Error. Nicht aus Code direkt geprüft.

- **Unterschiede zu F-1.1 / F-2.1 / F-3.1 in der State-Coverage:**
  - **Eigene Async-State-Klasse**: weder F-1.1 (sync Conversion) noch F-2.1 (sync Transcribe) noch F-3.1 (sync CRUD) hatten async-multi-stage-Pipelines. Die Konflation `queued+started→processing` und das Fehlen von `stage-progress` ist eine **podcast-flow-spezifische** Schwäche.
  - **Cancel-Pattern ist neu**: F-2.1 hatte „Live Stop Recording" als Symmetrie zu Start, aber das war kein Job-Cancel. Hier ist Cancel **soft** (Frontend-only), was bei der Library/Document-Inventur kein Thema war.
  - **Ergebnis-Persistenz** ist hier flüchtig: Audio-Blob nur im Browser-Memory, File auf Server nach Download gelöscht. Im Gegensatz zur Library, wo Inhalte persistent sind. Implikation: Browser-Reload nach Generation = Podcast weg.
  - **Service-Gate** (`!gemini_api_key_set`) lässt das Pane sichtbar, anders als der Audio-Tab-Service-Gate (`!deepgram_api_key_set`), der den ganzen Tab-Block hidden macht. Tab-übergreifende Inkonsistenz innerhalb derselben Seite.

---

## Befunde (separat — nicht in Tabelle gemischt)

Diese Auffälligkeiten gehen **über fehlende States hinaus** und sind echte Implementierungs-Befunde aus dem Code-Reading. F4-REVIEW (Stage 2) muss entscheiden, welche davon Heuristik-Findings werden und welche separate Bug-Tickets:

1. **Tab-Button "Text zu Podcast" `aria-disabled` ohne JS-Click-Block.** Template setzt `aria-disabled="true"` wenn `!gemini_api_key_set`. Aber JS in `audio_converter.js` (Zeilen 35–54) iteriert über alle `.tab-btn` und switcht ohne Disabled-Check. Resultat: User klickt gating-tabbed Button, Pane wird trotzdem aktiviert, sieht den `c-alert--warning` und Service-Gate-Buttons, aber kann nichts tun. Pre-existing-Item, war auch in F-2.1 nicht behandelt. Sev: niedrig (Single-User, Service-Config-Issue tritt selten auf). Disposition: nur Finding (Bug-Ticket-Kandidat falls Service-Gating UX-Pattern in F4-PATTERNS standardisiert wird). Code-Anker: [templates/audio_converter.html:23-27](templates/audio_converter.html#L23-L27), [static/js/audio_converter.js:35-54](static/js/audio_converter.js#L35-L54).

2. **Service-Gate-Banner Inkonsistenz Live/File vs Podcast.** Live/File-Tab Configuration-Error rendert über das `{% if not deepgram_api_key_set %}`-Banner direkt im Pane mit voll erhaltenem Markup darunter (auch in F-2.1 #11 nur teilweise erfasst — F-2.1 hatte den Code-Pfad anders gelesen). Im aktuellen Stand sehen Live/File und Podcast beide das Banner UND den Pane-Inhalt mit disabled-Buttons. **Konsistent** im aktuellen Stand. Die F-2.1-Beobachtung „der ganze Rest der Seite ist tot" ist mittlerweile veraltet — der Code wurde inzwischen geändert. Disposition: F-2.1-Doc-Korrektur (out-of-scope hier; nur Beobachtung). Code-Anker: [templates/audio_converter.html:42-48,80-86,120-126](templates/audio_converter.html#L42-L48).

3. **Legacy `/generate-podcast` Google-TTS-Pfad — Dead-Code-Kandidat.** Die Route in [app_pkg/podcasts.py:72-125](app_pkg/podcasts.py#L72-L125) hat **keinen Frontend-Trigger**: `grep "/generate-podcast"` (ohne `-gemini-`) im static/ + templates/ liefert nur den `#generate-podcast-btn` (das ist die UI-ID, kein URL-Aufruf). `app_pkg/podcasts.py` registriert die Route, der `google_tts_service` wird im Service-Singleton (`app.py:60` o.ä.) initialisiert, aber **niemand fetcht** auf diese URL. Auch `/api/get-google-voices` (Zeile 127) und der zugehörige Voice-Picker sind nirgends im UI erreichbar. Master kann entscheiden, ob das in einer Hygiene-Welle entfernt wird. Disposition: nur Finding (Dead-Code-Removal in eigener Welle). Code-Anker: [app_pkg/podcasts.py:72-136](app_pkg/podcasts.py#L72-L136).

4. **Worker pflegt `job.meta` nicht — kein Stage-Progress publizierbar.** `tasks.generate_podcast_task` macht nur `logger.info` für Stage-Events (filter / chunking-decision / per-chunk-progress). RQ stellt `job.meta` als persistierten Dict zur Verfügung, der vom Frontend per `/podcast-status/<job_id>` ausgelesen werden könnte (z.B. `{"stage": "tts_chunk_3_of_7"}`). Heute weiß der User nur „läuft seit Ns". Bei Multi-Chunk-Podcast (10+ Min Audio = 5+ Chunks) ist der Counter zwar sichtbar, aber ohne Kontext, ob 50%/90%/Stuck. Sev: mittel (Wartezeit-UX-Friktion). Disposition: Finding — Pattern-Sprint F4-PATTERNS sollte entscheiden ob `job.meta`-Stage-Progress die Investition wert ist (Worker-Code + Backend-Route + Frontend-Render). Code-Anker: [tasks.py:32-58](tasks.py#L32-L58), [services/gemini/tts.py:154-216](services/gemini/tts.py#L154-L216), [app_pkg/podcasts.py:181-191](app_pkg/podcasts.py#L181-L191).

5. **`queued` und `started` werden zu `processing` konflatiert.** Backend-Route [app_pkg/podcasts.py:188-189](app_pkg/podcasts.py#L188-L189): `elif status in ['queued', 'started']: return jsonify({"status": "processing"})`. User-relevante Information geht verloren: „Der Worker hat den Job nicht aufgegriffen" (queued, kann auf 0 Worker-Verfügbarkeit hindeuten — zwar im Single-User-Setup nur ein Worker und keine parallel-Last, aber während Worker-Restart denkbar) vs. „Der Worker arbeitet aktiv". Sev: niedrig–mittel (Single-User-App, wenig sichtbarer Effekt). Disposition: Finding — F4-PATTERNS soll entscheiden ob differenzieren. Code-Anker: [app_pkg/podcasts.py:181-191](app_pkg/podcasts.py#L181-L191).

6. **Polling-Loop hat kein Frontend-Timeout und keinen `r.ok`-Check.** [static/js/audio_converter.js:839-852](static/js/audio_converter.js#L839-L852): Loop fragt `/podcast-status/<id>` ab und prüft nur `if (status === 'failed')`. Bei `r.ok===false` (z.B. 404 NoSuchJobError nach RQ-result-TTL-Ablauf — RQ-default 500s — oder 500 transport-error) wird `r.json()` per `safeJSON` geparst. Wenn der Body `{"error": "..."}` enthält statt `{"status": "..."}`, wird `status` zu `undefined`, das ist nicht `'pending'` und nicht `'processing'` → Loop bricht ab → kein `failed`-Throw → kein `completed`-Branch → fällt in den Download-Pfad mit invaldem job_id → throw → catch → showAlert(danger). **Funktional ist das OK** (User bekommt Failure-Banner), aber spricht nicht den eigentlichen Fehler aus. Sev: niedrig. Disposition: nur Finding. Code-Anker: [static/js/audio_converter.js:837-862](static/js/audio_converter.js#L837-L862).

7. **RQ-Statuses `deferred` / `scheduled` / `stopped` / `canceled` werden ungehandelt durchgereicht.** [app_pkg/podcasts.py:190-191](app_pkg/podcasts.py#L190-L191): `else: return jsonify({"status": status})` — der raw RQ-Status fließt ins Frontend. Frontend hat nur `'pending'`, `'processing'`, `'completed'`, `'failed'`-Branches. Andere Werte: Loop bricht ab (siehe Befund 6). Praktisch tritt `deferred` ein wenn ein Dependency-Job läuft (kein Anwendungsfall hier), `scheduled` bei `enqueue_at` (auch nicht hier), `stopped`/`canceled` bei explicit `job.cancel()`/`send_stop_job_command` (von außen, nicht aus dem Pfad hier). Real wenig wahrscheinlich, aber Code-Defensiv-Mängel. Sev: niedrig. Disposition: nur Finding. Code-Anker: s.o.

8. **Generate-Btn `.hidden`-Toggle vs Skript-Btn Text-Swap = Loading-Pattern-Inkonsistenz.** `#generate-script-btn` zeigt Loading via Text-Swap "Generiert …" + `.disabled`. `#generate-podcast-btn` zeigt Loading via `.hidden` + Cancel-Btn-Übernahme. Innerhalb derselben Pane zwei verschiedene Loading-Patterns. F-2-Pattern hat showAlert/showToast/Helper konvergiert, aber nicht die Loading-Toggle-Pattern. Sev: niedrig. Disposition: Finding — F4-PATTERNS ist die Stelle für „Loading-State-Convention". Code-Anker: [static/js/audio_converter.js:728-731,762-763](static/js/audio_converter.js#L728-L731) vs. [static/js/audio_converter.js:634-646,815](static/js/audio_converter.js#L634-L646).

9. **Cancel ist Frontend-only — Worker läuft weiter.** [static/js/audio_converter.js:648-652](static/js/audio_converter.js#L648-L652) setzt nur `podcastCancelRequested = true`. Polling-Loop bricht ab, Worker arbeitet bis Job fertig (kann Minuten dauern), Output-File landet in `OUTPUT_DIR`, wird **nie heruntergeladen** → bleibt liegen. Frontend-Message ist ehrlich („Backend-Job läuft im Hintergrund weiter."), aber:
  - Kein Cleanup-Mechanismus in `OUTPUT_DIR` für orphaned Files.
  - RQ bietet `Job.cancel()` (cancel queued only) und `send_stop_job_command(connection, job_id)` (stop running) — beides ungenutzt.
  - User könnte denselben Cancel-Knopf-Klick mit der Erwartung „spar mir den Gemini-Credit" interpretieren, was hier nicht eintritt — Credit wird konsumiert.
  Sev: mittel (Disk-Space-Wachstum + Credit-Verschwendung-Erwartungs-Mismatch). Disposition: Finding + Bug-Ticket-Kandidat (echte Backend-Cancel via RQ-API + UI-Microcopy-Korrektur). Code-Anker: [static/js/audio_converter.js:648-652,854-858](static/js/audio_converter.js#L648-L652), [app_pkg/podcasts.py:138-164](app_pkg/podcasts.py#L138-L164).

10. **Browser-Reload während Polling = job_id verloren.** Kein LocalStorage / Cookie für laufenden `job_id`. User reloadet → leerer Podcast-Pane → Worker läuft im Hintergrund weiter → File landet in OUTPUT_DIR und ist verwaist. Sev: mittel (gleiche Problematik wie Befund 9 — orphaned Files + Credit-Verschwendung). Disposition: Finding — Pattern-Sprint F4-PATTERNS soll entscheiden ob `localStorage.podcastInFlight = job_id` eingebaut wird, um Reload-Resume zu ermöglichen. Code-Anker: [static/js/audio_converter.js:768-885](static/js/audio_converter.js#L768-L885).

11. **Audio-Blob-URL wird nicht revoked.** [static/js/audio_converter.js:866-871](static/js/audio_converter.js#L866-L871): `const audioUrl = URL.createObjectURL(audioBlob); podcastAudioSource.src = audioUrl; downloadPodcastBtn.href = audioUrl;` — beim **nächsten** Generate-Klick wird die alte Blob-URL nicht via `URL.revokeObjectURL` freigegeben. Bei mehrfacher Generation in derselben Tab-Session sammeln sich Blobs im Browser-Memory. Single-User-App-Kontext mildert; Memory-Hygiene-Spuren-Lücke. Sev: niedrig. Disposition: nur Finding. Code-Anker: [static/js/audio_converter.js:866-871](static/js/audio_converter.js#L866-L871).

12. **`/generate-gemini-podcast` hat keine Allowlist auf `language` / `tts_model`.** [app_pkg/podcasts.py:138-164](app_pkg/podcasts.py#L138-L164): nur `dialogue` + `language` + `tts_model` werden aus `data` gepickt, **ohne Validation**. Worker-`generate_podcast_task` reicht das an `services.gemini.tts.generate_podcast` durch, das `tts_model` validiert (mit Default-Fallback). `language` läuft komplett ungeprüft durch — wird im Worker nur in `prompts.py`/`script.py` für Sprach-Hints verwendet, ohne Strict-Validation. `dialogue` ist eine Liste mit dict-keys `speaker`/`style`/`text`, aber niemand prüft die Struktur — bei kaputtem Input würde der Worker später crashen → `failed`-Status. F-013 hat die Allowlist nur für `/format-dialogue-with-llm` etabliert, nicht für `/generate-gemini-podcast`. Sev: niedrig (Single-User, kein Untrusted-Input), aber Defensive-Lücke. Disposition: Finding (Konsistenz-Gap mit F-013). Code-Anker: [app_pkg/podcasts.py:147-149](app_pkg/podcasts.py#L147-L149) vs. [app_pkg/podcasts.py:265-286](app_pkg/podcasts.py#L265-L286).

13. **`raw_text`-Eingabe hat keinen Max-Length-Check.** [app_pkg/podcasts.py:256-258](app_pkg/podcasts.py#L256-L258): nur `not raw_text or not raw_text.strip()`-Check. Bei sehr großem Quelltext (z.B. 100k+ Zeichen, Buch-Auszug) würde der LLM-Call entweder lange dauern, Token-Limit erreichen, oder fehlschlagen. Frontend gibt auch keinen Hint. Sev: niedrig (Single-User, eigene Diskretion), aber Cost-Awareness-Lücke. Disposition: nur Finding. Code-Anker: [app_pkg/podcasts.py:256-258](app_pkg/podcasts.py#L256-L258), [templates/audio_converter.html:189-191](templates/audio_converter.html#L189-L191).

14. **`#generate-script-btn` Loading-Spinner-Icon ist statisch.** [templates/audio_converter.html:194-198](templates/audio_converter.html#L194-L198): Das SVG (Sun/Spark-Icon) ist innerhalb `.generate-script-btn__icon`. Während Loading wird nur `.generate-script-btn__label`-Text geändert ([static/js/audio_converter.js:728-731](static/js/audio_converter.js#L728-L731)). Kein CSS-Animation auf dem SVG (z.B. `animation: spin 1s linear infinite`). Visuell statischer Button mit Text-Swap = wirkt weniger „aktiv" als ein animierter Spinner. Sev: niedrig (kosmetisch). Disposition: Finding — F4-PATTERNS könnte standardisieren (Spinner-Icon-Animation oder explicit Loading-Indicator). Code-Anker: [templates/audio_converter.html:194-198](templates/audio_converter.html#L194-L198), `static/css/style.css` für `.mic-button__spinner` (existiert bereits — könnte als Vorlage dienen).

15. **Skript-Textarea nicht readonly während Podcast-Generierung.** [static/js/audio_converter.js:768-885](static/js/audio_converter.js#L768-L885): User kann `#podcast-script` mid-pipeline bearbeiten. Hat **keine** Auswirkung auf den laufenden Job (Worker hat sein eigenes `dialogue[]` schon bekommen), kann aber User verwirren — „ich habe es geändert, warum klingt der Podcast wie vorher?". F-2 hat das Live-Transcript-Pattern (`setLiveTextareaReadonly`) etabliert; analog wäre denkbar. Sev: niedrig (Single-User, gewohnheitsmäßiges Tippen, eher kosmetisch). Disposition: Finding — F4-PATTERNS soll entscheiden ob das Live-Pattern auch hier gelten soll. Code-Anker: [static/js/audio_converter.js:142-160,815-885](static/js/audio_converter.js#L142-L160).

16. **Skript-Parsing ist fragil.** [static/js/audio_converter.js:782-806](static/js/audio_converter.js#L782-L806): `for (const line of scriptText.split('\n'))` mit String-Split-Pattern. Regeln:
  - Skip wenn leer, beginnt mit `#`, beginnt mit `**`.
  - Erforderlich: `:`-Trenner.
  - Speaker-Bracket-Style-Notation: `Speaker [style]: text` parst durch `split('[')` und `split(']')`.
  Probleme:
  - Wenn Sprecher-Name `:`/`[`/`]` enthält (z.B. „Dr. Müller [warm]: ..." → würde funktionieren, aber „[`Section 1`] Kate: text" wird komplett verschluckt).
  - Nicht-Standard-Markdown wie `>` Zitat-Markup, `-` List-Markup, oder Skript-Sections (`# Intro`/`## Outro`) gehen alle in den Skip-Branch (Skip aus dem `# / **`-Filter ist intentional, aber List-Items werden trotz `:`-Vorhandensein zu invaliden Speaker-Namen).
  - Voice-Map ist hardcoded auf {Kate, Max} → andere Speaker fallen auf Default 'Kore'. Bei `numSpeakers=2` kann das den Effekt der Setup-Karte unterlaufen.
  - **Nur das erste `:`-Vorkommen pro Zeile** wird als Trenner genommen (`split(':', 1)` in JS-Land — Achtung: JS `split(str, limit)` semantik anders als Python; in JS limitiert das nur die Zahl der Outputs, nicht das splitten — also der String wird voll gesplittet, dann auf 1 Element gekürzt. `parts[0]` ist der Speaker-Teil, dann substring-rückgewinnung des Texts via Length-Berechnung. Funktional korrekt, aber subtil.)
  Sev: mittel (User-fehlendes-Format → Empty-dialogue → showAlert(danger) "Skript konnte nicht gelesen werden" → das ist abgedeckt). Edge-Cases nicht graceful. Disposition: Finding — F4-PATTERNS soll entscheiden ob ein robusteres Parser-Helper sinnvoll ist (oder ein schema'd Editor). Code-Anker: [static/js/audio_converter.js:780-812](static/js/audio_converter.js#L780-L812).

17. **`URL.createObjectURL` ohne `URL.revokeObjectURL`** — siehe Befund 11. (Doppelt aufgenommen aus Code-Review-Sicht; eine Disposition.)

18. **Worker-File wird beim ersten Download gelöscht — kein Re-Download möglich.** [app_pkg/podcasts.py:230-233](app_pkg/podcasts.py#L230-L233): `os.unlink(real_path)` direkt nach `BytesIO`-Read. Wenn der Browser-Download fehlschlägt (z.B. User hat Pop-Up-Blocker oder Container-Restart zwischen Generate und Download), kann der User nicht erneut auf `/podcast-download/<job_id>` triggern — der File ist weg, RQ-Job ist `finished` (nicht failed), aber `job.result` zeigt auf einen non-existenten Pfad → 404 „File not found on server". Frontend-seitig gibt es eh keinen Re-Download-Knopf — der User muss den ganzen Generate-Flow erneut anstoßen (= Credit erneut verbrennen). Sev: mittel (Edge-Case mit hohem Cost). Disposition: Finding + Bug-Ticket-Kandidat (z.B. File-TTL 1h statt sofort-Löschung; oder Re-Download-Pfad in der UI). Code-Anker: [app_pkg/podcasts.py:230-233](app_pkg/podcasts.py#L230-L233), [static/js/audio_converter.js:860-866](static/js/audio_converter.js#L860-L866).

19. **Toast bei Download-Click feuert immer, auch ohne echten Download.** [static/js/audio_converter.js:654-659](static/js/audio_converter.js#L654-L659): `if (!downloadPodcastBtn.getAttribute('href')) return;` — guard nur gegen leeren `href`. Wenn `href` zu invaler Blob-URL gesetzt ist (z.B. nach `URL.revokeObjectURL` falls das eingebaut wird) oder Browser-Download-Block, feuert der Toast trotzdem grün-success. Sev: niedrig. Disposition: nur Finding. Code-Anker: [static/js/audio_converter.js:654-659](static/js/audio_converter.js#L654-L659).

20. **Helper-Reuse-Spuren aus `_utils.js`** (analog F-3.1 Pflicht für Phase-2-Konsistenz):
  - **Verwendet** in `audio_converter.js`-podcast-Block:
    - `safeJSON` ([static/js/audio_converter.js:750-754](static/js/audio_converter.js#L750-L754) Skript-Generation; [:830-835](static/js/audio_converter.js#L830-L835) Generate-Podcast; [:846](static/js/audio_converter.js#L846) Polling) — vollständig.
    - `showAlert(level)` ([static/js/audio_converter.js:712-715,758-760,772-775,808-811,855-857,877-879](static/js/audio_converter.js#L712-L715)) — vollständig.
    - `showToast` ([static/js/audio_converter.js:657](static/js/audio_converter.js#L657) Download-Confirm) — punktuell.
    - `confirmIfLong` ([static/js/audio_converter.js:700-703](static/js/audio_converter.js#L700-L703) Reset-Prompt) — punktuell.
  - **Nicht verwendet** (weil nicht relevant in podcast-flow):
    - `formatFileSize` — kein File-Upload im Podcast-Flow.
    - `formatDatetimeLocalNow` — keine Date-Inputs.
    - `fallbackCopyText` — kein Copy-Btn im Podcast-Flow.
  - Beobachtung: **Helper-Reuse für podcast-flow ist sehr solide** — alle relevanten Async/Feedback-Pfade gehen über `safeJSON` + `showAlert`/`showToast`. Inkonsistenzen aus F-2.1 (`alert()`-Pfade, drei konkurrierende Error-UX-Patterns) sind im Podcast-Block durchgehend bereinigt.

---

## Live-Walkthrough-Lücken (besonders wertvoll für F4-REVIEW)

Async-States und Multi-Stage-Pipelines sind **besonders schwer aus Code abzuleiten** — Code-only-Inventur deckt geschätzt 60–70% (niedriger als F-1.1 / F-2.1 / F-3.1 wegen Async-Komplexität). Master/F4-REVIEW würde von einer Live-Job-Generierung in folgenden Test-Szenarien profitieren:

**Test-Anleitung 1 — Kurzer Podcast (Single-Chunk, ~30–60s Generation):**
- Setup: Monolog, Deutsch, conversational, Flash-TTS.
- Quelltext: ~200 Wörter beliebiger Lese-Stoff (z.B. Wikipedia-Intro).
- Schritt: Skript generieren (synchron) → Podcast generieren.
- Beobachten: Zeitachse Counter-Updates (1. Tick @ 2s, 2. Tick @ 4s), Banner-Auftritt bei Erfolg, Audio-Player-Auftritt + scrollIntoView, Download-Toast.
- Lücken die das füllt: Counter-Sichtbarkeit-Timing (Befund 6), Result-Container-Reveal-Animation (#24 nicht im Code, aber Browser-Default), Audio-Player-Render-Delay.

**Test-Anleitung 2 — Multi-Chunk-Podcast (3+ Chunks, ~3–5 Min Generation):**
- Setup: Dialog, Deutsch, conversational, Flash-TTS.
- Skript: manuell ~120 Zeilen `Kate [warm]: ... / Max [neutral]: ...`-Format einkopieren (Split-Schwelle ist 80 Zeilen / 3000 Zeichen).
- Schritt: Podcast generieren (sollte 2–3 Chunks ergeben).
- Beobachten: Counter-Wert während Chunk-Übergänge (`INTER_CHUNK_DELAY = 1.0s`), Cancel-Btn-Counter-Glitch zwischen Chunks, ob das UI den Worker-Stage-Wechsel sichtbar macht (Hypothese: nein, Counter zählt nur Wand-Sekunden).
- Lücken die das füllt: Stage-Progress-Sichtbarkeit (Befund 4), Long-Job-Counter-Verhalten (#19 unverifizierbarer State).

**Test-Anleitung 3 — Cancel-mid-Generation:**
- Setup: wie Test 2, aber nach ~30s Cancel-Btn klicken.
- Beobachten:
  - Welcher Counter-Wert war zuletzt sichtbar?
  - Wie schnell verschwindet der Cancel-Btn (sollte sofort durch `setPodcastGenerating(false)`, aber 2s-Tick könnte stalen).
  - Zeigt der Banner („Backend-Job läuft im Hintergrund weiter.") auf?
  - **Mintbox-seitig**: `docker logs converter-worker -f` während der nächsten 2 Min — läuft der Worker den Job zu Ende? File in `OUTPUT_DIR` hinterher? Disk-Wachstum?
- Lücken die das füllt: Befund 9 (Cancel-Soft-Behavior); Disk-Forensik.

**Test-Anleitung 4 — Browser-Reload während Generation:**
- Setup: wie Test 2, nach ~30s Cmd+R (Tab-Reload).
- Beobachten:
  - Pane ist leer.
  - Worker läuft weiter (`docker logs`).
  - Output-File landet in `OUTPUT_DIR`, niemand holt es ab.
- Lücken die das füllt: Befund 10 (Reload-Resume nicht implementiert).

**Test-Anleitung 5 — Network-Drop während Polling:**
- Setup: wie Test 2, nach ~10s Network-Tab in DevTools auf „Offline" stellen.
- Beobachten: Polling-Loop catch-Pfad. Welche Banner erscheinen? Wenn Network wieder online: läuft der Loop weiter oder ist er weg?
- Lücken die das füllt: Befund 6 (Polling-Robustheit).

**Test-Anleitung 6 — Service-Gate (`gemini_api_key_set=False`):**
- Setup: `.env` GEMINI_API_KEY entfernen + Container-Restart.
- Beobachten: Pane-Layout (Banner-Mountpoint sichtbar, Pane-Inhalt sichtbar mit disabled-Buttons; vs. F-2.1 #11 Beobachtung „der ganze Rest der Seite ist tot"). Was ist die aktuelle Realität?
- Lücken die das füllt: Befund 2 (Service-Gate-Konsistenz vs. Live/File).

**Test-Anleitung 7 — Re-Download-Pfad:**
- Setup: Erfolgreichen Podcast generieren, herunterladen, dann den Browser-Download-Dialog abbrechen (oder Pop-Up-Blocker aktiv).
- Beobachten: Toast feuert grün-success („Podcast heruntergeladen"). Ist die Datei auf der Disk? Klick auf Download-Btn nochmal: passiert was?
- Wenn der File auf Server gelöscht ist (`os.unlink`), erneuter `/podcast-download/<job_id>`-Call → 404. Aber der `href` zeigt auf Browser-cached-Blob-URL — Browser probiert das, kommt zum gleichen Result.
- Lücken die das füllt: Befund 18 (File-Cleanup-vs-Re-Download), Befund 19 (Toast-immer-grün).

**Code-only-Inventur-Coverage geschätzt:** 65%. Die obigen 7 Test-Anleitungen würden in F4-REVIEW geschätzt 25% Lücken-Schluss bringen (besonders Befunde 4, 9, 10, 18). Der Rest (10% — Worker-Crash, OOM, sehr lange Jobs) wäre nur durch destruktives Testing reproduzierbar und out-of-scope für F4-REVIEW.

---

## Disposition-Übersicht (Befund 1–19, vorläufig — final in F4-REVIEW)

- **Finding + Bug-Ticket-Kandidat (3):** Befund 9 (Cancel-Soft + Worker läuft weiter + Credit-Verschwendung), Befund 18 (File-Cleanup-vs-Re-Download), Befund 4 (Stage-Progress kein job.meta — Bug-Ticket nur falls UX-Friktion in F4-REVIEW als „mittlere Priorität" eingestuft).
- **Nur Finding (15):** Befund 1 (Tab-aria-disabled-Click), Befund 2 (Service-Gate-Inkonsistenz Doc-Korrektur), Befund 5 (queued/started-Konflation), Befund 6 (Polling-Robustheit), Befund 7 (RQ-Status-Pass-through), Befund 8 (Loading-Pattern-Inkonsistenz), Befund 10 (Reload-Resume-Fehlen), Befund 11 (Blob-URL-Revoke-Lücke), Befund 12 (Allowlist-Gap), Befund 13 (raw_text-Max-Length), Befund 14 (Spinner-Icon-statisch), Befund 15 (Skript-Textarea-readonly-pattern), Befund 16 (Skript-Parsing-Fragilität), Befund 19 (Download-Toast-immer-grün), Befund 20 (Helper-Reuse-Beobachtung — solide). Befund 17 ist redundant mit Befund 11 (transparent als „doppelt aufgenommen" markiert) und zählt nicht extra.
- **Pre-Existing-Item / Erwähnung (1):** Befund 3 (Legacy `/generate-podcast` Dead-Code-Kandidat — Hygiene-Welle).

---

**Hinweis:** Diese Stufe-1-Datei enthält bewusst **keine** Pattern- oder Microcopy-Vorschläge. Diese folgen in Stufe 3 (`F4-PATTERNS`-Sprint) nach dem Heuristik-Review (`F4-REVIEW`-Sprint). Die Stage-2-Review wird auch entscheiden müssen, welche der 19 Befunde oben echte Bug-Tickets werden und welche als Findings in den Sev-Score-Pool wandern. Die Async-Pipeline-Mapping-Sektion am Doc-Anfang ist ein neuer Beitrag zu künftigen async-Features (z.B. wenn der Highlights/Reader-Layer aus dem Readwise-Ersatz async-Importe bekommt).
