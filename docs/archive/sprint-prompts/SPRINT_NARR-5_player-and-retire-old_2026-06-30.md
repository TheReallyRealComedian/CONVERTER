# Sprint NARR-5 — Library-Audio-Player + Alt-Podcast-Flow stilllegen (L, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **636**). Du committest jede Phase selbst (Hash + push), **fokussiert**. Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **P1 = additiv (UI), P2 = subtraktiv (Deletion).** Kein Schema/Dep/Token.
>
> **Kontext**: NARRATION-Reframe, der **letzte** Sprint. NARR-1B/2/3/4 ☑ (Renderer + Persistenz + async Endpoint + Skill, **live**). Voller Entwurf [docs/narration_reframe.md](docs/narration_reframe.md). **P1 löst Olis konkreten Schmerz** (kein UI-Pfad zum Audio — er muss die URL von Hand bauen). **P2 macht den „großen Wurf" rund** (das alte unkontrollierte Podcast-Ding weg, das den Reframe ausgelöst hat).

## Warum & Entscheidungen (gesetzt)

- **P1 zuerst + eigener Sign-off**: der Player ist additiv + Olis akuter Bedarf → er kriegt ihn verifiziert, **bevor** P2 löscht. P2 (Deletion) ist isoliert + kann nicht den Player gefährden.
- **Retirement-Scope = die GANZE Alt-Podcast-Generierung** (sie ist „ersetzt/raus", Oli-Entscheid 2026-06-28). Raus: die Podcast-UI in `audio_converter.html`, die Alt-Podcast-Routes, `generate_podcast_task`, der genai-Synth (`tts.py::generate_podcast` + Chunking-Helfer + `synthesis.py`/`audio.py` falls **nur** vom Alt-Flow genutzt), die **Hack-Funktionen** (`calculate_tag_guidance`/`_TAG_DENSITY_MAP`, `format_dialogue_with_llm`, `parse_dialogue`), ihre `GeminiService`-Facade-Methoden, die zugehörigen Tests.
- **Behalten** (Narration braucht es): `GoogleTTSService`-Klasse (Narration nutzt `synthesize_narration`), `OUTPUT_DIR` + das `podcast_data`-Volume, `services/narration_render.py`/`narration_library.py`/`app_pkg/narration.py`, `tasks.py::generate_narration_task` + `update_job_stage`. **`_GEMINI_VOICES`** (Voice-Katalog): behalten (klein, kanonisch) — der Skill nutzt ihn, ein künftiges Read könnte ihn exposen.
- **Lösch-Disziplin**: **Caller-first** (UI/Route raus → dann die Funktion → `pytest` nach jedem Schritt), nie eine Funktion löschen, solange noch ein Caller steht. `grep -rn <name>` vor jedem Delete, um Dangling-Refs zu fangen.
- **Retry**: ein **Session**-Endpoint (`POST /api/narrations/<id>/retry`, `@login_required`, owner-scoped) re-enqueued aus dem **gespeicherten** `metadata.transcript`/`speakers`/`mode`/`tts_model` (kein neuer Agent-Call, kein Token nötig — Oli klickt in der UI). Nur bei `narration_status == 'failed'`.

## Verifizierte Code-Fakten (Master-gegroundet)

- **`library_detail.html`**: `type-badge`-Switch (Z.19–25, branched auf `conversion_type` — `audio_narration` ergänzen) · `#content-body` (Z.59, die Reader-Fläche; der **Player** gehört als Karte **darüber**, in `library-reader-hide`) · `c-btn-row` (Z.50, Action-Buttons) · `{% block head_extra %}` (Z.5) für Player-CSS/JS · `content_html` rendert den Transkript-Markdown (bleibt). `PageData` (inline) seedet die JS-Werte.
- **`library_detail.js`**: `window.PageData.conversionId` etc.; `fetch('/api/conversions/<id>')`-Muster. **Player-JS**: bei `audio_narration` `/api/narrations/<id>` pollen (NARR-3-Status, rekonziliert) → `ready` → `<audio src="/api/narrations/<id>/audio">` zeigen; `pending` → Spinner + weiterpollen; `failed` → Fehler + Retry-Button.
- **Alt-Podcast-Routes** ([app_pkg/podcasts.py](app_pkg/podcasts.py)): `/generate-podcast` (Z.74), `/api/get-google-voices` (129), `/generate-gemini-podcast` (140 → enqueued `generate_podcast_task`), `/podcast-status` (189), `/podcast-cancel` (240), `/podcast-download` (314), `/api/get-gemini-voices` (363), `/format-dialogue-with-llm` (368 → `gemini_service.format_dialogue_with_llm`). **Die Podcast-UI sitzt in [templates/audio_converter.html](templates/audio_converter.html)** (`download-podcast-btn` Z.256 etc.) — der Audio-Transkriptions-Teil der Seite **bleibt**, nur der Podcast-Generierungs-Teil raus.
- **Hack-Funktionen**: `format_dialogue_with_llm` ([services/gemini/script.py](services/gemini/script.py), Facade [services/gemini/__init__.py:34](services/gemini/__init__.py)) → nutzt `calculate_tag_guidance` (script.py:81, def [services/gemini/prompts.py:75](services/gemini/prompts.py)) + `parse_dialogue` (script.py:161, def [services/gemini/dialogue.py:33](services/gemini/dialogue.py)). Caller von `format_dialogue_with_llm`: nur `/format-dialogue-with-llm` (podcasts.py:416). Caller von `generate_podcast_task`: nur `/generate-gemini-podcast` (podcasts.py:176). `generate_podcast` (tts.py) Caller: nur `generate_podcast_task` (tasks.py:66). → **alle nur vom Alt-Flow**; nach UI/Route-Entfernung tot.
- **`dialogue.py::filter_metadata_lines`/`split_long_dialogue_turns`**: vom genai-Synth (`tts.py::generate_podcast`) genutzt — gehen mit, falls der Synth gelöscht wird. **`narration_render` nutzt sie NICHT** (eigene Chunking-Logik). `grep` bestätigen.
- **Tests**: [tests/test_podcasts.py](tests/test_podcasts.py) (30 Tests) decken den Alt-Flow ab → mit ihm entfernen/anpassen. Narration-Tests (`test_narration_*`) **unberührt**.
- **Live-Smoke-Pflicht** (P1): die Test-Suite rendert keine Templates (CLAUDE.md-Limit) → der Player wird **live** geprüft (dark+light, Audio spielt, States).

## Phase 1 — Library-Audio-Player + Retry (additiv) — Olis Schmerz

1. **`POST /api/narrations/<int:id>/retry`** ([app_pkg/narration.py](app_pkg/narration.py), `@login_required`, owner-404, type-404): nur wenn `narration_status == 'failed'` (sonst 409); aus `metadata` (`transcript`/`speakers`/`mode`/`tts_model`) `generate_narration_task` neu enqueuen, `metadata` → `status='pending'`, neue `job_id`, `error=None`, commit; → 202 `{status:'pending', job_id}`. (CSRF-geschützt — Session, kein exempt.) +Tests.
2. **Player-UI** in `library_detail.html` — eine Karte **über** `#content-body`, **nur** für `audio_narration` (`{% if conversion.conversion_type == 'audio_narration' %}`), in `library-reader-hide`: `<audio controls>` (src wird per JS gesetzt, wenn `ready`) + ein Status-Bereich (pending: Spinner „Wird vertont…"; failed: Fehlertext + „Erneut versuchen"-Button). `type-badge`-Switch um `audio_narration` → „Vertonung". Player-CSS token-driven (`--nm-*`), DE-Microcopy.
3. **Player-JS** in `library_detail.js`: bei `audio_narration` (aus `PageData.conversionType`) `/api/narrations/<id>` pollen (Throttle, Stop bei terminal); `ready` → `<audio src=/api/narrations/<id>/audio>` einblenden + Dauer/Stimmen aus `metadata`; `failed` → Fehler + Retry-Button (`POST …/retry` über den CSRF-Wrapper, dann weiterpollen); `pending` → Spinner. `PageData` um `conversionType` + die Narration-`metadata` erweitern (Template-seitig).
4. Pre-Flight `pytest` grün (Retry-Endpoint getestet). **Live-Smoke** (echte Instanz, eingeloggt, dark+light, narration 76/77): Player erscheint nur bei `audio_narration`, `ready` spielt das Audio, pending-Spinner, failed→Retry; 0 Console-Errors; Nicht-Narration-Detail unverändert.

**Stop + Bericht** (Oli kann den Player jetzt abnehmen).

## Phase 2 — Alt-Podcast-Flow stilllegen + Hack-Funktionen löschen (subtraktiv)

**Caller-first, `pytest` nach jedem Schritt:**
1. **UI raus**: den Podcast-Generierungs-Teil aus `templates/audio_converter.html` (Buttons/JS/Markup) — der **Audio-Transkriptions-Teil bleibt**. Zugehöriges JS in `static/js/` (audio_converter.js o.ä.) mit.
2. **Routes raus** ([app_pkg/podcasts.py](app_pkg/podcasts.py)): `/generate-podcast`, `/generate-gemini-podcast`, `/format-dialogue-with-llm`, `/podcast-status`, `/podcast-cancel`, `/podcast-download`, `/api/get-google-voices`, `/api/get-gemini-voices` — **alle Alt-Podcast-Routes**. (Prüfen, ob `podcasts.py::register` danach leer ist → dann das Modul + die `app.py`-Registrierung entfernen; sonst nur die toten Routes.)
3. **Task + Synth raus**: `tasks.py::generate_podcast_task`; `services/gemini/tts.py::generate_podcast` + die Chunking-Helfer (`_generate_with_chunking`/`_create_dialogue_chunks`); `services/gemini/synthesis.py` + `services/gemini/audio.py`, **falls** `grep` zeigt, dass sie nur vom Alt-Synth genutzt werden (Narration nutzt sie nicht). `GeminiService.generate_podcast`-Facade mit.
4. **Hack-Funktionen raus**: `services/gemini/script.py::format_dialogue_with_llm` (+ `GeminiService.format_dialogue_with_llm`-Facade) · `services/gemini/prompts.py::calculate_tag_guidance` + `_TAG_DENSITY_MAP` + die Paraphrase-Prompt-Builder · `services/gemini/dialogue.py::parse_dialogue` (+ `filter_metadata_lines`/`split_long_dialogue_turns` falls mit dem Synth tot). Nach jedem `grep -rn <name>` → keine Dangling-Refs.
5. **Tests**: `tests/test_podcasts.py` (30) entfernen/auf die verbleibenden Endpoints eindampfen; alle anderen grün halten.
6. **`GeminiService` prüfen**: bleibt es nach den Facade-Deletions noch sinnvoll genutzt (Script-Gen woanders? `pdf_extraction`?)? Falls die Klasse leer/tot wird, dokumentieren — aber **nicht** überdehnen; im Zweifel die Klasse + den `gemini_service`-Singleton stehen lassen (out of scope, nur die Alt-Podcast-Pfade raus).
7. `pytest` grün (deutlich weniger Tests — der Alt-Flow ist weg; Narration + Rest grün).

**Stop + Bericht.**

## Phase 3 — Wrap

1. **`docs/narration_reframe.md`** — NARR-5 ☑; den Sequenz-Hinweis (Hack-Funktionen werden in NARR-5 gelöscht) auf **erledigt** stellen.
2. **`CLAUDE.md`** — die *Gemini Models*-Zeile „genai-Pfad (Alt-Podcast-Flow)" + die Podcast-Architecture-Notes auf den Stand bringen (Alt-Flow weg; **„Podcast generation" → „treue Narration"** im Titel/Intro). Den Player in der NARR-Architecture-Bullet erwähnen.
3. **README.md** — Podcast→Narration-Reframe spiegeln, falls dort beschrieben.
4. **STATUS.md** + **BACKLOG.md**: NARR-5 ☑ done (Hashes); **NARRATION-Cluster komplett** (1✅→1B✅→2✅→3✅→4✅→5✅); nächstes Master-Item **Web-Article-Save (P2)**. **Bullet-Guard**.
5. **Memory** (`reference_*`): falls eine nicht-offensichtliche Retirement-Lehre auftaucht (z.B. der genai-Synth war vom Cloud-Renderer sauber entkoppelt → Deletion gefahrlos) — sonst nicht nötig.
6. **Sprint-Doc** einchecken. Finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. Deploy-Notiz: reines `git pull` + `up -d --build` (Templates/JS ins Image → `--build`; **kein** Schema/Dep/Token). Player live nach Deploy; der Alt-Podcast-Flow ist weg.

## Bewusst NICHT (Scope-Grenze)

- **`GoogleTTSService`-Klasse + `OUTPUT_DIR` + Narration-Code** unberührt (Narration braucht sie).
- **Audio-Transkriptions-Teil** von `audio_converter.html` bleibt (nur die Podcast-**Generierung** raus).
- **Kein** Schema/Dep/Token; **kein** Anfassen des `GeminiService`-Singletons über die Facade-Deletions hinaus (falls die Klasse sonst tot wird: dokumentieren, nicht im Akt löschen — separates Item).
- **Kein** Voice-Katalog-Read-Endpoint bauen (`_GEMINI_VOICES` bleibt nur als Datum).

## Akzeptanz

- [ ] **P1**: `audio_narration`-Player im Library-Detail (nur für den Typ; `<audio>` bei `ready`, Spinner bei `pending`, Fehler+Retry bei `failed`) + `POST /api/narrations/<id>/retry` (Session, failed-only, re-enqueue aus metadata) + Tests + **Live-Smoke** (dark+light, Audio spielt, 0 Console-Errors).
- [ ] **P2**: Alt-Podcast-Flow weg (UI + alle Alt-Routes + `generate_podcast_task` + genai-Synth + Hack-Funktionen `calculate_tag_guidance`/`format_dialogue_with_llm`/`parse_dialogue` + Facades), caller-first, **keine Dangling-Refs** (`grep` clean); `test_podcasts.py` entfernt/eingedampft.
- [ ] `GoogleTTSService`/`OUTPUT_DIR`/Narration + Audio-Transkription unberührt; `pytest` grün.
- [ ] Docs (narration_reframe/CLAUDE.md/README/STATUS/BACKLOG) auf „Reframe komplett"; NARRATION-Cluster ☑.
