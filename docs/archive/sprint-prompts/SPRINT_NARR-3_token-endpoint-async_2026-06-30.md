# Sprint NARR-3 — Token-Endpoint + async RQ + Reconcile (Generierung live verbinden) (L, 4 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **610**). Du committest jede Phase selbst (Hash + push), **fokussiert**. Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Backend + ein neuer Token** (`NARRATION_TOKEN`, env, fail-closed — **kein** Schema/Dep). Tests mocken den Renderer (`GoogleTTSService`) + die RQ-Singletons am etablierten Boundary (**keine** echten Gemini-Calls, **kein** echtes Redis).
>
> **Kontext**: NARRATION-Reframe, voller Entwurf [docs/narration_reframe.md](docs/narration_reframe.md). NARR-1B (Renderer) ☑ + NARR-2 (Persistenz) ☑. **Dieser Sprint verbindet beides live**: Agent POSTet eine Turn-Liste → `pending`-Conversion → RQ-Worker rendert + schreibt das WAV → Web rekonziliert beim Pollen auf `ready`. **Hier entsteht zum ersten Mal echtes Audio.**

## Warum & Entscheidungen (gesetzt — verifiziert)

- **⚠️ Worker hat KEINEN DB-Zugriff** (docker-compose: Worker mountet `podcast_data:/app/output_podcasts`, **nicht** `app_data:/app/data`; die SQLite-DB ist web-only). Der DB-freie Worker ist die **bewusste** Architektur (der Alt-Podcast-Flow fasst die DB nie an — der Worker returnt nur einen Pfad). **Architektur (Option B, gesetzt):**
  - **POST-Endpoint (web, hat DB)**: legt die `pending`-Conversion an + enqueued den Task.
  - **Worker (DB-frei)**: rendert → schreibt `narration_<conversion_id>.wav` (auf das geteilte `podcast_data`-Volume) → returnt; **flippt KEINE DB**. Fehler → `raise` (RQ markiert den Job `failed`, Exception in `job.exc_info`).
  - **Web rekonziliert beim Pollen** (`GET /api/narrations/<id>`): `pending` + Audio-Datei existiert → `ready` (+ Dauer aus dem WAV); Job `failed`/abgelaufen + keine Datei → `failed`. (Verworfen: Option A „`app_data` auf den Worker mounten + Worker flippt DB" — SQLite-Multi-Writer + Infra-Änderung gegen die Architektur.)
- **`NARRATION_TOKEN` (eigener Token)**, nicht `CARD_TOKEN`: Narration kostet **echtes GCP-Geld pro Call** (anders als die freien DB-Write-Surfaces Cards/Tags/Docwrite, die `CARD_TOKEN` teilen) → ein **unabhängig revozierbarer** Token ist gerechtfertigt. **Fail-closed** (503 ohne `NARRATION_TOKEN`) → bis Oli ihn setzt, ist der Endpoint sicher zu. `_authorize_card_write` **spiegeln** (nicht refactoren — cards.py unberührt), `_bearer_token`/`_resolve_target_user` aus `app_pkg.ingest` wiederverwenden.
- **Endpoint-Modul**: alles in **`app_pkg/narration.py`** (NARR-2 hat es als register-Modul angelegt + den Serve-Endpoint). NARR-3 ergänzt POST + Status + den Reconcile-Helper dort. (Bereits in `app.py` registriert — keine app.py-Registrierungs-Änderung.)
- **Reads (Status) = Session** (`@login_required`, owner-404) wie die Library-Reads; **Write (POST) = Token**. Beide owner-scoped über `_resolve_target_user` (Write) bzw. `get_owned_conversion` (Read).

## Verifizierte Code-Fakten (Master-gegroundet)

- **`_authorize_card_write`** ([app_pkg/cards.py:128](app_pkg/cards.py)): `expected = os.environ.get('CARD_TOKEN')`; ohne → 503; `_bearer_token()` + `hmac.compare_digest` → 401; `_resolve_target_user()` → 503 wenn None; sonst `(target, None)`. **Spiegel-Vorlage.** `_bearer_token`/`_resolve_target_user` aus [app_pkg/ingest.py:68](app_pkg/ingest.py) (importierbar, nutzt cards.py schon: `from .ingest import _bearer_token, _resolve_target_user`).
- **RQ-Task-Muster** [tasks.py:43](tasks.py) `generate_podcast_task(dialogue, language, tts_model)`: erstellt `GeminiService(api_key)` in-task, rendert, `job = get_current_job()` → `filename = f'{job.id}.wav'`, `shutil.move(temp, os.path.join(OUTPUT_DIR, filename))`, returnt `final_path`. **DB-frei.** `update_job_stage(stage, **extras)` schreibt `job.meta`. `os.makedirs(OUTPUT_DIR, exist_ok=True)` beim Import.
- **Enqueue-Muster** [app_pkg/podcasts.py:175](app_pkg/podcasts.py): `_app_module.task_queue.enqueue(generate_podcast_task, dialogue, language, tts_model, meta={...}, job_timeout=…)`; `job.meta.get('user_id')` für owner-Checks. (Genaue `job_timeout`/`result_ttl`-Args dort file-level grounden.)
- **`task_queue`** = `Queue(connection=redis_conn)` ([app.py:65](app.py)); Module-Singletons (`task_queue`, `Job`, `redis_conn`) via `import app as _app_module` (Test-Patch-Punkt). `google_tts_service`/`gemini_service` ebenfalls dort.
- **Renderer** (NARR-1B): `GoogleTTSService(creds).synthesize_narration(turns, voices, *, style_prompt=None, mode='two_speaker', language_code='de-DE', model_name=DEFAULT_NARRATION_MODEL) -> wav_path` (Temp-WAV). Creds = `GOOGLE_APPLICATION_CREDENTIALS` (im Worker-Env vorhanden).
- **Persistenz** (NARR-2, [services/narration_library.py](services/narration_library.py)): `narration_to_markdown(turns)`, `narration_audio_path(id)`, `build_narration_metadata(id, *, status, tts_model, speakers, transcript, …)`, `narration_metadata/_status/_audio_filename(conversion)`. **Serve** `GET /api/narrations/<id>/audio` ([app_pkg/narration.py](app_pkg/narration.py)) gated auf `narration_status == 'ready'`.
- **`Conversion`** ([models.py:36](models.py)): `title`(≤255 NOT NULL), `content`(NOT NULL), `metadata_json`(Text), `lifecycle_status` default `'inbox'`. id erst nach `db.session.flush()` bekannt → für den id-abgeleiteten `audio_filename`/`narration_audio_path` **flush vor** dem metadata-Set.
- **Validierung wiederverwenden**: `narration_render.validate_turns(turns, voices, mode)` (wirft ValueError) — der POST validiert den Kontrakt damit, mappt ValueError → 400.

## Phase 1 — Worker-Task (DB-frei) + Reconcile-Helper + Tests

1. **`tasks.py::generate_narration_task(conversion_id, turns, voices, style_prompt, mode, language_code, model_name)`** (DB-frei, Muster wie `generate_podcast_task`):
   - `creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')`; ohne → `raise ValueError`.
   - `from services import GoogleTTSService` → `svc = GoogleTTSService(creds)` → `temp = svc.synthesize_narration(turns, voices, style_prompt=…, mode=…, language_code=…, model_name=…)`.
   - `final = narration_audio_path(conversion_id)` (id-abgeleitet, OUTPUT_DIR); `update_job_stage('finalizing')`; `shutil.move(temp, final)`; return `final`.
   - **Kein DB-Zugriff.** Exception → loggen + `raise` (Job failed). (Temp-Cleanup macht der Renderer schon im except.)
2. **Reconcile-Helper** (web-seitig, in `app_pkg/narration.py` oder `services/narration_library.py` als pure-ish Funktion, die DB + Redis bekommt): `reconcile_narration(conversion) -> None` — wenn `narration_status(conversion) != 'pending'`: return; sonst:
   - **Audio-Datei existiert** (`os.path.exists(narration_audio_path(conversion.id))`) → metadata `narration_status='ready'`, `duration_seconds = _wav_duration(path)` (stdlib `wave`: `frames/framerate`), `metadata_json` zurückschreiben, commit.
   - sonst Job prüfen (job_id aus metadata): `job is None` (abgelaufen, keine Datei) → `failed` + error „Job nicht mehr auffindbar."; `job.is_failed` → `failed` + `error=(job.exc_info or '')[:500]`; sonst (queued/started) → bleibt `pending`.
3. **Tests** (`tests/test_narration_task.py`): `generate_narration_task` mit gemocktem `GoogleTTSService.synthesize_narration` (gibt Temp-WAV) → Datei landet unter `narration_<id>.wav` (OUTPUT_DIR gepatcht); creds fehlen → ValueError; Renderer wirft → propagiert. Reconcile (gemockte Conversion + gepatchte Datei-Existenz + gemockter Job): file-exists→ready+duration; job-failed→failed; job-None→failed; running→pending. `pytest` grün ≥ 610.

**Stop + Bericht.**

## Phase 2 — POST /api/narrations (Token-Gate + Validierung + pending-Conversion + Enqueue)

1. **`_authorize_narration_write()`** in `app_pkg/narration.py` (Spiegel von `_authorize_card_write`, aber `NARRATION_TOKEN`): `expected = os.environ.get('NARRATION_TOKEN')`; ohne → 503; `_bearer_token()` + `hmac.compare_digest` → 401; `_resolve_target_user()` → 503; sonst `(target, None)`. (`from app_pkg.ingest import _bearer_token, _resolve_target_user`.)
2. **`POST /api/narrations`** (Token, **CSRF-exempt**): Body `{title, language?, tts_model?, mode, voices, turns, style_prompt?}` (Turn-Kontrakt, s. Design-Doc).
   - `target, err = _authorize_narration_write(); if err: return err`.
   - JSON-Body-Validierung; `validate_turns(turns, voices, mode)` (ValueError → 400 mit der Message); `title` non-blank (sonst `derive_title`/Fallback); `tts_model` default `DEFAULT_NARRATION_MODEL`.
   - **`pending`-Conversion anlegen**: `Conversion(user_id=target.id, conversion_type='audio_narration', title=…, content=narration_to_markdown(turns), lifecycle_status='inbox')`; `db.session.add` + **`flush()`** (→ `conversion.id`); dann `metadata_json = json.dumps(build_narration_metadata(conversion.id, status='pending', tts_model=…, speakers=voices, transcript=turns, …))`; `commit()`.
   - **Enqueue**: `job = _app_module.task_queue.enqueue(generate_narration_task, conversion.id, turns, voices, style_prompt, mode, language_code, tts_model, meta={'user_id': target.id, 'conversion_id': conversion.id}, job_timeout=…)`. **job_id in die metadata** zurückschreiben (`metadata['job_id'] = job.id`; `metadata_json` update; commit) — der Reconcile braucht ihn.
   - **202** `{'narration_id': conversion.id, 'job_id': job.id, 'status': 'pending'}`.
   - `app.extensions['csrf'].exempt(api_create_narration)` am Ende von `register`.
3. **Tests** (`tests/test_narration_write.py`, Token-Scaffolding aus test_cards/test_tag_*): Auth-Matrix (kein `NARRATION_TOKEN`→503, falsch→401, fehlend→401, gut→202); bad-shape→400; `validate_turns`-Verstoß→400; **pending-Conversion angelegt** (Typ/content/metadata/owner=Target); **enqueue aufgerufen** (gemocktes `task_queue.enqueue` capturen: Task=`generate_narration_task`, args=conversion.id+turns+…, meta); **CSRF-exempt** unter erzwungenem CSRF beweisen (conftest-Caveat wie Ingest). `pytest` grün.

**Stop + Bericht.**

## Phase 3 — GET /api/narrations/<id> (Status + Reconcile) + Serve-Reconcile

1. **`GET /api/narrations/<int:conversion_id>`** (`@login_required`, owner-404): `conversion = get_owned_conversion(id)`; `conversion_type != 'audio_narration'` → 404; **`reconcile_narration(conversion)`** (flippt ready/failed wenn nötig); return `conversion.to_dict()` (enthält `metadata.narration_status`/`duration_seconds`/`error` etc.). Das ist der **Poll-Endpoint des Agenten/der UI**.
2. **Serve-Reconcile**: der NARR-2-Serve-Endpoint (`GET /api/narrations/<id>/audio`) ruft `reconcile_narration(conversion)` **vor** dem `narration_status == 'ready'`-Gate (damit ein `pending`-aber-Datei-da-Element direkt servt). Sonst unverändert.
3. **Tests** (in `tests/test_narration_serve.py` o.ä.): Status-Endpoint owner-404/type-404; `pending`+Datei → Read flippt auf `ready`+duration und gibt's zurück; `pending`+Job-failed → `failed`+error; `ready` bleibt `ready` (idempotent, kein Re-Reconcile); Serve nach Reconcile servt das frisch-ready-Element. `pytest` grün.

**Stop + Bericht.**

## Phase 4 — Wrap + .env-Notiz

1. **`docs/narration_reframe.md`** — NARR-3 ☑ (Token-Endpoint/Worker-DB-frei/Reconcile); den `NARRATION_TOKEN`-Env-Bedarf + die Option-B-Architektur final festhalten (falls von Phase 1–3 abgewichen).
2. **`CLAUDE.md`** — der **`NARRATION_TOKEN`** in die Service-Singleton-/Env-Doku (fail-closed wie `CARD_TOKEN`); der `audio_narration`-Flow (POST→pending→Worker→Reconcile) in die Architecture-Notes.
3. **STATUS.md** + **BACKLOG.md**: NARR-3 ☑ done (Hashes); Sequenz `1✅→1B✅→2✅→3✅→(4,5)`. **`NARRATION_TOKEN` als Olis Real-Welt-Schritt** notieren (Mac `.env` + Mintbox `.env` + converter-mcp-Config). **Bullet-Guard**.
4. **Memory** (`reference_*`): die Worker-DB-frei-+-Web-Reconcile-Architektur (Option B; warum: Worker mountet kein `app_data`) + der eigene `NARRATION_TOKEN` (billing-rationale) — wiederverwendbar. MEMORY.md-Pointer.
5. **Sprint-Doc** einchecken. Finaler `pytest`.

**Stop + Schluss-Bericht** — inkl. der **gesammelten Deploy-/Real-Welt-Kette** (jetzt alles bündelbar): Mintbox `git pull` + `docker compose up -d --build` (**Dep-Bump `google-cloud-texttospeech` aus NARR-1B** → Image rebuildt; kein Schema/Migration); **`NARRATION_TOKEN` in beide `.env` + converter-mcp**; GCP-Stack ist projektweit schon freigeschaltet (NARR-1B). **Danach End-to-End live möglich**: POST eine echte Turn-Liste → Audio in der Library.

## Bewusst NICHT (Scope-Grenze)

- **Kein** Mounten von `app_data` auf den Worker / kein Worker-DB-Zugriff (Option A verworfen).
- **Kein** Claude-Skill (NARR-4), **kein** UI-Player/Alt-Flow-Stilllegung (NARR-5).
- **`cards.py`/`_authorize_card_write` unberührt** (spiegeln, nicht refactoren); `generate_podcast_task` + Alt-Flow unberührt; der NARR-2-Serve-Endpoint nur um den Reconcile-Aufruf ergänzt.
- **Kein** Schema/Dep; einziger neuer Knopf = `NARRATION_TOKEN` (env, fail-closed).

## Akzeptanz

- [ ] `generate_narration_task` (DB-frei): Renderer → `narration_<id>.wav`; creds-fehlt→ValueError; Renderer-Wurf→propagiert.
- [ ] `reconcile_narration` (web): file-exists→ready+duration, job-failed/gone→failed, running→pending; idempotent für terminale States.
- [ ] `POST /api/narrations` (NARRATION_TOKEN, CSRF-exempt, fail-closed 503/401): validate_turns→400, pending-Conversion (audio_narration, content=Markdown, metadata+job_id, owner=Target), enqueue `generate_narration_task`, 202.
- [ ] `GET /api/narrations/<id>` (session, owner-404, type-404, Reconcile) → to_dict; Serve-Endpoint ruft Reconcile vor dem ready-Gate.
- [ ] Tests: Worker-Task + Reconcile + Auth-Matrix + Validierung + pending-Create + Enqueue-Capture + CSRF-exempt + Status/Serve-Reconcile. `pytest` grün ≥ 610.
- [ ] Docs/CLAUDE.md/STATUS/BACKLOG/Memory; `NARRATION_TOKEN` als Real-Welt-Schritt notiert. Kein Schema/Dep; cards.py + Alt-Flow unberührt.
