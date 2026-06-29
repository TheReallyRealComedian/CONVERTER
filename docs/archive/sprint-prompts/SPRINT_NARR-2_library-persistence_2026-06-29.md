# Sprint NARR-2 — Library-Persistenz für Vertonungen (audio_narration-Conversion) (M, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **580**). Du committest jede Phase selbst (Hash + push), **fokussiert**. Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. **Backend**, **kein Schema-Touch** (metadata_json ist die Escape-Hatch), **kein** neuer Dep/Token. Backend pytest-getestet.
>
> **Kontext**: NARRATION-Reframe, voller Entwurf [docs/narration_reframe.md](docs/narration_reframe.md). NARR-1B (Renderer) ☑. **Dieser Sprint baut die Persistenz-Schale** — er **generiert noch kein Audio** (das macht NARR-3s Worker). Tests erzeugen synthetische `audio_narration`-Conversions + Dummy-WAV-Dateien, um Serve + Delete-Cleanup zu prüfen.

## Warum & Entscheidungen (gesetzt)

Erzeugtes Vertonungs-Audio soll **erstklassiges Library-Element** werden (speicher-/abrufbar wie Docs), statt wie heute im `podcast_data`-Volume zu liegen + beim Download gelöscht zu werden. **Conversion wiederverwenden** (wie `audio_transcription`) — **kein Schema-Touch**:

- **`conversion_type='audio_narration'`** (zu `ALLOWED_CONVERSION_TYPES`).
- **`content`** = die Turn-Liste als **lesbares Markdown** (speaker-beschriftet) → die Vertonung ist ein normaler Reader-/Such-/Tag-/Highlight-fähiger Library-Citizen, `list_conversions`/`derive_title` greifen gratis. (`content` ist **NOT NULL** → nie leer.)
- **Audio-Felder in `metadata_json`** (Text/JSON-Escape-Hatch, **keine** neue Tabelle, **keine** Migration).
- **Audio-Datei**: persistenter, deterministischer Pfad **`OUTPUT_DIR/narration_<id>.wav`** (auf dem `podcast_data`-Volume, liegt schon auf **beiden** Containern — Worker schreibt ab NARR-3, Web liefert). **Nicht** löschen-beim-Serve.
- **Serve-Endpoint** session-authed (`@login_required`, owner-404, Traversal-Guard) in **neuem `app_pkg/narration.py`** (register-Pattern) — NARR-3 ergänzt dort den Token-POST.
- **Delete-Cleanup** in der bestehenden `api_delete_conversion`-Route (**post-commit** Unlink der Audio-Datei, traversal-guarded) — **nicht** als ORM-`before_delete`-Event: Conversions haben **keinen** Cascade-Lösch-Pfad (nur die Route löscht sie), und ein Unlink **nach** erfolgreichem Commit ist sicherer als mitten im Flush (kein gelöschte-Datei-aber-Row-rollback). (Der `Highlight`-`before_delete` ist das Event-Vorbild — hier bewusst die einfachere/sicherere Route-Variante.)

## Verifizierte Code-Fakten (Master-gegroundet)

- **`Conversion`** ([models.py:36](models.py)): `conversion_type`(≤30, indexed), `title`(≤255, **NOT NULL**), `content`(Text, **NOT NULL**), `source_filename`/`source_mimetype`/`source_size_bytes`, **`metadata_json`**(Text), `lifecycle_status`(default `'inbox'`), `created_at`. `highlights`-Relationship `cascade='all, delete-orphan'`. `to_dict()` ([models.py:74](models.py)) liefert `metadata = json.loads(metadata_json) if metadata_json else {}` + `content` + Typ etc.
- **`ALLOWED_CONVERSION_TYPES`** ([app_pkg/library.py:16](app_pkg/library.py)) = Set `{document_to_markdown, audio_transcription, dialogue_formatting, markdown_input, ai_newsletter}` → **`'audio_narration'` ergänzen**.
- **`api_delete_conversion`** ([app_pkg/library.py:543](app_pkg/library.py)): `conversion = get_owned_conversion(id)` → `db.session.delete(conversion)` → `commit()`. **`get_owned_conversion`** = der Owner-404-Helper (wiederverwenden; falls modul-level importierbar → importieren, sonst Owner-Check spiegeln).
- **Serve-Muster** `podcast_download` ([app_pkg/podcasts.py:314](app_pkg/podcasts.py)): Owner-Check, `real_path = os.path.realpath(file_path)`, **`Path(real_path).is_relative_to(Path(os.path.realpath(OUTPUT_DIR)))`** → sonst 403; `send_file(...)`. ⚠️ **podcast_download löscht nach dem Serve** (`os.unlink`) — der Narration-Serve **darf das NICHT** (persistent).
- **`OUTPUT_DIR = '/app/output_podcasts'`** ([app_pkg/config.py:10](app_pkg/config.py)), `podcast_data`-Volume auf beiden Containern (`docker-compose.yml:24+55`).
- **`Highlight` `before_delete`** ([models.py:353](models.py)) = das ORM-Event-Pattern (Core-Op auf der Flush-`connection`) — hier als Referenz, aber wir nehmen die Route-Variante.
- **Route-Modul-Konvention**: jedes `app_pkg/`-Modul exponiert `register(app)` (kein Blueprint, flache Endpoint-Namen); der App-Factory ruft sie. **`app_pkg/narration.py` neu** → in der Registrierungs-Liste ergänzen (file-level grounden, wo).

## Phase 1 — Persistenz-Helfer (Typ, metadata-Kontrakt, content-Markdown, Audio-Pfad) + Tests

1. **`ALLOWED_CONVERSION_TYPES`** um `'audio_narration'` ergänzen ([app_pkg/library.py](app_pkg/library.py)).
2. **Pures Helfer-Modul** (z.B. `services/narration_library.py` oder als Funktionen in `app_pkg/narration.py`-Modul-Ebene; pure, kein Flask-Context):
   - **`narration_to_markdown(turns) -> str`** — die Turn-Liste als lesbares, speaker-beschriftetes Markdown (z.B. `**Anna:** …\n\n**Ben:** …`; Single-Speaker = Absätze ggf. ohne Label). **Nicht-leer** (content ist NOT NULL); HTML/Markdown-Sonderzeichen im Text bleiben wörtlich (kein Mangling — das ist Lese-Content, der durch den geteilten Renderer läuft).
   - **`narration_audio_path(conversion_id) -> str`** = `os.path.join(OUTPUT_DIR, f'narration_{conversion_id}.wav')`.
   - **`build_narration_metadata(...)`** / eine dokumentierte Kontrakt-Konstante: das `metadata_json`-Schema:
     ```json
     {"narration_status": "pending|ready|failed",
      "audio_filename": "narration_<id>.wav", "audio_mimetype": "audio/wav",
      "duration_seconds": int|null, "tts_model": "gemini-2.5-flash-tts",
      "speakers": {"Anna":"Kore","Ben":"Puck"}, "transcript": [{"speaker","text"}],
      "error": null}
     ```
     Plus kleine Lese-Helfer (`narration_status(conversion)`, `narration_audio_filename(conversion)`), die `metadata_json` robust parsen (fehlend/kaputt → leere Defaults).
3. **Tests** (`tests/test_narration_library.py`, pure): `narration_to_markdown` (beide Speaker beschriftet, aller Text drin, nicht-leer, Single-Speaker-Form); `narration_audio_path` deterministisch; metadata-Lese-Helfer (ready/pending/failed, kaputtes JSON → Defaults). `pytest` grün ≥ 580.

**Stop + Bericht.**

## Phase 2 — Serve-Endpoint + Delete-Cleanup + Integrations-Tests

1. **`app_pkg/narration.py`** (neu, `register(app)`-Pattern) — **`GET /api/narrations/<int:conversion_id>/audio`** `@login_required`:
   - `conversion = get_owned_conversion(conversion_id)` (owner-404; fremd/fehlend → 404).
   - `conversion.conversion_type != 'audio_narration'` → **404** (kein Typ-Leak).
   - Status aus `metadata_json`: `!= 'ready'` → **404** „Audio nicht verfügbar." (pending/failed haben keine Datei; UI in NARR-5).
   - Pfad = `narration_audio_path(conversion_id)` (bzw. aus `metadata.audio_filename`, aber **immer** gegen OUTPUT_DIR auflösen). Datei fehlt → 404.
   - **Traversal-Guard** wie podcast_download (`os.path.realpath` + `Path(...).is_relative_to(Path(os.path.realpath(OUTPUT_DIR)))` → sonst 403).
   - **`send_file(real_path, mimetype='audio/wav', download_name=f'narration_{id}.wav')` — KEIN `os.unlink`** (persistent).
   - `app_pkg/narration.py` in der App-Factory-Registrierung ergänzen.
2. **Delete-Cleanup** in `api_delete_conversion` ([app_pkg/library.py:543](app_pkg/library.py)): wenn `conversion.conversion_type == 'audio_narration'` → den Audio-Pfad **vor** dem Delete merken; nach `db.session.commit()` (also nur bei erfolgreichem Row-Delete) die Datei **best-effort + traversal-guarded** unlinken (try/except + `is_relative_to(OUTPUT_DIR)`). Nicht-Narration-Conversions: unverändert.
3. **Tests** (`tests/test_narration_serve.py`): synthetische `audio_narration`-Conversion (Owner = Test-User) + **Dummy-WAV** unter `narration_<id>.wav` (OUTPUT_DIR im Test auf ein tmp-Dir patchen/monkeypatchen):
   - Serve: status=ready + Datei → **200** `audio/wav`, Bytes stimmen; **Datei existiert nach dem Serve noch** (kein Delete-on-serve).
   - owner-404 (fremde Conversion); wrong-type-404 (z.B. `markdown_input`); status≠ready → 404; fehlende Datei → 404; Traversal-Guard.
   - Delete-Cleanup: `DELETE /api/conversions/<id>` einer `audio_narration` → Row weg **und** Audio-Datei unlinkt; Delete einer Nicht-Narration rührt keine Audio-Datei an; Delete bei schon-fehlender Datei wirft nicht (best-effort).
   - `pytest` grün.

**Stop + Bericht.**

## Phase 3 — Wrap

1. **`docs/narration_reframe.md`** — NARR-2 ☑ (Persistenz-Modell/Serve/Cleanup); den metadata_json-Kontrakt + den Audio-Pfad final festhalten (falls von Phase 1 abgewichen).
2. **STATUS.md** + **BACKLOG.md**: NARR-2 ☑ done (Hashes); Sequenz `1✅→1B✅→2✅→3…`. **Bullet-Guard** (`grep -nE '(- \*\*.*){2,}' BACKLOG.md`).
3. **Sprint-Doc** (dieses File) einchecken. Finaler `pytest`. (Memory: nur falls eine nicht-offensichtliche Persistenz-Lehre auftaucht — sonst nicht nötig.)

**Stop + Schluss-Bericht** — inkl. Deploy-Notiz: kein Schema/Token/Dep → der spätere Mintbox-Deploy (bündelt sich mit NARR-1Bs Dep-Bump + NARR-3) ist `git pull` + `up -d --build`. Serve/Cleanup sind live, aber bis NARR-3 erzeugt nichts `audio_narration`-Conversions → am laufenden Verhalten ändert sich nichts.

## Bewusst NICHT (Scope-Grenze)

- **Keine** Generierung (kein Renderer-Call, kein Worker-Task, kein Token-Endpoint — NARR-3).
- **Kein** Schema-Touch (alles über `metadata_json` + bestehende Spalten), **kein** Dep/Token, **kein** UI (NARR-5).
- **`podcast_download`** + der Alt-Podcast-Flow + `OUTPUT_DIR`-Nutzung der Alt-Jobs **unberührt** (Narration nutzt denselben Volume, aber den `narration_*`-Namespace — kollidiert nicht mit `<job_id>.wav`).
- **Kein** ORM-`before_delete`-Event für die Audio-Datei (Route-Variante, s. Entscheidungen).

## Akzeptanz

- [ ] `'audio_narration'` in `ALLOWED_CONVERSION_TYPES`; `narration_to_markdown` (nicht-leer, speaker-beschriftet) + `narration_audio_path` (deterministisch) + metadata-Kontrakt/Lese-Helfer, pure + getestet.
- [ ] `GET /api/narrations/<id>/audio` (`@login_required`, owner-404, type-404, status≠ready-404, Traversal-Guard, **kein** Delete-on-serve) in `app_pkg/narration.py`, registriert.
- [ ] Delete-Cleanup in `api_delete_conversion` (post-commit, best-effort, traversal-guarded; nur `audio_narration`).
- [ ] Tests: pure Helfer + Serve (200/owner-404/type-404/not-ready/missing/traversal/no-delete) + Delete-Cleanup. `pytest` grün ≥ 580.
- [ ] Kein Schema/Dep/Token/UI; `podcast_download` + Alt-Flow unberührt.
