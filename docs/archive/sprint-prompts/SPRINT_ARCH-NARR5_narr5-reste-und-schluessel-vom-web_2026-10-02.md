# SPRINT ARCH-NARR5 — NARR-5-Reste entfernen, den GCP-Schlüssel vom Web-Container nehmen

**Größe**: S + S (3 Phasen: Code · zwei Deploys mit Belegen · Wrap) · **Datum**: 2026-10-02 · **Herkunft**: Befund W-1 und Urteil E-6 aus [ARCH-AUDIT](../audit-outputs/AUDIT_ARCHITEKTUR_2026-10-01.md), Rang 2 der Priorisierung; BACKLOG-Item ARCH-NARR5

## Warum

NARR-5 (2026-06-30) hat den Alt-Podcast-Flow stillgelegt — die **Verwender** — und die **Anbieter** stehen lassen. Drei Monate später kostet das dreierlei:

1. **Ein Geheimnis an der Internet-Kante ohne Leser.** `app.py` Z. 61 baut beim Import `GoogleTTSService(GOOGLE_CREDENTIALS_PATH)` — einen gRPC-TTS-Client in jedem der zwei Web-Prozesse. Keine Route liest das Objekt (Narrationen rendert der Worker, mit einer eigenen Instanz je Auftrag, [tasks.py](../../../tasks.py)). Der GCP-Schlüssel-Bind am aus dem Internet erreichbaren Web-Container ([docker-compose.yml](../../../docker-compose.yml) Z. 45) existiert **nur dafür**; ohne die Datei startet der Web-Prozess nicht.
2. **Ein schwebendes SDK im lebenden Pfad.** Die zwei lebenden Helfer (`concatenate_with_pydub`/`concatenate_with_wave`, `is_pydub_available`) liegen im Paket `services/gemini/`, dessen `__init__` `google.genai` importiert. Jeder Narrations-, Dokument- und Transkriptions-Auftrag zahlt das SDK beim `import tasks`, ohne es dort zu nutzen; ein genai-Importfehler nähme den Renderer mit.
3. **Eine dokumentierte Entscheidung, deren Ausstiegsklausel erfüllt ist.** CLAUDE.md führt `GeminiService` + `gemini_service` als „bewusst stehen gelassen (Seam für ein künftiges Gemini-Text-Feature)". Das Feature kam (Cloud-PDF) und baut seinen Client selbst. **Oli hat die Entfernung am 2026-10-02 freigegeben** („ja, GeminiService raus").

## Gegroundeter Ist-Zustand (Master, gemessen 2026-10-02)

- **Shim** [app.py](../../../app.py): Z. 45 `from services import DeepgramService, GeminiService, GoogleTTSService` · Z. 49 `GEMINI_API_KEY` · Z. 50 `GOOGLE_CREDENTIALS_PATH` · Z. 60 `gemini_service = …` · Z. 61 `google_tts_service = …` · Z. 103–104 `if __name__ == '__main__': app.run(…, debug=True)`. Über den Seam gelesen werden nur `task_queue` (4), `fetch_job` (3), `deepgram_service`, `async_playwright`, `DEEPGRAM_API_KEY` (je 1) — `gemini_service`, `google_tts_service`, `GEMINI_API_KEY`: **0 Leser** (selbst nachgezählt).
- **Decorator** [app_pkg/decorators.py](../../../app_pkg/decorators.py): Maps mit drei Schlüsseln (`deepgram`, `google_tts`, `gemini`); `require_service(` hat genau zwei Aufrufe, beide `'deepgram'` ([app_pkg/audio.py](../../../app_pkg/audio.py)).
- **Paket `services/gemini/`** (199 LOC): `__init__.py` (Klasse `GeminiService`, importiert `client`) · `client.py` (`from google import genai`, `create_client`, **`is_pydub_available`**) · `audio.py` (die zwei WAV-Concat-Funktionen, nur Stdlib + pydub) · `voices.py` (Stimmen-Katalog, 49 Zeilen, kein Code-Konsument). Importeure: [services/narration_render.py](../../../services/narration_render.py) Z. 38 (`services.gemini.audio`), [services/google_tts_service.py](../../../services/google_tts_service.py) Z. 6 (`is_pydub_available`), `services/__init__.py` Z. 17 (`_LAZY['GeminiService']`), `app.py`. Der Stimmen-Katalog steht zusätzlich in [docs/narration_skill.md](../../narration_skill.md), [docs/converter_mcp_narration_brief.md](../../converter_mcp_narration_brief.md) und [docs/narration_tag_doctrine.md](../../narration_tag_doctrine.md).
- **`GoogleTTSService`**: `__init__` baut `texttospeech.TextToSpeechClient()` über die Env-Variable; `list_voices` und `synthesize_speech` (Z. 21–99) haben keinen Aufrufer; **`synthesize_narration` lebt** (Worker, `tasks.generate_narration_task`; ein Test ruft sie über `__new__`).
- **Fixtures ohne Nutzer** in [tests/conftest.py](../../../tests/conftest.py): `mock_gemini`, `mock_google_tts`, `gemini_api_key_set`.
- **Compose**: Schlüssel-Bind `:ro` an Web (Z. 45) und Worker (Z. 87); `GOOGLE_APPLICATION_CREDENTIALS=${…}` an beiden (Z. 59, 100) — am Web zusätzlich über `env_file: .env`. Sentinel `test_credentials_bind_is_read_only` ([tests/test_compose_socket.py](../../../tests/test_compose_socket.py)) verlangt heute den Bind an **beiden**.
- **Auf dem Pin gemessen** (Wegwerf-Container aus `converter-app:latest` = `1e6dee70a12e`, ohne Volumes, ohne Schlüssel in der Env): `import app` lädt **1 576 Module in 0,88 s, maxrss 131 MB**; `google.genai`, `google.cloud.texttospeech`, `grpc`, `deepgram` sind danach geladen. Web-Prozesse in Prod: 186 MB und 340 MB RSS. Mounts am Web: `/app/data`, `/app/output_podcasts`, `/app/google-credentials.json` (RW=false).
- **`google-genai` bleibt eine Abhängigkeit**: [services/pdf_cloud.py](../../../services/pdf_cloud.py) und [scripts/probe_configured_models.py](../../../scripts/probe_configured_models.py) bauen ihren Client selbst.
- **Baseline**: 1389 passed + 1 skipped, Mac und Pin. Prod-Image `1e6dee70a12e`, Rollback-Tag `pre-job-id-reuse` (`8bf21a3e2f35`, darf mit seinem Image weg). Volume sauber, 0 `pending`.

## Gesperrte Entscheidungen

1. **Reihenfolge ist zwingend: erst der Code in Prod, dann der Bind.** Wird der Bind gestrichen, solange der alte Code läuft, stirbt der Web-Container beim Boot (`DefaultCredentialsError`). Deshalb zwei Deploys. Folge für den Rückweg: nach Deploy 2 ist ein Rollback auf das alte Image **nur zusammen mit der alten Compose-Datei** möglich — im Bericht und im STATUS so benennen.
2. **Was fällt:** in `app.py` die Importe `GeminiService`/`GoogleTTSService`, `GEMINI_API_KEY`, `GOOGLE_CREDENTIALS_PATH`, beide Singletons und der `__main__`-Zweig (bootstrapt doppelt, `debug=True`); in `decorators.py` die Schlüssel `google_tts` und `gemini`; `GoogleTTSService.list_voices` und `.synthesize_speech`; das Paket `services/gemini/` **ganz** (Klasse, `create_client`, `voices.py`); der `_LAZY`-Eintrag `GeminiService`; die drei toten Fixtures.
3. **Was bleibt, unverändert:** `GoogleTTSService.__init__` und `synthesize_narration`, der ganze Renderer, `require_service` als Mechanik mit dem einen Schlüssel `deepgram` (kein Umbau des Decorators), die Abhängigkeit `google-genai`, die Env-Zeilen `GEMINI_API_KEY`/`DEEPGRAM_API_KEY`, der Schlüssel-Bind am **Worker**.
4. **Die lebenden Helfer ziehen wörtlich um** nach `services/wav_concat.py`: `concatenate_with_pydub`, `concatenate_with_wave`, `is_pydub_available` — Funktionskörper byte-gleich (per `git mv` der `audio.py`, damit die Historie mitgeht), nur der Modul-Docstring und die Importzeilen der zwei Aufrufer ändern sich. Kein Verhalten, kein Log-Text.
5. **Stimmen-Katalog:** vor dem Löschen von `voices.py` belegen, dass jede Stimme daraus in [docs/narration_skill.md](../../narration_skill.md) steht (Mengenvergleich im Bericht). Fehlt eine, wandert der Katalog im Wrap vollständig in dieses Doc — nichts geht verloren.
6. **Sentinels, die den Gewinn festhalten** (Vorbild `test_launcher_import_surface_is_minimal`, Subprozess): nach `import services.narration_render`, `import services.google_tts_service` und `import tasks` ist `google.genai` **nicht** geladen; `app` hat die Attribute `gemini_service`/`google_tts_service` nicht mehr. Wenn sich `import app` in einem sauberen Subprozess billig aufbauen lässt, auch dafür „kein `google.genai`"; wenn nicht, im Bericht sagen, warum.
7. **Compose (erst in Phase 2, eigener Commit):** der Bind am Web fällt, Z. 59 fällt mit (wirkungslos, `env_file` liefert den Namen weiter — das bleibt so, kein Überschreiben auf leer); der Sentinel verlangt danach den Bind **nur am Worker** und verbietet ihn überall sonst.
8. **Nicht in diesem Sprint:** der Singleton-Seam selbst (E-2 hält mit Auflage), `task_queue`/`fetch_job`, das Job-Gerüst, `import tasks` im Web auf einen String-Verweis umstellen, Factory, irgendein Build-Thema (ARCH-BUILD), Abhängigkeiten heben.

**Arbeitsweise:** inline, **kein Workflow, keine Subagenten** ohne Olis ausdrückliches Wort. Commit + Push je Phase, dann Stop + Bericht. Nichts Tragendes im Session-Scratch. Editiert wird nur auf dem Mac.

---

# Phase 1 — Code (zwei Commits, kein Deploy)

**Commit A — `google_tts`-Teil und der Umzug der Helfer.** `services/wav_concat.py` anlegen (Entscheidung 4), die zwei Aufrufer umhängen; `google_tts_service` samt Import und `GOOGLE_CREDENTIALS_PATH` aus `app.py`; Schlüssel `google_tts` aus beiden Decorator-Maps; die zwei toten Methoden; Fixture `mock_google_tts`; der `__main__`-Zweig; der Kopf-Docstring von `app.py` auf den Ist-Stand (er nennt „blueprints" und Singletons, die es dann nicht mehr gibt).

**Commit B — der `gemini`-Teil (von Oli freigegeben).** Stimmen-Vergleich (Entscheidung 5), dann `services/gemini/` löschen, `_LAZY`-Eintrag, `gemini_service` und `GEMINI_API_KEY` aus `app.py`, Schlüssel `gemini` aus den Maps, Fixtures `mock_gemini` und `gemini_api_key_set`; Docstrings, die `gemini_service` als Beispiel nennen (`app_pkg/__init__.py` Z. 8, `app_pkg/decorators.py`, `tests/conftest.py` Kopf), auf den Ist-Stand.

**Vor jedem Löschen: der Leser-Nachweis über alle Kanäle** (absolute und relative Importe, `_LAZY` und andere String-Maps, Templates und `static/js/`, `scripts/` und `tests/`) — mit **Positivkontrolle**: ein bekannt genutzter Name (`deepgram_service`) muss als genutzt zählen. Die Zahlen in den Bericht.

**Sentinels** nach Entscheidung 6, in beiden Commits jeweils der Teil, der dann gilt.

**Gates:** `python3 -m pytest tests/ -q -p no:cacheprovider` am Mac (Baseline 1389 + 1 Skip, Abweichung benennen); **Container-Suite auf dem Pin** (stdin-Rezept aus CLAUDE.md, `COPYFILE_DISABLE=1 tar --no-xattrs`); `test_launcher_import_surface_is_minimal` grün; der Compose-Sentinel bleibt in dieser Phase **unverändert grün** (der Bind steht noch).

## Stop
Bericht: Leser-Nachweis mit Positivkontrolle · Stimmen-Vergleich · Diff je Datei · der Beleg „Funktionskörper byte-gleich" für den Umzug · die neuen Sentinels und was sie messen · Suite Mac und Container · Abweichungen vom Prompt. Dann warten.

---

# Phase 2 — Zwei Deploys mit Belegen

## Deploy 1 — der Code

1. **Fenster:** Queue leer, Worker idle, kein `mineru_*`, 0 `pending` (DB `mode=ro`), die drei Job-Verzeichnisse leer. **DB-Backup** nach dem Rezept in CLAUDE.md (die Probe-Narrationen schreiben auf Olis Konto), Kopie mit `mode=ro&immutable=1` prüfen. Rollback-Tag `converter-app:pre-arch-narr5`; `pre-job-id-reuse` darf samt Image weg. **Nie `docker image prune -a`.**
2. **Vorher messen:** RSS der Web-Prozesse (`ps` im Container); im Wegwerf-Container aus dem **alten** Image (`--network none`, ohne Volumes, ohne Schlüssel): Module, Sekunden und maxrss nach `import app`, dazu ob `google.genai` geladen ist (Master-Vorwert: 1 576 · 0,88 s · 131 MB · ja).
3. `git pull --ff-only`, `docker compose up -d --build` aus dem Projektverzeichnis; `docker inspect`: was neu angelegt wurde; Login 200.
4. **Nachher messen:** dieselbe Wegwerf-Messung am **neuen** Image; RSS der Web-Prozesse nach dem Start.
5. **Eine Narration Ende-zu-Ende, die den umgezogenen Code wirklich fährt:** der WAV-Concat läuft nur bei **mehreren Chunks** — der Text muss also über der Chunk-Grenze des Renderers liegen (Grenze aus `services/narration_render.py` lesen, im Bericht nennen). Über `POST /api/narrations` mit dem Token aus der Container-Env (**nie ausgeben**), pollen bis `ready`; belegen: im Worker-Log lief die Konkatenation (welcher Zweig: pydub oder wave), die Dauer ist plausibel für den Text, `narration_jobs/` ist danach leer. Die Zeile **strikt nach ihrer id** samt WAV entfernen.
6. `docker exec markdown-converter-worker python scripts/probe_configured_models.py` → exit 0 (der genai-Client des Cloud-PDF-Pfads lebt unabhängig vom gelöschten Paket).

## Deploy 2 — der Bind

7. **Commit C** (Mac): Compose nach Entscheidung 7 und der Sentinel auf „nur Worker"; Suite am Mac grün; Push.
8. Vor dem Deploy am laufenden Web-Container belegen, dass kein Prozess die Schlüsseldatei offen hält (`/proc/<pid>/fd` der gunicorn-Prozesse).
9. `git pull --ff-only`, `docker compose up -d` (**ohne** `--build`); `docker inspect`: nur der Web-Container ist neu, Worker, Launcher und Redis unverändert.
10. **Belege:** `docker exec markdown-converter-web ls /app/google-credentials.json` → *No such file*; Mounts am Web ohne den Schlüssel; am Worker weiter `RW=false`; Login 200 lokal und über die Kante; ein `flask --app app …`-Lesekommando im Web-Container läuft (die CLI bootet ohne Schlüssel); **eine kurze Narration Ende-zu-Ende** (ein Satz) inklusive `GET /api/narrations/<id>/audio` → 200 — sie beweist Submit, Adoption und Auslieferung durch einen Web-Prozess ohne Schlüssel; danach strikt nach id entfernen.
11. **Nachher-Inventar:** Volume wie vorher (nur Olis Narrationen), DB wie vorher (Zeilenzahl, `max(id)`, 0 `pending`), keine Wegwerf-Reste, Mintbox-Checkout sauber.

Sicherheits-Regeln wie immer: `docker exec` nie mit `-u 0`; DB-Lesen `mode=ro`; keine Token, Schlüssel oder Cookie-Werte in Ausgaben; keine unversionierte Datei auf der Mintbox.

## Stop
Bericht: Fenster · Deploy 1 mit Vorher/Nachher-Tabelle (Module, Sekunden, maxrss, `google.genai` geladen, RSS der Web-Prozesse) · die Mehr-Chunk-Narration mit Log-Beleg · Modell-Probe · Deploy 2 mit den Belegen aus 10 · Inventar · der benannte Rückweg (Image **und** Compose-Datei).

---

# Phase 3 — Wrap

- **CLAUDE.md** — hier fällt eine dokumentierte Entscheidung, die Stellen einzeln nachziehen, je ein bis zwei Sätze: *Key Files* (`services/gemini/` → `services/wav_concat.py`; der „dormant"-Satz weg), *Gemini Models* („Entfernt (NARR-5)": `GeminiService`/`gemini_service` sind seit ARCH-NARR5 entfernt, mit dem Grund), *Service-singleton pattern* (die Liste der Singletons auf den Ist-Stand), *Build-Kontext* und der SEC-NONROOT-Absatz (der Schlüssel liegt nur noch am Worker; der Web-Container hat keinen Host-Pfad mehr), Test-Baseline. Die gemessene Vorher/Nachher-Tabelle gehört in den STATUS-Eintrag, in CLAUDE.md nur das Ergebnis in einem Satz.
- **STATUS.md**, **BACKLOG.md** (ARCH-NARR5 schließen; ⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber).
- **Kein Brief ans converter-mcp** — keine Agent-Fläche berührt (die Narrations-Fläche bleibt in Form und Verhalten); im Wrap so sagen.
- **Memory** nur bei übertragbarer Lehre; `reference_flow_retirement_shared_package` trägt seit ARCH-AUDIT schon „Verwender entfernt, Anbieter geblieben" — prüfen, ob ein Satz „ein Geheimnis-Mount folgt seinem letzten Leser" dort hineingehört, kein neuer Eintrag ohne Not.
- Commit + Push, Stop + Bericht.

## Nicht-Ziele

Kein Umbau des Seams, kein Job-Gerüst, kein Eingriff in den Renderer oder in `synthesize_narration`, keine Abhängigkeit gehoben oder entfernt, kein Build-Thema, kein Frontend. Nichts am Worker-Bind.
