# SPRINT ARCH-AUDIT — Architektur- & Wucherungs-Audit (Karte · Hot-Spots · Verstöße · Wucherung · Tech-Debt)

**Größe**: M (2 Phasen; **beide ohne Code-Änderung** — Phase 1 ist lesend und messend, Phase 2 schneidet Items und schreibt Docs) · **Datum**: 2026-10-01 · **Vorhaben**: Code-Check-Reihe, Audit 2 von 5 (Notion-Katalog „Audit — Architektur- & Wucherungs-Audit", Projekt MINTBOX, 2026-05-16; Audit 1 = SEC-AUDIT 2026-09-25)

## Warum

Oli, 2026-09-25: *„ich würde gerne mal wieder einen code-check über die applikation laufen lassen"*, Reihung nach Security: **ARCH → CONSIST → TEST → DOC**. Der Security-Zyklus ist seit 2026-10-01 geschlossen (F-1…F-13, F-6-SSRF, Hygiene, Schlüssel-Rotation). Die letzte Architektur-Reflektion war die **Cleanup-Welle im Mai** (Stages 0–7, geschlossen 2026-05-11, [docs/cleanup_plan.md](../../cleanup_plan.md), [docs/inventory_2026-05.md](../../inventory_2026-05.md)). Seitdem: **440 Commits, 86 Sprints**, die Suite von 71 auf 1341 Tests, und drei Re-Architektur-Wellen in derselben Region (Dokument-Dienst DOC-API/ENGINE/LOCAL/WEB/WEB-ASYNC, Nebenläufigkeit SYNC-FREEZE/LOST-UPDATE, Reader LESEMODUS/READER-*/RICH-MEDIA) plus der Security-Zyklus mit vier Compose-/Dockerfile-Umbauten. Die Notion-Vorlage nennt genau diese Trigger: *„nach mehreren Sprints in derselben Region"*, *„nach Re-Architektur-Wellen um zu sehen, was nachgewuchert ist"*, *„periodisch alle ~5 Sprints"* — wir sind bei 86.

Dieser Sprint ist **nicht** „ist der Code gut", sondern: **wo ist die Summe vieler isoliert richtiger Sprints unwartbar geworden, wo ist eine dokumentierte Entscheidung inzwischen teurer als ihre Alternative, und was davon lohnt einen Cleanup-Sprint.** CLAUDE.md dokumentiert viele Entscheidungen ausdrücklich (kein Blueprint, Singletons im Shim, DB-freier Worker, ein Stylesheet, lazy Engine-Imports) — das Audit **unterscheidet Entscheidung von Drift** und urteilt über die Entscheidung mit gemessenen Kosten, nicht mit Geschmack.

## Gegroundeter Ist-Zustand (Master — gemessen 2026-10-01 am Mac-Clone, nicht neu herleiten; Abweichungen im Bericht benennen)

**Größe.** Python **14 005 LOC** in 61 Modulen: `app_pkg/` 25 (Routen je Feature + Factory + Config + Renderer + ASGI-Adapter), `services/` 32 (Engines, Tore, Scheduler, Launcher), Root `app.py` (104) · `tasks.py` (240) · `worker.py` (29) · `models.py` (541, acht Klassen: `User`, `ApiToken`, `Conversion`, `Highlight`, `Tag`, `Card`, `Review`, `Collection`). Frontend: 13 JS-Dateien **5 972 LOC** (`library_detail.js` 1 830, `review.js` 931, `audio_converter.js` 827, `markdown_converter.js` 519), `style.css` **2 952 LOC** (ein Stylesheet, bewusst), 12 Templates. Tests: 74 Dateien, 1 145 Testfunktionen, **1341 + 1 Skip** gesammelt. `scripts/` 15 (5 Smokes, 2 Renderer-Gates, 4 Nebenläufigkeits-Messungen, 1 Modell-Probe, 3 Einmal-Backfills/Cleanups). `docs/` 64 Dateien + 109 Sprint-Prompts im Archiv. `corpus/` trägt **462 getrackte Dateien** (Bake-off-Harness, 11 `.py`, Ergebnisse); das Verzeichnis ist per `.dockerignore` draußen. 660 Commits gesamt.

**Routen.** **75** `@app.route` in **19** Modulen mit `register(app)`: library 12 · cards 11 · tags 9 · collections 6 · audio/document_api/highlights/learn/narration je 4 · mobile_auth 3 · auth/documents/docwrite/notion/markdown je 2 · `__init__`/ingest/kindle/mermaid je 1. **Kein Blueprint — Entscheidung** (CLAUDE.md *Routing pattern*: Endpoint-Namen flach, Templates nutzen `url_for("login")`). Folge, gemessen: jede View ist eine **Closure in `register()`**, und `register` ist in jedem Routen-Modul die längste Funktion — **library 551 Zeilen, cards 494, tags 385, narration 226, document_api 190, audio 163, collections 144, markdown 141**. Helfer außerhalb der Closure existieren (library 18 Funktionen, cards 27, learn 20), aber die Views selbst sind weder importierbar noch einzeln adressierbar.

**Schichten und Importe** (AST, Top-Level und in Funktionen; Skript in Anhang B):
- **`services` → `app_pkg`**: sieben Service-Module importieren `app_pkg.config` (`deepgram_service`, `document_conversions`, `gemini.client`, `narration_library`, `narration_render`, `pdf_cloud`, `transcription_jobs`). Umgekehrt liest `app_pkg.config` die mineru-Frist aus `services.mineru_invocation` (laut Docstring in `services/__init__.py`) — **ein Zyklus zwischen App-Config und Service-Schicht**, der nur hält, weil `config.py` SDK-frei bleibt (Sentinel `test_launcher_import_surface_is_minimal`).
- **Routen-Module → `app`**: acht Stellen `import app as _app_module` **in Funktionen** (audio 165/221, narration 151/232, decorators 44, markdown 141, document_api 349/412), weil `app.py` die Service-Singletons hält (`deepgram_service`, `gemini_service`, `google_tts_service`, `task_queue`, `Job`, `async_playwright`, `redis_conn`, `fetch_job`) und die Tests sie **am Root-Modul** patchen (`app.py`-Docstring: *„the blueprints look these up via import app as _app_module"* — das Wort „blueprints" steht dort, obwohl es keine gibt). Das ist der **Singleton-Seam**: Root-Modul ↔ Feature-Module zirkulär, per späten Import aufgelöst. `app.py` (104 Zeilen) wurde seit Mai **28-mal** geändert.
- **Lokale Importe in Funktionen** gesamt 28: dazu `app_pkg/__init__` 5 (`mobile_auth`, `models`, `cards._naive_utc` — in CLAUDE.md als „top-level wäre zirkulär" dokumentiert), `services.document_router` 9 (**Entscheidung**: Engines lazy, DOC-WEB) und `tasks` 4 (DB-freier Worker, lazy).
- **Cross-Feature-Helfer**: `app_pkg.ingest` liefert `_bearer_token`/`_resolve_target_user` an narration und document_api; `cards._naive_utc` an die Factory; `learn.local_day_bounds`/`LOCAL_TZ` (seit NOTION-TZ in `config`) an library und notion. Muster: Helfer entstehen im Feature, in dem sie zuerst gebraucht wurden, und werden dann quer importiert.
- 16 relative Importe (`from .x`), der Rest absolut. **Keine** `TODO/FIXME/XXX/HACK`-Marker, 2 `noqa`, 0 `type: ignore`.

**Gespiegelte Implementierungen** (gezählt): drei `_authorize_*` (`cards._authorize_card_write`, `narration._authorize_narration_write`, `document_api._authorize_document_access` — CLAUDE.md nennt die Spiegelung ausdrücklich: *„gespiegelt (cards.py unberührt)"*), drei `reconcile_*` (`audio.reconcile_transcription`, `narration.reconcile_narration`, `document_api.reconcile_document_conversion` — dreimal Option B: Worker schreibt Datei, Web rekonziliert file-first), die Status-Literale `'pending'`/`'ready'`/`'failed'` an **16** Stellen als String, `secure_filename` an 7 Stellen bei 9 `request.files`-Zugriffen.

**Hot-Spots nach Churn** (Dateien in Commits seit 2026-05-01, Code/Templates/CSS): `style.css` **62** · `library.py` 30 · `app.py` 28 · `library_detail.html` 27 · `library_detail.js` 26 · `app_pkg/__init__.py` 24 · `library.html` 20 · `models.py` 19 · `cards.py` 19 · `config.py` 18 · `corpus/bakeoff/harness/adapters.py` 17 · `review.js` 15 · `review.html` 14 · `tasks.py` 13 · `document_converter.html` 12. Die Library/Reader-Region (`library.py` + `library_detail.*` + `library.html`) ist der Magnet: LESEMODUS, READER-SCROLLBAR, READER-STIL, RICH-MEDIA, LOST-UPDATE, R2-* landeten alle dort. ⚠️ Die Autoren-Dimension der Vorlage („viele Autoren") ist **nicht anwendbar** (ein Mensch, ein Assistent) — im Bericht so sagen.

**Zwei Sammel-Module.** `app_pkg/__init__.py` (**743 LOC**, 31 Funktionen) trägt Factory, Cookie-Interface, `request_loader`, `unauthorized_handler`, SQLite-Pragmas, Startup-Lock, CSRF-Inversion, **zwei Migrationsfunktionen**, Error-Handler, Security-Header, CSRF-Endpoint, drei Template-Filter und **drei CLI-Kommandos** (`_register_cli_commands` 191 Zeilen: `create-user`, `set-password`, `reset-collection`). `app_pkg/config.py` (283 LOC, **29 Konstanten**, 18 Änderungen) hält Zeit-Umschläge für vier Teilsysteme (Gemini, Deepgram/Audio, RQ, Dokument-Jobs), Budget-Sätze, SQLite-Timeout, Thread-Zahl, `LOCAL_TZ`, `OUTPUT_DIR`, `RQ_SERIALIZER`.

**Dokumentierte Entscheidungen, die kein Verstoß sind, sondern zu beurteilen** (Quelle CLAUDE.md; das Audit urteilt *hält / hält mit Auflage / hält nicht* mit Kostenargument): (1) kein Blueprint, Closures in `register()`; (2) Service-Singletons in `app.py` als Test-Patch-Punkt, `import app` in Funktionen; (3) Option B dreifach (DB-freier Worker, Web rekonziliert file-first) mit bewusst gespiegelten Auth-Helfern; (4) **ein** `style.css` mit TOC; (5) Engines lazy im Router, SDK-Klassen lazy in `services/__init__` (PEP 562); (6) `GeminiService` dormant als Seam; (7) `constraints.txt`-Freeze (DEPS-FLOAT existiert); (8) `OUTPUT_DIR`/Volume `podcast_data`/`output_podcasts` als Namensraum für Narration **und** Dokument-Jobs **und** Transkriptionen (Namen aus der Podcast-Ära); (9) `corpus/` im Repo; (10) `scripts/` als Ort für Smokes, Gates, Messungen und Einmal-Backfills nebeneinander.

**Schon im Backlog, nicht neu erfinden**: DEPS-FLOAT (M), CSP-BASELINE (M, Play-CDN), RICH-MEDIA-ASSETS, DOC-SPAN-MERGE, PDF-KNOPF-STUCK, PDF-LAZY-IMG, PREVIEW-PAPER-SYNC, DARK-THUMB-CONTRAST, READER-WINDOW-SCROLL, SEC-LOGOUT-POST, SEC-TOKEN-EXPIRY, RICH-MEDIA-IOS, LEARN-SKIP-IOS, MESOMERIE-V2, SEC-MAILBOX-PW, DOC-SVC (XL, extern).

⚠️ **Messfalle Dead Code (Master selbst hineingelaufen):** ein naiver Importeur-Grep meldete `app_pkg.markdown_render`, `app_pkg.pdf_egress`, `services.audio_chunker`, `services.google_tts_service`, `services.gemini.voices`, `services.scheduler.sm2_scheduler` als „ohne Importeur" — alle sechs sind live. Gründe: relative Importe (`from .markdown_render import …`), der PEP-562-Lader in `services/__init__.py` (String-Namen), `url_for('…')` in Templates und JS, Patch-Pfade in Tests. **Jeder Dead-Code-Befund braucht den Nachweis über alle vier Kanäle** (absolute + relative Importe, `_LAZY`-Tabelle, `url_for`/Endpoint-Namen in `templates/` und `static/js/`, `scripts/`), sonst ist er kein Befund.

## Schmerz-Hypothesen des Masters (zu prüfen, nicht zu bestätigen)

1. Die `register()`-Closures sind die teuerste Einzelentscheidung: 500-Zeilen-Funktionen, Views nicht adressierbar, jeder Sprint in library/cards editiert mitten in einer Closure. Prüfen, ob es eine Mechanik gibt, die **Endpoint-Namen flach hält** und trotzdem Modul-Funktionen erlaubt (`app.add_url_rule` mit `endpoint=`, Blueprint ohne Präfix mit gleichnamigen Endpoints — `url_for` kennt `bp.name`, Templates müssten nicht zwingend umgeschrieben werden, wenn der Blueprint-Name leer bleibt oder ein `endpoint`-Alias gesetzt wird; **messen am Test-Client, nicht vermuten**).
2. Der Singleton-Seam in `app.py` ist eine Test-Entscheidung aus Stage 6 (Mai), die seither 28 Shim-Änderungen und acht späte Importe gekostet hat; `grep -rn "patch('app\." tests/` zeigt, wie viele Tests daran hängen.
3. Die Option-B-Dreifaltigkeit (drei Reconciles, drei Auth-Helfer, 16 Status-Literale, dreimal atomare `result_<id>.json`) ist Parallel-Implementierung eines Konzepts; die Spiegelung war gewollt (unabhängige Revozierbarkeit der Tokens), aber die **Job-Mechanik** (enqueue, Umschlag, Datei, Reconcile) ist nicht token-spezifisch.
4. `app_pkg/__init__.py` ist ein God-File mit mindestens drei Extraktionen zu S (`migrations.py`, `cli.py`, `security.py`), jede mit bestehenden Sentinels gedeckt.
5. Die Library/Reader-Region trägt vier Verantwortungen in einem Modul und einer JS-Datei (Liste, Detail/Reader, Fortschritt, Platzieren/Tags) — das erklärt den Churn besser als „viel Feature".
6. Namens-Altlasten aus der Podcast-Ära (`podcast_data`, `output_podcasts`, `OUTPUT_DIR`, „ConversionHistory" im CLAUDE.md-Key-Files-Abschnitt gegen `Conversion` im Code, „blueprints" im `app.py`-Docstring) sind **Konsistenz**, nicht Architektur — hier nur als Liste für CONSIST-/DOC-AUDIT sammeln.

## Gesperrte Entscheidungen

1. **Kein Code-Edit in beiden Phasen.** Kein Refactor, kein „kleiner Fix nebenbei", keine Dependency-Änderung, kein Löschen von Dead Code. Das Audit liefert Befund und Items; der Umbau ist je Item ein eigener Sprint mit eigenem Gate. Erlaubte Schreibzugriffe: das Befund-Doc, BACKLOG/STATUS/CLAUDE.md im Wrap, ggf. Memory.
2. **Messen, nicht behaupten.** Jeder Hot-Spot trägt die Zahl (Churn, LOC, Funktionslänge), jeder Verstoß die Fundstelle mit Zeile, jede Parallel-Implementierung den Diff der Zwillinge. Kommandos und Ausgaben ins Befund-Doc (Anhang C), damit TEST-AUDIT und ein zweiter ARCH-Lauf in sechs Monaten vergleichbar sind.
3. **Befund-Format = Notion-Vorlage wörtlich** (Anhang A): Karte (Mermaid) · Hot-Spots-Tabelle mit Hot-Score · Architektur-Verstöße 1–4 · Wucherungs-Findings · Tech-Debt-Priorisierung · Strategische Empfehlung (2–4 Sätze) · Validierungs-Checkliste beantwortet. **Plus eine eigene Sektion „Dokumentierte Entscheidungen"** (die zehn aus dem Ist-Zustand, je *hält / hält mit Auflage / hält nicht* mit Kostenargument). Aufwand-Skala der Vorlage: XS Text-Edit · S eine Funktion · M Modul-Umbau · L Cross-Modul · XL Strukturwechsel. Schweregrad 1–4 ehrlich verteilt.
4. **Entscheidung ≠ Drift.** Was CLAUDE.md als Entscheidung führt, wird als Entscheidung beurteilt, nicht als Verstoß gelistet. Was CLAUDE.md behauptet und der Code nicht mehr tut, ist **Drift** — wandert in eine Sektion „Input für DOC-AUDIT" (nur Liste, nicht fixen) oder, wenn es Code-Drift ist, in die Findings.
5. **Keine Kosmetik.** Namenskonventionen, Antwortformen, Fehlertexte, Import-Reihenfolgen, Docstring-Stil → Sektion „Input für CONSIST-AUDIT" (Liste), keine Findings. Testqualität → „Input für TEST-AUDIT". Das Audit bleibt bei Struktur.
6. **Tech-Debt-Ratio nur mit benannter Formel** (z. B. Summe der Remediation-Schätzungen in Sprint-Tagen ÷ geschätzter Bestand) und als Näherung markiert; sonst *„nicht aus Input ableitbar"* — die Vorlage erlaubt das ausdrücklich.
7. **Werkzeuge.** Stdlib-AST (Anhang B), `git log`, `grep`. `vulture`/`radon` sind in einem Mac-venv erlaubt (`python3 -m venv /tmp/arch-venv`, nicht ins Repo, nicht in `requirements.txt`); jeder Treffer ist eine Hypothese, die über die vier Kanäle aus der Messfalle oben bestätigt wird. **Kein Mintbox-Zugriff nötig**; falls doch (z. B. `pip freeze` im Container), nur lesend.
8. **Ablage**: `docs/archive/audit-outputs/AUDIT_ARCHITEKTUR_2026-10-01.md` (Konvention der Reihe). Ins BACKLOG wandern **nur Top-5** als Items mit Code, Größe, Region, Gate — nicht alle Findings (Notion-Workflow). Cluster **nach Code-Region**, nicht nach Audit-Reihenfolge.
9. ⚠️ Editiert wird nur auf dem Mac. Keine Token, keine Pfade mit Geheimnissen im Doc.

---

# Phase 1 — Befund

## 1.1 Input zusammenstellen (die drei Artefakte der Vorlage)

1. **Directory-Tree**: `tree -L 3 -I '__pycache__|node_modules|corpus|.git' .` (Mac; `brew install tree` falls fehlend, sonst `find … -maxdepth 3`). `corpus/` separat nur als Zahl.
2. **Git-Hot-Spots**: `git log --since=2026-05-01 --name-only --pretty=format: -- '*.py' '*.js' '*.html' '*.css' | grep -v '^$' | sort | uniq -c | sort -rn | head -30` — die Master-Zahlen oben sind der Stand 2026-10-01, reproduzieren und ins Doc.
3. **Verdächtige Kern-Dateien** (Master-Vorschlag, aus Churn + LOC + Hypothesen; ergänzen oder streichen mit Begründung): `app.py`, `app_pkg/__init__.py`, `app_pkg/config.py`, `app_pkg/library.py`, `app_pkg/cards.py`, `app_pkg/tags.py`, `app_pkg/narration.py` + `app_pkg/audio.py` + `app_pkg/document_api.py` (die drei Option-B-Zwillinge), `tasks.py`, `models.py`, `static/js/library_detail.js`, `static/css/style.css`.

Dazu lesen: CLAUDE.md vollständig (die *Architecture Notes* sind das Protokoll der Entscheidungen), [docs/cleanup_plan.md](../../cleanup_plan.md) und [docs/inventory_2026-05.md](../../inventory_2026-05.md) (der Mai-Stand — was damals bereinigt wurde, darf nicht als neu gefunden werden), [docs/reader_architecture.md](../../reader_architecture.md), [docs/css_principles.md](../../css_principles.md), `app.py`, `services/__init__.py`, alle `register(app)`-Module, `tasks.py`, `worker.py`, `models.py`, `app_pkg/config.py`.

Das Messskript aus Anhang B liefert Import-Graph (Top-Level und in Funktionen), Routen je Modul und die längste Funktion je Modul — ausführen, Ausgabe ins Doc, dann erweitern (Cross-Feature-Matrix innerhalb `app_pkg`, Fan-in/Fan-out je Modul).

## 1.2 Die fünf Stufen der Vorlage

In der Reihenfolge der Vorlage, je Stufe mit Messung:

1. **Kartografie** — Komponenten und Grenzen (Web/Worker/Launcher/Redis + Engines + Tore), Entry-Points (`app.py`, `worker.py`, `services.mineru_launcher`, `scripts/*`), kritische Pfade (Login → Session; Upload → Job → Reconcile; Markdown → Renderer → PDF/EPUB/Reader; Karte → Scheduler), Abhängigkeits-Graph als Mermaid (Module als Knoten, Importe als Kanten; Zyklen rot).
2. **Hot-Spots** — Tabelle mit Volatilität (Churn), Logik-Konzentration (LOC, längste Funktion, Routen), Bug-Anziehung (Commits mit `fix(` je Datei: `git log --since=2026-05-01 --pretty=format:%s --name-only` nach Datei auswerten), Hot-Score (Formel benennen). Mindestens die 15 aus dem Ist-Zustand; Autoren-Spalte „nicht anwendbar".
3. **Architektur-Verstöße** — Tight Coupling (Singleton-Seam, Cross-Feature-Helfer), God-Files (`__init__.py`, die `register()`-Closures), Schicht-Verletzungen (`services → app_pkg.config`, der Config-↔-mineru_invocation-Zyklus), zyklische Abhängigkeiten (alle, aus dem AST, mit dem späten Import, der sie auflöst), Cross-cutting ohne Layer (Auth-Helfer in `ingest`, Zeit-Helfer in `cards`/`learn`/`config`, Status-Literale).
4. **Wucherungs-Patterns** — Cargo-Cult (die drei gespiegelten `_authorize_*`: Diff zeigen, was identisch und was absichtlich verschieden ist), Dead Code (nur mit Vier-Kanal-Nachweis; Kandidaten: `services/document_pipeline.py` — fährt der Cloud-Pfad real darüber?, `app_pkg/documents.py`-Reste nach DOC-WEB-ASYNC, `scripts/backfill_*`/`cleanup_tags.py` als Einmal-Werkzeuge, `services/gemini/*` je Datei, `corpus/`-Harness-Reste, nicht mehr referenzierte CSS-Selektoren in `style.css` gegen `templates/` + `static/js/` + Renderer-Klassen aus `markdown_render.py`), Time Bombs (`Notion-Version: 2022-06-28`, `mineru:3.4.4` nur lokal ohne Registry-Digest, Playwright-Pin ↔ Image-Tag-Kopplung, `constraints.txt`-Freeze, SQLite-Einzeldatei als Skalengrenze — je mit Urteil, ob es für Single-User trägt), Parallel-Implementierungen (die drei Reconciles: Diff; die drei atomaren Ergebnis-Schreiber; Status-Literale), Abandoned Refactorings (Blueprint-Vokabular ohne Blueprints, Podcast-Namen, `GeminiService`-Seam ohne Caller seit NARR-5, halb zentralisierte Helfer).
5. **Tech-Debt-Metrik** — je Hot-Spot Remediation XS–XL, Ratio nur mit Formel (Entscheidung 6), **strategische Empfehlung**: was zuerst, was später, was nie — mit Begründung aus Churn (wo der nächste Sprint sowieso hinfasst) und Risiko (wo ein Umbau ein Gate hat: Suite, `gate_render_bytes.py`, Smokes).

## 1.3 VERIFY-Liste (Master-vorab; jedes Item mit Beleg auflösen, nicht mit Meinung)

- **Closures**: `wc -l` je `register()`, Zahl der Views je Modul, Zahl der Helfer außerhalb; **Prototyp am Test-Client (nur im venv/Scratch, nicht im Repo)**: lässt sich eine View als Modul-Funktion per `app.add_url_rule(rule, endpoint='login', view_func=…)` registrieren, sodass `url_for('login')` unverändert trägt? Ergebnis als Mechanik-Vorschlag mit Aufwand je Modul.
- **Singleton-Seam**: `grep -rnE "patch\(['\"]app\." tests/ | wc -l` und die Liste der gepatchten Namen; welche Routen-Funktionen lesen welchen Singleton; ob `current_app.extensions` oder ein `app_pkg.services_registry` dieselbe Test-Patchbarkeit böte (Test-Client-Beleg im Scratch, kein Repo-Edit).
- **Config-Zyklus**: `app_pkg.config` ↔ `services.mineru_invocation` — welche Namen in welche Richtung; welche der 29 Konstanten nur **ein** Konsument hat (gehören zu ihm), welche mehrere (gehören in eine neutrale Schicht). Tabelle Konstante → Konsumenten.
- **Option-B-Zwillinge**: `diff` der drei `reconcile_*` und der drei `_authorize_*` (normalisiert); welche Zeilen sind identisch, welche bewusst verschieden (Kontrakte in `docs/*_contract.md` nennen die Unterschiede); Schätzung für ein gemeinsames `app_pkg/jobs.py` mit den drei als Parametrisierung, und was es an Tests bräuchte.
- **`__init__.py`-Extraktionen**: welche Sentinels decken Migrationen, CLI, CSRF-Inversion, Header (`tests/test_csrf_inversion.py`, `test_proxy_fix.py`, `test_set_password*`, …) — reicht die Suite als Gate für ein reines Verschieben?
- **Library-Region**: Verantwortungs-Karte für `library.py` (12 Routen → Gruppen), `library_detail.js` (1 830 Zeilen → Funktionsgruppen: Reader, Highlights, Progress, Player, Figuren, Platzieren), `library_detail.html`; welche Gates existieren (`smoke_markdown_reader`, `smoke_reader_media`, `measure_highlight_anchors`, `gate_render_bytes`).
- **`style.css`**: Selektoren-Inventar gegen Templates + JS + Renderer-Klassen; Zahl der nie referenzierten Selektoren mit Positivkontrolle (ein bekannt genutzter muss als genutzt zählen); die TOC-Sektionen gegen die Feature-Module (deckt sich die Gliederung noch?).
- **`corpus/` und `scripts/`**: was in `corpus/` getrackt ist (Harness vs. Ergebnisse vs. Gold), ob die Harness noch läuft (Import-Check), welche Skripte Einmal-Werkzeuge sind (Datum des letzten Aufrufs aus Git/STATUS) — Vorschlag: Archiv-Verzeichnis statt Löschen.
- **`GeminiService`-Seam**: Caller-Grep (Code, Tests, Skripte); Kosten des Stehenlassens (Import im `app.py`, `_LAZY`-Eintrag, Tests) vs. Löschen — Urteil als Entscheidung, nicht als Finding.
- **Mai-Abgleich**: die 18 Findings aus [docs/cleanup_plan.md](../../cleanup_plan.md) — welche sind nachgewuchert (gleiches Muster wieder da)? Das ist die Vorlage-Frage *„was nach Re-Architektur-Wellen nachgewuchert ist"*.

## 1.4 Bericht

Befund-Doc unter `docs/archive/audit-outputs/AUDIT_ARCHITEKTUR_2026-10-01.md` im Format aus Anhang A plus Sektion „Dokumentierte Entscheidungen" (Entscheidung 3/4), die Input-Listen für CONSIST/TEST/DOC-AUDIT, Anhang B (Tree + Hot-Spot-Liste + Skript-Ausgabe), Anhang C (Mess-Belege), beantwortete Validierungs-Checkliste. **Commit + Push** (`docs(ARCH-AUDIT): Befund — …`). Kein weiterer Edit.

## Stop
Bericht an den Master: die Karte in fünf Sätzen · Top-5 Hot-Spots mit Score · die Verstöße mit Schweregrad · die Wucherungs-Findings · die strategische Empfehlung · die Urteile zu den zehn Entscheidungen · was nicht ableitbar war. Dann warten. Der Master wählt die Top-5 für den Schnitt.

---

# Phase 2 — Schnitt + Wrap (kein Code)

- **Top-5 → BACKLOG-Items** (nur die im Sign-off freigegebenen): je Item Code (`ARCH-<REGION>`, z. B. `ARCH-ROUTES`, `ARCH-JOBS`, `ARCH-FACTORY`, `ARCH-CONFIG`, `ARCH-LIBRARY` — Vorschlag, der Bericht benennt sie), Größe, Region, Befund-Nummern, **Gate** (womit ein Umbau Verhaltensneutralität belegt: Suite-Baseline 1341 + 1 Skip, `gate_render_bytes.py` bei Renderer-Nähe, die Smokes bei UI-Nähe, Container-Suite auf dem Pin), Reihung mit Begründung (Cluster nach Region; was der nächste Feature-Sprint ohnehin anfasst, zuerst). Prio-Einordnung: P1 nur, wenn der Befund laufende Kosten belegt, sonst P2.
- **Input-Listen** für CONSIST-AUDIT, TEST-AUDIT, DOC-AUDIT bleiben im Befund-Doc; im jeweiligen Reihe-Satz des BACKLOG (Item SEC-AUDIT, „benannte Reihe") nur ein Verweis.
- **CLAUDE.md**: ein Satz im Kopf-Absatz hinter der Audit-Liste (Reihe: Security 09/25 → Architektur 10/01, Befund-Doc, Top-5 als Items) — kein zweiter Architektur-Roman; das Doc ist der Ort. Drift-Funde in CLAUDE.md **nicht** fixen (DOC-AUDIT), nur im Befund listen.
- **STATUS.md** (Eintrag mit Karte-in-Kurz, Top-5, Entscheidungs-Urteilen), **BACKLOG.md** (ARCH-AUDIT schließen, Items anlegen; ⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber).
- **Kein Brief ans converter-mcp** (keine Agent-Fläche berührt) — im Wrap so sagen.
- **Memory** nur bei übertragbarer Lehre; Kandidat: *Dead-Code-Nachweis braucht vier Kanäle (relative Importe, lazy Lader, `url_for`, Tests)* — prüfen, ob `feedback_zero_count_needs_positive_control` das schon trägt; sonst `reference_*`.
- Commit + Push, Stop + Bericht.

## Nicht-Ziele

- **Kein Refactor, kein Löschen, kein Dependency-Bump, kein Deploy.** Nichts auf der Mintbox.
- **Keine** Konsistenz-, Test- oder Doku-Befunde als Findings (Listen für die Folge-Audits).
- **Keine** Re-Litigation gesperrter Entscheidungen ohne Kostenargument; „ich würde es anders bauen" ist kein Finding.
- **Keine** Frontend-Performance, a11y, UX (eigene Vorlagen im Katalog, zurückgestellt).

---

## Anhang A — Der Prüfkatalog (Notion „Audit — Architektur- & Wucherungs-Audit", MINTBOX, 2026-05-16, wörtlich)

Quelle: https://app.notion.com/p/Audit-Architektur-Wucherungs-Audit-3623f5db30d281e4a2cbc39a830973f5

```
ROLLE
Du bist Senior Software-Architekt mit Spezialisierung auf Tech-Debt-Identifikation und strukturelle Code-Health. Du analysierst gewachsenen Code in Schichten und bist trainiert darin, Wucherungen, Hot-Spots und akkumulierte Design-Drift zu erkennen. Du arbeitest auf Deutsch, präzise, ohne Marketing-Tonalität.

KONTEXT
- Produkt: CONVERTER — Flask-Multimedia-Konverter (Markdown↔PDF/EPUB/Reader, Dokument→Markdown mit Cloud- und lokaler Engine, Transkription, Narration, Lernkarten mit FSRS, Agent-Schreibflächen per Token), Single-User, Docker (Web/Worker/Launcher/Redis) hinter Host-nginx
- Codebase-Größe: 14 005 LOC Python in 61 Modulen · 5 972 LOC JS in 13 Dateien · 2 952 LOC CSS · 12 Templates · 1341 Tests · 660 Commits
- Letzte Architektur-Reflektion: Cleanup-Welle Mai 2026 (geschlossen 2026-05-11); seitdem 440 Commits, 86 Sprints
- Aktuelle Schmerz-Hypothese: siehe Sprint-Prompt „Schmerz-Hypothesen des Masters" (Closures, Singleton-Seam, Option-B-Dreifaltigkeit, Factory-God-File, Library-Region, Podcast-Namen)

INPUT
---
[DIRECTORY-TREE + GIT-HOTSPOT-LISTE + 5-10 VERDÄCHTIGE KERN-FILES]
---

AUFGABE

Führe einen mehrschichtigen Architektur-Audit in fünf Stufen durch:

1. CODEBASE-KARTOGRAFIE
   - Hauptkomponenten + Verantwortlichkeits-Grenzen
   - Entry-Points + kritische Pfade
   - Dependency-Beziehungen (welches Modul ruft welches)

2. HOT-SPOT-IDENTIFIKATION
   - Häufig geänderte Files (Git-Volatilität → potentieller Schmerz)
   - Files mit hoher Logik-Konzentration (LoC-Cluster)
   - "Magnet-Files" die in vielen Bug-Fixes auftauchen
   - Files mit vielen unterschiedlichen Autoren über die Zeit

3. ARCHITEKTUR-VERSTÖSSE
   - Tight Coupling (Module hängen unnötig voneinander ab)
   - God-Objects / God-Files (zu viele Verantwortlichkeiten)
   - Schicht-Verletzungen (Lower-Layer importiert Higher-Layer)
   - Zyklische Dependencies
   - Cross-cutting Concerns ohne sauberen Layer

4. WUCHERUNGS-PATTERNS
   - Cargo-Cult-Code (kopiert ohne Verständnis, leicht abgewandelt)
   - Dead Code (nicht erreichbar / nicht genutzt)
   - Time Bombs (hardcodierte Daten, deprecated Dependencies, Scale-Limits)
   - Parallel-Implementierungen desselben Konzepts an verschiedenen Stellen
   - Abandoned Refactorings (halb-migrierte Patterns)

5. TECH-DEBT-METRIK & PRIORISIERUNG
   - Pro Hot-Spot: Remediation-Cost-Schätzung (XS/S/M/L/XL)
   - Approximierter Tech-Debt-Ratio: < 5% gesund, 5-10% akzeptabel, 10-20% Warnung, > 20% systemisch
   - Strategische Empfehlung: Was zuerst, was später, was nie

OUTPUT-FORMAT

### Karte
[ASCII- oder Mermaid-Diagramm der Komponenten + Hauptabhängigkeiten]

### Hot-Spots
| Rang | File/Modul | Volatilität | Logik-Konzentration | Bug-Anziehung | Hot-Score |
|---|---|---|---|---|---|

### Architektur-Verstöße
| # | Verstoß-Typ | Fundstelle | Warum problematisch | Schweregrad 1-4 |
|---|---|---|---|---|

### Wucherungs-Findings
| # | Pattern | Fundstelle | Was zu tun | Aufwand |
|---|---|---|---|---|

### Tech-Debt-Priorisierung
| Rang | Item | Schweregrad × Frequenz | Aufwand | Empfohlene Aktion |
|---|---|---|---|---|

### Strategische Empfehlung
2-4 Sätze: Was sind die 1-3 wichtigsten architektonischen Bewegungen für die nächsten Sprints? Reihenfolge begründet.

REGELN
- Keine Spekulation — nur Findings die aus dem Input ableitbar sind.
- Keine kosmetischen Probleme (die gehören in andere Audits).
- Schweregrad ehrlich verteilen: nicht alles ist kritisch, nicht alles kosmetisch.
- Aufwand-Skala: XS=Text-Edit, S=einzelne Funktion, M=Modul-Umbau, L=Cross-Modul-Refactor, XL=Strukturwechsel.
- Empfehlungen müssen umsetzbar sein — kein "sollte refactored werden", sondern "extract X aus Y in neues Modul Z, ~M Aufwand".
- Bei unklarer Datenlage: markiere "nicht aus Input ableitbar" statt zu raten.

VALIDIERUNGS-CHECKLISTE (am Ende des Befund-Docs beantworten)
- [ ] Sind die identifizierten Hot-Spots wirklich Hot-Spots (Git-Volatilität stichprobenhaft gegenprüfen)?
- [ ] Ist die Schweregrad-Verteilung plausibel (nicht alles "kritisch", nicht alles "kosmetisch")?
- [ ] Sind Aufwand-Schätzungen ehrlich (nicht alles XL)?
- [ ] Sind die Empfehlungen konkret und umsetzbar?
- [ ] Decken die Top-5 wirklich die schmerzhaftesten Stellen ab (Bauch-Check)?
```

Nutzung laut Vorlage: Output durchgehen, **Top-5** markieren, **nur diese** als BACKLOG-Items mit Prio, Cleanup-Sprints **nach Architektur-Strang clustern** (welche Findings teilen eine Code-Region), nach dem Cleanup die Checkliste erneut, ggf. neuer Lauf nach 3–5 Sprints. Die Autoren-Dimension („viele Autoren") ist für dieses Repo nicht anwendbar — im Befund so markieren, nicht weglassen.

## Anhang B — Master-Messskript (Stand 2026-10-01; ausführen, erweitern, Ausgabe ins Doc)

Churn (Code, Templates, CSS; seit der Cleanup-Welle):

```bash
git log --since=2026-05-01 --name-only --pretty=format: -- '*.py' '*.js' '*.html' '*.css' | grep -v '^$' | sort | uniq -c | sort -rn | head -30
```

Import-Graph, späte Importe, Routen, längste Funktion je Modul (Stdlib, read-only; im Repo-Root ausführen):

```python
import ast, pathlib, re, collections
root = pathlib.Path('.')
mods = sorted(p for p in list(root.glob('app_pkg/**/*.py')) + list(root.glob('services/**/*.py'))
              + [root/'app.py', root/'tasks.py', root/'worker.py', root/'models.py'] if '__pycache__' not in str(p))
INTERNAL = ('app_pkg', 'services', 'models', 'tasks', 'app', 'worker')
def modname(p):
    s = str(p)[:-3].replace('/', '.')
    return s[:-9] if s.endswith('.__init__') else s
names = {modname(p): p for p in mods}
top_imports, late_imports, routes, longest = collections.defaultdict(set), collections.defaultdict(list), {}, {}
for m, p in names.items():
    src = p.read_text(); tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            tgt = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            if isinstance(node, ast.ImportFrom) and node.level:      # relative import → auflösen
                base = m.rsplit('.', node.level)[0] if '.' in m else m
                tgt = f'{base}.{node.module}' if node.module else base
            if tgt and tgt.split('.')[0] in INTERNAL: top_imports[m].add(tgt)
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in ast.walk(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                tgt = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                if tgt and tgt.split('.')[0] in INTERNAL: late_imports[m].append((node.lineno, tgt))
    fdefs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if fdefs:
        big = max(fdefs, key=lambda n: n.end_lineno - n.lineno)
        longest[m] = (len(fdefs), big.name, big.end_lineno - big.lineno + 1)
    routes[m] = len(re.findall(r'@app\.route\(', src))
print('# services -> app_pkg/models/app (Schichtverletzungen)')
for m in sorted(top_imports):
    bad = sorted(t for t in top_imports[m] if m.startswith('services') and t.split('.')[0] in ('app_pkg', 'models', 'app', 'tasks'))
    if bad: print(f'  {m}: {bad}')
print('# späte Importe in Funktionen (Zirkel-Verdacht)')
for m in sorted(late_imports): print(f'  {m}: {len(late_imports[m])} {late_imports[m][:4]}')
print('# Routen je Modul'); [print(f'  {m}: {c}') for m, c in sorted(routes.items(), key=lambda kv: -kv[1]) if c]
print('# Funktionen / längste Funktion je Modul')
for m, (n, f, L) in sorted(longest.items(), key=lambda kv: -kv[1][2])[:20]: print(f'  {m}: {n} Funktionen, längste {f} = {L} Zeilen')
```

⚠️ Das Skript sieht **keine** Kanten über den PEP-562-Lader (`services._LAZY`), über `url_for` in Templates/JS und über Test-Patches — für Dead-Code-Aussagen die vier Kanäle aus dem Ist-Zustand prüfen.
