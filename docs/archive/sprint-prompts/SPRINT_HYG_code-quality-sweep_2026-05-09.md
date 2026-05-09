# Sprint HYG — Code-Quality-Sweep der Stage-4-Restposten

**Datum**: 2026-05-09

**Ziel**: Acht verbleibende Hygiene-Findings aus Cleanup Stage 4 schließen. Alle Low-Severity, alle Code-Quality / Defensive Boundaries, kein Behavior-Change im Standard-Pfad. Nach diesem Sprint sind die Stage-4-Findings durch (außer denen, die bewusst als wontfix markiert sind) und der Cleanup-Block kippt in die CVE-Wellen.

**Vorbedingung**:
- Pytest 43/43 grün auf `main`. Letzter Code-Touch: SEC-Sprint (commit `6a18086`, 2026-05-09). Sprint-Close-Commit `c7b86c0` lokal, kann gepusht sein oder noch nicht — irrelevant für den Sprint, weil dort nur STATUS/BACKLOG drin ist.
- Findings-Volltext: [docs/cleanup_plan.md](docs/cleanup_plan.md), Sektion „Findings (populated by Stage 4)".

**Out-of-scope**:
- Alle CVE-Bumps (`CVE-LOW`, `CVE-PDF`, `CVE-RQ`, `CVE-DG`) — eigene Folge-Sprints.
- F-014 (PDF-extraction silent fallback) — bereits als wontfix markiert (intentional silent fallback in heuristic hot loop).
- UX-/Frontend-Patterns — F-3-Welle kommt nach allen Cleanup-Sprints.

---

## Phase 1 — Implementation

Pre-Flight:

1. `pytest tests/` — muss 43/43 grün sein.
2. Findings-Volltext lesen: [docs/cleanup_plan.md](docs/cleanup_plan.md), Einträge F-002, F-007, F-008, F-011, F-012, F-015, F-016, F-017.

**Findings dieses Sprints** (alle 8 Pflicht):

### F-002 — Narrow `except` in `highlight_code`
**Wo**: `app_pkg/markdown.py` (per `grep -n "highlight_code\|ClassNotFound" app_pkg/markdown.py`).

**Was**: `except Exception` für Pygments-Lexer-Fallback ist zu breit. Sollte auf `pygments.util.ClassNotFound` eingeengt werden — anderer Exception-Typ (z.B. ImportError) verdient Logging.

**Default**: `from pygments.util import ClassNotFound` + `except ClassNotFound:`.

### F-007 — `secure_filename(None)` AttributeError absichern
**Wo**: `app_pkg/markdown.py` (per `grep -n "secure_filename\|output_filename" app_pkg/markdown.py`), historisch `app.py:336-340`.

**Was**: `request.form.get('output_filename')` kann `None` zurückgeben; `secure_filename(None)` wirft `AttributeError`, der vom breiten `except` als „PDF generation failed" maskiert wird. Downstream `if not safe_filename:` ist im None-Fall dead.

**Default**: explizite Empty-Default-Strategie: `output_filename = request.form.get('output_filename') or ''` vor `secure_filename`. Falls leer → bestehende Empty-Filename-Behandlung greift sauber.

**Test**: ein neuer Test (z.B. `test_convert_markdown_missing_filename_field_handled`) — POST ohne `output_filename`-Field; erwartet sauberer Pfad statt 500.

### F-008 partial — 4 Logging-Sites ohne `exc_info=True`
**Wo** (4 Sites zu finden via `grep -rn "logger.error\|app.logger.error" app_pkg/`):
- markdown PDF generation (vermutlich `app_pkg/markdown.py`)
- Deepgram-key issuance (vermutlich `app_pkg/audio.py`)
- podcast TTS-temp cleanup (vermutlich `app_pkg/podcasts.py`)
- Notion-MCP transport failure (vermutlich `app_pkg/integrations/notion.py`)

**Was**: Symmetrische Error-Pfade haben heterogenes Logging — die 4 Sites droppen den Stacktrace, andere haben ihn. Standardisieren auf always-on `exc_info=True` in allen `.error`-Pfaden.

**Default**: `exc_info=True` an allen 4 Sites ergänzen. Behavior-Change ist „Log-Output enthält jetzt Stack" — dokumentieren im Commit-Body, nicht im Code. Keine Tests nötig (Logging-Output, nicht Verhaltens-Fixture).

### F-011 — Service-availability-Decorator
**Wo**: 6 Endpoints, historisch `app.py:524, 538, 583, 625, 640, 784` — heute verteilt über `app_pkg/audio.py`, `app_pkg/podcasts.py`, evtl. weitere. Per `grep -rn 'jsonify.*not configured.*503\|return.*503' app_pkg/`.

**Was**: dasselbe 5-zeilige `if not deepgram_service: return jsonify({"error": "Deepgram not configured"}), 503` an 6 Stellen für 3 Services.

**Default**:
1. Decorator-Helper in **neuem** Modul `app_pkg/decorators.py` — `@require_service('deepgram')`, `@require_service('google_tts')`, `@require_service('gemini')`. Service-Name wird zur Lookup-Key auf den Singletons in `app.py`.
2. DE-Microcopy für die Error-Message (z.B. „Deepgram-Service nicht konfiguriert. API-Key fehlt in der Server-Konfiguration."). Aktuell ist die Message schon englisch — die ändern wir hier mit, weil's eine User-facing-String ist.
3. Tests: ein neuer Test pro Service, der via Singleton-Mock auf `None` setzt und 503+DE-JSON erwartet (z.B. `test_audio_transcribe_returns_503_when_deepgram_not_configured`). 3 Tests gesamt.

### F-012 — Dead `if not file:` check
**Wo**: `app_pkg/documents.py` (per `grep -n "if not file" app_pkg/documents.py`).

**Was**: `request.files['document_file']` raised `KeyError` oder gibt `FileStorage` zurück — letzteres ist immer truthy. Der vorherige `if 'document_file' not in request.files`-Check deckt das KeyError-Szenario schon ab. Der `if not file:`-Block ist unerreichbar.

**Default**: Block ersatzlos entfernen. Kein Test nötig (Code wird schmaler, Verhalten unverändert).

### F-015 — Timeout-Alignment
**Wo**: `services/gemini/client.py` (oder `tts.py` — 300s), `services/deepgram_service.py` (~600s), `app_pkg/podcasts.py` (RQ job_timeout 600s). Per `grep -rn "timeout=\|job_timeout" services/ app_pkg/`.

**Was**: Drei Magic-Timeouts ohne dokumentierten Bezug zueinander.

**Default**: **Konsolidierung in `app_pkg/config.py`** als benannte Konstanten:
```python
TIMEOUT_GEMINI_SECONDS = 300
TIMEOUT_DEEPGRAM_SECONDS = 600
TIMEOUT_RQ_JOB_SECONDS = 600
```
Wert-Erhalt zwingend (kein Behavior-Change). Module importieren und referenzieren statt Magic-Number. Falls einer der Werte aus einer Env-Var kommt: Existing-Pattern beibehalten, nur die Default-Konstante zentralisieren. Kein Test nötig (Konstanten-Verschiebung).

### F-016 — Doppel-Log in `_filter_metadata_lines`
**Wo**: [services/gemini/dialogue.py](services/gemini/dialogue.py) (callee-side, „Filtered N metadata lines") + Caller-Site irgendwo in `services/gemini/` (caller-side, „Filtered out N metadata lines"). Per `grep -rn "metadata lines" services/gemini/`.

**Was**: Zwei Log-Lines pro Aufruf, dieselbe Zahl. Caller-side wickelt schon mit `if filtered_count > 0`, ist also sauberer.

**Default**: Callee-side Log-Line droppen. Caller-side bleibt. Kein Test nötig.

### F-017 — `isinstance(data, dict)`-Check für `request.get_json()`
**Wo**: 6 Stellen, historisch `app.py:588, 644, 788, 875, 905, 953` — heute verteilt. Per `grep -rn "request.get_json()" app_pkg/`.

**Was**: Wenn ein Client JSON-List `[]` oder Scalar `"foo"` schickt, raised `data.get(...)` `AttributeError` → 500. Inkonsistent mit `api_create_conversion`'s defensive Pattern.

**Default**: einheitlicher Inline-Check pro Stelle:
```python
data = request.get_json(silent=True)
if not isinstance(data, dict):
    return jsonify({"error": "Ungültiger Request-Body. JSON-Objekt erwartet."}), 400
```
Drei Zeilen pro Stelle, kein Helper nötig. **Aber:** wenn der Sub-Thread merkt, dass die 6 Stellen sehr ähnlich sind und ein Decorator (`@expects_json_object`) sich aufdrängt: kurz im Bericht erwähnen, der Default bleibt aber Inline.

**Tests**: 1–2 neue Tests, die auf einen der Endpoints einen JSON-List-Body schicken und 400 erwarten (z.B. `test_api_create_conversion_rejects_non_dict_body`).

---

**Mechanik-Leitplanken**:

- **DE-Microcopy** in allen neuen Error-JSON-Bodies (Fehler max 2 Sätze).
- **Kein Behavior-Change** im Standard-Pfad. Tests müssen vor und nach jedem Finding grün sein.
- **`pytest tests/` zwingend grün** nach jeder einzelnen Finding-Behebung.
- **Erwartete Final-Test-Anzahl**: 43 + 1 (F-007) + 3 (F-011) + 1–2 (F-017) = **48–49 Tests grün**.
- **Sub-Batch-Vorschlag**: alle 8 Findings sind unverkoppelt. Ein Sub-Batch pro Finding ist OK, aber Holistic in zwei Batches geht auch (Batch A: F-002, F-007, F-012, F-015, F-016 — read-only-Refactors ohne Tests; Batch B: F-008, F-011, F-017 — Logging + Tests). Sub-Thread entscheidet pragmatisch.

**Erwartete Files**:

```
app_pkg/markdown.py            # F-002 + F-007 + F-008 (1 Logging-Site)
app_pkg/audio.py               # F-008 (1 Site) + F-011 (Decorator-Anwendung)
app_pkg/podcasts.py            # F-008 (1 Site) + F-011 (Decorator-Anwendung) + F-015 (Timeout-Konstante)
app_pkg/integrations/notion.py # F-008 (1 Site)
app_pkg/documents.py           # F-012
app_pkg/decorators.py          # NEU — F-011 require_service-Decorator
app_pkg/config.py              # F-015 — TIMEOUT_*_SECONDS Konstanten
services/gemini/client.py oder tts.py  # F-015 — Timeout-Konstante importieren
services/deepgram_service.py   # F-015 — Timeout-Konstante importieren
services/gemini/dialogue.py    # F-016 — callee-side Log-Line entfernen
tests/test_markdown.py         # F-007 Test
tests/test_audio.py            # F-011 Deepgram-503-Test
tests/test_podcasts.py         # F-011 Gemini-503-Test, ggf. Google-TTS-503
tests/test_library.py oder test_podcasts.py  # F-017 Test (je nach Endpoint)
```

Nach Phase 1: STOP — Bericht. Welche Findings durch, ob Decorator-Helper sich für F-017 doch aufgedrängt hat, neue Test-Anzahl, ob Sub-Batches gebildet wurden.

---

## Phase 2 — Verify

1. `pytest tests/` grün (48–49 erwartet).
2. **Live-Smoke** auf `localhost:5656`:
   - Markdown-Converter: POST ohne `output_filename`-Field → kein Crash, sauberer Default-Pfad (F-007).
   - Markdown-Converter: Code-Block mit unbekannter Sprache (z.B. ` ```fooobar ` ) → Render funktioniert (Pygments-Fallback, F-002).
   - Audio-/Podcast-Endpoints: ein Service auf `None` setzen (z.B. `.env` GEMINI_API_KEY leer + Container-Restart) → 503+DE-JSON statt 500 (F-011).
   - Beliebigen JSON-API-Endpoint mit `[]` als Body anrufen via `curl -X POST -H 'Content-Type: application/json' -d '[]' …` → 400+DE-JSON (F-017).
3. Kein DevTools-Console-Fehler nach den Smoke-Pfaden.
4. Logs nach einem provozierten Error-Pfad ansehen — `exc_info` jetzt überall enthalten (F-008).

Nach Phase 2: STOP — Bericht. Liste der gesmokten Pfade, eventuelle Auffälligkeiten.

---

## Phase 3 — Commit (lokal — KEIN Push)

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Default: ein Commit für `HYG` insgesamt. Wenn Sub-Batches gefahren wurden: separate Commits in derselben Branch (Bezug auf Findings-Codes in der Message).
- Branch: direkt auf `main` ist OK.
- **`git push` ist NICHT Teil dieses Sprints.** Der Push ist explizit Master-Hoheit (Sign-off-Gate). Phase 3 endet mit `git commit` lokal + STATUS/BACKLOG-Pflege. Push folgt nach Master-Sign-off.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — 8 Findings, ein neues Modul (`app_pkg/decorators.py`), Konstanten-Konsolidierung in `app_pkg/config.py`, 4–6 neue Tests, keine Schema-Migration, keine UI-Welle. Verkopplung niedrig (alle additive, alle isoliert). Sub-Batch-Strategie sinnvoll falls Holistic zu groß wirkt.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Lesen offensichtlicher Bugs auffallen die nicht in der HYG-Liste stehen: Issue-Format („gefunden, beschrieben, **nicht** gefixt") in den Bericht — siehe Memory `feedback_no_silent_fixes.md`.
- Wenn `app_pkg/decorators.py` neu entsteht und dort weitere ähnliche Patterns kandidiert sind (z.B. `@expects_json_object` aus F-017): kurz im Bericht erwähnen.
- Wenn die F-008-Logging-Sites mehr als 4 sind (z.B. 5 oder 6 weil weitere im Code dazugekommen sind seit cleanup_plan.md geschrieben wurde): einfach mit-fixen, aber im Bericht zählen.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „HYG ☑ done 2026-05-XX → commit `<hash>`. Stage-4-Findings-Block damit durch (F-014 wontfix). Verbleibende Sequenz: CVE-LOW → … Aktueller Pytest-Stand: 48–49/48–49 grün."
- **BACKLOG.md**: Sektion „1. HYG" raus → Erledigt-Liste; Sektion „2. CVE-LOW" rückt auf Position 1, alle Folge-Sprint-Nummern -1.
- **Memory**: nichts zu erwarten — Hygiene-Findings sind selten übertragbar. Falls überraschend etwas auftaucht (z.B. „Decorator-Pattern für CONVERTER-Services X" als wiederverwendbare Konvention): `feedback_*.md` schreiben.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Findings sind in cleanup_plan.md vollständig beschrieben, Default-Empfehlungen oben gesetzt.)_
