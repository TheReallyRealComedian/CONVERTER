# Sprint SEC — Security-Hardening-Sweep

**Datum**: 2026-05-09

**Ziel**: Drei verbleibende Sicherheits-Findings aus Cleanup Stage 4 schließen. Defense-in-Depth — keine akute Exploit-Lage (LAN-only, login-protected, single-user), aber strukturell sauberer Stand vor weiteren CVE-Wellen.

**Vorbedingung**:
- Pytest 38/38 grün auf `main`. Letzter Code-Touch: F-2 Cluster II (commit `30de7cb`, 2026-05-09).
- F-006 für `document_converter` ist bereits durch (F-1 Cluster D, commit `e68b6dd`).
- F-006 für `audio_converter` ist bereits durch (F-2 Cluster II Pattern 13, commit `30de7cb`).
- Vorlagen für die Backend-Whitelist-Mechanik: [app_pkg/documents.py:30](app_pkg/documents.py#L30) (precomputed `accept`-String, fließt nach Template + `window.PageData`, Backend liefert 400+DE-JSON für unsupported extensions) plus die analoge Implementierung im audio-Endpoint aus Cluster II.

**Out-of-scope**:
- F-002, F-007, F-008-partial, F-011, F-012, F-015, F-016, F-017 — diese Hygiene-Findings kommen im Folge-Sprint `HYG`.
- CVE-Upgrades (`CVE-LOW`, `CVE-PDF`, `CVE-RQ`, `CVE-DG`) — eigene Folge-Sprints.
- UX-/Frontend-Patterns — F-3-Welle kommt nach allen Cleanup-Sprints.

---

## Phase 1 — Implementation

Pre-Flight:

1. `pytest tests/` — muss 38/38 grün sein.
2. Findings-Vollbeschreibung lesen: [docs/cleanup_plan.md](docs/cleanup_plan.md), Sektion „Findings (populated by Stage 4)", Einträge F-005, F-006, F-013.
3. F-006-Mechanik-Vorlagen ansehen:
   - F-1 Cluster D commit `e68b6dd` — Backend-Whitelist im document_converter.
   - F-2 Cluster II commit `30de7cb` — Backend-Whitelist im audio_converter.

**Findings dieses Sprints** (alle drei Pflicht):

### F-005 — Path-Traversal in `podcast_download`
**Wo**: [app_pkg/podcasts.py:194-195](app_pkg/podcasts.py#L194) (`if not real_path.startswith(os.path.realpath(OUTPUT_DIR))`).

**Was**: `startswith` ist prefix-collision-anfällig — `/app/output_podcasts2/foo.wav` würde fälschlich akzeptiert, falls jemals ein Sibling-Dir entsteht. `job.result` ist aktuell nicht user-controlled (vom Worker gesetzt), also nicht aktiv exploitable, aber das Pattern ist fragile Defense-in-Depth.

**Default-Empfehlung**: `Path.is_relative_to()` (Python 3.9+, im Container vorhanden) statt `commonpath` — modern und expressiv. Einzeiler-Fix:

```python
from pathlib import Path
if not Path(real_path).is_relative_to(Path(os.path.realpath(OUTPUT_DIR))):
    return jsonify({"error": "Datei außerhalb des Podcast-Verzeichnisses."}), 400
```

DE-Microcopy beim Reject-Pfad. Falls der Sub-Thread `commonpath` für besser hält: kurz im Bericht erwähnen, kann auch das werden.

**Test**: ein neuer Test (z.B. `test_podcast_download_path_traversal_rejected`) — simulieren dass `job.result` einen Pfad außerhalb `OUTPUT_DIR` liefert und 400/403 erwarten. Mock-Strategie analog `tests/test_podcasts.py` (vorhandene Job-Mocks).

### F-006 für `markdown_converter` — Backend-Whitelist + DE-Fehler-JSON
**Wo**: [app_pkg/markdown.py](app_pkg/markdown.py) (genauen Endpoint per `grep -n "convert_markdown\|markdown_file" app_pkg/markdown.py`).

**Was**: Aktuell akzeptiert `convert_markdown` jede Extension (Frontend-`accept=".md,.markdown"` in `templates/markdown_converter.html:74` ist nur UI-Hint, nicht enforced). `.read().decode('utf-8')` kann bei Binary-Upload `UnicodeDecodeError` werfen, der vom breiten `except Exception` als „Could not generate PDF" gemeldet wird — falscher Fehler.

**Default-Empfehlung**: Single-Source-of-Truth `ACCEPTED_EXTENSIONS = {"md", "markdown"}` im Module-Scope von `app_pkg/markdown.py`. Daraus precomputed `accept`-String → Template (analog [app_pkg/documents.py:30](app_pkg/documents.py#L30)). Im Route-Handler: Extension-Check → bei Verstoß 400 + DE-JSON `{"error": "Dateiformat nicht unterstützt. Erlaubt: .md, .markdown"}`. Das Template-Update auf `accept`-aus-`window.PageData` (Cluster D-Pattern) ist Pflicht — sonst driftet Frontend wieder.

**Test**: ein neuer Test analog F-1 Cluster D (`test_transform_document_unsupported_extension_returns_400`). Erwarteter Test-Name: `test_convert_markdown_unsupported_extension_returns_400`. In `tests/test_markdown.py`.

### F-013 — Input-Allowlist für User-supplied Config-Strings
**Wo**: drei Stellen, alle in `app_pkg/`:
- Deepgram `language`-Parameter — `app_pkg/audio.py` (per `grep -n "language\|deepgram" app_pkg/audio.py`).
- Google TTS `voice`/`language`/`speaking_rate`/`pitch` — vermutlich `app_pkg/podcasts.py` oder eigener Endpoint, per `grep -rn "google_tts\|speaking_rate\|pitch" app_pkg/`.
- Gemini `narration_style`/`script_length`/`num_speakers` — vermutlich `app_pkg/podcasts.py`, per `grep -n "narration_style\|script_length\|num_speakers" app_pkg/podcasts.py`.

**Was**: User-controlled Strings/Floats fließen ungeprüft in SDK-Calls. Kein Sicherheitsrisiko (login-gated, downstream APIs reject bad input), aber bad input → 500 statt sauberem 400. `narration_style`, `script_length`, `language` haben ohnehin enumerierte Allowlists in den Service-Klassen — die expose-en und in der Route validieren ist die ehrliche Variante.

**Default-Empfehlung**:
1. Pro Service die Allowlist-Konstanten **public** machen (falls nicht schon — bei `services/gemini/` und `services/deepgram_service.py` reinschauen).
2. Im Route-Handler: validieren vor SDK-Call. Bei Verstoß: 400 + DE-JSON `{"error": "Ungültiger Wert für <feld>. Erlaubt: a, b, c."}`.
3. Helper-Frage: drei Stellen, drei Allowlists. Eine zentrale Helper-Funktion `_validate_choice(value, allowed, field_name)` ist denkbar, lokal in jeweiligem `app_pkg/<modul>.py` oder in einem neuen `app_pkg/validation.py`. **Default-Empfehlung**: erstmal lokal duplizieren wenn der Helper trivial ist (3 Zeilen) — wenn der Sub-Thread mehr als 5 Zeilen pro Stelle braucht, dann in `app_pkg/validation.py` zentralisieren.
4. Floats (`speaking_rate`, `pitch`): Range-Check (z.B. 0.25 ≤ rate ≤ 4.0) — Werte aus Google-TTS-Doku oder dem aktuellen Service-Code übernehmen.

**Test**: 1–3 neue Tests, je nach Helper-Strategie. Mindestens einer pro betroffenem Endpoint mit ungültigem Wert, 400-Response erwartet.

---

**Mechanik-Leitplanken**:

- **DE-Microcopy** in allen Error-JSON-Bodies (Fehler max 2 Sätze).
- **Test-Daten-Hygiene**: keine echten Service-Calls, alles SDK-Boundary-Mock (Cluster I/II + F-1 Cluster D als Vorlage).
- **`pytest tests/` zwingend grün** nach jeder einzelnen Finding-Behebung — wenn nach F-005 rot, nicht weiter zu F-006.
- **Erwartete Final-Test-Anzahl**: 38 + 1 (F-005) + 1 (F-006-markdown) + 1–3 (F-013) = **41–43 Tests grün**.

**Erwartete Files**:

```
app_pkg/podcasts.py            # EDIT — F-005 + ggf. F-013 (Gemini/Google-TTS-Validation)
app_pkg/markdown.py            # EDIT — F-006 ACCEPTED_EXTENSIONS + Validation + 400+DE-JSON
app_pkg/audio.py               # EDIT — F-013 (Deepgram language)
app_pkg/validation.py          # NEU (optional, nur falls F-013-Helper > 5 LOC pro Stelle)
templates/markdown_converter.html  # EDIT — accept aus window.PageData (analog F-1 Cluster D)
static/js/markdown_converter.js    # EDIT — ggf. accept-Whitelist konsumieren (analog F-1 Cluster D)
tests/test_markdown.py             # EDIT — neuer Test für F-006
tests/test_podcasts.py             # EDIT — neuer Test für F-005, ggf. F-013
tests/test_audio.py                # EDIT — neuer Test für F-013-Deepgram
```

Nach Phase 1: STOP — Bericht. Welche Findings durch, welche Default-Empfehlungen befolgt vs. abgewichen, ob ein zentraler `validation.py`-Helper entstanden ist, neue Test-Anzahl.

---

## Phase 2 — Verify

1. `pytest tests/` grün (41–43 erwartet, je nach F-013-Test-Schnitt).
2. **Live-Smoke** auf `localhost:5656`:
   - Markdown-Converter: `.txt`-Datei hochladen → 400+DE-JSON-Banner, kein Crash, kein „Could not generate PDF"-Fehler.
   - Podcast-Download: regulärer Download funktioniert weiter (kein Regression).
   - Audio-Tab: Transcription mit gültiger `language` läuft, mit ungültiger (`language=xx-XX`) → 400+DE-Banner.
   - Podcast-Tab: Generation mit gültigen Parametern läuft, ungültiger `narration_style` → 400+DE-Banner.
3. Kein DevTools-Console-Fehler nach den Smoke-Pfaden.

Nach Phase 2: STOP — Bericht. Liste der gesmokten Pfade, eventuelle Auffälligkeiten.

---

## Phase 3 — Commit

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Default: ein Commit für `SEC` insgesamt. Wenn der Sub-Thread Findings als separate Sub-Batches gefahren hat: separate Commits in derselben Branch (Bezug auf F-005 / F-006 / F-013 in der Message).
- Branch: direkt auf `main` ist OK.
- Push erst nach Master-Sign-off.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — drei Findings, 2–3 Backend-Endpoints angefasst, 3–5 neue Tests, ein Template-Touch (markdown_converter), keine Schema-Migration, keine UI-Welle. Verkopplung zwischen den drei Findings ist niedrig (alle defensive boundaries an verschiedenen Code-Pfaden) — sequenziell pro Finding ist OK.

---

## Konstitutiv mit-genommen, falls berührt

- Falls beim Lesen von `app_pkg/markdown.py` ein offensichtlicher Bug auffällt der nicht F-006 ist: Issue-Format („gefunden, beschrieben, **nicht** gefixt") in den Bericht — siehe Memory `feedback_no_silent_fixes.md`.
- Falls F-013-Validation eine bestehende Service-Konstante explizit macht (bisher private): kurz im Bericht erwähnen, ob das Surface-Change relevant ist.
- Falls `app_pkg/validation.py` neu entsteht: kurz Begründung im Bericht (welche Patterns sich wiederholt haben).

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „SEC ☑ done 2026-05-XX → commit `<hash>`. Verbleibende Sequenz: HYG → CVE-LOW → … Aktueller Pytest-Stand: <neue Anzahl>/<neue Anzahl> grün."
- **BACKLOG.md**: Sektion „1. SEC" raus → Erledigt-Liste; Sektion „2. HYG" rückt auf Position 1, alle Folge-Sprint-Nummern -1.
- **Memory**: nur wenn übertragbare Lehre auftaucht (z.B. „Path-Traversal-Pattern in CONVERTER: Path.is_relative_to() Standard"). Defensiv: lieber nichts.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Findings sind in cleanup_plan.md vollständig beschrieben, Default-Empfehlungen oben gesetzt.)_
