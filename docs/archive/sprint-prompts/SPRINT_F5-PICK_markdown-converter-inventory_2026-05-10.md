# Sprint F5-PICK — F-5.1 UX-Inventur `markdown_converter`

**Datum**: 2026-05-10

**Ziel**: Stufe 1 (Inventur) der dreistufigen UX-Cascade-Methodik für `markdown_converter`. **Schwester-Feature zu `document_converter`** (F-1-Welle): Markdown-Datei → PDF-Konversion mit Save-to-Library-Flow. Erwartung ist hohe Cross-Feature-H4-Quote (~50%+) wegen direkter Pattern-Übertragbarkeit aus F-1. **Kein Bewerten, kein Bug-Fix, kein Pattern-Vorschlag** — das kommt in F-5.2 / F-5.3.

**Vorbedingung**:
- Pytest 65/65 grün im Container. Letzter Code-Touch: F4-IMPL-B (commits `3ef7f9e` + `3ac8786`, 2026-05-10). F-4 strukturell abgeschlossen für podcast-flow.
- Touch-Pfade des Features:
  - **Template**: [templates/markdown_converter.html](templates/markdown_converter.html) — auf <15 KB seit Stage-5 (von 24 KB extrahiert). Drop-Zone, Convert-Button, Save-to-Library, PDF-Iframe-Render.
  - **JS**: [static/js/markdown_converter.js](static/js/markdown_converter.js) — eigenes File seit Stage-5.
  - **Route-Modul**: [app_pkg/markdown.py](app_pkg/markdown.py) — `convert_markdown` Endpoint mit ACCEPTED_EXTENSIONS-Whitelist (durch SEC-Sprint), `highlight_code` für Pygments (durch HYG-narrow-except).
  - **Shared**: [static/css/style.css](static/css/style.css), [static/js/_utils.js](static/js/_utils.js) (`showAlert`, `showToast`, `formatFileSize`, `safeJSON`, `formatDatetimeLocalNow`, `.sr-only`).
- **Was schon durch frühere Sprints erledigt ist** (NICHT als Befund aufnehmen):
  - **F-006 Backend-Whitelist** für markdown_converter durch SEC-Sprint (commit `6a18086`) — `ACCEPTED_EXTENSIONS = {"md", "markdown"}` als Single-Source-of-Truth in `app_pkg/markdown.py`, fließt nach Template, Backend liefert 400+DE-JSON für unsupported.
  - **F-002 narrow-except in `highlight_code`** durch HYG-Sprint (commit `1d1c30a`) — `pygments.util.ClassNotFound`-narrow.
  - **F-007 secure_filename(None)-Guard** durch HYG-Sprint.
  - **F-008 Logging-Sites mit `exc_info=True`** für markdown PDF generation durch HYG-Sprint.
  - **F-011 `@require_service`-Decorator** für relevante Endpoints durch HYG-Sprint.
  - **F-1-Hot-Fix Jinja2 Generator-Expression** in document_converter (kein direkter markdown_converter-Touch, aber Test-Suite-Lehre erinnern: Templates werden nicht gerendert in pytest, Live-Smoke nach Template-Änderung Pflicht).
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren — `markdown_converter` ist Schwester zu `document_converter`, also F-1.1 ist die direkte Strukturvorlage):
  - F-1.1 Inventur: [docs/ui_inventory_document_converter_2026-05.md](docs/ui_inventory_document_converter_2026-05.md) — 24 Elemente, 6 fehlende States, 5 Code↔live-Divergenzen, 9 separate Befunde (3 als Bug-Tickets).
  - F-2.1 Inventur: [docs/ui_inventory_audio_converter_2026-05.md](docs/ui_inventory_audio_converter_2026-05.md) — 47 Elemente, audio-spezifische States.
  - F-3.1 Inventur: [docs/ui_inventory_library_detail_2026-05.md](docs/ui_inventory_library_detail_2026-05.md) — 21 Elemente, Live-Walkthrough-Lücken-Sektion.
  - F-4.1 Inventur: [docs/ui_inventory_podcast_flow_2026-05.md](docs/ui_inventory_podcast_flow_2026-05.md) — 17 Elemente, Async-Pipeline-Mapping-Sektion.
- **Schwester-Feature-Erwartung** (sehr wichtig für Effizienz dieser Inventur):
  - Drop-Zone-Mechanik analog F-1 (Drop, Click-to-Browse, Drag-Highlight).
  - Convert-Button + Loading-State + Result-Display.
  - Save-to-Library-Pfad mit Filename-Eingabe.
  - F-1 hatte 14 Patterns + 3 Bug-Tickets. Erwartung F-5: ~80% direkt aus F-1 übertragbar (gleiche User-Aktionen, gleiche UI-Komponenten), ~20% markdown-spezifisch (PDF-Iframe-Render, Pygments-Code-Highlighting in PDF, Output-Format-Optionen).
- **Kontext für die Methodik**: Single-User-App, LAN-only, login-protected. Primäre `markdown_converter`-Aufgabe: Markdown-Datei hochladen → PDF generieren → herunterladen oder zur Library speichern. Nicht so daily-usage-zentral wie library_detail oder podcast-flow, aber substantieller User-Workflow.

**Out-of-scope**:
- Heuristik-Review (Stufe 2) — eigener Folge-Sprint `F5-REVIEW`.
- Patterns + Microcopy (Stufe 3) — eigener Folge-Sprint `F5-PATTERNS`.
- Implementation — eigene Folge-Sprints `F5-IMPL-*`.
- Code-Änderungen jeglicher Art. Bugs als „separater Befund" dokumentieren, nicht fixen — `feedback_no_silent_fixes.md`.
- Andere Features (`library` List-View, `mermaid_converter`, `login`) — eigene Folge-Wellen.
- Bereits durch frühere Sprints erledigte Items (siehe Vorbedingung) — nicht als Inventur-Befund aufnehmen, höchstens im Header-Verweis.

---

## Phase 1 — Inventur (read-only)

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Methodik-Vorlagen lesen**: F-1.1 als primäre Strukturvorlage (Schwester-Feature). F-3.1 + F-4.1 für Live-Walkthrough-Lücken-Sektion-Pattern. Im F-1.1 die 14 Patterns + 24 Elemente kennen, weil viele direkt übertragbar sind.
3. **Touch-Pfade lesen** (Pflicht-Reihenfolge):
   - `templates/markdown_converter.html` zuerst — DOM-Struktur, Form-Felder, Button-Labels, Drop-Zone-Markup, PDF-Iframe-Mountpoint.
   - `static/js/markdown_converter.js` zweitens — Event-Handler, AJAX-Calls, State-Mutations, Helper-Reuse aus `_utils.js`.
   - `app_pkg/markdown.py` drittens — Route-Handler-Signature, ACCEPTED_EXTENSIONS, PDF-Generation-Pipeline, Save-to-Library-Pfad.
   - `static/css/style.css` letztens — relevante `c-`-Prefix-Klassen aus dem Template.

**Inventur-Aufgabe** (kein Bewerten, nur Mapping):

Für jedes interaktive Element kartieren:

| Spalte | Was rein |
|--------|----------|
| `#` | Lauf-Nummer |
| `Element-Typ` | Button / Input / Drop-Zone / Iframe / Loading-Indicator / etc. |
| `Label` (im Template) | Text wie er im DOM steht (deutsch oder englisch markieren) |
| `Aktion` | Was passiert (Endpoint? State-Change? Navigation? PDF-Render?) |
| `Vorhandene States` | default, hover, focus, disabled, loading, error, success, empty |
| `Fehlende States` | dieselbe Liste — die nicht belegt sind |
| `F-1-Korrespondenz` | (NEU für F-5.1, sehr wichtig) Pattern-Nummer aus F-1 das hier direkt anwendbar wäre, oder „F-1-Pattern X zu prüfen ob übertragbar". Falls kein F-1-Korrespondent existiert: „markdown-spezifisch". |
| `Notizen` | Auffälligkeiten, Code↔live-Divergenzen, mögliche Bugs (als „Befund Nr. X") |

**Ergänzungs-Sektionen**:

- **Code↔live-Divergenzen**: Stellen wo Template, JS und Route-Handler nicht zusammenpassen.
- **F-1-Korrespondenz-Sektion** (NEU für F-5.1): am Doc-Anfang oder -Ende eine Übersicht welche F-1-Patterns direkt übertragbar sind, welche teil-übertragbar mit Modifikation, welche nicht anwendbar (markdown-spezifisch). Hilft F-5.2/F-5.3 die Cross-Feature-H4-Quote schnell zu greifen.
- **Markdown-spezifische States**: PDF-Iframe-Render-Lifecycle (loading → rendered → error), Code-Highlighting-Output (Pygments-CSS-Embed), eventuelle Multi-File-Zip-Output (falls vorhanden — checken).
- **Separate Befunde** (nummeriert): Bugs, Inkonsistenzen, Out-of-Scope-Beobachtungen. Jeder mit Code-Anker, Beschreibung, Disposition-Vorschlag.
- **Live-Walkthrough-Lücken**: Code-only-Inventur ist 80-90% des Werts, States die nur runtime sichtbar sind als „Code-deduced, nicht live verifiziert" markieren.
- **Bereits-durch-frühere-Sprints-erfüllt-Sektion**: Header-Liste (F-006, F-002, F-007, F-008, F-011) als Verweis dass diese Items aus dem Inventur-Scope ausgenommen sind.

**Output-Doc**: `docs/ui_inventory_markdown_converter_2026-05.md`. Struktur 1:1 wie F-1.1 / F-3.1 / F-4.1 + neue F-1-Korrespondenz-Sektion.

Nach Phase 1: STOP — Bericht. Element-Anzahl, F-1-Korrespondenz-Verteilung (direkt übertragbar / teil-übertragbar / markdown-spezifisch), fehlende-States-Anzahl, Divergenzen-Anzahl, Befunde-Anzahl mit Disposition-Verteilung, Live-Walkthrough-Lücken-Status.

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Inventur-Doc nochmal und prüft:

1. **Vollständigkeit**: jedes interaktive Element aus dem Template ist in der Tabelle.
2. **F-1-Korrespondenz-Konsistenz**: jede Element-Zeile hat einen F-1-Korrespondenz-Eintrag (Pattern-Nummer oder „markdown-spezifisch"). Übersichts-Sektion und Element-Zeilen-Einträge stimmen überein.
3. **Bereits-durch-frühere-Sprints-erfüllt-Disziplin**: F-006/F-002/F-007/F-008/F-011-Aspekte sind nicht als Befunde aufgenommen, nur im Header-Verweis erwähnt.
4. **Konsistenz**: jeder Befund hat Code-Anker.
5. **Disziplin**: kein Pattern-Vorschlag, keine Severity-Bewertung, kein Bug-Fix.
6. **Helper-Reuse-Spuren**: in den Notizen markieren wo `markdown_converter.js` schon `_utils.js`-Helper nutzt vs. wo Inline-Code dupliziert.

Nach Phase 2: STOP — Bericht.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-5.1 / Stufe 1: UI inventory of markdown_converter".
- Body: Statistik (Element-Anzahl, F-1-Korrespondenz-Verteilung, fehlende-States, Divergenzen, Befunde, Live-Walkthrough-Lücken-Status).
- Branch: direkt auf `main` ist OK.
- `git push origin main`. Wenn Auto-Mode-Classifier blockt: Master pushed von Hand.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

**Inventur-spezifischer STOP-Trigger**: bei akut wirkendem Bug (Crash-Pfad, Datenverlust-Risiko) im Bericht „akut" flaggen — Master entscheidet ob Hot-Fix-Sprint vorgezogen wird.

---

## Größe

**S** — eine Output-Datei (`docs/ui_inventory_markdown_converter_2026-05.md`), Code-Reading + Mapping, kein Code-Touch. F-5 ist Schwester-zu-F-1 und sollte mit der F-1-Vorlage effizient laufen.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Code-Reading von `app_pkg/markdown.py` Auffälligkeiten in **anderen** Routen auffallen (z.B. die `/api/notion/...`-Endpoints): kurz im Bericht aufzählen, **nicht** in die Inventur-Doc.
- Englische UI-Strings: erwähnen falls welche durchrutschen, **nicht** als separater Befund — DE-Pass kommt in F-5.3.
- Wenn das PDF-Iframe-Render-Verhalten Sub-Thread-spezifisches Browser-Wissen braucht: als „Live-Walkthrough-Lücke" markieren.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F5-PICK ☑ done 2026-05-XX → commit `<hash>`. Inventur-Doc unter `docs/ui_inventory_markdown_converter_2026-05.md`. <Element-Anzahl> Elemente, F-1-Korrespondenz <X> direkt übertragbar / <Y> teil-übertragbar / <Z> markdown-spezifisch, <fehlende> States, <Divergenzen> Divergenzen, <Befunde> Befunde. Verbleibende Sequenz: F5-REVIEW → F5-PATTERNS → F5-IMPL → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F-N…" raus → Sektion „1. F5-REVIEW" rückt auf Position 1 mit Verweis auf die neue Inventur-Doc.
- **Memory**: nichts erwartet — Inventur-Methodik ist seit F-1.1/F-2.1/F-3.1/F-4.1 etabliert. F-1-Korrespondenz-Sektion ist neue Output-Variante aber keine Methodik-Lehre.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Inventur-Methodik klar, F-1-Korrespondenz-Erwartung im Sprint-Prompt verankert, Schwester-Feature-Effizienz erwartet.)_
