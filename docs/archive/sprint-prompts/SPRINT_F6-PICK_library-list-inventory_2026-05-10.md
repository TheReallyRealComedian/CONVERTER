# Sprint F6-PICK — F-6.1 UX-Inventur `library` List-View

**Datum**: 2026-05-10

**Ziel**: Stufe 1 (Inventur) der dreistufigen UX-Cascade-Methodik für die **`library` List-View** (die Übersichts-Seite mit Liste/Filter/Tags/Suche, von der aus zu `library_detail` navigiert wird). **Geschwister-Feature zu `library_detail`** (F-3-Welle, schon abgeschlossen): selbe Datenklasse, andere View — List statt Detail. Erwartung ist mittlere bis hohe Cross-Feature-H4-Quote zu `library_detail` und allen anderen Features (Helper-Reuse aus `_utils.js`). **Kein Bewerten, kein Bug-Fix, kein Pattern-Vorschlag** — das kommt in F-6.2 / F-6.3.

**Vorbedingung**:
- Pytest 66/66 grün im Container. Letzter Code-Touch: F5-IMPL (commits `07e9aa6` + `9e6999c` + `ff22840`, 2026-05-10). F-5 strukturell abgeschlossen für `markdown_converter`.
- Touch-Pfade des Features:
  - **Template**: [templates/library.html](templates/library.html) — die List-View-Seite (zu unterscheiden von `library_detail.html` aus F-3).
  - **JS**: [static/js/library.js](static/js/library.js) — eigenes File für die List-View (zu unterscheiden von `library_detail.js`).
  - **Route-Modul**: [app_pkg/library.py](app_pkg/library.py) — `library_view`-Endpoint plus `api_*`-Endpoints für CRUD (manche schon durch F3-IMPL getouched für library_detail).
  - **Shared**: [static/css/style.css](static/css/style.css), [static/js/_utils.js](static/js/_utils.js) (`showAlert`, `showToast`, `formatFileSize`, `safeJSON`, `formatDatetimeLocalNow`, **`saveViewState`/`loadViewState`** [neu aus F5-IMPL für View-State-Persistenz], `.sr-only`-Utility), [app_pkg/__init__.py](app_pkg/__init__.py) (`file_size`-Jinja-Filter aus F3-IMPL).
- **Was schon durch frühere Sprints erledigt ist** (NICHT als Befund aufnehmen):
  - **F-005 Path-Traversal-Guard** durch SEC-Sprint (`api_*`-Endpoints).
  - **F-007 secure_filename(None)-Guard** durch HYG-Sprint.
  - **F-008 Logging-Sites mit `exc_info=True`** durch HYG-Sprint.
  - **F-011 `@require_service`-Decorator** für relevante Endpoints durch HYG-Sprint.
  - **F-013 Input-Allowlists** für relevante Endpoints durch SEC-Sprint.
  - **F-3-Welle**: library_detail vollständig (PICK→REVIEW→PATTERNS→IMPL). Cross-Reference für library-List-View, aber **kein library_detail-Touch in F-6.1 Inventur** (nur als Helper-Reuse-Quelle und Code-Anker-Referenz nutzen).
  - **F-006 markdown Backend-Whitelist** in markdown_converter (kein library-Bezug).
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - **F-3.1 Inventur**: [docs/ui_inventory_library_detail_2026-05.md](docs/ui_inventory_library_detail_2026-05.md) — **primäre Geschwister-Feature-Vorlage**, 21 Elemente, Live-Walkthrough-Lücken-Sektion. Selbe Datenklasse, viele Helper überlappen.
  - F-1.1 Inventur: [docs/ui_inventory_document_converter_2026-05.md](docs/ui_inventory_document_converter_2026-05.md).
  - F-2.1 Inventur: [docs/ui_inventory_audio_converter_2026-05.md](docs/ui_inventory_audio_converter_2026-05.md).
  - F-4.1 Inventur: [docs/ui_inventory_podcast_flow_2026-05.md](docs/ui_inventory_podcast_flow_2026-05.md).
  - F-5.1 Inventur: [docs/ui_inventory_markdown_converter_2026-05.md](docs/ui_inventory_markdown_converter_2026-05.md) — jüngste Vorlage mit F-1-Korrespondenz-Spalte (NEUe Spalte für Schwester-Feature-Hebel — analog hier: F-3-Korrespondenz-Spalte für Geschwister-Feature).
- **Geschwister-Feature-Erwartung** (wichtig für Methodik):
  - **`library` (List) und `library_detail` (Detail)** sind verschiedene Views auf dieselbe `ConversionHistory`-Datenklasse. List-View hat Sortierung/Filter/Suche/Bulk-Aktionen die Detail-View nicht hat; Detail-View hat Edit-Pfade die List-View nicht hat.
  - Helper-Reuse-Erwartung: `formatFileSize` / `formatDatetimeLocalNow` / `file_size`-Jinja-Filter / `showAlert` / `safeJSON` / Sidebar-Active-State analog F-3-IMPL.
  - F-3-Korrespondenz-Spalte (NEU für F-6.1): pro Element notieren ob ein F-3-Pattern (P1–P14, P15) hier wieder anwendbar ist, teil-anwendbar, oder nicht-anwendbar (List-spezifisch).
- **Cross-Feature-H4-Erwartung** (mittlere Quote):
  - **List-spezifische Findings ohne F-3-Korrespondent**: Sortierung/Filter/Suche-State, Bulk-Selektion, Tag-Filter-Pattern, Pagination (falls vorhanden).
  - **F-3-übertragbare Findings**: Helper-Reuse, DE-Microcopy, a11y-Annotations, file_size-Filter-Konvergenz, Datum-Format-Lokalisierung.
  - **Cross-Feature-Helper-Reuse ohne F-3-Bezug**: `saveViewState`/`loadViewState` aus F-5-IMPL für View-State-Persistenz (Sortierung, Filter, Suche-State) — **explizit prüfen** ob library-List-View einen View-State hat der persistiert werden sollte (zweite Call-Site für den Helper).
- **Kontext für die Methodik**: Single-User-App, LAN-only, login-protected. **Primäre `library` List-View-Aufgabe**: Konvertierungs-Historie übersichtlich anzeigen, navigieren zu Detail, eventuell filtern/suchen/bulk-managen. **Memory-Gewicht**: laut [project_readwise_replacement.md](file:///Users/olivergluth/.claude/projects/-Volumes-MintHome-CODE-CONVERTER/memory/project_readwise_replacement.md) ist Library zentraler Reader-Ersatz für Readwise — daily-usage-Schmerz hoch wenn UX-Reibung in der List-View liegt.

**Out-of-scope**:
- Heuristik-Review (Stufe 2) — eigener Folge-Sprint `F6-REVIEW`.
- Patterns + Microcopy (Stufe 3) — eigener Folge-Sprint `F6-PATTERNS`.
- Implementation — eigene Folge-Sprints `F6-IMPL-*`.
- Code-Änderungen jeglicher Art. Bugs als „separater Befund" dokumentieren, nicht fixen — `feedback_no_silent_fixes.md`.
- **`library_detail`-Code-Refactors** — F-3 ist abgeschlossen für die Detail-View; library-List-View ist eigenes Feature. Wenn beim Code-Reading library_detail-Auffälligkeiten erscheinen (z.B. weitere F-3.2 BT-Kandidaten): nur kurz im Bericht erwähnen, nicht in die Inventur-Doc.
- **F-3.2 Bug-Tickets BT7 + BT8** (textarea-escape, window.open-noopener) — gehören zu library_detail, nicht zu library-List-View. Aus dem Scope auch wenn beim library-Inventur-Code-Reading die Stellen vorbeikommen.
- Andere Features (`mermaid_converter`, `login`) — eigene Folge-Wellen.
- Bereits durch frühere Sprints erledigte Items (siehe Vorbedingung) — nicht als Inventur-Befund aufnehmen, höchstens im Header-Verweis.

---

## Master-Annotation (vorab eingebettet)

### 1. Geschwister-Feature-Hebel zu F-3 (analog Schwester-Feature-Hebel zu F-1 in F-5.1)

`library_detail` (F-3) und `library` (List-View, F-6) sind **Geschwister-Features** auf derselben Datenklasse. Aber: andere View-Klasse, andere Aufgaben. Cross-Feature-H4 ist erwartet **mittel** (~30–50%) — höher als F-4.2's 0% weil Helper-Reuse-Konvergenz auf etablierte Helper, niedriger als F-5.2's 86% weil keine direkte Pattern-Übertragbarkeit (List-View hat keine Notion-Form, keine Auto-Save-Field-Updates, kein Tag-Chip-Editor).

**Methodik-Konsequenz**:
- **F-3-Korrespondenz-Spalte** in der Inventur-Tabelle (analog F-1-Korrespondenz aus F-5.1): pro Element notieren ob ein F-3-Pattern (P1 Auto-Save / P3 Delete / P4 Notion / P5 Datum / P6 DE-Microcopy / P7 showAlert-Reuse / P8 Toast-Level / P9 Tag-Chips / P10 a11y / P11 Sidebar-Active / P12 file_size / P13 Page-Title / P14 safeJSON-Login-Redirect / P15 Banner-Mountpoint) hier anwendbar ist, teil-anwendbar (mit Liste-spezifischer Modifikation), oder nicht-anwendbar (List-spezifisch — z.B. Sortier-Header, Filter-Pills, Bulk-Auswahl).
- **F-3-Korrespondenz-Übersicht** am Doc-Anfang oder -Ende: Gesamt-Quote der direkt-anwendbaren / teil-anwendbaren / nicht-anwendbaren F-3-Patterns.

### 2. Helper-Reuse-Spuren-Sektion mit Fokus auf View-State-Helper

Wie in F-5.1 etabliert: **Helper-Reuse-Spuren-Sektion** dokumentiert pro etabliertem Helper aus `_utils.js`, ob `library.js` ihn nutzt (✓), nicht nutzt mit Inline-Duplikat (✗ + Code-Anker), oder n/a (kein Anwendungs-Anlass).

**Spezifischer Master-Fokus für F-6.1**:
- **`saveViewState`/`loadViewState` aus F-5-IMPL** ist die **zweite Call-Site-Frage**: hat library-List-View einen View-State (Sortierung / Filter / Suche / aktiver Tag) der über Reload erhalten bleiben sollte? Wenn ja: H4-Konvergenz-Befund (Helper exists, library nutzt es nicht). Wenn nein: in der Helper-Reuse-Spuren-Sektion als „n/a — kein persistierungs-würdiger State" mit Begründung.
- **`confirmInPlace` aus F-4-IMPL** ist die **zweite Call-Site-Frage** für Bulk-Delete: hat library-List-View einen Bulk-Delete-Pfad? Wenn ja: H4-Konvergenz-Befund mit Helper-Extraktion-Vorschlag-Trigger (siehe BACKLOG-Disposition für `confirmInPlace`). Wenn nein: in Helper-Reuse-Spuren-Sektion als „n/a".

### 3. Cross-Feature-Konvergenz mit allen anderen Features

Helper-Reuse-Spuren-Sektion sollte **alle Helper aus `_utils.js`** abdecken plus den `file_size`-Jinja-Filter aus `app_pkg/__init__.py` (F3-IMPL). DE-Microcopy-Pass-Erwartung: 2 EN-Strings in `library.js` sind aus dem BACKLOG-P3-Reminder bekannt — als Befund aufnehmen mit Disposition „F6-PATTERNS DE-Microcopy-Folde", nicht im F6-PICK fixen.

### 4. List-spezifische States als eigene Sub-Sektion

Analog F-4.1 hatte „Async-Pipeline-Mapping" und F-5.1 hatte „Reader-Mode-Cluster": F-6.1 soll eine **List-View-States-Sub-Sektion** haben mit Mapping über:
- **Sortierung-State** (welche Spalte ist sortiert, asc/desc, persistent oder ephemeral?).
- **Filter-State** (aktive Filter-Pills, Tag-Filter, Datum-Range, ...).
- **Suche-State** (Such-Query-Input, Live-Search vs. Submit, Debouncing).
- **Bulk-Selektion-State** (Checkboxes, Select-All, Selected-Count, Bulk-Aktion-Toolbar).
- **Pagination-State** (falls vorhanden — Page-Index, Items-per-Page, Infinite-Scroll vs. Numeric-Pages).
- **Empty-State** (keine Konvertierungen, gefilterte Liste leer).

Pro List-View-State-Klasse: Code-Anker, vorhandene-States, fehlende-States, Helper-Reuse-Status.

---

## Phase 1 — Inventur (read-only)

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Methodik-Vorlagen lesen**: F-3.1 als primäre Geschwister-Vorlage. F-5.1 für Korrespondenz-Spalten-Pattern (analog hier F-3-Korrespondenz). F-1.1 / F-2.1 / F-4.1 für allgemeine Inventur-Methodik.
3. **F-3.1 + F-3.2 + F-3.3 + F-3-IMPL-Erledigt-Eintrag** (in BACKLOG.md) lesen — die F-3-Pattern-IDs (P1–P15) müssen für die F-3-Korrespondenz-Spalte greifbar sein.
4. **Touch-Pfade lesen** (Pflicht-Reihenfolge):
   - `templates/library.html` zuerst — DOM-Struktur, Liste-Markup, Sortier-Header, Filter-Pills, Suche-Input, Bulk-Toolbar (falls), Empty-State, Pagination.
   - `static/js/library.js` zweitens — Event-Handler für Sortierung/Filter/Suche/Bulk, AJAX-Calls (falls), State-Mutations, Helper-Reuse aus `_utils.js`.
   - `app_pkg/library.py` drittens — `library_view`-Endpoint (Render mit Filter/Sort), `api_*`-Endpoints (Bulk-Delete, Bulk-Update, …, falls).
   - `static/css/style.css` letztens — relevante `c-`-Prefix-Klassen aus dem Template, Liste-Styling, Filter-Pill-Styling.

**Inventur-Aufgabe** (kein Bewerten, nur Mapping):

Für jedes interaktive Element kartieren:

| Spalte | Was rein |
|--------|----------|
| `#` | Lauf-Nummer |
| `Element-Typ` | Button / Sortier-Header / Filter-Pill / Suche-Input / Checkbox / Tag-Chip / Liste-Row / Pagination-Control / Bulk-Toolbar / etc. |
| `Label` (im Template) | Text wie er im DOM steht (deutsch oder englisch markieren — für DE-Pass-Vorbereitung in F6-PATTERNS) |
| `Aktion` | Was passiert (Endpoint? URL-Param-Update? Filter-Apply? Navigation? Bulk-Op?) |
| `Vorhandene States` | default, hover, focus, disabled, loading, active, selected, error, empty |
| `Fehlende States` | dieselbe Liste — die nicht belegt sind |
| `F-3-Korrespondenz` | Pattern-Nummer aus F-3 (P1–P15) das hier direkt anwendbar wäre, oder „F-3-Pattern X zu prüfen ob übertragbar" oder „list-spezifisch" |
| `Helper-Reuse-Status` | welche `_utils.js`-Helper genutzt (✓), welche fehlen wo Inline-Code dupliziert (✗) |
| `Notizen` | Auffälligkeiten, Code↔live-Divergenzen, mögliche Bugs (als „Befund Nr. X") |

**Ergänzungs-Sektionen**:

- **Code↔live-Divergenzen**: Stellen wo Template, JS und Route-Handler nicht zusammenpassen.
- **F-3-Korrespondenz-Sektion**: am Doc-Anfang oder -Ende eine Übersicht welche F-3-Patterns direkt übertragbar sind / teil-übertragbar / nicht-anwendbar (list-spezifisch). Hilft F-6.2/F-6.3 die Cross-Feature-H4-Quote schnell zu greifen.
- **List-View-States-Sub-Sektion** (Master-Annotation 4): Mapping über Sortierung / Filter / Suche / Bulk-Selektion / Pagination / Empty-State.
- **Helper-Reuse-Spuren-Sektion**: pro `_utils.js`-Helper plus `file_size`-Jinja-Filter Status. Master-Fokus auf `saveViewState`/`loadViewState` (zweite Call-Site?) und `confirmInPlace` (Bulk-Delete?).
- **Separate Befunde** (nummeriert): Bugs, Inkonsistenzen, Out-of-Scope-Beobachtungen. Jeder mit Code-Anker, Beschreibung, Disposition-Vorschlag. **2 EN-Strings in `library.js` als ein Befund** mit Disposition „F6-PATTERNS DE-Microcopy-Folde".
- **Live-Walkthrough-Lücken**: States die nur runtime sichtbar sind als „Code-deduced, nicht live verifiziert" markieren.
- **Bereits-durch-frühere-Sprints-erfüllt-Sektion**: Header-Liste (F-005 / F-007 / F-008 / F-011 / F-013 / F-3-Welle) als Verweis dass diese Items aus dem Inventur-Scope ausgenommen sind.

**Output-Doc**: `docs/ui_inventory_library_list_2026-05.md`. Struktur 1:1 wie F-3.1 / F-5.1 + F-3-Korrespondenz-Spalte + List-View-States-Sub-Sektion + Helper-Reuse-Spuren-Sektion.

Nach Phase 1: STOP — Bericht. Element-Anzahl, F-3-Korrespondenz-Verteilung (direkt übertragbar / teil-übertragbar / list-spezifisch), List-View-States-Coverage, Helper-Reuse-Status (besonders `saveViewState`/`loadViewState`-Call-Site-Frage und `confirmInPlace`-Bulk-Delete-Frage), fehlende-States-Anzahl, Divergenzen-Anzahl, Befunde-Anzahl mit Disposition-Verteilung, Live-Walkthrough-Lücken-Status. Plus: ob Sub-Thread einen akut-flag-Kandidaten gefunden hat (Crash-Pfad / Datenverlust-Risiko) — siehe Stop-Regel.

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Inventur-Doc nochmal und prüft:

1. **Vollständigkeit**: jedes interaktive Element aus dem Template ist in der Tabelle.
2. **F-3-Korrespondenz-Konsistenz**: jede Element-Zeile hat einen F-3-Korrespondenz-Eintrag (Pattern-Nummer oder „list-spezifisch"). Übersichts-Sektion und Element-Zeilen-Einträge stimmen überein.
3. **List-View-States-Sub-Sektion**: alle relevanten States kartiert (Sortierung / Filter / Suche / Bulk / Pagination / Empty), Code-Anker pro State.
4. **Helper-Reuse-Spuren-Sektion**: pro `_utils.js`-Helper plus `file_size`-Jinja-Filter Status mit Code-Anker für Inline-Duplikate. Master-Fokus-Helper (`saveViewState`/`loadViewState`, `confirmInPlace`) explizit beantwortet (genutzt / Call-Site-Anlass-vorhanden / n/a-mit-Begründung).
5. **Bereits-durch-frühere-Sprints-erfüllt-Disziplin**: F-005/F-007/F-008/F-011/F-013/F-3-Welle-Aspekte sind nicht als Befunde aufgenommen, nur im Header-Verweis erwähnt.
6. **Konsistenz**: jeder Befund hat Code-Anker.
7. **Disziplin**: kein Pattern-Vorschlag, keine Severity-Bewertung, kein Bug-Fix.

Nach Phase 2: STOP — Bericht.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-6.1 / Stufe 1: UI inventory of library list view".
- Body: Statistik (Element-Anzahl, F-3-Korrespondenz-Verteilung, List-View-States-Coverage, Helper-Reuse-Status, fehlende-States, Divergenzen, Befunde, Live-Walkthrough-Lücken-Status).
- Branch: direkt auf `main` ist OK.
- `git push origin main`. Wenn Auto-Mode-Classifier blockt: Master pushed von Hand. (Memory `feedback_push_is_normal.md`.)

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

**Inventur-spezifischer STOP-Trigger**: bei akut wirkendem Bug (Crash-Pfad, Datenverlust-Risiko) im Bericht „akut" flaggen — Master entscheidet ob Hot-Fix-Sprint vorgezogen wird oder gefolded.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S–M** — eine Output-Datei (`docs/ui_inventory_library_list_2026-05.md`), Code-Reading + Mapping, kein Code-Touch. M-Anteil weil List-View komplexer als Detail-View sein kann (Sortierung + Filter + Suche + Bulk + Pagination), Geschwister-Feature-Effizienz aus F-3-Code-Reading senkt aber den Aufwand. Erwartete Element-Anzahl 20-40 (Spannweite F-3.1's 21 bis F-5.1's 32).

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Code-Reading von `app_pkg/library.py` Auffälligkeiten in den library_detail-Pfaden auffallen die F-3 nicht abgedeckt hat: kurz im Bericht aufzählen, **nicht** in die Inventur-Doc (sind library_detail, F-3 ist abgeschlossen, gehören zu Sammel-Bug-Pass).
- Englische UI-Strings in `library.js`: nicht als separater Befund pro String, sondern als ein Sammel-Befund mit Anzahl + Code-Ankern + Disposition „F6-PATTERNS DE-Microcopy-Folde" (analog BACKLOG-P3-Reminder).
- Wenn `saveViewState`/`loadViewState` als zweite Call-Site offensichtlich passt (z.B. List-View hat aktive Filter im URL-Param aber nicht im localStorage): als H4-Konvergenz-Befund aufnehmen mit Code-Anker.
- Wenn `confirmInPlace` als zweite Call-Site offensichtlich passt (z.B. List-View hat Bulk-Delete-Button mit konfirmation-`confirm()`): als H4-Konvergenz-Befund mit Helper-Extraktion-Trigger-Hinweis.
- Wenn beim Code-Reading neue F-3.2-BT-Kandidaten (in library_detail-Code) vorbeikommen: kurz im Bericht erwähnen, **nicht** in die F-6.1-Inventur-Doc — gehören zu library_detail.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F6-PICK ☑ done 2026-05-XX → commit `<hash>`. Inventur-Doc unter `docs/ui_inventory_library_list_2026-05.md`. <Element-Anzahl> Elemente, F-3-Korrespondenz <X> direkt übertragbar / <Y> teil-übertragbar / <Z> list-spezifisch, <List-States> List-View-States kartiert, Helper-Reuse-Status (`saveViewState` zweite Call-Site: ja/nein, `confirmInPlace` zweite Call-Site: ja/nein), <fehlende> States, <Divergenzen> Divergenzen, <Befunde> Befunde inkl. EN-Strings-Sammel-Befund. Verbleibende Sequenz: F6-REVIEW → F6-PATTERNS → F6-IMPL → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F6-PICK" raus → Erledigt-Liste; Master fügt F6-REVIEW als Position 1 beim nächsten Dispatch hinzu — Sub-Thread fügt es **nicht** selbst hinzu (Master-Edit-Zone).
- **Memory**: nichts erwartet — Inventur-Methodik etabliert. F-3-Korrespondenz-Sektion ist analog zur F-1-Korrespondenz-Sektion in F-5.1 (Geschwister-Feature-Hebel statt Schwester-Feature-Hebel) — schon als Methodik-Variante eingeführt, keine neue Lehre. **Falls** beim Apply auffällt dass die Helper-Vorschlags-Disziplin („zweite Call-Site"-Regel) eine bessere Präzision braucht: defensive `feedback_*.md`.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Inventur-Methodik klar, F-3-Korrespondenz-Erwartung im Sprint-Prompt verankert, Geschwister-Feature-Hebel + Helper-Reuse-Master-Fokus eingebettet.)_
