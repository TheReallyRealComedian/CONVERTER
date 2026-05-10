# Sprint F5-IMPL — F-5 Implementation `markdown_converter` (alle 13 Patterns, 3 Sub-Batches)

**Datum**: 2026-05-10

**Ziel**: Alle 13 Patterns aus F5-PATTERNS in einem Sprint implementieren, intern strukturiert als drei Sub-Batches (Cross-Feature-H4-Konvergenz → Reader-Mode-State → Error-Recovery+Async-Pre-Check). Dies ist ein **bewusst aggressiv geschnittener Sprint** — Master hat 1-Sprint statt 2-Sprint-Split gewählt, gestützt auf F-3-IMPL-Lehre (15 Patterns in 3 Sub-Batches funktionierte) und F-4-IMPL-B-Lehre (10 Patterns in einem Sweep ohne Split funktionierte bei verkoppelten Touches). Aufwand-Verteilung sehr XS-lastig (XS=8 / S=4 / M=1 / L=0) wegen Schwester-Feature-Übernahme aus F-1.3. Die Sub-Batch-Mechanik dient als Schutz vor Überforderung — nach jedem Sub-Batch ein STOP-Punkt mit Bericht. **Ca/Cb-Split-Fallback** für Sub-Batch C wenn Sub-Thread nach C-Pre-Flight Verkopplung als zu groß einschätzt.

**Vorbedingung**:
- Pytest 65/65 grün auf `main`. Letzter Code-Touch: F5-PATTERNS (commit `06f74b5`, 2026-05-10).
- **Eingabe**: [docs/ui_patterns_markdown_converter_2026-05.md](docs/ui_patterns_markdown_converter_2026-05.md). Sub-Thread liest **vor jedem Sub-Batch** die zugehörigen Pattern-Blöcke nochmal.
  - 13 Pattern-IDs P1–P13.
  - 8 Patterns mit `🔥 Smoke-Pflicht in F5-IMPL`-Tag: P5, P6, P7, P8, P9, P10, P11, P13 (alle 8 ⚠️ code-only-Findings aus F-5.2 adressiert).
  - 2 Helper-Vorschläge in dedizierter Doc-Sektion: `saveViewState/loadViewState` (für P5), `attachAutoDismissToServerBanners` (für P2).
  - Reader-Mode-Default-Wahl aus F-5.3 Master-Annotation 4 in Patterns P5/P6/P7 verankert (gemeinsamer `localStorage.markdown.viewState`-Key, `activeElement.tagName`-Esc-Handler-Check, Reader-Dark-lokaler-Scope-Sieg).
- **Methodik-Vorlagen** (Multi-Pattern-Sprints):
  - F-3-IMPL commits `843574b` / `40dd02e` / `b3e666a` (15 Patterns in 3 Sub-Batches Foundation → Notion → Polish, drei separate Commits via temp-Backup→revert→stepwise-replay-Mechanik wegen Cross-Sub-Batch-Hunks).
  - F-4-IMPL-B commit `3ef7f9e` (10 Patterns in einem Sweep ohne Sub-Batch-Split, Backend-zuerst → CSS+HTML → Frontend-JS → Tests Apply-Reihenfolge).
  - F-1 Cluster A-E + Polish als Vorlagen für isolierte Pattern-Cluster.
  - F-1.3 als **Pattern-Übernahme-Quelle** für Sub-Batch A — die F-1-Patterns die Sub-Batch-A-Patterns korrespondieren sind dort schon ausgearbeitet (Code-Strukturen, Microcopy-Strings, Helper-Reuse).
- **Test-Coverage**: aktuelle Suite 65/65 grün. Erwartete neue Tests:
  - 1-2 für P8 (PDF-Gen-Error-Recovery — Backend-Test für `redirect()`+`flash()`-Pfad auf Error-Branch in [app_pkg/markdown.py:246](app_pkg/markdown.py#L246)).
  - 0-1 für P9 (Sample-Merge-Template-Render — vermutlich Frontend-only, ggf. Backend-Render-Test für korrektes Empty-Markdown-State).
  - 0-1 für P12 (Empty-Submit-Pre-Check — Backend-Allowlist-Test ggf. analog F-013).
  - Erwartete Final-Anzahl: **66–69 Tests grün**.
- **Memory-Layer-Pflicht-Lese**: `feedback_no_silent_fixes.md` (Bugs während Implementation als separates Bug-Ticket dokumentieren), `feedback_pragmatic_merge.md` (Risiko-Kalibrierung), `feedback_push_is_normal.md` (Push direkt nach Commit OK), `reference_converter_dep_bump_constraints.md` (Container-Smoke-Disziplin, pytest im Container nicht lokal).

**Out-of-scope**:
- Weitere F-N-Wellen für andere Features (`library` List-View, `mermaid_converter`, `login`).
- WAVE-CLOSE.
- **BT4 (Inline-`<style>` in Template)**: Master-Empfehlung **draußen lassen** — reine HTML-Reorganisation ohne UX-H-Aspekt, separater Hygiene-Pass. Sub-Thread folded BT4 **nicht** mit auch wenn nahegelegen.
- F-1-Cross-Feature-Backports — F-5.3 Patterns sind `markdown_converter`-spezifische Ableitungen aus F-1.3, keine F-1-Code-Refactors.

---

## Master-Annotation (vorab eingebettet)

### 1. Sub-Batch-Strategie 3 Sub-Batches A/B/C, Ca/Cb-Fallback

| Sub-Batch | Patterns | Anzahl | Smoke-Pflicht | Begründung |
|-----------|----------|--------|---------------|------------|
| **A — Cross-Feature-H4-Konvergenz zu F-1** | P1, P2, P3, P4 | 4 | — | Schwester-Feature-Übernahme aus F-1.3, helper-reuse-dominiert (`showAlert`, `safeJSON`, `formatFileSize`, a11y-Annotations). XS-lastig. Kein Smoke-Pflicht — F-1-Patterns sind dort schon live verifiziert, hier 1:1-Übernahme mit Code-Anker-Anpassung. |
| **B — Reader-Mode-State und Visual-Layout** | P5, P6, P7 | 3 | P5, P6, P7 | Stark verkoppelte Reader-Mode-Logik mit gemeinsamer `localStorage.markdown.viewState`-Mechanik. P5 ist M-Aufwand-Anchor, P6 + P7 sind XS+S. Holistic-Apply empfohlen. **Reader-Mode-Default-Wahl** aus F-5.3 Master-Annotation 4 ist im Pattern verankert. |
| **C — Error-Recovery + Async-Pre-Check** | P8, P9, P10, P11, P12, P13 | 6 | P8, P9, P10, P11, P13 | Mischung aus Backend (P8 Flash+Redirect, P9 Template-Bug, ggf. P12 Pre-Check), Frontend (P10, P11, P13), Template (P9). 5 von 6 Smoke-Pflicht. **Ca/Cb-Split-Fallback** wenn Sub-Thread Verkopplung als zu groß einschätzt: Ca = P8 + P9 (isolierte Quick-Wins) / Cb = P10 + P11 + P12 + P13 (Frontend-Cluster). |

**Pflicht-Reihenfolge**: A → B → C, **ohne Auslassen**. STOP-Punkt mit Bericht nach jedem Sub-Batch — Master kann zwischen Sub-Batches korrigieren oder Sprint abbrechen.

**Holistic vs. sequenziell pro Sub-Batch**: Sub-Thread entscheidet pragmatisch.
- **Sub-Batch A (4 Patterns, helper-reuse-Cluster)**: holistic empfohlen (Helper-Imports, alert()-Replacements, a11y-Annotations sind verkoppelte Touches).
- **Sub-Batch B (3 Patterns, Reader-Mode-Cluster)**: holistic empfohlen (gemeinsamer `localStorage`-Key, Init-Sequenz).
- **Sub-Batch C (6 Patterns, gemischt)**: sequenziell empfohlen mit P8 zuerst als Quick-Win-Trigger (Score 15.0, Befund-16-Folding), dann P9 (BT2-Template-Bug isoliert), dann P10–P13.

### 2. Apply-Reihenfolge-Empfehlung innerhalb Sub-Batch C

Sub-Thread folgt **default**, kann pragmatisch abweichen mit Begründung im Bericht:

1. **P8 zuerst** — höchster Quick-Win-Score 15.0, isolierter Backend-Touch (1-2 Zeilen `redirect()`+`flash()` in [app_pkg/markdown.py:246](app_pkg/markdown.py#L246)), 🔥 Smoke-Pflicht via PDF-Gen-Error-Trigger forcieren. **Befund-16-Pattern wird damit quasi-Hot-Fix** mit der ersten C-Welle ohne separaten Hot-Fix-Sprint (analog Master-Folding-Entscheidung aus F-5.2). BT1-Verzahnung mit-aufgelöst.
2. **P9 zweitens** — Score 10.0, isolierter Template-Bug-Fix (Sample-Merge in `templates/markdown_converter.html`), BT2-Verzahnung mit-aufgelöst, 🔥 Smoke-Pflicht via Empty-Markdown-State-Page-Load.
3. **P12 + P13** — Async-Pre-Check + Submit-Loading, ähnliche Frontend-Pfade, evtl. holistic-Apply.
4. **P10 + P11** — restliche Frontend-Touches.

### 3. Smoke-Pflicht-Kalibrierung — Drei 🔥-Pflicht-Live-Master-Patterns von 8

8 Smoke-Pflicht-Patterns (P5–P11, P13) ist viel. Sub-Thread soll Smoke-Disziplin analog F-4-IMPL-B kalibrieren — **drei Pflicht-Live-Master-Smoke-Patterns**, andere 5 als „code-evident verifiziert im Container":

**🔥-Pflicht-Live-Master-Smoke** (vor Apply oder direkt nach Apply, Sub-Thread entscheidet):
- **P5 (Reader-Mode-localStorage-Persistenz)** — Browser-Reload nach Reader-Mode-Aktivierung + Width-Auswahl + Theme; nach Reload State korrekt rehydriert. Test-Anleitung aus F-5.3 P5-Smoke-Mechanik.
- **P7 (Two-Dark-Modes-Reihenfolge)** — Theme-Toggle ↔ Reader-Toggle in unterschiedlichen Reihenfolgen, visueller Konsistenz-Check (Reader-Dark-lokaler-Scope-Sieg über Theme-Dark wenn Reader aktiv).
- **P8 (PDF-Gen-Error-Recovery)** — PDF-Gen-Error forcieren via z.B. Theme-Datei-Manipulation oder einem manipulierten POST; Erwartung: flash-Banner statt 500-Page.

**Code-evident-verifiziert** (Container-side ohne Live-Browser-Smoke):
- P6 (Esc-Key-Listener-Activeelement-Check) — `grep` für `activeElement.tagName`-Pattern + DOM-Test in JSDom optional.
- P9 (Sample-Merge-Template-Bug) — Template-Render-Smoke in Container für Empty-Markdown-State.
- P10 + P11 — Code-Reading-Verifikation der Mechanik.
- P13 (Submit-Loading-Persistenz) — Code-Reading + DOM-Inspect.

Sub-Thread berichtet pro 🔥-Pflicht-Pattern Smoke-Ergebnis im Sub-Batch-Bericht.

### 4. Helper-Vorschlags-Disposition

- **`saveViewState/loadViewState`** — Master-Empfehlung: **extrahieren nach `_utils.js`**. Begründung: P5 selbst hat sub-call-sites (Mode-On/Off, Width, Reader-Theme via gemeinsamem Key), und zweite Call-Site für `library`-View-State-Welle (Sortierung, Filter) ist absehbar. Helper-Disziplin „bei zweiter Call-Site nach _utils.js ziehen" (analog `confirmInPlace`-Disposition aus F-4) ist hier proaktiv erfüllt.
- **`attachAutoDismissToServerBanners`** — Master-Empfehlung: **lokal in `markdown_converter.js` belassen**. Begründung: single-call-site (P2 Auto-Dismiss für Server-Flash-Banner), kein zweiter Apply-Pfad bisher in F-5-Welle absehbar. Folge-Sprints können extrahieren wenn zweite Call-Site auftaucht (z.B. `library` Save-Toast-Konvergenz).

Sub-Thread kann abweichen wenn beim Apply technische Gründe gegen die Master-Empfehlung sprechen (z.B. `saveViewState` bringt Race-Condition mit Theme-CSS-Fetch und braucht eigene Mechanik) — Bericht-Pflicht.

### 5. BT-Folde-Disposition

- **BT1 ↔ P8** (verknüpft): mit-gelöst durch Pattern-Apply, kein separater BT-Apply nötig.
- **BT2 ↔ P9** (verknüpft): mit-gelöst durch Pattern-Apply, kein separater BT-Apply nötig.
- **BT3 (`updateStyle()` doppelt beim Page-Load)**: Master-Empfehlung **mit-folden in P7-Init-Touch** weil P7 (Two-Dark-Modes-Reihenfolge) den Style-Setter-Pfad ohnehin anfasst. `updateStyle()` Call-Site-Konsolidierung ist nahegelegen.
- **BT5 (toter `<link id="preview-style" href="">` ohne JS-Setter)**: Master-Empfehlung **mit-folden in P5-Init-Touch** weil P5 (Reader-Mode-Init) den Theme-CSS-Fetch-Pfad anfasst und der tote `<link>` dort auffallen würde.
- **BT4 (Inline-`<style>` im Template)**: **draußen lassen** (siehe Out-of-scope). Reine HTML-Reorganisation ohne UX-H-Aspekt; separater Hygiene-Pass. Auch wenn beim Apply nahegelegen — **nicht** mit-folden.

### 6. Reader-Mode-Default-Wahl-Validierung

F-5.3 Master-Annotation 4 hat konkrete Defaults für P5/P6/P7 verankert (siehe Patterns-Doc). F5-IMPL-Sub-Thread:
- **Folgt Defaults beim Apply** als Default-Pfad.
- **Kann abweichen** wenn beim Apply technische Probleme auftreten (z.B. `localStorage.markdown.viewState` Race-Condition mit Theme-CSS-Fetch beim Page-Load → braucht z.B. async-await-Init-Sequenz statt sync-Read).
- **Bericht-Pflicht** bei Abweichung mit Begründung. **Nicht** Variante-A/B/C-Diskussion eröffnen — bei Problem konkrete pragmatische Alternative wählen und im Bericht dokumentieren.

---

## Phase 1 — Implementation (drei Sub-Batches mit STOP-Punkten)

### Pre-Flight (vor Sub-Batch A)

1. `pytest tests/` im Container — muss **65/65 grün** sein. (Container-side, weil pytest-Pflicht aus `reference_converter_dep_bump_constraints.md`.)
2. `git status -s` → clean tree erwartet.
3. **Pattern-Doc + Findings-Doc + Inventur-Doc** kurz überfliegen.
4. **F-1.3 Patterns-Doc** lesen für Sub-Batch-A-Übernahme-Vorbereitung — die F-1-Korrespondenz-Spalte in F-5.3 zeigt welche F-1.3-Pattern-Blöcke 1:1 mit Code-Anker-Anpassung übernommen werden.
5. **`_utils.js`-Helper-Bestand verifizieren**: `grep -n "^function\|window\\." static/js/_utils.js` — erwartet `showAlert`, `showToast`, `formatFileSize`, `safeJSON`, `formatDatetimeLocalNow`, `confirmIfLong`, `.sr-only`-Utility (CSS).

---

### Sub-Batch A — Cross-Feature-H4-Konvergenz zu F-1 (4 Patterns)

**Patterns** (per F-5.3 Patterns-Doc): P1 (Save-Failure-Banner — F1+F2 konsolidiert, S, F-1.3 P4-Übernahme), P2 (Auto-Dismiss-Hook für Server-Flash-Banner — XS, F-1.3 P7-Übernahme + lokaler `attachAutoDismissToServerBanners`-Helper), P3 (File-Info-Display nach File-Auswahl — S mit Aufwand-Hochstufung XS→S weil Display insgesamt fehlt nicht nur KB/MB-Unit-Bug, F-1.3 P12-Adaption), P4 (Iframe-a11y-Annotations — XS, F-1.3 P13-Übernahme mit Iframe-spezifischen Anchor `tabindex` / `role` / `aria-label`).

**Mechanik (Holistic-Rewrite empfohlen)**:

1. F-1.3-Pattern-Blöcke für P4/P7/P13/P12 lesen, Code-Anker auf `static/js/markdown_converter.js` + `templates/markdown_converter.html` umlegen.
2. P1 zuerst (Save-Failure-Banner für `saveMarkdownToLibrary`-Pfad — drei Browser-`alert()`-Calls durch `showAlert(<container>, 'danger', ...)` ersetzen plus `safeJSON`-Reuse für Login-Redirect-Detection plus Recovery-Microcopy für H9-Aspekt).
3. P3 + P4 zweitens (a11y-Annotations am Iframe + File-Info-Display nach File-Auswahl mit `formatFileSize` + DE-Microcopy).
4. P2 als drittes (Auto-Dismiss-Helper lokal in `markdown_converter.js`, attached an Server-Flash-Banner beim DOMContentLoaded).
5. **Kein Smoke-Pflicht** — F-1-Patterns sind in `document_converter` live verifiziert, hier 1:1-Übernahme mit Code-Anker-Anpassung. Code-evident-Verifikation via `grep` für `alert(`-Ersatz und Helper-Imports.
6. Tests: optional 1 Test für `attachAutoDismissToServerBanners`-Helper falls JSDom-fähig.
7. `pytest tests/` im Container muss grün bleiben (65-66 erwartet).

**Live-Smoke nach Sub-Batch A** (falls Master-Browser-Access gegeben — Sub-Thread kann skippen wenn nicht möglich):

- Save-Failure-Pfad: DevTools-Network-Throttle Offline → Save-Click → Failure-Banner statt `alert()`-Modal.
- Auto-Dismiss: Server-Flash nach Successful-PDF-Gen → Banner erscheint und dismissed nach N Sekunden.
- a11y-Iframe: DevTools-Inspect zeigt `tabindex` + `role="document"` + `aria-label`.
- File-Info: File-Auswahl zeigt Filename + Size unter dem Input.
- Console clean.

**STOP nach Sub-Batch A** — Bericht: welche der 4 Patterns durch, F-1-Übernahme-Disziplin (welche F-1.3-Pattern-Code-Strings 1:1 wiederverwendet), Test-Stand, ob `attachAutoDismissToServerBanners` lokal oder doch nach `_utils.js` extrahiert.

---

### Sub-Batch B — Reader-Mode-State und Visual-Layout (3 Patterns, alle 🔥 Smoke-Pflicht)

**Patterns**: P5 (Reader-Mode-State + Width-Buttons-Initial — F9+F11 konsolidiert, M-Aufwand, gemeinsamer `localStorage.markdown.viewState`-Key), P6 (Esc-Key-Listener-`activeElement.tagName`-Check statt Document-Global-Refactor — F12, XS), P7 (Two-Dark-Modes-Reihenfolge mit Reader-Dark-Lokal-Sieg — F8, S).

**Mechanik (Holistic-Apply empfohlen wegen Reader-Mode-State-Kopplung)**:

1. **`saveViewState/loadViewState`-Helper in `_utils.js` anlegen** (Master-Annotation 4): generischer Wrapper um `localStorage` mit JSON-Serialize/Parse + Try-Catch für Quota-Exceeded + Default-Value-Fallback. P5 ist erste Call-Site; zweite (`library`-View-State) absehbar.
2. P5 zuerst (Reader-Mode-On/Off + Width + Theme aus `localStorage.markdown.viewState` rehydrieren beim Page-Load via IIFE oder DOMContentLoaded; State-Updates schreiben zurück; Width-Buttons mit `aria-pressed`-Markierung beim Init).
3. P6 zweitens (Esc-Handler-Logik so anpassen dass `if (document.activeElement.tagName === 'TEXTAREA' || document.activeElement.tagName === 'INPUT') return;` vor Reader-Exit greift — sehr klein, isolierter Touch).
4. P7 als drittes (Reader-Dark-lokaler-Scope-Sieg über Theme-Dark wenn Reader aktiv — CSS-Spezifizität via `.reader-mode .reader-dark` oder JS-Branch in `updateStyle()`-Pfad; **BT3 mit-folden** wenn Apply-Pfad `updateStyle()`-Call-Site anfasst).
5. **🔥 Smoke-Pflicht für P5 + P7** (Master-Live-Smoke), **code-evident für P6** (grep für `activeElement`-Pattern):
   - P5 (Master-Smoke): Browser-Reload nach Reader-Mode-Aktivierung + Width-Auswahl + Reader-Theme; nach Reload State korrekt rehydriert.
   - P7 (Master-Smoke): Theme-Toggle aktivieren → Reader-Mode an → erwartete Reader-Dark sichtbar; Reader-Mode aus → erwartete Theme-Dark sichtbar; in beiden Reihenfolgen.
   - P6 (Container-Smoke): `grep -n "activeElement.tagName" static/js/markdown_converter.js` zeigt Pattern.
6. **BT5 mit-folden wenn nahegelegen** — toter `<link id="preview-style" href="">` ohne JS-Setter ist im P5-Init-Touch-Pfad; entfernen wenn `updateStyle()`-Call-Site refactored.
7. Tests: optional 1 Test für `saveViewState/loadViewState`-Helper-Roundtrip falls JSDom-fähig.
8. `pytest tests/` im Container muss grün bleiben (65-66 erwartet).

**Live-Smoke nach Sub-Batch B** (Master-Pflicht für P5+P7):

- Reader-Mode-Persistenz: Browser-Reload erhält Mode + Width + Reader-Theme.
- Two-Dark-Modes: Reader-Dark übersteuert Theme-Dark wenn Reader aktiv.
- Esc-Key: in Textarea → Esc → kein Reader-Exit; außerhalb Textarea → Esc → Reader-Exit.
- BT3 / BT5 Status: Sub-Thread berichtet ob mit-gefoldet oder nahegelegen-aber-nicht-gefoldet.

**STOP nach Sub-Batch B** — Bericht: alle drei Patterns durch, Smoke-Verifikation für P5+P7 (Master-Smoke), `saveViewState/loadViewState`-Helper-Status, BT3/BT5-Folde-Disposition, ob Reader-Mode-Default-Wahl beim Apply abgewichen wurde mit Begründung.

---

### Sub-Batch C — Error-Recovery + Async-Pre-Check (6 Patterns, 5 🔥 Smoke-Pflicht)

**Patterns** (per F-5.3 Patterns-Doc): P8 (PDF-Gen-Error-Recovery — F3+BT1, XS, Score 15.0 Quick-Win #1), P9 (Sample-Merge-Template-Bug — F6+BT2, XS, Score 10.0), P10 (Theme-CSS-Fetch-Recovery — F5, S, 🔥 Smoke-Pflicht via DevTools-404-Block), P11 (markdown-it-CDN-Fallback-DE — Befund 18 / F-5.1, XS, 🔥 Smoke-Pflicht via DevTools-CDN-Block), P12 (Empty-Submit-Pre-Check — F4, XS, Score 10.0), P13 (Submit-Loading-Persistenz — F7, XS, Score 10.0, 🔥 Smoke-Pflicht via Network-Throttle).

**Pre-Flight für Sub-Batch C**:

Sub-Thread liest Patterns P8–P13 nochmal komplett und entscheidet:
- **Default**: 1 Sweep mit P8 zuerst, dann sequenziell P9 → P12 → P13 → P10 → P11.
- **Ca/Cb-Split-Fallback**: wenn Verkopplung als zu groß eingeschätzt (z.B. P10+P11 brauchen gemeinsame Helper-Konvergenz, P12+P13 sind im selben Submit-Handler-Code-Pfad mit Holistic-Touch-Risiko): **Ca = P8 + P9 (isolierte Quick-Wins)** / **Cb = P10 + P11 + P12 + P13 (Frontend-Cluster)**, mit STOP-Punkt zwischen Ca und Cb. Bericht-Pflicht mit Begründung.

**Mechanik (sequenziell, P8 zuerst)**:

1. **P8 zuerst** (Master-Annotation 2 Apply-Reihenfolge):
   - Backend-Touch in [app_pkg/markdown.py:246](app_pkg/markdown.py#L246): `render_template('markdown_converter.html', markdown_text=markdown_text)` ersetzen durch `flash(<DE-Microcopy>, 'danger')` + `return redirect(url_for('markdown_converter'))`.
   - DE-Microcopy aus F-5.3 P8-Block übernehmen (max 2 Sätze, kein Emoji).
   - **🔥 Smoke-Pflicht (Master-Live-Smoke)**: PDF-Gen-Error forcieren — z.B. via manipuliertem POST mit kaputter Theme-Auswahl, oder via Theme-Datei-Path-Manipulation, oder via Playwright-Mock. Erwartung: flash-Banner auf der `markdown_converter`-Page sichtbar statt 500-Page.
   - **BT1-Verzahnung mit-aufgelöst** (Re-Render-Context-Lücke ist durch `redirect()` ohnehin nicht mehr möglich).
   - 1-2 neue Tests in `tests/test_markdown.py` (wenn vorhanden, sonst neu): `test_convert_markdown_pdf_gen_error_redirects_with_flash`.
2. **P9 zweitens** (Sample-Merge-Template-Bug-Fix — Template-Touch in `templates/markdown_converter.html` für Empty-Markdown-State):
   - Pattern-Block aus F-5.3 P9 lesen für konkrete Mechanik.
   - **🔥 Smoke-Pflicht (Container-Smoke)**: Template-Render auf Empty-Markdown-State zeigt korrekten Sample-Text statt Merge-Bug.
   - **BT2-Verzahnung mit-aufgelöst**.
3. **P12 + P13 drittens** (Frontend-Async-Pre-Check + Submit-Loading-Persistenz):
   - P12 (Empty-Submit-Pre-Check): Frontend-Vorab-Check vor Form-Submit, DE-Microcopy.
   - P13 (Submit-Loading-Persistenz): Submit-Btn-Loading-State darf nicht durch CSRF-Refresh-Roundtrip-Restore zu früh weggenommen werden.
   - Gemeinsamer Submit-Handler-Code-Pfad — holistic-Apply möglich.
4. **P10 + P11 viertens**:
   - P10 (Theme-CSS-Fetch-Recovery): Fetch-Failure für Theme-CSS-URL → User-sichtbarer Hinweis statt nur `console.error`.
   - P11 (markdown-it-CDN-Fallback-DE): markdown-it CDN-Dependency-Failure-Pfad mit DE-Microcopy als Fallback-Hinweis.
5. **🔥 Smoke-Pflicht-Mapping**:
   - **Master-Live-Smoke**: P8 (PDF-Gen-Error forcieren).
   - **Container-Smoke**: P9 (Template-Render), P10 (DevTools-404-Block für Theme-CSS-URL → User-Hinweis sichtbar via Code-Reading), P11 (Code-Reading), P13 (Code-Reading + DOM-Inspect).
   - Sub-Thread kann Smoke-Mechanik anpassen wenn Container-Smoke für ein Pattern nicht aussagekräftig genug ist.
6. Tests: 1-2 für P8 (siehe oben), 0-1 für P9 (Template-Render), 0-1 für P12 (Pre-Check Backend-Allowlist falls analog F-013 erweitert).
7. `pytest tests/` im Container muss grün bleiben (66-69 erwartet).

**Live-Smoke nach Sub-Batch C**:

- P8 (Master): PDF-Gen-Error → flash-Banner (nicht 500-Page).
- P9 (Container): Empty-Markdown-State zeigt korrekten Sample.
- P12 + P13 (Code-Reading + ggf. Browser): Empty-Submit zeigt Pre-Check-Banner; Submit-Loading bleibt durchgehend bis Server-Response.
- P10 + P11 (Code-Reading): Theme-CSS-Fetch-Failure-Pfad sichtbar; P11-Mechanik code-evident.

**STOP nach Sub-Batch C** — Bericht: alle 6 Patterns durch, Sub-Batch-vs-Ca/Cb-Disposition, Smoke-Verifikation pro Pattern, Test-Final-Anzahl, BT-Folde-Status (BT1+BT2 mit-gelöst), ob beim Code-Reading neue Findings auffielen die nicht in F-5.2 dokumentiert waren.

---

## Phase 2 — Verify (gesamter Sprint)

1. `pytest tests/` im Container final grün (**66-69 erwartet**).
2. `grep -c "alert(" static/js/markdown_converter.js` → 0 (alle drei `alert()`-Calls aus `saveMarkdownToLibrary` durch `showAlert` ersetzt).
3. `grep -n "showAlert\|safeJSON\|formatFileSize\|saveViewState\|loadViewState" static/js/markdown_converter.js` zeigt Helper-Imports und Call-Sites.
4. **End-to-End-Smoke** (Master-Pflicht für die drei 🔥-Pflicht-Live-Master-Smoke-Patterns):
   - **Standard-Pfad**: Markdown eingeben → Submit → PDF-Download (Submit-Loading sichtbar; Auto-Dismiss-Banner nach Success). Save-to-Library-Pfad: erfolgreich, Failure-Banner statt `alert()` bei Network-Throttle.
   - **Reader-Mode-Pfad**: Reader-Mode an → Width-Auswahl → Reader-Theme → Browser-Reload → State rehydriert. Theme-Toggle ↔ Reader-Toggle in beiden Reihenfolgen → Reader-Dark-Lokal-Sieg.
   - **PDF-Gen-Error-Pfad**: PDF-Gen-Error forcieren → flash-Banner statt 500-Page.
5. DevTools-Console final clean.
6. Sub-Batches A/B/C sind alle drei in `git diff` reflektiert — keine Sub-Batch-Auslassung.
7. F-1-Übernahme-Disziplin: `grep` zeigt dieselben Helper-Patterns wie `document_converter.js` (wo F-1 dort lebt).

Nach Phase 2: STOP — Bericht. Liste der gesmokten Pfade, Final-Test-Anzahl, etwaige Drift-Befunde.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- **Default: drei Commits, einer pro Sub-Batch** (analog F-3-IMPL `843574b` / `40dd02e` / `b3e666a`). Subjects z.B. „F-5 Sub-Batch A: Cross-Feature-H4-Konvergenz zu F-1 (P1, P2, P3, P4)" / „F-5 Sub-Batch B: Reader-Mode-State und Visual-Layout (P5, P6, P7)" / „F-5 Sub-Batch C: Error-Recovery und Async-Pre-Check (P8, P9, P10, P11, P12, P13)".
- Falls Sub-Thread alle Sub-Batches in einem Commit bündeln will (z.B. weil Helper-Konvergenz cross-Sub-Batch lief): kurz im Bericht erwähnen, Default ist drei Commits.
- Bei Ca/Cb-Split: vier Commits (A / B / Ca / Cb).
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commits ist Teil des Sprints. Wenn der Auto-Mode-Classifier blockt: im Phase-3-Bericht erwähnen, Master pusht von Hand (siehe Memory `feedback_push_is_normal.md`; F-3-IMPL, F-4-IMPL-A, F-5.1, F-5.2, F-5.3 wurden alle so gepusht).

---

## Stop-Regel

Nach **jeder Phase** UND **nach jedem Sub-Batch** Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für F5-IMPL**:
- Wenn der Sub-Thread während Sub-Batch A merkt, dass die Verkopplung höher ist als gedacht und Sub-Batch B oder C nicht mehr handhabbar wäre: **STOP nach Sub-Batch A**, Master fragen — Sprint kann re-skopt werden.
- Wenn ein 🔥 Smoke-Pflicht-Pattern (P5/P7/P8) sich als nicht reproduzierbar erweist: STOP, **nicht** silent-fixen — Master entscheidet ob Pattern aus Scope fällt oder Befund neu zu bewerten ist.
- Wenn beim Code-Reading weitere Findings auffallen die nicht in F-5.2 dokumentiert sind: als „aufgefallen, nicht gefixt" in den Bericht — siehe Memory `feedback_no_silent_fixes.md`.
- Wenn Reader-Mode-Default-Wahl aus F-5.3 Master-Annotation 4 beim Apply technische Probleme zeigt (z.B. localStorage-Race-Condition): konkrete pragmatische Alternative wählen, **nicht** Variante-A/B/C-Diskussion eröffnen, im Bericht begründen.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**L** — 13 Patterns in einem Sprint, 3 Sub-Batches (mit Ca/Cb-Fallback), 8 Smoke-Pflicht-Patterns mit Drei-Pflicht-Live-Master-Kalibrierung, 1 neuer Helper in `_utils.js` (`saveViewState/loadViewState`), 1 lokaler Helper (`attachAutoDismissToServerBanners`), 2-4 neue Tests, mehrere Code-Bereiche (`static/js/markdown_converter.js`, `templates/markdown_converter.html`, `static/css/style.css`, `app_pkg/markdown.py`, `static/js/_utils.js`, evtl. `tests/test_markdown.py`). Aufwand-Verteilung XS-lastig (XS=8 / S=4 / M=1) wegen Schwester-Feature-Übernahme aus F-1.3 — strukturell leichter als F-3-IMPL (XL bei 15 Patterns mit gemischter Aufwand-Verteilung) und F-4-IMPL-Gesamt (Cluster I L wegen P1 Cancel-Mechanik).

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Code-Reading von `app_pkg/markdown.py` Auffälligkeiten in den **anderen** Routen auffallen (`download_markdown`, `save_markdown` etc.) die **nicht** zu den 13 Patterns gehören: kurz im Bericht aufzählen, **nicht** in den Sprint-Diff. Master fold-et bei der nächsten Hygiene-Welle.
- Wenn ein Helper-Vorschlag aus F-5.3 im Verlauf des Sprints überflüssig wird (z.B. weil ein anderer Pattern dieselbe Logik schon gelöst hat): kurz im Bericht aufzählen + in `_utils.js` nicht anlegen.
- Wenn beim Live-Smoke ein Befund aus F-5.1 plötzlich anders aussieht als beschrieben (z.B. F8 Two-Dark-Modes funktioniert doch konsistent): im Bericht aufnehmen — code-deduced-Inventur kann irren.
- Wenn `saveViewState/loadViewState` beim Apply technische Probleme zeigt (z.B. JSON-Serialize-Edge-Case mit Theme-CSS-URL-Object): konkrete pragmatische Alternative wählen.
- Wenn BT4 (Inline-`<style>`) beim Apply nahegelegen scheint und die Versuchung groß ist: **nicht** mit-folden (siehe Out-of-scope) — Sub-Thread kann das im Bericht erwähnen, bleibt aber draußen.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F5-IMPL ☑ done 2026-05-XX → commits `<hash-A>` (Sub-Batch A), `<hash-B>` (Sub-Batch B), `<hash-C>` (Sub-Batch C). Pytest <neue Anzahl>/<neue Anzahl> grün. Live-Smoke clean (Standard + Reader-Mode + PDF-Gen-Error-Pfade). 13 Patterns implementiert + BT1/BT2 mit-gelöst + BT3/BT5 mit-gefoldet (BT4 out-of-scope) + 1 neuer Helper in `_utils.js` + 1 lokaler Helper. **F-5 strukturell abgeschlossen** für `markdown_converter`. Verbleibende Sequenz: F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F5-IMPL" raus → Erledigt-Liste mit allen 13 Pattern-IDs zur Traceability. Sektion „2. F-N…" rückt auf Position 1 mit Hinweis dass die nächste F-N-Welle ein neues Feature anpicken muss (`library` List-View, `mermaid_converter`, `login` — Master-Entscheidung).
- **Memory**: nur wenn übertragbare Lehren auftauchen (z.B. „Schwester-Feature-Übernahme XS-Lastigkeit beim Implementation-Sprint — Cross-Feature-H4-Sub-Batch lief reibungslos via 1:1-Pattern-Apply"). Defensiv.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — alle Patterns sind in F-5.3 vollständig spezifiziert mit Reader-Mode-Default-Wahl, Microcopy steht, Aufwandsschätzung ist da, Sub-Batch-Strategie ist im Sprint-Prompt verankert, Smoke-Pflicht-Kalibrierung ist über Drei-Pflicht-Live-Master-Patterns aufgelöst.)_
