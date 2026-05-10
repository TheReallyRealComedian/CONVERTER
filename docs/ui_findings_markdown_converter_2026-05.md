# UX-Heuristik-Findings: markdown_converter (2026-05-10)

**Methodik:** Stufe 2 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Heuristisches Review der strukturierten Inventur aus Stufe 1.
**Quelle:** [docs/ui_inventory_markdown_converter_2026-05.md](ui_inventory_markdown_converter_2026-05.md) — 32 echte interaktive Elemente in 44 Tabellenzeilen, 19 nummerierte Befunde, F-1-Korrespondenz-Übersicht am Doc-Ende.
**Heuristiken:** Nielsen H1 (Sichtbarkeit des Systemzustands), H4 (Konsistenz und Standards), H6 (Wiedererkennen statt Erinnern), H9 (Fehlermeldungen / Hilfe bei Fehlern).
**Produkt-Kontext:** Single-User (Oliver), LAN-only, login-protected. **`markdown_converter` ist zentrales Reader-Workflow-Feature** für Olivers Daily-Reading-Pfad: Markdown im Textarea schreiben oder per File hochladen → Live-Preview im Iframe → optional Theme/Reader-Mode + Width → "Convert to PDF" Download oder "In Library speichern". Daily-Usage-Schmerz-Gewichtung relevant für Library-Save-Pfad und PDF-Download — Reader-Mode-State-Persistenz wird täglich getroffen wenn Oliver in Reader-Mode liest.

**Schwester-Feature-Hebel (Cross-Feature-Inversion zu F-4.2):** `markdown_converter` ist das **direkte Schwester-Feature zu `document_converter`** (F-1) — beide Konversions-Pages mit File-Upload + Submit-Form-Pattern. F-5.1-Korrespondenz-Übersicht zeigt 12 von 14 F-1-Patterns als direkt oder teil-übertragbar (≈86%) und nur 2 als nicht-anwendbar (P2 Result-Area, P5 Drag-Active-Highlight). **Erwartung-Verschiebung gegenüber F-4.2:** F-4.2 hatte 0% Cross-Feature-H4 (F-2-Helper-Konvergenz war bereits durchgezogen); hier ist die Lage umgekehrt — F-1 hat die Konvergenz **noch nicht** auf `markdown_converter` durchgezogen, also dominiert die F-1-Übernahme. **F-1.2 ist primäre Heuristik-Klassifikations-Quelle** für Findings mit F-1-Korrespondenz; H/Sev werden 1:1 übernommen außer Daily-Usage-Schmerz-Gewichtung schiebt.

**Live-Walkthrough-Hinweis:** F-5.1 ist Code-only-Inventur mit 6 dokumentierten Code↔live-Divergenz-Verdachten und 7 Live-Walkthrough-Lücken am Doc-Ende. Findings, deren visueller Effekt aus reinem Code-Reading nicht endgültig beurteilbar ist, sind in der Severity-Spalte mit `⚠️ code-only` gekennzeichnet — Master kann zwischen F5-REVIEW und F5-PATTERNS Walkthrough nachreichen, gerade für die Sev-3-Findings F3 (PDF-Gen-Error-Re-Render) und Sev-2-Findings F6 (Sample-Text-Merge) und F8 (Two-Dark-Modes-Interaktion).

---

## Findings (sortiert absteigend nach Schweregrad)

| #   | Element / Befund | Problem (1–2 Sätze) | Heuristik | Severity | Inventur-Anker | F-1-Korrespondenz | Disposition |
|-----|------------------|---------------------|-----------|----------|----------------|-------------------|-------------|
| F1  | Save-to-Library Failure-Pfad nutzt Browser-`alert()` (Befund 1; Inventur #11/#44) | `saveMarkdownToLibrary()` nutzt `alert('Failed to save: ' + err.message)` ([../static/js/markdown_converter.js:146](../static/js/markdown_converter.js#L146)) und zwei weitere `alert()`-Calls (leerer Textarea-Pfad :102, Submit-Fallback :321) — in-page Banner via `c-alert--danger` wäre konsistent. **Identische Falle wie F-1.2 F7**: Save-Failures gehen in Browser-Modal statt in den `.editor-pane .px-6.pt-4`-Banner-Container, der für den File-Extension-Pfad bereits genutzt wird. | H4 | **3** | Befund 1 | F-1 P4 (direkt) | nur Finding |
| F2  | Save-to-Library Failure Recovery via `alert()` (siehe F1) | `alert()` zwingt zum manuellen "OK", verschwindet spurlos — keine in-page-Trace, keine Möglichkeit, die Meldung erneut anzusehen. **Identisch zu F-1.2 F8**. | H9 | **3** | Befund 1 | F-1 P4 (direkt) | nur Finding |
| F3  | PDF-Generation-Failure-Re-Render hat unvollständigen Template-Context (Befund 16; akut-flag) | [../app_pkg/markdown.py:246](../app_pkg/markdown.py#L246) ruft `render_template('markdown_converter.html', markdown_text=markdown_text)` ohne `themes`/`accepted_extensions`/`accepted_extensions_accept`. Template iteriert `{% for theme in themes %}` und nutzt `accept="{{ accepted_extensions_accept }}"` — bei Flask's Default-`Undefined` wird der Re-Render in einen secondary 500 fallen, statt das geplante flash-Banner ("Error: Could not generate PDF") zu zeigen. Error-Recovery-Pfad ist selbst kaputt. | H9 | **3** ⚠️ code-only | Befund 16 | — (md-spezifisch, akut) | Finding + Bug-Ticket BT1 |
| F4  | Empty-Markdown-Submit hat keinen Frontend-Vorab-Check (Befund 2; Inventur #28) | Submit-Handler ([../static/js/markdown_converter.js:309-349](../static/js/markdown_converter.js#L309-L349)) prüft nur File-Extension — leerer Textarea ohne File geht durch und wird Server-side im Flash-Roundtrip aufgefangen ([../app_pkg/markdown.py:136-138](../app_pkg/markdown.py#L136-L138)). **F-1-P1-teil**: F-1 hatte Empty-Submit komplett silent (Sev 4); hier macht Server-Flash zwar einen Roundtrip, aber Latenz und Re-Render-Kosten sind sichtbar. Inline-Pre-Check via `showAlert` würde den Roundtrip sparen. **Severity-Abweichung von F-1.2 F1 (Sev 4) → Sev 2**, weil Server-Flash existiert (kein silent fail). | H1 | **2** | Befund 2 | F-1 P1 (teil) | nur Finding |
| F5  | Theme-CSS-Fetch-Failure ist silent (Befund 3; Inventur #23) | [../static/js/markdown_converter.js:263](../static/js/markdown_converter.js#L263) `.catch(err => console.error(...))` — kein User-Feedback, wenn `/static/css/pdf_styles/<theme>.css` 404 oder Server-Fail liefert. Preview rendert mit altem oder leerem Theme weiter; User sieht nicht, dass die Theme-Auswahl tatsächlich fehlschlug. | H9 | **2** ⚠️ code-only | Befund 3 | — (md-spezifisch; P9-Loading-Analogie für Fetch-Latenz) | nur Finding |
| F6  | Initial-Sample-Markdown verschmilzt mit Server-`markdown_text` beim Error-Re-Render (Befund 4; Inventur #17) | [../templates/markdown_converter.html:43-69](../templates/markdown_converter.html#L43-L69) hat ein langes Sample-Markdown **gefolgt von** `{{ markdown_text or '' }}` direkt vor dem `</textarea>`. Wenn der `convert_markdown`-Error-Branch die Page mit User-Input re-rendert (Befund 16), erscheint der User-Inhalt **angehängt an das Sample** statt ersetzend. Beobachtbar nur bei PDF-Gen-Failure — daher Severity-Skew gegenüber kosmetischer Wertung. | H1 | **2** ⚠️ code-only | Befund 4 | — (md-spezifisch, Template-Bug) | Finding + Bug-Ticket BT2 |
| F7  | Submit-Button-Loading-Text "Preparing…" zu kurz sichtbar (Befund 12; Inventur #27/#28) | [../static/js/markdown_converter.js:345-347](../static/js/markdown_converter.js#L345-L347): `submitBtn.innerHTML = originalLabel; form.submit();` — Loading-Text wird **vor** dem `form.submit()`-Call zurückgesetzt. User sieht "Preparing…" nur kurz während des CSRF-Refresh-Roundtrips, dann sofort zurück zu "Convert to PDF" — die Server-PDF-Gen-Zeit (mehrere Sekunden für Playwright-Browser-Boot) hat **keinen Loading-Indikator** mehr sichtbar. F-1-P9 (Submit-Loading) ist hier nur teil-erfüllt. | H1 | **2** ⚠️ code-only | Befund 12 | F-1 P9 (teil) | nur Finding |
| F8  | Zwei voneinander unabhängige Dark-Modes (Befund 11; Inventur #1/#36) | Globaler `data-global-theme` (Theme-Toggle Layout-#1) und Reader-Mode-only `data-theme` (Reader-Dark-Btn #36) wirken nicht aufeinander. `isDarkActive()` ([../static/js/markdown_converter.js:178-182](../static/js/markdown_converter.js#L178-L182)) checkt **beide** als Dark-Trigger; aber Buttons setzen unterschiedliche Attribute. Edge-Case: globaler Theme dunkel → Reader-Mode betreten → erneut Dark applied; Reader-Mode verlassen → `data-theme` entfernt, aber `data-global-theme="dark"` bleibt. Interner H4-Bruch innerhalb der Page (zwei Dark-Pfade, kein konsolidiertes Mental-Model). | H4 | **2** ⚠️ code-only | Befund 11 | — (md-spezifisch, intern) | nur Finding |
| F9  | Reader-Mode-On/Off-State wird nicht persistiert (Befund 9; Inventur #12) | [../static/js/markdown_converter.js:30-46](../static/js/markdown_converter.js#L30-L46): `localStorage.readerPrefs` hält `dark`/`fontSize`/`width`, aber **nicht** den Mode-Toggle selbst. User der Reader-Mode bevorzugt muss bei jedem Page-Load einmal toggeln. Recognition-over-Recall-Bruch — User-Erwartung "Modus bleibt wo ich ihn gelassen habe" wird gebrochen. **Daily-Usage-Schmerz mittel**: wenn Oliver Reader-Mode regelmäßig nutzt, trifft jeder Reload. | H6 | **2** | Befund 9 | — (md-spezifisch, Reader-Mode-only) | nur Finding |
| F10 | Kein File-Info-Display nach File-Auswahl (Befund 5; Inventur #19) | `handleFileSelect()` ([../static/js/markdown_converter.js:266-275](../static/js/markdown_converter.js#L266-L275)) füllt den Textarea-Inhalt, rendert aber keine File-Info-UI. Im Gegensatz zu F-1 #13/#14 (Filename + Size in MB) sieht der User den Filename nirgends, kann nicht ohne weiteres "Datei abwählen". Sichtbarer System-Status "ich habe X hochgeladen" fehlt. **F-1-P12 nicht-anwendbar mangels Anzeige insgesamt** (KB/MB-Fallback erst sinnvoll nach Display-Pattern in F5-PATTERNS). | H1 | **1** | Befund 5 | F-1 P12 (teil/nicht-anwendbar) | nur Finding |
| F11 | Width-Buttons haben vor erstem Reader-Toggle keinen `.active`-Visual (Befund 10; Inventur #32–#35) | [../static/js/markdown_converter.js:35-40](../static/js/markdown_converter.js#L35-L40) ruft `updateWidthButtons(prefs.width || 'medium')` erst **bei** Reader-Toggle-On. Bevor Reader-Mode betreten wird, fehlt das `.active`-Marker auf allen vier Width-Buttons. Toolbar ist allerdings ohnehin `display:none` initial — Befund nur sichtbar im allerersten Reader-Toggle-Frame, danach State-stabil. Recognition-over-Recall-Bruch nur kurz sichtbar. | H6 | **1** ⚠️ code-only | Befund 10 | — (md-spezifisch, Reader-Toolbar-only) | nur Finding |
| F12 | Esc-Key-Listener ist Document-global (Befund 17; Inventur #38) | [../static/js/markdown_converter.js:94-98](../static/js/markdown_converter.js#L94-L98): `document.addEventListener('keydown', ...)` löst Reader-Mode-Exit unabhängig vom Fokus aus. Bei Textarea-Fokus + Reader-Mode-aktiv könnte Esc User unerwartet aus dem Reader-Mode rauswerfen, wenn z.B. Browser-Autocomplete geschlossen werden soll. Recognition-over-Recall-Bruch (User erwartet Esc-Verhalten kontext-abhängig). | H6 | **1** ⚠️ code-only | Befund 17 | — (md-spezifisch, Reader-Mode-only) | nur Finding |
| F13 | Iframe `#preview-iframe` ohne `tabindex`/`role` (Befund 13; Inventur #40) | [../templates/markdown_converter.html:142](../templates/markdown_converter.html#L142) hat `title="PDF Preview"` (≈aria-label), aber kein `tabindex="0"`, kein `role="region"`. Screenreader/Tastatur-User können den Iframe-Content nicht gezielt fokussieren — analog zu F-1.2 F18 (`<pre>` ohne a11y). **F-1-P13 direkt übertragbar**: H6 Sev 1 1:1 übernommen, Iframe statt `<pre>` als Fokus-Ziel. | H6 | **1** | Befund 13 | F-1 P13 (direkt) | nur Finding |
| F14 | Flash-Banner haben keinen Auto-Dismiss (Befund 7; Inventur #13/#14) | Inline-Close-`×` vorhanden ([../templates/markdown_converter.html:31](../templates/markdown_converter.html#L31)), aber kein Auto-Dismiss-Timer. Flash-Banner stehen bis User klickt. **F-1-P7 direkt übertragbar**, aber Severity-Skew gegenüber F-1.2 F12 (H1 Sev 2, dort fehlten Close-Btn UND Auto-Dismiss): hier ist Close-`×` schon da, nur Auto-Dismiss für non-danger Levels fehlt → Sev 1 (verringert auf rein kosmetisch). | H1 | **1** | Befund 7 | F-1 P7 (direkt) | nur Finding |
| F15 | markdown-it als externer CDN-Load (Befund 18; Inventur #40) | [../templates/markdown_converter.html:154](../templates/markdown_converter.html#L154) lädt `https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js`. Bei CDN-Ausfall / offline ist die Live-Preview kaputt — Code fängt das ab und rendert "Preview unavailable: markdown-it library failed to load" im Iframe ([../static/js/markdown_converter.js:158-163](../static/js/markdown_converter.js#L158-L163)). Server-PDF-Pipeline arbeitet unabhängig (Python markdown-it). Recovery-Microcopy ist EN und ohne konkreten Hinweis. **Sev 1**: Ausfall ist selten (jsdelivr ist stabil), Error-Pfad existiert bereits. | H9 | **1** ⚠️ code-only | Befund 18 | — (md-spezifisch, Architektur) | nur Finding |

---

## Reine Bug-Tickets (ohne eigenständiges Heuristik-Finding **oder** als Implementations-Anker für Findings — separates Ticket-Material)

Drei BTs sind reine Code-Hygiene ohne UX-H-Aspekt (BT3–BT5, vergleichbar zu F-3.2's BT7/BT8 und F-4.2's BT4); zwei BTs sind Finding-linked Implementations-Anker für Sev-3- bzw. Sev-2-Findings (BT1, BT2). **Kein Fix-Pfad-Vorschlag, keine konkrete Microcopy** — das macht F5-PATTERNS bzw. F5-IMPL.

- **BT1: PDF-Generation-Failure-Re-Render unvollständiger Template-Context.** [../app_pkg/markdown.py:246](../app_pkg/markdown.py#L246) ruft `render_template('markdown_converter.html', markdown_text=markdown_text)` ohne `themes`/`accepted_extensions`/`accepted_extensions_accept`-Context. Template-Iterations- und `accept`-Attribut-Stellen würden bei Flask-Default-`Undefined` einen `UndefinedError` werfen → secondary 500 statt freundlichem flash-Banner. → siehe Finding F3. Reproduktion: Forciertes PDF-Gen-Failure auslösen (z.B. durch Playwright-Crash via gemockten Style-Fail oder via injizierten markdown_text der den Renderer crasht), Page-Re-Render beobachten — sollte 500 mit Jinja-`UndefinedError` liefern statt der geplanten flash-Banner-Page. **Master-Annotation: in F5-IMPL mit-fixen, kein Hot-Fix-Sprint.** F5-PATTERNS greift Befund 16 als eigenes Pattern "PDF-Gen-Error-Recovery" mit Microcopy-Aspekt auf.
- **BT2: Sample-Text-Merge-Template-Bug.** [../templates/markdown_converter.html:43-69](../templates/markdown_converter.html#L43-L69) hat Sample-Markdown **vor** `{{ markdown_text or '' }}`; bei Server-Re-Render-mit-User-Input verschmilzt User-Inhalt mit Sample. Selbe Wurzel-Familie wie BT1 (beide entstehen nur im PDF-Gen-Error-Pfad), aber unterschiedlicher Trigger (BT1 = fehlender Context, BT2 = Sample-Anhängung an User-Input). → siehe Finding F6. Reproduktion: nur sichtbar nach BT1-Fix — wenn die Re-Render-Page wieder ladbar ist, würde User-Input unter dem Sample-Text doppelt angezeigt.
- **BT3: `updateStyle()` wird beim Page-Load doppelt aufgerufen.** [../static/js/markdown_converter.js:281](../static/js/markdown_converter.js#L281) und [../static/js/markdown_converter.js:289](../static/js/markdown_converter.js#L289) — der zweite Aufruf macht denselben Theme-CSS-Fetch identisch nochmal. **Aus F-5.2 als Bug-only katalogisiert, weil keine UX-Heuristik-Komponente vorhanden ist** (User sieht heute keinen Unterschied — beide Fetches liefern dasselbe CSS, der zweite Fetch wird vom Browser-Cache absorbiert; reine Code-Hygiene plus minimaler Network-Cost). Code-Anker s.o. Reproduktion: DevTools-Network beim Page-Load beobachten — zwei identische Requests an `/static/css/pdf_styles/default.css` (zweiter aus Cache).
- **BT4: Inline-`<style>` im Template.** [../templates/markdown_converter.html:156-168](../templates/markdown_converter.html#L156-L168) deklariert `.orientation-btn { ... }` und `.orientation-btn.active { ... }` per `<style>`-Tag im `{% block scripts %}` statt in [../static/css/style.css](../static/css/style.css). **Aus F-5.2 als Bug-only katalogisiert, weil keine UX-Heuristik-Komponente vorhanden ist** (User sieht keinen Unterschied; reine Architektur-Drift; CSP-strict würde inline-styles blocken — aber CSP ist im Single-User-LAN-Setup nicht im Spiel). Code-Anker s.o. Reproduktion: Template-Source inspizieren — `<style>`-Block im scripts-Block.
- **BT5: Tot `<link id="preview-style" rel="stylesheet" href="">`** im `head_extra`-Block ([../templates/markdown_converter.html:6](../templates/markdown_converter.html#L6)) hat **immer leeren `href`** und wird nirgendwo per JS gesetzt. Tot-Code-Kandidat. **Aus F-5.2 als Bug-only katalogisiert, weil keine UX-Heuristik-Komponente vorhanden ist** (User sieht keinen Unterschied — leeres `<link>` löst keinen Request aus; reine Code-Aufräum). Code-Anker s.o. Reproduktion: DevTools-Elements zeigt das Tag mit leerem `href`; grep nach `getElementById('preview-style')` in JS gibt keine Treffer.

---

## Aus F-5.2 ausgenommene Inventur-Befunde

Zwei Inventur-Befunde fielen beim Heuristik-Filter heraus, mit expliziter Begründung (analog F-3.2-/F-4.2-Konvention für ausgenommene Items):

- **Befund 8 (Flash-Strings sind durchgehend EN; [../app_pkg/markdown.py:137](../app_pkg/markdown.py#L137), :143, :157, :245)**: aus dem Heuristik-Review explizit ausgenommen laut Sprint-Prompt — **gehört zur DE-Pass-Welle in F5-PATTERNS**. F-5.1 hat das als sprint-konstitutiven Hinweis markiert ("DE-Pass kommt in F-5.3"). Hier kein eigenes Finding, weil DE-Pass eine separate Pattern-Investition mit eigener Microcopy-Disziplin ist (analog F-2.2 F18 → F-2-IMPL Cluster Polish-1).
- **Befund 19 (`window.PageData`-Inline-Block; [../templates/markdown_converter.html:149-153](../templates/markdown_converter.html#L149-L153))**: aus dem Heuristik-Review explizit ausgenommen, weil **konformer Codestyle**, kein UX-H-Aspekt. CLAUDE.md-Pattern "Templates inline only small `window.PageData = {…}` blocks" wird hier eingehalten. F-5.1 hat das als reine Notiz markiert. Filtert sich beim Heuristik-Filter heraus (H1/H4/H6/H9 treffen alle nicht).

---

## Cross-Feature-H4-Sektion (Schwester-Feature-Konvergenz zu F-1)

**Pattern-Konvergenz-Quote: 86% (12 von 14 anwendbaren F-1-Patterns).** Drastisch höher als F-2.2 (~41%), F-3.2 (~35%) und F-4.2 (0%) — Schwester-Feature-Inversion bestätigt. **Begründung:** F-1 hat seine 14 Patterns in der UX-Cascade etabliert, aber die Konvergenz noch **nicht** auf `markdown_converter` durchgezogen. F-5.1 hat in der Korrespondenz-Übersicht 4 direkt anwendbare, 8 teil-übertragbare bzw. bereits-erfüllte und nur 2 nicht-anwendbare Patterns ausgewiesen.

### Direkt übertragbare F-1-Patterns (mit `markdown_converter`-Findings)

| F-1-Pattern | F-1.2-Finding-Quelle | markdown_converter-Finding | Heuristik | Severity | Code-Anker |
|-------------|---------------------|----------------------------|-----------|----------|------------|
| **P4 — Save-Failure Browser-`alert()` ersetzen** | F-1.2 F7 (H4 Sev 3) | F1 (1:1 übernommen) | H4 | 3 | [../static/js/markdown_converter.js:146](../static/js/markdown_converter.js#L146) |
| **P4 — Save-Failure Recovery via in-page Banner** | F-1.2 F8 (H9 Sev 3) | F2 (1:1 übernommen) | H9 | 3 | [../static/js/markdown_converter.js:146](../static/js/markdown_converter.js#L146) |
| **P7 — Alert-Auto-Dismiss** | F-1.2 F12 (H1 Sev 2) | F14 (Sev verringert auf 1) | H1 | 1 | [../templates/markdown_converter.html:31](../templates/markdown_converter.html#L31) |
| **P13 — Result-Content a11y** | F-1.2 F18 (H6 Sev 1) | F13 (1:1 übernommen, Iframe statt `<pre>`) | H6 | 1 | [../templates/markdown_converter.html:142](../templates/markdown_converter.html#L142) |
| **P14 — Download Success-Toast** | (kein F-1.2-Finding; F-1-Pattern aus F-1-Patterns-Doc) | (kein F-5.2-Finding) | — | — | F5-PATTERNS-Diskussion |

**Severity-Abweichungen begründet:**
- **F14 (P7-Auto-Dismiss):** F-1.2 F12 hatte Sev 2, weil Close-Button **und** Auto-Dismiss fehlten. Hier ist Close-`×` bereits im Template inline vorhanden ([../templates/markdown_converter.html:31](../templates/markdown_converter.html#L31)) — nur Auto-Dismiss-Timer für non-danger-Levels fehlt → Sev verringert auf 1 (rein kosmetisch).
- **P14 (Download-Toast):** F-1 hat das Pattern für `document_converter` etabliert, aber F-1.2 hat dafür kein eigenes Finding ausgewiesen (P14 ist erst in F-1-PATTERNS-Doc benannt). Für `markdown_converter` ist die Disposition unklar — Browser-Download-Bar nach `form.submit()` deckt redundant ab, ein expliziter Toast wäre Polish. **F5-PATTERNS-Diskussion** ohne F-5.2-Finding.

### Teil-übertragbare F-1-Patterns (mit modifizierten `markdown_converter`-Findings)

| F-1-Pattern | F-1.2-Finding-Quelle | markdown_converter-Finding | Heuristik | Severity | Anpassung |
|-------------|---------------------|----------------------------|-----------|----------|-----------|
| **P1 — Empty-Submit silent** | F-1.2 F1 (H1 Sev 4) + F2 (H9 Sev 4) | F4 (Sev verringert auf 2) | H1 | 2 | F-1-Falle war komplett silent (`display:none`-Anker); hier macht Server-Flash-Roundtrip Feedback, nur ohne Inline-Pre-Check. Daher Sev 2 statt 4, und nur H1-Reihe (kein H9-Split). |
| **P9 — Submit-Loading-Indikation** | F-1.2 F14 (H1 Sev 2; Drop-Zone-Loading) | F7 (1:1 H1 Sev 2) | H1 | 2 | F-1 hatte fehlende Drop-Zone-Loading; hier hat Submit-Btn Loading-Text "Preparing…", aber nur kurz vor `form.submit()` sichtbar. Selbe Schmerz-Klasse, andere Mechanik (Text-Restore vs. Drop-Zone-State-fehlt). |
| **P12 — Filename KB/MB-Fallback** | F-1.2 F17 (H4 Sev 1) | F10 (umklassifiziert auf H1 Sev 1) | H1 | 1 | F-1 hatte MB-Display; hier fehlt File-Info-Display **insgesamt**. Pattern-Anwendung verschiebt sich von "MB-Bug fixen" zu "Display einführen". Heuristik-Verschiebung von H4 (Konsistenz-Bruch) auf H1 (System-Status nicht sichtbar). |

**P6.1 Mini-Pattern-Hinweis:** F-1.2 hat F9 (H4 Sev 3 — Drop-Zone-Label vs. Backend-Akzeptanz). Hier ist F-1-P6 (Format-Label/Accept-Mismatch) durch SEC-Sprint-F-006 bereits-erfüllt für das `accept`-Attribut, aber das sichtbare Label "Or Upload a File" ([../templates/markdown_converter.html](../templates/markdown_converter.html), Inventur #20) hat **keinen Format-Hint** ("Erlaubt: .md, .markdown" als sichtbares Label-Beispiel). F-5.2 nimmt das **nicht** als eigenes Finding auf, weil der Mini-Aspekt unter der Sev-1-Schwelle liegt; F5-PATTERNS kann es als Mini-Pattern (P6.1) aufgreifen.

### Bereits konvergente F-1-Patterns (markdown_converter erfüllt das F-1-Pattern strukturell — kein Finding nötig)

| F-1-Pattern | Bereits erfüllt durch | Code-Anker |
|-------------|------------------------|------------|
| **P3 — Save-Btn `.saved`-Reset** | setTimeout-Reset-Routine entfernt `.saved` korrekt nach 2s | [../static/js/markdown_converter.js:135-139](../static/js/markdown_converter.js#L135-L139) |
| **P6 — Format-Label/`accept` Single-Source-of-Truth** | SEC-Sprint-F-006: `accept` wird aus `ACCEPTED_EXTENSIONS = ('md', 'markdown')` generiert | [../app_pkg/markdown.py:25](../app_pkg/markdown.py#L25), [../app_pkg/markdown.py:131-134](../app_pkg/markdown.py#L131-L134); Label-Hint-Mini siehe oben |
| **P8 — Frontend-Vorab-Check unsupported file** | Submit-Handler prüft Extension via `fileExtensionAllowed()` und nutzt `showAlert(... 'danger', 'Dateiformat nicht unterstützt. Erlaubt: .md, .markdown.')` (DE-Microcopy bereits da!) | [../static/js/markdown_converter.js:317-322](../static/js/markdown_converter.js#L317-L322) |
| **P10 — Result-Area scrollIntoView** | Konvergent durch Architektur (Layout-Lösung statt Scroll-Lösung) — Iframe-Preview ist immer im Viewport sichtbar (Split-Pane) | [../templates/markdown_converter.html](../templates/markdown_converter.html) (Editor-Pane + Preview-Pane Split-Layout) |
| **P11 — Drop-Zone Keyboard-Pfad** | Konvergent durch Architektur — visible native `<input type="file">` ist out-of-the-box keyboard-erreichbar | [../templates/markdown_converter.html](../templates/markdown_converter.html) Inventur #19 |

**Begründung der Verschärfung "teil-erfüllt → bereits konvergent" für P10 und P11**: F-5.1 hatte beide als "teil bzw. bereits-erfüllt" markiert. F-5.2 verschärft auf "bereits konvergent", weil die markdown_converter-Architektur das F-1-Problem strukturell vermeidet (P10: Split-Pane-Layout statt Result-Area + Scroll; P11: Native-File-Input statt Custom-Drop-Zone) — kein UX-Schmerz mehr verbleibend, kein Pattern-Finding-Bedarf. Master kann das in F5-PATTERNS gegenchecken.

### Nicht-anwendbare F-1-Patterns

| F-1-Pattern | Begründung |
|-------------|-----------|
| **P2 — Result-Area persistiert nach Clear** | Kein Result-Area, kein Clear-Button — der Workflow ist Live-Preview + Server-PDF-Download. Keine Stale-State-Problematik. |
| **P5 — Drag-Active-Highlight transparent** | Keine Drop-Zone, nur visible File-Input ohne Drag-Active-Styling. Pattern setzt Drop-Zone voraus. |

### Konvergenz-Quote-Berechnung

- **Anwendbare F-1-Patterns**: 12 (von 14 total; 2 nicht-anwendbar)
- **Direkt übertragbar**: 4 (P4, P7, P13, P14) — produzieren 4 Findings (F1, F2, F14, F13) plus F5-PATTERNS-Diskussion (P14)
- **Teil-übertragbar mit Anpassung**: 3 (P1, P9, P12) — produzieren 3 Findings (F4, F7, F10)
- **Bereits konvergent**: 5 (P3, P6, P8, P10, P11) — kein Finding-Bedarf
- **Pattern-Konvergenz-Quote**: 12/14 = **86%**, im erwarteten Bereich 80-90% (Master-Annotation bestätigt).
- **Cross-Feature-H4-Findings (Finding-level)**: 7 von 15 Findings (F1, F2, F4, F7, F10, F13, F14) ≈ **47%** — höher als F-2.2 (~41%) und F-3.2 (~35%), zwischen Schwester-Feature-Inversion und residualen md-spezifischen Befunden.

---

## Markdown-spezifische Heuristik-Sub-Sektion

8 Findings ohne F-1-Korrespondenz, die markdown-eigen sind (Reader-Mode-Lifecycle, Iframe-Preview-Render, PDF-Gen-Error-Pfad, Theme-CSS-Fetch). Diese sind der "neue" Heuristik-Anteil dieses Sprints.

| # | Finding | Heuristik | Severity | Befund-Anker | Sub-Cluster |
|---|---------|-----------|----------|--------------|-------------|
| F3 | PDF-Generation-Failure-Re-Render unvollständig | H9 | **3** ⚠️ | Befund 16 | Error-Recovery |
| F5 | Theme-CSS-Fetch-Failure silent | H9 | **2** ⚠️ | Befund 3 | Error-Recovery |
| F6 | Sample-Text-Merge mit Server-`markdown_text` | H1 | **2** ⚠️ | Befund 4 | Error-Recovery (Template-Bug) |
| F8 | Zwei voneinander unabhängige Dark-Modes | H4 | **2** ⚠️ | Befund 11 | Reader-Mode-State |
| F9 | Reader-Mode-On/Off-State nicht persistiert | H6 | **2** | Befund 9 | Reader-Mode-State |
| F11 | Width-Buttons kein Initial-Active vor Reader-Toggle | H6 | **1** ⚠️ | Befund 10 | Reader-Mode-State |
| F12 | Esc-Key-Listener Document-global | H6 | **1** ⚠️ | Befund 17 | Reader-Mode-State |
| F15 | markdown-it CDN-Dependency | H9 | **1** ⚠️ | Befund 18 | Architektur (Dependency) |

**Heuristik-Verteilung (md-spez Findings):** H1 = 1 (F6), H4 = 1 (F8), H6 = 4 (F9, F11, F12 + Reader-Mode-Sub), H9 = 3 (F3, F5, F15). H6 dominiert wegen Reader-Mode-Lifecycle-Lücken; H9 dominiert wegen Error-Pfade (PDF-Gen, Theme-Fetch, CDN-Fail). Daily-Usage-Schmerz konzentriert sich auf F9 (Reader-Mode-Persistenz, daily für Oliver wenn er Reader-Mode regelmäßig nutzt) und F3 (PDF-Gen-Error-Re-Render, selten aber bei Trigger hart).

---

## Schwerpunkt-Cluster

Vier thematische Cluster, in denen sich die schweren Findings konzentrieren — analog F-3.2's "Silent-Failure-Familie / Notion-Side State-Wipe / Cross-Feature-Helper-Drift" und F-4.2's "Cancel-und-Cleanup-Recovery / Async-State-Visibility / Polling-Robustheit / Speaker-Format-Hilfe":

### Cluster 1 — Cross-Feature-H4-Helper-Reuse zu F-1 (F1, F2, F14, F13, F10; Sev 1–3)

**Daily-Usage-Schmerz mittel** (Save-Library-Pfad ist häufiger Endpoint). Fünf Findings, die direkt aus F-1-Pattern-Übernahmen entstehen — F-1 hat etabliert, `markdown_converter` ist nicht konvergiert.
- **F1 + F2 (P4-Save-`alert()`)**: drei `alert()`-Calls in `saveMarkdownToLibrary`, davon zwei mit EN-Strings, einer mit DE-String aus dem File-Extension-Pfad. Identisch zu F-1.2 F7+F8.
- **F14 (P7-Auto-Dismiss)**: Close-`×` schon da, nur Timer fehlt — günstigster Cross-Feature-Konvergenz-Punkt.
- **F13 (P13-Iframe-a11y)**: `tabindex="0"` + `role="region"` ergänzen, analog F-1's `<pre>`-a11y.
- **F10 (P12-File-Info-Display)**: kein File-Info-Anzeige nach Auswahl — Pattern-Anwendung erfordert kleine UI-Erweiterung (Filename + `formatFileSize` aus `_utils.js`).

**Fix-Pfad in F5-PATTERNS:** ein Cluster-Pattern "Cross-Feature-Helper-Konvergenz zu F-1" der `showAlert` für `saveMarkdownToLibrary` einführt + Auto-Dismiss-Timer ergänzt + Iframe-a11y-Annotations setzt + File-Info-Display + `formatFileSize` reuse. Existing-helper-reuse, XS bis S Aufwand.

### Cluster 2 — Reader-Mode-State und Visual-Layout (F9, F11, F12, F8; Sev 1–2 H6+H4)

**Daily-Usage-Schmerz mittel-hoch wegen F9** (wenn Oliver Reader-Mode regelmäßig nutzt). Vier Findings, die das Reader-Mode-Lifecycle und seine State-Konsistenz betreffen.
- **F9 (Reader-Mode-On/Off nicht persistiert)**: zentralster Daily-Usage-Schmerz dieses Clusters. `localStorage.readerPrefs` hält Sub-Prefs aber nicht den Mode-Toggle.
- **F11 (Width-Buttons-Initial-Active)**: nur kurz im allerersten Reader-Toggle-Frame sichtbar — sub-kosmetisch.
- **F12 (Esc-Key Document-global)**: Recognition-over-Recall-Bruch bei Textarea-Fokus.
- **F8 (Zwei Dark-Modes)**: globaler Theme-Toggle vs. Reader-only Dark-Btn — interner H4-Bruch innerhalb der Page.

**Fix-Pfad in F5-PATTERNS:** Reader-Mode-State-Konsolidierung als ein Pattern: (a) Reader-Mode-Toggle-Persistenz in `localStorage.readerPrefs.modeOn`; (b) `updateWidthButtons(prefs.width)` auch im Initial-Pfad aufrufen; (c) Esc-Key-Listener auf Reader-Mode-Container scopen oder bei Textarea-Fokus suppressen; (d) Dark-Mode-Konsolidierung — gemeinsame Quelle für `data-theme` und `data-global-theme`.

### Cluster 3 — Error-Recovery-Pfade (F3, F5, F6, F2; Sev 2–3 H9 ⚠️ code-only)

**Daily-Usage-Schmerz niedrig pro Vorfall, hoch pro Konsequenz.** Vier Findings, die Fehler-Pfade auf der Page betreffen — alle mit `⚠️ code-only`-Marker (PDF-Gen-Error-Trigger ist selten und schwer zu reproduzieren).
- **F3 (PDF-Gen-Error-Re-Render)**: akut-Befund 16. Error-Recovery-Pfad selbst kaputt — secondary 500 statt freundlichem flash-Banner. Master-Disposition: Finding + Bug-Ticket BT1, in F5-IMPL mit-fixen.
- **F5 (Theme-CSS-Fetch silent)**: kein User-Feedback bei Theme-CSS-404. Recovery-Anleitung fehlt.
- **F6 (Sample-Text-Merge im Error-Re-Render)**: Template-Bug, nur sichtbar wenn BT1 gefixt ist und Page wieder rendert. Folge-Bug.
- **F2 (Save-`alert()` Recovery)**: bereits in Cluster 1, Querverweis hier wegen H9-Familie.

**Fix-Pfad in F5-PATTERNS:** Pattern "PDF-Gen-Error-Recovery" mit Microcopy-Aspekt (BT1-Fix als Strukturvoraussetzung plus dedizierter Banner-Text für PDF-Gen-Failure — Microcopy in F5-PATTERNS); Pattern "Theme-CSS-Fetch-Recovery" mit User-sichtbarem Hinweis (analog F-3.2-Finding-Verbesserung); BT2 als folgender Template-Fix (nach BT1) eingeplant.

### Cluster 4 — Async-Pre-Check und Loading-Visibility (F4, F7; Sev 2 H1 ⚠️ code-only)

**Daily-Usage-Schmerz niedrig.** Zwei Findings, die Submit-Pfad-Pre-Check und Loading-Indikation betreffen.
- **F4 (Empty-Markdown Submit ohne Pre-Check)**: F-1-P1-teil — Server-Flash macht Roundtrip, kein Inline-Pre-Check.
- **F7 (Submit-Loading "Preparing…" zu kurz)**: F-1-P9-teil — Loading-Text vor `form.submit()` zurückgesetzt, Server-PDF-Gen-Zeit ohne Indikator.

**Fix-Pfad in F5-PATTERNS:** kleines Pattern "Submit-Pre-Check + Loading-Persistenz" der den Empty-Markdown-Pre-Check via `showAlert` ergänzt und den Loading-Text bis zur Server-Antwort hält (alternativ dedizierter Server-Render-Loading-State — Microcopy in F5-PATTERNS).

---

## Disposition-Verteilung

- **Nur Findings (kommen in F5-PATTERNS):** 13 — F1, F2, F4, F5, F7, F8, F9, F10, F11, F12, F13, F14, F15
- **Findings + Bug-Tickets (kommen in F5-PATTERNS **plus** separates Bug-Ticket):** 2 Findings → **2 unique Bug-Tickets** — F3 (BT1), F6 (BT2)
- **Nur Bug-Tickets (kommen **nicht** in F5-PATTERNS):** 3 — BT3 (updateStyle()-doppelt), BT4 (Inline-`<style>`), BT5 (Tot-`<link>`)
- **Aus F-5.2 ausgenommen:** 2 Inventur-Befunde — Befund 8 (Flash-Strings EN, DE-Pass-Welle in F5-PATTERNS), Befund 19 (window.PageData konformer Codestyle, kein UX-H-Aspekt)

**Inventur-Befund-Coverage (alle 19 disponiert):**
- Befund 1 → F1 (H4) + F2 (H9) — **eine Inventur-Befund-Wurzel, zwei Heuristik-Reihen** (analog F-1.2 F7/F8 und F-2.2 F1/F2)
- Befund 2 → F4 (H1; Sev verringert von F-1.2-Sev-4)
- Befund 3 → F5 (H9; md-spezifisch)
- Befund 4 → F6 + BT2
- Befund 5 → F10 (H1; umklassifiziert von F-1.2-H4)
- Befund 6 → BT3 (kein UX-H — pure Code-Hygiene)
- Befund 7 → F14 (H1; Sev verringert von F-1.2-Sev-2)
- Befund 8 → ausgenommen (DE-Pass-Welle in F5-PATTERNS)
- Befund 9 → F9 (H6; md-spezifisch)
- Befund 10 → F11 (H6; md-spezifisch)
- Befund 11 → F8 (H4; intern md-spezifisch)
- Befund 12 → F7 (H1; F-1-P9-teil)
- Befund 13 → F13 (H6; F-1-P13-direkt)
- Befund 14 → BT4 (kein UX-H — Architektur)
- Befund 15 → BT5 (kein UX-H — Tot-Code)
- Befund 16 → F3 + BT1 (akut, Master-Disposition Finding+BT)
- Befund 17 → F12 (H6; md-spezifisch)
- Befund 18 → F15 (H9; md-spezifisch)
- Befund 19 → ausgenommen (kein UX-H — konformer Codestyle)

**Abweichungen von F-5.1-Disposition (begründet):**
- **Befund 6 (updateStyle()-doppelt):** F-5.1-Disposition war "F5-REVIEW". F-5.2 ordnet als "nur Bug-Ticket" (BT3) ein, weil **keine UX-Heuristik-Komponente** vorhanden ist (User sieht heute keinen Unterschied — der zweite Fetch wird vom Browser-Cache absorbiert). Filtert sich beim Heuristik-Filter heraus, gehört zur Code-Hygiene-Welle. Analog zu F-3.2's BT7/BT8 und F-4.2's BT4.
- **Befund 14 (Inline-`<style>` im Template):** F-5.1-Disposition war "F5-REVIEW". F-5.2 ordnet als "nur Bug-Ticket" (BT4) ein, weil **keine UX-Heuristik-Komponente** vorhanden ist (User sieht keinen Unterschied; reine Architektur-Drift; CSP-strict ist im Single-User-LAN-Setup nicht aktiv). Analog zur Befund-6-Behandlung.
- **Befund 15 (Tot `<link id="preview-style">`):** F-5.1-Disposition war "F5-REVIEW". F-5.2 ordnet als "nur Bug-Ticket" (BT5) ein, weil **keine UX-Heuristik-Komponente** vorhanden ist (leeres `<link>` löst keinen Request aus, User sieht nichts). Analog zur Befund-6-Behandlung.
- **Befund 5 (Kein File-Info-Display):** F-5.1-Korrespondenz markierte als "F-1-Pattern P12 nicht-anwendbar mangels Anzeige". F-5.2 klassifiziert F10 mit **H1 statt H4** (F-1.2 F17 hatte H4 für MB-Bug). Begründung: F-1's H4-Klassifikation drehte sich um Konvention-Bruch (immer-MB statt KB/MB-Fallback); hier fehlt das Display **insgesamt** — der Schmerz ist primär System-Status-fehlt (H1), nicht Konvention-Bruch.
- **P10 (Result-Area scrollIntoView) und P11 (Drop-Zone Keyboard-Pfad):** F-5.1 markierte beide als "teil bzw. bereits-erfüllt" (P10 weil Layout-Lösung statt Scroll-Lösung; P11 weil native Input keyboard-erreichbar). F-5.2 verschärft auf "bereits konvergent" (kein Finding-Bedarf), weil die markdown_converter-Architektur das F-1-Problem strukturell vermeidet — kein UX-Schmerz mehr verbleibend.
- **Inventur-Befund-Aufteilung:** F-5.1 hatte 19 nummerierte Befunde. F-5.2 produziert 15 Findings + 5 Bug-Tickets, weil ein Befund in 2 Heuristik-Reihen aufgespalten wurde (Befund 1 → F1+F2), zwei Befunde Finding+BT-Disposition haben (Befund 4 → F6+BT2; Befund 16 → F3+BT1), drei Befunde zu Bug-only fielen (Befund 6, 14, 15) und zwei Befunde aus F-5.2 ausgenommen sind (Befund 8 DE-Pass, Befund 19 konformer Codestyle).

---

## Zusammenfassung

- **Heuristik-Findings gesamt:** 15
- **Davon Schweregrad 4 (kritisch):** 0 — kein Datenverlust-/Cost-/Blockade-Pfad auf der primären Reader-Aufgabe (Live-Preview funktioniert; PDF-Download und Library-Save haben Reibung aber keine Blockade)
- **Davon Schweregrad 3:** 3 (F1 H4 Save-`alert()`, F2 H9 Save-Recovery, F3 H9 PDF-Gen-Error-Re-Render ⚠️)
- **Davon Schweregrad 2:** 6 (F4 H1 Empty-Submit, F5 H9 Theme-CSS-silent ⚠️, F6 H1 Sample-Merge ⚠️, F7 H1 Submit-Loading-kurz ⚠️, F8 H4 Two-Dark-Modes ⚠️, F9 H6 Reader-Mode-Persistenz)
- **Davon Schweregrad 1:** 6 (F10 H1 File-Info-fehlt, F11 H6 Width-Buttons-Initial ⚠️, F12 H6 Esc-Document-global ⚠️, F13 H6 Iframe-a11y, F14 H1 Auto-Dismiss-fehlt, F15 H9 markdown-it-CDN ⚠️)
- **Reine Bug-Tickets (mit Ticket-Material):** 5 (BT1 PDF-Gen-Re-Render-Context, BT2 Sample-Merge-Template, BT3 updateStyle()-doppelt, BT4 Inline-`<style>`, BT5 Tot-`<link>`) — davon 2 mit Finding-Verknüpfung (BT1↔F3, BT2↔F6), 3 pure Bugs ohne H-Aspekt (BT3, BT4, BT5)
- **Pattern-Konvergenz-Quote (F-1-Patterns):** 12/14 = **86%** — im erwarteten Schwester-Feature-Inversions-Bereich 80-90%, drastisch höher als F-2.2 (~41%), F-3.2 (~35%) und F-4.2 (0%)
- **Cross-Feature-H4-Findings (Finding-level):** 7 von 15 (≈47%) — F1, F2, F4, F7, F10, F13, F14 mit F-1-Pattern-Korrespondenz; höher als F-2.2 (~41%) und F-3.2 (~35%) wegen Schwester-Feature-Status
- **Markdown-spezifische Findings:** 8 von 15 (≈53%) — F3, F5, F6, F8, F9, F11, F12, F15 ohne F-1-Korrespondent
- **`⚠️ code-only`-markierte Findings:** 8 (F3, F5, F6, F7, F8, F11, F12, F15) — Findings, deren visueller Effekt aus reinem Code-Reading nicht endgültig beurteilbar ist

**Heuristik-Verteilung:** H1 = 5 Findings (F4, F6, F7, F10, F14). H4 = 2 Findings (F1, F8). H6 = 4 Findings (F9, F11, F12, F13). H9 = 4 Findings (F2, F3, F5, F15). Summe 5+2+4+4 = 15 ✓ — Befund 1 wird über zwei Heuristik-Reihen abgebildet (F1 H4 + F2 H9), das Doppel-Zählen für eine Inventur-Wurzel ist konsistent zur F-1.2-/F-2.2-/F-3.2-Konvention. **H1 dominiert** wegen Visibility-Lücken (Empty-Submit-Roundtrip, File-Info-fehlt, Submit-Loading-kurz, Sample-Merge); **H6 deutlich** wegen Reader-Mode-Lifecycle-Lücken; **H9 deutlich** wegen Error-Pfade (PDF-Gen, Theme-Fetch, CDN-Fail, Save-Recovery).

**Schweregrad-Skala:**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse / Datenverlust- oder Cost-Pfad)

**Befund 16-Disposition (Master-Annotation aufgegriffen):** F3 trägt **Disposition Finding + Bug-Ticket BT1** mit primärer Heuristik **H9** und Severity **3 ⚠️ code-only** — Sub-Thread folgt der Master-Annotation 1:1, **kein Hot-Fix-Sprint** vorgeschlagen, **kein Code-Fix in der Doc**. F5-IMPL übernimmt den Fix; F5-PATTERNS greift "PDF-Gen-Error-Recovery" als eigenes Pattern mit Microcopy-Aspekt auf (Cluster 3).

**Master-Walkthrough-Empfehlung vor F5-PATTERNS:** Empfehlung **ja**, kurze Walkthrough-Session (15–30 Min) für die `⚠️ code-only`-Sev-3- und Sev-2-Findings — primär F3 (PDF-Gen-Error-Trigger forcieren via injizierten Style-Crash oder via curl-Trigger-Test um secondary 500 zu beobachten), F8 (globaler Theme-Toggle ↔ Reader-Mode-Dark-Toggle Reihenfolge-Test mit DevTools-Element-Inspector für `data-global-theme`/`data-theme`), F6 (nur sichtbar nach BT1-Fix, daher post-F5-IMPL-Smoke-pflichtig). F5 (Theme-CSS-Fetch silent) per DevTools-Network-404-Block 5-Min-Verifikation. F9 (Reader-Mode-Persistenz) ist code-evident — Walkthrough nicht zwingend, aber kurze Verifikation des "Reload startet immer in non-Reader"-Verhaltens günstig. Wenn Master-Bandwidth knapp ist: F5-PATTERNS kann auch ohne Walkthrough beginnen, aber die Pattern-Vorschläge für Cluster 3 (Error-Recovery) sollten das `⚠️ code-only`-Risiko explizit in den Phasen-Stops markieren, damit ein Implementierungs-Smoke vor Merge zwingend ist (analog F-3-IMPL und F-4-IMPL Smoke-Pflicht-Patterns).
