# Sprint F5-REVIEW — F-5.2 Heuristik-Review `markdown_converter`

**Datum**: 2026-05-10

**Ziel**: Stufe 2 (Heuristik-Review) der dreistufigen UX-Cascade-Methodik für `markdown_converter`. Die 19 Befunde aus der F-5.1-Inventur durch Nielsens Heuristiken H1/H4/H6/H9 filtern, Severity-ranken (1=kosmetisch … 4=kritisch), Disposition pro Befund finalisieren. **Schwester-Feature-Hebel**: `markdown_converter` ist das direkte Schwester-Feature zu `document_converter` (F-1) — die F-1.1-Korrespondenz-Spalte aus der Inventur ist die zentrale Heuristik-Filter-Eingabe für H4 (Konsistenz), und H1/H4/H6/H9-Klassifikationen aus F-1.2 sind direkt übertragbar wo F-1-Patterns korrespondieren. **Kein Pattern-Vorschlag, keine Microcopy** — das kommt in F-5.3.

**Vorbedingung**:
- Pytest 65/65 grün auf `main`. Letzter Code-Touch: F5-PICK (commit `efb13d2`, 2026-05-10).
- **Eingabe**: [docs/ui_inventory_markdown_converter_2026-05.md](docs/ui_inventory_markdown_converter_2026-05.md) (Sub-Thread liest komplett vor Phase 1).
  - 32 echte interaktive Elemente in 44 Tabellenzeilen.
  - 19 separate Befunde (Befund 19 ist konformer-Codestyle-Notiz `window.PageData`).
  - **F-1-Korrespondenz-Übersicht** am Doc-Ende: 4 direkt übertragbar (P4 / P7 / P13 / P14) + 8 teil-übertragbar bzw. bereits-erfüllt (P1 / P3 / P6 / P8 / P9 / P10 / P11 / P12) + 2 nicht-anwendbar (P2 / P5) → ≈86% Cross-Feature-H4-Quote-Erwartung.
  - **Helper-Reuse-Spuren-Sektion**: `showAlert` ✓ partial in Submit-Handler, drei Browser-`alert()`-Calls in `saveMarkdownToLibrary` ✗ (Befund 1, F-1-P4-Korrespondenz), `safeJSON` ✗ ungenutzt, `showToast` ✗ ungenutzt, `formatFileSize` ✗ mangels File-Info-Display.
  - 6 Code↔live-Divergenz-Verdachte (alle code-deduced).
  - 7 Live-Walkthrough-Lücken am Doc-Ende.
  - **Befund 16 akut-flag** im F5-PICK-Phase-1-Bericht: PDF-Gen-Error-Branch [app_pkg/markdown.py:246](app_pkg/markdown.py#L246) re-rendert Template ohne `themes`/`accepted_extensions`/`accepted_extensions_accept`-Context → secondary 500-Page statt flash-Banner. Master-Disposition: **in F5-IMPL mit-fixen, kein Hot-Fix-Sprint**. Siehe Master-Annotation unten.
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - F-1.2 Heuristik-Review: [docs/ui_findings_document_converter_2026-05.md](docs/ui_findings_document_converter_2026-05.md) — **primäre Vorlage und Heuristik-Klassifikations-Quelle** (Schwester-Feature).
  - F-2.2 Heuristik-Review: [docs/ui_findings_audio_converter_2026-05.md](docs/ui_findings_audio_converter_2026-05.md) — ~41% Cross-Feature-H4.
  - F-3.2 Heuristik-Review: [docs/ui_findings_library_detail_2026-05.md](docs/ui_findings_library_detail_2026-05.md) — ~35% Cross-Feature-H4.
  - F-4.2 Heuristik-Review: [docs/ui_findings_podcast_flow_2026-05.md](docs/ui_findings_podcast_flow_2026-05.md) — 0% Cross-Feature-H4 (Inversions-Vergleichsfall).
- **Produkt-Kontext** (für Severity-Bewertung): Single-User-App (Oliver), LAN-only, login-protected. Primäre `markdown_converter`-Aufgabe: Markdown eingeben oder hochladen → Theme wählen → Live-Preview im Iframe → optional Reader-Mode + Width-Auswahl + PDF-Download → optional „In Library speichern" (mit Notion-Integration). Daily-Usage-Schmerz-Gewichtung: zentraler Reader-Workflow für Oliver, Library-Save-Pfad und PDF-Download sind häufige Endpoints.
- **Methodik-Begründung**: Duan et al. *Heuristic Evaluation with LLMs* (CHI 2024), gefiltert durch Nielsen H1/H4/H6/H9.

**Out-of-scope**:
- Patterns + Microcopy (Stufe 3) — eigener Folge-Sprint `F5-PATTERNS`.
- Implementation — eigener Folge-Sprint `F5-IMPL`.
- Code-Änderungen jeglicher Art.
- **Befund 19 (konformer-Codestyle-Notiz `window.PageData`)**: keine UX-H-Komponente, als „aus F-5.2 ausgenommen, kein Heuristik-Aspekt" katalogisieren.
- Andere Features.

---

## Master-Annotation (vorab eingebettet)

### 1. Cross-Feature-H4-Erwartung 80-90% (Schwester-Feature-Inversion zu F-4.2)

`markdown_converter` ist das **direkte Schwester-Feature zu `document_converter`** (F-1) — beide sind Konversions-Pages mit File-Upload/Text-Input + Submit-Form-Pattern, und die F-1.1-Inventur diente als primäre Strukturvorlage für F-5.1. Die F-5.1-Korrespondenz-Übersicht am Doc-Ende zeigt 12 von 14 F-1-Patterns als direkt oder teil-übertragbar (≈86%) und nur 2 als nicht-anwendbar.

**Erwartung-Verschiebung gegenüber F-4.2**: F-4.2 hatte 0% Cross-Feature-H4 weil F-2 die Helper-Konvergenz im `audio_converter.js`-Podcast-Block schon durchgezogen hatte. Hier ist die Lage umgekehrt: F-1 hat die Konvergenz **noch nicht** auf `markdown_converter` durchgezogen, also ist die Cross-Feature-H4-Quote **drastisch hoch**. **Erwartung 80-90%**, nicht 15-25%.

**Methodik-Konsequenz**:
- **F-1-Korrespondenz-Spalte aus der Inventur ist die zentrale H4-Filter-Eingabe**. Direkt übertragbare F-1-Patterns sind in F-1.2 bereits heuristik-klassifiziert (H1/H4/H6/H9 + Severity) — die Klassifikation wandert 1:1 mit, nur Code-Anker und Microcopy-Beispiele werden auf den `markdown_converter`-Code umgemünzt.
- **Konvergenz-Items dürfen nicht künstlich „neu" entdeckt werden**. Wenn Befund 1 (drei `alert()`-Calls in `saveMarkdownToLibrary`) zu F-1-P4 korrespondiert, dann ist das Finding genau **eine Wiederholung von F-1-P4 für `markdown_converter`** — H4-Severity wird übernommen, nicht neu vergeben.
- **Bereits-erfüllte F-1-Patterns** (P3 setTimeout-Reset, P6 SEC-Sprint Format-Label, P8 SEC-Sprint Frontend-Vorab-Check, P11 native File-Input-Visibility) tauchen **nicht** als Findings auf — sie werden in der Cross-Feature-H4-Sektion als „bereits konvergent" gelistet, also positives Inventar.
- **Markdown-spezifische Befunde ohne F-1-Korrespondent** (Befund 4 Sample-Text-Merge, Befund 6 doppelter `updateStyle()`-Init-Call, Befund 9 Reader-Mode-State, Befund 10 Width-Buttons-Initial-Active, Befund 11 zwei Dark-Modes, Befund 14 Inline-`<style>`, Befund 15 toter `<link>`, Befund 16 PDF-Gen-Error-Re-Render, Befund 17 Esc-Key-Listener-Document-global, Befund 18 markdown-it CDN-Dependency) brauchen eigene Heuristik-Klassifikation und Severity. Das ist der „neue" Heuristik-Anteil dieses Sprints.

### 2. Befund 16 (PDF-Gen-Error-Re-Render) — in F5-IMPL mit-fixen

Master-Entscheidung: **kein Hot-Fix-Sprint**. Begründung:
- Trivial 1-2-Zeilen-Fix (`redirect(url_for('markdown_converter'))` statt `render_template` im Error-Branch).
- Kein Datenverlust-Risiko, nur sichtbare 500-Page statt freundlichem Banner.
- F5-IMPL kommt nach 2 weiteren S-Sprints (F5-REVIEW + F5-PATTERNS).
- In F5-PATTERNS wird Befund 16 als eigenes Pattern „PDF-Gen-Error-Recovery" mit Master-Annotation aufgegriffen.

**Sub-Thread-Disposition für F5-REVIEW**: Befund 16 trägt **Disposition: Finding + Bug-Ticket** (analog F-3.2-BT-Pattern). Heuristik primär **H9 (Help users recognise/diagnose/recover from errors)** weil Error-Recovery-Pfad selbst kaputt ist. Severity-Vorschlag **Sev 3** (täglich-trifft-nicht aber bei PDF-Failure trifft hart) — Sub-Thread kann abweichen. **Kein** Bug-Ticket-Inhalt mit Code-Fix-Vorschlag, das macht F5-PATTERNS.

### 3. F-1-Korrespondenz-Spalte als Heuristik-Filter-Eingabe

**Mechanik**: Sub-Thread liest F-1.2-Findings-Doc und mappt jeden F-5.1-Befund mit F-1-Korrespondenz auf das entsprechende F-1-Finding. F-1-Finding hat bereits:
- Heuristik (H1/H4/H6/H9)
- Severity (1-4)
- Beschreibung-Schema
- Disposition-Pattern

**Sub-Thread übernimmt**:
- Heuristik 1:1.
- Severity 1:1 — außer Daily-Usage-Schmerz-Gewichtung schiebt (z.B. wenn `markdown_converter` häufiger genutzt wird als `document_converter` und ein F-1-Sev-2 daher hier Sev 3 ist; Sub-Thread begründet im Bericht).
- Code-Anker auf `markdown_converter`-Code (statt `document_converter`).
- Disposition-Vorschlag.

**Sub-Thread weicht ab nur wenn**:
- F-1-Sev war auf `document_converter`-Spezifika begründet die hier nicht greifen.
- F-1-Heuristik war auf einen Sub-Aspekt fokussiert der hier anders liegt (z.B. drag-drop-spezifisch — wäre in F-1 nicht so klassifiziert weil F-1 keine drag-drop-Reibung hatte).

**Cross-Feature-H4-Sektion strukturiert dies**:
- 4 direkt-übertragbare Patterns → 4 Findings mit „F-1-Korrespondenz: P4/P7/P13/P14" und übernommener H/Sev.
- 8 teil-übertragbar oder bereits-erfüllt → Sub-Thread klassifiziert pro Pattern: was ist bereits-erfüllt und gehört in „bereits konvergent"-Liste? was bleibt teil-Befund mit übernommener Heuristik?
- 2 nicht-anwendbar → werden nicht zu Findings.
- 10 markdown-spezifische Befunde → eigene Heuristik-Klassifikation.

---

## Phase 1 — Heuristik-Review

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Inventur-Doc komplett lesen**: alle 19 Befunde, F-1-Korrespondenz-Übersicht, Helper-Reuse-Spuren-Sektion, Live-Walkthrough-Lücken.
3. **F-1.2-Findings-Doc lesen** und Korrespondenz-Mapping vorbereiten (welches F-1-Finding gehört zu welchem F-1-Pattern aus der Korrespondenz-Übersicht).
4. **F-2.2 / F-3.2 / F-4.2 lesen** für Format-Treue und für die Cross-Feature-H4-Disziplin (F-4.2 zeigt insbesondere wie ehrliche Begründung für unerwartete Quote aussieht).

**Heuristik-Filter** (mit Schwester-Feature-Hebel):

- **H1 — Visibility of system status**: Iframe-Loading-State, Theme-CSS-Fetch-Latenz-Sichtbarkeit, Submit-Loading-Sichtbarkeit nach CSRF-Refresh-Roundtrip, File-Info-Display (Filename + Size nach Auswahl).
- **H4 — Consistency and standards**: **dominanter Filter dieses Sprints**. F-1-Korrespondenz steuert. Helper-Reuse-Drift (`alert()` statt `showAlert`, `safeJSON` ungenutzt, `showToast` ungenutzt). Cross-Feature-Konvergenz-Vektor.
- **H6 — Recognition rather than recall**: Reader-Mode-Toggle-Sichtbarkeit (welche Buttons sind aktiv), Width-Buttons-Default-Markierung, Theme-Auswahl-Persistenz-Erwartung, PDF-Download-vs-Library-Save-Workflow.
- **H9 — Help users recognise, diagnose, recover from errors**: PDF-Gen-Error-Re-Render (Befund 16 akut), Theme-CSS-Fetch-Failure silent (Befund 3), Save-Failure-`alert()` (Befund 1, F-1-P4-Korrespondenz), CSRF-Refresh-vs-Re-Submit-Race.

**Severity-Skala 1–4** (analog F-1.2 / F-2.2 / F-3.2 / F-4.2):

- **Sev 4 — Kritisch**: blockiert Kernaufgabe oder Datenverlust/Cost-Risiko.
- **Sev 3 — Hoch**: spürbarer Reibungspunkt der täglich trifft.
- **Sev 2 — Mittel**: stört einmalig.
- **Sev 1 — Kosmetisch**: würde gefixt wenn man sowieso vorbeikommt.

**`⚠️ code-only`-Marker**: 6 Code↔live-Divergenz-Verdachte aus F-5.1 + 7 Live-Walkthrough-Lücken. Findings die diesen Status erben mit `⚠️ code-only`-Hinweis in Severity-Spalte ergänzen — Master entscheidet vor F-5.3 ob Walkthrough nötig.

**Disposition-Logik pro Befund** (analog F-3.2 / F-4.2):

- **Nur Finding** (kommt in Patterns-Sprint).
- **Finding + Bug-Ticket** (kommt in Patterns-Sprint **plus** separates Bug-Ticket — z.B. Befund 16 PDF-Gen-Error).
- **Nur Bug-Ticket** (kommt **nicht** in Patterns-Sprint, pure Bug ohne UX-H-Komponente — z.B. Befund 14 Inline-`<style>` falls kein UX-H-Aspekt erkannt, Befund 15 toter `<link>`).
- **Bereits konvergent** (NEU für F-5.2 wegen Schwester-Feature-Status): F-1-Pattern bereits durch frühere Sprints für `markdown_converter` erfüllt. In Cross-Feature-H4-Sektion als positives Inventar listen, nicht in Findings-Tabelle. Quelle: F-5.1 Korrespondenz-Übersicht „bereits-erfüllt"-Markierung.
- **Aus F-5.2 ausgenommen** — wenn ein Inventur-Befund beim Heuristik-Filter „verschwindet", explizit notieren mit Begründung („gehört zu Hygiene-Welle X" oder „kein UX-H-Aspekt erkannt"). Mindest-Kandidat: Befund 19 (konformer-Codestyle-Notiz).

**Output-Doc**: `docs/ui_findings_markdown_converter_2026-05.md`. Struktur (1:1 wie F-1.2 / F-2.2 / F-3.2 / F-4.2 plus Schwester-Feature-Sektion):

1. Header mit Inventur-Quelle, Sprint-Datum, Methodik-Hinweis, **Schwester-Feature-Hebel-Notiz** (Cross-Feature-H4-Erwartung 80-90%, F-1.2 als primäre Heuristik-Quelle).
2. **Findings-Tabelle**: # · Heuristik (H1/H4/H6/H9) · Severity (1–4, mit `⚠️ code-only` wo nicht live verifiziert) · Beschreibung in 1–2 Sätzen · Inventur-Befund-Anker (Nr. aus F-5.1) · F-1-Korrespondenz (Pattern-Code aus F-1, oder „—" wenn markdown-spezifisch) · Disposition.
3. **Reine Bug-Tickets** (separat, nicht in Findings-Tabelle): Liste mit Code-Anker, Beschreibung, Reproduktion. **Kein Fix-Pfad-Vorschlag** — das macht F5-PATTERNS.
4. **Disposition-Verteilung**: Statistik. Sub-Thread begründet wo F-5.1-Pre-Disposition-Vorschlag abgewichen wurde.
5. **Cross-Feature-H4-Sektion** (NEU strukturiert für Schwester-Feature):
   - **Direkt übertragbare Konvergenz-Items**: Liste der Findings mit F-1-Korrespondenz, übernommener Heuristik+Severity, Code-Anker auf `markdown_converter`.
   - **Bereits konvergente F-1-Patterns**: Liste der F-1-Patterns die für `markdown_converter` schon erfüllt sind (kein Finding nötig). Quelle: F-5.1 „bereits-erfüllt"-Markierungen plus F-1-Korrespondenz.
   - **Nicht-anwendbare F-1-Patterns**: 2 Items (P2 Result-Area, P5 Drag-Active-Highlight) mit Begründung.
   - **Konvergenz-Quote**: ((direkt übertragbar + bereits konvergent) / (alle anwendbaren F-1-Patterns)) %. Erwartung 80-90%.
6. **Markdown-spezifische Heuristik-Sub-Sektion** (NEU für F-5.2): Findings ohne F-1-Korrespondent — die 10 Befunde aus F-5.1 die markdown-eigen sind (Reader-Mode-State, zwei Dark-Modes, Iframe-Loading, PDF-Gen-Error-Re-Render, Esc-Key-Listener-Document-global, etc.). Kurze Heuristik-Verteilung dieser markdown-spezifischen Findings.
7. **Schwerpunkt-Cluster**: 2-4 thematische Cluster wo schwere Findings sich konzentrieren. Erwartung: 1× „Cross-Feature-H4-Helper-Reuse zu F-1" (Befund 1 + Befund 7 + ggf. Befund 13 + ggf. 14) + 1× „Reader-Mode-State-Persistenz und Visual-Layout (H1+H6)" (Befund 9 + 10 + 11) + 1× „Error-Recovery-Pfade (H9)" (Befund 16 PDF-Gen + Befund 3 Theme-CSS-Fetch-Silent + Befund 1 Save-Failure-`alert()`) + ggf. „CDN-Dependency und Iframe-a11y" (Befund 18 + Befund 13).

Nach Phase 1: STOP — Bericht. Statistik (Findings pro Sev-Stufe, Bug-Tickets-Anzahl, Cross-Feature-H4-Quote, Schwerpunkt-Cluster). Plus: Disposition für Befund 16 (Master-Annotation prüfen, ob Sub-Thread Heuristik/Severity abweicht). Plus: ob Master-Walkthrough-Nachreichung empfohlen wird (analog F-3.2 / F-4.2).

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Findings-Doc nochmal und prüft:

1. **Vollständigkeit**: jeder der 19 Inventur-Befunde ist disponiert (entweder in Findings-Tabelle, in Bug-Tickets-Sektion, in Cross-Feature-H4-„bereits konvergent"-Liste, oder als „aus F-5.2 ausgenommen, gehört zu …"). Mindest-Kandidat für Letzteres: Befund 19.
2. **F-1-Korrespondenz-Übernahme**: jeder Finding mit F-1-Korrespondenz hat dieselbe Heuristik+Severity wie das F-1-Finding, **oder** Sub-Thread hat Abweichung im Bericht begründet (z.B. Daily-Usage-Schmerz-Gewichtung).
3. **Heuristik-Klarheit**: jeder Finding hat genau eine Primary-Heuristik. Sekundär-Heuristiken in Beschreibung erwähnen aber nicht in Filter-Spalte.
4. **Severity-Konsistenz**: Sev 4 nur für Datenverlust/Cost/Blockade-Pfade. Sev 1 nur für rein kosmetisch. Daily-Usage-Schmerz-Gewichtung dokumentiert wo Stufe verschoben.
5. **Cross-Feature-H4-Quote**: liegt im erwarteten Bereich (80-90%) oder Abweichung ist begründet (analog F-4.2-Begründungsdisziplin in entgegengesetzter Richtung).
6. **Bereits-konvergente F-1-Patterns**: Liste ist vollständig und Code-Anker stimmen mit F-5.1-Markierungen überein.
7. **`⚠️ code-only`-Marker**: jeder Finding aus den 6 Code-deduced-Inventur-Verdachten + den 7 Live-Walkthrough-Lücken trägt den Marker.
8. **Befund 16-Disposition**: Master-Annotation aufgegriffen — als „Finding + Bug-Ticket" disponiert, **kein** Hot-Fix-Sprint-Vorschlag, **kein** Code-Fix in der Doc.
9. **Disziplin**: kein Pattern-Vorschlag, keine konkrete Microcopy, kein Bug-Fix.

Nach Phase 2: STOP — Bericht. „Findings-Doc konsistent" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-5.2 / Stufe 2: heuristic review of markdown_converter".
- Body: Statistik (Findings pro Sev-Stufe, Bug-Tickets-Anzahl, Cross-Feature-H4-Quote, Schwerpunkt-Cluster). Master-Annotations-Aufgriff (Befund 16 als Finding+BT disponiert für F5-IMPL-Mit-Fix, F-1-Korrespondenz-Übernahme-Quote).
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints. Wenn der Auto-Mode-Classifier blockt: im Phase-3-Bericht erwähnen, Master pusht von Hand. (Siehe Memory `feedback_push_is_normal.md` und Sprint-Prompt-Konvention.)

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S–M** — eine Output-Datei (`docs/ui_findings_markdown_converter_2026-05.md`), Heuristik-Filter über 19 Inventur-Befunde, F-1-Korrespondenz-Mapping als zentrale Mechanik (M-Anteil weil F-1.2-Cross-Read und Per-Pattern-Übernahme-Disziplin Aufwand sind), Cross-Feature-H4-Sektion neu strukturiert mit Konvergenz-Quote, kein Code-Touch, keine Tests, kein Smoke.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim F-1-Korrespondenz-Mapping ein F-5.1-Befund **mit „teil-übertragbar"-Markierung** beim Heuristik-Filter klar als „bereits konvergent" oder umgekehrt als „voll H4-Finding" rauskommt: in Cross-Feature-H4-Sektion explizit notieren mit Begründung („Sub-Thread hat F-5.1-`teil-übertragbar`-Markierung verschärft auf `bereits konvergent`, weil …").
- Wenn beim Lesen der `⚠️ code-only`-Befunde Zweifel an der Code-Deduktion auftauchen: kurz im Bericht aufzählen, **nicht** in der Findings-Doc als „bereits erfüllt" — Master verifiziert.
- Wenn ein markdown-spezifischer Befund (ohne F-1-Korrespondenz) sich beim Heuristik-Filter als **zu spezifisch für eine einzelne Heuristik** zeigt (z.B. Befund 11 zwei Dark-Modes — H4 für Konsistenz vs. H6 für Recognition): Primary einzeln wählen mit Begründung, Sekundär in Beschreibung.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F5-REVIEW ☑ done 2026-05-XX → commit `<hash>`. Findings-Doc unter `docs/ui_findings_markdown_converter_2026-05.md`. <Findings-Anzahl> Findings (Sev 4: X, Sev 3: Y, Sev 2: Z, Sev 1: W) + <Bug-Tickets> Bug-Tickets. Cross-Feature-H4-Quote: <%>%. Befund 16 als Finding+BT für F5-IMPL-Mit-Fix disponiert. Verbleibende Sequenz: F5-PATTERNS → F5-IMPL → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F5-REVIEW" raus → Erledigt-Liste; Sektion „2. F-N…" rückt auf Position 1 (zwischengeschoben werden F5-PATTERNS und F5-IMPL durch den Master beim nächsten Dispatch — der Sub-Thread fügt sie nicht hinzu, das ist Master-Edit-Zone).
- **Memory**: nur wenn übertragbare Lehren für Schwester-Feature-Inventur-Methodik auftauchen (z.B. „Schwester-Feature-Inversion bei Cross-Feature-H4-Erwartung — F-X.2-Korrespondenz-Spalte als Heuristik-Filter-Eingabe spart Klassifikations-Aufwand"). Defensiv.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Heuristik-Review-Methodik ist seit F-1.2 / F-2.2 / F-3.2 / F-4.2 etabliert, Schwester-Feature-Hebel und Befund-16-Disposition in Master-Annotation oben verankert.)_
