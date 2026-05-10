# Sprint F6-REVIEW — F-6.2 Heuristik-Review `library` List-View

**Datum**: 2026-05-10

**Ziel**: Stufe 2 (Heuristik-Review) der dreistufigen UX-Cascade-Methodik für die `library` List-View. Die 16 Befunde aus der F-6.1-Inventur durch Nielsens Heuristiken H1/H4/H6/H9 filtern, Severity-ranken (1=kosmetisch … 4=kritisch), Disposition pro Befund finalisieren. **Geschwister-Feature-Hebel zu F-3**: die F-3-Korrespondenz-Spalte aus der F-6.1-Inventur ist die zentrale Heuristik-Filter-Eingabe für H4 (Konsistenz), und H1/H4/H6/H9-Klassifikationen aus F-3.2 sind direkt übertragbar wo F-3-Patterns korrespondieren. **Kein Pattern-Vorschlag, keine Microcopy** — das kommt in F-6.3.

**Vorbedingung**:
- Pytest 66/66 grün auf `main`. Letzter Code-Touch: F6-PICK (commit `538b5e7`, 2026-05-10).
- **Eingabe**: [docs/ui_inventory_library_list_2026-05.md](docs/ui_inventory_library_list_2026-05.md) (Sub-Thread liest komplett vor Phase 1).
  - 21 echte interaktive Elemente in 30 Tabellenzeilen.
  - **16 separate Befunde** mit Disposition-Vorschlag aus F-6.1: 4 Finding-+-Bug-Ticket-Kandidaten, 10 nur-Finding, 1 Sammel-Befund (EN-Strings), 1 Helper-Reuse-Beobachtung (#16).
  - **F-3-Korrespondenz-Übersicht** am Doc-Ende: 6 direkt übertragbar (P1 / P3 / P6 / P8 / P14 / P15) + 2 teil-übertragbar (P2 / P5) + 3 bereits-erfüllt (P9 / P11 / P12) + 4 nicht-anwendbar (P4 / P7 / P10 / P13) → **Pattern-Konvergenz-Quote 53%**.
  - **List-View-States-Sub-Sektion**: 6 Klassen kartiert (Sortierung / Filter+Favorites / Suche / Bulk-existiert-nicht / Pagination / Empty-State nicht filter-aware) — alle 4 ersten URL-getrieben.
  - **Helper-Reuse-Spuren-Sektion**: `fallbackCopyText` ✓ und `showToast` ✓ genutzt. `showAlert` ✗ (kein Mountpoint — Befund-Trigger für F-3-P15-Korrespondenz). `safeJSON` ✗ in PUT+DELETE-Pfaden. **`saveViewState/loadViewState` zweite Call-Site: nein** (URL-Persistierung ist Design-Wahl). **`confirmInPlace` zweite Call-Site: nein** (kein Bulk-Delete-Pfad).
  - **10 fehlende States**, **4 Code↔live-Divergenzen** (alle code-deduced), **9 Live-Walkthrough-Lücken** (Locale-Check, Card-Hover-Lift, CSRF-Verhalten, etc.).
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - **F-3.2 Heuristik-Review**: [docs/ui_findings_library_detail_2026-05.md](docs/ui_findings_library_detail_2026-05.md) — **primäre Geschwister-Feature-Vorlage und Heuristik-Klassifikations-Quelle**. 17 Findings + 8 Bug-Tickets, ~35% Cross-Feature-H4-Quote.
  - F-1.2 Heuristik-Review: [docs/ui_findings_document_converter_2026-05.md](docs/ui_findings_document_converter_2026-05.md) — 19 Findings.
  - F-2.2 Heuristik-Review: [docs/ui_findings_audio_converter_2026-05.md](docs/ui_findings_audio_converter_2026-05.md) — ~41% Cross-Feature-H4.
  - F-4.2 Heuristik-Review: [docs/ui_findings_podcast_flow_2026-05.md](docs/ui_findings_podcast_flow_2026-05.md) — 0% (Inversions-Vergleichsfall).
  - F-5.2 Heuristik-Review: [docs/ui_findings_markdown_converter_2026-05.md](docs/ui_findings_markdown_converter_2026-05.md) — 86% Pattern-Konvergenz / 47% Finding-Quote (Schwester-Feature-Hebel).
- **Produkt-Kontext** (für Severity-Bewertung): Single-User-App (Oliver), LAN-only, login-protected. **Library als zentraler Reader-Ersatz** für Readwise-Replacement laut [project_readwise_replacement.md](file:///Users/olivergluth/.claude/projects/-Volumes-MintHome-CODE-CONVERTER/memory/project_readwise_replacement.md) — **Daily-Usage-Schmerz hoch** für List-View-Reibung (Sortierung-Reset, Filter-Stale, Tag-Click-Reibung, Suche-Live-vs-Submit-Inkonsistenz, Copy-Btn-Daten-Verlust). Liste-Navigation ist Häufigkeits-Hotspot, weil jede Konvertierung von dort aus startet/wieder gefunden wird.
- **Methodik-Begründung**: Duan et al. *Heuristic Evaluation with LLMs* (CHI 2024), gefiltert durch Nielsen H1/H4/H6/H9.

**Out-of-scope**:
- Patterns + Microcopy (Stufe 3) — eigener Folge-Sprint `F6-PATTERNS`.
- Implementation — eigener Folge-Sprint `F6-IMPL`.
- Code-Änderungen jeglicher Art.
- **F-3.2 BT7 + BT8** (textarea-escape, window.open-noopener) — gehören zu library_detail, nicht zu library-List-View. Aus Scope.
- **Befund 16 (Helper-Reuse-Beobachtung als Meta-Befund)** aus F-6.1: nicht als eigener Heuristik-Finding, sondern in der Cross-Feature-H4-Sektion als positives Inventar / methodische Reflexion (siehe Master-Annotation 5).
- Andere Features (`mermaid_converter`, `login`) — eigene Folge-Wellen.

---

## Master-Annotation (vorab eingebettet)

### 1. F-3-Korrespondenz-Spalte als Heuristik-Filter-Eingabe

`library` List-View ist **Geschwister-Feature zu `library_detail`** (F-3) — selbe Datenklasse, andere View-Klasse. Die F-6.1-Korrespondenz-Übersicht zeigt 6 direkt-übertragbare F-3-Patterns + 2 teil-übertragbare + 3 bereits-erfüllt + 4 nicht-anwendbar. **F-3.2-Findings-Doc ist die primäre Heuristik-Klassifikations-Quelle** für Findings mit F-3-Korrespondenz.

**Methodik-Konsequenz** (analog F-5.2's F-1-Korrespondenz-Mechanik):
- **F-3-Findings mit Korrespondenz wandern 1:1 mit**: Heuristik (H1/H4/H6/H9) und Severity (1–4) werden übernommen, **außer** Daily-Usage-Schmerz-Gewichtung schiebt (z.B. wenn library List-View ein F-3-Pattern häufiger trifft als library_detail → Sev hochstufen; Sub-Thread begründet im Bericht).
- **Konvergenz-Items dürfen nicht künstlich „neu" entdeckt werden**. Wenn F-6.1 Befund 2 (silent-fail `toggleFavorite`) zu F-3.2-Finding (P1 Auto-Save Failure-Banner-Pattern) korrespondiert: H4+H9 und Severity wandern mit.
- **Bereits-erfüllte F-3-Patterns** (P9 Tag-Chips Server-side, P11 Sidebar-Active via `base.html`-path-Match, P12 file_size-Filter): **nicht als Findings** — in Cross-Feature-H4-Sektion als „bereits konvergent"-Liste.
- **Nicht-anwendbare F-3-Patterns** (P4/P7/P10 Notion-only, P13 Title-Edit-only): erscheinen nicht als Findings, in Cross-Feature-H4-Sektion mit Begründung gelistet.
- **List-spezifische Findings ohne F-3-Korrespondent** (Sortierung-Reset, Filter-Stale, Suche-Live-vs-Submit-Inkonsistenz, Pagination-URL-Artifact, Empty-State nicht filter-aware, Copy-Btn-Daten-Verlust): eigene Heuristik-Klassifikation. Das ist der „neue" Heuristik-Anteil dieses Sprints.

### 2. Cross-Feature-H4-Finding-Quote-Erwartung 35-50%

Pattern-Konvergenz 53% auf F-6.1-Pattern-Ebene heißt: **Cross-Feature-H4-Finding-Quote** wahrscheinlich **35-50%** (Pattern-zu-Finding-Übertragung hat Verluste durch list-spezifische Differenzen). Vergleichswerte:
- F-3.2: 35% (`library_detail` zu allen anderen — nicht-Geschwister).
- F-2.2: ~41%.
- F-5.2: 47% (Schwester-Feature zu F-1).
- F-4.2: 0% (Helper-Konvergenz schon gemacht).

**Sub-Thread darf abweichen** wenn ehrliche Begründung — F-4.2 hat 0%-Quote mit „Helper schon konvergiert" begründet, analog hier könnte z.B. „URL-Persistierung-Design-Wahl macht 2-3 erwartete Findings hinfällig" eine Begründung sein. **Keine künstlichen H4-Findings konstruieren** wie F-4.2-Annotation verlangt.

### 3. Befund 1 (Copy-Btn 200-char-Preview) — Sev-3-Vorschlag

Master-Disposition-Vorschlag: **Finding + Bug-Ticket**, Heuristik primär **H9** (Help users recognise/diagnose/recover from errors — User denkt, sie haben Full-Content kopiert, hat aber nur Preview = Daten-Verlust-light-Pfad ohne Recovery-Hinweis), Severity-Vorschlag **Sev 3** (Daily-Usage-Schmerz-Gewichtung — Copy-Pfad ist häufig genutzt für Workflow-Ketten, irreführendes Verhalten trifft hart wenn User es nicht merkt). Sub-Thread kann abweichen mit Begründung. **Kein** Bug-Ticket-Inhalt mit Code-Fix-Vorschlag, das macht F6-PATTERNS.

### 4. Befunde 2+3 (silent-fail toggleFavorite / deleteConversion) — F-3-P1/P3-Übernahme

**Direkte F-3-Korrespondenz** zu F-3.2-Findings (silent-fail Auto-Save-Update + silent-fail Delete). H4+H9-Heuristik und Severity (Sev 3 in F-3.2 erinnert sich, Sub-Thread liest die F-3.2-Findings-Doc) wandern 1:1 mit. **Verzahnung-Notiz**: BT-Tickets aus F-3.2 für diese F-3-Findings sind dort schon disponiert; in F-6.2 als „Finding + Bug-Ticket-Kandidat (analog F-3-BT-Mechanik)" katalogisieren.

### 5. `saveViewState/loadViewState` URL-Persistierungs-Reflexion in Cross-Feature-H4-Sektion

F-6.1 hat eine **wichtige methodische Reflexion** durchgezogen: `saveViewState/loadViewState`-Helper bleibt single-call-site, **weil URL-Query-Param-Persistierung die etablierte und korrekte Design-Wahl** für List-View-State ist (bookmarkable, sharable, browser-back-restoriert). Das ist **keine** H4-Verletzung sondern eine begründete Design-Wahl.

**Cross-Feature-H4-Sektion** soll dies als **positive Helper-Disziplin-Beobachtung** dokumentieren (analog F-4.2's „F-2-Helper-Konvergenz schon gemacht"-Begründung der 0%-Quote, hier umgekehrt: „Helper-Reuse-Drift hat begründete Alternative"). Methodische Lehre für künftige Wellen: **Helper-Reuse-Drift bedeutet nicht zwingend H4-Verletzung wenn die Alternative eine begründete Design-Wahl ist** (URL vs. localStorage, Server-Side-Sort vs. Client-Side-Filter, etc.).

Daraus folgt: F-6.1 Befund 16 (Helper-Reuse-Beobachtung als Meta-Befund) wird **nicht zum Heuristik-Finding** — es wird zur methodischen Reflexion in der Cross-Feature-H4-Sektion (siehe Out-of-scope).

### 6. Live-Walkthrough-Disposition deferred zur F-6.3-Phase

F-6.1 hat 9 Live-Walkthrough-Lücken dokumentiert. Master-Entscheidung analog F-5.2/F-5.3-Methodik: **kein expliziter Master-Walkthrough vor F6-PATTERNS**. Stattdessen tragen ⚠️ code-only-Findings in F-6.2 den `⚠️ code-only`-Marker, und in F6-PATTERNS werden Patterns für ⚠️-Findings mit `🔥 Smoke-Pflicht in F6-IMPL`-Tag versehen (analog F-3-IMPL- und F-4-IMPL- und F-5-IMPL-Methodik).

**Sub-Thread soll** ⚠️ code-only-Marker konsequent setzen für alle 4 Code↔live-Divergenz-Verdachte und alle Findings die ohne Browser-Verifikation nicht abschließend bewertbar sind (Locale-Check, Card-Hover-Robustheit, CSRF-Verhalten, etc.).

---

## Phase 1 — Heuristik-Review

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Inventur-Doc komplett lesen**: alle 16 Befunde, F-3-Korrespondenz-Übersicht, List-View-States-Sub-Sektion, Helper-Reuse-Spuren-Sektion, Live-Walkthrough-Lücken.
3. **F-3.2-Findings-Doc lesen** und Korrespondenz-Mapping vorbereiten (welches F-3-Finding gehört zu welchem F-6.1-Befund mit F-3-Korrespondenz).
4. **F-5.2 / F-4.2 lesen** für Format-Treue und Cross-Feature-H4-Disziplin (F-4.2's ehrliche Begründung für 0%-Quote ist Vorlage; F-5.2's 47% mit Schwester-Feature-Hebel-Begründung ist Vorlage).

**Heuristik-Filter** (mit Geschwister-Feature-Hebel und Daily-Usage-Schmerz-Gewichtung):

- **H1 — Visibility of system status**: Sortier-Header-Active-State, Filter-Active-Indicator-Sichtbarkeit, Suche-Loading vs. Submit-Required-Unterschied, Pagination-Active-Page-Markierung, Favorite-Toggle-Loading-Sichtbarkeit, Delete-Loading-Sichtbarkeit, Empty-State filter-aware vs. global-empty.
- **H4 — Consistency and standards**: **dominanter Filter dieses Sprints** wegen F-3-Korrespondenz. F-3-Korrespondenz-Spalte aus F-6.1 steuert. Helper-Reuse-Drift wo nicht durch URL-Design-Wahl gerechtfertigt.
- **H6 — Recognition rather than recall**: Filter-Pill-Active-Visual, Tag-Chip-Click-vs-Hover, Card-Hover-Lift als Click-Affordance, Sortier-Asc/Desc-Indikator, Bulk-Aktion-Existenz-Sichtbarkeit (gibt es überhaupt einen Bulk-Pfad?), Copy-Btn-Tooltip-Klarheit.
- **H9 — Help users recognise, diagnose, recover from errors**: Copy-Btn-Daten-Verlust-Recovery (Befund 1), silent-fail toggleFavorite (Befund 2), silent-fail deleteConversion (Befund 3), Suche-no-results-Recovery, Filter-leere-Liste-Recovery, CSRF-Refresh-Race in API-Calls.

**Severity-Skala 1–4** (analog F-3.2 / F-5.2):

- **Sev 4 — Kritisch**: blockiert Kernaufgabe oder Datenverlust/Cost-Risiko. **Daily-Usage-Schmerz-Gewichtung wichtig** wegen Library-als-Readwise-Ersatz.
- **Sev 3 — Hoch**: spürbarer Reibungspunkt der täglich trifft.
- **Sev 2 — Mittel**: stört einmalig.
- **Sev 1 — Kosmetisch**: würde gefixt wenn man sowieso vorbeikommt.

**`⚠️ code-only`-Marker**: 4 Code↔live-Divergenz-Verdachte aus F-6.1 + 9 Live-Walkthrough-Lücken. Findings die diesen Status erben mit `⚠️ code-only`-Hinweis in Severity-Spalte ergänzen — Master entscheidet vor F-6.3 ob Walkthrough nötig oder als 🔥 Smoke-Pflicht in F6-IMPL.

**Disposition-Logik pro Befund** (analog F-3.2 / F-4.2 / F-5.2):

- **Nur Finding** (kommt in Patterns-Sprint).
- **Finding + Bug-Ticket** (kommt in Patterns-Sprint **plus** separates Bug-Ticket — z.B. Befund 1 Copy-Btn / Befunde 2+3 silent-fails).
- **Nur Bug-Ticket** (kommt **nicht** in Patterns-Sprint, pure Bug ohne UX-H-Komponente).
- **Bereits konvergent** (Geschwister-Feature-Hebel): F-3-Pattern bereits durch frühere Sprints für library List-View erfüllt. In Cross-Feature-H4-Sektion als positives Inventar listen, nicht in Findings-Tabelle. Quelle: F-6.1 Korrespondenz-Übersicht „bereits-erfüllt"-Markierung (P9 Tag-Chips, P11 Sidebar-Active, P12 file_size-Filter).
- **Aus F-6.2 ausgenommen** — wenn ein Inventur-Befund beim Heuristik-Filter „verschwindet", explizit notieren mit Begründung. Mindest-Kandidat: Befund 16 (Helper-Reuse-Meta-Beobachtung → in Cross-Feature-H4-Sektion absorbiert).

**Output-Doc**: `docs/ui_findings_library_list_2026-05.md`. Struktur 1:1 wie F-1.2 / F-2.2 / F-3.2 / F-4.2 / F-5.2 plus Geschwister-Feature-Sektion:

1. Header mit Inventur-Quelle, Sprint-Datum, Methodik-Hinweis, **Geschwister-Feature-Hebel-Notiz** (F-3.2 als primäre Heuristik-Quelle, Daily-Usage-Schmerz-Gewichtung wegen Library-Readwise-Replacement).
2. **Findings-Tabelle**: # · Heuristik (H1/H4/H6/H9) · Severity (1–4, mit `⚠️ code-only` wo nicht live verifiziert) · Beschreibung in 1–2 Sätzen · Inventur-Befund-Anker (Nr. aus F-6.1) · F-3-Korrespondenz (Pattern-Code aus F-3 wie „P1" / „P3" / „P14", oder „—" wenn list-spezifisch) · Disposition.
3. **Reine Bug-Tickets** (separat, nicht in Findings-Tabelle): Liste mit Code-Anker, Beschreibung, Reproduktion. **Kein Fix-Pfad-Vorschlag** — das macht F6-PATTERNS.
4. **Disposition-Verteilung**: Statistik. Sub-Thread begründet wo F-6.1-Pre-Disposition-Vorschlag abgewichen wurde.
5. **Cross-Feature-H4-Sektion** (Geschwister-Feature-strukturiert):
   - **Direkt übertragbare Konvergenz-Items**: Liste der Findings mit F-3-Korrespondenz, übernommener Heuristik+Severity, Code-Anker auf `library.js` / `library.html` / `library.py`.
   - **Bereits konvergente F-3-Patterns**: P9 Tag-Chips / P11 Sidebar-Active / P12 file_size-Filter als positives Inventar mit Code-Anker.
   - **Nicht-anwendbare F-3-Patterns**: P4 / P7 / P10 (Notion-only) / P13 (Title-Edit-only) mit Begründung.
   - **Helper-Reuse-Reflexion** (Master-Annotation 5): `saveViewState/loadViewState`-URL-Persistierungs-Begründung als positive Disziplin-Notiz; `confirmInPlace`-kein-Bulk-Delete-Begründung; `showAlert`-Mountpoint-fehlt-als-Befund-Verzahnung (das ist ein **echter** H4-Finding, weil F-3-P15 Banner-Mountpoint korrespondiert).
   - **Konvergenz-Quote**: Cross-Feature-H4-Finding-Quote in %. Erwartung 35-50%.
6. **List-spezifische Heuristik-Sub-Sektion**: Findings ohne F-3-Korrespondent (List-View-State-Klassen: Sortierung / Filter / Suche / Pagination / Empty / Copy-Btn-Verhalten). Kurze Heuristik-Verteilung dieser list-spezifischen Findings.
7. **Schwerpunkt-Cluster**: 2-4 thematische Cluster wo schwere Findings sich konzentrieren. Erwartung-Vorschlag: 1× „Silent-Failure-Familie" (Befunde 1/2/3 — Daily-Usage-Schmerz-Hotspots) + 1× „Cross-Feature-H4-Helper-Reuse zu F-3" (Banner-Mountpoint / safeJSON / showAlert) + 1× „List-View-State-Visibility und Empty-State-Recovery" (Filter-leer / Suche-leer / Sortier-Active-Marker) + ggf. „Tag-Click-und-Navigation-UX".

Nach Phase 1: STOP — Bericht. Statistik (Findings pro Sev-Stufe, Bug-Tickets-Anzahl, Cross-Feature-H4-Finding-Quote, Schwerpunkt-Cluster). Plus: Disposition für Befund 1 (Master-Annotation 3 prüfen, ob Sub-Thread Heuristik/Severity abweicht). Plus: F-3-Übernahme-Disziplin (welche F-3-Findings 1:1 wiederverwendet, wo Daily-Usage-Schmerz Sev-Abweichung verlangt).

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Findings-Doc nochmal und prüft:

1. **Vollständigkeit**: jeder der 16 Inventur-Befunde ist disponiert (entweder in Findings-Tabelle, in Bug-Tickets-Sektion, in Cross-Feature-H4-„bereits konvergent"-Liste, oder als „aus F-6.2 ausgenommen, gehört zu …"). Mindest-Kandidat für Letzteres: Befund 16 (Helper-Reuse-Meta-Beobachtung).
2. **F-3-Korrespondenz-Übernahme**: jeder Finding mit F-3-Korrespondenz hat dieselbe Heuristik+Severity wie das F-3-Finding, **oder** Sub-Thread hat Abweichung im Bericht begründet (z.B. Daily-Usage-Schmerz-Gewichtung wegen Library-Readwise-Replacement).
3. **Heuristik-Klarheit**: jeder Finding hat genau eine Primary-Heuristik. Sekundär-Heuristiken in Beschreibung erwähnen aber nicht in Filter-Spalte.
4. **Severity-Konsistenz**: Sev 4 nur für Datenverlust/Cost/Blockade-Pfade. Sev 1 nur für rein kosmetisch. Daily-Usage-Schmerz-Gewichtung dokumentiert wo Stufe verschoben.
5. **Cross-Feature-H4-Quote**: liegt im erwarteten Bereich (35-50%) oder Abweichung ist begründet (analog F-4.2-Begründungsdisziplin).
6. **Bereits-konvergente F-3-Patterns**: Liste ist vollständig (P9 / P11 / P12) und Code-Anker stimmen mit F-6.1-Markierungen überein.
7. **Helper-Reuse-Reflexion in Cross-Feature-H4-Sektion**: `saveViewState/loadViewState`-URL-Persistierungs-Begründung als positive Disziplin-Notiz; `confirmInPlace`-Begründung; `showAlert`-Mountpoint-Befund als echter H4-Finding (Verzahnung zu F-3-P15).
8. **`⚠️ code-only`-Marker**: jeder Finding aus den 4 Code-deduced-Inventur-Verdachten + den 9 Live-Walkthrough-Lücken trägt den Marker.
9. **Befund-1-Disposition**: Master-Annotation 3 aufgegriffen — als „Finding + Bug-Ticket" disponiert mit H9 / Sev 3 (oder Sub-Thread-Abweichung mit Begründung).
10. **Befunde-2+3-Disposition**: Master-Annotation 4 aufgegriffen — F-3-P1/P3-Korrespondenz mit H4+H9 / Sev 3 übernommen.
11. **Befund-16-Disposition**: Master-Annotation 5 aufgegriffen — in Cross-Feature-H4-Sektion absorbiert, nicht als Heuristik-Finding.
12. **Disziplin**: kein Pattern-Vorschlag, keine konkrete Microcopy, kein Bug-Fix.

Nach Phase 2: STOP — Bericht. „Findings-Doc konsistent" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-6.2 / Stufe 2: heuristic review of library list view".
- Body: Statistik (Findings pro Sev-Stufe, Bug-Tickets-Anzahl, Cross-Feature-H4-Finding-Quote, Schwerpunkt-Cluster, F-3-Übernahme-Anzahl, Helper-Reuse-Reflexions-Hinweis).
- Branch: direkt auf `main` ist OK.
- `git push origin main`. Wenn Auto-Mode-Classifier blockt **oder** `.git/objects/<hash>`-SMB-Permission blockt (analog F-5.3, F-6.1): im Phase-3-Bericht erwähnen, Master committet/pusht von Hand via SSH zu Mintbox. (Siehe Memory `feedback_push_is_normal.md`.)

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S–M** — eine Output-Datei (`docs/ui_findings_library_list_2026-05.md`), Heuristik-Filter über 16 Inventur-Befunde, F-3-Korrespondenz-Mapping als zentrale Mechanik (M-Anteil wie F-5.2 weil F-3.2-Cross-Read und Per-Pattern-Übernahme-Disziplin Aufwand sind), Cross-Feature-H4-Sektion mit Helper-Reuse-Reflexion strukturiert, kein Code-Touch, keine Tests, kein Smoke.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim F-3-Korrespondenz-Mapping ein F-6.1-Befund **mit „teil-übertragbar"-Markierung** beim Heuristik-Filter klar als „bereits konvergent" oder umgekehrt als „voll H4-Finding" rauskommt: in Cross-Feature-H4-Sektion explizit notieren mit Begründung.
- Wenn beim Lesen der `⚠️ code-only`-Befunde Zweifel an der Code-Deduktion auftauchen: kurz im Bericht aufzählen, **nicht** in der Findings-Doc als „bereits erfüllt" — Master verifiziert.
- Wenn ein list-spezifischer Befund (ohne F-3-Korrespondenz) sich beim Heuristik-Filter als **zu spezifisch für eine einzelne Heuristik** zeigt: Primary einzeln wählen mit Begründung, Sekundär in Beschreibung.
- Wenn beim Code-Reading neue F-3.2-BT-Kandidaten in library_detail-Code auffallen (z.B. weil ein gemeinsamer Helper-Pfad mit library.js geteilt wird): kurz im Bericht erwähnen, **nicht** in die F-6.2-Findings-Doc — F-3 ist abgeschlossen, gehört zu Sammel-Bug-Pass.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F6-REVIEW ☑ done 2026-05-XX → commit `<hash>`. Findings-Doc unter `docs/ui_findings_library_list_2026-05.md`. <Findings-Anzahl> Findings (Sev 4: X, Sev 3: Y, Sev 2: Z, Sev 1: W) + <Bug-Tickets> Bug-Tickets. Cross-Feature-H4-Finding-Quote: <%>%. Befund 1 als Finding+BT für F6-IMPL-Mit-Fix disponiert (H9 Sev 3). Befunde 2+3 als F-3-P1/P3-Korrespondenz mit H4+H9 / Sev 3 übernommen. Verbleibende Sequenz: F6-PATTERNS → F6-IMPL → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F6-REVIEW" raus → Erledigt-Liste; Master fügt F6-PATTERNS als Position 1 beim nächsten Dispatch hinzu — Sub-Thread fügt es **nicht** selbst hinzu (Master-Edit-Zone).
- **Memory**: nur wenn übertragbare Lehren auftauchen. **Master-Annotation 5 (Helper-Reuse-Drift mit begründeter Design-Wahl ist keine H4-Verletzung)** ist ein Kandidat — wenn Sub-Thread beim Apply diese Reflexion bestätigt findet, **feedback-Memory schreiben** mit Pointer auf F-6.2 als Präzedenzfall.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Heuristik-Review-Methodik ist seit F-1.2 / F-2.2 / F-3.2 / F-4.2 / F-5.2 etabliert, Geschwister-Feature-Hebel + Master-Disposition-Vorschläge für Befunde 1/2/3 + Helper-Reuse-Reflexion + Live-Walkthrough-Disposition in Master-Annotationen oben verankert.)_
