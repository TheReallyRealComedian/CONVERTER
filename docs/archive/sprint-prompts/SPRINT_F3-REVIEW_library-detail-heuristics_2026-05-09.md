# Sprint F3-REVIEW — F-3.2 Heuristik-Review `library_detail`

**Datum**: 2026-05-09

**Ziel**: Stufe 2 (Heuristik-Review) der dreistufigen UX-Cascade-Methodik für `library_detail`. Die 18 Befunde aus der F-3.1-Inventur durch Nielsens Heuristiken H1/H4/H6/H9 filtern, Severity-ranken (1=kosmetisch … 4=kritisch), Disposition pro Befund finalisieren (nur Finding / Finding + Bug-Ticket / nur Bug-Ticket). **Kein Pattern-Vorschlag, keine Microcopy** — das kommt in F-3.3.

**Vorbedingung**:
- Pytest 48/48 grün auf `main`. Letzter Code-Touch: F3-PICK (commit `e9cfd1a`, 2026-05-09).
- **Eingabe**: [docs/ui_inventory_library_detail_2026-05.md](docs/ui_inventory_library_detail_2026-05.md) (Sub-Thread liest komplett vor Phase 1).
  - 21 interaktive Elemente.
  - 18 separate Befunde mit Pre-Disposition aus F-3.1: 6 Finding-+-Bug-Ticket-Kandidaten, 11 nur Finding, 1 Pre-Existing-Item.
  - ~12 fehlende States, ~7 Code↔live-Divergenzen (alle code-deduced).
  - Live-Walkthrough-Lücken-Sektion am Doc-Ende — die markierten Befunde sind nicht live verifiziert.
- **Methodik-Vorlagen** (sehr wichtig — Output-Format 1:1 reproduzieren):
  - F-1.2 Heuristik-Review: [docs/ui_findings_document_converter_2026-05.md](docs/ui_findings_document_converter_2026-05.md) — 19 Findings (Sev 4: 2, Sev 3: 7, Sev 2: 7, Sev 1: 3) + 3 reine Bug-Tickets.
  - F-2.2 Heuristik-Review: [docs/ui_findings_audio_converter_2026-05.md](docs/ui_findings_audio_converter_2026-05.md) — 32 Findings + 9 Bug-Tickets, **41% Cross-Feature-H4** (Helper-Reuse aus F-1).
- **Produkt-Kontext** (für Severity-Bewertung): Single-User-App (Oliver), LAN-only, login-protected. Primäre `library_detail`-Aufgaben: Conversion-Eintrag lesen, Highlights/Notizen verwalten, Notion-Export. **Reader-Funktion im Readwise-Ersatz-Kontext** — daily-usage, daher hoher Schmerz-Faktor bei Sev 3+ Findings die täglich treffen.
- **Methodik-Begründung** (Stufe 2 in Kürze): Duan et al. *Heuristic Evaluation with LLMs* (CHI 2024), gefiltert durch Nielsen H1/H4/H6/H9. Kaskade > Monster-Prompt: jede Stufe enger Fokus, Master kann zwischen Stufen korrigieren.

**Out-of-scope**:
- Patterns + Microcopy (Stufe 3) — eigener Folge-Sprint `F3-PATTERNS`.
- Implementation — eigene Folge-Sprints `F3-IMPL-*`.
- Code-Änderungen jeglicher Art.
- Andere Features (`library`, `markdown_converter`, `mermaid_converter`, `login`, podcast-flow) — eigene Folge-Wellen.
- **Konstitutive Befunde aus F-3.1** (api_create_conversion-Strict-Validation vs. Fallback-Render; Notion-MCP-Down-String-Doppelung): bewusst aus diesem Sprint ausgenommen, gehören zur `library`-Welle bzw. einer Notion-Konsolidierung. Im F3-REVIEW-Output **nicht** mit-bewerten.

---

## Master-Annotation (vorab eingebettet)

**Live-Walkthrough-Status**: zwischen F3-PICK und F3-REVIEW hat der Master **noch keinen** Live-Walkthrough nachgereicht (siehe F-3.1-Inventur-Doc Live-Walkthrough-Lücken-Sektion). Sub-Thread arbeitet Code-deduced. Bei Befunden, die in der Inventur als „Code-deduced, nicht live verifiziert" markiert sind, in der Severity-Spalte einen `⚠️ code-only`-Hinweis ergänzen, damit der Master beim Pattern-Sprint (F-3.3) entscheiden kann ob noch Live-Verifikation erfolgen soll bevor implementiert wird.

---

## Phase 1 — Heuristik-Review

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Inventur-Doc komplett lesen**: alle 18 Befunde, alle ~12 fehlende-States-Notizen, alle ~7 Divergenzen, plus Live-Walkthrough-Lücken-Sektion.
3. **Methodik-Vorlagen lesen**: F-1.2 + F-2.2 Heuristik-Review-Docs (Output-Format 1:1).

**Heuristik-Filter** (Nielsens vier ausgewählte Heuristiken):

- **H1 — Visibility of system status**: User weiß was das System gerade tut/erreicht hat. Beispiele aus library_detail-Befunden: Auto-Save-silent-fail, Delete-silent-fail, Notion-Submit-Banner-Persistenz, Loading-States bei AJAX-Calls.
- **H4 — Consistency and standards**: Gleiche Aktion sollte gleiche Sprache/UI sprechen. Cross-Feature-Konvergenz-Vektor: wo `library_detail.js` Inline-Code hat während `_utils.js`-Helper existieren (`showAlert`, `showToast`, `formatFileSize`, `safeJSON`). Auch: Sidebar-Active-State-Konsistenz mit anderen Routen.
- **H6 — Recognition rather than recall**: User soll Optionen sehen, nicht erinnern. Beispiele: Tag-Chip-Rendering, Datum-Format, Page-Title-Information, dirty-indicator.
- **H9 — Help users recognise, diagnose, recover from errors**: bei Fehler nicht nur „Error", sondern was schief lief + wie weiter. Beispiele: Toast-Level-Mismatch (success für Fehler), Error-via-Toast vs. via-Banner, Re-Toggle-Inputs-Wipe ohne Warnung.

**Severity-Skala 1–4**:

- **Sev 4 — Kritisch**: blockiert Kernaufgabe oder Datenverlust-Risiko. Beispiel-Pattern aus F-1/F-2: silent-fail bei Save/Submit auf primärem Pfad, falscher Status nach destruktiver Aktion.
- **Sev 3 — Hoch**: spürbarer Reibungspunkt der täglich trifft, aber Nutzer kommt durch. Beispiel: Cross-Feature-Inkonsistenz die sich kognitiv aufstaut.
- **Sev 2 — Mittel**: stört einmalig, aber nicht handlungsrelevant. Beispiel: Datum-Format ist UTC statt lokal.
- **Sev 1 — Kosmetisch**: würde gefixt wenn man sowieso vorbeikommt. Beispiel: Page-Title fehlt, Helper könnte besser benannt sein.

**Daily-Usage-Schmerz-Hinweis**: weil `library_detail` die Reader-Funktion im Readwise-Ersatz-Kontext ist, soll Severity-Bewertung den **täglichen Treffer-Häufigkeits**-Faktor mit-gewichten. Eine Sev-3-Inkonsistenz, die täglich 10× passiert (z.B. Auto-Save-silent-fail beim Notiz-Tippen), wiegt schwerer als eine Sev-3-Inkonsistenz die alle 2 Wochen einmal sichtbar wird (z.B. Notion-Export). Im Findings-Doc dokumentieren wenn das Schmerz-Gewichtung beeinflusst hat.

**Disposition-Logik pro Inventur-Befund**:

- **Nur Finding** (~kommt in Patterns-Sprint): wenn die UX-Reibung systematisch ist, aber der Code-Pfad korrekt funktioniert.
- **Finding + Bug-Ticket** (~kommt in Patterns-Sprint **plus** separates Bug-Ticket): wenn die UX-Reibung von einem echten Bug verursacht ist, der unabhängig vom Pattern-Cluster gefixt werden könnte (z.B. CSS-Spezifität-Bug, JS-Handler-Scope-Bug).
- **Nur Bug-Ticket** (~kommt **nicht** in Patterns-Sprint): wenn es ein purer Bug ist ohne UX-Heuristik-Komponente. Selten — meist haben Bugs auch H-Aspekte.

**Cross-Feature-H4-Erwartung**: F-2.2 hatte ~41% Cross-Feature-H4 (durch Helper-Reuse aus F-1). F-3.2 Erwartung ähnlich oder höher, weil `_utils.js` jetzt 4 etablierte Helper hat (`showAlert`, `showToast`, `formatFileSize`, `safeJSON`) — wo `library_detail.js` Inline-Code dupliziert, sind das H4-Findings. Beobachtung der Helper-Reuse-Spuren aus F-3.1 ist die direkte Eingabe.

**Output-Doc**: `docs/ui_findings_library_detail_2026-05.md`. Struktur (1:1 wie F-1.2 / F-2.2):

1. Header mit Inventur-Quelle, Sprint-Datum, Methodik-Hinweis (Duan et al. + Nielsen H1/H4/H6/H9).
2. **Findings-Tabelle**: # · Heuristik (H1/H4/H6/H9) · Severity (1–4, mit `⚠️ code-only` wo nicht live verifiziert) · Beschreibung in 1–2 Sätzen · Inventur-Befund-Anker (Nr. aus F-3.1) · Disposition.
3. **Reine Bug-Tickets** (separat, nicht in Findings-Tabelle): Liste mit Code-Anker (`file:line`), Beschreibung, Reproduktion, Vorgeschlagener Fix-Pfad.
4. **Disposition-Verteilung**: Statistik (X nur Findings, Y Findings + Bug-Tickets, Z nur Bug-Tickets). Sub-Thread soll begründen wo der F-3.1-Disposition-Vorschlag abgewichen wurde.
5. **Cross-Feature-H4-Sektion**: Liste der Findings die durch Helper-Reuse aus `_utils.js` lösbar sind. Cross-Feature-Konvergenz-Quote (% der Findings).
6. **Schwerpunkt-Cluster**: 1-3 thematische Cluster wo die schweren Findings sich konzentrieren (analog F-1 „Empty-Submit-Silent / Result-Persistenz / Save-Button-Stale-Visual" oder F-2 „Drag-Drop-Lüge / 11+ alert() / Config-Error-Global / Englische Strings").

Nach Phase 1: STOP — Bericht. Statistik (Findings-Anzahl pro Sev-Stufe, Bug-Tickets-Anzahl, Cross-Feature-H4-Quote, Schwerpunkt-Cluster-Anzahl). Plus: ob Master-Walkthrough-Nachreichung für irgendwelche `⚠️ code-only`-Findings empfohlen wird bevor F-3.3 startet.

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Findings-Doc nochmal mit Distanz und prüft:

1. **Vollständigkeit**: jeder der 18 Inventur-Befunde ist in der Findings-Doc disponiert (entweder in Findings-Tabelle, in Bug-Tickets-Sektion, oder explizit als „aus F-3.2 ausgenommen, gehört zu …" begründet).
2. **Heuristik-Klarheit**: jeder Finding hat genau eine Primary-Heuristik (H1/H4/H6/H9). Sekundär-Heuristiken können in der Beschreibung erwähnt werden, aber Filter-Spalte ist eindeutig.
3. **Severity-Konsistenz**: Sev 4 nur für Datenverlust-/Blockade-Pfade. Sev 1 nur für rein kosmetische Items. Daily-Usage-Schmerz-Gewichtung dokumentiert wo sie eine Stufe verschoben hat.
4. **Cross-Feature-H4-Anker**: jeder als „Cross-Feature-H4" markierte Finding hat Code-Anker zu der Stelle in `library_detail.js` wo Inline-Code statt Helper-Reuse passiert, plus Verweis auf den Helper aus `_utils.js`.
5. **Disziplin**: kein Pattern-Vorschlag im Doc (passiert nur in F-3.3). Keine konkrete Microcopy. Kein Bug-Fix.

Nach Phase 2: STOP — Bericht. „Findings-Doc konsistent, alle 18 Befunde disponiert" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-3.2 / Stufe 2: heuristic review of library_detail".
- Body: kurze Statistik (Findings pro Sev-Stufe, Bug-Tickets-Anzahl, Cross-Feature-H4-Quote, Schwerpunkt-Cluster).
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — eine Output-Datei (`docs/ui_findings_library_detail_2026-05.md`), Heuristik-Filter über die 18 Inventur-Befunde, Severity-Ranking, kein Code-Touch, keine Tests, kein Smoke. Wenn die Cross-Feature-H4-Quote überraschend hoch ist (z.B. >60%) und mehr Findings entstehen als 18: kein Re-Skopung-Trigger, normaler Sprint-Verlauf — F-3.3 fold-et das in den Pattern-Cluster-Vorschlag.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Heuristik-Filter offensichtliche **fehlende Heuristik-Aspekte** auffallen (z.B. ein Befund treibt sowohl H1 als auch H9 stark): in Beschreibung erwähnen, Primary-Heuristik trotzdem einzeln wählen — Doppel-Heuristik-Findings würden die Tabelle unübersichtlich machen.
- Wenn ein Inventur-Befund beim Heuristik-Filter „verschwindet" (keine der vier Heuristiken trifft): explizit als „aus F-3.2 ausgenommen, weil …" notieren — nicht stillschweigend droppen.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F3-REVIEW ☑ done 2026-05-XX → commit `<hash>`. Findings-Doc unter `docs/ui_findings_library_detail_2026-05.md`. <Findings-Anzahl> Findings (Sev 4: X, Sev 3: Y, Sev 2: Z, Sev 1: W) + <Bug-Tickets> Bug-Tickets. Cross-Feature-H4-Quote: <%>%. Verbleibende Sequenz: F3-PATTERNS → F3-IMPL-* → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F3-REVIEW" raus → Erledigt-Liste; Sektion „2. F3-PATTERNS" rückt auf Position 1, alle Folge-Sprint-Nummern -1.
- **Memory**: nichts erwartet — Heuristik-Review-Methodik ist seit F-1.2/F-2.2 etabliert. Falls überraschend doch (z.B. „Reader-Features brauchen eine zusätzliche Heuristik H10 — Help and Documentation"): `feedback_*.md` schreiben.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Heuristik-Review-Methodik ist seit F-1.2 / F-2.2 klar etabliert, Vorlagen und Output-Format vorhanden. Master-Walkthrough-Nachreichung wurde bewusst nicht zwischen F3-PICK und F3-REVIEW gemacht — falls nach F-3.2 nötig, wird sie zwischen F3-REVIEW und F3-PATTERNS eingeschoben.)_
