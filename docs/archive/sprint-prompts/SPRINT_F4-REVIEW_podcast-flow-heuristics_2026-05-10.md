# Sprint F4-REVIEW — F-4.2 Heuristik-Review `podcast-flow`

**Datum**: 2026-05-10

**Ziel**: Stufe 2 (Heuristik-Review) der dreistufigen UX-Cascade-Methodik für `podcast-flow`. Die 19 Befunde aus der F-4.1-Inventur durch Nielsens Heuristiken H1/H4/H6/H9 filtern, Severity-ranken (1=kosmetisch … 4=kritisch), Disposition pro Befund finalisieren. **Async-Komplexität verlangt Schwerpunkt-Anpassung**: H1 (System-Status) und H9 (Error-Recovery) sind hier wahrscheinlich überrepräsentiert wegen der 5/6 unzureichenden Async-State-Klassen. **Kein Pattern-Vorschlag, keine Microcopy** — das kommt in F-4.3.

**Vorbedingung**:
- Pytest 51/51 grün auf `main`. Letzter Code-Touch: F4-PICK (commit `d974439`, 2026-05-09).
- **Eingabe**: [docs/ui_inventory_podcast_flow_2026-05.md](docs/ui_inventory_podcast_flow_2026-05.md) (Sub-Thread liest komplett vor Phase 1).
  - 17 echte interaktive Elemente + 3 Async-State-Pseudo-Elemente.
  - 19 separate Befunde mit Pre-Disposition aus F-4.1: 3 Finding-+-Bug-Ticket-Kandidaten (Befund 4, 9, 18), 15 nur Finding, 1 Pre-Existing-Erwähnung (Befund 3 Legacy `/generate-podcast`).
  - **5/6 Async-State-Klassen unzureichend**: queued+started konflatiert, stage-progress fehlt komplett (Worker pflegt kein `job.meta`), cancelled ist Frontend-Lüge (Worker läuft weiter), nur finished + failed sind sauber.
  - 5 Code↔live-Divergenz-Verdachte (alle code-deduced).
  - Async-Pipeline-Mapping-Sektion am Doc-Anfang als ASCII-Sequenz-Diagramm.
  - 7 Live-Walkthrough-Test-Anleitungen-Sektion am Doc-Ende.
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - F-1.2 Heuristik-Review: [docs/ui_findings_document_converter_2026-05.md](docs/ui_findings_document_converter_2026-05.md) — 19 Findings + 3 Bug-Tickets.
  - F-2.2 Heuristik-Review: [docs/ui_findings_audio_converter_2026-05.md](docs/ui_findings_audio_converter_2026-05.md) — 32 Findings + 9 Bug-Tickets, ~41% Cross-Feature-H4.
  - F-3.2 Heuristik-Review: [docs/ui_findings_library_detail_2026-05.md](docs/ui_findings_library_detail_2026-05.md) — 17 Findings + 8 Bug-Tickets, ~35% Cross-Feature-H4.
- **Produkt-Kontext** (für Severity-Bewertung): Single-User-App (Oliver), LAN-only, login-protected. Primäre `podcast-flow`-Aufgabe: Quelltext eingeben → Mode wählen → Generieren → warten (Long-Running, mehrere Minuten) → Download. **Long-running async ist der zentrale UX-Reibungspunkt**. Daily-usage-Schmerz-Gewichtung: wenn Oliver Podcasts häufig generiert (Reader-Workflow + Audio-Konsumption), schlagen async-Status-Lücken besonders zu — Cancel-Lüge (Worker brennt CPU+Token weiter), fehlender Stage-Progress (User weiß nicht ob 30s oder 8min noch warten), File-Cleanup-Disk-Wachstum.
- **Methodik-Begründung**: Duan et al. *Heuristic Evaluation with LLMs* (CHI 2024), gefiltert durch Nielsen H1/H4/H6/H9.

**Out-of-scope**:
- Patterns + Microcopy (Stufe 3) — eigener Folge-Sprint `F4-PATTERNS`.
- Implementation — eigener Folge-Sprint `F4-IMPL`.
- Code-Änderungen jeglicher Art.
- **Befund 3 (Legacy `/generate-podcast` Dead-Code-Kandidat)**: aus dem Heuristik-Review ausgenommen, gehört zu einer Hygiene-Welle (Removal-Decision ist nicht UX-H-Frage).
- **F-2.1-Doc-Korrektur** (Service-Gate-Verhalten veraltet): aus dem Scope, gehört zu einer Doc-Hygiene-Welle. Master pflegt das ggf. nach.
- Andere Features.

---

## Master-Annotation (vorab eingebettet)

**Cross-Feature-H4-Erwartung niedriger als F-3.2** (~15-25% statt 35%). Begründung: F-2 hat den Helper-Reuse für `audio_converter.js`-podcast-Block schon durchgezogen — Sub-Thread-Inventur (Befund 20) bestätigt: keine `alert()`-Pfade, keine drei konkurrierenden Error-UI-Patterns mehr. Stattdessen ist die Schmerz-Quelle **Async-spezifisch**: Visibility (H1) und Error-Recovery (H9) dominieren. **Das ist eine erwartete Verteilungs-Verschiebung**, kein Methodik-Problem — der Sub-Thread soll nicht künstlich H4-Findings konstruieren.

**Async-Heuristik-Schwerpunkt**:
- **H1 (Visibility of system status)**: 5/6 unzureichende Async-State-Klassen sind überwiegend H1-Findings. Stage-Progress-Fehlen, queued/started-Konflation, Cancel-Visual-vs-Worker-Realität.
- **H9 (Help users recognise/diagnose/recover from errors)**: Worker-Failure-Pfad, Job-Stale-Detection, Network-Drop während Polling, Cancel-Mid-Job (Worker-Cleanup).
- **H4 (Consistency)**: deutlich weniger als sonst — Helper-Reuse ist sauber. Wenige Findings erwartet.
- **H6 (Recognition over Recall)**: Speaker-Format-Hilfe (Pattern „Kate [warm]: …" im Placeholder), Mode-Radio-Default-Erkennung, Status-Display-Klarheit.

**Daily-Usage-Schmerz besonders relevant**: Long-Running-Jobs sind exponiert. Eine Sev-3-H1-Visibility-Lücke die täglich beim Generieren trifft (z.B. Stage-Progress-Fehlen) wiegt schwerer als eine Sev-3-H4-Konsistenz-Lücke die einmal pro Woche sichtbar wird.

**Befund-Vorab-Einschätzung** (Master-Erwartung, Sub-Thread soll unabhängig prüfen):
- **Befund 9 (Cancel-Lüge — Worker läuft nach Cancel weiter)**: erwartet **Sev 4** als Daily-Usage-Daten/Cost-Verlust-Pfad (CPU-Burn, TTS-Token, Audio-File-Output ohne User-Wissen). Strukturanalogon zu F-2.1's „Drag-Drop-Lüge" (Sev 4). Sub-Thread kann abweichen wenn Begründung gut.
- **Befund 18 (File-Cleanup-vs-Re-Download)**: erwartet **Sev 3** wegen Disk-Wachstum + Re-Download-Pfad-Reibung.
- **Befund 4 (Stage-Progress fehlt)**: erwartet **Sev 3 (H1)** mit Daily-Usage-Schmerz-Gewichtung — Long-Running-Job ohne Fortschritts-Anzeige ist klassischer H1-Bruch.

---

## Phase 1 — Heuristik-Review

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Inventur-Doc komplett lesen**: alle 19 Befunde, Async-Pipeline-Mapping, Live-Walkthrough-Test-Anleitungen, Out-of-Scope-Notizen.
3. **Methodik-Vorlagen lesen**: F-1.2 + F-2.2 + F-3.2 Heuristik-Review-Docs.

**Heuristik-Filter** (mit Async-Schwerpunkt-Anpassung):

- **H1 — Visibility of system status**: System sollte zeigen was es gerade tut + erreicht hat. Async-Schwerpunkte: Stage-Progress-Sichtbarkeit, queued vs. started unterscheidbar, Cancel-Confirmation, Job-Stale-Erkennung, Polling-Drop-Indikation.
- **H4 — Consistency and standards**: Cross-Feature-Konvergenz-Vektor (Helper-Reuse aus `_utils.js`). **Erwartung: deutlich niedriger als sonst**, weil F-2 das schon durchgezogen hat. Sub-Thread soll keine künstlichen H4-Findings konstruieren.
- **H6 — Recognition rather than recall**: Speaker-Format-Hilfe, Mode-Radio-Default-Erkennung, Status-Mapping (queued/started/etc. → User-verständliche DE-Microcopy in Stufe 3), Skript-vs-Quelltext-Workflow.
- **H9 — Help users recognise, diagnose, recover from errors**: Worker-Failure-Pfad-Sichtbarkeit, Cancel-Recovery (Worker-Stop), Network-Drop-Recovery beim Polling, Re-Download-Pfad nach Cleanup, Stale-Job-Detection-Recovery.

**Severity-Skala 1–4** (analog F-3.2):

- **Sev 4 — Kritisch**: blockiert Kernaufgabe oder Datenverlust/Cost-Risiko. Beispiel: Cancel-Lüge (Worker brennt Token weiter) — erwartet für Befund 9.
- **Sev 3 — Hoch**: spürbarer Reibungspunkt der täglich trifft. Beispiel: Stage-Progress-Fehlen.
- **Sev 2 — Mittel**: stört einmalig.
- **Sev 1 — Kosmetisch**: würde gefixt wenn man sowieso vorbeikommt.

**`⚠️ code-only`-Marker**: 5 Befunde aus F-4.1 sind Code↔live-Divergenz-Verdachte. Plus 7 Live-Walkthrough-Test-Anleitungen-Bereiche. Findings die diesen Status erben, mit `⚠️ code-only`-Hinweis in Severity-Spalte ergänzen — Master entscheidet vor F-4.3 ob Walkthrough nötig.

**Disposition-Logik pro Befund** (analog F-3.2):

- **Nur Finding** (kommt in Patterns-Sprint).
- **Finding + Bug-Ticket** (kommt in Patterns-Sprint **plus** separates Bug-Ticket).
- **Nur Bug-Ticket** (kommt **nicht** in Patterns-Sprint, pure Bug ohne UX-H-Komponente).
- **Aus F-4.2 ausgenommen** — wenn ein Inventur-Befund beim Heuristik-Filter „verschwindet", explizit notieren mit Begründung („gehört zu Hygiene-Welle X" oder „kein UX-H-Aspekt erkannt").

**Output-Doc**: `docs/ui_findings_podcast_flow_2026-05.md`. Struktur (1:1 wie F-1.2 / F-2.2 / F-3.2):

1. Header mit Inventur-Quelle, Sprint-Datum, Methodik-Hinweis.
2. **Findings-Tabelle**: # · Heuristik (H1/H4/H6/H9) · Severity (1–4, mit `⚠️ code-only` wo nicht live verifiziert) · Beschreibung in 1–2 Sätzen · Inventur-Befund-Anker (Nr. aus F-4.1) · Disposition.
3. **Reine Bug-Tickets** (separat, nicht in Findings-Tabelle): Liste mit Code-Anker, Beschreibung, Reproduktion, Vorgeschlagener Fix-Pfad.
4. **Disposition-Verteilung**: Statistik. Sub-Thread begründet wo F-4.1-Disposition-Vorschlag abgewichen wurde.
5. **Cross-Feature-H4-Sektion**: erwartet niedrig (~15-25%). Liste der Findings die durch Helper-Reuse lösbar sind. Konvergenz-Quote.
6. **Async-Heuristik-Sub-Sektion** (NEU für podcast-flow): Verteilung der Findings über die 5 unzureichenden Async-State-Klassen (queued+started, stage-progress, cancelled, sowie Polling-Edge-Cases). Welche Heuristik dominiert pro State. Hilft F-4.3 die Patterns nach Async-State-Mapping zu strukturieren.
7. **Schwerpunkt-Cluster**: 2-4 thematische Cluster wo schwere Findings sich konzentrieren. Erwartung: 1× „Async-State-Visibility (H1)" + 1× „Cancel/Cleanup-Recovery (H9)" + ggf. „Speaker-Format-Hilfe (H6)" + ggf. kleine „Cross-Feature-H4-Restposten".

Nach Phase 1: STOP — Bericht. Statistik (Findings pro Sev-Stufe, Bug-Tickets-Anzahl, Cross-Feature-H4-Quote, Async-Heuristik-Verteilung, Schwerpunkt-Cluster). Plus: ob Master-Walkthrough-Nachreichung empfohlen wird (analog F-3.2).

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Findings-Doc nochmal und prüft:

1. **Vollständigkeit**: jeder der 19 Inventur-Befunde ist disponiert (entweder in Findings-Tabelle, in Bug-Tickets-Sektion, oder als „aus F-4.2 ausgenommen, gehört zu …").
2. **Heuristik-Klarheit**: jeder Finding hat genau eine Primary-Heuristik. Sekundär-Heuristiken in Beschreibung erwähnen aber nicht in Filter-Spalte.
3. **Severity-Konsistenz**: Sev 4 nur für Datenverlust/Cost/Blockade-Pfade. Sev 1 nur für rein kosmetisch. Daily-Usage-Schmerz-Gewichtung dokumentiert wo Stufe verschoben.
4. **Async-Heuristik-Sub-Sektion**: Findings-Mapping zu Async-States ist konsistent mit Findings-Tabelle. Keine doppelten Einträge.
5. **Cross-Feature-H4-Quote**: liegt im erwarteten Bereich (~15-25%) oder Abweichung ist begründet.
6. **`⚠️ code-only`-Marker**: jeder Finding aus den 5 Code-deduced-Inventur-Verdachten + den 7 Live-Walkthrough-Bereichen trägt den Marker.
7. **Disziplin**: kein Pattern-Vorschlag, keine konkrete Microcopy, kein Bug-Fix.

Nach Phase 2: STOP — Bericht. „Findings-Doc konsistent" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-4.2 / Stufe 2: heuristic review of podcast-flow".
- Body: Statistik (Findings pro Sev-Stufe, Bug-Tickets-Anzahl, Cross-Feature-H4-Quote, Async-Heuristik-Verteilung, Schwerpunkt-Cluster).
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints. Wenn der Auto-Mode-Classifier blockt: im Phase-3-Bericht erwähnen, Master pusht von Hand.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — eine Output-Datei (`docs/ui_findings_podcast_flow_2026-05.md`), Heuristik-Filter über die 19 Inventur-Befunde, Severity-Ranking, Async-Heuristik-Sub-Sektion, kein Code-Touch, keine Tests, kein Smoke.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Heuristik-Filter ein Befund offensichtlich **mehrere Heuristiken stark trifft** (z.B. Cancel-Lüge ist H1+H9): Primary-Heuristik einzeln wählen, Sekundär in Beschreibung.
- Wenn ein Inventur-Befund beim Heuristik-Filter „verschwindet": explizit als „aus F-4.2 ausgenommen, weil …" notieren — nicht stillschweigend droppen.
- Wenn beim Lesen der `⚠️ code-only`-Befunde Zweifel an der Code-Deduktion auftauchen (z.B. „Befund 9 könnte schon durch ein Fix in einem Folge-Commit gelöst sein"): kurz im Bericht aufzählen, **nicht** in der Findings-Doc als „bereits erfüllt" — Master verifiziert.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F4-REVIEW ☑ done 2026-05-XX → commit `<hash>`. Findings-Doc unter `docs/ui_findings_podcast_flow_2026-05.md`. <Findings-Anzahl> Findings (Sev 4: X, Sev 3: Y, Sev 2: Z, Sev 1: W) + <Bug-Tickets> Bug-Tickets. Cross-Feature-H4-Quote: <%>%. Verbleibende Sequenz: F4-PATTERNS → F4-IMPL → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F4-REVIEW" raus → Erledigt-Liste; Sektion „2. F4-PATTERNS" rückt auf Position 1, alle Folge-Sprint-Nummern -1.
- **Memory**: nur wenn übertragbare Lehren für Async-Features auftauchen (z.B. „Async-State-Mapping-Konvention für künftige async UX-Wellen"). Defensiv.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Heuristik-Review-Methodik ist seit F-1.2 / F-2.2 / F-3.2 etabliert, Async-Schwerpunkt-Anpassung in Master-Annotation oben verankert.)_
