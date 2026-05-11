# Sprint WAVE-CLOSE — Strukturelles Closing der Cleanup-/UX-Cascade-Welle

**Datum**: 2026-05-11

**Ziel**: Strukturelles Closing der Cleanup-Welle (Stages 0–7 plus F-1 bis F-6 UX-Cascade plus SEC / HYG / CVE-Sprints). `docs/cleanup_plan.md`-Header auf "fully closed" updaten, `OVERSEER_HANDOFF.md` archivieren, P3-BACKLOG-Hygiene durchziehen, UX-Cascade-Methodik in `CLAUDE.md` unter „Historical" für künftige Wellen verankern. **Kein Code-Touch** (außer Doc-Files).

**Vorbedingung**:
- Pytest 71/71 grün auf `main`. Letzter Code-Touch: F6-IMPL (commits `8049d3e` / `741794a` / `ca41270` / `09f4bf3`, 2026-05-11). F-6 strukturell abgeschlossen für `library` List-View. Vorherige Wellen-Closes: F-1 bis F-5 plus Cleanup-Stages 0-7 plus SEC plus HYG plus CVE-LOW plus CVE-PDF plus CVE-RQ plus CVE-DG.
- **Eingabe**:
  - [docs/cleanup_plan.md](docs/cleanup_plan.md) — Cleanup-Plan-Doc, Header steht vermutlich noch auf WIP / Stages-Status. Sub-Thread liest komplett vor Phase 1.
  - [OVERSEER_HANDOFF.md](OVERSEER_HANDOFF.md) — historischer Hand-off-Doc vom 2026-05-03 Maschinen-Wechsel. Sub-Thread liest Header und prüft ob noch lebende Inhalte enthalten sind.
  - [BACKLOG.md](BACKLOG.md) — P3-Reminder-Sektion komplett lesen (~10-12 Items inklusive ausstehender Master-Smokes für F-5-IMPL P8 und F-6-IMPL P2/P3/P11).
  - [STATUS.md](STATUS.md) — „Zuletzt durch"-Sektion zur Closing-Story-Verifikation.
  - [CLAUDE.md](CLAUDE.md) — „Historical"-Sektion am Doc-Ende für die UX-Cascade-Convention-Ergänzung.
- **Master-Wahl**: WAVE-CLOSE direkt nach F-6, ohne dedizierte mermaid_converter- und login-F-N-Wellen. Begründung: F-1 bis F-6 haben die hochfrequenten Features abgedeckt; mermaid_converter und login bleiben als P3-Reminder im BACKLOG falls später UX-Reibung auftaucht.

**Out-of-scope**:
- Neue F-N-Wellen für mermaid_converter / login — bleiben als P3-Reminder.
- Tatsächliche Master-Smoke-Durchführung (P8 / P2 / P3 / P11) — bleibt Master-Aufgabe, nicht in Sub-Thread-Scope. Sub-Thread kann Reminder-Formulierung schärfen.
- Code-Änderungen jeglicher Art außerhalb der Doc-Files.
- Memory-Erweiterungen außer wenn beim Closing eine übertragbare Methodik-Lehre auffällt (z.B. „UX-Cascade-Methodik als wiederholbarer Sprint-Pfad").

---

## Master-Annotation (vorab eingebettet)

### 1. cleanup_plan.md — Header auf „fully closed 2026-05" updaten

**Mechanik**: Header-Sektion am Doc-Anfang umschreiben:
- Status: WIP → **fully closed 2026-05-11**.
- Sequenz-Zusammenfassung: alle Cleanup-Stages 0-7 plus F-001…F-018 plus SEC plus HYG plus CVE-LOW plus CVE-PDF plus CVE-RQ plus CVE-DG plus F-1 bis F-6 UX-Cascade verlinken.
- Optional: Pre/Post-Test-Anzahl-Statistik (z.B. „Test-Suite 0 → 71/71 grün im Container").
- Footer-Notiz: weitere UX-Wellen für mermaid_converter / login bleiben als P3-Reminder im BACKLOG, kein blockierender Pfad.

**Disziplin**: keine inhaltliche Doc-Re-Strukturierung — nur Header-Update + ggf. Schluss-Sektion mit Closing-Story. Falls beim Lesen offensichtliche Outdated-Items im Body auffallen: kurz im Bericht erwähnen, **nicht** inline editen (außer Sub-Thread sieht klaren Grund für minimale Aktualisierung — Bericht-Pflicht).

### 2. OVERSEER_HANDOFF.md — Archivierung

**Mechanik**: nach `docs/archive/OVERSEER_HANDOFF_2026-05-03.md` verschieben (`git mv`), kein Inhalt-Edit. Archiv-Marker am Header optional (z.B. „**Archiviert 2026-05-11** — durch CLAUDE.md / STATUS.md / BACKLOG.md ersetzt.").

**Disziplin**: keine Löschung — Audit-Trail erhalten. `git mv` statt manuelles Kopieren+Löschen wegen History-Erhalt.

**Verzahnung**: in CLAUDE.md prüfen ob `OVERSEER_HANDOFF.md` referenziert wird (Sub-Thread sucht via grep) und Pfad auf `docs/archive/OVERSEER_HANDOFF_2026-05-03.md` updaten.

### 3. UX-Cascade-Methodik in CLAUDE.md unter „Historical"

**Mechanik**: in der existierenden „Historical"-Sektion am CLAUDE.md-Ende einen neuen Bullet-Point ergänzen — analog der bestehenden F-1/F-2-Methodik-Lehren-Sektion. Inhalt:

- **2026-05 UX-Cascade F-3 bis F-6 abgeschlossen** mit dreistufiger Duan-Kaskade-Methodik (Inventur → Heuristik-Review → Patterns + Microcopy) plus Implementation-Cluster (1-3 Sub-Batches je nach Patterns-Anzahl).
- **Geschwister-/Schwester-Feature-Hebel-Methodik**: F-X-Korrespondenz-Spalte aus Inventur-Stufe wandert als Heuristik-Filter-Eingabe in Review-Stufe und als Pattern-Übernahme-Quelle in Patterns-Stufe. F-3 als Geschwister-Quelle für F-6, F-1 als Schwester-Quelle für F-5.
- **Methodik-Lehren**: Helper-Reuse-Drift mit begründeter Design-Wahl ≠ H4-Verletzung (`feedback_helper_reuse_design_choice.md`, Präzedenzfall F-6.2). Smoke-Sequenz schlägt Pattern-Text bei Spannung (`feedback_smoke_beats_pattern_text.md`, Präzedenzfall F-5-IMPL P7). XS-Lastigkeit bei Schwester-/Geschwister-Übernahme durch 1:1-Pattern-Identität-Erhaltung.
- **Output-Konvention**: pro Feature drei Docs in `docs/` (`ui_inventory_<feature>_<datum>.md`, `ui_findings_<feature>_<datum>.md`, `ui_patterns_<feature>_<datum>.md`) plus Sprint-Prompt-Docs in `docs/archive/sprint-prompts/`.

**Disziplin**: keine Re-Strukturierung der bestehenden „Historical"-Sektion, nur Ergänzung. Master-Edit-Zone-Konformität (CLAUDE.md ist explizit Master-Edit-Zone, aber Sub-Thread darf für WAVE-CLOSE editen weil das Closing-Tätigkeit ist).

### 4. P3-BACKLOG-Hygiene

**Mechanik**: Sub-Thread geht **jedes P3-Reminder-Item** durch und klassifiziert:
- **Bleibt offen**: noch relevant, Reminder behalten (z.B. mermaid_converter / login Future-F-N, Playwright-UI-Tests, Tesseract-NC-33-System-Kontext).
- **Durch nachfolgende Welle erfüllt**: Doc-Korrektur durch F-Welle absorbiert oder Code-Fix durch IMPL-Sprint erfolgt → **entfernen** mit kurzer Erfüllungs-Note im Sprint-Bericht.
- **Schärfen**: Formulierung präziser machen oder Code-Anker aktualisieren wenn veraltet.

**Master-Vorschläge** (Sub-Thread prüft und kann abweichen):
- **F-2.1-Doc-Korrektur Service-Gate** — F-4.1-Befund 2 hat das schon dokumentiert. **Vorschlag: entfernen** mit Note „in F-4.1 absorbiert".
- **getUserMedia-in-socket.onopen-Bug** — bleibt offen, eigene Audio-UX-Welle.
- **2 EN-Strings in library.js** — F-6-IMPL Sub-Batch A P5 DE-Microcopy-Sweep hat das gefixt. **Vorschlag: entfernen** mit Note „in F-6-IMPL absorbiert".
- **F-3.2 BT7+BT8** — bleiben offen, Sammel-Bug-Pass-Kandidaten.
- **Opacity-Übergang Notion-Target-Switch** — bleibt offen.
- **aria-live Save-Success-Hint** — bleibt offen, dedizierte a11y-Welle.
- **Playwright-UI-Tests** — bleibt offen, eigener Sprint.
- **Tesseract-NC-33** — bleibt offen, System-Kontext.
- **3 EN-Strings in app_pkg/markdown.py** — bleibt offen, DE-Microcopy-Sammel-Welle-Kandidat.
- **P8-Master-Live-Smoke** — bleibt offen, Master-Aufgabe.
- **F6-IMPL Master-Live-Smokes P2+P3+P11** — bleibt offen, Master-Aufgabe.

**Neu ergänzen als P3-Reminder**:
- **mermaid_converter F-N-Welle (deferred)** — kein dedizierter Sprint mehr in der Cleanup-Welle, Folge-Sprint wenn UX-Reibung auftaucht oder Feature-Erweiterung ansteht.
- **login F-N-Welle (deferred)** — analog mermaid_converter, sehr kleine Page, niedrige Daily-Usage-Schmerz-Gewichtung.

### 5. STATUS.md — Aktueller-Sprint-Block + Sequenz-Plan-Disposition

**Mechanik**: nach WAVE-CLOSE-Done:
- „**Aktueller Sprint**"-Block: ein letzter „WAVE-CLOSE ☑ done 2026-05-XX"-Eintrag mit Closing-Statistik (Wellen-Zusammenfassung, Final-Test-Anzahl, BACKLOG-Hygiene-Verteilung).
- „**Sequenz-Plan**"-Sektion: **entfernen oder umetikettieren auf „Steady-State"**. BACKLOG wird von sequenzieller Roadmap zu P0/P1-Inbox + P3-Reminder-Liste.
- „**Zuletzt durch**"-Sektion: beibehalten als Audit-Trail, ggf. ältere Einträge in `docs/archive/STATUS_history_<datum>.md` verschieben wenn STATUS.md zu lang wird (Sub-Thread entscheidet pragmatisch — defensiv: nur trimmen wenn > ~50 Zeilen).

### 6. BACKLOG.md — Strukturwechsel von „In Sequenz" zu „Steady-State"

**Mechanik**: nach WAVE-CLOSE-Done:
- Sektion „**In Sequenz — UX-Cascade-Fortsetzung**" entfernen (war Position 1, mit F-N-Pattern). F-N-Items wandern als P3-Reminder.
- Sektion „**In Sequenz — Wave-Close**" entfernen (war Position 2-3). WAVE-CLOSE selbst ist done.
- Sektion „**P3 — nicht in Sequenz / Reminder**" wird zur **primären BACKLOG-Sektion**. Neu strukturieren in zwei Sub-Sektionen:
  - **P3 — Aktive Reminder** (Bugs, Folde-Kandidaten, Master-Smokes, Doc-Hygiene).
  - **P3 — Deferred F-N-Wellen** (mermaid_converter, login, Playwright-UI-Tests).
- Top-of-Doc-Header anpassen: „Source-of-Truth für offene Items. **Cleanup-Welle abgeschlossen 2026-05** — Items sind P0/P1-Inbox plus P3-Reminder, keine sequenzielle Roadmap mehr."

### 7. Memory-Disposition

**Erwartung**: keine neue Memory-Eintrag. Die fünf etablierten feedback-/reference-/project-/user-Memos plus zwei Geschwister-/Schwester-Übernahme-Methodik-Memos decken die Wellen-Lehren ab.

**Sub-Thread kann abweichen** wenn beim Closing eine übertragbare Lehre auffällt die nicht abgedeckt ist (z.B. „WAVE-CLOSE-Mechanik als wiederholbarer Closing-Sprint" — aber das ist eher CLAUDE.md-Historical-Inhalt, nicht Memory).

---

## Phase 1 — Doc-Updates

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Eingaben lesen** (Pflicht-Reihenfolge):
   - `docs/cleanup_plan.md` — Header + Body überfliegen.
   - `OVERSEER_HANDOFF.md` — Header + Inhalt-Sample.
   - `BACKLOG.md` — P3-Reminder-Sektion komplett.
   - `STATUS.md` — „Zuletzt durch"-Sektion.
   - `CLAUDE.md` — „Historical"-Sektion.
3. **`grep -r "OVERSEER_HANDOFF" .`** — finde alle Referenzen die nach Archivierung zu aktualisieren sind.

**Apply-Reihenfolge**:

1. **`docs/cleanup_plan.md`** Header-Update auf „fully closed 2026-05-11" (Master-Annotation 1).
2. **`OVERSEER_HANDOFF.md` archivieren** via `git mv` nach `docs/archive/OVERSEER_HANDOFF_2026-05-03.md` (Master-Annotation 2). Plus Archiv-Marker am Header.
3. **`CLAUDE.md`** „Historical"-Sektion-Ergänzung (Master-Annotation 3). Plus OVERSEER_HANDOFF-Pfad-Update wenn Referenz vorhanden.
4. **`BACKLOG.md`** Strukturwechsel zu „Steady-State" plus P3-Hygiene (Master-Annotationen 4+6). Header umschreiben, sequenzielle Sektionen entfernen, P3-Sub-Sektionen einrichten.
5. **`STATUS.md`** Aktueller-Sprint-Block plus Sequenz-Plan-Disposition (Master-Annotation 5).

Nach Phase 1: STOP — Bericht. Statistik (Files geändert, Zeilen-Diff, P3-Items entfernt vs. behalten vs. neu, BACKLOG-Strukturwechsel-Status, CLAUDE.md-Ergänzung-Anchor).

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigenen Doc-Edits nochmal und prüft:

1. **cleanup_plan.md**: Header steht auf „fully closed 2026-05-11" mit Closing-Story. Body inhaltlich unverändert (außer Sub-Thread hat klaren Grund für minimale Aktualisierung dokumentiert).
2. **OVERSEER_HANDOFF.md** archiviert via `git mv`. CLAUDE.md-Referenz aktualisiert wenn vorhanden. Keine anderen toten Referenzen via `grep`.
3. **CLAUDE.md**: „Historical"-Bullet ergänzt mit UX-Cascade-Methodik-Zusammenfassung. Bestehende Historical-Sektion-Struktur unverändert.
4. **BACKLOG.md**: Top-Header umgeschrieben, sequenzielle Sektionen entfernt, P3-Sub-Sektionen aktiv vs. deferred sauber getrennt. Erledigt-Liste (rolling) bleibt erhalten als Audit-Trail.
5. **STATUS.md**: Aktueller-Sprint-Block leer oder „Steady-State"-Marker, Sequenz-Plan entfernt/umetikettiert, Zuletzt-durch bleibt.
6. **P3-Hygiene-Disziplin**: entfernte Items haben Erfüllungs-Note im Sprint-Bericht. Neue Items haben klare Disposition.
7. **Konsistenz Markdown-Links**: alle Links in den geänderten Docs funktionieren (kein toter Verweis auf `OVERSEER_HANDOFF.md` Top-Level).

Nach Phase 2: STOP — Bericht. „Doc-Stack konsistent" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- **Ein Commit**. Subject z.B. „WAVE-CLOSE: cleanup welle abgeschlossen 2026-05".
- Body: Closing-Story (Wellen-Sequenz F-1 bis F-6 plus Cleanup-Stages plus SEC/HYG/CVE plus Test-Anzahl-Statistik), P3-Hygiene-Verteilung (entfernt vs. behalten vs. neu), BACKLOG-Strukturwechsel-Hinweis, OVERSEER_HANDOFF-Archivierung-Notiz.
- Branch: direkt auf `main` ist OK.
- `git push origin main`. Wenn Auto-Mode-Classifier blockt **oder** `.git/objects/<hash>`-SMB-Permission blockt: Bericht, Master pusht von Hand via SSH zu Mintbox (siehe Memory `feedback_push_is_normal.md`).

---

## Stop-Regel

Nach **jeder Phase** Bericht an Master, nicht weiter bis Sign-off.

**WAVE-CLOSE-spezifischer STOP-Trigger**: wenn beim Apply ein Item-Status unklar ist (z.B. ein P3-Reminder kann durch eine Welle erfüllt sein, aber Sub-Thread ist nicht sicher): kurz im Bericht erwähnen, **nicht** silent entfernen — Master entscheidet.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S–M** — fünf Doc-Files (`docs/cleanup_plan.md`, `OVERSEER_HANDOFF.md` Move, `CLAUDE.md`, `BACKLOG.md`, `STATUS.md`), kein Code-Touch, keine Tests. M-Anteil wegen BACKLOG-Strukturwechsel + P3-Hygiene-Disziplin (jedes Item klassifizieren).

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Lesen von `cleanup_plan.md` substantielle outdated-Inhalte im Body auffallen: kurz im Bericht aufzählen, **nicht** inline editen außer Sub-Thread sieht klaren Grund für minimale Aktualisierung — Bericht-Pflicht.
- Wenn beim P3-Hygiene-Pass ein Item-Status unklar ist: in Bericht mit Status-Frage an Master, nicht silent entfernen.
- Wenn beim grep nach „OVERSEER_HANDOFF" mehrere Stellen außerhalb CLAUDE.md gefunden werden: alle aktualisieren mit Bericht-Liste.
- Wenn beim BACKLOG-Strukturwechsel die Erledigt-Liste (rolling) zu lang ist für sinnvolle Top-of-Doc-Anzeige: trimmen mit Note „älteste 10 in commit-history zugänglich" — Sub-Thread entscheidet pragmatisch.

Alles andere bleibt liegen.

---

## STATUS- und BACKLOG-Updates nach Abschluss

Sub-Thread pflegt am Ende (Phase 1 hat sie schon zentral angefasst; Phase 2-Check verifiziert nur):

- **STATUS.md**: „WAVE-CLOSE ☑ done 2026-05-XX → commit `<hash>`. Cleanup-Welle 2026-05 abgeschlossen. Sequenz F-1 bis F-6 UX-Cascade plus Stages 0-7 plus SEC plus HYG plus CVE-LOW plus CVE-PDF plus CVE-RQ plus CVE-DG. Pytest-Final 71/71 grün. BACKLOG umstrukturiert auf Steady-State (P0/P1-Inbox plus P3-Reminder). OVERSEER_HANDOFF.md archiviert. CLAUDE.md Historical-Sektion erweitert."
- **BACKLOG.md**: Sektion „1. WAVE-CLOSE" raus → Erledigt-Liste. Header umgeschrieben (siehe Master-Annotation 6).
- **Memory**: erwartet keine neue Eintrag — Master-Annotation 7. Falls beim Closing übertragbare Lehre auffällt: defensiv `feedback_*.md` oder `reference_*.md`.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — alle Closing-Aufgaben sind in Master-Annotationen oben spec'd, Master-Vorschläge für P3-Hygiene-Items pro-Item explizit aufgeführt, Sub-Thread kann pro Item abweichen mit Bericht-Pflicht.)_
