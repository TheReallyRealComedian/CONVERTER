# MASTER BACKLOG HANDOFF — CONVERTER

**Datum**: 2026-05-26
**Empfänger**: Folge-Master-Thread, der den CONVERTER-Backlog übernimmt.
**Lese-Reihenfolge**: dieses Doc → `CLAUDE.md` (Working-Practice + Architecture) → `STATUS.md` (aktueller Stand) → `BACKLOG.md` (offene Items) → `docs/reader_architecture.md` (Architektur-Memo, aktive Referenz für R2/R3).
**Memory**: `/Users/olivergluth/.claude/projects/-Users-olivergluth-CODE-CONVERTER/memory/MEMORY.md` ist Index; 7 Files (4 feedback + 3 reference).

---

## 1. Aktueller Stand

**Deploy/Build-Frontier**:
- Server: `localhost:5656` auf Mintbox + `converter.smallpieces.de` — Live-Stack-Stand laut STATUS.md, kein Push-Drift seit `c9a4516`.
- Mac-Dev: Docker-Override-Stack über `docker-compose.override.yml` (gitignored, MAC1-Sprint). Container-DB hat Smoke-Artefakte aus R1-B-A/B-B/B-C/R2-A/READER-FIX-A — bewusst hinterlassen als Foundation für R2-B/R2-C-Smokes.
- HEAD: `c9a4516` (READER-FIX-A STATUS+BACKLOG-Update, lokal+remote synchron).
- Letzter Code-Commit: `7f331c1` (READER-FIX-A: multi-node highlight wrap + sidebar-card expand).
- Image-Stand persistent (mehrfach rebuilt in dieser Session via `docker compose up -d --build`).

**Pytest**: 131/131 grün im Container. Stand vor Session: 71/71. +60 Tests in 9 Sprints.

**Strategischer Pivot**: CONVERTER wandert vom Multimedia-Konverter zum **Readwise-Reader-Replacement**. Pivot dokumentiert in `docs/reader_architecture.md`. Bestehende Features (Markdown→PDF, Document→Markdown, Audio-Transcript, Podcast, Notion-Send) bleiben — die Library wird zum Reader-Layer obendrauf. Status:

| Cluster | Status |
|---|---|
| R1 Reader-Core (Reading-View + Highlights + Notes + Tags) | ☑ komplett |
| R2-A Tag-Junction + CSV-Migration | ☑ done |
| READER-MODE (Distraction-Free + Progress + Mouseup) | ☑ done |
| READER-FIX-A (Multi-Node-Wrap + Card-Expand) | ☑ done |
| R2-B Filtered Views + Reading-Progress-Persist | offen |
| R2-C Lifecycle (Inbox/Later/Archive) | offen |
| R3 Web-Article-Save · R4 Ghostreader/RSS/Daily-Review/EPUB | nicht angefasst |

---

## 2. In-flight

**Kein Sprint mid-flight.** READER-FIX-A ist abgeschlossen, alle Code-Commits gepusht, kein Sub-Thread aktiv. Saubere Übergabe-Stelle.

**Exakter nächster Schritt** (Master-Entscheidung offen):
- Entweder direkt R2-B (Filtered Views + Reading-Progress-Persist, M) oder R2-C (Lifecycle, M). Beide haben offene Architektur-Knoten, die einen Mini-Workshop vorab brauchen (analog READER-PLAN).
- Alternative: P3-Master-Smoke-Sweep (7 XS-Items, siehe BACKLOG „P3 Aktive Reminder") plus MAC1-A pip-timeout-Patch (XS) als 60-Min-Hygiene-Block vor dem nächsten R2-Sprint.

**User-Wahl bei letztem Master-Turn**: nach READER-FIX-A-Closing wurde Pause-Empfehlung gegeben, User-Entscheidung über nächste Session steht aus.

---

## 3. Offene Top-Items

**P1-Inbox** (siehe `BACKLOG.md`):
- **R2-B** Filtered Views + Reading-Progress-Persistierung — Tag-Filter-Chip-Row mit URL-Persistierung, Reading-Progress-Schema-Wahl (bool/percent/last-read). Baut auf `Conversion.tag_refs.any(...)`-Pattern aus R2-A.
- **R2-C** Lifecycle-Status Inbox/Later/Archive — Schema-Wahl offen (Enum-Spalte via `_run_pending_migrations` ALTER TABLE vs. eigene Tabelle).
- **MAC1-FOLLOWUP-A** pip-timeout-Patch XS (Dockerfile-Zeile).
- **MAC1-FOLLOWUP-B** Image-Slim NVIDIA-CUDA-Wheels M (Torch-Transitives evaluieren).

**P3-Aktive Reminder** (XS Master-Browser-Aufgaben außer Tesseract):
- R1-A-Light-Mode-Smoke
- R2-A-Dark-Mode-Smoke
- HYG2-Live-Mic-Smoke (3 Sequenzen)
- P8-PDF-Error-Recovery-Smoke
- F6-IMPL P2/P3/P11-Smokes
- READER-MODE-Mobile-Toggles (Polish-Sprint-Kandidat)
- READER-MODE-Keyboard-Shortcuts (Polish-Sprint-Kandidat)
- Hover-Sync für Multi-Node-Highlights (NEU aus READER-FIX-A, Polish wenn UX-Feedback)
- Tesseract-NC-33 (Mintbox-System-Kontext, kein CONVERTER-Touch)

---

## 4. Entscheidungen + offene Fragen

**Persistierte Architektur-Entscheidungen** (siehe `docs/reader_architecture.md` Decision-Log):
- Highlight-Schema = eigene Tabelle mit FK auf Conversion
- Highlight-Anker = Text-Quote + Prefix + Suffix (W3C Web Annotation Style)
- Notes-Storage = Single-`note`-Feld am Highlight
- Tag-Schema = `Tag` + zwei Junction-Tabellen (`conversion_tags` + `highlight_tags`)
- Conversion-Class-Name bleibt unverändert (YAGNI)
- R1-B in drei Sub-Sprints A/B/C gesplittet
- R2-A in zwei Sprints gesplittet (R2-A Tags + R2-C Lifecycle)
- `Tag.get_or_create()`-Classmethod als DRY-Anker für 3 Call-Sites
- CSV-Migration-Idempotenz via leerer-Spalte-als-Marker

**Offene Fragen für die nächsten R2-Workshops**:
- **R2-B Reading-Progress-Schema**: bool (gelesen ja/nein) vs. percent (0-100) vs. last_read_position (Scroll-Offset in Pixel oder char-Index). Pro/Contra je Variante im Workshop klären.
- **R2-B Tag-Filter-Persistierung**: URL-Query-Param (`?tag=ki`) reicht, oder zusätzlich localStorage für Cross-Session-Persistenz? Reader hat URL als Source.
- **R2-C Lifecycle-Schema**: neue Spalte `Conversion.lifecycle_status` (Enum mit Default `'inbox'`) via Inline-ALTER-TABLE-Helper, ODER eigene Tabelle wenn Status-Historie/Timestamps gebraucht werden. Default-Empfehlung: Spalte, weil Single-User-App keine Audit-Trail-Anforderung hat.
- **R2-C Default-Status für neue Conversions**: `inbox` ist Reader-Standard. Alle bestehenden Conversions bei der Migration auf `inbox` setzen? Oder auf `archive` (weil das bestehende Konversions-History faktisch Archiv ist)? UX-Frage.

**Master-Entscheidung steht für**: ob direkt R2-B oder R2-C startet, oder erst Master-Smoke-Sweep.

---

## 5. Gotchas aus diesem Thread

**Nicht in Memory/BACKLOG dokumentiert, aber für den Folge-Master relevant**:

1. **Zeitdimensions-Regel-Verletzung**: in dieser Session habe ich „nach Mitternacht" als Pause-Argument gebracht, obwohl bei Oliver 18:20 war. Commit-Timestamps korrelieren nicht mit Realzeit des Users. Memory `feedback_no_time_dimension.md` ist daraufhin angelegt, aber die Regel ist in CONVERTER-`CLAUDE.md` nicht als Hard-Rule-Block dokumentiert (META_FEEDBACK-Propagation ausstehend, siehe Punkt unten).

2. **Sub-Thread-Push-Auth-Reibung**: in zwei Sprints (R2-A, READER-FIX-A war in derselben Session ohne Reibung) konnte der Sub-Thread nicht pushen — macOS-Keychain-Prompt blockierte HTTPS-Auth in der Sandbox-Session. Der Master-Thread (selbe Sandbox, anderer Process-Kontext) konnte trotzdem pushen. **Pattern für Folge-Master**: bei Sub-Thread-Push-Block → Master-Session probiert es selbst, bevor manuelle Intervention gefordert wird.

3. **Pre-Commit-Patch-Pattern als Master-Disposition**: mehrfach in dieser Session genutzt (R1-B-A Anchor-Koordinaten-Mismatch, R1-B-A Pygments-Class-Kollision, R1-B-C Cross-Format-Popover-Bridge, R2-A Library-Search Junction-Branch, READER-FIX-A Tag-Render-Bug + Tag-Chip-stopPropagation). Wenn ein Phase-2-Smoke einen Bug findet, der **UX-Konsistenz** mit dem aktuellen Sprint-Scope erzwingt (nicht ein neues Feature), darf Master den Pre-Commit-Patch via Sign-off erlauben statt Folge-Sprint anordnen. Spart Cluster-Closing-Inkonsistenzen. Working-Practice-Erweiterung, kein Memory-Eintrag — die Disposition liegt beim Master-Urteil.

4. **CSS-Klassen-Familie-Koexistenz bei Schritt-für-Schritt-Migration**: in R1-B-C wurden `.highlight-tag-chip*`-Klassen neu eingeführt, während die alte `.tag-chip*`-Familie (CSV-Conversion-Tags) parallel weiterlief. Konsolidations-Zeitpunkt war erst bei R2-A, als die CSV abgelöst wurde. Sub-Thread hat das antizipiert und Klassen-Namespace-Trennung diszipliniert beibehalten. **Pattern für Folge-Master**: bei phasenweiser Frontend-Migration alte Klassen unangetastet lassen bis die ablösende Migration komplett ist; in R2-A passierte das beim CSV→Junction-Wechsel. Verwandtes Memory: `feedback_css_class_collision_in_markdown_views.md` deckt den Pygments/Tailwind-Fall ab, der Migrations-Phasen-Fall ist eigene Disziplin.

5. **9-Sprint-Session-Validierung der Working-Practice**: in dieser Session liefen 8 Code-Sprints + 1 Master-Workshop sequenziell durch. Master = Dispatch / Sub-Thread = Execute funktioniert auch unter Hochfrequenz, weil das Pre-Commit-Patch-Pattern (Punkt 3) und die explizite Sub-Phase-Trennung (R1-B in A/B/C, R2-A in zwei Sprints, READER-FIX-A in 1.1/1.2) Sprint-Größen unter L hielten. Kein Sub-Thread musste mid-Sprint eskaliert werden.

6. **Smoke-Foundation-Recycling**: Container-DB hält Highlights/Tags aus R1-B-A bis R2-A bewusst — doc 2 (4 Highlights), doc 4 (Cross-Format-Highlight), doc 7 (Multi-Anker-Demo), doc 9 (Tag „produkt"). Diese Smoke-Artefakte sind für R2-B/R2-C als bereits-bestehende Test-Daten direkt wiederverwendbar — der Folge-Master muss nicht neu seeden für die ersten Smoke-Schritte.

7. **Cross-Format-Highlight-Edge-Case faktisch verschwunden**: nach READER-FIX-A bleibt der `crossFormatHighlightIds`-Set in der Praxis leer, weil Range-Walking alle Multi-Node-Selections wrapt. Das Set bleibt im Code als Defense-Mechanik für absolute Edge-Cases (z.B. degenerierte Selections mit nur whitespace-only Text-Nodes). Folge-Master soll das **nicht aufräumen** — defensive Architektur ohne tote Code-Pfade.

8. **User-Feedback-getriebene Folge-Sprints**: READER-MODE und READER-FIX-A waren nicht vor-geplant in der ursprünglichen R-Cluster-Roadmap. Sie entstanden aus konkreten User-Live-Test-Beobachtungen mit Readwise-Reader-Screenshot-Referenz. **Master-Disziplin für Folge-Master**: nach jedem Cluster-Closing den User zum Live-Test einladen — die wichtigsten UX-Befunde kommen aus dem ersten echten Benutzungs-Versuch.

---

## Quer-Verweise

- `CLAUDE.md` Sektion „Working Practice" — Master=Dispatch-Pattern, Sprint-Größen, Pre-Flight-Pflichten.
- `docs/reader_architecture.md` — Architektur-Memo, aktive Referenz für R2-B/R2-C-Workshops.
- `docs/archive/sprint-prompts/SPRINT_*.md` — alle Sprint-Prompts dieser Session liegen archiviert (MAC1, R1A, R1BA, R1BB, R1BC, R2A, READER-MODE, READER-FIX-A).
- `docs/archive/OVERSEER_HANDOFF_2026-05-03.md` — Vorgänger-Hand-off vom Maschinen-Wechsel, archiviert. Historischer Kontext für die Overseer-zu-Master-Practice-Umstellung am 2026-05-09.
- Memory-Index: `/Users/olivergluth/.claude/projects/-Users-olivergluth-CODE-CONVERTER/memory/MEMORY.md`.
