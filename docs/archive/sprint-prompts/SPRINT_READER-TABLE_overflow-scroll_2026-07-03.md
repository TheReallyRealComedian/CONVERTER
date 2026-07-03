# Sprint READER-TABLE — Breite Tabellen scrollen statt overflowen (XS/S, 2 Phasen)

> **Executor-Doc.** Nach jeder Phase **Stop + Bericht** (Phase 1 = Code+Smoke, Phase 2 = Wrap). Pre-Flight: `pytest tests/` grün (Baseline **628**). Jede Phase selbst committen + pushen. Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. Reiner CSS-Touch (+ ggf. Mini-JS-Fallback) — **kein** Schema/Dep/Token, kein Template-Zwang.
>
> **Kontext (Oli-Report mit Screenshot, 2026-07-03)**: Im Library-Reader sprengen **breite Markdown-Tabellen** die mittlere Content-Spalte und laufen rechts unter die Sidebar-Karten (Beispiel: DDL-Tabelle mit Code-Spalten). Erwartung: die Tabelle bleibt in der Content-Spalte und **scrollt horizontal in einem eigenen Käfig**.

## Verifizierte Fakten (Master-gegroundet)

- **Ursache**: `.reader-view table` ([static/css/style.css:1352](static/css/style.css)) setzt `border-collapse/margin/width:100%/font-size` — **kein** `overflow`-Handling, kein `max-width`-Constraint der greift (Tabellen-Mindest-Inhaltsbreite schlägt `width:100%`). Der Container (`#content-body`/`.c-surface`) clippt nicht.
- **Zweite Fläche**: `.preview-content-area` (Markdown-Converter-Preview; style.css:1091 hat nur Dark-Theme-Kosmetik für `table`) — **mutmaßlich derselbe Bug**, prüfen + mitfixen (gleiches Muster). `grep -rln "reader-view" templates/` → nur `library_detail.html`; der Converter-Reader/Preview nutzt eigene Klassen.
- **NICHT anfassen**: `app_pkg/markdown.py::_wrap_wide_tables`/`_TableColumnCounter` + `.landscape-table` — das ist der **PDF-Pfad** (Playwright, `@page landscape`), eigenes Feature, funktioniert. EPUB ebenso eigener Pfad. Der geteilte Markdown-Renderer (`render_markdown_to_html`) bekommt **keinen** Wrapper-div (er speist auch PDF/EPUB — ein Web-Scroll-Wrapper dort hätte Print-/E-Reader-Nebenwirkungen).
- **Highlights sind text-verankert** (Text-Quote exact/prefix/suffix, Memory `reference_dom_range_walking`): reine CSS-Layout-Änderungen an Tabellen gefährden die Anker nicht.
- **Test-Suite-Limit**: pytest fängt CSS nicht — der Gate ist der **Live-Smoke** (CLAUDE.md).

## Design (gesetzt, mit dokumentiertem Fallback)

**CSS-first** auf den Web-Flächen — kein Renderer-Touch:
```css
.reader-view table {
    display: block;          /* macht die Tabelle zum scrollbaren Block */
    overflow-x: auto;        /* breite Tabellen scrollen im eigenen Käfig */
    width: max-content;      /* schmale Tabellen bleiben inhaltsbreit */
    max-width: 100%;         /* nie breiter als die Content-Spalte */
    /* border-collapse/margin/font-size bleiben */
}
```
(`width: 100%` ersetzen durch `max-content` + `max-width: 100%`.) Dasselbe Muster für die Converter-Preview-Fläche, **falls** dort reproduzierbar.

**Bewusstes Optik-Delta**: schmale Tabellen sind danach **inhaltsbreit** statt auf 100 % gestreckt (Borders enden am Inhalt). Das ist üblich + meist hübscher — aber der **Smoke entscheidet** (Memory `feedback_smoke_beats_pattern_text`): sieht es im echten Bestand schlecht aus, ist der dokumentierte **Fallback** ein Mini-JS-Wrap im Reader-Init (alle `table` in `.reader-view` client-seitig in `<div class="table-scroll">` mit `overflow-x:auto` wrappen; Tabelle behält `width:100%`) — text-neutral, Highlight-Anker-safe.

## Phase 1 — Fix + Live-Smoke

1. CSS-Fix `.reader-view table` wie oben (style.css hat TOC/Section-Kommentare — an der bestehenden Stelle editieren, Kommentar knapp warum).
2. **Converter-Preview prüfen**: breite Tabelle in der Markdown-Converter-Preview reproduzieren → falls gleicher Bug, gleiches Muster auf deren Table-Selektor; falls dort schon ok (anderes Layout), Befund notieren, nichts anfassen.
3. **Live-Smoke (der Gate — dark + light)**:
   - Library-Detail mit **Olis echtem DDL-Dokument** (der Screenshot-Fall, Conversion mit `itonics.budget_fact`-Tabelle): Tabelle bleibt in der Content-Spalte, scrollt horizontal, nichts läuft unter die Sidebar. Auch im **Reader-Mode** (Vollbild-Reader) prüfen.
   - Ein Dokument mit **schmaler Tabelle**: Optik-Delta bewerten (inhaltsbreit ok? Borders sauber?). Sieht es schlecht aus → JS-Wrap-Fallback statt CSS-Variante (dann erneut smoken).
   - **Highlight-Regression**: ein bestehendes Highlight in einem Tabellen-Dokument rendert weiter; neues Highlight über Tabellen-Text setzen funktioniert.
   - Converter-Preview (falls angefasst): breite + schmale Tabelle.
   - 0 Console-Errors.
4. `pytest` grün (628 — reiner CSS-Touch, keine neuen Tests erwartet; falls JS-Fallback nötig wurde: Node-Syntax-Check).

**Stop + Bericht** (inkl. Smoke-Screenshots-Beschreibung + welcher Weg es wurde).

## Phase 2 — Wrap

1. BACKLOG: READER-TABLE ☑ (XS, unter Done); STATUS: Klausel (pytest unverändert, reiner CSS-Touch). **Bullet-Guard** vor Commit.
2. CLAUDE.md **nur falls** JS-Fallback gebaut wurde (dann im Library-/Reader-Bullet erwähnen); beim reinen CSS-Fix kein CLAUDE.md-Touch.
3. Memory nur bei nicht-offensichtlicher Lehre (unwahrscheinlich bei XS).
4. Deploy-Notiz: Mintbox `git pull` + `docker compose up -d --build` (CSS ins Image gebacken → `--build`; danach Browser-Hard-Reload — Memory: kein Cache-Busting). Kein Schema/Dep/Token.

**Stop + Schluss-Bericht.**

## Bewusst NICHT

- **Kein** Touch am geteilten `render_markdown_to_html` (PDF/EPUB-Nebenwirkungen), an `_wrap_wide_tables`/`.landscape-table` (PDF-Feature) oder am EPUB-Pfad.
- **Kein** `table-layout: fixed`/`word-break` (zerstört Code-Spalten-Lesbarkeit).
- **Kein** Anfassen der Sidebar/des Grid-Layouts — der Fix gehört an die Tabelle, nicht an den Container.

## Akzeptanz

- [ ] Olis DDL-Dokument: Tabelle bleibt in der Content-Spalte, scrollt horizontal (normal + Reader-Mode, dark + light).
- [ ] Schmale Tabellen nicht verschlechtert (Smoke-Urteil; sonst JS-Wrap-Fallback).
- [ ] Highlights in Tabellen-Dokumenten unversehrt (bestehend + neu setzen).
- [ ] Converter-Preview geprüft (gefixt oder Befund „nicht betroffen").
- [ ] PDF-/EPUB-Pfad unberührt; `pytest` 628 grün; Wrap + Bullet-Guard.
