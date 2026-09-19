# SPRINT READER-SCROLLBAR — die Scroll-Leiste im Lesemodus tritt zurück

**Größe**: S (eine Phase, Wrap inklusive) · **Datum**: 2026-09-19 · **Vorhaben**: Reader

## Warum

Oli, 2026-09-19, mit Screenshot: *„im reader-mode ist die scroll-leiste zu groß und ablenkend"*. Das Bild zeigt den Lesemodus des Markdown-Konverters in Dark: neben der Textspalte steht eine **breite, weiße** klassische Scroll-Leiste — mitten im Bild, nicht am Fensterrand, rechts von ihr geht die dunkle Fläche weiter.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten, am Code belegt 2026-09-19)

**Welcher Reader**: der Lesemodus des **Markdown-Konverters** (iframe), nicht der Library-Reader. Beleg aus dem Screenshot: Papier `#0d1117` mit h2-Unterlinie = die Dunkel-Fassung der Vorlage `default` ([static/css/pdf_styles/dark/default.css:12](../../../static/css/pdf_styles/dark/default.css), READER-STIL).

**Warum die Leiste weiß und breit ist**: der Vorschau-iframe ist ein **eigenes Dokument** (`srcdoc`). Die App kennt für Dark sehr wohl Regeln — `color-scheme: dark` ([static/css/style.css:2595](../../../static/css/style.css)) und `[data-global-theme="dark"] ::-webkit-scrollbar` mit 8 px, Track `--nm-bg`, Thumb `--nm-shadow-light` (:2658–2660) —, aber **keine davon überquert die iframe-Grenze**. `buildIframeDoc` ([static/js/markdown_converter.js:263–278](../../../static/js/markdown_converter.js)) schreibt in das iframe-Dokument nur `html,body{margin:0;background;color}`: **kein `color-scheme`, keine Scrollbar-Regel**. Im iframe zeichnet der Browser deshalb seine helle Standard-Leiste (~15 px, weißer Track) — in ein dunkles Dokument.

**Warum sie mitten im Bild steht**: im Lesemodus scrollt **der iframe selbst** (`.main-container.reader-mode .preview-iframe`: `height: calc(100vh - 80px)`, `max-width: var(--reader-width)`, style.css:819). Die Leiste sitzt also an der rechten Kante der **Textspalte**, nicht am Fenster — rechts daneben liegt die Scope-Fläche in `--reader-bg`.

**Wann man sie sieht**: bei klassischen (immer sichtbaren) Scroll-Leisten — macOS „Rollbalken immer einblenden" oder angeschlossene Maus; Olis Setup offenkundig. ⚠️ Headless Chromium zeichnet ebenfalls klassische Leisten (LEARN-SKIP-Nebenbefund: 17 px) — der Defekt ist im Smoke **messbar**, nicht nur sichtbar.

**Hell**: dieselbe Geometrie, helle Leiste auf hellem Papier — weniger grell, genauso breit, genauso mitten im Bild.

**Die drei Papiere sind verschieden** (READER-STIL): `default` `#0d1117` · `academic-latex` rgb(19,19,22) · `newspaper-bodoni` rgb(27,25,21) · ohne Zwilling der generische Fallback `#1a1a2e`. Ein hart gesetzter Track-Ton passt also höchstens auf eines davon.

## Gesperrte Entscheidungen

1. **Der Track folgt dem Papier** — `transparent`, kein harter Farbwert. Dieselbe Lehre wie `syncReaderPaper` (LESEMODUS): der Stil bestimmt das Papier, alles um ihn herum liest es ab, statt es zu raten.
2. **Beide Themes, alle Vorlagen, auch der Fallback ohne Dunkel-Zwilling.** Der Auftrag ist „zu groß und ablenkend", nicht „in Dark weiß".
3. **`color-scheme` gehört ins iframe-Dokument** (passend zu `isDarkActive()`), nicht nur die Scrollbar-Regel — sonst bleiben die übrigen browser-gezeichneten Teile (Scrollbar-Ecke, Overscroll) hell.
4. **Standard zuerst** (`scrollbar-width` / `scrollbar-color`), `::-webkit-scrollbar` nur als belegter Rückfall. Welcher Browser bei Oli läuft, ist unbekannt → **Chromium und WebKit** prüfen (das Playwright-Basis-Image bringt beide mit).
5. **Dezent, aber auffindbar**: der Thumb darf nicht verschwinden — eine Leiste, die man nicht mehr findet, ist der nächste Bug-Report.
6. **Kein `!important`** (LESEMODUS-Doktrin). **Kein Umbau des Scroll-Containers** — den iframe auf Inhaltshöhe zu ziehen und das Fenster scrollen zu lassen wäre die größere Lösung (Leiste am Fensterrand), berührt aber Esc, Aa-Popover und `syncReaderPaper`; als benannte Möglichkeit hinterlassen, nicht bauen.

---

# Phase 1 — Fix, Beleg, Wrap

## 1.1 Den Befund herstellen

Im echten Browser, Lesemodus, Dokument lang genug zum Scrollen: Breite der Leiste und Farbe ihres Tracks messen — Dark × drei Vorlagen + Fallback, Hell × eine. Ein Pixel-Streifen aus dem Screenshot an der rechten iframe-Kante ist die Messung (Hausmuster Eckpixel aus [scripts/smoke_markdown_reader.py](../../../scripts/smoke_markdown_reader.py)).

## 1.2 Fix

In `buildIframeDoc` — dort entsteht das Dokument, dort gehört es hin. Die Regel reist mit jedem Neuaufbau (`srcdoc` wird bei jedem Tastendruck neu gesetzt), es gibt keinen zweiten Ort zu pflegen. ⚠️ `generate_pdf` baut sein HTML serverseitig und sieht davon nichts — das PDF bleibt unberührt; trotzdem im Bericht bestätigen.

## 1.3 Ein Blick zum Nachbarn

Der Library-Reader scrollt das **Fenster**; dort greifen die App-Regeln. Einmal in Dark ansehen und berichten, **ob** dort derselbe Defekt existiert. Kein Fix ohne Befund.

## 1.4 Der Beleg

[scripts/smoke_markdown_reader.py](../../../scripts/smoke_markdown_reader.py) erweitern (heute 57 Checks): je Zustand aus 1.1 — berechnetes `color-scheme` und `scrollbar-width` am iframe-`html`, Leistenbreite unter einer benannten Schranke, **Track-Pixel == Papierton** (außerhalb des Thumbs), Thumb vom Papier unterscheidbar. In Chromium **und** WebKit. `pytest tests/` grün (Baseline **1129 + 1 Skip**) — die Suite sieht von alledem nichts, der Smoke ist der Gate.

## 1.5 Wrap

- **CLAUDE.md**, Lesemodus-Bullet: ein Satz — der iframe ist ein eigenes Dokument, App-Regeln (`color-scheme`, Scrollbar) überqueren die Grenze nicht; was im iframe gelten soll, schreibt `buildIframeDoc`. Track transparent wegen der drei Papiere.
- **STATUS.md**, **BACKLOG.md** (⚠️ Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md`, exit 1 = sauber): READER-SCROLLBAR schließen; „Fenster scrollt statt iframe" als benannte Möglichkeit.
- **Kein Brief ans converter-mcp** — keine Agent-Fläche berührt (im Wrap so sagen, Hausregel seit 2026-09-19).
- **Memory** nur bei übertragbarer Lehre; Kandidat: *ein iframe erbt nichts — Theme-Signale, die der Browser selbst zeichnet (`color-scheme`, Scrollbars), müssen ins innere Dokument.* Prüfen, ob `reference_theme_scope_tokens_equal_specificity_order` das schon trägt; dann ergänzen statt duplizieren.
- **Im Bericht benennen**: Breite und Track-Farbe vorher/nachher je Zustand · welche Properties in welcher Engine greifen · der Befund am Library-Reader · dass das PDF unverändert ist.

## Stop
Smoke grün auf dem deployten Stand. **Commit + Push** (Fix und Smoke getrennt, Wrap eigener Commit). Dann warten.

## Nicht-Ziele

- **Kein** Umbau des Scroll-Containers, **kein** Anfassen von `syncReaderPaper`, Aa-Popover, Esc.
- **Kein** Fix am Library-Reader ohne gemessenen Befund.
- **Keine** neuen Vorlagen, **kein** Touch an `generate_pdf`.
- ⚠️ **Editiert wird nur auf dem Mac.** Die Mintbox ist Runtime — Deploy und Smoke ja, Arbeitsplatz nein; Wegwerf-User strikt nach `user_id` abräumen.
