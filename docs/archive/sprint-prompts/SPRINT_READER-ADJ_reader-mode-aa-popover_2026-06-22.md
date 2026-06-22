# Sprint READER-ADJ — Reader-Mode mit „Aa"-Popover (Library + markdown-converter) (L, 3 Phasen)

> **Executor-Doc.** Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline **393**). Du committest jede Phase selbst (eigener Hash + push). Arbeitsverzeichnis `/Users/olivergluth/CODE/CONVERTER`. Working-Practice in `CLAUDE.md` (Sektion „Working Practice"). **Frontend-Sprint** (JS/Templates/CSS, kein Backend) → die Test-Suite rendert keine Templates, **Live-Smoke ist der echte Gate** (dark+light, beide Flächen, 0 Console-Errors).

## Worum es geht (Olis Klärung 2026-06-22)

Vor langer Zeit wurde „Reader" als die **Distraction-Free-Ansicht im `library_detail`** umgesetzt (Sidebars ein-/ausklappen). Die ist gut und **bleibt unangetastet.** Was Oli *eigentlich* meinte, ist ein **Reader-Mode wie im markdown-converter**: eine **separate, zuschaltbare Leseansicht mit Spaltenbreite + Textgröße**. Der existiert dort schon — aber seine Steuerung, die floating **`.reader-toolbar`**, **hängt dauerhaft überm Text und stört** (Olis langjähriger Schmerz).

**Entscheidungen (Oli):**
1. **Controls = „Aa"-Popover on-demand** (ein dezentes Icon am Rand → Klick öffnet ein kleines Panel mit Breite + Textgröße → wegklicken/Esc = weg). **Nie** dauerhaft überm Text (Safari-/Kindle-Reader-Muster).
2. **Scope = beide Flächen konsistent**: neuen Reader-Mode in der **Library-Leseansicht** bauen **UND** die markdown-converter `.reader-toolbar` auf dasselbe „Aa"-Popover-Muster umstellen.

## Master-Design-Entscheidungen (gesetzt — Oli kann beim Sign-off vetoen)

- **Eine geteilte Komponente** `static/js/reader_settings.js` + geteiltes CSS, von **beiden** Seiten genutzt. Extrahiert die **vorhandene** Logik (`changeFontSize`/`changeWidth`/`WIDTH_MAP`/`applyWidth`/`getReaderPrefs`/`saveReaderPrefs`) — **nicht neu erfinden**, nur umziehen + generisch machen (Ziel-Container als Parameter statt fix `.main-container`).
- **„Aa"-Popover-Inhalt**: Textgröße `A−`/`A+` + Breite-Presets (narrow/medium/wide/ultrawide). **Trigger-Icon dezent am Rand/Eck** (wie die Distraction-Free-Floater), **nicht** über der Textspalte. Dismiss auf Outside-Click **und** Esc. DS-konform, token-driven (Popover-Optik vom `highlight-action-popover` übernehmen).
- **Library-Reader-Mode** = ein **Toggle** (spiegelt den markdown-converter-„Reader Mode"): fokussierte, **zentrierte** Leseansicht der `.reader-view`, blendet umgebende Chrome (Sidebars/Nav) aus, wendet `--reader-width` + `--reader-font-size` an, `Aa`-Popover für die Einstellungen, **Esc** verlässt. **In-Reader-Interaktionen bleiben** (Markieren/Highlights, Fortschrittsbalken). Die bestehenden Distraction-Free-Floater **bleiben** für den Nicht-Reader-Modus.
- **Dark**: der markdown-converter behält seinen reader-scoped Dark-Toggle (Preview-iframe). Die Library nutzt das **globale Theme** (kein separater Dark-Toggle im Library-Aa-Popover v1).
- **Persistenz**: geteiltes `readerPrefs` (`loadViewState`/`saveViewState`) → konsistente Lese-Settings über beide Reader.

## Verifizierte Code-Fakten (Master-gegroundet — bau darauf, reuse)

- **markdown_converter.js** ([static/js/markdown_converter.js](static/js/markdown_converter.js)): `toggleReaderMode()` (toggelt `.main-container.reader-mode` + `body.reader-active`), `changeFontSize(delta)` (`--reader-font-size`, clamp 12–32), `changeWidth(size)`/`applyWidth`/`WIDTH_MAP={narrow:600px,medium:800px,wide:1000px,ultrawide:80%}` (`--reader-width`), `getReaderPrefs`/`saveReaderPrefs` über `READER_PREFS_KEY='readerPrefs'`, Esc-to-exit-Handler. **Das ist die zu extrahierende Logik.**
- **markdown_converter.html** ([templates/markdown_converter.html](templates/markdown_converter.html)): die floating **`.reader-toolbar`** (Z. ~121–133: A−/A+, 4 Width-Buttons, Dark, Exit-×) = **das zu ersetzende Element**; der „Reader Mode"-Button (Z. 13) bleibt der Einstieg.
- **style.css** ([static/css/style.css](static/css/style.css)): `.reader-mode`-Regeln (ab ~792), `.reader-toolbar` (ab ~877, **zu retiren/ersetzen**), `--reader-width`/`--reader-font-size` greifen auf `.preview-iframe`/`.preview-page`/`.preview-content-area`; **`body.reader-active` blendet `#sidebar`/`.grid`/`#main-content`/Header aus** (Z. ~863–872) — **dieser Chrome-Hide-Mechanismus ist für den Library-Reader wiederverwendbar.**
- **library_detail.html** ([templates/library_detail.html](templates/library_detail.html)): `.reader-view` (Z. ~57) rendert den Artikel; die Distraction-Free-Floater (Z. ~26) = **behalten**.
- **library_detail.js** ([static/js/library_detail.js](static/js/library_detail.js)): `highlightActionPopover()` / `showHighlightActionPopover()` / `hideHighlightActionPopover()` + Outside-Click-Dismiss = **Popover-Präzedenz** (Mechanik + Optik fürs Aa-Panel übernehmen). Highlighting hängt an `.reader-view` → im Reader-Mode erhalten.
- **_utils.js**: `loadViewState(key, default)` / `saveViewState(key, state)`.

## Phase 1 — Geteilte Reader-Settings-Komponente + „Aa"-Popover + markdown-converter umstellen

Dies zuerst, weil die Fläche (markdown-converter) die Logik schon trägt → niedrigstes Risiko, der Refactor beweist die geteilte Komponente.

1. **`static/js/reader_settings.js`** — extrahiere `changeFontSize`/`changeWidth`/`WIDTH_MAP`/`applyWidth`/`updateWidthButtons`/`getReaderPrefs`/`saveReaderPrefs` aus markdown_converter.js in ein wiederverwendbares Modul. **Generisch**: Ziel-Container als Parameter (statt fest `.main-container`), damit Library + markdown-converter denselben Code nutzen. `readerPrefs`-Schema + `loadViewState`/`saveViewState` bleiben.
2. **„Aa"-Popover** (geteiltes Markup-Fragment + CSS, token-driven, DS-konform — Optik vom `highlight-action-popover`): dezenter **`Aa`-Trigger am Rand** (nicht über dem Text), Klick toggelt ein kleines Panel mit `A−`/`A+` + den 4 Breite-Presets; **Outside-Click + Esc schließen** das Panel. Aktiver Breite-/Größe-Zustand sichtbar.
3. **markdown-converter umstellen**: die floating `.reader-toolbar` **raus**, ersetzt durch den `Aa`-Trigger + Popover. `toggleReaderMode`/Dark/Exit-×/Esc-Verhalten bleibt (Dark + Exit dürfen im Popover oder als dezente Eck-Affordanz wohnen — **kein** Wieder-Einführen einer Dauer-Leiste überm Text). `markdown_converter.js` nutzt jetzt `reader_settings.js`. Tote `.reader-toolbar`-CSS entfernen.
4. **Live-Smoke markdown-converter** (lokale Instanz, MacChrome **dark+light**, **0 Console-Errors**): Reader-Mode an → Aa-Popover öffnet on-demand, Textgröße + Breite wirken auf die Preview, **keine Leiste mehr überm Text**, Outside-Click/Esc schließt das Panel, `readerPrefs` persistiert über Reload, Esc verlässt den Reader-Mode. `node --check` der berührten JS.

**Stop + Bericht.**

## Phase 2 — Library-Reader-Mode (`library_detail`)

1. **„Reader Mode"-Toggle** in `library_detail` (Einstieg dezent, z.B. neben den vorhandenen Reader-Affordanzen): aktiviert eine **fokussierte, zentrierte** Leseansicht der `.reader-view` — Chrome-Hide über den **wiederverwendeten `body.reader-active`-Mechanismus** (Sidebars/Nav aus), Spalte zentriert auf `--reader-width`, Text auf `--reader-font-size`.
2. **`Aa`-Popover** (aus Phase 1) in der Leseansicht: Breite + Textgröße, dieselbe on-demand-Mechanik, dezenter Rand-Trigger. Persistenz via geteiltem `readerPrefs`.
3. **Esc** verlässt den Reader-Mode (wie markdown-converter). **Erhalten bleiben**: Highlighting/Markieren auf `.reader-view`, der Reading-Progress-Balken, und die bestehenden **Distraction-Free-Floater** (unangetastet — koexistieren).
4. **Live-Smoke library_detail** (echter Artikel mit Markup, MacChrome **dark+light**, **0 Console-Errors**): Reader-Mode an → zentrierte Spalte, Aa-Popover wirkt (Breite/Größe), Esc raus; **Highlighting funktioniert im Reader-Mode weiter**; Distraction-Free-Floater unverändert; `readerPrefs` über beide Reader konsistent (im markdown-converter gesetzte Größe wirkt auch hier). `node --check`.

**Stop + Bericht.**

## Phase 3 — Wrap

1. **STATUS.md** + **BACKLOG.md**: READER-ADJ ☑ done mit Hashes; „Aktiv offen"-Block leeren bzw. auf **Web-Article-Save (P2)** als nächstes zeigen (alle P1 durch). STATUS „Aktueller Sprint" = READER-ADJ, MCP-DOCWRITE → Vorheriger. **Bullet-Guard.**
2. **Doc**: kurzer Eintrag in [docs/reader_architecture.md](docs/reader_architecture.md) — die Reader-Mode-/Aa-Popover-Mechanik + die Drei-Teilung (Distraction-Free-Floater = feingranulares Sidebar-Collapse · Reader-Mode = fokussierte Leseansicht mit Breite/Größe · geteiltes `reader_settings.js` über markdown-converter + library).
3. **Memory** (`reference_*`, optional nach deiner Einschätzung): die geteilte Reader-Settings-Komponente + das „Controls on-demand statt Dauer-Leiste überm Text"-Pattern, falls als reusable UX-Lehre wertvoll.
4. Finaler `pytest tests/` grün (**393**, unverändert — kein Backend-Touch).

**Stop + Schluss-Bericht** — inkl. Deploy-Hinweis: reiner Frontend-/Template-/Static-Touch, **keine Migration, kein neuer Dep**; Mintbox `git pull` + `docker compose up -d --build` (Templates ins Image → `--build`, nicht `restart`) + Browser-Hard-Reload.

## Bewusst NICHT (Scope-Grenze)

- **Distraction-Free-Floater** (Sidebar-Collapse) im `library_detail` **nicht** ändern — bleiben wie sie sind.
- **Kein** neues Feature über Breite + Textgröße hinaus (kein Schriftart-Wechsel, kein Zeilenabstand-Regler v1 — YAGNI, kann Folge-Item werden).
- **Kein** reader-scoped Dark-Toggle in der Library v1 (globales Theme).
- **Kein** Backend-/Schema-Touch, keine neuen Endpoints.
- **Keine** neue Dauer-Leiste überm Text — der Anti-Pattern, den dieser Sprint *behebt*, darf nirgends wieder auftauchen.

## Akzeptanz

- [ ] markdown-converter: floating `.reader-toolbar` **weg**, Breite + Textgröße (+ Dark) über das **on-demand `Aa`-Popover**; Verhalten + `readerPrefs`-Persistenz unverändert; Esc verlässt; dark+light gesmoked, 0 Console-Errors
- [ ] library_detail: **neuer Reader-Mode-Toggle** → fokussierte zentrierte Leseansicht, Breite + Textgröße über dasselbe `Aa`-Popover, Esc raus; Highlighting + Progress intakt; Distraction-Free-Floater unangetastet; dark+light gesmoked
- [ ] `Aa`-Popover **nie** dauerhaft überm Text; Dismiss auf Outside-Click + Esc; Trigger dezent am Rand
- [ ] **eine geteilte** `reader_settings.js` von beiden Flächen genutzt; `readerPrefs` konsistent
- [ ] `pytest tests/` grün (393, kein Backend-Touch); `node --check` der berührten JS
