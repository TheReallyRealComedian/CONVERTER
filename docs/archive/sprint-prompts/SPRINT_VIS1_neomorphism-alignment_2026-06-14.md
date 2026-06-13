# Sprint VIS1 — Neomorphism-Design-System-Angleichung (L+)

> **Executor-Doc.** Phasen strikt nacheinander, nach jeder Phase **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün. Dies ist ein **visueller** Sprint — `pytest` fängt CSS-/Layout-Bugs **nicht** (rendert Templates, prüft keine Optik), **Live-Smoke nach jedem Screen ist Pflicht** (CLAUDE.md Test-Suite-Limit). Zeilennummern unten = Orientierung (driften), geh über Datei + Selektor/Funktion.

## Kontext / Warum

Das Design System unter `/Users/olivergluth/CODE/Neomorphism Design System` wurde **aus CONVERTER selbst extrahiert** — CONVERTER ist die Quelle. Aber im Design-Review wurden **mehrere Regeln nachgeschärft** („the rules we just tightened in design review"), und die fließen jetzt zurück. Das ist **kein Reskin** (Farben/Radii/Shadow-Primitives sind identisch), sondern eine **Regel-Angleichung** + **Brand-Font-Wechsel**.

**Pflichtlektüre vorab (in dieser Reihenfolge), das ist die Source-of-Truth:**
1. `/Users/olivergluth/CODE/Neomorphism Design System/CLAUDE_CODE_HANDOFF.md` — die 8 harten Regeln + „How to apply".
2. `…/readme.md` — die volle Design-Sprache.
3. `…/tokens/*.css` — die exakten Werte (font, colors, typography, spacing, shadows, dark).
4. `…/components/**/*.{jsx,prompt.md}` — Ziel-Muster pro Komponente. **Besonders** `components/data-display/LibraryCard.jsx` (+`.prompt.md`) = die Elevation-Budget-Referenz, dazu `Badge`, `Tag`, `Segmented`, `Button`, `IconButton`, und `components/forms/*`, `components/surfaces/*`.
5. `…/guidelines/elevation-budget.card.html` + die anderen `*.card.html` — visuelle Spezimen.
6. `…/ui_kits/file-transformer/` — Hi-Fi-Klick-Recreation ganzer Screens (höchste Fidelity-Referenz).

## Die 8 harten Regeln (aus dem Handoff — alle einhalten)

1. **Eine Flächenfarbe** — Page/Cards/Inputs/Sidebar alle `--nm-bg`. Tiefe nur durch Shadow.
2. **Tiefe = Shadow, nie Borders** — kein `border:` zur Trennung; `--nm-raised*`/`--nm-pressed*` oder die Hairline `--nm-sep-*`.
3. **Elevation-Budget — max 2–3 Ebenen sichtbar.** Innerhalb einer gehobenen Fläche ist der Default **flach**; nur das **eine** aktive/primäre Element ist plastisch (gepresstes aktives Toggle, Primary-Button). Kein eigenes raised/pressed an jedem Badge/Tag/Button. Sekundär-Info trägt Status per **Tönung** (12% Tint + muted Text), nicht per Elevation. Sekundär-Aktionen (Copy/Löschen/Flag) erst auf **Hover**.
4. **Shadows need air** — Gaps zwischen neomorphen Geschwistern **≥32px** (`--nm-gap-control`), nie unter 24px; **48px** zwischen Gruppen (`--nm-gap-section`). Nie am 16px-`md`-Schritt clustern.
5. **Pressed-in = aktiv/selektiert** — Inputs immer gepresst, Toggles flippen auf gepresst wenn an.
6. **Nur weiche Radii** — 12–20px Flächen, 100px Pills für Chips/Toggles/Badges.
7. **Status als Tint, nie Fill oder Bar** — Alerts/Badges = 12% Tint + farbiger Text. Keine soliden Fills, **keine farbigen Status-Left-Border-Bars.**
8. **Leiser Akzent** — Periwinkle als Alltag, das warme Orange nur für die seltene Primary-CTA. Danger = weicher Terracotta-Wash + Danger-Ink, kein gesättigtes Rot.

**Plus**: kein Hardcode — **immer `--nm-*`-Vars**, nie Hex/Shadow-Literal.

### Gesegnete Ausnahmen — NICHT „bereinigen"
- **Die Fortschritts-Bar bleibt.** Regel 7 verbietet farbige *Status*-Bars (Left-Border-Marker), **nicht** den Reading-Progress-Indikator — das Orange-CTA-Reading-Progress-Gradient ist im System ausdrücklich gesegnet (readme: „warm orange … used sparingly (e.g. the reading-progress gradient)"). Card-Progress (R2-B) + Reader-Progress-Bar (R2-F/G furthest-read) + das „Gelesen"-Label **bleiben funktional erhalten**.
- **Gesegnete Unicode-Glyphen bleiben** (kein „De-Emoji"): `★/☆` Favorit, `⚑/⚐` Lese-Liste, `×` Dismiss, `▲/▼` Reorder, `🌙/☾` Reader-Dark-Toggle, Block-Glyphen `▌▌▌` Reader-Width-Stepper.
- **Echte Emojis** gibt es im Chrome ohnehin nicht — falls welche auftauchen, raus (Regel: keine Emojis).

## Gesperrte Entscheidungen (Workshop Master + Oliver 2026-06-14 — nicht neu diskutieren)

- **Scope = volle Angleichung, ALLE Screens** (Library, Reader/Detail, Markdown, Document, Audio, Mermaid, Login, Tags-Page).
- **Brand-Font Inter → Nunito** (gesetzt, sichtbarste Änderung).
- **Library-Karte: R2-Funktion erhalten, Ruhezustand flach.** Das Segmented-Status-Control + Queue/Favorit **bleiben funktional** (Ein-Klick-Triage von der Karte bleibt — R2-C/E nicht zurückbauen), aber der **Ruhezustand wird flach**: nur das **aktive** Segment ist gepresst (das eine Level-2-Element), Copy/Löschen erst auf **Hover**. Type/Status per Tönung. NICHT der `LibraryCard.jsx`-read-only-Ort-Chip-Weg — der entfernt die Triage; nutze die Referenz für Tönung/Spacing/Hover/Flachheit, aber behalte die interaktiven Controls.
- **Token-Strategie: in die bestehende EINE `style.css` `:root` mergen**, NICHT die `tokens/*.css` vendoren/`@import`en. CONVERTER-Konvention = ein handgeschriebenes Stylesheet (CLAUDE.md: „not split by design"); der Handoff sagt explizit „follow the codebase's established patterns … don't introduce a parallel set".

---

## Phase 1 — Token-Fundament + Nunito (global, additiv)

Dateien: `static/css/style.css` (`:root` + Dark-Block), `templates/base.html`.

1. **Font Inter → Nunito**, drei Stellen: `base.html` Google-Fonts-`<link>` (~Z.9, auf `family=Nunito:wght@400;500;600;700&display=swap`), `base.html` Tailwind-Config `sans:`-Alias (~Z.23), `style.css` `--nm-font` (~Z.40, `'Nunito', -apple-system, …`). Mono-Stack unverändert.
2. **Neue Tokens in `:root` ergänzen** (Werte 1:1 aus den `tokens/*.css`): `--nm-gap-control: 32px`, `--nm-gap-section: 48px`, `--nm-space-2xl: 48px`; die `--text-*`-Skala (xs..3xl), `--weight-{regular,medium,semibold,bold}`, `--leading-{tight,snug,normal,relaxed}`, `--eyebrow-size`/`--eyebrow-spacing`. `--nm-surface-grad` als Token promoten (aktuell inline in `.c-card`) + `.c-card` darauf umstellen. `--nm-sep-top`/`--nm-sep-bottom` als Token; die bestehenden `.neo-separator-*`-Klassen diese Tokens konsumieren lassen (DRY, kein Doppel-Wert).
3. **Dark-Block angleichen**: prüfen, dass die komponierten Dark-Shadows den **weißen Bevel-Highlight droppen** (Design-System-`dark.css` definiert `--nm-raised{,-sm,-lg}` ohne `inset … rgba(255,255,255,…)`; Glare auf Dark). Falls CONVERTERs Dark-Block den Bevel noch trägt → entfernen. Dark-Tints 15%, `--nm-surface-grad` dark re-raked — gegen `dark.css` abgleichen.
4. **Optionale Semantik-Aliase** (`--surface-page`, `--text-label` …) nur übernehmen, wenn du sie tatsächlich nutzt — kein toter Token-Ballast.

Erwartung: außer dem Font **keine** sichtbare Regression (rein additive Tokens). Smoke: App lädt, Nunito rendert (DevTools computed `font-family`), dark+light, nichts bricht. `pytest` grün (base.html-Change → Jinja-Render-Test würde Fehler zeigen).

**Stop + Bericht.**

## Phase 2 — Library-Liste + Karte (die Flaggschiff-Referenz)

Dateien: `templates/library.html` (das `conversion_card`-Macro), `static/css/style.css`, ggf. `static/js/library.js` (nur falls Hover-Reveal JS braucht — bevorzugt rein CSS via `:hover`).

Ziel-Muster = `LibraryCard.jsx`, aber mit **erhaltener R2-Funktion** (s. gesperrte Entscheidung):
1. **Karte = einzige gehobene Ruhe-Fläche** (Level 1): `--nm-surface-grad`, `--nm-raised` → Hover `--nm-raised-lg` + `translateY(-2px)`. Innen **alles flach**.
2. **Type + Status per Tönung statt Elevation**: Type-Badge von hardcoded Hexes (~Z.1582-1597, `#1e40af` etc.) auf Tint+muted-Text-Muster; Status-Badge (R2-C) flach (Tint + Dot). Tags = flache `--nm-tint-accent`-Pills, kein Shadow.
3. **Das eine Level-2-Element** = das aktive Status-Segment (gepresst). Die nicht-aktiven Segmente flach. Queue-Btn/Favorit als **flache Glyph-Buttons** (kein Ruhe-Shadow), Hover hebt dezent.
4. **Copy/Löschen erst auf Hover** (opacity 0→1, `pointer-events`), Ruhezustand ruhig.
5. **Spacing ≥32px** zwischen Control-Clustern; Grid-Gaps prüfen (Karten-Grid, Filter-Bar, Ort-/Tag-Leisten). Keine 16px-Cluster.
6. **Progress-Bar bleibt** (gesegnet) — flach an der Karten-Unterkante, nicht als Status-Bar missverstehen.
7. **Tab-Leiste/Ort-Filter/Tag-Leiste** (R2-E) auf Budget + Spacing prüfen; aktiver Tab/Chip = das gepresste Element der jeweiligen Gruppe.

**Live-Smoke (Pflicht, echte Klicks, dark+light)**: R2-Funktion muss **überleben** — Ein-Klick-Triage (Inbox→Lese-Liste→zurück), Status-Segment, Queue-Toggle + Reorder, Favorit, Tab-Wechsel, Ort-/Tag-Filter, Typeahead, „+N weitere", Progress-Balken sichtbar. Hover-Reveal von Copy/Löschen. Keine Console-Errors.

**Stop + Bericht** (Screenshot-Beschreibung Ruhe- vs. Hover-Zustand der Karte).

## Phase 3 — Reader / Library-Detail

Dateien: `templates/library_detail.html`, `static/css/style.css`, ggf. `static/js/library_detail.js` (nur CSS-nahe Hooks).

1. **Detail-Sidebar-Cards** (Status, Lese-Fortschritt/R2-G, Lese-Liste, Markierungen, Notion, Tags) auf Budget: jede `c-surface--flat`-Card ist die gehobene Ebene, innen flach; aktives Segment/Toggle = das eine gepresste Element; ≥32px Gaps zwischen den Cards (`--nm-gap-section`).
2. **Reader-View / Highlights**: Highlight-Popover, Sidebar-Highlight-Cards, Tag-Chips auf Budget + Tint. Highlight-Tint bleibt (`--nm-tint-highlight`).
3. **R2-F/G-Elemente bleiben funktional**: Abschluss-Leiste (Zurück/Archivieren), „Gelesen"-Label, furthest-read-Bar, „Als ungelesen"-Reset-Button — nur visuell auf Budget ziehen, **Verhalten nicht anfassen**.
4. **Reader-Toolbar** (Width-Stepper, Dark-Toggle, Sidebar-Floater) auf Pill-Radii + Budget.

**Live-Smoke (Pflicht, echtes Scrollen — Memory-Caveat `reference_scroll_progress_persistence`: occluded MCP-Tab pausiert rAF; der furthest-read-Bar-Kern ist seed-getrieben also direkt beobachtbar)**: Progress-Bar = furthest-read, Reset → NULL + Label weg, Abschluss-Leiste, Highlight setzen/löschen, Tags, dark+light. DB/Theme nach Smoke restaurieren.

**Stop + Bericht.**

## Phase 4 — Konverter-Tools + Login + Tags-Page (Breite)

Sub-batchbar; **stopp + berichte, falls es lang wird** — Master entscheidet dann ggf. den Rest in einen VIS2 abzuspalten. Reihenfolge-Vorschlag:
- **4A**: `markdown_converter` (Split-Editor/Preview, Pane-Header, Reader-Mode, Theme-Select, PDF-Buttons) + `document_converter` (DropZone — `components/forms/DropZone.jsx` als Referenz).
- **4B**: `audio_converter` (Mic-Button, Mode-Radios, Prompt-Editor, Podcast-Flow) + `mermaid_converter`.
- **4C**: `login` (max-w-sm, das eine gehobene Surface + gepresste Inputs) + `tags`-Manager-Page (`.tag-manager-card`-Reihen auf Budget).

Pro Screen: Tokens/Font erben schon aus P1; anzuwenden sind **Elevation-Budget, ≥32px-Spacing, Status-als-Tint, Hardcoded-Hex→Var, Pill-Radii, Hover-Reveal sekundärer Aktionen**. Referenzen: die passenden `components/`-Muster + `ui_kits/file-transformer/`-Screens. **Live-Smoke pro Sub-Batch** (jeder Tool-Screen einmal durchklicken, dark+light, keine Console-Errors, Kern-Funktion intakt).

**Stop + Bericht** (nach jedem Sub-Batch ok, mindestens am Phase-Ende).

## Phase 5 — Wrap-up

**Commit-Disziplin wie R2-x: Code pro Phase committen (eigener Hash), Doc-Wrap separat, alle pushen (HEAD == origin halten).** (Du kannst auch pro Phase committen+pushen — dann ist nichts un-deployed liegengelassen.)

1. `STATUS.md` + `BACKLOG.md`: VIS1 ☑ done mit Hashes; falls Phase 4 (oder Teile) abgespalten → VIS2 als Folge-P1 eintragen.
2. `docs/reader_architecture.md`: kurze Notiz, dass die Reader-Screens auf die nachgeschärften DS-Regeln angeglichen sind (Elevation-Budget, Nunito, Spacing) — Decision-Log-Zeile.
3. **Memory**: deine Einschätzung — eine generalisierbare Lehre („Design-System wurde aus dieser App extrahiert + nachgeschärft → Rück-Angleichung ist Elevation-Budget-Audit, nicht Reskin") könnte einen `reference_*`-Eintrag wert sein; nicht erzwingen.
4. **Bullet-Guard** vor dem Doc-Commit: `grep -nE '(- \*\*.*){2,}' BACKLOG.md STATUS.md`.
5. `pytest tests/` final grün.

**Stop + Schluss-Bericht** — inkl. Olivers offenem Schritt: **Mintbox-Deploy** (`git pull` + `docker compose up -d --build`; **keine Migration** — reiner CSS/Template/Static-Touch; danach **Browser-Hard-Reload** wegen Static-Asset-Cache). Falls R2-F/R2-G dort noch nicht deployed sind, zieht **ein** Deploy alles zusammen.

## Out of scope
- Funktionale Änderungen (R2-Verhalten, Routen, Endpoints) — rein visuell/CSS/Markup.
- Die `tokens/*.css` vendoren / Split-Stylesheet — Tokens werden in die eine `style.css` gemergt.
- Neue Komponenten/Features — nur Angleichung des Bestands an die Regeln.
- React-Code aus `components/` 1:1 übernehmen — das sind **Referenzen**, nicht zum Pasten (CONVERTER = Jinja + Vanilla + handgeschriebenes CSS).
