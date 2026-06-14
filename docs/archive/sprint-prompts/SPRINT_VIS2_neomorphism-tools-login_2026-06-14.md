# Sprint VIS2 — Neomorphism-Angleichung: Audio + Mermaid + Login + Tags (M)

> **Executor-Doc. Fortsetzung von VIS1** (`SPRINT_VIS1_neomorphism-alignment_2026-06-14.md`) — der Abschluss der „alle Screens"-Entscheidung. VIS1 hat das **Fundament gemergt** (Tokens, Nunito, globale Komponenten) + Library + Reader + Markdown + Document gemacht. VIS2 zieht die **vier verbleibenden Screens** nach. Phasen nacheinander, nach jeder **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün. Visueller Sprint → **Live-Smoke pro Screen Pflicht** (`pytest` fängt CSS nicht).

## Pflichtlektüre vorab
1. **`SPRINT_VIS1_neomorphism-alignment_2026-06-14.md`** — die **8 harten Regeln**, die gesperrten Entscheidungen, die gesegneten Ausnahmen (Progress-Bar, Unicode-Glyphen), die Token-Strategie. Gilt 1:1 weiter.
2. `/Users/olivergluth/CODE/Neomorphism Design System/` — `CLAUDE_CODE_HANDOFF.md`, `readme.md`, die passenden `components/**` (`Input`, `Textarea`, `Select`, `DropZone`, `Button`, `Alert`, `Surface`, `PaneHeader`) + `ui_kits/file-transformer/` (Login + Audio-Screens).
3. **`static/css/style.css`** — das Fundament steht schon (lies, was VIS1 etabliert hat, bevor du etwas neu erfindest).

## Was VIS1 bereits etabliert hat (erben, NICHT neu bauen)
- **Nunito** global (base.html + login.html + `--nm-font`), neue Tokens in `:root` (`--nm-gap-control:32px`, `--nm-gap-section:48px`, `--text-*`, `--weight-*`, `--leading-*`, `--nm-surface-grad`, `--nm-sep-{top,bottom}`).
- **Tone-Tokens** `--nm-tone-{info,success,purple,warning,teal,danger,slate}(-ink)` für Badges; **Alert-Ink-Tokens** `--nm-alert-{danger,success,warning,info}-ink`. **Nie Hex hardcoden** — diese Tokens nutzen.
- **`.c-alert`** ist schon regel-7-konform (Tint + Ink + raised, kein border-left).
- **`.c-btn-row`** (24px-Boden für raised `.c-btn`-Geschwister) = die **Toolbar-Cluster-Konvention**. Wo immer in VIS2 ein Cluster aus raised `.c-btn` in einem Header/Toolbar sitzt → `c-btn-row` statt `gap-2`.
- **Segmented/Toggle-Muster**: flacher Track, **aktives = das eine gepresste Element** (`--nm-pressed-sm`), inaktiv flach + muted. Toggles: off=flach, on=pressed (Regel 5).
- **Hover-Reveal sekundärer Aktionen** via reinem CSS (`:hover`/`:focus-within`, `opacity` + `pointer-events`).
- **DropZone** ist bereits der korrekte „deeply-pressed well" — als Referenz für andere Drop-Flächen.

## Die anzuwendenden Regeln (Kurzform — Details in VIS1)
Elevation-Budget (max 2–3 Ebenen, innen flach, nur aktiv/primär plastisch, Sekundär auf Hover) · ≥32px zwischen raised-Geschwistern / ≥24px nie unterschreiten / 48px zwischen Gruppen · Status als Tint nie Fill/Bar · pressed=aktiv · Pill-Radii für Chips/Toggles/Badges · kein Hardcode (Tokens) · German „du", keine Emojis (gesegnete Unicode-Glyphen bleiben).

---

## Phase 1 — Audio-Tools (`audio_converter.html`, der schwerste Screen)

Dateien: `templates/audio_converter.html`, `static/css/style.css` (Sektion `AUDIO CONVERTER PAGE`), ggf. inline `<style>`.
1. **Mic-Button** (`#mic-button`) auf Budget: Ruhe raised, aktiv/aufnehmend = pressed/accent (das eine plastische Element des Recording-Blocks).
2. **Mode-Radios** (`.mode-radio-*`) als Segmented-/Toggle-Muster: flacher Track, aktiver Modus gepresst, inaktiv flach.
3. **Prompt-Editor** (`.prompt-editor-content`) + Inputs/Textareas → gepresst (carved-in), Tokens.
4. **Podcast-Flow** (Generierungs-Status, Polling-UI, Audio-Player): Budget + ≥32px-Spacing, Status per Tint nicht Bar; Buttons-Cluster → `c-btn-row`.
5. **Hardcoded Hexes** → Tokens. **Live-Smoke** (dark+light): Mic-Toggle-Zustände, Mode-Wechsel, Podcast-Flow rendert, 0 Console-Errors. Kern-Funktion (Transkription/Podcast) **nicht** anfassen — rein visuell.

**Stop + Bericht.**

## Phase 2 — Mermaid + Login + Tags

- **2A Mermaid** (`mermaid_converter.html`): kleiner Screen — Editor/Preview-Flächen + Buttons auf Budget/`c-btn-row`, Hardcode→Token.
- **2B Login** (`login.html`): die EINE gehobene `max-w-sm`-Surface + gepresste Inputs, Primary-Button = das plastische Element; Spacing ≥32px zwischen Feldern/Gruppen. (Font ist schon Nunito aus VIS1.)
- **2C Tags-Manager** (`tags.html` / `static/js/tags.js`-gerenderte `.tag-manager-card`-Reihen): Karten auf Budget (Reihe = flach getönt, nicht raised-pillow), ≥32px-Gaps, Delete-Aktion dezent/Hover, Hardcode→Token.

**Live-Smoke pro Sub-Batch** (dark+light, Kern-Funktion intakt, 0 Console-Errors). **Stop + Bericht.**

## Phase 3 — Wrap-up

**Commit-Disziplin wie VIS1: Code pro Phase committen (eigener Hash), Doc-Wrap separat, alle pushen (HEAD == origin).**
1. `STATUS.md` + `BACKLOG.md`: VIS2 ☑ done mit Hashes; **damit ist die „alle Screens"-Angleichung (VIS1+VIS2) komplett** — festhalten.
2. `docs/reader_architecture.md`: ggf. Halbsatz, dass die ganze App jetzt auf den nachgeschärften DS-Regeln steht.
3. **Bullet-Guard**: `grep -nE '(- \*\*.*){2,}' BACKLOG.md STATUS.md`.
4. `pytest tests/` final grün.

**Stop + Schluss-Bericht** — inkl. Olivers offenem Schritt: **Mintbox-Deploy** zieht jetzt R2-F + R2-G + VIS1 + VIS2 zusammen. **Wichtig (VIS1-Fund)**: Template-Änderungen brauchen in Prod `docker compose up -d --build` (kein Override-Mount → Templates ins Image gebacken; reiner `restart` reicht NICHT); keine Migration; danach Browser-Hard-Reload (Static-Cache).

## Out of scope
- Funktionale Änderungen (Audio/Podcast/Mermaid-Logik, Routen) — rein visuell/CSS/Markup.
- Die in VIS1 etablierten Tokens/Komponenten neu definieren — nur konsumieren.
- `pdf_styles/*` (gerenderter PDF-Output, nicht App-Chrome).
