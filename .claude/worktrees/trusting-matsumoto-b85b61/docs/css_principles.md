# CSS & STYLE Playbook

**Das komplette Referenzdokument für professionelle CSS-Entwicklung 2025/2026.**
Dieses Playbook ist framework- und theme-agnostisch. Es beschreibt Architektur, Prinzipien, Werkzeuge und Techniken, die unabhaengig vom visuellen Design gelten.

---

## Inhaltsverzeichnis

1. [Architektur-Methodologien](#1-architektur-methodologien)
2. [Namenskonventionen](#2-namenskonventionen)
3. [Dateistruktur & Organisation](#3-dateistruktur--organisation)
4. [Design Tokens](#4-design-tokens)
5. [Cascade Layers (`@layer`)](#5-cascade-layers-layer)
6. [Specificity Management](#6-specificity-management)
7. [CSS Custom Properties (Variablen)](#7-css-custom-properties-variablen)
8. [Modernes CSS: Die wichtigsten Features](#8-modernes-css-die-wichtigsten-features)
9. [Responsive Design: Fluid & Intrinsic](#9-responsive-design-fluid--intrinsic)
10. [Performance-Optimierung](#10-performance-optimierung)
11. [Code-Qualitaet & Linting](#11-code-qualitaet--linting)
12. [CSS Nesting: Regeln & Fallstricke](#12-css-nesting-regeln--fallstricke)
13. [Farb-Management](#13-farb-management)
14. [Typografie-System](#14-typografie-system)
15. [Spacing-System](#15-spacing-system)
16. [Animation & Transitions](#16-animation--transitions)
17. [Accessibility (a11y)](#17-accessibility-a11y)
18. [Dark Mode & Theming](#18-dark-mode--theming)
19. [Browser-Kompatibilitaet](#19-browser-kompatibilitaet)
20. [Anti-Patterns & haeufige Fehler](#20-anti-patterns--haeufige-fehler)
21. [Checkliste: Neues Projekt aufsetzen](#21-checkliste-neues-projekt-aufsetzen)
22. [Glossar](#22-glossar)

---

## 1. Architektur-Methodologien

Die Wahl der CSS-Architektur ist die wichtigste Entscheidung vor dem ersten geschriebenen Selektor. Sie bestimmt Wartbarkeit, Skalierbarkeit und Teamproduktivitaet.

### 1.1 BEM (Block Element Modifier)

Entwickelt vom Yandex-Team. Jede Klasse folgt dem Muster `.block__element--modifier`.

```css
.card { }                      /* Block */
.card__title { }               /* Element */
.card__title--highlighted { }  /* Modifier */
```

**Vorteile:**
- Flache Selektoren mit einheitlicher Spezifitaet (0,1,0)
- Explizite Strukturkodierung — jeder Selektor ist selbsterklaerend
- Teams berichten von 40-60% weniger CSS-Bugs
- Framework-agnostisch, funktioniert ueberall

**Nachteile:**
- Verbose Klassennamen (`.navigation__menu-item--highlighted`)
- Kann HTML aufblaehen

**Wann einsetzen:** Grosse Teams, langlebige Projekte, Design-Systeme.

### 1.2 ITCSS (Inverted Triangle CSS)

Von Harry Roberts. Organisiert CSS in sieben aufsteigenden Schichten. Drei Metriken steigen mit jeder Schicht: Explizitheit, Spezifitaet, Lokalitaet.

```
Settings    → Variablen, Konfiguration (niedrigste Spezifitaet)
Tools       → Mixins, Funktionen
Generic     → Reset, Normalize
Elements    → h1-h6, a, p, img
Objects     → .o-container, .o-grid
Components  → .c-card, .c-button
Utilities   → .u-hidden, .u-sr-only (hoechste Spezifitaet)
```

ITCSS laesst sich direkt auf CSS Cascade Layers abbilden:
```css
@layer settings, generic, elements, objects, components, utilities;
```

**Wann einsetzen:** Mittel-grosse bis grosse Projekte. Natuerliche Architektur fuer natives CSS mit `@layer`.

### 1.3 SMACSS (Scalable and Modular Architecture for CSS)

Kategorisiert Regeln in Base, Layout (`l-`), Module, State (`is-`), Theme.

**Wann einsetzen:** Modernisierung von Legacy-Codebases — weniger preskriptiv als BEM, eher Organisationsleitlinie.

### 1.4 OOCSS (Object-Oriented CSS)

Von Nicole Sullivan. Trennt Struktur von Skin und Container von Inhalt.

**Kernprinzipien:**
1. Trenne Struktur (width, height, margin) von Skin (color, background, border)
2. Trenne Container von Inhalt — Styles sollen nicht von der Position im DOM abhaengen

Tailwind CSS ist letztlich OOCSS auf die Spitze getrieben.

### 1.5 CUBE CSS

Von Andy Bell. Vier Schichten:
- **Composition** — Layout-Primitiven (Stack, Cluster, Sidebar)
- **Utility** — Einzelzweck-Klassen
- **Block** — Minimale Komponenten-Styles
- **Exception** — Variationen ueber `data-`Attribute statt Modifier-Klassen

Kernunterschied zu BEM: Modifier werden als `data-state="reversed"` statt als CSS-Klassen ausgedrueckt.

**Wann einsetzen:** Moderne Projekte mit wenig CSS, die die Kaskade aktiv nutzen wollen.

### 1.6 Empfehlung nach Szenario

| Szenario | Empfohlene Methodik |
|---|---|
| Grosses Team, etabliertes Projekt | BEM + ITCSS |
| Design-System / Component Library | BEM + ITCSS + Token-Hierarchie |
| Legacy-Codebase modernisieren | SMACSS + BEM schrittweise einfuehren |
| Modernes Projekt, wenig CSS | CUBE CSS + `@layer` |
| Enterprise, Multi-Team | SMACSS-Organisation + BEM-Naming + `@layer` |

### 1.7 Der Hybrid-Ansatz 2025

Erfolgreiche Teams kombinieren:
- **ITCSS-Schichten** fuer Architektur
- **BEM** fuer Naming
- **OOCSS-Prinzipien** fuer Wiederverwendbarkeit
- **`@layer`** fuer native Kaskadenkontrolle

---

## 2. Namenskonventionen

### 2.1 BEMIT (BEM + ITCSS Namespaces)

Von Harry Roberts. Ergaenzt BEM um Namespace-Prefixe:

| Prefix | Zweck | Beispiel |
|--------|-------|---------|
| `o-` | Objects (Layout-Primitiven) | `.o-container`, `.o-grid` |
| `c-` | Components | `.c-button`, `.c-card` |
| `u-` | Utilities | `.u-hidden`, `.u-sr-only` |
| `l-` | Layout | `.l-sidebar`, `.l-main` |
| `is-` / `has-` | State | `.is-active`, `.has-children` |
| `js-` | JavaScript-Hooks | `.js-toggle`, `.js-modal` |
| `t-` | Theme | `.t-dark`, `.t-brand` |

**Regel:** Verwende `js-`-Klassen ausschliesslich fuer JavaScript-Selektoren, nie fuer Styling. So kann CSS refactored werden ohne JS zu brechen.

### 2.2 Custom-Property-Naming

```
--{kategorie}-{eigenschaft}-{variante}-{zustand}
```

Beispiele:
```css
--color-primary
--color-primary-hover
--font-size-body
--space-md
--shadow-raised-sm
--transition-normal
```

### 2.3 Datei-Naming

- Kebab-case fuer Dateinamen: `button-group.css`, `ai-chat.css`
- Nummerische Prefixe fuer Sortierung bei ITCSS: `01-settings/`, `02-tools/`
- Komponenten-Dateien neben ihren JS-Dateien in komponentenbasierten Frameworks

---

## 3. Dateistruktur & Organisation

### 3.1 Das 7-1 Pattern (Sass-Projekte)

Sieben Ordner plus eine zentrale Import-Datei:

```
/styles
  /abstracts/     → Variablen, Mixins, Funktionen
  /base/          → Reset, Typografie, Globales
  /components/    → Buttons, Cards, Forms
  /layout/        → Header, Footer, Grid, Sidebar
  /pages/         → Seitenspezifisches CSS
  /themes/        → Dark Mode, Brand-Varianten
  /vendors/       → Third-Party CSS
  main.scss       → Import-Datei
```

### 3.2 ITCSS-basierte Struktur (natives CSS)

```
/css
  style.css              → Entry-Point: @layer + @import
  /01-settings/          → tokens.css (Custom Properties)
  /02-generic/           → reset.css, normalize.css
  /03-elements/          → typography.css, links.css
  /04-objects/            → container.css, grid.css
  /05-components/        → button.css, card.css, form.css
  /06-utilities/         → hidden.css, sr-only.css
```

### 3.3 Modularer Ansatz mit @import

Fuer Projekte ohne Build-Pipeline (reines CSS):

```css
/* style.css — Entry Point */
@layer framework;
@import url("bootstrap.min.css") layer(framework);

@import url("tokens.css");
@import url("base.css");
@import url("layout.css");
@import url("components.css");
@import url("overrides.css");
@import url("dark.css");
```

**Wichtig:** `@import` muss vor allen anderen Regeln stehen (ausser `@charset` und `@layer`-Deklarationen).

### 3.4 Performance-Hinweis: @import vs. \<link\>

`@import` erzeugt sequentielle Fetches (Wasserfall-Effekt). Fuer Produktions-Performance:
- `<link rel="preload" as="style">` fuer kritische CSS-Dateien im HTML-Head
- Oder: Build-Schritt der alle CSS-Dateien konkateniert (`cat *.css > bundle.css`)
- Fuer interne Tools ist der Wasserfall-Effekt bei kleinen Dateien (<5KB) akzeptabel

---

## 4. Design Tokens

Design Tokens sind die kleinsten, unteilbaren Einheiten eines Design-Systems. Sie speichern visuelle Entscheidungen plattformunabhaengig.

### 4.1 Das Drei-Schichten-Modell

```
Primitive Tokens    →  Rohwerte            →  --blue-500: #2563eb
Semantische Tokens  →  Zweck/Bedeutung     →  --color-primary: var(--blue-500)
Komponenten-Tokens  →  Komponentenspezif.  →  --button-bg: var(--color-primary)
```

**Regel:** Verwende in Komponenten-CSS nur semantische oder Komponenten-Tokens. Nie Primitive direkt.

```css
/* Primitiv — nie direkt in Komponenten */
:root {
    --blue-600: #2563EB;
    --gray-900: #111827;
    --gray-50: #F9FAFB;
}

/* Semantisch — Zweck statt Farbe */
:root {
    --color-primary: var(--blue-600);
    --color-text: var(--gray-900);
    --color-surface: var(--gray-50);
}

/* Theming durch Ueberschreiben der semantischen Schicht */
[data-theme="dark"] {
    --color-primary: #818cf8;
    --color-text: #e1e1ff;
    --color-surface: #27272c;
}
```

### 4.2 Token-Kategorien

Ein vollstaendiges Token-System deckt ab:

| Kategorie | Beispiel-Tokens |
|-----------|----------------|
| Farben | `--color-primary`, `--color-surface`, `--color-text`, `--color-success` |
| Typografie | `--font-size-body`, `--font-family-sans`, `--line-height-tight` |
| Spacing | `--space-xs`, `--space-sm`, `--space-md`, `--space-lg`, `--space-xl` |
| Border Radius | `--radius-sm`, `--radius-md`, `--radius-lg`, `--radius-pill` |
| Shadows | `--shadow-sm`, `--shadow-md`, `--shadow-lg` |
| Transitions | `--transition-fast`, `--transition-normal` |
| Z-Index | `--z-dropdown`, `--z-modal`, `--z-toast` |
| Layout | `--sidebar-width`, `--header-height` |

### 4.3 Skalierungen: T-Shirt-Groessen

Fuer Spacing, Typografie und andere Skalen haben sich T-Shirt-Groessen bewaehrt:

```css
:root {
    --space-xs:  4px;
    --space-sm:  8px;
    --space-md:  16px;
    --space-lg:  24px;
    --space-xl:  32px;
    --space-xxl: 48px;
}
```

Vorteile: Intuitiv fuer Nicht-Techniker, endliche Skala verhindert willkuerliche Werte.

### 4.4 W3C Design Tokens Standard (DTCG)

Die W3C Design Tokens Community Group hat 2025 die erste stabile Spezifikation veroeffentlicht. Vendor-neutrales JSON-Format:

```json
{
  "color": {
    "primary": {
      "$value": "#2563eb",
      "$type": "color"
    }
  },
  "spacing": {
    "md": {
      "$value": "16px",
      "$type": "dimension"
    }
  }
}
```

**Style Dictionary v4** (Amazon) generiert aus diesen Token-Definitionen plattformspezifische Outputs — CSS Custom Properties, SCSS, Swift, Kotlin.

---

## 5. Cascade Layers (`@layer`)

Cascade Layers sind der wichtigste Architektur-Baustein in modernem CSS. Sie stehen in der Kaskade *ueber* der Spezifitaet.

### 5.1 Grundprinzip

Ein Selektor in einem spaeteren Layer gewinnt immer, unabhaengig von seiner Spezifitaet:

```css
/* Layer-Reihenfolge deklarieren — einmal, am Anfang */
@layer reset, framework, base, components, utilities;

@layer reset {
    *, *::before, *::after { box-sizing: border-box; margin: 0; }
}

@layer components {
    .button { color: white; background: blue; }
}

@layer utilities {
    .text-center { text-align: center; }
    /* Gewinnt IMMER ueber components — egal welche Spezifitaet */
}
```

### 5.2 Third-Party CSS in Layer importieren

```css
@layer framework;
@import url("bootstrap.min.css") layer(framework);
```

Eigene Styles (unlayered oder in spaeteren Layers) gewinnen automatisch ueber Bootstrap — ohne `!important`-Hacks.

### 5.3 !important und Layer-Umkehr

**Kritisch:** `!important` kehrt die Layer-Reihenfolge um!

```
Normale Deklarationen (spaeterer Layer gewinnt):
  reset < framework < base < components < utilities < unlayered

!important Deklarationen (frueherer Layer gewinnt):
  unlayered < utilities < components < base < framework < reset
```

Praktische Konsequenz:
- `!important` in einem Reset-Layer hat die hoechste Prioritaet — ideal fuer Accessibility-Resets
- Ein `!important` in `framework` (Bootstrap) schlaegt `!important` in `components`
- Um Bootstraps `!important` zu schlagen: unlayered `!important` verwenden

### 5.4 Empfohlene Strategie

```css
/* Bootstrap in Layer → unsere normalen Styles gewinnen automatisch */
@layer framework;
@import url("bootstrap.min.css") layer(framework);

/* Eigene Styles bleiben unlayered (hoechste Prioritaet fuer normal) */
.card { background: white; }          /* schlaegt Bootstrap .card */

/* Fuer Bootstrap-Utility-Overrides: unlayered !important */
.text-muted { color: gray !important; }  /* schlaegt BS layered !important */
```

---

## 6. Specificity Management

### 6.1 Die vollstaendige Kaskaden-Aufloesung 2025/2026

Von niedrigster zu hoechster Prioritaet:

1. **Origin & Importance** (User-Agent < User < Author)
2. **Context** (Shadow DOM Boundaries)
3. **Cascade Layers** (deklarierte Reihenfolge)
4. **Spezifitaet** (ID > Klasse > Element)
5. **Scoping Proximity** (`@scope`)
6. **Source Order** (spaetere Regel gewinnt)

### 6.2 `:where()` — Null-Spezifitaets-Wrapper

Jeder Selektor innerhalb von `:where()` hat Spezifitaet (0,0,0):

```css
/* Design-System-Defaults — leicht ueberschreibbar */
:where(.btn) { background: blue; }

/* Jede einfache Klasse ueberschreibt das */
.btn { background: green; }  /* gewinnt */
```

Ideal fuer: Bibliotheken, Default-Styles, die leicht anpassbar sein sollen.

### 6.3 `:is()` — Spezifitaet des spezifischsten Arguments

```css
:is(.card, #sidebar) .title { }
/* Spezifitaet = (1,1,0) — nimmt die hoechste Spezifitaet aller Argumente */
```

### 6.4 `@scope` — DOM-Teilbaum-Begrenzung

Seit Dezember 2025 Baseline in allen Browsern:

```css
@scope (.card) {
    .title { font-weight: bold; }  /* Gilt nur innerhalb von .card */
}

/* Donut-Scope — verhindert Durchsickern in verschachtelte Komponenten */
@scope (.card) to (.nested-widget) {
    .title { color: red; }
    /* Gilt in .card, aber NICHT innerhalb von .nested-widget */
}
```

**Scoping Proximity:** Bei verschachtelten Scopes gewinnt der naechstgelegene — loest Theme-Konflikte elegant.

### 6.5 Harry Roberts' Specificity Graph

Visualisierung der Spezifitaet ueber die Stylesheet-Laenge. Sollte stets **sanft nach oben steigen**:

```
Spezifitaet
    ^
    |         ________________  ← Utilities
    |     ___/                  ← Components
    |  __/                      ← Objects
    | /                         ← Elements
    |/                          ← Reset
    +------------------------→  Stylesheet-Position
```

ITCSS erzwingt dies durch Konvention, `@layer` durch native Sprachfeatures.

---

## 7. CSS Custom Properties (Variablen)

### 7.1 Grundlagen

```css
:root {
    --color-primary: #2563eb;
}

.button {
    background: var(--color-primary);
    /* Fallback-Wert */
    color: var(--color-text, #333);
}
```

### 7.2 Scope und Vererbung

Custom Properties vererben sich im DOM-Baum:

```css
:root { --gap: 16px; }           /* Global */
.sidebar { --gap: 8px; }         /* Lokaler Override */
.sidebar .item { gap: var(--gap); }  /* Verwendet 8px */
```

### 7.3 Dynamische Berechnung

```css
:root {
    --base-size: 16;
    --scale: 1.25;
}

h1 { font-size: calc(var(--base-size) * var(--scale) * var(--scale) * 1px); }
```

### 7.4 Animierbare Custom Properties mit @property

Standardmaessig sind Custom Properties nicht animierbar. Mit `@property` aendern:

```css
@property --gradient-angle {
    syntax: '<angle>';
    initial-value: 0deg;
    inherits: false;
}

.element {
    --gradient-angle: 0deg;
    background: linear-gradient(var(--gradient-angle), red, blue);
    transition: --gradient-angle 0.5s;
}
.element:hover {
    --gradient-angle: 180deg;
}
```

### 7.5 Best Practices

- Definiere Tokens in `:root`, komponentenspezifische Properties im Komponenten-Selektor
- Verwende `var()` mit Fallback fuer optionale Properties
- Keine komplexen Berechnungen in Token-Definitionen — halte Tokens als reine Werte
- Custom Properties sind case-sensitive: `--Color` und `--color` sind unterschiedlich

---

## 8. Modernes CSS: Die wichtigsten Features

### 8.1 Container Queries (`@container`)

Komponenten reagieren auf die Groesse ihres Containers statt des Viewports:

```css
.card-container {
    container-type: inline-size;
    container-name: card;
}

@container card (min-width: 400px) {
    .card { flex-direction: row; }
}

@container card (max-width: 399px) {
    .card { flex-direction: column; }
}
```

**Container Query Units:** `cqw`, `cqi`, `cqb` — relativ zum Container statt zum Viewport.

**Wann einsetzen:** Wiederverwendbare Komponenten die in unterschiedlich breiten Containern leben (Cards, Widgets, Sidebar-Inhalte).

**Wann NICHT einsetzen:** Seiten-Layout (dafuer weiterhin Media Queries).

### 8.2 `:has()` — Der Parent-Selector

Das meistgenutzte und meistgeliebte CSS-Feature laut State of CSS 2025:

```css
/* Cards MIT Bild anders stylen */
.card:has(img) { grid-template-rows: auto 1fr; }

/* Form-Gruppe mit invalidem Input markieren */
.form-group:has(:invalid) { border-color: red; }

/* Grid-Layout ab 6 Kindern aendern */
.grid:has(:nth-child(6)) { grid-template-columns: repeat(3, 1fr); }

/* Geschwister-Styling */
h2:has(+ p) { margin-bottom: 0.5em; }
```

**Performance-Warnung:** Breite Selektoren wie `*:has(.something)` vermeiden.

### 8.3 View Transitions

Filmreife Uebergaenge zwischen DOM-Zustaenden:

```css
/* Multi-Page-Apps — eine Zeile genuegt */
@view-transition { navigation: auto; }

/* Benannte Element-Transitions */
.card { view-transition-name: card-hero; }

::view-transition-old(card-hero) { animation: fade-out 0.3s; }
::view-transition-new(card-hero) { animation: fade-in 0.3s; }
```

### 8.4 Anchor Positioning

Eliminiert JavaScript-basierte Positionsberechnungen fuer Tooltips, Popovers, Dropdowns:

```css
.tooltip {
    position: fixed;
    position-anchor: --trigger;
    top: anchor(bottom);
    left: anchor(center);
    position-try-fallbacks: flip-block;  /* Auto-Repositionierung bei Overflow */
}
```

**Browser-Support:** Chrome/Edge 125+. Safari und Firefox in Entwicklung.

### 8.5 Scroll-Driven Animations

Animationen verknuepft mit Scroll-Position statt Zeit — ohne JavaScript, GPU-beschleunigt:

```css
.progress-bar {
    animation: grow linear;
    animation-timeline: scroll();  /* Trackt Scroll-Position */
}

.reveal {
    animation: fade-in linear;
    animation-timeline: view();    /* Trackt Sichtbarkeit im Viewport */
    animation-range: entry 0% entry 100%;
}
```

---

## 9. Responsive Design: Fluid & Intrinsic

### 9.1 clamp() — Die Schluessseltechnik

`clamp(minimum, preferred, maximum)` interpoliert linear zwischen Viewport-Groessen:

```css
/* Fluid Typography */
font-size: clamp(1rem, 0.5rem + 2vw, 2rem);

/* Fluid Spacing */
padding: clamp(16px, 1rem + 1vw, 32px);

/* Fluid Widths */
width: clamp(200px, 50%, 800px);
```

**Accessibility-Regel:** Den preferred Value als Kombination aus `rem` und `vw` formulieren. Reines `vw` bricht die Browser-Zoom-Funktion (WCAG-Verstoss).

```css
/* Schlecht — Zoom funktioniert nicht */
font-size: clamp(1rem, 4vw, 2rem);

/* Gut — Zoom-kompatibel */
font-size: clamp(1rem, 0.5rem + 2vw, 2rem);
```

### 9.2 Fluid Token-System

Eliminiert den Bedarf an Breakpoint-spezifischen Ueberschreibungen:

```css
:root {
    --font-size-title: clamp(22px, 1.25rem + 0.75vw, 28px);
    --space-lg:        clamp(16px, 1rem + 1vw, 24px);
    --space-xl:        clamp(24px, 1.25rem + 1vw, 32px);
}
```

**Regel:** Nur grosse Werte fluid machen. Spacing unter 16px und Schriftgroessen unter 14px sollten fix bleiben — fluide Werte unter diesen Schwellen verursachen mehr Probleme als sie loesen (Touch-Targets werden unzuverlaessig, Text wird unleserlich).

### 9.3 Intrinsic Design

Statt Breakpoints zu definieren, nutzen Layouts Content-Sizing:

```css
/* minmax() — passt sich natuerlich an */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr));
    gap: var(--space-md);
}

/* fit-content() — maximal so breit wie noetig */
.sidebar { width: fit-content(300px); }
```

### 9.4 Der Konsens 2025

| Technik | Anwendung |
|---------|-----------|
| Media Queries | Seiten-Layout (Sidebar ein/aus, Navigation) |
| Container Queries | Komponenten-Layout (Card horizontal/vertikal) |
| `clamp()` | Sizing und Typografie (fluide Skalierung) |
| `minmax()` / `auto-fit` | Grid-Layouts (intrinsische Anpassung) |

### 9.5 Layout-Primitiven (Every Layout)

Zwölf komposierbare, medienquery-freie Building Blocks:

| Primitiv | Zweck |
|----------|-------|
| **Stack** | Vertikaler Flow mit konsistentem Gap |
| **Box** | Padding-Container |
| **Center** | Horizontale Zentrierung mit max-width |
| **Cluster** | Horizontale Gruppe mit Wrap |
| **Sidebar** | Zwei-Spalten mit Sidebar-Mindestbreite |
| **Switcher** | Automatischer Wechsel horizontal/vertikal |
| **Cover** | Vertikale Zentrierung (Hero-Bereiche) |
| **Grid** | Auto-responsive Grid |
| **Frame** | Aspect-Ratio-Container |
| **Reel** | Horizontal-Scroll-Container |

Der **Switcher** (die "Holy Albatross"-Technik):

```css
.switcher > * {
    flex-grow: 1;
    flex-basis: calc((30rem - 100%) * 999);
    /* Wechselt bei 30rem Container-Breite */
}
```

### 9.6 Viewport-Units fuer Mobile

Das klassische `100vh`-Problem auf Mobile (Adressleiste wird nicht beruecksichtigt):

| Unit | Bedeutung | Empfehlung |
|------|-----------|------------|
| `svh` | Small Viewport Height (Adressleiste sichtbar) | Default fuer die meisten Layouts |
| `dvh` | Dynamic Viewport Height (aendert sich) | Vermeiden wegen Reflow |
| `lvh` | Large Viewport Height (Adressleiste versteckt) | Fuer Fullscreen-Overlays |

```css
.hero { min-height: 100svh; }
```

---

## 10. Performance-Optimierung

### 10.1 Critical CSS

Nur Above-the-Fold-Styles inline im `<head>`. Ziel: unter 14 KB komprimiert (TCP Congestion Window).

```html
<head>
    <style>/* Critical CSS inline */</style>
    <link rel="preload" href="style.css" as="style" onload="this.rel='stylesheet'">
</head>
```

Tools: **Critical** (Addy Osmani), **Penthouse**.

### 10.2 content-visibility: auto

Ueberspringt Rendering von Off-Screen-Elementen:

```css
.section {
    content-visibility: auto;
    contain-intrinsic-size: auto 500px;  /* Geschaetzte Hoehe verhindert Scrollbar-Jitter */
}
```

Messbarer Gewinn: bis zu 7x schnelleres initiales Rendering in Demos.

**Vorsicht:** Safari's Cmd+F findet moeglicherweise keinen Text in versteckten Elementen.

### 10.3 will-change

- **Vor** der Animation setzen, nicht waehrend
- Per JavaScript setzen und danach wieder entfernen
- Jedes Element mit `will-change` kann eine separate Compositing-Layer erzeugen (Speicherverbrauch!)
- Nur als letztes Mittel bei nachgewiesenen Performance-Problemen

```css
/* Schlecht — permanent */
.card { will-change: transform; }

/* Gut — nur bei Bedarf */
.card:hover { will-change: transform; }
```

### 10.4 Layout Thrashing vermeiden

Entsteht durch verschraenktes Lesen und Schreiben von Layout-Properties in JavaScript:

```javascript
// Schlecht — erzwingt synchrone Reflows
elements.forEach(el => {
    const width = el.offsetWidth;      // Read → Reflow
    el.style.width = width + 10 + 'px'; // Write → Invalidiert Layout
});

// Gut — Reads buendeln, dann Writes
const widths = elements.map(el => el.offsetWidth);  // Alle Reads
elements.forEach((el, i) => {
    el.style.width = widths[i] + 10 + 'px';  // Alle Writes
});
```

### 10.5 CSS-Dateigröße reduzieren

- **PurgeCSS** entfernt ungenutzte Selektoren (96% Reduktion bei Tailwind-Projekten)
- **cssnano** optimiert als PostCSS-Plugin (Shorthand-Normalisierung, Kommentar-Entfernung)
- Animiere nur `transform` und `opacity` — alles andere triggert Layout/Paint

### 10.6 Animationen: transform/opacity bevorzugen

| Property | Triggert | Performance |
|----------|----------|-------------|
| `transform` | Composite only | Exzellent |
| `opacity` | Composite only | Exzellent |
| `background-color` | Paint + Composite | Gut |
| `width`, `height` | Layout + Paint + Composite | Schlecht |
| `top`, `left` | Layout + Paint + Composite | Schlecht |

---

## 11. Code-Qualitaet & Linting

### 11.1 Stylelint

Das zentrale Werkzeug fuer CSS-Qualitaet. Version 17 (2026) ist vollstaendig ESM-basiert.

```javascript
// stylelint.config.mjs
export default {
    extends: [
        "stylelint-config-standard",
        "stylelint-config-recess-order"   // Property-Sortierung
    ],
    plugins: [
        "stylelint-plugin-defensive-css"
    ],
    rules: {
        "declaration-no-important": true,
        "selector-max-id": 0,
        "max-nesting-depth": 3,
        "selector-class-pattern": "^[a-z][a-z0-9]*(-[a-z0-9]+)*$"
    }
};
```

### 11.2 Wertvolle Plugins

| Plugin | Zweck |
|--------|-------|
| `stylelint-declaration-strict-value` | Erzwingt Tokens fuer Farben, Font-Sizes |
| `stylelint-plugin-use-baseline` | Warnt vor Features ohne Browser-Support |
| `stylelint-plugin-logical-css` | Foerdert logische Properties |
| `stylelint-plugin-defensive-css` | Verhindert haeufige CSS-Fehler |

### 11.3 Die komplette Pipeline

**Lokal (Pre-Commit):**
```bash
# Husky + lint-staged — nur gestagte Dateien (1-2 Sekunden)
npx lint-staged
# → stylelint --fix
# → prettier --write
```

**CI/CD:**
```yaml
# GitHub Actions — alle CSS-Dateien bei jedem PR
- run: npx stylelint "**/*.css"
```

**Build:**
```
PostCSS → Autoprefixer → cssnano → Production Bundle
```

### 11.4 Prettier + Stylelint

Prettier uebernimmt Formatierung (Einrueckung, Zeilenumbrueche).
Stylelint uebernimmt Konventionen und Fehler.
`stylelint-config-prettier` verhindert Regel-Konflikte.

---

## 12. CSS Nesting: Regeln & Fallstricke

### 12.1 Natives CSS Nesting vs. Sass

Natives CSS Nesting ist seit 2023 Baseline in allen Browsern. Aber es gibt **kritische Unterschiede zu Sass:**

#### Was funktioniert (valides natives CSS Nesting):

```css
.card {
    /* Pseudo-Klassen */
    &:hover { transform: translateY(-2px); }
    &:focus-visible { outline: 2px solid blue; }

    /* Compound-Selektoren (Klasse am gleichen Element) */
    &.active { background: blue; }
    &.is-loading { opacity: 0.5; }

    /* Pseudo-Elemente */
    &::before { content: ''; }
    &::after { content: ''; }

    /* Descendant-Selektoren */
    & .title { font-weight: bold; }
    & > .icon { margin-right: 8px; }
    & + .sibling { margin-top: 16px; }

    /* Media Queries (inline) */
    @media (max-width: 768px) {
        padding: 8px;  /* Gilt fuer .card bei diesem Breakpoint */
    }
}
```

#### Was NICHT funktioniert (Sass-only, bricht in nativem CSS):

```css
.ios-sidebar {
    /* KAPUTT — &-suffix ist kein valider CSS-Selektor! */
    &-brand { }       /* Soll .ios-sidebar-brand werden — wird ignoriert */
    &-link { }        /* Soll .ios-sidebar-link werden — wird ignoriert */
    &--featured { }   /* Soll .ios-sidebar--featured werden — wird ignoriert */
}
```

**Grund:** In Sass ist `&` ein Text-Replacement. In nativem CSS ist `&` eine Selector-Referenz — kein String. `&-brand` ist syntaktisch ungueltig und wird vom Browser stillschweigend ignoriert.

### 12.2 BEM-Elemente korrekt nesten

```css
/* FALSCH — broken in nativem CSS */
.card {
    &__title { font-size: 1.5rem; }
    &__body { padding: 1rem; }
    &--featured { border: 2px solid gold; }
}

/* RICHTIG — flache BEM-Selektoren, States genested */
.card { border-radius: 8px; }

.card__title {
    font-size: 1.5rem;
    &:first-child { margin-top: 0; }  /* Pseudo-Klasse — funktioniert */
}

.card__body { padding: 1rem; }

.card--featured {
    border: 2px solid gold;
    & .card__title { color: gold; }  /* Descendant — funktioniert */
}
```

### 12.3 Verschachtelungstiefe begrenzen

Andy Bell's Faustregel: **Maximal zwei Ebenen.**

```css
/* Gut — flach und lesbar */
.nav {
    & .nav-item {
        &:hover { color: blue; }
    }
}

/* Schlecht — zu tief, schwer zu debuggen */
.nav {
    & .nav-list {
        & .nav-item {
            & .nav-link {
                &:hover { color: blue; }
            }
        }
    }
}
```

### 12.4 Spezifitaets-Hinweis

`&` wird intern zu `:is()` gewrappt. Das kann die Spezifitaet beeinflussen:

```css
.card {
    & .title { }
}
/* Wird zu: :is(.card) .title — Spezifitaet (0,2,0) statt (0,1,1) */
/* In der Praxis selten ein Problem, aber gut zu wissen */
```

---

## 13. Farb-Management

### 13.1 oklch() — Der neue Standard

OKLCH ist perzeptuell uniform — eine Lightness-Aenderung sieht fuer alle Farbtoene gleich aus:

```css
/* oklch(Lightness, Chroma, Hue) */
--color-primary: oklch(0.55 0.2 250);

/* Dunklere Variante — einfach L reduzieren */
--color-primary-dark: oklch(0.40 0.2 250);

/* Entsaettigte Variante — C reduzieren */
--color-primary-muted: oklch(0.55 0.05 250);
```

### 13.2 color-mix() — Farben dynamisch ableiten

Keine manuellen Hex-Berechnungen mehr:

```css
/* Hover-State: 15% dunkler */
--btn-hover: color-mix(in oklch, var(--color-primary), black 15%);

/* Tint: 12% Opacity-Overlay */
--tint-primary: color-mix(in srgb, var(--color-primary) 12%, transparent);

/* Zwei Farben mischen */
--color-blend: color-mix(in oklch, var(--color-a), var(--color-b));
```

**Vorteil fuer Theming:** Tints und Hover-States leiten sich automatisch von den Basis-Farben ab. Bei Dark-Mode-Wechsel aendern sich die abgeleiteten Farben automatisch mit.

### 13.3 Farbsysteme aufbauen

```css
:root {
    /* Basis-Palette (Primitive Tokens) */
    --hue-primary: 250;
    --hue-success: 150;
    --hue-danger: 25;

    /* Semantische Farben — konsistente L/C, variabler Hue */
    --color-primary: oklch(0.55 0.2 var(--hue-primary));
    --color-success: oklch(0.55 0.2 var(--hue-success));
    --color-danger:  oklch(0.55 0.2 var(--hue-danger));

    /* Abgeleitete Varianten */
    --color-primary-hover: color-mix(in oklch, var(--color-primary), black 15%);
    --color-primary-tint:  color-mix(in srgb, var(--color-primary) 12%, transparent);
}
```

### 13.4 Kontrast sicherstellen

- WCAG AA: Mindestens 4.5:1 fuer normalen Text, 3:1 fuer grossen Text
- WCAG AAA: Mindestens 7:1 fuer normalen Text
- Teste mit DevTools: Chrome → Elements → Computed → Contrast Ratio
- `color-contrast()` (CSS-Funktion, noch Draft) wird automatische Kontrastberechnung ermoeglichen

---

## 14. Typografie-System

### 14.1 Fluid Type Scale

```css
:root {
    /* Fluid: interpoliert zwischen 375px und 1200px Viewport */
    --font-size-title1:   clamp(22px, 1.25rem + 0.75vw, 28px);
    --font-size-title2:   clamp(18px, 1rem + 0.5vw, 22px);
    --font-size-title3:   clamp(17px, 0.95rem + 0.35vw, 20px);
    --font-size-headline: clamp(15px, 0.85rem + 0.25vw, 17px);

    /* Fix: Kleine Groessen bleiben fix (Lesbarkeit) */
    --font-size-body:     16px;
    --font-size-subhead:  15px;
    --font-size-footnote: 13px;
    --font-size-caption:  12px;
}
```

Tools wie **Utopia** (utopia.fyi) generieren komplette Fluid-Type-Scales als Custom-Property-Sets.

### 14.2 Font Stack Best Practices

```css
/* System Font Stack — kein externer Font noetig */
--font-sans: system-ui, -apple-system, BlinkMacSystemFont,
             'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;

/* Monospace Stack */
--font-mono: 'SF Mono', 'SFMono-Regular', Consolas,
             'Liberation Mono', Menlo, monospace;

/* Mit externem Font (Inter) */
--font-sans: 'Inter', system-ui, -apple-system, sans-serif;
```

### 14.3 Typografie-Verbesserungen

```css
/* Ausgewogene Zeilenlaengen fuer Headings */
h1, h2, h3, h4, h5, h6 {
    text-wrap: balance;
}

/* Schoener Textumbruch — vermeidet Waisen-Woerter */
body {
    text-wrap: pretty;
}

/* Optimale Lesbarkeitsbreite */
.prose {
    max-inline-size: 65ch;  /* 65 Zeichen pro Zeile */
}
```

### 14.4 Font Loading

```html
<!-- Preconnect fuer externe Fonts -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<!-- Font mit display=swap laden (verhindert FOIT) -->
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
```

```css
/* Optional: Font-Display in @font-face */
@font-face {
    font-family: 'Inter';
    src: url('inter.woff2') format('woff2');
    font-display: swap;      /* Text sofort sichtbar, Font tauscht spaeter */
    font-weight: 100 900;    /* Variable Font */
}
```

---

## 15. Spacing-System

### 15.1 8px Base Grid

Das am weitesten verbreitete Spacing-System:

```css
:root {
    --space-xs:  4px;    /* 0.5x */
    --space-sm:  8px;    /* 1x   */
    --space-md:  16px;   /* 2x   */
    --space-lg:  24px;   /* 3x   */
    --space-xl:  32px;   /* 4x   */
    --space-xxl: 48px;   /* 6x   */
}
```

### 15.2 Fluid Spacing

Nur fuer grosse Werte — kleine Werte bleiben fix:

```css
:root {
    --space-xs:  4px;                              /* Fix */
    --space-sm:  8px;                              /* Fix */
    --space-md:  16px;                             /* Fix */
    --space-lg:  clamp(16px, 1rem + 1vw, 24px);   /* Fluid */
    --space-xl:  clamp(24px, 1.25rem + 1vw, 32px); /* Fluid */
    --space-xxl: clamp(32px, 1.5rem + 2vw, 48px);  /* Fluid */
}
```

### 15.3 Logische Properties

Verwende logische Properties statt physischer Richtungen — sauberer und RTL-kompatibel:

| Physisch | Logisch |
|----------|---------|
| `margin-left` | `margin-inline-start` |
| `margin-right` | `margin-inline-end` |
| `margin-top` | `margin-block-start` |
| `margin-bottom` | `margin-block-end` |
| `padding-left/right` | `padding-inline` |
| `padding-top/bottom` | `padding-block` |
| `width` | `inline-size` |
| `height` | `block-size` |
| `text-align: left` | `text-align: start` |
| `border-left` | `border-inline-start` |

### 15.4 Gap statt Margin

Bevorzuge `gap` in Flex/Grid-Containern ueber Margins an Kindern:

```css
/* Schlecht — Spacing-Logik am falschen Element */
.stack > * + * { margin-top: 16px; }

/* Gut — Spacing am Container */
.stack {
    display: flex;
    flex-direction: column;
    gap: var(--space-md);
}
```

---

## 16. Animation & Transitions

### 16.1 Transition Best Practices

```css
/* Spezifische Properties auflisten — nie "all" */
.button {
    transition: background-color 0.15s ease, transform 0.15s ease;
    /* NICHT: transition: all 0.15s ease; (triggert Layout auf alles) */
}
```

### 16.2 Reduced Motion respektieren

```css
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
```

### 16.3 interpolate-size: allow-keywords

Ermoeglicht smooth Transitions auf `height: auto`:

```css
html {
    interpolate-size: allow-keywords;
}

.accordion-content {
    height: 0;
    overflow: hidden;
    transition: height 0.3s ease;
}

.accordion-content.open {
    height: auto;  /* Smooth animiert! */
}
```

Kein JavaScript-Autosize mehr noetig fuer Akkordeons, Dropdowns, expandierbare Bereiche.

### 16.4 Timing Functions

| Funktion | Verwendung |
|----------|-----------|
| `ease` | Standard, gut fuer die meisten Faelle |
| `ease-out` | Elemente die erscheinen (schneller Start, langsames Ende) |
| `ease-in` | Elemente die verschwinden (langsamer Start) |
| `ease-in-out` | Elemente die Position wechseln |
| `cubic-bezier(0.32, 0.72, 0, 1)` | iOS-aehnliches Spring-Feeling |

### 16.5 @keyframes fuer komplexe Animationen

```css
@keyframes slide-in {
    from { transform: translateY(-8px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
}

.panel {
    animation: slide-in 0.3s ease;
}
```

---

## 17. Accessibility (a11y)

### 17.1 Screen-Reader-Only Content

```css
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
}
```

### 17.2 Focus Styles

```css
/* Nie focus komplett entfernen — nur visuelles Outline anpassen */
:focus-visible {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
}

/* focus (ohne -visible) nur fuer Keyboard-Navigation */
:focus:not(:focus-visible) {
    outline: none;
}
```

### 17.3 Touch Targets

Minimum 44x44px fuer alle interaktiven Elemente auf Mobile:

```css
@media (max-width: 768px) {
    .btn-sm { min-height: 44px; }
    .form-control-sm { min-height: 44px; }
    .form-check-input { width: 22px; height: 22px; }
}
```

### 17.4 Farb-Kontrast

- Teste alle Text-Farbe/Hintergrund-Kombinationen gegen WCAG 2.1 AA (4.5:1)
- Verwende nie Farbe als einzigen Indikator (Icons, Unterstreichungen, Labels ergaenzen)
- Dark Mode separat testen — gleiche Farbe kann verschiedene Kontrastwerte haben

### 17.5 Prefers-Contrast

```css
@media (prefers-contrast: more) {
    :root {
        --color-text: #000;
        --color-surface: #fff;
        --border-width: 2px;
    }
}
```

---

## 18. Dark Mode & Theming

### 18.1 Implementierungs-Strategien

**Strategie A: CSS Custom Properties + Daten-Attribut (empfohlen)**

```css
:root {
    --color-bg: #ffffff;
    --color-text: #111827;
}

[data-theme="dark"] {
    --color-bg: #1a1a2e;
    --color-text: #e0e0e8;
}
```

```javascript
// Toggle
document.documentElement.setAttribute('data-theme',
    currentTheme === 'dark' ? 'light' : 'dark'
);
```

**Strategie B: prefers-color-scheme (System-Praeferenz)**

```css
@media (prefers-color-scheme: dark) {
    :root { --color-bg: #1a1a2e; }
}
```

**Strategie C: Kombination (Best Practice)**

```javascript
// 1. Gespeicherte Praeferenz pruefen
// 2. Falls nicht vorhanden: System-Praeferenz verwenden
const theme = localStorage.getItem('theme')
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
document.documentElement.setAttribute('data-theme', theme);
```

### 18.2 Anti-Flash: Theme vor Paint anwenden

```html
<head>
    <script>
        (function(){
            var t = localStorage.getItem('theme');
            if (!t) t = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', t);
        })();
    </script>
    <!-- CSS danach laden -->
</head>
```

### 18.3 Dark Mode Design-Regeln

1. **Nie einfach Farben invertieren** — Dark Mode braucht eigene, sorgfaeltig gewahlte Farben
2. **Oberflaechenfarben:** Leicht unterschiedliche Grautone fuer Elevation/Depth
3. **Schatten:** Dunkler und subtiler als im Light Mode
4. **Textfarben:** Nicht reines Weiss (#fff) — zu grell. Besser: #e0e0e8 oder aehnlich
5. **Akzentfarben:** Hoehere Lightness im Dark Mode fuer ausreichend Kontrast
6. **Tints/Overlays:** Hoehere Opacity im Dark Mode (12% → 15% oder mehr)

### 18.4 Dark Mode mit color-mix() automatisieren

```css
:root {
    --color-primary: #2563eb;
    --tint-opacity: 12%;
}

[data-theme="dark"] {
    --color-primary: #818cf8;
    --tint-opacity: 15%;
}

/* Tint leitet sich automatisch ab */
--color-primary-tint: color-mix(in srgb, var(--color-primary) var(--tint-opacity), transparent);
```

---

## 19. Browser-Kompatibilitaet

### 19.1 Baseline-Features (sicher verwendbar)

Diese Features sind in allen modernen Browsern verfuegbar:

| Feature | Baseline seit | Nutzung |
|---------|--------------|---------|
| CSS Custom Properties | 2017 | Ueberall |
| CSS Grid | 2017 | Ueberall |
| `clamp()` | 2020 | Ueberall |
| `:is()`, `:where()` | 2021 | Ueberall |
| `aspect-ratio` | 2021 | Ueberall |
| Container Queries | 2023 | Ueberall |
| CSS Nesting | 2023 | Ueberall |
| `:has()` | 2023 | Ueberall |
| `@layer` | 2022 | Ueberall |
| `color-mix()` | 2023 | Ueberall |
| `@scope` | 2025 | Ueberall |
| View Transitions | 2025 | Ueberall |

### 19.2 Eingeschraenkt verfuegbar

| Feature | Status | Fallback-Strategie |
|---------|--------|-------------------|
| Anchor Positioning | Chrome/Edge only | JavaScript-Positionierung |
| `@starting-style` | Chrome/Edge/Safari | Keine Animation beim Erscheinen |
| Scroll-Driven Animations | Chrome/Edge | JavaScript-basiertes Scrolling |
| `field-sizing: content` | Chrome only | JavaScript-Autosize |

### 19.3 Progressive Enhancement

```css
/* Feature Query — sicher neues CSS nutzen */
@supports (container-type: inline-size) {
    .widget { container-type: inline-size; }
}

/* Fallback fuer aeltere Browser */
.card { display: flex; }
@supports (display: grid) {
    .card { display: grid; }
}
```

### 19.4 Stylelint Plugin fuer Baseline-Check

```javascript
plugins: ["stylelint-plugin-use-baseline"],
rules: {
    "plugin/use-baseline": [true, { "browser-support": "widely-available" }]
}
```

---

## 20. Anti-Patterns & haeufige Fehler

### 20.1 `!important` als Spezifitaets-Waffe

```css
/* Schlecht */
.button { color: blue !important; }

/* Gut — Spezifitaet durch Architektur loesen */
@layer framework, components;
/* oder: @scope, :where(), bessere Selektoren */
```

`!important` ist akzeptabel fuer:
- Utility-Klassen (`.hidden { display: none !important; }`)
- Accessibility-Overrides
- Third-Party-CSS-Overrides (als temporaerer Fix)

### 20.2 Deep Nesting

```css
/* Schlecht — Specificity Nightmare */
.page .content .sidebar .nav .item .link:hover { }

/* Gut — flache Selektoren */
.nav-link:hover { }
```

### 20.3 Magic Numbers

```css
/* Schlecht */
.header { height: 73px; padding-top: 17px; }

/* Gut — benannte Werte */
.header { height: var(--header-height); padding-top: var(--space-md); }
```

### 20.4 Harte Pixel-Breakpoints ueberall

```css
/* Schlecht — starre Breakpoints fuer jede Groesse */
@media (max-width: 768px) { .title { font-size: 18px; } }
@media (max-width: 480px) { .title { font-size: 16px; } }

/* Gut — ein fluid Token */
.title { font-size: clamp(16px, 1rem + 0.5vw, 22px); }
```

### 20.5 `transition: all`

```css
/* Schlecht — triggert Transitions auf JEDE Property-Aenderung */
.card { transition: all 0.3s ease; }

/* Gut — nur noetige Properties */
.card { transition: transform 0.3s ease, box-shadow 0.3s ease; }
```

### 20.6 Inline Styles fuer alles

Inline Styles haben Spezifitaet (1,0,0,0) und sind nur durch `!important` ueberschreibbar. Akzeptabel fuer:
- Dynamische Werte aus JavaScript (z.B. berechnete Positionen)
- `display: none` als initialer Zustand (vor JS-Initialisierung)

### 20.7 BEM-Suffix-Nesting in nativem CSS

```css
/* KAPUTT — haeufigster Fehler beim Umstieg von Sass */
.card {
    &__title { }    /* Wird NICHT zu .card__title */
    &--featured { }  /* Wird NICHT zu .card--featured */
}
```

Siehe Kapitel 12 fuer die korrekte Verwendung.

### 20.8 ID-Selektoren fuer Styling

```css
/* Schlecht — Spezifitaet (1,0,0) ist schwer ueberschreibbar */
#header { background: blue; }

/* Gut — Klassen verwenden */
.header { background: blue; }
```

IDs gehoeren ins HTML fuer: Anchor-Links, JavaScript-Hooks, ARIA-Referenzen. Nicht fuer CSS-Styling.

---

## 21. Checkliste: Neues Projekt aufsetzen

### Phase 1: Fundament

- [ ] **Architektur waehlen** (BEM + ITCSS empfohlen fuer die meisten Projekte)
- [ ] **Design Tokens definieren** — Farben, Typografie, Spacing, Shadows, Radii, Transitions
- [ ] **Reset/Normalize** einbinden (Josh Comeau's Modern CSS Reset oder Andy Bell's Reset)
- [ ] **`@layer`-Reihenfolge** deklarieren
- [ ] **Dateistruktur** anlegen (7-1, ITCSS, oder modular mit @import)

### Phase 2: Token-System

- [ ] **Farb-Tokens** — Primitive + Semantische Schicht + Dark Mode Varianten
- [ ] **Typografie-Tokens** — Fluid Type Scale mit `clamp()`
- [ ] **Spacing-Tokens** — 8px Grid, grosse Werte fluid
- [ ] **Shadow-Tokens** — Elevation Levels (sm, md, lg)
- [ ] **Transition-Tokens** — Fast (0.15s) und Normal (0.3s)
- [ ] **Z-Index-Tokens** — Definierte Skala (dropdown, modal, toast)

### Phase 3: Globale Styles

- [ ] **Reset** — `box-sizing: border-box`, Margin-Reset
- [ ] **html** — `interpolate-size: allow-keywords`, Font-Smoothing
- [ ] **body** — Font-Family, Farben, `text-wrap: pretty`
- [ ] **Headings** — `text-wrap: balance`
- [ ] **Links** — Farben, Hover-States
- [ ] **Focus Styles** — `:focus-visible` mit sichtbarem Outline

### Phase 4: Framework-Integration

- [ ] Third-Party CSS in `@layer(framework)` importieren
- [ ] **Preload-Hint** fuer Framework-CSS im HTML
- [ ] Bridge-Tokens definieren (eigene Tokens → Framework-Variablen)

### Phase 5: Qualitaet

- [ ] **Stylelint** konfigurieren
- [ ] **Prettier** fuer CSS-Formatierung
- [ ] **Husky + lint-staged** fuer Pre-Commit-Checks
- [ ] `selector-max-id: 0`, `declaration-no-important: true`

### Phase 6: Responsive

- [ ] Fluid Tokens verifizieren (375px bis 1200px testen)
- [ ] Media Queries nur fuer Layout-Shifts (nicht fuer Sizing)
- [ ] Touch Targets auf Mobile pruefen (min 44x44px)
- [ ] Safe Area Insets fuer Notch-Geraete (`env(safe-area-inset-bottom)`)
- [ ] `100svh` statt `100vh` fuer Mobile Full-Height

### Phase 7: Dark Mode

- [ ] Token-Overrides fuer `[data-theme="dark"]` definieren
- [ ] Anti-Flash Script im `<head>` (vor CSS)
- [ ] Kontrast-Checks fuer Dark Mode separat
- [ ] `color-scheme: dark` setzen
- [ ] `color-mix()` fuer automatische Tints/Hover-States

---

## 22. Glossar

| Begriff | Definition |
|---------|-----------|
| **Baseline** | Feature verfuegbar in allen grossen Browsern (Chrome, Firefox, Safari, Edge) |
| **BEM** | Block Element Modifier — Namenskonvention: `.block__element--modifier` |
| **Cascade** | CSS-Algorithmus zur Bestimmung welche Regel gewinnt bei Konflikten |
| **Compound Selector** | Mehrere Selektoren am gleichen Element: `.card.active` |
| **Custom Property** | CSS-Variable: `--name: value` / `var(--name)` |
| **Design Token** | Kleinste Einheit eines Design-Systems (Farbe, Groesse, Abstand) |
| **FOIT** | Flash of Invisible Text — Text unsichtbar waehrend Font laedt |
| **FOUT** | Flash of Unstyled Text — Fallback-Font sichtbar, dann Swap |
| **Intrinsic Design** | Layout passt sich natuerlich an Content und Container an |
| **ITCSS** | Inverted Triangle CSS — Architektur-Methodik von Harry Roberts |
| **Layer** | CSS Cascade Layer (`@layer`) — Prioritaetsstufe in der Kaskade |
| **Logical Properties** | Schreibmodus-relative Properties (`inline-start` statt `left`) |
| **Progressive Enhancement** | Basis-Funktionalitaet fuer alle, erweiterte Features fuer moderne Browser |
| **Reflow** | Browser berechnet Layout neu (teuer, vermeiden) |
| **Repaint** | Browser zeichnet Pixel neu (weniger teuer als Reflow) |
| **Scope** | Begrenzung von CSS auf einen DOM-Teilbaum (`@scope`) |
| **Specificity** | Gewichtung eines CSS-Selektors: (ID, Klasse, Element) |

---

*Dieses Playbook wird aktiv gepflegt. Letzte Aktualisierung: April 2026.*
