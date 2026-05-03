# Neomorphism Design System — Dossier

Light & Dark Mode | Referenzimplementierung: MintAccounting / Email Automation UI, April 2026

---

## 1. Grundprinzip

Neomorphism erzeugt Tiefe **ausschliesslich durch Schatten**. Keine Borders, keine harten Kanten. Jedes Element wirkt, als waere es aus einer einheitlichen Oberflaeche herausgedrueckt (*raised*) oder hineingedrueckt (*pressed*). Der Trick: **Element und Hintergrund teilen dieselbe Farbe.** Tiefe entsteht durch zwei gegenlaeufige Schatten — einen dunklen (Schattenseite) und einen hellen (Lichtseite).

```
                  Lichtquelle (oben-links)
                        \
         ┌──────────────────────────┐
  hell → │                          │ ← dunkel
         │      RAISED ELEMENT      │
  hell → │                          │ ← dunkel
         └──────────────────────────┘
                        \
                  Schattenwurf (unten-rechts)
```

### Die zwei Zustaende

| Zustand | Shadow-Typ | Verwendung |
|---------|-----------|------------|
| **Raised** | `outer` — Schatten nach aussen | Cards, Buttons (ruhend), Tags, Alerts |
| **Pressed** | `inset` — Schatten nach innen | Inputs, aktive Sidebar-Links, Toggle-Tracks, Buttons (gedrueckt) |

---

## 2. Einheitliche Oberflaeche — Das Kernprinzip

**Alles hat dieselbe Hintergrundfarbe.** Body, Cards, Inputs, Buttons, Sidebar — alle `#e0e5ec`. Tiefe entsteht NUR durch Schatten, nie durch Farbunterschiede zwischen Parent und Child.

```css
--nm-bg: #e0e5ec;

body       { background: var(--nm-bg); }
.card      { background: var(--nm-bg); }  /* oder subtiler Gradient, s. unten */
.btn       { background: var(--nm-bg); }
input      { background: var(--nm-bg); }
.sidebar   { background: var(--nm-bg); }
```

### Konsequenzen

- **Keine Borders** — durchgehend `border: none`
- **Trennlinien** innerhalb von Cards/Tables nicht durch `border-bottom`, sondern durch:
  ```css
  box-shadow: 0 1px 0 rgba(0,0,0,0.06),
              0 2px 0 rgba(255,255,255,0.4);
  ```
  Das ergibt eine feine Doppellinie (dunkel + hell), die wie eine eingefraeste Rille wirkt.
- **Subtle Gradient auf raised Containern** (optional):
  ```css
  background: linear-gradient(145deg, #dde2e9, #edf0f4);
  ```
  Der 145-Grad-Winkel korrespondiert mit der Lichtrichtung (oben-links) und gibt dem flachen Element eine dezente Richtung. Wird bei Cards, Tables und Accordion-Items eingesetzt. Buttons bekommen einen leicht blaeulichen Gradient.

---

## 3. Shadow-Architektur

### 3.1 Primitives

Alle Schatten werden aus **zwei Primitives** aufgebaut:

| Token | Wert | Rolle |
|-------|------|-------|
| `--nm-shadow-dark` | `#a3b1c6` | Schattenseite (unten-rechts) — blau-grau, nicht reines Grau |
| `--nm-shadow-light` | `#ffffff` | Lichtseite (oben-links) — reines Weiss |

### 3.2 Composed Shadows

Jeder Shadow besteht aus **drei Teilen**: dunkler Schatten + heller Schatten + subtiler Inset-Border fuer Anti-Aliasing:

```css
/* --- RAISED: Element steht hervor --- */

--nm-raised:      8px 8px 16px var(--nm-shadow-dark),     /* Schatten unten-rechts */
                 -8px -8px 16px var(--nm-shadow-light),    /* Licht oben-links */
                  inset 0 0 0 1px rgba(255,255,255,0.6);   /* Edge-Highlight */

--nm-raised-sm:   4px 4px 8px var(--nm-shadow-dark),
                 -4px -4px 8px var(--nm-shadow-light),
                  inset 0 0 0 0.5px rgba(255,255,255,0.5);

--nm-raised-lg:  12px 12px 24px var(--nm-shadow-dark),
                -12px -12px 24px var(--nm-shadow-light),
                  inset 0 0 0 1px rgba(255,255,255,0.6);

/* --- PRESSED: Element ist eingedrueckt --- */

--nm-pressed:     inset 6px 6px 14px var(--nm-shadow-dark),
                  inset -4px -4px 10px var(--nm-shadow-light);

--nm-pressed-sm:  inset 3px 3px 8px var(--nm-shadow-dark),
                  inset -2px -2px 5px var(--nm-shadow-light);
```

**Detail:** Die Pressed-Shadows sind bewusst **asymmetrisch** — die dunkle Seite ist staerker (6px/14px) als die helle (-4px/10px). Das verstaerkt den Eindruck, dass das Element wirklich *eingedrueckt* ist statt nur invertiert.

### 3.3 Groessen-Zuordnung

| Shadow | Offset | Blur | Einsatz |
|--------|--------|------|---------|
| `sm` | 4px | 8px | Buttons, Tags, kleine Elemente |
| (normal) | 8px | 16px | Cards, Accordion, Table-Container |
| `lg` | 12px | 24px | Modals, Overlays |

### 3.4 Responsive Reduktion (<=767px)

Auf Mobile werden Offsets halbiert:

```css
@media (max-width: 767.98px) {
    --nm-raised:     4px 4px 8px ...    /* statt 8px/16px */
    --nm-raised-sm:  2px 2px 5px ...    /* statt 4px/8px  */
    --nm-raised-lg:  6px 6px 12px ...   /* statt 12px/24px */
    --nm-pressed:    inset 4px 4px 10px ... /* statt 6px/14px */
    --nm-pressed-sm: inset 2px 2px 5px ...  /* statt 3px/8px */
}
```

---

## 4. Spacing — Die goldene Regel

### Minimum-Abstand = Shadow-Offset + Blur-Radius

Bei `--nm-raised` (8px Offset, 16px Blur) braucht jedes Element **mindestens 24px Abstand** zum Nachbarn, damit die Schatten nicht ineinander laufen. Unter dieser Schwelle wird es "vollgestopft".

### 8px Grid

| Token | Wert | Verwendung |
|-------|------|------------|
| `--space-xs` | 4px | Micro-Gaps, Label-Margins |
| `--space-sm` | 8px | Table-Cell-Padding, Badge-Padding |
| `--space-md` | 16px | Card-Header/Footer-Padding, Accordion-Padding, Standard-Gap |
| `--space-lg` | clamp(16px, 1rem + 1vw, 24px) | Card-Body-Padding, Sidebar-Padding |
| `--space-xl` | clamp(24px, 1.25rem + 1vw, 32px) | Content-Vertical-Padding |
| `--space-xxl` | clamp(32px, 1.5rem + 2vw, 48px) | Content-Horizontal-Padding, Section-Abstaende |

**Fluid Spacing:** Die drei groesseren Tokens (`lg`, `xl`, `xxl`) verwenden CSS `clamp()` und skalieren fliessend zwischen Mobile (375px) und Desktop (1200px) Viewports. Damit entfallen harte Breakpoint-Spruenge bei Abstaenden.

### Anwendung im Layout

```css
/* Hauptbereich: grosszuegig */
.nm-content {
    padding: 32px 48px;       /* oben/unten 32, seitlich 48 */
    max-width: 1400px;
}

/* Abstand zwischen Sektionen (Cards, Tabellen): ~48px */
/* Realisiert ueber Bootstrap mb-5 (3rem = 48px) */

/* Grid-Gap zwischen nebeneinander liegenden Cards: 24px */
/* Realisiert ueber Bootstrap g-4 (1.5rem = 24px) */

/* Titel zu erstem Element: 24px */
h1 { margin-bottom: 24px; }
```

### Border-Radius

| Token | Wert | Verwendung |
|-------|------|------------|
| `--radius-sm` | 12px | Buttons, Inputs, Sidebar-Links |
| `--radius-md` | 16px | Cards, Tables, Accordion |
| `--radius-lg` | 20px | Modals |
| `--radius-xl` | 20px | Grosse Overlays |
| `--radius-pill` | 100px | Badges, Tags, Toggle-Tracks |

Neomorphism braucht **groessere Radii** als Flat Design — Schatten wirken an scharfen Ecken unnatuerlich.

---

## 5. Farbsystem

### 5.1 Palette

```
SURFACE     #e0e5ec   Einheitliche Oberflaeche
TEXT        #2d3436   Primaertext (7.2:1 Kontrast)
TEXT-SEC    #636e72   Sekundaertext (4.6:1)
TEXT-MUTED  #8395a7   Labels, Captions

ACCENT      #a8b4d8   Pastel Blau — aktive States, Tints
ACC-HOVER   #8fa0c4   Dunkleres Blau — Links, Hover
ACTION      #b0a0d0   Pastel Purple — Link-Hover, Action-Badges, Sidebar-Gradient
ACC-CTA     #ff9f43   Orange — nur CTAs (max 10% Flaeche)

SUCCESS     #7ec8a0   Pastel Gruen
DANGER      #e17055   Pastel Rot
WARNING     #f6b93b   Pastel Gelb
INFO        #74b9ff   Pastel Blau (heller)
```

### 5.2 Kontrast-Regeln (WCAG)

| Kombination | Ratio | Level | Verwendung |
|------------|-------|-------|------------|
| `#2d3436` auf `#e0e5ec` | 7.2:1 | AAA | Alle Fliesstexte |
| `#636e72` auf `#e0e5ec` | 4.6:1 | AA | Sekundaertexte |
| `#8395a7` auf `#e0e5ec` | 3.1:1 | AA Large | Nur Labels/Captions (>=14px bold / >=18px) |

**Wichtig:** `#a8b4d8` (Accent) hat nur **1.5:1** auf `#e0e5ec` — **niemals als Textfarbe!** Nur fuer Hintergrund-Tints oder Icon-Akzente. Fuer klickbare Links: `#8fa0c4` (2.8:1, akzeptabel mit visuellen Cues).

### 5.3 Tints (halbtransparente Hintergruende)

Fuer Alerts, Hover-States und Status-Hintergruende: 12%-Opacity der Akzentfarben, automatisch abgeleitet via `color-mix()`:

```css
--nm-tint-accent:  color-mix(in srgb, var(--nm-accent) 12%, transparent);
--nm-tint-success: color-mix(in srgb, var(--nm-success) 12%, transparent);
--nm-tint-danger:  color-mix(in srgb, var(--nm-danger) 12%, transparent);
--nm-tint-warning: color-mix(in srgb, var(--nm-warning) 12%, transparent);
--nm-tint-info:    color-mix(in srgb, var(--nm-info) 12%, transparent);
```

**Vorteil von `color-mix()`:** Tints leiten sich automatisch von den Basisfarben ab. Wenn im Dark Mode die Basisfarben ueberschrieben werden, passen sich die Tints automatisch an — kein manuelles Duplizieren noetig. Im Dark Mode wird die Opazitaet auf 15% erhoeht (s. Sektion 13.5).

### 5.4 Button-Text: IMMER dunkel

Neumorphische Buttons haben Pastell-Hintergruende. Weisser Text wuerde unter 3:1 Kontrast fallen. Daher:

```css
.btn         { color: var(--nm-text); }   /* #2d3436 — immer dunkel */
.btn-primary { color: var(--nm-text); }   /* auch primary: dunkler Text */
.btn-success { color: var(--nm-text); }
.btn-danger  { color: var(--nm-text); }
```

---

## 6. Typografie

| Token | Groesse | Gewicht | Verwendung |
|-------|---------|---------|------------|
| `--font-size-title1` | clamp(22px, 1.25rem + 0.75vw, 28px) | 700 | Seitentitel (h1) |
| `--font-size-title2` | clamp(18px, 1rem + 0.5vw, 22px) | 700 | Sektions-Titel, grosse Metric-Values |
| `--font-size-title3` | clamp(17px, 0.95rem + 0.35vw, 20px) | 700 | h3 |
| `--font-size-headline` | clamp(15px, 0.85rem + 0.25vw, 17px) | 600 | h4, Sidebar-Brand, kleine Metric-Values |
| `--font-size-body` | 16px | 400 | Fliesstext |
| `--font-size-callout` | 16px | 400 | Hervorgehobene Absaetze |
| `--font-size-subhead` | 15px | 500 | Buttons, Labels, Card-Headers |
| `--font-size-footnote` | 13px | 400 | Sender-Tags, Settings-Descriptions, btn-sm |
| `--font-size-caption1` | 12px | 600 | Table-Headers, Metric-Labels (UPPERCASE) |
| `--font-size-caption2` | 11px | 500 | Tab-Labels, Badge-Text |

**Fluid Typography:** Die vier groessten Stufen (title1–headline) verwenden `clamp()` und skalieren fliessend zwischen 375px und 1200px Viewport-Breite. Kleinere Stufen bleiben fix — unterhalb 16px macht Fluid Sizing keinen Sinn.

**Font:** Inter (Google Fonts) -> System-Fallbacks -> sans-serif

```css
--nm-font: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display',
           'Helvetica Neue', Helvetica, Arial, sans-serif;
--font-monospace: 'SF Mono', 'SFMono-Regular', Consolas, 'Liberation Mono',
                  Menlo, monospace;
```

Einbindung:
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
```

**Letter-Spacing:** `-0.02em` fuer Headlines, `0.03-0.04em` fuer UPPERCASE Labels
**Line-Height:** 1.5 durchgehend

---

## 7. Komponenten-Rezepte

### 7.1 Card

```css
.card {
    background: linear-gradient(145deg, #dde2e9, #edf0f4);
    border: none;
    border-radius: 16px;
    box-shadow: var(--nm-raised);
    outline: 1px solid transparent;  /* Anti-Aliasing auf runden Ecken */
    overflow: hidden;
}

.card-header {
    background: transparent;
    box-shadow: 0 1px 0 rgba(0,0,0,0.06),
                0 2px 0 rgba(255,255,255,0.4);
    padding: 16px;
    font-weight: 600;
    font-size: 15px;
}

.card-body {
    padding: 24px;
}

.card-footer {
    background: transparent;
    box-shadow: inset 0 1px 0 rgba(0,0,0,0.06),
                inset 0 2px 0 rgba(255,255,255,0.4);
    padding: 8px 16px;
}
```

### 7.2 Button (drei Zustaende)

```css
.btn {
    background: var(--nm-bg);
    border: none;
    border-radius: 12px;
    box-shadow: var(--nm-raised-sm);
    color: #2d3436;
    padding: 8px 16px;
    font-weight: 500;
    transition: all 0.15s ease;
}

/* Hover: staerker raised + leichter Lift */
.btn:hover {
    box-shadow: var(--nm-raised);
    transform: translateY(-1px);
}

/* Active/Pressed: eingedrueckt + leichtes Einsinken */
.btn:active {
    box-shadow: var(--nm-pressed-sm);
    transform: translateY(1px);
}

/* Focus: pressed + Akzent-Ring */
.btn:focus-visible {
    box-shadow: var(--nm-pressed-sm), 0 0 0 3px rgba(168,180,216,0.3);
    outline: none;
}

/* Primary: subtiler Richtungs-Gradient */
.btn-primary {
    background: linear-gradient(145deg, #c8d0e8, #d8dde4);
}

/* Danger: Pastell-Rot-Gradient */
.btn-danger {
    background: linear-gradient(145deg, #e88a70, #d8c0b8);
}

/* Success: Pastell-Gruen-Gradient */
.btn-success {
    background: linear-gradient(145deg, #8ed4ae, #c8ddd0);
}
```

### 7.3 Input (eingedrueckt)

```css
input, select {
    background: var(--nm-bg);
    border: none;
    border-radius: 12px;
    box-shadow: var(--nm-pressed-sm);
    padding: 10px 12px;
    font-size: 15px;
    color: #2d3436;
}

input:focus {
    box-shadow: var(--nm-pressed-sm), 0 0 0 3px rgba(168,180,216,0.3);
    outline: none;
}

input::placeholder { color: #8395a7; }
```

### 7.4 Toggle Switch

```css
.toggle {
    width: 51px;
    height: 31px;
    border-radius: 16px;
    background: var(--nm-bg);
    box-shadow: var(--nm-pressed-sm);     /* Track ist "eingedrueckt" */
    position: relative;
}

.toggle::after {                          /* Knob */
    content: '';
    width: 25px;
    height: 25px;
    border-radius: 50%;
    position: absolute;
    top: 3px; left: 3px;
    background: linear-gradient(145deg, #f4f4f6, #dce1e8);
    box-shadow: 2px 2px 5px var(--nm-shadow-dark),
               -2px -2px 5px var(--nm-shadow-light);  /* Knob ist "raised" */
}

.toggle:checked {
    background: #7ec8a0;                  /* Track wird gruen */
}
.toggle:checked::after {
    transform: translateX(20px);
}
```

### 7.5 Table (in raised Container)

```css
.table-responsive {
    background: linear-gradient(145deg, #dde2e9, #edf0f4);
    border-radius: 16px;
    box-shadow: var(--nm-raised);
    overflow: hidden;
}

.table {
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
}

/* Zeilen-Trennlinie: Shadow statt Border */
.table td, .table th {
    border: none;
    box-shadow: 0 1px 0 rgba(0,0,0,0.04);
    padding: 8px 16px;
}

/* Header */
.table th {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #8395a7;
    box-shadow: 0 1px 0 rgba(0,0,0,0.08);    /* etwas staerker */
}

/* Striped: fast unsichtbar */
.table tr:nth-child(odd) { background: rgba(0,0,0,0.012); }

/* Hover: Akzent-Tint */
.table tr:hover { background: rgba(168,180,216,0.12); }
```

### 7.6 Accordion

```css
.accordion-item {
    background: linear-gradient(145deg, #dde2e9, #edf0f4);
    border: none;
    border-radius: 16px !important;
    box-shadow: var(--nm-raised);
    margin-bottom: 16px;             /* Mindestabstand! */
    overflow: hidden;
}

.accordion-button {
    background: var(--nm-bg);
    border: none;
    font-weight: 600;
    font-size: 15px;
    padding: 16px;
    box-shadow: none !important;
}

.accordion-button:not(.collapsed) {
    color: #8fa0c4;                  /* Akzent-Farbe wenn offen */
}
```

### 7.7 Sidebar-Link

```css
.sidebar-link {
    padding: 10px 16px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 500;
    color: #2d3436;
    /* Normal: kein Shadow — flat */
}

.sidebar-link:hover {
    box-shadow: var(--nm-pressed-sm);   /* eingedrueckt bei Hover */
}

.sidebar-link.active {
    box-shadow: var(--nm-pressed-sm);   /* eingedrueckt = aktiv */
    color: #8fa0c4;                     /* Akzent-Farbe */
}
```

### 7.8 Badge (Pill)

```css
.badge {
    font-weight: 500;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 100px;
    /* Kein Shadow — Badges sind flach, Farbe reicht fuer Hervorhebung */
}
```

### 7.9 Tag / Chip (raised pill)

```css
.tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--nm-bg);
    box-shadow: var(--nm-raised-sm);
    border: none;
    border-radius: 100px;
    padding: 4px 10px 4px 12px;
    font-size: 13px;
}
```

### 7.10 Alert

```css
.alert {
    border: none;
    border-radius: 12px;
    box-shadow: var(--nm-raised-sm);
    font-size: 15px;
}
/* Farben ueber Tints, nicht opake Hintergruende: */
.alert-success { background: rgba(126,200,160,0.12); color: #5a9a7a; }
.alert-danger  { background: rgba(225,112,85,0.12);  color: #a85040; }
.alert-warning { background: rgba(246,185,59,0.12);  color: #a07828; }
.alert-info    { background: rgba(116,185,255,0.12); color: #4a7ab5; }
```

### 7.11 Modal

```css
.modal-content {
    background: var(--nm-bg);
    border: none;
    border-radius: 20px;
    box-shadow: var(--nm-raised-lg);
}
.modal-header { border: none; box-shadow: 0 1px 0 rgba(0,0,0,0.05); }
.modal-footer { border: none; box-shadow: 0 -1px 0 rgba(0,0,0,0.05); }
```

---

## 8. Transitionen

| Token | Dauer | Einsatz |
|-------|-------|---------|
| `0.15s ease` | Schnell | Hover-States, Farbaenderungen, Button-Lift |
| `0.3s ease` | Normal | Shadow-Wechsel (raised <-> pressed), Layout-Shifts, Sidebar-Toggle |

Shadow-Transitionen sind bewusst **0.3s** — schneller wirkt bei Neomorphism "nervoes", weil die Tiefenaenderung subtil ist und Wahrnehmungszeit braucht.

---

## 9. Trennlinien-Technik

Neomorphism verbietet `border`. Interne Trennlinien werden durch **Shadow-Doppellinien** realisiert:

```css
/* Separator nach unten (z.B. Card-Header): */
box-shadow: 0 1px 0 rgba(0,0,0,0.06),       /* dunkle Linie */
            0 2px 0 rgba(255,255,255,0.4);    /* helle Linie darunter */

/* Separator nach oben (z.B. Card-Footer): */
box-shadow: inset 0 1px 0 rgba(0,0,0,0.06),
            inset 0 2px 0 rgba(255,255,255,0.4);

/* Table-Zeile (subtiler): */
box-shadow: 0 1px 0 rgba(0,0,0,0.04);
```

---

## 10. Anti-Aliasing Tricks

```css
/* Auf raised Containern mit border-radius: */
outline: 1px solid transparent;
/* Verhindert pixelige Kanten auf manchen Browsern */

/* Font-Rendering: */
-webkit-font-smoothing: antialiased;
-moz-osx-font-smoothing: grayscale;
```

---

## 11. Anti-Patterns (was NICHT tun)

| Anti-Pattern | Warum nicht | Stattdessen |
|---|---|---|
| Weisser Card-Hintergrund auf grauem Body | Bricht Einheitsprinzip | Alles `#e0e5ec` |
| `border: 1px solid` | Zerstoert "aus-einer-Masse"-Illusion | Shadow-Trennlinien |
| Weisser Text auf Pastell-Buttons | Kontrast < 3:1 | Dunkler Text `#2d3436` |
| Grosse Shadow-Offsets auf Mobile | Uebertrieben | Halbieren ab <=767px |
| Accent `#a8b4d8` als Textfarbe | 1.5:1 Kontrast | Nur als Tint/Icon; fuer Text `#8fa0c4` |
| Abstande < Shadow-Offset+Blur | Schatten laufen ineinander | Mindestens 24px (bei 8+16px Shadow) |
| Schnelle Shadow-Transitions (<0.15s) | Wirkt nervoes | 0.3s ease |

---

## 12. Dark Mode

Neomorphism im Dark Mode folgt denselben Prinzipien wie im Light Mode — einheitliche Oberflaeche, Tiefe nur durch Schatten. Die Umsetzung erfordert jedoch eigene Farben, angepasste Schatten und invertierte Trennlinien.

### 12.1 Aktivierung & Toggle-Mechanismus

**Selektor:** `[data-bs-theme="dark"]` auf dem `<html>`-Element (Bootstrap 5.3 Standard).

**Persistenz:** `localStorage` mit Key `theme-mode`. Fallback auf System-Praeferenz:

```javascript
const theme = localStorage.getItem('theme-mode')
    || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
document.documentElement.setAttribute('data-bs-theme', theme);
```

**Anti-Flash:** Ein Inline-Script im `<head>` setzt das Theme **vor dem ersten Paint**, damit kein weisser Blitz erscheint:

```html
<head>
    <script>
        (function(){
            var t = localStorage.getItem('theme-mode');
            if (!t) t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            document.documentElement.setAttribute('data-bs-theme', t);
        })();
    </script>
</head>
```

**Zusaetzlich:**
- `color-scheme: dark` wird gesetzt, damit native Formularelemente (Scrollbars, Checkboxen) sich anpassen
- `window.MintTheme.isDark()` liefert den aktuellen Mode fuer JavaScript (z.B. Chart.js-Farben)
- Bei Wechsel wird ein `themeChanged` CustomEvent dispatched, damit Charts und andere JS-Komponenten neu rendern

### 12.2 Oberflaechenfarben

Das Kernprinzip bleibt: **Alles hat dieselbe Hintergrundfarbe.** Im Dark Mode ist das `#2a2d3a`.

| Token | Light | Dark |
|-------|-------|------|
| `--nm-bg` | #e0e5ec | #2a2d3a |
| `--nm-text` | #2d3436 | #e0e0e8 |
| `--nm-text-secondary` | #636e72 | #a0a4b8 |
| `--nm-text-muted` | #8395a7 | #6b7090 |

```css
[data-bs-theme="dark"] {
    --nm-bg: #2a2d3a;
    --nm-text: #e0e0e8;
    --nm-text-secondary: #a0a4b8;
    --nm-text-muted: #6b7090;

    /* Bootstrap Bridge */
    --bs-body-bg: var(--nm-bg);
    --bs-body-color: var(--nm-text);
    color-scheme: dark;
}
```

**Nicht reines Schwarz (#000) und nicht reines Weiss (#fff)** — beides wuerde die weiche Neomorphism-Aesthetik zerstoeren.

### 12.3 Shadow-Architektur (Unterschiede zum Light Mode)

#### Primitives

| Token | Light | Dark |
|-------|-------|------|
| `--nm-shadow-dark` | #a3b1c6 (blau-grau) | #1e2028 (fast schwarz) |
| `--nm-shadow-light` | #ffffff (weiss) | #363a4c (helles Dunkelgrau) |

#### Kritischer Unterschied: Kein Edge-Highlight

Im Light Mode haben Raised Shadows **3 Layer** — der dritte ist ein `inset 0 0 0 1px rgba(255,255,255,0.6)` Edge-Highlight fuer Anti-Aliasing. Im Dark Mode wird dieser **entfernt**, weil ein weisser Hairline auf dunklem Hintergrund grell leuchtet statt zu glaetten.

```
Light (3 Layer):
  8px 8px 16px  var(--nm-shadow-dark)       ← Schattenseite
 -8px -8px 16px var(--nm-shadow-light)      ← Lichtseite
  inset 0 0 0 1px rgba(255,255,255,0.6)    ← Edge-Highlight ✓

Dark (2 Layer):
  8px 8px 16px  var(--nm-shadow-dark)       ← Schattenseite
 -8px -8px 16px var(--nm-shadow-light)      ← Lichtseite
                                             ← KEIN Edge-Highlight ✗
```

#### Composed Shadows (Dark Mode)

```css
[data-bs-theme="dark"] {
    --nm-shadow-dark: #1e2028;
    --nm-shadow-light: #363a4c;

    --nm-raised:     8px 8px 16px var(--nm-shadow-dark),
                    -8px -8px 16px var(--nm-shadow-light);
    --nm-raised-sm:  4px 4px 8px var(--nm-shadow-dark),
                    -4px -4px 8px var(--nm-shadow-light);
    --nm-raised-lg: 12px 12px 24px var(--nm-shadow-dark),
                   -12px -12px 24px var(--nm-shadow-light);
    --nm-pressed:    inset 6px 6px 14px var(--nm-shadow-dark),
                     inset -4px -4px 10px var(--nm-shadow-light);
    --nm-pressed-sm: inset 3px 3px 8px var(--nm-shadow-dark),
                     inset -2px -2px 5px var(--nm-shadow-light);
}
```

Pressed Shadows behalten in beiden Modi dieselbe 2-Layer-Struktur — dort gab es nie einen Edge-Highlight.

### 12.4 Farbsystem

Die Farben werden im Dark Mode **leicht entsaettigt und verschoben**, nicht einfach invertiert:

| Rolle | Light | Dark |
|-------|-------|------|
| Accent | #a8b4d8 | #7b8abd |
| Accent-Hover | #8fa0c4 | #9aa4d0 |
| Action | #b0a0d0 | #9888b8 |
| CTA | #ff9f43 | #ff9f43 (gleich) |
| Success | #7ec8a0 | #6abf8a |
| Danger | #e17055 | #e07060 |
| Warning | #f6b93b | #e8b040 |
| Info | #74b9ff | #60a8f0 |

**CTA-Orange bleibt identisch** — es hat auf beiden Hintergruenden ausreichend Kontrast.

#### Alert-Textfarben (aufgehellt)

Auf dunklen Tint-Hintergruenden brauchen Alert-Texte hellere Farben als im Light Mode:

| Alert | Light-Text | Dark-Text |
|-------|-----------|-----------|
| Success | #5a9a7a | #70c898 |
| Danger | #a85040 | #e88070 |
| Warning | #a07828 | #e0b848 |
| Info | #4a7ab5 | #80b8f0 |

### 12.5 Tints (erhoehte Opazitaet)

Im Light Mode verwenden Tints **12%** Opazitaet. Im Dark Mode werden **15%** verwendet, weil 12% auf dunklem Hintergrund fast unsichtbar waeren:

```css
[data-bs-theme="dark"] {
    --nm-tint-accent:  color-mix(in srgb, var(--nm-accent) 15%, transparent);
    --nm-tint-success: color-mix(in srgb, var(--nm-success) 15%, transparent);
    --nm-tint-danger:  color-mix(in srgb, var(--nm-danger) 15%, transparent);
    --nm-tint-warning: color-mix(in srgb, var(--nm-warning) 15%, transparent);
    --nm-tint-info:    color-mix(in srgb, var(--nm-info) 15%, transparent);
}
```

Da `color-mix()` die Dark-Mode-Basisfarben referenziert, passen sich die Tints doppelt an: andere Basisfarbe + hoehere Opazitaet.

### 12.6 Trennlinien (invertiert)

Die Shadow-basierte Trennlinien-Technik (s. Sektion 9) invertiert im Dark Mode. Die Rollen tauschen: die "helle" Linie wird zum dezenten Highlight, die "dunkle" Linie wird dominant.

| Rolle | Light | Dark |
|-------|-------|------|
| Dunkle Linie | `rgba(0,0,0,0.06)` | `rgba(255,255,255,0.04)` |
| Helle Linie | `rgba(255,255,255,0.4)` | `rgba(0,0,0,0.3)` |

```css
/* Light: Card-Header Separator */
box-shadow: 0 1px 0 rgba(0,0,0,0.06),
            0 2px 0 rgba(255,255,255,0.4);

/* Dark: Card-Header Separator */
box-shadow: 0 1px 0 rgba(255,255,255,0.04),
            0 2px 0 rgba(0,0,0,0.3);
```

Das Prinzip bleibt identisch (Doppellinie = eingefraeste Rille), nur die Lichtverhaeltnisse drehen sich.

### 12.7 Komponenten-Anpassungen

#### Cards & Tables

Gradient-Richtung bleibt bei 145 Grad, Farben werden dunkel:

```css
[data-bs-theme="dark"] .card,
[data-bs-theme="dark"] .table-responsive {
    background: linear-gradient(145deg, #2e3142, #262936);
}
```

#### Buttons

Jede Button-Variante bekommt einen eigenen Dark-Gradient:

```css
[data-bs-theme="dark"] {
    & .btn         { background: var(--nm-bg); color: var(--nm-text); }
    & .btn-primary { background: linear-gradient(145deg, #353a52, #2a2e40); }
    & .btn-success { background: linear-gradient(145deg, #2a4038, #243530); }
    & .btn-danger  { background: linear-gradient(145deg, #3a3048, #302838); }
    & .btn-warning { background: linear-gradient(145deg, #4a3a20, #3a2e18); }
    & .btn-info    { background: linear-gradient(145deg, #353a52, #2a2e40); }
}
```

**Buttontext:** Im Dark Mode automatisch hell (`--nm-text` ist jetzt #e0e0e8).

#### Table-Rows

```css
/* Light: schwarze Transparenz fuer Stripes */
tr:nth-child(odd) { background: rgba(0,0,0,0.012); }

/* Dark: weisse Transparenz fuer Stripes */
tr:nth-child(odd) { background: rgba(255,255,255,0.02); }
```

#### Mobile Bars (Frosted Glass)

```css
--topbar-bg:    rgba(42, 45, 58, 0.88);   /* leicht transparent */
--bottombar-bg: rgba(42, 45, 58, 0.92);
```

#### Scrollbar (WebKit)

```css
[data-bs-theme="dark"] {
    & ::-webkit-scrollbar       { width: 8px; }
    & ::-webkit-scrollbar-track { background: var(--nm-bg); }
    & ::-webkit-scrollbar-thumb { background: var(--nm-shadow-light); border-radius: 4px; }
}
```

### 12.8 Responsive Shadows (Mobile, Dark Mode)

Wie im Light Mode werden alle Shadow-Offsets auf Mobile (<=767px) halbiert. Die Dark-Mode-Variante enthaelt ebenfalls keinen Edge-Highlight:

```css
[data-bs-theme="dark"] {
    @media (max-width: 767.98px) {
        --nm-raised:     4px 4px 8px var(--nm-shadow-dark),
                        -4px -4px 8px var(--nm-shadow-light);
        --nm-raised-sm:  2px 2px 5px var(--nm-shadow-dark),
                        -2px -2px 5px var(--nm-shadow-light);
        --nm-raised-lg:  6px 6px 12px var(--nm-shadow-dark),
                        -6px -6px 12px var(--nm-shadow-light);
        --nm-pressed:    inset 4px 4px 10px var(--nm-shadow-dark),
                         inset -3px -3px 7px var(--nm-shadow-light);
        --nm-pressed-sm: inset 2px 2px 5px var(--nm-shadow-dark),
                         inset -2px -2px 4px var(--nm-shadow-light);
    }
}
```

### 12.9 Anti-Patterns (Dark-Mode-spezifisch)

| Anti-Pattern | Warum nicht | Stattdessen |
|---|---|---|
| Inset-Edge-Highlight beibehalten (`rgba(255,255,255,0.6)`) | Leuchtet grell auf dunklem Hintergrund | Edge-Highlight im Dark Mode entfernen |
| 12% Tint-Opazitaet auf dunkel | Kaum sichtbar | 15% verwenden |
| Light-Mode Trennlinien unveraendert | `rgba(0,0,0,0.06)` verschwindet auf dunkel | Invertieren: `rgba(255,255,255,0.04)` |
| Light-Mode Gradients beibehalten | Zu helle Flaechen, brechen Einheitsprinzip | Eigene dunkle Gradients definieren |
| Reines Schwarz (#000) als Hintergrund | Kein Kontrast fuer hellen Shadow-Layer | Getoentes Dunkelgrau (#2a2d3a) |
| Reines Weiss (#fff) als Textfarbe | Zu grell, ermuedend | Off-White (#e0e0e8) |
| `color-scheme: dark` vergessen | Native Elemente (Scrollbars, Inputs) bleiben hell | Immer `color-scheme: dark` setzen |

---

## 13. Minimal-Template zum Kopieren

### Light Mode

```css
:root {
    --nm-bg: #e0e5ec;
    --nm-text: #2d3436;
    --nm-text-secondary: #636e72;
    --nm-text-muted: #8395a7;
    --nm-accent: #a8b4d8;
    --nm-accent-hover: #8fa0c4;
    --nm-action: #b0a0d0;
    --nm-shadow-dark: #a3b1c6;
    --nm-shadow-light: #ffffff;

    --nm-raised:     8px 8px 16px var(--nm-shadow-dark),
                    -8px -8px 16px var(--nm-shadow-light),
                     inset 0 0 0 1px rgba(255,255,255,0.6);
    --nm-raised-sm:  4px 4px 8px var(--nm-shadow-dark),
                    -4px -4px 8px var(--nm-shadow-light),
                     inset 0 0 0 0.5px rgba(255,255,255,0.5);
    --nm-pressed:    inset 6px 6px 14px var(--nm-shadow-dark),
                     inset -4px -4px 10px var(--nm-shadow-light);
    --nm-pressed-sm: inset 3px 3px 8px var(--nm-shadow-dark),
                     inset -2px -2px 5px var(--nm-shadow-light);

    /* Tints — auto-derived via color-mix */
    --nm-tint-accent: color-mix(in srgb, var(--nm-accent) 12%, transparent);

    --nm-font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
```

### Dark Mode

```css
[data-bs-theme="dark"] {
    --nm-bg: #2a2d3a;
    --nm-text: #e0e0e8;
    --nm-text-secondary: #a0a4b8;
    --nm-text-muted: #6b7090;
    --nm-accent: #7b8abd;
    --nm-accent-hover: #9aa4d0;
    --nm-action: #9888b8;
    --nm-shadow-dark: #1e2028;
    --nm-shadow-light: #363a4c;

    /* Shadows — KEIN Edge-Highlight (3. Layer entfaellt) */
    --nm-raised:     8px 8px 16px var(--nm-shadow-dark),
                    -8px -8px 16px var(--nm-shadow-light);
    --nm-raised-sm:  4px 4px 8px var(--nm-shadow-dark),
                    -4px -4px 8px var(--nm-shadow-light);
    --nm-pressed:    inset 6px 6px 14px var(--nm-shadow-dark),
                     inset -4px -4px 10px var(--nm-shadow-light);
    --nm-pressed-sm: inset 3px 3px 8px var(--nm-shadow-dark),
                     inset -2px -2px 5px var(--nm-shadow-light);

    /* Tints — 15% statt 12% fuer Sichtbarkeit auf dunkel */
    --nm-tint-accent: color-mix(in srgb, var(--nm-accent) 15%, transparent);

    color-scheme: dark;
}
```

### Basis-Komponenten (funktionieren in beiden Modi)

```css
* { box-sizing: border-box; }
body {
    font-family: var(--nm-font);
    background: var(--nm-bg);
    color: var(--nm-text);
    font-size: 16px;
    line-height: 1.5;
}

.card     { background: linear-gradient(145deg, #dde2e9, #edf0f4);
            border: none; border-radius: 16px;
            box-shadow: var(--nm-raised); }
.btn      { background: var(--nm-bg); border: none; border-radius: 12px;
            box-shadow: var(--nm-raised-sm); color: var(--nm-text); }
.btn:hover   { box-shadow: var(--nm-raised); transform: translateY(-1px); }
.btn:active  { box-shadow: var(--nm-pressed-sm); transform: translateY(1px); }
input     { background: var(--nm-bg); border: none; border-radius: 12px;
            box-shadow: var(--nm-pressed-sm); }
input:focus  { box-shadow: var(--nm-pressed-sm),
               0 0 0 3px rgba(168,180,216,0.3); outline: none; }

/* Dark-Mode Card-Gradient Override */
[data-bs-theme="dark"] .card {
    background: linear-gradient(145deg, #2e3142, #262936);
}
```

### Anti-Flash Script (im `<head>`)

```html
<script>
    (function(){
        var t = localStorage.getItem('theme-mode');
        if (!t) t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        document.documentElement.setAttribute('data-bs-theme', t);
    })();
</script>
```

---

Die Kernidee in einem Satz: **Eine Oberflaeche, eine Farbe, zwei Schatten, grosszuegige Abstaende — in Hell und Dunkel.**
