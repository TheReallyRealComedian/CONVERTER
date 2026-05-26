# Sprint READER-MODE — Distraction-Free + Reading-Progress + Mark-on-Mouseup

**Datum**: 2026-05-25

**Ziel**: Drei UX-Erweiterungen, die zusammen den `library_detail`-Reader auf das Readwise-Reader-Niveau heben. Frontend-only, kein Schema-Touch, kein Backend-Touch.

1. **Distraction-Free-Toggle**: beide Spalten (globale Sidebar links, Detail-Sidebar rechts mit Markierungen/Notion/Details) per Toggle-Button ein- und ausklappbar. State in localStorage persistiert. Reader-View dehnt sich entsprechend aus.
2. **Reading-Progress-Bar**: schmaler Balken (3px) am oberen Rand des Reader-Containers, der mit dem Scroll-Fortschritt im Doc wächst. Persistierung der Scroll-Position ist **R2-B-Vorbehalt**, hier nur die visuelle Live-Anzeige.
3. **Mark-on-Mouseup**: der schwebende „Markieren"-Button aus R1-B-A wird entfernt. `mouseup`-Event mit nicht-leerer Selection im Reader-View → Highlight wird **direkt** persistiert (Reader-Style à la Readwise Reader). Cmd/Ctrl-Halten = nur Kopieren, kein Highlight. Selection-Längen-Threshold gegen Accidental-Selections.

**Vorbedingung**:
- Pytest 131/131 grün auf `main` (zuletzt R2-A done, commit `25df0ef`).
- R1-B-A bis R2-A liefern die Foundation: `<article class="reader-view">` mit Markdown-Render, Highlight-Schema + API + Popover-Pattern, Tag-Picker im Popover, Sidebar mit Markierungen-Liste, Cross-Format-Sidebar-Card-Popover-Bridge.
- Bestehende JS-Mechanik in [static/js/library_detail.js](static/js/library_detail.js): `saveCurrentSelection()` triggert via `mousedown` auf den Markieren-Button. `showHighlightActionPopover(anchorEl)` öffnet den Edit-Popover (mit Note + Tag-Picker + Löschen). Beides bleibt — wir refactorn nur den Trigger-Pfad.
- User-Feedback vom 2026-05-25 mit Readwise-Reader-Screenshots als visuelle Referenz: dort ist der Reading-Progress-Balken oben, beide Sidebars sind toggle-bar, Selection markiert direkt.

**Out-of-scope**:
- **Reading-Progress-Persistierung** (last_read_position-Schema) — R2-B-Vorbehalt.
- **R2-B Filtered Views** und **R2-C Lifecycle-Status** — separate Sub-Sprints.
- **Toolbar-Buttons im Reader-Header** (Font-Size, Theme-Toggle, TTS) — kommt in einem späteren Polish-Sprint, nicht hier.
- **Mobile-Layout** — keine spezielle Mobile-Optimierung in diesem Sprint, Desktop-First.
- **Keyboard-Shortcuts** für Sidebar-Toggle — könnte nice sein, aber YAGNI für jetzt.
- **Bestehender Click-on-Highlight-for-Popover-Pfad** bleibt unverändert.

---

## Phase 1 — Implementation

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. `pytest tests/` → 131/131 grün als Baseline.
3. Mac-Stack live, R2-A-State aktiv (Container-DB hat Tag-Junction-Daten).
4. **Lese** [base.html](templates/base.html) um die globale-Sidebar-Struktur zu verstehen — die linke Spalte (Markdown to PDF / Document Converter / Audio / Mermaid / Library) sitzt vermutlich dort als shared layout-Element. Toggle muss in der base oder per Detail-View-spezifischer Override gehen.

### Files

```
templates/base.html               # EDIT — globale Sidebar-Toggle-Button (oben links Header oder als Floater)
templates/library_detail.html     # EDIT — Detail-Sidebar-Toggle-Button, Reading-Progress-Bar-Container, Markieren-Button entfernt
static/js/library_detail.js       # EDIT — Mark-on-Mouseup-Logic, Progress-Bar-Listener, Toggle-Logic + localStorage, Markieren-Button-Code raus
static/js/base.js                 # EDIT — globale Sidebar-Toggle-Logic + localStorage (wenn base.html das Toggle hostet)
static/css/style.css              # EDIT — Reader-Mode-Block (Sidebar-Transitions, Progress-Bar, Detail-Sidebar-Hide-State, Globale-Sidebar-Hide-State, Reader-Container-Expand-State, Markieren-Button-Removal)
```

Fünf Files, alle Edits, keine Neuanlagen.

### Mechanik

**1. Distraction-Free-Toggle**:

*Globale Sidebar* (links):
- Toggle-Button: kleines Icon (z.B. Hamburger oder Caret-Left) oben am Rand der Sidebar oder als Floater im Reader-Container.
- Sub-Thread-Disposition zur Platzierung — Empfehlung: Floater am linken Rand des Reader-Containers (`position: fixed; left: 0; top: 50%`), so dass er erreichbar bleibt auch wenn die Sidebar eingeklappt ist.
- Click toggelt CSS-Klasse `global-sidebar--collapsed` am body oder am sidebar-element.
- State: `localStorage.getItem('reader.globalSidebar') === 'collapsed'` → eingeklappt initial.

*Detail-Sidebar* (rechts, nur in `library_detail.html`):
- Toggle-Button: rechts oben im Reader-Container, analoges Pattern.
- Click toggelt CSS-Klasse `detail-sidebar--collapsed` am detail-page-wrapper.
- State: `localStorage.getItem('reader.detailSidebar') === 'collapsed'`.

**CSS-Mechanik**:
```css
/* Smooth transition */
.global-sidebar { transition: transform 200ms ease, margin-left 200ms ease; }
.global-sidebar--collapsed { transform: translateX(-100%); margin-left: -SIDEBAR_WIDTH; }

.detail-sidebar { transition: transform 200ms ease, width 200ms ease, opacity 150ms ease; }
.detail-sidebar--collapsed { transform: translateX(100%); width: 0; opacity: 0; pointer-events: none; }

/* Reader-Container nimmt verfügbaren Platz */
.detail-grid { transition: grid-template-columns 200ms ease; }
.detail-grid--detail-collapsed { grid-template-columns: 1fr 0; }
```

Konkrete Selektor-Namen passt Sub-Thread an die existierenden Klassen aus [library_detail.html](templates/library_detail.html) und [base.html](templates/base.html) an.

**Toggle-Button-Microcopy / ARIA**:
- `aria-label="Sidebar ein-/ausblenden"` deutsch
- Icon-Wahl: Sub-Thread disponiert (Hamburger / Caret-Left/Right / EyeOff — letzteres wäre semantisch näher am „distraction-free")
- Keine Text-Labels am Button (Floater bleibt minimal)

**2. Reading-Progress-Bar**:

Markup (in `library_detail.html`, am Anfang des Reader-Containers, vor dem Article):
```html
<div class="reading-progress" aria-hidden="true">
  <div class="reading-progress__fill" id="reading-progress-fill"></div>
</div>
```

CSS:
```css
.reading-progress {
  position: sticky;
  top: 0;
  z-index: 10;
  height: 3px;
  background: var(--nm-text-muted);
  opacity: 0.15;
}
.reading-progress__fill {
  height: 100%;
  width: 0;
  background: linear-gradient(90deg, var(--nm-tint-accent), var(--nm-text));
  transition: width 80ms ease-out;
}
.reading-progress--hidden { display: none; }
```

JS (in `library_detail.js`):
```js
function initReadingProgress() {
  const container = document.getElementById('content-body');
  const fill = document.getElementById('reading-progress-fill');
  const wrapper = document.querySelector('.reading-progress');
  if (!container || !fill) return;
  
  function update() {
    const scrollable = container.scrollHeight - container.clientHeight;
    if (scrollable <= 0) {
      wrapper.classList.add('reading-progress--hidden');
      return;
    }
    wrapper.classList.remove('reading-progress--hidden');
    const percent = (container.scrollTop / scrollable) * 100;
    fill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  }
  
  container.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
}
```

Beachte: Scroll passiert vermutlich auf `#content-body` (max-h-overflow), nicht auf `document.documentElement`. Sub-Thread verifiziert das beim Smoke und passt den Scroll-Container-Selektor an.

**3. Mark-on-Mouseup**:

*Entfernen* aus `library_detail.html` und `library_detail.js`:
- `#highlight-create-btn` (HTML + CSS-Klasse `.highlight-create-btn`)
- `positionHighlightCreateBtn`-Funktion + `selectionchange`-Listener für die Button-Positionierung
- `mousedown`-Handler auf dem Button
- Existing `saveCurrentSelection`-Function **bleibt**, wird nur anders getriggert

*Neu*: `mouseup`-Listener auf dem `<article class="reader-view">`:

```js
function initMarkOnMouseup() {
  const reader = document.querySelector('.reader-view');
  if (!reader) return;
  
  reader.addEventListener('mouseup', (evt) => {
    // Cmd/Ctrl gehalten = User wollte kopieren, kein Highlight
    if (evt.metaKey || evt.ctrlKey) return;
    
    // Kurzer Defer damit Selection fertig propagiert ist
    setTimeout(() => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) return;
      if (!reader.contains(sel.anchorNode) || !reader.contains(sel.focusNode)) return;
      
      const text = sel.toString();
      if (text.trim().length < MIN_HIGHLIGHT_LENGTH) return;  // z.B. 3 chars
      
      // Existing-Logic — saveCurrentSelection nimmt Selection aus dem DOM
      saveCurrentSelection();
    }, 10);
  });
}
```

**Konstanten** (oben in `library_detail.js`):
```js
const MIN_HIGHLIGHT_LENGTH = 3;  // Selections kürzer als 3 chars werden ignoriert
```

**Toast-Wording**:
- Aktuell zeigt `saveCurrentSelection` Toast „Markiert." bei Erfolg.
- **Master-Entscheidung**: Toast bleibt, aber als `info`-Level statt `success` (weniger prominent). Reader-Mode ist still, aber komplett-still ohne Feedback wäre unsicher (User weiß nicht ob Save erfolgreich war). 
- Alternativ Sub-Thread-Disposition: ganz weglassen wenn der Highlight-Span sofort gelb erscheint (sichtbares Feedback reicht). Eigene Entscheidung erlaubt — Bericht-Item.

**Cross-Format-Toast** (existing): bei `applyHighlight === false` zeigt Code aktuell „Markierung gespeichert, Anzeige nicht möglich (Formatierungsgrenze)." als `warning`. Bleibt unverändert.

**Edge-Case-Vorgabe**:
- Selection startet außerhalb Reader-View und endet innerhalb (oder umgekehrt): Frühe Bail im Listener via `reader.contains(anchorNode) && reader.contains(focusNode)`-Check.
- Schnelle Klick-Sequenzen (Triple-Click selektiert Absatz): standard Browser-Verhalten, soll funktionieren. Triple-Click ist legitime Markier-Geste.
- Selection bei aktivem Popover (User hat Highlight angeklickt, dann anderen Text selektiert): Popover schließt sich nicht automatisch — Sub-Thread entscheidet ob das Click-outside-Pattern den Popover schließt bevor neue Selection ankommt.

### Code-Quality-Gates

- UI-Strings deutsch.
- `showToast` für Banner.
- Helper-Reuse: `localStorage`-Wrapper falls existierend in [_utils.js](static/js/_utils.js) wiederverwenden.
- Live-Smoke nach Frontend-Änderung Pflicht.
- Pytest 131/131 grün vor Phase-Ende (kein Backend-Touch, sollte trivial sein).

### Phase-1-Stop

Nach Phase 1: STOP — Bericht. Toggle-Button-Platzierungs-Disposition, Icon-Wahl, Scroll-Container-Selektor-Verifikation, Toast-Disposition (info statt success oder ganz weg), Globale-Sidebar-Position-Verifikation (sitzt sie wirklich in base.html oder per-Page).

---

## Phase 2 — Verify

**Pytest**:

1. `docker compose exec markdown-converter pytest tests/` → 131/131 grün (kein Backend-Touch, Erwartung trivial).

**Live-Smoke** (Browser, Smoke-User, Test-Doc mit langem Content — z.B. doc 2 Quartalsbericht oder ein eigens erzeugtes langes Markdown-Doc für Scroll-Smoke):

2. **Detail-View** öffnen mit langem Doc → Reading-Progress-Bar sichtbar am oberen Rand des Reader-Containers, Fill = 0%.
3. **Scroll** im Reader → Fill wächst smooth mit Scroll-Fortschritt. Bei 100% Scroll → Fill = 100%.
4. **Kurzes Doc** (das nicht scrollt): Reading-Progress-Bar versteckt (`reading-progress--hidden`).
5. **Detail-Sidebar-Toggle** klicken → rechte Sidebar schiebt smooth raus (200ms), Reader-View dehnt sich aus.
6. **Detail-Sidebar-Toggle** nochmal klicken → rechte Sidebar schiebt rein, alles wieder normal.
7. **Page-Reload** nach Toggle: localStorage merkt sich den State, Sidebar bleibt eingeklappt.
8. **Globale-Sidebar-Toggle** klicken → linke Sidebar schiebt raus. Page-Reload persistiert.
9. **Beide Sidebars eingeklappt** → Reader-View nimmt volle Breite ein. Distraction-Free-Mode.
10. **Mark-on-Mouseup**: Text mit Maus markieren → Maustaste loslassen → Highlight wird sofort persistiert (gelb, Span im DOM, DB-Insert, Sidebar-Card-Update). **Kein schwebender „Markieren"-Button** mehr.
11. **Cmd-Halten + Selektieren** → KEIN Highlight (Copy-Pfad). Klassische Copy-Geste funktioniert weiterhin via Cmd+C.
12. **Selektions-Threshold**: 2-Char-Selection (z.B. „ab") → kein Highlight. 3-Char-Selection (z.B. „abc") → Highlight.
13. **Click auf bestehenden Highlight** → Popover öffnet (Note + Tag-Picker + Löschen) — existing R1-B-A/B/C-Verhalten unverletzt.
14. **Cross-Format-Sidebar-Card-Click** → Popover öffnet über Card (existing R1-B-C-Bridge) — unverletzt.
15. **Toast-Wirkung**: nach Highlight-Save Toast erscheint kurz (info-Level wenn Sub-Thread auf info umgestellt hat). Subjektive Beurteilung: wirkt das angemessen still?
16. **Dark-Mode**: Toggle in beide Themes → Progress-Bar, Sidebar-Transitions, Highlight-Spans alle lesbar.

**Edge-Case-Smoke**:

17. Triple-Click-Absatz-Selektion → Absatz-Highlight wird erstellt (legitime Geste).
18. Selection startet außerhalb Reader-View (z.B. im Titel-Header) und endet innen → KEIN Highlight (bail wegen anchorNode/focusNode-Check).
19. Page-Resize während Scrolls → Progress-Bar adjustiert sich (resize-Listener).
20. Mehrere Highlights schnell hintereinander setzen (4-5 in 10 Sekunden) → alle persistiert, Sidebar-Karten alle korrekt, keine Race.

Nach Phase 2: STOP — Bericht. Liste der gesmokten Pfade, Toast-Disposition-Wirkung, Sidebar-Toggle-Smoothness-Beurteilung, Mark-on-Mouseup-Akzeptanz.

---

## Phase 3 — Commit + Push + Image-Rebuild

- Plain-prose Commit-Message, mehrere `-m`-Flags.
- Ein Commit. Subject z.B. „READER-MODE: distraction-free toggles + reading-progress-bar + mark-on-mouseup".
- Body soll erwähnen: drei Sub-Features, Mouseup-Refactor mit Cmd-Bypass + 3-Char-Threshold, Sidebar-Toggle-localStorage, Progress-Bar-Listener.
- Image-Rebuild via `docker compose up -d --build`.
- Branch direkt auf `main`. Push direkt nach Commit.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

**Zusätzlich für READER-MODE**:
- Wenn die Globale-Sidebar nicht in `base.html` sitzt sondern per-Page (z.B. in jedem Template separat): **STOP** Bericht-Item, Master entscheidet ob Refactor in base.html oder per-Page-Toggle-Duplikation.
- Wenn der `mouseup`-Listener mit anderen Selection-Handlern in `library_detail.js` interferiert (z.B. mit Klick-auf-Highlight-für-Popover): **STOP** Bericht-Item, Master entscheidet ob Event-Phasen-Trennung oder feinere Bedingungen.
- Wenn die Sidebar-Toggle-CSS-Transition Layout-Sprünge erzeugt (z.B. Reader-Container reflowed visuell schlechter als per `transform`): nicht aggressiv refactorn, im Bericht erwähnen.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — drei Frontend-Sub-Features die UX-konzeptionell zusammenhängen aber technisch unabhängig sind. Schema-Touch null, Backend-Touch null, ~150-200 LOC JS + ~50 LOC CSS + ~10 LOC HTML. Wenn sich beim Smoke zeigt, dass die Globale-Sidebar-Position oder die Mouseup-Trigger-Logik unerwartete Komplikationen haben: eskalation auf L, Master entscheidet ob splitten in „READER-MODE-A Sidebars+Progress" und „READER-MODE-B Mark-on-Mouseup".

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Refactor der Markieren-Button-Entfernung andere Dead-Code-Stellen (z.B. `selectionchange`-Listener, der nur für die Button-Positionierung war) übrig bleiben: aufräumen.
- Wenn die `.detail-grid`-CSS-Mechanik in `library_detail.html` einen Polish-Touch braucht (z.B. Grid-Columns sind nicht responsiv): kleiner Tweak OK, kein Refactor.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „READER-MODE ☑ done 2026-05-25 → commit `<hash>` (distraction-free Sidebars + Reading-Progress-Bar + Mark-on-Mouseup). Beide Sidebars per Toggle ein-/ausklappbar, State in localStorage; Reading-Progress-Bar 3px am Reader-Container-Top mit Scroll-Listener (nicht persistiert, R2-B-Vorbehalt); Mark-on-Mouseup mit Cmd-Bypass + 3-Char-Threshold ersetzt schwebenden Markieren-Button. Pytest 131/131 grün (kein Backend-Touch). **Master-Aktivität nächste**: R2-B Filtered Views + Reading-Progress-Persistierung oder R2-C Lifecycle."
- **BACKLOG.md**: READER-MODE in Erledigt-Liste; falls neue P3-Reminder aufgetaucht sind (z.B. „Reading-Progress-Persistierung in R2-B wiederverwenden", „Toggle-Button-Mobile-Layout", „Keyboard-Shortcut für Sidebar-Toggle"), als P3-Bullets ergänzen.
- **Memory**: wenn die Mouseup-mit-Defer-für-Selection-Mechanik als Pattern reift (z.B. R2-B könnte Selection-basierte UI auch brauchen): `reference_mouseup_selection_pattern.md`. Nichts erzwingen.

---

## Phase-0-Entscheidungen

_(Phase 0 nicht aktiviert — Mechanik klar nach User-Feedback vom 2026-05-25 mit Readwise-Reader-Screenshots als visuelle Referenz. Toggle-Buttons als Floater am Reader-Container-Rand (Master-Empfehlung), Reading-Progress als 3px sticky-bar mit Gradient-Fill, Mark-on-Mouseup mit Cmd-Bypass und 3-Char-Threshold gegen Accidental-Selection. Toast bleibt als info-Level, sub-Thread kann auf weglassen umstellen.)_
