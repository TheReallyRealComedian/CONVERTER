# Sprint READER-FIX-A — Multi-Node-Highlight + Sidebar-Card-Expansion

**Datum**: 2026-05-26

**Ziel**: Zwei UX-Limitations aus dem R1-B-A/B-B-Cluster beheben, die nach dem READER-MODE-Live-Test als alltägliche Reibung aufgefallen sind:

1. **Multi-Node-Highlight-Wrap**: aktuell wirft `Range.surroundContents` `InvalidStateError` wenn die Selection über mehrere Text-Nodes spannt (z.B. über `<strong>`/`<em>`/Heading-Boundary). R1-B-A löste das als „graceful fallback" mit DB-Persist aber ohne DOM-Span — der Toast „Formatierungsgrenze, Anzeige nicht möglich" macht das Markieren über Markdown-Format-Elements hinweg praktisch unbrauchbar. Lösung: **Range-Walking** statt `surroundContents` — pro überspanntem Text-Node ein eigenes `<span>` mit derselben `data-highlight-id`. Etablierte Technik (Hypothes.is, Annotator.js).

2. **Sidebar-Card-Expansion**: Cards zeigen aktuell nur 80-Char-Snippet + 60-Char-Note-Preview + max 3 Tag-Chips. Click → scroll-to-Highlight + Flash. User will beim Click die **volle Markierung** lesen können. Lösung: Card hat collapsed/expanded State, Click toggelt + scrollt parallel.

**Vorbedingung**:
- Pytest 131/131 grün auf `main` (zuletzt READER-MODE done, commit `e255c3f` + `02ec054`).
- R1-B-A bis READER-MODE liefern Foundation. Bestehender `applyHighlight`-Pfad in [static/js/library_detail.js](static/js/library_detail.js) ist Refactor-Quelle.
- `crossFormatHighlightIds` Set in `library_detail.js` markiert aktuell Highlights ohne DOM-Span. Nach Phase 1.1 sollte das Set in der Praxis fast leer bleiben — Defense bleibt aber drin für absolute Edge-Cases (z.B. Selection spannt über `.reader-view`-Grenze hinaus, aber das bailt eh schon im mouseup-Listener).
- `renderHighlightList`-Function rendert Sidebar-Cards aus `highlightsState`. Card-Markup ist Quelle für Phase-1.2-Refactor.

**Out-of-scope**:
- **R2-B Filtered Views + Reading-Progress-Persistierung** — separater Sprint.
- **R2-C Lifecycle-Status** — separater Sprint.
- **Highlight-Color-Coding** (verschiedene Farben pro Highlight) — späterer Polish-Sprint.
- **Highlight-Edit-via-Card** (Note + Tag-Picker inline in expanded Card) — Master-Disposition: Edit bleibt im Popover-Trigger (Click auf Highlight im Reader), expanded Card ist nur Read-View mit Edit-Button als optionalem Quick-Trigger. Sub-Thread disponiert ob Edit-Button in expanded Card aufgenommen wird.
- **Multi-Selection-Highlight-Merge** (zwei separate Selections zu einem Highlight vereinen) — YAGNI.

---

## Phase 1.1 — Multi-Node-Highlight-Wrap

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. `pytest tests/` → 131/131 grün als Baseline.
3. Mac-Stack live. Container-DB hat Test-Highlights aus den vorigen Sprints (inkl. den Cross-Format-Highlight `id=4` aus doc 4 — der wird nach diesem Fix sichtbar werden).
4. **Lese** bestehende `applyHighlight`/`rangeForOffsets`/`locateHighlightOffset`-Logik in `library_detail.js` (vermutlich Z.300-450 nach READER-MODE-Refactor).

### Files

```
static/js/library_detail.js   # EDIT — applyHighlight refactored zu Multi-Node-Wrap via Range-Walking, removeHighlight aggregiert über data-highlight-id, optional Hover-Sync
static/css/style.css          # EDIT — falls Hover-Sync via Class implementiert: .highlight-card[data-highlight-id="X"]:hover-Aggregation
tests/test_highlights.py      # EDIT — falls bestehende Tests Annahmen über Single-Span-Pattern enthalten (Backend-Tests sollten nicht betroffen sein, aber Verify)
```

Drei Files, alle Edits. Frontend-only, kein Schema/Backend-Touch.

### Mechanik

**Algorithmus — Range-Walking-Wrap**:

```js
function wrapSelectionAsHighlight(range, highlightId) {
  // 1. Sammle alle Text-Node-Bereiche innerhalb des Range
  const textRanges = collectTextRangesInRange(range);
  if (textRanges.length === 0) return false;
  
  // 2. Wrap in UMGEKEHRTER Reihenfolge (Ende → Anfang)
  //    damit frühere DOM-Mutationen die späteren Offsets nicht verschieben
  for (let i = textRanges.length - 1; i >= 0; i--) {
    const { textNode, startOffset, endOffset } = textRanges[i];
    const subRange = document.createRange();
    subRange.setStart(textNode, startOffset);
    subRange.setEnd(textNode, endOffset);
    
    const span = document.createElement('span');
    span.className = 'highlight';
    span.setAttribute('data-highlight-id', String(highlightId));
    try {
      subRange.surroundContents(span);
    } catch (err) {
      // Sollte nicht mehr passieren — Sub-Range ist immer single-text-node
      console.warn('subRange surroundContents failed', err);
    }
  }
  return true;
}

function collectTextRangesInRange(range) {
  const result = [];
  const root = range.commonAncestorContainer;
  const walker = document.createTreeWalker(
    root.nodeType === Node.TEXT_NODE ? root.parentNode : root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!range.intersectsNode(node)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );
  let node = walker.nextNode();
  while (node) {
    // Skip empty text nodes
    if (!node.nodeValue || !node.nodeValue.trim()) {
      node = walker.nextNode();
      continue;
    }
    const startOffset = (node === range.startContainer) ? range.startOffset : 0;
    const endOffset = (node === range.endContainer) ? range.endOffset : node.nodeValue.length;
    if (startOffset < endOffset) {
      result.push({ textNode: node, startOffset, endOffset });
    }
    node = walker.nextNode();
  }
  return result;
}
```

**Edge-Cases**:
- Range startet+endet im selben Text-Node (häufigster Fall, R1-B-A-Path) → `collectTextRangesInRange` liefert genau 1 Range → 1 Span wie bisher. Backward-compatible.
- Range spannt mehrere Block-Elements (z.B. `<p>` zu `<p>`) → N Text-Nodes → N Spans → optisch ein durchgehender Highlight, weil das Format dazwischen weiterläuft.
- Whitespace-only Text-Nodes (zwischen Elements) → übersprungen (cleaner Output).
- Range startet in einem leeren Container (`<p></p>`) → keine Text-Nodes → no-op.

**Re-Apply beim Doc-Load** (`applyHighlight`):

Aktuell ruft `applyHighlight(highlight)` `locateHighlightOffset` für `exact` mit `prefix`/`suffix`-Disambiguation, dann `rangeForOffsets`, dann `surroundContents`. **Neu**: `rangeForOffsets` muss ein Range zurückgeben (kein bail bei Multi-Node mehr), dann ruft `wrapSelectionAsHighlight` mit der `highlight.id`.

`rangeForOffsets` muss erweitert werden: aktuell bailt sie wenn startNode != endNode. Neu: setzt `range.setStart(startNode, startOffset)` und `range.setEnd(endNode, endOffset)` — Browser unterstützt das nativ, der Range spannt dann mehrere Text-Nodes. `wrapSelectionAsHighlight` kümmert sich um das Wrapping.

**Cleanup — `removeHighlight(id)`**:

Aktuell: findet `span.highlight[data-highlight-id="X"]` (single), unwrappt mit `parent.replaceChild(textNode, span); parent.normalize()`.

Neu: findet **alle** Spans mit `[data-highlight-id="X"]`, unwrappt jeden, am Ende `reader.normalize()` einmal aggregiert (statt N-mal). 

```js
function removeHighlightSpans(id) {
  const reader = document.querySelector('.reader-view');
  if (!reader) return;
  const spans = reader.querySelectorAll(`span.highlight[data-highlight-id="${id}"]`);
  spans.forEach(span => {
    const parent = span.parentNode;
    while (span.firstChild) parent.insertBefore(span.firstChild, span);
    parent.removeChild(span);
  });
  reader.normalize();
}
```

**Hover-Sync** (optional, Sub-Thread-Disposition):

Aktuell CSS `.highlight:hover` greift pro Span einzeln. Bei Multi-Node-Highlight sehen die Spans optisch zusammen aus, aber Hover färbt nur den gehoverten Span. UX-Frage: stört das? Sub-Thread bewertet im Smoke. Falls ja: JS-event-handler aggregiert über `data-highlight-id` und setzt eine `.highlight--hovered`-Klasse auf alle Spans der Gruppe.

**Cross-Format-Toast und `crossFormatHighlightIds`-Set**:

Bleiben drin als Defense für absolute Edge-Cases (z.B. Selection erweitert sich post-mouseup durch DOM-Manipulation in unerwartete Bereiche). Nach Phase 1.1 sollte das Set in der Praxis leer bleiben. Wenn `wrapSelectionAsHighlight` `false` returnt (z.B. weil `collectTextRangesInRange` 0 liefert), bleibt der bestehende Cross-Format-Fallback-Pfad als Sicherheitsnetz.

### Phase-1.1-Stop

Nach Phase 1.1: STOP — Bericht. Algorithmus-Korrektheit (TreeWalker mit acceptNode für Range-Intersection getestet?), Edge-Case-Disposition für Whitespace-only Text-Nodes, Hover-Sync-Disposition, Cross-Format-Set-Status (sollte in der Praxis leer sein nach Smoke), ob bestehende Tests Annahmen über Single-Span-Pattern enthielten und ggf. angepasst wurden.

---

## Phase 1.2 — Sidebar-Card-Expansion

### Files

```
templates/library_detail.html  # EDIT — falls Card-Markup im Template (sonst rein JS-render)
static/js/library_detail.js    # EDIT — renderHighlightList-Card-Builder erweitert um expanded-State + Toggle-Handler
static/css/style.css           # EDIT — neue Block .highlight-card[data-expanded] mit Typografie für full text + smooth max-height-Transition
```

Drei Files, alle Edits.

### Mechanik

**Card-State**:
- `data-expanded="false"` (default) oder `data-expanded="true"`
- Click auf Card toggelt das Attribut + ruft existing `scrollToHighlight(id)` + Flash (bestehender Pfad bleibt).
- State **nicht** in localStorage persistiert (per-Doc-View ephemeral, nicht persistent).

**Card-Markup-Erweiterung**:

Aktuell (Vermutung, Sub-Thread verifiziert):
```html
<div class="highlight-card" data-highlight-id="X">
  <div class="highlight-card__snippet">[80 chars von exact]…</div>
  <div class="highlight-card__note">[60 chars von note]…</div>
  <div class="highlight-card__tags">[Chip][Chip][Chip] [+N]</div>
</div>
```

Neu:
```html
<div class="highlight-card" data-highlight-id="X" data-expanded="false">
  <div class="highlight-card__exact">[full text — CSS clamps in collapsed state]</div>
  <div class="highlight-card__note">[full note — CSS clamps in collapsed state]</div>
  <div class="highlight-card__tags">[Chip][Chip]...[Chip] (full list in expanded, max 3 + +N in collapsed)</div>
</div>
```

CSS-Mechanik via `line-clamp` (Standard im modernen CSS):

```css
.highlight-card__exact {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  /* ... */
}
.highlight-card[data-expanded="true"] .highlight-card__exact {
  -webkit-line-clamp: unset;
  overflow: visible;
}
.highlight-card__note {
  /* analog mit -webkit-line-clamp: 1 oder 2 */
}
.highlight-card[data-expanded="true"] .highlight-card__note {
  -webkit-line-clamp: unset;
}
.highlight-card__tags {
  /* collapsed: zeigt max 3 + +N via JS-Render */
}
```

**Tag-Truncation in expanded**:
- Collapsed: max 3 sichtbare Chips, `+N`-Pseudo-Chip wenn mehr (R1-B-B-Pattern bleibt).
- Expanded: alle Chips sichtbar, kein +N.
- Sub-Thread implementiert via JS-conditional in `renderHighlightList`.

**Smooth-Transition** (optional, Sub-Thread-Disposition):
- `max-height`-Animation ist tricky bei unbekannter Inhaltshöhe. Alternative: `grid-template-rows: 0fr` → `1fr` transition (moderner CSS-Trick, gut unterstützt).
- Wenn zu komplex: einfach `display: -webkit-box; -webkit-line-clamp` umschalten ohne Animation. Acceptable.

**Edit-Button in expanded Card** (Master-Disposition: optional, Sub-Thread entscheidet):
- Wenn drin: kleiner „Bearbeiten"-Button rechts unten in der expanded Card öffnet existing `showHighlightActionPopover(card)` (analog R1-B-C Cross-Format-Bridge).
- Wenn nicht drin: Card bleibt Read-View, Edit nur via Click auf Highlight im Reader. Akzeptabel — User-Mental-Model: Sidebar = Übersicht, Reader = Aktion.
- Empfehlung: **drinlassen**, weil das die Edit-Erreichbarkeit aus der Sidebar verbessert ohne große Komplexität.

**Click-Routing**:
- Click auf Card-Body → toggelt expand + scroll-to-Highlight + Flash
- Click auf Edit-Button (wenn drin) → öffnet Popover (Event-Propagation stoppen)
- Click auf Tag-Chip in der Card → optional auf Tag-Filter-View navigieren (R2-B-Vorbehalt) oder no-op
- Click außerhalb Card → kein State-Change

### Phase-1.2-Stop

Nach Phase 1.2: STOP — Bericht. Card-Markup-Strategie (line-clamp vs. grid-rows-transition), Edit-Button-Disposition (drin/draußen), Animations-Disposition, Tag-Chip-Click-Verhalten (no-op vs. R2-B-Tease), allgemeine UX-Beurteilung des Expand-Patterns.

---

## Phase 2 — Verify

**Pytest** (im rebuild Container):

1. `docker compose exec markdown-converter pytest tests/` → 131/131 grün (Frontend-only, Erwartung trivial).

**Live-Smoke** (Browser, Smoke-User, Test-Docs):

**Multi-Node-Highlight-Smokes**:

2. **doc 4 „Sprecher-Dialog HOST/GAST"**: der existing cross-format-Highlight `id=4` (`exact = "HOST: Willkommen "`) muss **sichtbar** sein als DOM-Spans im Reader nach Page-Load. Beide Wörter (`HOST:` und `Willkommen`) bekommen jeweils einen Span, gleiches gelbes Background, sieht wie ein durchgehender Highlight aus.
3. **Neue Multi-Node-Selection**: Markiere Text der ein `<strong>` überspannt (z.B. „Auslieferungsquote lag bei **94 Prozent**." aus doc 2). Mouseup → 3 Spans entstehen (vor strong, im strong, nach strong), alle gleicher data-highlight-id, optisch durchgehend.
4. **Heading-Boundary**: Markiere Text von Ende eines Absatzes über eine `<h2>`-Heading bis zum nächsten Absatz. Mehrere Spans, alle gleicher ID.
5. **Listen-Item-Boundary**: Markiere Text von einem `<li>` zum nächsten. Funktioniert.
6. **Single-Node-Selection (Regression)**: Normale Markierung innerhalb eines Absatzes ohne Format-Element → 1 Span wie bisher. R1-B-A-Pfad nicht gebrochen.
7. **Reload**: Multi-Node-Highlight aus Smoke 3 bleibt nach Reload mit allen N Spans erhalten.
8. **Delete**: Click auf einen Multi-Node-Highlight-Span → Popover → Löschen → alle N Spans verschwinden gleichzeitig, DB-Row weg.
9. **Cross-Format-Toast**: muss in den Smokes 2-5 nie erscheinen. Wenn er auftaucht: STOP, Bericht-Item.

**Sidebar-Card-Expansion-Smokes**:

10. **Card-Click expand**: Click auf eine Card mit langem `exact` (z.B. > 80 chars) → Card expandiert, voller Text sichtbar. Scroll im Reader läuft parallel (existing Flash-Animation).
11. **Card-Click collapse**: Zweiter Click → Card collapsed zurück.
12. **Note-Expand**: Card mit langer Note (> 60 chars) — expanded zeigt vollen Note-Text.
13. **Tag-Expand**: Card mit > 3 Tags — collapsed zeigt 3 + „+N", expanded zeigt alle.
14. **Edit-Button** (wenn drin): Click auf Edit-Button in expanded Card → Popover öffnet (Note-Edit + Tag-Picker), Card bleibt expanded im Hintergrund.
15. **Cross-Format-Card-Expand**: Sollte nach Phase 1.1 in der Praxis nicht mehr existieren, aber Defense: wenn doch eine cross-format-Card da ist (z.B. von vor Phase 1.1 persistiert), Click expandiert die Card normal, scrollToHighlight bailt graceful (kein Ziel).

**Cross-Domain-Edge-Cases**:

16. **Highlight über `<code>`-Block**: Markiere Text der einen Inline-`<code>`-Element überspannt. Funktioniert.
17. **Highlight über Codeblock**: Markiere Text der eine Pygments-`<pre>`/`<code>`-Section anfasst. Edge-Case — Pygments-internes Markup mit Klassen wie `.highlight` (siehe Memory `feedback_css_class_collision_in_markdown_views.md`) darf nicht kollidieren mit unserem `span.highlight[data-highlight-id]`-Selektor. Vermutlich OK, aber explizit verifizieren.
18. **Dark-Mode**: Multi-Node-Highlights und Sidebar-Card-Expansion lesbar in beiden Themes.

Nach Phase 2: STOP — Bericht. Smoke-Pfade-Übersicht, Cross-Format-Set-Status (sollte fast leer sein), Hover-Sync-UX-Beurteilung, Tag-Chip-Click-Verhalten, Edit-Button-Disposition wenn relevant.

---

## Phase 3 — Commit + Push + Image-Rebuild

- Plain-prose Commit-Message, mehrere `-m`-Flags.
- Ein Commit. Subject z.B. „READER-FIX-A: multi-node highlight wrap via range-walking + sidebar-card expand-on-click".
- Body soll erwähnen: (1) Range-Walking statt `surroundContents` mit N Spans pro Highlight, (2) collectTextRangesInRange + Reverse-Order-Wrap, (3) removeHighlightSpans aggregiert über data-highlight-id, (4) Cross-Format-Fallback bleibt als Defense, (5) Card-Expand-State mit line-clamp/grid-rows-Animation, (6) Edit-Button-Disposition.
- Image-Rebuild via `docker compose up -d --build`.
- Branch direkt auf `main`. Push direkt nach Commit.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

**Zusätzlich für READER-FIX-A**:
- Wenn `Range.surroundContents` im Sub-Range trotz Single-Text-Node-Garantie failt (sehr seltener Edge-Case mit Custom-Elements oder Shadow-DOM): **STOP**, Master fragen.
- Wenn TreeWalker mit `acceptNode` + Range-Intersection in einer Browser-Version (Chromium-Variante) unerwartet anders verhält: **STOP**, Bericht-Item.
- Wenn die `data-highlight-id`-Aggregation beim Hover-Sync visuelle Glitches erzeugt (z.B. Multi-Span-Flackern bei mouseenter): nicht aggressiv lösen, im Bericht erwähnen.
- Wenn line-clamp-CSS in einer relevanten Browser-Version (Safari) anders rendert: Alternative `grid-template-rows: 0fr` → `1fr` Pattern.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — zwei Frontend-Sub-Phasen die unabhängig sind. Phase 1.1 Multi-Node ist die schwierigere (Range-Walking + Reverse-Order-Wrap + Edge-Cases), Phase 1.2 Card-Expansion ist S. Wenn beim Smoke unerwartete Range-API-Komplikationen auftauchen oder line-clamp-Browser-Quirks: eskalation auf L, Master entscheidet ob splitten.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Refactor von `applyHighlight` Dead-Code-Stellen aus R1-B-A übrig bleiben (z.B. `crossFormatHighlightIds`-Set wird vielleicht überflüssig wenn Multi-Node-Wrap nie mehr false returnt): aufräumen ist OK, aber Set behalten als Defense-Mechanik ist auch OK.
- Wenn die Card-Markup-Refactor (Phase 1.2) eine Verbesserung an `renderHighlightList`-Performance ermöglicht (z.B. DocumentFragment-Batching bei N>10 Cards): kleine Optimization OK.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „READER-FIX-A ☑ done 2026-05-26 → commit `<hash>` (Multi-Node-Highlight-Wrap via Range-Walking + Sidebar-Card-Expand-on-Click). Cross-Format-Limitation aus R1-B-A behoben — Highlights spannen jetzt über `<strong>`/`<em>`/Heading-/Listen-Boundaries. Sidebar-Cards zeigen full exact + full note + alle Tags bei Click. Pytest 131/131 grün."
- **BACKLOG.md**: READER-FIX-A in Erledigt-Liste; falls neue P3-Reminder aufgetaucht sind (z.B. „Hover-Sync für Multi-Node-Highlights via JS-event statt CSS"), als P3-Bullets ergänzen.
- **Memory**: wenn die Range-Walking-Mechanik als wiederverwendbares Pattern reift (für künftige Reader-Annotations- oder Selection-Features): `reference_dom_range_walking.md`. Plus: wenn die Reverse-Order-Wrap-Mechanik (Mutations am Ende zuerst) eine generelle Lehre ist: separate Memory. Sub-Thread-Disposition.

---

## Phase-0-Entscheidungen

_(Phase 0 nicht aktiviert — Mechanik klar nach User-Feedback vom 2026-05-26: Multi-Node-Wrap via Range-Walking ist etablierte Technik (Hypothes.is, Annotator.js), Reverse-Order-Wrap gegen Offset-Drift ist Standard, Card-Expand via line-clamp ist Standard-CSS. Edit-Button-Disposition optional. Hover-Sync optional.)_
