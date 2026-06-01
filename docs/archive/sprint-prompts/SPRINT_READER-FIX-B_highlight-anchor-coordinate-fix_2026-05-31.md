# Sprint READER-FIX-B — Highlight-Anker-Koordinaten-Fix + Speicher-Feedback

**Datum**: 2026-05-31

**Ziel**: Die Kern-Interaktion des Readers — Text markieren — funktioniert für normale Lese-Fluss-Auswahlen wieder. Root-Cause: Markierungen werden beim Speichern in einem anderen Text-Koordinatensystem abgelegt (`selection.toString()`, mit Block-Umbrüchen) als beim Wieder-Anzeigen gesucht wird (`readerRawText`, Roh-Konkatenation ohne Umbrüche). Block-übergreifende Auswahlen (über Absatz-/Listen-/Überschriften-Grenzen) werden nie wiedergefunden → unsichtbare Markierung, „Formatierungsgrenze"-Toast, kein Hinscrollen, angehängte Tags nur auf einer leicht übersehbaren Sidebar-Card. Plus: Speicher-Feedback so schärfen, dass nie wieder „keine Ahnung ob gespeichert" entsteht.

**Vorbedingung**: HEAD `453bd4e`, lokal+remote synchron. Pytest **151/151 grün**. R2-B ☑ done. Dieser Sprint ist ein **Hot-Fix vor R2-C** — User-Live-Test (2026-05-31) hat den Bug aufgedeckt. **Frontend-only**: kein Schema-, kein Backend-, kein Test-Suite-Touch erwartet → **pytest bleibt 151/151**, die Korrektheits-Gate ist **Live-Smoke mit echten Maus-Drag-Auswahlen** (siehe Methodik-Hinweis).

**Out-of-scope** (Scope-Creep-Schutz):
- **Dokument-Tag vs. Markierungs-Tag-Verwirrung** (Finding 4 aus dem Live-Test: zwei Tag-Systeme nebeneinander — Filter-Chips = Conversion-Tags, Popover-Tags = Highlight-Tags) — eigener UX-Schnitt, nur in BACKLOG loggen.
- **R2-C Lifecycle** — Folge-Sprint.
- **Playwright-UI-Test-Framework einführen** (deferred L) — dieser Bug ist das stärkste Argument dafür, aber das Framework-Setup ist ein eigener Sprint. Hier reicht rigoroser manueller Smoke.
- **Reading-Progress / Filter / Tag-Manager** — alles R2-B, fertig, nicht anfassen.

---

## Root-Cause-Analyse (Master, code-gelesen 2026-05-31)

Alle Anker in [static/js/library_detail.js](static/js/library_detail.js):

1. **Speichern** (`saveCurrentSelection` Z.600): `const exact = sel.toString()` (Z.603). Bei Auswahl über Block-Grenzen enthält `selection.toString()` **Zeilenumbrüche** (`\n`) und ggf. kollabierte Whitespace — browser-/layout-abhängig.
2. **Kontext** (`extractSelectionContext` Z.579): berechnet `preLen` = Roh-Text-Start-Offset durch Text-Node-Walk bis `range.startContainer` — also bereits im `readerRawText`-Koordinatensystem. `prefix`/`suffix` kommen aus `readerRawText` (Z.595/596). **Aber `exact` kommt aus `selection.toString()`** → Anker ist halb im einen, halb im anderen System.
3. **Wieder-Anzeigen** (`locateHighlightOffset` Z.737): sucht `exact` via `fullText.indexOf(exact)` in `readerRawText(reader)` (Z.571 — purer `nodeValue`-Concat, **keine** Block-Umbrüche).
4. → `indexOf` findet das umbruch-behaftete `exact` nicht → `-1` → `applyHighlight` (Z.880) returnt `false` → `crossFormatHighlightIds.add` (Z.629/653) → kein Span, Warning-Toast (Z.636), `scrollToHighlight` scheitert (Z.722).

**Warum READER-FIX-A das nicht fing**: der Multi-Node-Wrap (Z.842) repariert nur das *Wrappen* über **Inline**-Grenzen (`<strong>`/`<em>`, gleicher Block → kein `\n` in `toString()`, `exact` matcht weiter). **Block**-Grenzen brechen eine Stufe früher beim *Finden* — unberührt. Zusätzlich liefen die READER-FIX-A-Smokes über synthetische DevTools-Ranges, die das `selection.toString()`-Umbruch-Verhalten echter Maus-Drags nicht reproduzieren → „verschwunden" war ein Test-Artefakt.

---

## Methodik-Hinweis (verpflichtend, gilt für Phase 1 + Phase 3)

**Smoke ausschließlich mit echten Maus-Drag-Auswahlen** (Chrome-MCP `left_click_drag` / echte Maus über die Reader-Text-Knoten), **nicht** mit `document.createRange()` oder programmatisch gesetzten Selections. Synthetische Ranges reproduzieren das `selection.toString()`-Verhalten nicht und haben genau diesen Bug maskiert. Wo eine Auswahl beschrieben ist, ziehe sie als echten Drag.

---

## Phase 0 — entfällt

Root-Cause ist code-gelesen und im Master-Turn bestätigt, Fix-Richtung prescribed, Scope vom User entschieden (Anker-Fix + Feedback-Politur). **Direkt Phase 1.** Die empirische Prüfung passiert als Repro in Phase 1, nicht als Phase-0-Audit.

---

## Phase 1 — Repro + Root-Cause-Anker-Fix

Pre-Flight: `pytest tests/` grün (151).

### 1.1 — Repro zuerst (kein Edit)

Container auf `localhost:5656` (Mac-Dev-Stack). Öffne ein Doc mit mehreren Absätzen. Zieh eine **echte Maus-Auswahl über eine Absatz-Grenze** (Ende Absatz 1 → Anfang Absatz 2). Erwartung: „Formatierungsgrenze"-Toast, kein Span. Belege in der Konsole die Divergenz:
- `window.getSelection().toString()` → enthält `\n`.
- der gleiche Bereich in `readerRawText`-Terms → ohne `\n`.

Berichte die **tatsächlich beobachtete** Divergenz (Umbruch? kollabierter Whitespace? NBSP?) — daran wird der Fix gemessen, nicht an der Master-Hypothese. Falls die Divergenz anders ist als „Block-Umbruch": melden, dann justieren wir.

### 1.2 — Fix (Muss-Teil: Save-Koordinaten-Konsistenz)

**Erwartete Files**: `static/js/library_detail.js` (alleiniger Code-Touch des Anker-Fixes).

- `extractSelectionContext` (Z.579) so erweitern, dass es **Start- UND End-Offset** im `readerRawText`-System liefert (aktuell wird nur bis `range.startContainer` gewalkt). Technik: Text-Node-Walk, beim Treffen von `range.startContainer` `+= range.startOffset` (→ `rawStart`), beim Treffen von `range.endContainer` `+= range.endOffset` (→ `rawEnd`).
- **`exact` aus `readerRawText` slicen**, nicht aus `selection.toString()`: `exact = fullText.slice(rawStart, rawEnd)`. `prefix = fullText.slice(rawStart - HIGHLIGHT_CONTEXT_LEN, rawStart)`, `suffix = fullText.slice(rawEnd, rawEnd + HIGHLIGHT_CONTEXT_LEN)`. Damit ist der gespeicherte Anker **byte-identisch** zu dem, was `locateHighlightOffset` durchsucht — ein Koordinatensystem.
- `saveCurrentSelection` (Z.600): `exact` aus dem neuen `extractSelectionContext`-Rückgabewert beziehen statt `sel.toString()`. Die Längen-Prüfung (Z.605, `HIGHLIGHT_EXACT_LIMIT`) gegen das neue `exact` laufen lassen. POST-Body (Z.617) unverändert in der Form `{exact, prefix, suffix}`.
- Ergebnis: block-übergreifende Auswahlen re-locaten jetzt, und der **vorhandene** READER-FIX-A-Multi-Node-Wrap (`wrapSelectionAsHighlight` Z.842) rendert sie als mehrere Spans. `crossFormatHighlightIds` wird für echte Auswahlen real leer.

### 1.3 — Backward-Compat (Soll-Teil: Rescue alter Highlights)

Bestehende Highlights in Olivers DB sind mit `selection.toString()` (umbruch-behaftet) gespeichert — die würden sonst kaputt bleiben. **Best-effort-Fallback** in `locateHighlightOffset` (Z.737):
- Fast-Path zuerst: `indexOf(exact)` exakt (deterministisch für neue Anker — unverändert).
- Bei Miss: **Whitespace-toleranter Fallback** — normalisierten `fullText` (Whitespace-Runs → ein Space, getrimmt) mit einer Roh-Offset-Index-Map bauen, `exact` gleich normalisieren, suchen, Position auf Roh-Offsets zurückmappen, dann normal weiter mit Prefix/Suffix-Scoring.
- **Priorität**: Wenn der Fallback in Phase 1 fragil/aufwändig wird (Index-Mapping-Edge-Cases), **descope** ihn — der Save-Koordinaten-Fix (1.2) ist das Muss, alte un-rescue-bare Highlights kann Oliver löschen + neu ziehen (funktionieren dann sofort). Dann Follow-up-Item statt Hängenbleiben. Melde die Entscheidung.

### 1.4 — Display-Konsistenz

`renderHighlightList` zeigt `h.exact` in der Card (`exactEl.textContent` Z.681). Roh-`exact` über Block-Grenzen kann Text ohne sichtbaren Umbruch mashen. **Nur für die Anzeige** optional Whitespace kollabieren (`.replace(/\s+/g, ' ')`) — **nie** den gespeicherten Such-Key anfassen. Klein, optional.

### Quality-Gates Phase 1
- `pytest tests/` grün (unverändert 151 — Frontend-only).
- Keine neuen `_utils.js`-Helper für Single-Call-Site-Logik (Memory `feedback_helper_reuse_design_choice.md`).
- UI-Strings deutsch.

### Verify Phase 1 (echte Maus-Drags!)
Smoke-Matrix, jede Auswahl als **echter Drag**, jeweils: Span(s) erscheinen? Überleben Reload? Aus der Sidebar anspringbar?
1. Innerhalb eines Absatzes (Single-Node) → 1 Span (R1-B-A-Regression).
2. Über Inline-Format (`<strong>`/`<em>` mitten im Absatz) → N Spans (READER-FIX-A-Regression).
3. **Über Absatz-Grenze** → N Spans, sichtbar, anspringbar (DER FIX).
4. Über Überschrift-Grenze (Absatz ↔ H2).
5. Über Listen-Items (`<li>` → `<li>`).
6. In/aus Blockquote oder Code-Block.
7. **Reload** → 3–6 re-applien und rendern (der Locate-Pfad — hier biss der Bug).
8. Alte (pre-Fix) Cross-Format-Highlights in Olivers DB → vom Fallback gerettet, oder klar als löschen+neu-ziehen dokumentiert.
9. Degenerierte Auswahl (nur Whitespace) → graceful Cross-Format, klares Feedback.

**Commit** (plain-prose, mehrere `-m`, Co-Authored-By), z.B.
`git commit -m "READER-FIX-B-anchor: exact im readerRawText-Koordinatensystem speichern" -m "Block-uebergreifende Markierungen re-locaten und rendern via Multi-Node-Wrap, whitespace-toleranter Locate-Fallback rescued Alt-Highlights, Cross-Format real leer fuer echte Auswahlen" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`
Push direkt. Bei Push-Auth-Block (macOS-Keychain) → an Master melden (Master-Session kann pushen).

**STOP — Bericht an Master (inkl. der in 1.1 beobachteten Divergenz + Fallback-Entscheidung). Nicht in Phase 2 bis Sign-off.**

---

## Phase 2 — Speicher-Feedback-Politur

Pre-Flight: `pytest tests/` grün.

**Erwartete Files**: `static/js/library_detail.js` (ggf. `static/css/style.css` für Card-Expand-Persist-Styling).

Leitprinzip: **Der User darf nach keiner Highlight-Aktion raten, ob sie gespeichert wurde.** Konkret:

1. **Tag-Add bestätigt auf allen Pfaden** (`addTagToHighlight` Z.996): aktuell Toast nur bei `201` (Z.1035). Bei `200` (idempotenter Re-Add) ist es still → „keine Ahnung". Auch bei 200 bestätigen (z.B. „Tag bereits vorhanden." oder schlicht „Tag hinzugefügt."). DE, max 2 Sätze.
2. **Card-Expand-State über Re-Render erhalten** (`renderHighlightList` Z.658 resettet `data-expanded='false'` Z.673): wenn der User in einer expandierten Sidebar-Card einen Tag hinzufügt/entfernt, klappt die Card durch den Re-Render zu und der frische Tag „verschwindet" optisch. Fix: Expand-State der **aktiven** Highlight-Card (`activeHighlightId`) über den Re-Render hinweg bewahren (vor Re-Render expandierte IDs merken, danach wiederherstellen — oder die aktive Card re-expanden). Genau Olivers „Tag wird nicht angezeigt"-Erlebnis.
3. **Feedback-Audit aller Mutations-Pfade**: `saveCurrentSelection` (create), `addTagToHighlight`, `removeTagFromHighlight` (Z.1040), `saveHighlightNote` (Z.1064), `deleteActiveHighlight` (Z.1093) — jede Aktion hat ein **sichtbares** Resultat (Span, Card-Update, oder Toast). Create bleibt bewusst still **wenn** der gelbe Span sichtbar ist (Reader-Mode-Ruhe, Readwise-Stil — kein Toast-Spam). Nach dem Anker-Fix ist der Span sichtbar → Create-Feedback ist gelöst; **nicht** zusätzlich toasten.
4. **Cross-Format-Residual klarer** (jetzt selten): eine frisch erzeugte echte Cross-Format-Card ist der einzige Ort für diese Markierung + ihre Tags. Damit sie nicht übersehen wird: die frische Cross-Format-Card auto-expanden (oder sichtbar markieren), damit der User sieht, dass die Markierung gelandet ist. Der Warning-Toast (Z.636) bleibt.

### Quality-Gates + Verify Phase 2
- `pytest tests/` grün (151).
- Smoke (echte Drags + Klicks): Tag-Add an sichtbarer Markierung → Tag erscheint sofort auf der Card, Card **bleibt offen**, Toast da; Re-Add desselben Tags → Bestätigung (nicht still); Tag-Remove/Note-Save/Delete → jeweils sichtbares Feedback; eine (provozierte) echte Cross-Format-Markierung → Card auto-expanded + Warning sichtbar.
- **Commit** (plain-prose, mehrere `-m`), z.B. `READER-FIX-B-feedback: Tag-Add auf allen Pfaden bestaetigen, Card-Expand-State ueber Re-Render erhalten, Cross-Format-Residual-Card auto-expand`. Push direkt.

**STOP — Bericht an Master. Nicht in Phase 3 bis Sign-off.**

---

## Phase 3 — Verify + Doc-Korrektur + Abschluss

1. `pytest tests/` final grün (151).
2. **Voller Smoke-Matrix-Durchlauf** (Phase-1-Matrix 1–9) **mit echten Drags**, plus Phase-2-Feedback-Checks — am Stück, als Abnahme.
3. **Doc-Korrektur** (wichtig — die „verschwunden"-Überclaims richtigstellen):
   - **STATUS.md**: READER-FIX-B als neuer „Aktueller Sprint"-Block mit Commit-Hashes; R2-B zu „Vorgänger". Klarstellen, dass Cross-Format für Block-Grenzen bis hierher offen war.
   - **BACKLOG.md**: READER-FIX-B ☑ done 2026-05-31 im „Erledigt"-Block; **Finding 4** (Dokument-Tag vs. Markierungs-Tag-Verwirrung) als neues **P3**-UX-Item eintragen; falls Fallback (1.3) descoped → „Alt-Highlight-Rescue / Re-Anchor" als P3-Follow-up; R2-C bleibt nächstes P1.
   - **docs/reader_architecture.md**: Decision-Log-Eintrag „Highlight-Anker in EINEM Koordinatensystem (`readerRawText`), nie `selection.toString()` als Such-Key" + Korrektur-Notiz, dass die READER-FIX-A-„Cross-Format-verschwunden"-Bewertung nur Inline-Grenzen galt und auf synthetischen Smokes beruhte.
4. **Memory — Eintrag anlegen** (verallgemeinerbar, hoher Wert):
   - `feedback_*`-Eintrag: **(a)** Anker für Selection/Annotation-Features müssen Save und Locate im selben Text-Koordinatensystem verankern — `selection.toString()` ≠ Roh-`nodeValue`-Concat (Block-Umbrüche, Whitespace-Normalisierung). **(b)** Selection-Features **immer mit echter Maus-Drag** smoken — synthetische DevTools-Ranges maskieren das `selection.toString()`-Verhalten (genau dieser Bug überlebte READER-FIX-A so). MEMORY.md-Pointer ergänzen. Verlinken mit `reference_dom_range_walking.md` (Wrap-Mechanik bleibt korrekt — der Fehler lag im Locate/Save, nicht im Wrap).
5. Push bestätigen (alle READER-FIX-B-Commits auf `origin/main`).

**STOP — Schluss-Bericht an Master.**

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute. Bei „mach jetzt einfach"/Frust: einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — Frontend-only, aber heikel: Anker-Koordinaten-Logik ist subtil, Backward-Compat-Fallback mit Index-Mapping, Feedback-Politur über mehrere Pfade, und eine **rigorose Live-Smoke-Matrix mit echten Drags** (die Korrektheits-Gate, da pytest die JS-Ebene strukturell nicht abdeckt). Kein Schema, kein Backend, pytest unverändert 151. Zwei Commits (Anker, Feedback).

---

## Konstitutiv mit-genommen, falls berührt

- **Finding 4 in BACKLOG loggen** (Dokument-Tag vs. Markierungs-Tag) — kein Fix hier, nur Erfassung, damit der Faden nicht verloren geht.
- **Korrektur der „Cross-Format-verschwunden"-Claims** in STATUS/reader_architecture — der Sprint berührt genau diese Stelle, also gleich richtigstellen.

Alles andere bleibt liegen (R2-C, MAC1-FOLLOWUPs, restliche P3-Smokes, `.pyc`-Hygiene, Dual-Reset-Polish).

---

## BACKLOG- und STATUS-Updates nach Abschluss

- ✓ Sprint READER-FIX-B durch (2026-05-31), zwei Commit-Hashes.
- 📋 Finding 4 (Tag-System-Verwirrung, P3 UX); ggf. Alt-Highlight-Rescue (P3) falls Fallback descoped; ggf. Playwright-UI-Test-Priorität anheben (dieser Bug als Beleg).
- STATUS.md / BACKLOG.md / reader_architecture.md wie in Phase 3.
