# UX-Patterns + Microcopy: markdown_converter (2026-05-10)

**Methodik:** Stufe 3 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Konkrete Patterns + DE-Microcopy auf Basis der Heuristik-Findings aus Stufe 2. Konsolidiert die 15 Stufe-2-Findings + 5 Bug-Tickets auf 13 Pattern-Blöcke (2 konsolidiert + 11 einzeln). Bug-Tickets BT1 und BT2 sind in den Patterns ihrer verknüpften Findings (BT1 ↔ P8, BT2 ↔ P9) mit-adressiert; BT3, BT4, BT5 sind pure Bug-Tickets ohne UX-H-Komponente und nicht Teil von F-5.3 (Sammel-Bug-Pass oder mit-genommen wenn nahegelegene Patterns berührt werden).
**Quelle Findings:** [docs/ui_findings_markdown_converter_2026-05.md](ui_findings_markdown_converter_2026-05.md)
**Quelle Inventur:** [docs/ui_inventory_markdown_converter_2026-05.md](ui_inventory_markdown_converter_2026-05.md)
**F-1 / F-2 / F-3 / F-4 Patterns als Referenz:** [docs/ui_patterns_document_converter_2026-05.md](ui_patterns_document_converter_2026-05.md), [docs/ui_patterns_audio_converter_2026-05.md](ui_patterns_audio_converter_2026-05.md), [docs/ui_patterns_library_detail_2026-05.md](ui_patterns_library_detail_2026-05.md), [docs/ui_patterns_podcast_flow_2026-05.md](ui_patterns_podcast_flow_2026-05.md)
**Helper-API:** [static/js/_utils.js](../static/js/_utils.js) — `safeJSON(response)`, `fallbackCopyText(text)`, `showAlert(containerEl, level, msg, options?)`, `showToast(msg, options?)`, `formatFileSize(bytes)`, `formatDatetimeLocalNow()`, `confirmIfLong(text, msg, options?)`. CSS-Utility `.sr-only` aus [static/css/style.css:996](../static/css/style.css#L996). Server-side: `file_size`-Jinja-Filter aus F-3-IMPL.
**Komponenten-Basis:** Existierende Neomorphism-Klassen aus [static/css/style.css](../static/css/style.css) — `c-btn`, `c-btn--primary`, `c-input`, `c-card`, `c-alert--danger/success/warning/info` (mit Close-Button + Auto-Dismiss aus F-1 Cluster C), `.toast-notification`, `.save-library-btn`, `.saved`, `.hidden`, `.sr-only`, `:focus-visible` für `c-btn`. Banner-Container `.editor-pane .px-6.pt-4` ist heute schon der Mountpoint für die File-Extension-Pre-Check-Microcopy ([static/js/markdown_converter.js:316-322](../static/js/markdown_converter.js#L316-L322)) — wird in P1 / P12 als Standard-Container für alle Markdown-Page-Banner eingesetzt.

**Microcopy-Regeln:** Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern, Deutsch durchgängig (Du-Form analog F-1 / F-2 / F-3 / F-4).
**Aufwand-Skala:** XS / S / M / L (Daumenregel: XS = 1–3 Zeilen, S = ein Handler-Cluster + Microcopy-Sweep, M = neue Mechanik mit State-Refactor, L = Cross-Stack-Refactor).
**Impact-Score-Formel:** `Score = Sev × 5 / Aufwand-Gewicht` mit Aufwand-Gewichten XS=1, S=2, M=4, L=8. Höher = besser. Bei konsolidierten Patterns wird die höchste Sev der adressierten Findings genommen (analog F-1.3 / F-2.3 / F-3.3 / F-4.3).

**Schwester-Feature-Übernahme aus F-1.3 (zentrale Mechanik dieses Sprints):** F-5.2 hat eine 86% Pattern-Konvergenz-Quote zu F-1 ausgewiesen — die höchste in der gesamten UX-Cascade. Sieben der 15 Findings (47% Cross-Feature-H4-Quote) haben eine direkte F-1.3-Pattern-Korrespondenz. F-5.3 übernimmt diese Patterns 1:1, **passt nur drei Felder an** (Code-Anker auf `markdown_converter`-Code, Microcopy wo markdown-spezifisch nötig, Adressiert-Findings auf F-5.2-Nummern). Aufwand wird übernommen außer es gibt einen klaren Begründungs-Grund für Abweichung. F-1-Pattern-Identität bleibt in jeder Pattern-Block-Header-Zeile sichtbar (Feld „F-1-Korrespondenz"). Die Patterns mit F-1-Korrespondenz sind P1 (F-1.3 P4), P2 (F-1.3 P7), P3 (F-1.3 P12-Adaption), P4 (F-1.3 P13), P12 (F-1.3 P1-teil), P13 (F-1.3 P9-teil).

**Live-Verifikation-Konvention** (NEU für F-5.3 wegen Master-Annotation 3 — kein Master-Walkthrough vor F5-PATTERNS): Patterns für die 8 ⚠️ code-only-Findings (F3, F5, F6, F7, F8, F11, F12, F15) tragen den Sub-Tag `🔥 Smoke-Pflicht in F5-IMPL` mit explizitem Smoke-Mechanik-Hinweis pro Pattern. F5-IMPL-Sub-Thread verifiziert vor Pattern-Apply per Live-Smoke (DevTools-Network-Block, Reload-Sequenz, Theme-Toggle-Reihenfolge). Wenn der Smoke zeigt, dass der Befund nicht reproduzierbar ist: Pattern-Apply STOP, Master fragen.

**Reader-Mode-Default-Wahl-Konvention** (Master-Annotation 4): Patterns für F8, F9, F11, F12 enthalten konkrete pragmatische Default-Mechanik (nicht Variante-A/B/C-Liste). „Master-Default-Wahl"-Marker im Pattern-Block. Sub-Thread kann abweichen wenn Begründung gut.

---

## Pattern-Blöcke

### Pattern 1: Save-to-Library Failure-Banner statt Browser-`alert()`
**Adressiert Findings:** F1 (H4 Sev 3), F2 (H9 Sev 3)
**F-1-Korrespondenz:** F-1.3 P4 (Save-Failure Browser-`alert()` ersetzen — direkt übernommen)
**Cluster:** 1 (Cross-Feature-H4-Helper-Reuse zu F-1)
**Live-Verifikation-Status:** — (kein code-only-Marker; Save-Pfad ist live-evident)

- **Pattern:** Aus F-1.3 P4 übernommen — alle drei `alert()`-Aufrufe in `static/js/markdown_converter.js` werden auf `showAlert(container, level, message)` umgestellt. Container ist der bereits existierende `.editor-pane .px-6.pt-4`-Banner-Mountpoint, der heute nur im File-Extension-Pre-Check-Pfad genutzt wird ([static/js/markdown_converter.js:316-322](../static/js/markdown_converter.js#L316-L322)). Damit fallen alle drei alert()-Stellen (Empty-Save-Pfad, Save-Failure-Pfad, File-Extension-Submit-Fallback) auf den selben in-page Banner-Pfad. Save-Failure-Banner ist persistent (F-1 Cluster-C-Default für `danger`), bleibt sichtbar bis User schließt — beseitigt die Recovery-Lücke aus F2 (alert() verschwindet spurlos nach OK-Klick).
- **Visuelle Hinweise:** Banner oben in der Editor-Pane (existierender Container), Danger-Tint, Close-× über `showAlert`-Default. Save-Button kehrt nach Fail in den Default-State zurück (`disabled=false`, Text aus DE-Microcopy unten).
- **Microcopy** (DE, Du-Form, max 2 Sätze; Buttons max 3 Wörter):
  - Save-Failure (generisch, ersetzt heutiges `alert('Failed to save: ' + err.message)` an [markdown_converter.js:146](../static/js/markdown_converter.js#L146)): „In Library speichern fehlgeschlagen. Verbindung prüfen und erneut versuchen."
  - Save-Failure mit Server-Detail (falls `err.message` informativ): „In Library speichern fehlgeschlagen: {detail}."
  - Empty-Save-Banner (ersetzt heutiges `alert('No markdown content to save.')` an [markdown_converter.js:102](../static/js/markdown_converter.js#L102)): „Kein Inhalt zum Speichern vorhanden."
  - File-Extension-Submit-Fallback (heutiger `alert(...)` an [markdown_converter.js:321](../static/js/markdown_converter.js#L321) wird obsolet, weil der `showAlert`-Branch direkt davor schon greift; alert-Fallback komplett entfernen).
  - Save-Btn default: „In Library speichern"
  - Save-Btn loading: „Speichert …"
  - Save-Btn success: „✓ Gespeichert"
- **Helper-Reuse:** `showAlert(.editor-pane .px-6.pt-4, 'danger', msg)` aus [static/js/_utils.js:39-78](../static/js/_utils.js#L39-L78) ersetzt drei `alert()`-Calls. Helper-Reuse-Quote für dieses Pattern: 100% (kein neuer Helper). Cross-Feature-Konvergenz auf F-1.3 P4, F-2.3 P5 (Alert-Konsolidierung), F-3.3 P1/P3 (Auto-Save/Delete-Banner).
- **Aufwand:** S — drei `alert()`-Stellen 1:1 ersetzen + Empty-Save-Branch in Banner-Form bringen + DE-Microcopy. Helper schon eingebunden (`_utils.js` lädt via `base.html`), kein neuer Container nötig.
- **Impact-Score:** 3 × 5 / 2 = **7.5**
- **Konsolidierung:** F1 (H4 Konsistenz mit anderem File-Extension-Pre-Check-Banner-Pfad) und F2 (H9 Recovery — alert() verschwindet ohne Trace) entstehen aus derselben Wurzel: drei `alert()`-Calls statt in-page Banner. Eine Lösung (showAlert + persistente Banner) erfüllt beide Heuristiken. F-1.3-Konsolidierungs-Pattern (F-1.2 F7 + F8 → F-1.3 P4) wird 1:1 reproduziert.

---

### Pattern 2: Alert-Banner Auto-Dismiss für non-danger Levels
**Adressiert Findings:** F14 (H1 Sev 1)
**F-1-Korrespondenz:** F-1.3 P7 (Alert-Banner ohne Close-Button und ohne Auto-Dismiss — direkt übernommen, mit Sev-Anpassung)
**Cluster:** 1 (Cross-Feature-H4-Helper-Reuse zu F-1)
**Live-Verifikation-Status:** — (Banner-Verhalten ist live-evident)

- **Pattern:** Aus F-1.3 P7 übernommen, mit Sev-1-Anpassung — F-1.3 P7 musste Close-Button **und** Auto-Dismiss-Timer ergänzen. Hier ist der Close-`×` bereits inline im Template vorhanden ([templates/markdown_converter.html:31](../templates/markdown_converter.html#L31)), nur Auto-Dismiss-Timer fehlt. Konvergenz-Pfad: Server-side gerenderte Flash-Banner (im `c-alert`-Wrapper) bekommen einen JS-Auto-Dismiss-Hook beim Page-Load, der für non-danger Levels (`success`, `info`, `warning`) nach 6 s den Banner aus dem DOM entfernt. Danger-Banner bleiben persistent (User soll lesen).
- **Visuelle Hinweise:** Fade-out 200 ms beim Auto-Dismiss (analog `showAlert`-Default). Close-`×` bleibt als manueller Eskalations-Pfad. Position unverändert (Server-Render im Flash-Container).
- **Microcopy** (DE, Du-Form):
  - Close-Button aria-label (existiert schon im `c-alert__close`): „Meldung schließen" — aus `showAlert`-Helper-Default ([static/js/_utils.js:62](../static/js/_utils.js#L62)). Server-side gerendetes `c-alert` hat ggf. nur ein static `aria-label` — zu konsolidieren auf denselben String.
  - keine sichtbare Begleit-Microcopy.
- **Helper-Reuse:** kein neuer Helper, aber Konvention-Reuse: das Auto-Dismiss-Verhalten von `showAlert` (für non-danger 6 s, [static/js/_utils.js:43-45](../static/js/_utils.js#L43-L45)) wird auf Server-side gerenderte `c-alert`-Banner gespiegelt — kleine Hilfsfunktion `attachAutoDismissToServerBanners()` im DOMContentLoaded-Branch von `markdown_converter.js`, die alle vorhandenen `.c-alert:not(.c-alert--danger)`-Banner mit dem Timer ausstattet. Code-Anker: [templates/markdown_converter.html:13-32](../templates/markdown_converter.html#L13-L32) Flash-Loop; [static/js/markdown_converter.js:157](../static/js/markdown_converter.js#L157) `window.addEventListener('load', ...)`-Init.
- **Aufwand:** XS — kleine Init-Hilfsfunktion mit `setTimeout(banner.remove, 6000)` für non-danger Banner. F-1.3 hatte für P7 Aufwand S; hier XS, weil Close-`×` schon da ist und die Auto-Dismiss-Mechanik bereits in `showAlert` etabliert ist (Spiegelung, kein Re-Build).
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 3: File-Info-Display nach File-Auswahl
**Adressiert Findings:** F10 (H1 Sev 1)
**F-1-Korrespondenz:** F-1.3 P12 (Filename KB/MB-Fallback — Adaption: nicht Unit-Bug fixen, sondern Display einführen)
**Cluster:** 1 (Cross-Feature-H4-Helper-Reuse zu F-1)
**Live-Verifikation-Status:** — (Display-Lücke ist live-evident)

- **Pattern:** Heuristik-Verschiebung gegenüber F-1.3 P12: F-1 hatte einen MB-Display-Bug (immer-MB statt KB/MB-Fallback). Hier fehlt das File-Info-Display **insgesamt** — nach File-Auswahl wird nur der Textarea-Inhalt befüllt, kein Filename- oder Size-Indikator. Pattern: in `handleFileSelect` ([static/js/markdown_converter.js:266-275](../static/js/markdown_converter.js#L266-L275)) ein neues `#markdown-file-info`-Span unter dem File-Input befüllen mit `${file.name} (${formatFileSize(file.size)})` plus kleinem Klein-`×`-Btn zum Reset (entfernt File aus Input + leert Textarea + räumt Span ab). Damit entsteht der bisher fehlende sichtbare System-Status „ich habe X hochgeladen".
- **Visuelle Hinweise:** Span direkt unter dem File-Input in der Editor-Pane, kleine Schrift-Grad, dezent grau (`var(--nm-text-faint)`). Klein-`×`-Btn rechts in der Span-Zeile, Hover-Tint danger. Übergang `opacity` 150 ms beim Reset.
- **Microcopy** (DE, Du-Form, max 3 Wörter pro Action):
  - Span-Format: „{filename} ({formatFileSize-Output})" — z.B. „notes.md (4,6 KB)"
  - Reset-Btn aria-label: „Datei abwählen"
  - aria-live-Hint nach File-Selection (höflich): „{filename}, {size}, ausgewählt."
  - aria-live-Hint nach Reset (höflich): „Datei abgewählt."
- **Helper-Reuse:** `formatFileSize(file.size)` aus [static/js/_utils.js:82-87](../static/js/_utils.js#L82-L87). Cross-Feature-Konvergenz auf F-1.3 P12 (`document_converter`) und F-2.3 P19 (`audio_converter`). Code-Anker: [static/js/markdown_converter.js:266-275](../static/js/markdown_converter.js#L266-L275) `handleFileSelect`; [templates/markdown_converter.html](../templates/markdown_converter.html) für neuen `#markdown-file-info`-Span unter dem File-Input.
- **Aufwand:** S — neuer Span im Template + Render-Logik in `handleFileSelect` + Reset-Btn-Handler + aria-live-Region + Microcopy. F-1.3 hatte für P12 Aufwand XS (nur MB-Bug-Fix); hier S, weil das Display-Element insgesamt neu eingeführt werden muss (Heuristik-Verschiebung von H4 auf H1, siehe F-5.2).
- **Impact-Score:** 1 × 5 / 2 = **2.5**

---

### Pattern 4: Iframe-Preview a11y-Annotations
**Adressiert Findings:** F13 (H6 Sev 1)
**F-1-Korrespondenz:** F-1.3 P13 (Result-Content `<pre>` ohne a11y-Annotations — direkt übernommen, mit Iframe-Anker statt `<pre>`)
**Cluster:** 1 (Cross-Feature-H4-Helper-Reuse zu F-1)
**Live-Verifikation-Status:** — (a11y-Annotation-Lücke ist statisch verifizierbar)

- **Pattern:** Aus F-1.3 P13 übernommen, mit Iframe-Anker — `#preview-iframe` ([templates/markdown_converter.html:142](../templates/markdown_converter.html#L142)) bekommt `tabindex="0"`, `role="region"` und ein DE-`aria-label`. Heute trägt der Iframe nur `title="PDF Preview"` (≈aria-label, EN). Mit den drei Attributen ist die Live-Vorschau per Tastatur direkt fokussierbar und Screenreader-ansagbar.
- **Visuelle Hinweise:** Sichtbarer Focus-Ring entsprechend `:focus-visible` (Neomorphism-Outline-Logik analog `c-btn:focus-visible`). Keine sichtbare Microcopy.
- **Microcopy** (DE, Du-Form):
  - aria-label (ersetzt EN `title="PDF Preview"`): „PDF-Vorschau"
  - title (Hover-Tooltip, behält dieselbe Microcopy): „PDF-Vorschau"
- **Helper-Reuse:** keine. Reine Template-Attribute. Code-Anker: [templates/markdown_converter.html:142](../templates/markdown_converter.html#L142).
- **Aufwand:** XS — drei Attribute am Template. F-1.3-Aufwand 1:1 übernommen.
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 5: Reader-Mode-State-Persistenz (Mode-On/Off + Width-Initial-Active)
**Adressiert Findings:** F9 (H6 Sev 2), F11 (H6 Sev 1)
**F-1-Korrespondenz:** — (markdown-spezifisch, Reader-Mode-only)
**Cluster:** 2 (Reader-Mode-State und Visual-Layout)
**🔥 Smoke-Pflicht in F5-IMPL** (für F11-Teil)
**Master-Default-Wahl** (Master-Annotation 4)

- **Pattern:** `localStorage.readerPrefs` wird um den Reader-Mode-On/Off-State erweitert (`prefs.modeOn`). Beim DOMContentLoaded-Init prüft `markdown_converter.js`: wenn `prefs.modeOn === true`, wird `toggleReaderMode()` einmal beim Page-Load ausgeführt, sodass der Reader-Mode-Container plus Width/Font/Dark-Sub-Prefs rehydriert werden. Beim manuellen `toggleReaderMode()`-Klick wird `prefs.modeOn = isActive` gesetzt und persistiert. F11 (Width-Buttons-Initial-Active) wird durch dieselbe Mechanik gelöst: `updateWidthButtons(prefs.width || 'medium')` wird **immer** beim Init ausgeführt (nicht nur beim Reader-Toggle-Branch wie heute), sodass die Width-Buttons selbst vor erstem Reader-Toggle den korrekten `.active`-Marker tragen — auch wenn die Toolbar `display:none` ist und der Marker nur kurz im allerersten Reader-Toggle-Frame sichtbar wäre.
- **Master-Default-Wahl:** gemeinsamer `localStorage.readerPrefs`-Key (existiert schon, [markdown_converter.js:17](../static/js/markdown_converter.js#L17)) — kein neuer Key, keine Migration. Default-Width bei fehlender Pref bleibt `'medium'` (aktueller Code-Default). Bei `prefs.modeOn === undefined` (Erst-Besuch) bleibt Reader-Mode aus.
- **Visuelle Hinweise:** Übergang in Reader-Mode beim Page-Load ist ohne Animation (User soll nicht erschrecken — Mode war beim letzten Besuch aktiv, Erwartung ist „so wie ich es verlassen habe"). Width-Buttons tragen den `.active`-Marker stabil ab Init.
- **Microcopy** (DE, Du-Form):
  - keine neuen sichtbaren Strings — Mode-Toggle und Width-Buttons sind icon-/glyph-basiert, kein Text-Label-Touch.
- **Helper-Reuse:** kein neuer `_utils.js`-Helper. Lokale Erweiterung von `getReaderPrefs()` / `saveReaderPrefs()` und Init-Pfad. **Helper-Vorschlag** für `_utils.js`, falls weitere Features State-Persistenz mit demselben Muster brauchen — siehe Helper-Vorschlag-Sektion am Doc-Ende. Code-Anker: [static/js/markdown_converter.js:17-46](../static/js/markdown_converter.js#L17-L46) `READER_PREFS_KEY` + `toggleReaderMode`; [static/js/markdown_converter.js:87-92](../static/js/markdown_converter.js#L87-L92) `updateWidthButtons`; [static/js/markdown_converter.js:157](../static/js/markdown_converter.js#L157) `window.addEventListener('load', ...)`-Init.
- **Aufwand:** M — neue State-Persistenz-Mechanik mit Init-Pfad-Sweep, Mode-Toggle-Handler-Erweiterung, Width-Buttons-Init-Aufruf, Test des State-Cycles (Reload mit Mode aktiv → rehydrate; Reload mit Mode aus → bleibt aus; Width-Wechsel persistiert; Dark-Pref-Konsistenz mit P7). Schema-Touch entfällt, aber die Mechanik ist nicht trivial weil Reader-Mode-Toggle-Funktion heute nur durch User-Klick getriggert wird.
- **Impact-Score:** 2 × 5 / 4 = **2.5** (höchste Sev der adressierten Findings: F9 mit Sev 2)
- **Smoke-Mechanik (für F11-Teil):** Reader-Mode-Toggle aktivieren → Width-Button visuell mit `.active`-Marker prüfen. Page reload → Width-Button-Marker auch ohne Reader-Mode-Toggle vorhanden (für den allerersten Reader-Toggle-Frame). Test in DevTools: `localStorage.readerPrefs` Inspect.
- **Konsolidierung:** F9 (Mode-On/Off-Persistenz) und F11 (Width-Buttons-Initial-Active) gehören zur selben State-Mechanik (`readerPrefs`-Hydrate beim Init). Eine Lösung erfüllt beide. F12 (Esc-Key-Scope) bleibt als eigenständiger Pattern P6 — andere Mechanik (Listener-Scope-Anpassung statt State-Persistenz).
- **Verzahnung-Hinweis (BT5-Verdacht):** Wenn der Init-Pfad ohnehin angefasst wird, kann BT5 (toter `<link id="preview-style" href="">` an [templates/markdown_converter.html:6](../templates/markdown_converter.html#L6)) mit-aufgeräumt werden — das tote `<link>` ist unmittelbar oberhalb des Init-Code-Touchs sichtbar. F5-IMPL entscheidet ob mit-fixen oder separater Bug-Pass.

---

### Pattern 6: Esc-Key-Listener auf Reader-Mode-Container scopen
**Adressiert Findings:** F12 (H6 Sev 1)
**F-1-Korrespondenz:** — (markdown-spezifisch, Reader-Mode-only)
**Cluster:** 2 (Reader-Mode-State und Visual-Layout)
**🔥 Smoke-Pflicht in F5-IMPL**
**Master-Default-Wahl** (Master-Annotation 4)

- **Pattern:** Heutiger `document.addEventListener('keydown', ...)` ([static/js/markdown_converter.js:94-98](../static/js/markdown_converter.js#L94-L98)) löst Reader-Mode-Exit unabhängig vom Fokus aus. **Master-Default-Wahl:** Listener bleibt auf `document`, aber der Handler prüft zusätzlich, ob der Fokus auf einem Form-Control liegt (`document.activeElement` ist `TEXTAREA`, `INPUT`, `SELECT` oder `BUTTON`). Wenn ja, wird Esc nicht als Reader-Exit interpretiert — User kann Browser-Autocomplete schließen oder native Field-Behavior nutzen, ohne aus dem Reader-Mode rauszufallen. Wenn der Fokus außerhalb von Form-Controls ist (z.B. auf Body oder im Iframe), bleibt der Esc-Reader-Exit-Pfad.
- **Master-Default-Wahl:** `activeElement.tagName`-Check statt Container-Scope-Listener — einfacher Fix, kein Listener-Refactor, kein Container-Anker-Wechsel. Trade-Off: User der mit Esc explizit aus dem Iframe-Fokus den Reader-Mode verlassen will, muss erst aus dem Iframe heraus tabben (1 zusätzlicher Tastendruck). Akzeptabel weil seltener als der Textarea-Fokus-Fall.
- **Visuelle Hinweise:** keine — reine Listener-Logik.
- **Microcopy** (DE, Du-Form):
  - keine neuen Strings. Reader-Mode-Toggle-Btn (#12) trägt heute schon einen Tooltip — Sub-Thread kann beim DE-Pass mit-prüfen, ob der Tooltip „Reader Mode (Esc to exit)" auf „Reader-Modus (Esc zum Beenden)" deutsch sollte; das ist DE-Pass-mit-genommen, kein eigener Microcopy-Block.
- **Helper-Reuse:** keine. Code-Anker: [static/js/markdown_converter.js:94-98](../static/js/markdown_converter.js#L94-L98) Esc-Listener.
- **Aufwand:** XS — eine zusätzliche Bedingung im Esc-Handler (`if (formControlTags.includes(document.activeElement.tagName)) return;`). Drei Code-Zeilen.
- **Impact-Score:** 1 × 5 / 1 = **5.0**
- **Smoke-Mechanik:** Reader-Mode aktivieren → Textarea fokussieren → Esc drücken → erwartet: Textarea-Fokus bleibt, Reader-Mode bleibt aktiv. Reader-Mode aktivieren → Body fokussieren (z.B. Click auf Reader-Mode-Toolbar-Hintergrund) → Esc drücken → erwartet: Reader-Mode wird verlassen.

---

### Pattern 7: Two-Dark-Modes-Konsolidierung (Reader-Dark dominiert wenn Reader aktiv)
**Adressiert Findings:** F8 (H4 Sev 2)
**F-1-Korrespondenz:** — (markdown-spezifisch, intern H4-Bruch)
**Cluster:** 2 (Reader-Mode-State und Visual-Layout)
**🔥 Smoke-Pflicht in F5-IMPL**
**Master-Default-Wahl** (Master-Annotation 4)

- **Pattern:** Heute existieren zwei voneinander unabhängige Dark-Pfade: globaler `data-global-theme` (Theme-Toggle Layout-#1) und Reader-Mode-only `data-theme` (Reader-Dark-Btn #36). `isDarkActive()` ([static/js/markdown_converter.js:178-182](../static/js/markdown_converter.js#L178-L182)) checkt beide, aber das Setzen ist asymmetrisch (Edge-Case: globaler Theme dunkel → Reader-Mode betreten → erneut Dark applied; Reader-Mode verlassen → `data-theme` entfernt, aber `data-global-theme="dark"` bleibt). **Master-Default-Wahl:** Reader-Dark übersteuert Theme-Dark wenn Reader aktiv (lokaler Scope-Sieg). Konkrete Mechanik:
  - `applyDarkMode(on)` setzt nur `data-theme` (Reader-Scope), nicht den globalen Theme-Pfad an. Globaler Theme-Toggle bleibt unverändert (greift `data-global-theme`).
  - `isDarkActive()` bleibt der Single-Source-of-Truth: liefert `true` wenn entweder `data-theme === 'dark'` ODER `data-global-theme === 'dark'`. Damit ist die Iframe-Render-Entscheidung konsistent.
  - Beim `toggleReaderMode()`-Exit ([markdown_converter.js:42-43](../static/js/markdown_converter.js#L42-L43)) wird `data-theme` entfernt — Reader-Scope-Dark fällt ab, globaler Theme-Dark bleibt sichtbar wenn er aktiv war. Damit kein „Dark-bleibt-hängen"-Bruch.
- **Master-Default-Wahl:** Reader-Dark als lokaler Scope-Sieg über Theme-Dark wenn Reader aktiv; beim Exit fällt der lokale Scope sauber ab. Trade-Off: User der globalen Dark-Theme nutzt und Reader-Mode betritt, kann den Reader-Light-Mode trotzdem aktivieren (Reader-Scope übersteuert) — gewollt, weil Reader-Mode eine Lese-Optimierung ist und User dort eine andere Tönung wählen darf.
- **Visuelle Hinweise:** Iframe-Render reagiert via `MutationObserver` auf beide Attribute ([markdown_converter.js:247-251](../static/js/markdown_converter.js#L247-L251)) — bleibt unverändert. Dark-Mode-Btn-Glyph (`☀️` / `🌙`) reflektiert `data-theme`-State (Reader-Scope), nicht `data-global-theme`.
- **Microcopy** (DE, Du-Form):
  - keine neuen sichtbaren Strings. Dark-Mode-Btn-Glyph bleibt unverändert.
- **Helper-Reuse:** keine. Code-Anker: [static/js/markdown_converter.js:48-61](../static/js/markdown_converter.js#L48-L61) `toggleDarkMode` + `applyDarkMode`; [static/js/markdown_converter.js:178-182](../static/js/markdown_converter.js#L178-L182) `isDarkActive`.
- **Aufwand:** S — `applyDarkMode` und `toggleReaderMode` Exit-Branch verifizieren, dass `data-theme` korrekt entfernt wird; `isDarkActive` bleibt unverändert (ist schon der gewünschte Single-Source-of-Truth-Read). Mögliche zweite Stelle: globaler Theme-Toggle-Code (im `base.html`-Layout) — der setzt `data-global-theme` und hat keinen Reader-Scope-Konflikt.
- **Impact-Score:** 2 × 5 / 2 = **5.0**
- **Smoke-Mechanik:** Sequenz 1: Globaler Theme-Toggle (oben rechts) auf Dark → Reader-Mode betreten → Reader-Dark-Btn klicken → erneutes Reader-Light-Klick → erwartet: Reader-Mode-Iframe-Render zeigt Light, globaler Theme-Toggle-State unverändert. Sequenz 2: Globaler Theme-Toggle auf Dark → Reader-Mode betreten → Reader-Mode verlassen → erwartet: globaler Dark-Theme bleibt aktiv, kein Reader-Scope-Restbestand.
- **Verzahnung-Hinweis (BT3-Verdacht):** Wenn `applyDarkMode` ohnehin angefasst wird, kann BT3 (`updateStyle()` doppelt beim Page-Load, [markdown_converter.js:281+289](../static/js/markdown_converter.js#L281)) im selben Init-Code-Touch mit-aufgeräumt werden — der zweite `updateStyle()`-Call ist unmittelbar nach dem Theme-Init-Branch sichtbar. F5-IMPL entscheidet ob mit-fixen oder separater Bug-Pass.

---

### Pattern 8: PDF-Generation-Error-Recovery via flash + redirect
**Adressiert Findings:** F3 (H9 Sev 3 ⚠️ code-only)
**Adressiert Bug-Ticket:** BT1 (PDF-Generation-Failure-Re-Render unvollständiger Template-Context)
**F-1-Korrespondenz:** — (markdown-spezifisch, Backend-Error-Recovery-Pfad)
**Cluster:** 3 (Error-Recovery-Pfade)
**🔥 Smoke-Pflicht in F5-IMPL**

- **Pattern:** Aus Master-Annotation 2 ausgeformt. Statt `render_template('markdown_converter.html', markdown_text=markdown_text)` im Error-Branch von [app_pkg/markdown.py:246](../app_pkg/markdown.py#L246) wird Flask `flash(<DE-Microcopy>, 'danger')` + `redirect(url_for('markdown_converter'))` verwendet. User landet auf der Markdown-Seite mit sichtbarem Danger-Banner aus dem bestehenden Flash-Render-Loop ([templates/markdown_converter.html:13-32](../templates/markdown_converter.html#L13-L32)) statt 500-Page. BT1 (fehlender `themes`/`accepted_extensions`/`accepted_extensions_accept`-Context, der den Re-Render in einen secondary 500 fallen lassen würde) wird durch dieses Pattern strukturell aufgelöst — der Re-Render-Pfad fällt komplett weg, weil `redirect` einen frischen GET-Request mit vollständigem Context auslöst.
- **Visuelle Hinweise:** Banner oben in der Editor-Pane (existierender Flash-Container), Danger-Tint, Close-`×`. Markdown-Text aus dem fehlerhaften Submit ist nicht im Textarea (redirect statt re-render) — User muss den Text aus History/Browser-Back wiederherstellen, falls er ihn noch braucht. Trade-Off ist akzeptabel weil Recovery-Frequenz niedrig (PDF-Gen-Failure ist selten) und der Verlust von User-Input ein bekannter Trade-Off bei flash+redirect-Patterns.
- **Microcopy** (DE, Du-Form, max 2 Sätze; ohne Emoji bei Fehlern):
  - PDF-Gen-Failure-Banner (ersetzt heutigen `flash(f'Error: Could not generate PDF: {e}', 'danger')` an [app_pkg/markdown.py:245](../app_pkg/markdown.py#L245)): „PDF-Erstellung fehlgeschlagen. Bitte erneut versuchen oder eine andere Vorlage wählen."
  - Optional mit Server-Detail (falls Logs zeigen, dass `e.message` informativ ist und nicht nur ein Stack-Trace): „PDF-Erstellung fehlgeschlagen: {detail}."
- **Helper-Reuse:** Flask `flash()` + bestehender Banner-Render im Template ([templates/markdown_converter.html:13-32](../templates/markdown_converter.html#L13-L32)). Bei P2-Apply trägt der Flash-Banner Auto-Dismiss für non-danger Levels — `danger` bleibt persistent (Master-Annotation-konform). Code-Anker: [app_pkg/markdown.py:245-246](../app_pkg/markdown.py#L245-L246).
- **Aufwand:** XS — 1-2 Zeilen Backend-Refactor (`render_template(...)` → `redirect(url_for('markdown_converter'))`), kein Frontend-Touch. Heutiger `flash(...)` bleibt — nur der `render_template`-Pfad wird durch `redirect` ersetzt.
- **Impact-Score:** 3 × 5 / 1 = **15.0**
- **Smoke-Mechanik:** PDF-Gen-Failure forcieren — z.B. via Theme-Datei-Manipulation (`/static/css/pdf_styles/default.css` temporär durch ungültiges CSS ersetzen, das Playwright crashen lässt) oder via Playwright-Mock (`async_playwright`-Singleton in `app.py` durch einen Throw-Mock ersetzen). Page-Re-Render beobachten — sollte Banner-Page mit DE-Microcopy zeigen statt 500 mit Jinja-`UndefinedError`.
- **Verzahnung-Hinweis:** BT1 wird durch dieses Pattern aufgelöst, kein separater Bug-Ticket-Apply nötig. Der Re-Render-Pfad fällt komplett weg.

---

### Pattern 9: Sample-Markdown-Template-Bug-Fix (Sample außerhalb User-Default)
**Adressiert Findings:** F6 (H1 Sev 2 ⚠️ code-only)
**Adressiert Bug-Ticket:** BT2 (Sample-Text-Merge-Template-Bug)
**F-1-Korrespondenz:** — (markdown-spezifisch, Template-Bug)
**Cluster:** 3 (Error-Recovery-Pfade)
**🔥 Smoke-Pflicht in F5-IMPL**

- **Pattern:** Heutiges Template ([templates/markdown_converter.html:43-69](../templates/markdown_converter.html#L43-L69)) hat Sample-Markdown **vor** `{{ markdown_text or '' }}` direkt im Textarea-Body. Wenn der Server-Re-Render-Pfad mit User-Input feuert (Befund 16), entstand User-Inhalt **angehängt an das Sample**. Dieses Pattern wird durch P8 strukturell aufgelöst (kein Re-Render-Pfad mehr — flash+redirect liefert frischen Context ohne `markdown_text`-Variable). Aber für den Fall, dass künftig wieder ein `markdown_text`-Re-Render eingeführt wird (oder andere Template-Konsumenten den Wert setzen), bleibt der Bug strukturell vorhanden. Pattern: Sample-Markdown nur dann rendern wenn `markdown_text` falsy ist — `{% if not markdown_text %}{# Sample #}{% endif %}{{ markdown_text or '' }}`. Damit ist der Sample-Pfad explizit „kein User-Input vorhanden", und der User-Input-Pfad explizit „Sample wird ersetzt".
- **Visuelle Hinweise:** keine — reine Template-Logik. Initial-Page-Load (kein User-Input) zeigt Sample wie heute. User-Input-Re-Render (falls je wieder existiert) ersetzt Sample durch User-Inhalt.
- **Microcopy** (DE, Du-Form):
  - keine neuen Strings. Sample-Markdown-Inhalt bleibt unverändert (englische Anleitungs-Beispiel-Datei — Sub-Thread kann beim DE-Pass mit-prüfen, ob die Sample-Anleitung deutsch sein soll; das ist DE-Pass-mit-genommen, kein eigener Microcopy-Block).
- **Helper-Reuse:** keine. Code-Anker: [templates/markdown_converter.html:43-69](../templates/markdown_converter.html#L43-L69) Sample-Block + `{{ markdown_text or '' }}`-Stelle.
- **Aufwand:** XS — ein `{% if not markdown_text %}…{% endif %}`-Wrap um den Sample-Block. Drei Template-Zeilen.
- **Impact-Score:** 2 × 5 / 1 = **10.0**
- **Smoke-Mechanik:** Empty-Markdown-Page-Load → erwartet: Sample sichtbar. Nach P8-Apply gibt es keinen Re-Render-Pfad mehr, der den Bug live triggert; das Pattern ist strukturelle Defensive-Code-Hygiene. Optional manueller Test: temporär in `markdown.py` einen Test-Endpoint einbauen, der `render_template('markdown_converter.html', markdown_text='Test-User-Input')` aufruft → erwartet: nur „Test-User-Input" im Textarea, kein Sample-Anhang.
- **Verzahnung-Hinweis:** BT2 wird durch dieses Pattern aufgelöst, kein separater Bug-Ticket-Apply nötig. P8 löst die Live-Symptomatik strukturell, P9 schließt die Template-Lücke defensiv für künftige Re-Render-Pfade.

---

### Pattern 10: Theme-CSS-Fetch-Failure User-sichtbar machen
**Adressiert Findings:** F5 (H9 Sev 2 ⚠️ code-only)
**F-1-Korrespondenz:** — (markdown-spezifisch, Theme-CSS-Fetch)
**Cluster:** 3 (Error-Recovery-Pfade)
**🔥 Smoke-Pflicht in F5-IMPL**

- **Pattern:** `updateStyle()` in [static/js/markdown_converter.js:253-264](../static/js/markdown_converter.js#L253-L264) hat heute `.catch(err => console.error(...))` — Fetch-Failure ist silent. Pattern: `.catch`-Branch zeigt einen `c-alert--warning`-Banner via `showAlert(.editor-pane .px-6.pt-4, 'warning', msg)` mit DE-Microcopy. Preview rendert weiter (mit altem oder leerem Theme — `currentThemeCSS` bleibt was es war), aber der User sieht, dass die Theme-Auswahl tatsächlich fehlschlug und kann eine andere Vorlage wählen oder Network prüfen.
- **Visuelle Hinweise:** Banner oben in der Editor-Pane (gleicher Container wie P1 + P12), Warning-Tint, Close-`×` über `showAlert`-Default. Auto-Dismiss nach 6 s (warning-Default). Theme-Selector-Dropdown bleibt auf der gewählten (aber nicht geladenen) Option — User kann erneut wählen oder zurückwechseln.
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Theme-CSS-Fetch-Failure: „Vorlage konnte nicht geladen werden. Vorschau zeigt jetzt ohne diese Vorlage."
  - mit Theme-Name (falls verfügbar): „Vorlage „{theme}" konnte nicht geladen werden. Vorschau zeigt jetzt ohne diese Vorlage."
- **Helper-Reuse:** `showAlert(container, 'warning', msg)` aus [static/js/_utils.js:39-78](../static/js/_utils.js#L39-L78). Code-Anker: [static/js/markdown_converter.js:253-264](../static/js/markdown_converter.js#L253-L264) `updateStyle`.
- **Aufwand:** S — `.catch`-Branch erweitern (statt `console.error` → `showAlert` plus DE-Microcopy + Theme-Name-Interpolation). `currentThemeCSS = ''`-Fallback explizit setzen, damit der MutationObserver-Re-Render mit leerem Theme rendert (heute bleibt `currentThemeCSS` auf dem alten Wert, was inkonsistent zur User-Auswahl ist).
- **Impact-Score:** 2 × 5 / 2 = **5.0**
- **Smoke-Mechanik:** DevTools-Network-Tab → Block-URL für `/static/css/pdf_styles/<theme>.css` → Theme-Selector wechseln → erwartet: Banner sichtbar mit DE-Microcopy, Preview rendert mit leerem Theme (oder altem Theme), kein silent fail.

---

### Pattern 11: markdown-it CDN-Fallback-Microcopy DE-isieren
**Adressiert Findings:** F15 (H9 Sev 1 ⚠️ code-only)
**F-1-Korrespondenz:** — (markdown-spezifisch, Architektur-Dependency)
**Cluster:** 3 (Error-Recovery-Pfade)
**🔥 Smoke-Pflicht in F5-IMPL**

- **Pattern:** Heutiger CDN-Fallback-Pfad in [static/js/markdown_converter.js:158-163](../static/js/markdown_converter.js#L158-L163) prüft `typeof markdownit === 'undefined'` und rendert „Preview unavailable: markdown-it library failed to load" (EN) im Iframe. Pattern: Microcopy auf DE umstellen plus konkreter Recovery-Hinweis (Internet/Reload). Server-PDF-Pipeline arbeitet unabhängig (Python markdown-it) — der Fallback betrifft nur die Live-Preview, der „Convert to PDF"-Pfad funktioniert weiterhin. Microcopy macht das transparent.
- **Visuelle Hinweise:** Iframe-srcdoc bleibt das Render-Ziel (kein Banner-Pfad — Iframe ist Live-Preview-Only-Container, der ohnehin sichtbar ist). Style bleibt analog zum bestehenden Fallback-HTML (orangener Text-Tint `#b45309`).
- **Microcopy** (DE, Du-Form, max 2 Sätze; ohne Emoji bei Fehlern):
  - Iframe-Fallback (ersetzt heutigen EN-String): „Live-Vorschau nicht verfügbar — markdown-it konnte nicht geladen werden. PDF-Erstellung funktioniert trotzdem; Internet-Verbindung prüfen oder Seite neu laden."
- **Helper-Reuse:** keine. Code-Anker: [static/js/markdown_converter.js:158-163](../static/js/markdown_converter.js#L158-L163) Fallback-Branch.
- **Aufwand:** XS — eine String-Substitution. Drei Code-Zeilen.
- **Impact-Score:** 1 × 5 / 1 = **5.0**
- **Smoke-Mechanik:** DevTools-Network-Tab → Block-URL für `https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js` → Page reload → erwartet: Iframe zeigt DE-Fallback-Microcopy mit Recovery-Hinweis, „Convert to PDF"-Form bleibt funktional.

---

### Pattern 12: Empty-Markdown-Submit Frontend-Pre-Check
**Adressiert Findings:** F4 (H1 Sev 2)
**F-1-Korrespondenz:** F-1.3 P1 (Empty-Submit silent — teil-übertragen, mit Sev-2-Anpassung wegen Server-Flash-Roundtrip)
**Cluster:** 4 (Async-Pre-Check und Loading-Visibility)
**Live-Verifikation-Status:** — (Empty-Submit-Verhalten ist live-evident)

- **Pattern:** Aus F-1.3 P1 teil-übertragen — F-1 hatte komplett silent (Sev 4); hier macht Server-Flash-Roundtrip zwar Feedback, aber Latenz und Re-Render-Kosten sind sichtbar. Pattern: Submit-Handler in [static/js/markdown_converter.js:309-349](../static/js/markdown_converter.js#L309-L349) prüft vor dem CSRF-Refresh, ob `markdown_text.value.trim()` leer ist UND keine Datei im Input ist. Wenn ja, `event.preventDefault()` plus `showAlert(.editor-pane .px-6.pt-4, 'warning', msg)` mit DE-Microcopy. Submit-Btn bleibt im Default-State (kein „Preparing…"-Loading-Touch), kein CSRF-Roundtrip, kein Server-Roundtrip. F-1's „roter Ring um Drop-Zone" entfällt hier (kein Drop-Zone — der File-Input ist nativ visible), Banner reicht.
- **Visuelle Hinweise:** Banner oben in der Editor-Pane (gleicher Container wie P1 + P10), Warning-Tint, Close-`×`. Auto-Dismiss nach 6 s. Fokus springt nach Banner-Render auf den Textarea (markdown_text).
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Empty-Submit-Banner: „Bitte zuerst Markdown im Feld eintragen oder eine Datei hochladen."
- **Helper-Reuse:** `showAlert(container, 'warning', msg)` aus [static/js/_utils.js:39-78](../static/js/_utils.js#L39-L78). Cross-Feature-Konvergenz auf F-1.3 P1, F-2.3 P1. Code-Anker: [static/js/markdown_converter.js:309-349](../static/js/markdown_converter.js#L309-L349) Submit-Handler.
- **Aufwand:** XS — Pre-Check-Branch am Anfang des Submit-Handlers (3-4 Zeilen). F-1.3-Aufwand war S (weil F-1 zusätzlich `c-drop-zone--invalid`-CSS-Klasse anlegen musste); hier XS, weil keine Drop-Zone existiert und nur der Banner reicht.
- **Impact-Score:** 2 × 5 / 1 = **10.0**

---

### Pattern 13: Submit-Loading-Persistenz bis Page-Navigation
**Adressiert Findings:** F7 (H1 Sev 2 ⚠️ code-only)
**F-1-Korrespondenz:** F-1.3 P9 (Submit-Loading-Indikation — teil-übertragen, andere Mechanik wegen native form.submit())
**Cluster:** 4 (Async-Pre-Check und Loading-Visibility)
**🔥 Smoke-Pflicht in F5-IMPL**

- **Pattern:** Heutiger Submit-Handler ([static/js/markdown_converter.js:345-347](../static/js/markdown_converter.js#L345-L347)) setzt `submitBtn.innerHTML = originalLabel` **vor** dem `form.submit()`-Call. Loading-Text „Preparing…" verschwindet sofort, User sieht während der Server-PDF-Gen-Zeit (mehrere Sekunden für Playwright-Browser-Boot) keinen Loading-Indikator mehr. Pattern: Restore-Zeile entfernen — der `form.submit()`-Call löst Page-Navigation aus, die Page wird ohnehin durch den Server-Response (PDF-Download oder Re-Render) ersetzt. Wenn der PDF-Download erfolgreich ist, bleibt die Page sichtbar (Browser zeigt Download-Dialog), aber der Submit-Btn-Text bleibt auf „Wird vorbereitet …" — User-Feedback ist „Aktion wurde durchgeführt". Bei P8-Apply (PDF-Gen-Error → flash+redirect) wird die Page neu geladen, der Btn-Text fällt zurück auf den Server-gerenderten Default.
- **Visuelle Hinweise:** Submit-Btn bleibt im Loading-State („Wird vorbereitet …", `disabled=true`) bis zur Page-Navigation oder bis zum Server-Response. Kein Spinner-Icon nötig — Text-Swap reicht.
- **Microcopy** (DE, Du-Form, max 3 Wörter Buttons; DE-Pass mit-genommen):
  - Submit-Btn default (Server-gerendert via [templates/markdown_converter.html](../templates/markdown_converter.html), DE-Pass): „PDF erstellen" (heute „Convert to PDF")
  - Submit-Btn loading (ersetzt heutiges „Preparing…"): „Wird vorbereitet …"
- **Helper-Reuse:** keine. Code-Anker: [static/js/markdown_converter.js:309-349](../static/js/markdown_converter.js#L309-L349) Submit-Handler; insbesondere [markdown_converter.js:344-347](../static/js/markdown_converter.js#L344-L347) Restore-Block, der entfällt; [templates/markdown_converter.html](../templates/markdown_converter.html) Submit-Btn-Default-Label für DE-Pass.
- **Aufwand:** XS — Restore-Zeile entfernen + DE-Microcopy-String-Substitution für Loading-Text + DE-Pass für Submit-Btn-Default-Label im Template. Drei Stellen.
- **Impact-Score:** 2 × 5 / 1 = **10.0**
- **Smoke-Mechanik:** DevTools-Network-Throttle auf „Slow 3G" (oder vergleichbar) → Submit klicken → erwartet: Submit-Btn zeigt „Wird vorbereitet …" für die gesamte CSRF-Refresh + Server-PDF-Gen-Zeit (mehrere Sekunden), kein vorzeitiger Restore-Flash. Bei erfolgreichem PDF-Download bleibt der Loading-Text sichtbar bis zur nächsten Page-Interaktion.

---

## Bug-Tickets ohne UX-H-Komponente (nicht in F-5.3)

Drei Bug-Tickets sind explizit nicht in F-5.3 adressiert (Sprint-Prompt Out-of-scope), weil keine UX-Heuristik-Komponente vorhanden ist:

- **BT3: `updateStyle()` wird beim Page-Load doppelt aufgerufen.** [static/js/markdown_converter.js:281+289](../static/js/markdown_converter.js#L281). Reine Code-Hygiene — User sieht heute keinen Unterschied (Browser-Cache absorbiert den zweiten Fetch). **Verzahnung-Hinweis:** wenn P7 (Two-Dark-Modes-Konsolidierung) den Init-Pfad anfasst, kann BT3 in derselben Stelle mit-aufgeräumt werden. F5-IMPL entscheidet ob mit-fixen oder separater Bug-Pass.
- **BT4: Inline-`<style>` im Template.** [templates/markdown_converter.html:156-168](../templates/markdown_converter.html#L156-L168) deklariert `.orientation-btn`-Styles inline statt in `style.css`. Reine Architektur-Drift, User sieht keinen Unterschied. CSP-strict ist im Single-User-LAN-Setup nicht aktiv. Sammel-Bug-Pass.
- **BT5: Toter `<link id="preview-style" href="">`.** [templates/markdown_converter.html:6](../templates/markdown_converter.html#L6) — leerer `href`, wird nirgendwo per JS gesetzt. Reine Tot-Code-Aufräumung. **Verzahnung-Hinweis:** wenn P5 (Reader-Mode-Persistenz) den Init-Pfad anfasst (DOMContentLoaded-Branch ist nahe), kann BT5 mit-genommen werden. F5-IMPL entscheidet ob mit-fixen oder separater Bug-Pass.

BT1 und BT2 sind in den Patterns ihrer verknüpften Findings adressiert (siehe Pattern-Block-Header):
- BT1 ↔ P8 (PDF-Gen-Error-Recovery)
- BT2 ↔ P9 (Sample-Markdown-Template-Bug-Fix)

---

## Cluster-Vorbereitung für Implementation

**Default-Empfehlung — 1-Sprint mit 3 Sub-Batches.** Bei 13 Patterns liegt der Sprint im 1-Sprint-mit-Sub-Batches-Bereich (F-3-IMPL hat 15 Patterns in 3 Sub-Batches geschafft, F-4-IMPL-B 10 Patterns in einem Sweep). Cluster sind teilweise verkoppelt (Cluster 1 und Cluster 4 teilen den `.editor-pane .px-6.pt-4`-Container; Cluster 3 P8 + P9 + P10 teilen die DE-Microcopy-Disziplin), aber zwischen Clustern keine harten Reihenfolge-Abhängigkeiten.

### Sub-Batch A (Cross-Feature-H4-Konvergenz zu F-1)

Patterns: **P1, P2, P3, P4** — 4 Patterns. Aufwand-Mix: 1× S, 2× XS, 1× S.

Begründung Gruppierung: alle vier Patterns sind direkte F-1.3-Übernahmen (P4-Korrespondenz F1+F2 → P1, P7-Korrespondenz F14 → P2, P12-Adaption F10 → P3, P13-Korrespondenz F13 → P4). Helper-Reuse-Quote ~100% (`showAlert`, `formatFileSize`, native Template-Attribute). Statisch verifizierbar, kein Smoke-Pflicht-Druck. Mechanisch der einfachste Cluster — guter Sprint-Anfang.

**Cross-Feature-H4-Sub-Gruppe (Konvergenz):** P1 (showAlert in Save-Pfad), P2 (Auto-Dismiss-Helper-Spiegel), P3 (formatFileSize-Reuse), P4 (Iframe-a11y-Annotations). Diese vier führen `markdown_converter` strukturell auf F-1-Niveau zurück.

### Sub-Batch B (Reader-Mode-State und Visual-Layout)

Patterns: **P5, P6, P7** — 3 Patterns. Aufwand-Mix: 1× M, 1× XS, 1× S.

Begründung Gruppierung: alle drei Patterns adressieren Cluster 2 aus den Findings (Reader-Mode-Lifecycle und Visual-Layout). Alle drei tragen `🔥 Smoke-Pflicht in F5-IMPL` (P5 für F11-Teil, P6 vollständig, P7 vollständig). Reihenfolge-Empfehlung: P6 (XS, isoliert) zuerst, dann P7 (S, Two-Dark-Mode-Konsolidierung), dann P5 (M, größte Investition mit State-Persistenz). Verzahnungen: P5 und P7 teilen den DOMContentLoaded-Branch, sinnvoll im selben Touch zu erledigen.

**Smoke-Sequenz vor Apply:** P6 — Esc + Textarea-Fokus-Test; P7 — Theme-Toggle + Reader-Dark-Reihenfolge-Test (Sequenz 1 + 2 aus dem Pattern-Block); P5 — Reader-Mode-Toggle + Reload + Width-Button-Visual-Inspect.

### Sub-Batch C (Error-Recovery + Async-Pre-Check)

Patterns: **P8, P9, P10, P11, P12, P13** — 6 Patterns. Aufwand-Mix: 5× XS, 1× S.

Begründung Gruppierung: Cluster 3 (P8-P11, Error-Recovery-Pfade) und Cluster 4 (P12-P13, Async-Pre-Check) zusammen, weil alle sechs Patterns überwiegend XS-Aufwand und überwiegend Smoke-Pflicht (P8, P9, P10, P11, P13 — alle mit ⚠️ code-only-Findings). P12 ist nicht code-only und nicht Smoke-Pflicht, fällt aber thematisch in den Submit-Pfad — sinnvoll zusammen mit P13 (Submit-Loading) zu erledigen, weil beide den Submit-Handler anfassen. Reihenfolge-Empfehlung: P8 (Backend, isoliert) zuerst, dann P9 (Template, BT2-Verzahnung mit P8), dann P10 (Frontend, isoliert), dann P11 (Microcopy-Substitution), dann P12 + P13 (beide im Submit-Handler).

**Smoke-Sequenz vor Apply:** P8 — PDF-Gen-Failure forcieren; P9 — Empty-Page-Load + post-P8-Verifikation; P10 — DevTools-Block für Theme-CSS-URL; P11 — DevTools-Block für CDN-URL; P13 — DevTools-Network-Throttle für Submit-Loading-Persistenz.

### Zwei-Sprint-Empfehlung (falls Sub-Batch C als zu groß empfunden wird)

6 Patterns in einem Sub-Batch ist die obere Grenze (F-3-IMPL Cluster I + II zusammen hatten ~6 Patterns, F-4-IMPL-B war 10 in einem Sweep). Falls F5-IMPL-Sub-Thread Sub-Batch C als zu unhandlich empfindet:

- **Sub-Batch Ca (Backend + Template Error-Recovery):** P8, P9 — 2 Patterns. Backend-Touch + Template-Defensive-Code. Smoke-Pflicht für P8.
- **Sub-Batch Cb (Frontend Error-Recovery + Submit-Pfad):** P10, P11, P12, P13 — 4 Patterns. Reine Frontend-Patterns, alle im DOMContentLoaded oder Submit-Handler. Smoke-Pflicht für P10, P11, P13.

Sub-Batch A und B bleiben unverändert.

---

## Top-5 Quick-Wins

**Aufwand-Gewicht:** XS=1, S=2, M=4, L=8. Score = Sev × 5 / Aufwand-Gewicht. Höher = besser.

| Rang | Pattern # | Adressiert | Sev | Aufwand | Impact-Score | Quick-Win |
|------|-----------|------------|-----|---------|--------------|-----------|
| 1 | P8 | F3 + BT1 — PDF-Gen-Error-Recovery | 3 | XS | 15.0 | ★ Top-5 |
| 2 | P9 | F6 + BT2 — Sample-Merge-Template-Bug | 2 | XS | 10.0 | ★ Top-5 |
| 3 | P12 | F4 — Empty-Submit-Pre-Check | 2 | XS | 10.0 | ★ Top-5 |
| 4 | P13 | F7 — Submit-Loading-Persistenz | 2 | XS | 10.0 | ★ Top-5 |
| 5 | P1 | F1 + F2 — Save-Failure-Banner | 3 | S | 7.5 | ★ Top-5 |
| 6 | P2 | F14 — Alert-Auto-Dismiss | 1 | XS | 5.0 | |
| 7 | P4 | F13 — Iframe-a11y | 1 | XS | 5.0 | |
| 8 | P6 | F12 — Esc-Key-Scope | 1 | XS | 5.0 | |
| 9 | P7 | F8 — Two-Dark-Modes-Konsolidierung | 2 | S | 5.0 | |
| 10 | P10 | F5 — Theme-CSS-Fetch-Recovery | 2 | S | 5.0 | |
| 11 | P11 | F15 — markdown-it CDN-Fallback DE | 1 | XS | 5.0 | |
| 12 | P3 | F10 — File-Info-Display | 1 | S | 2.5 | |
| 13 | P5 | F9 + F11 — Reader-Mode-Persistenz | 2 | M | 2.5 | |

**Top-5 Quick-Wins:**

1. **P8 — PDF-Gen-Error-Recovery** (15.0): höchster Impact-Score der Stage. Sev 3 mit 1-2 Zeilen Backend-Refactor (`render_template` → `redirect`). Beseitigt einen Sev-3-Error-Recovery-Pfad-Bruch und löst BT1 strukturell mit. Smoke-Pflicht (PDF-Gen-Failure forcieren), aber Apply-Risiko niedrig weil flash+redirect ein etabliertes Flask-Pattern ist.
2. **P9 — Sample-Merge-Template-Bug-Fix** (10.0): drei Template-Zeilen, schließt BT2 strukturell. Defensive-Code-Hygiene für künftige Re-Render-Pfade. Smoke-Pflicht entfällt nach P8-Apply weil dann der Live-Trigger weg ist — sinnvoll im selben Touch.
3. **P12 — Empty-Submit-Pre-Check** (10.0): F-1.3-Konvergenz mit `showAlert`-Helper, 3-4 Zeilen im Submit-Handler. Spart den Server-Flash-Roundtrip und macht User-Feedback inline.
4. **P13 — Submit-Loading-Persistenz** (10.0): drei Stellen (Restore-Zeile entfernen + DE-Microcopy für Loading-Text + DE-Pass für Submit-Btn-Default). Schließt einen System-Status-Bruch ohne neue Mechanik.
5. **P1 — Save-Failure-Banner** (7.5): F-1.3-Konvergenz, drei `alert()`-Calls 1:1 ersetzen. Hat den höchsten Sev (3) der F-1-Konvergenz-Patterns. Cross-Feature-H4-Pflicht.

P10 (Theme-CSS-Fetch-Recovery, Sev 2 + S, Score 5.0) liegt knapp dahinter und ist Pflicht-Fix für Cluster 3 — bei Sub-Batch-C-Bündelung mit P11 zusammen 1-Touch. P5 (Reader-Mode-Persistenz, Sev 2 + M, Score 2.5) ist die größte Investition des Sprints und reflektiert in seinem niedrigen Score den Aufwand, nicht die Priorität — bleibt Pflicht für Cluster 2.

---

## Smoke-Pflicht-Übersicht

Patterns mit `🔥 Smoke-Pflicht in F5-IMPL`-Sub-Tag (8 Findings adressiert: F3, F5, F6, F7, F8, F11, F12, F15 — alle ⚠️ code-only-markiert in F-5.2):

| Pattern | Adressiert | Cluster | Smoke-Mechanik (vor Apply) |
|---------|-----------|---------|----------------------------|
| **P5** | F11 (Width-Buttons-Initial-Active) | 2 | Reader-Mode-Toggle aktivieren → Width-Button-Visual mit `.active`-Marker prüfen. Page reload → DevTools-Inspect von `localStorage.readerPrefs`. |
| **P6** | F12 (Esc-Key Document-global) | 2 | Reader-Mode aktivieren → Textarea fokussieren → Esc → erwartet: Reader-Mode bleibt. Reader-Mode aktivieren → Body fokussieren → Esc → erwartet: Reader-Mode wird verlassen. |
| **P7** | F8 (Two-Dark-Modes) | 2 | Sequenz 1: Globaler Dark → Reader-Dark-Toggle → Reader-Light-Klick. Sequenz 2: Globaler Dark → Reader-Mode-Toggle in/out. Beide mit DevTools-Inspect von `data-theme` und `data-global-theme`. |
| **P8** | F3 (PDF-Gen-Error-Re-Render) + BT1 | 3 | PDF-Gen-Failure forcieren via Theme-Datei-Manipulation oder Playwright-Mock → Page-Re-Render beobachten. Erwartet: Banner-Page mit DE-Microcopy statt 500. |
| **P9** | F6 (Sample-Merge) + BT2 | 3 | Empty-Page-Load → Sample sichtbar. Optional manueller Test: temporärer Endpoint mit `markdown_text='Test'` → erwartet: nur Test, kein Sample. |
| **P10** | F5 (Theme-CSS-Fetch silent) | 3 | DevTools-Network-Block für `/static/css/pdf_styles/<theme>.css` → Theme-Selector wechseln → erwartet: Banner sichtbar, Preview ohne Theme. |
| **P11** | F15 (markdown-it CDN) | 3 | DevTools-Network-Block für jsdelivr-CDN-URL → Page reload → erwartet: Iframe-Fallback mit DE-Microcopy. |
| **P13** | F7 (Submit-Loading kurz) | 4 | DevTools-Network-Throttle „Slow 3G" → Submit klicken → erwartet: „Wird vorbereitet …" sichtbar bis Page-Navigation. |

**Anzahl Smoke-Pflicht-Patterns:** 8 (von 13). Adressiert alle 8 ⚠️ code-only-Findings aus F-5.2.

**Anzahl Patterns ohne Smoke-Tag:** 5 (P1, P2, P3, P4, P12) — F-1-Konvergenz oder live-evidente Verhalten, statisch verifizierbar.

---

## Cross-Feature-H4-Sektion (Schwester-Feature-Konvergenz zu F-1)

Pattern-Konvergenz-Quote: **86% (12 von 14 anwendbaren F-1-Patterns)** — aus F-5.2 1:1 übernommen. Drastisch höher als F-2.2 (~41%), F-3.2 (~35%) und F-4.2 (0%) — Schwester-Feature-Inversion bestätigt.

### Direkt übertragbare F-1-Patterns (mit `markdown_converter`-Patterns)

| F-1-Pattern | F-5.2-Finding | F-5.3-Pattern | Heuristik | Sev | Code-Anker |
|-------------|---------------|---------------|-----------|-----|------------|
| **F-1.3 P4 — Save-Failure Browser-`alert()` ersetzen** | F1 + F2 | **P1** (1:1 übernommen) | H4 + H9 | 3 | [static/js/markdown_converter.js:146](../static/js/markdown_converter.js#L146) |
| **F-1.3 P7 — Alert-Auto-Dismiss** | F14 | **P2** (übernommen, Aufwand XS statt S) | H1 | 1 | [templates/markdown_converter.html:31](../templates/markdown_converter.html#L31) |
| **F-1.3 P12 — Filename KB/MB-Fallback** | F10 | **P3** (Adaption: Display einführen statt Unit-Bug fixen, Aufwand S statt XS) | H1 | 1 | [static/js/markdown_converter.js:266-275](../static/js/markdown_converter.js#L266-L275) |
| **F-1.3 P13 — Result-Content a11y** | F13 | **P4** (1:1 übernommen, Iframe statt `<pre>`) | H6 | 1 | [templates/markdown_converter.html:142](../templates/markdown_converter.html#L142) |

### Teil-übertragbare F-1-Patterns (mit modifizierten F-5.3-Patterns)

| F-1-Pattern | F-5.2-Finding | F-5.3-Pattern | Heuristik | Sev | Anpassung |
|-------------|---------------|---------------|-----------|-----|-----------|
| **F-1.3 P1 — Empty-Submit silent** | F4 | **P12** (Sev 2 statt 4, kein Drop-Zone-Ring nötig) | H1 | 2 | F-1 hatte Drop-Zone + Banner; hier nur Banner, weil File-Input nativ visible. Aufwand XS statt S. |
| **F-1.3 P9 — Submit-Loading-Indikation** | F7 | **P13** (andere Mechanik wegen native form.submit()) | H1 | 2 | F-1 hatte Drop-Zone-Loading-Skeleton; hier Restore-Zeile entfernen + DE-Pass für „Wird vorbereitet …". Aufwand XS statt M. |

### Bereits konvergente F-1-Patterns (markdown_converter erfüllt strukturell — kein Pattern nötig)

| F-1-Pattern | Bereits erfüllt durch | Anmerkung |
|-------------|------------------------|-----------|
| **F-1.3 P3 — Save-Btn `.saved`-Reset** | setTimeout-Reset-Routine entfernt `.saved` korrekt nach 2s ([markdown_converter.js:135-139](../static/js/markdown_converter.js#L135-L139)) | F-5.2 Cross-Feature-H4-Sektion bestätigt — kein Finding, kein Pattern. |
| **F-1.3 P6 — Format-Label/`accept` Single-Source-of-Truth** | SEC-Sprint-F-006: `accept` wird aus `ACCEPTED_EXTENSIONS = ('md', 'markdown')` generiert ([app_pkg/markdown.py:25](../app_pkg/markdown.py#L25)) | F-5.2 hat einen Mini-Aspekt erwähnt (Label-Hint „Erlaubt: .md, .markdown" als sichtbares Label-Beispiel) — unter Sev-1-Schwelle, F-5.3 nimmt das **nicht** als eigenes Pattern auf. |
| **F-1.3 P8 — Frontend-Vorab-Check unsupported file** | Submit-Handler prüft Extension via `fileExtensionAllowed()` und nutzt `showAlert(... 'danger', 'Dateiformat nicht unterstützt. Erlaubt: .md, .markdown.')` ([markdown_converter.js:317-322](../static/js/markdown_converter.js#L317-L322)) | DE-Microcopy bereits vorhanden. F-5.3 nutzt denselben Banner-Mountpoint für P1 / P10 / P12. |
| **F-1.3 P10 — Result-Area scrollIntoView** | Konvergent durch Architektur — Iframe-Preview ist Split-Pane, immer im Viewport | F-5.2 hat „bereits konvergent" gegen F-5.1's „teil bzw. bereits-erfüllt" verschärft. |
| **F-1.3 P11 — Drop-Zone Keyboard-Pfad** | Konvergent durch Architektur — visible native `<input type="file">` ist out-of-the-box keyboard-erreichbar | F-5.2 hat „bereits konvergent" verschärft. |

**5 bereits-konvergente F-1-Patterns als positives Inventar.** Sprint-Prompt-konform: keine Patterns nötig, aber explizit gelistet damit F-5.3-Doc den vollständigen F-1-Konvergenz-Status zeigt.

### Nicht-anwendbare F-1-Patterns

| F-1-Pattern | Begründung |
|-------------|-----------|
| **F-1.3 P2 — Result-Area persistiert nach Clear** | Kein Result-Area, kein Clear-Button — der Workflow ist Live-Preview + Server-PDF-Download. Keine Stale-State-Problematik. |
| **F-1.3 P5 — Drag-Active-Highlight transparent** | Keine Drop-Zone, nur visible File-Input ohne Drag-Active-Styling. Pattern setzt Drop-Zone voraus. |

### Konvergenz-Quote-Zusammenfassung

- **Anwendbare F-1-Patterns**: 12 (von 14 total; 2 nicht-anwendbar)
- **Direkt übertragbar**: 4 (F-1.3 P4 → P1, F-1.3 P7 → P2, F-1.3 P12 → P3, F-1.3 P13 → P4)
- **Teil-übertragbar mit Anpassung**: 2 (F-1.3 P1 → P12, F-1.3 P9 → P13)
- **Bereits konvergent**: 5 (F-1.3 P3, P6, P8, P10, P11)
- **Pattern-Konvergenz-Quote**: 12/14 = **86%** (1:1 aus F-5.2 übernommen)
- **Cross-Feature-H4-Findings adressiert in F-5.3-Patterns**: 7 (F1, F2, F4, F7, F10, F13, F14) → 6 Patterns (P1 konsolidiert F1+F2, P2 für F14, P3 für F10, P4 für F13, P12 für F4, P13 für F7)

---

## Helper-Vorschläge (für F5-IMPL-Sub-Thread zur Entscheidung)

Beim Pattern-Schreiben sind zwei mögliche neue `_utils.js`-Helper aufgefallen, die für künftige Wiederverwendung sinnvoll wären — **nicht** still im jeweiligen Pattern mit-anlegen, sondern F5-IMPL-Sub-Thread entscheidet, ob die Helper im Pattern-Cluster mit-implementiert oder als separater Helper-Cluster vorgezogen werden:

- **`saveViewState(key, state)` / `loadViewState(key)`** — generischer State-Persistenz-Helper für JSON-LocalStorage-Patterns. Heute in P5 (Reader-Mode-Persistenz) als lokale Erweiterung von `getReaderPrefs()` / `saveReaderPrefs()` benötigt. Wenn künftig weitere Features State-Persistenz brauchen (z.B. Library-Filter-Persistenz, Notion-Toggle-State, Document-Converter-Theme-Wahl), wäre der Helper Cross-Feature angemessen. Bis dahin reicht die lokale `getReaderPrefs()`-Variante in `markdown_converter.js`.
- **`attachAutoDismissToServerBanners(container, levels?, durationMs?)`** — Spiegel der `showAlert`-Auto-Dismiss-Logik für Server-side gerenderte `c-alert`-Banner. Heute in P2 (Auto-Dismiss-Hook für Flash-Banner) benötigt. Wenn `library_detail` / `audio_converter` / `document_converter` / `podcast_flow` ihre Server-Flash-Banner ebenfalls auf Auto-Dismiss bringen wollen (heute hängen sie statisch ohne Timer), wird der Helper Cross-Feature relevant. Bis dahin reicht die lokale Init-Hilfsfunktion in `markdown_converter.js`.

**Disposition:** beide Helper bleiben im jeweiligen Pattern-Block als „Helper-Vorschlag" markiert; F5-IMPL-Sub-Thread entscheidet beim Cluster-Schnitt.

---

**Schweregrad-Skala (aus Stufe 2):**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse / Datenverlust- oder Cost-Pfad)
