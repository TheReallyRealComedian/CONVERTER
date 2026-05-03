# UI-Inventur: document_converter (2026-05-03)

**Methodik:** Stufe 1 der Duan-Kaskade. Statische Analyse + Live-Walkthrough.
**Live-Walkthrough:** erfolgreich. Container `markdown-converter-web` lief bereits auf `localhost:5656`. Browser via `mcp__Claude_in_Chrome` (Linux-Chrome auf Mintbox), Tab nach Login auf `/document-converter`. Datei-Picker via DataTransfer-API simuliert (das native `file_upload`-MCP-Tool wurde von Chrome mit "Not allowed" zurückgewiesen — funktional äquivalent: Datei landet im Input via `input.files = dt.files` + `change`-Event).

**Quellen:** `templates/document_converter.html`, `static/js/document_converter.js`, `app_pkg/documents.py`, `templates/base.html`, `templates/_partials/flash_messages.html`, `static/js/_utils.js`, `static/css/style.css` (für State-Klassen `.c-drop-zone`, `.c-btn`, `.c-alert`, `.save-library-btn`, `.toast-notification`).

---

## Element-Tabelle

Legende States: ✓ vorhanden im Code · ✗ fehlt · ? unklar/teils · n/a nicht zutreffend
Live-Spalte: ✓ live verifiziert · ✗ live nicht beobachtet · ↯ Differenz Code↔live · n/a nicht prüfbar (rein statisch)

Regions: **Layout** = `base.html` · **Header** = Page-Heading-Bereich · **Form** = Upload-Form · **Result** = Conversion-Result-Panel · **Feedback** = Alerts/Toasts · **Browser-native** = HTML5 Constraint Validation

| #  | Region   | Element-Typ      | Label/Text                                  | Aktion                                                                         | default | hover | focus / focus-visible | disabled | loading | error | success | empty | Live verifiziert |
|----|----------|------------------|---------------------------------------------|--------------------------------------------------------------------------------|---------|-------|------------------------|----------|---------|-------|---------|-------|-------------------|
| 1  | Layout   | Button           | Theme-Toggle (Sun/Moon-Icon)                | Toggle `data-global-theme` light↔dark + persist in `localStorage`              | ✓       | ✓ (`.theme-toggle-btn:hover`) | ? (nur `:active`-Regel; `:focus-visible` fehlt für diesen Button) | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ default sichtbar; nicht durchgeklickt |
| 2  | Layout   | Link             | "File Transformer" (Brand)                  | → `/markdown-converter` (markdown_converter)                                    | ✓       | ✓ (`hover:text-neo-accent`) | ? (kein eigenes Focus-Style sichtbar)                              | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ Tab-Order Position 2 |
| 3  | Layout   | Link             | "Markdown to PDF" (Sidebar-Nav)             | → `/markdown-converter`                                                         | ✓       | ✓ (`hover:text-neo-text`) | ?                                                                  | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ |
| 4  | Layout   | Link             | "Document Converter" (Sidebar-Nav, aktiv)   | → `/document-converter`; aktiver State via `neo-nav-active` Klasse              | ✓ + `active`-Klasse | ✓                | ?                                                                  | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ aktiver State sichtbar (eingedrückte Pille) |
| 5  | Layout   | Link             | "Audio Converter" (Sidebar-Nav)             | → `/audio-converter`                                                            | ✓       | ✓     | ?                                                                  | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ |
| 6  | Layout   | Link             | "Mermaid Converter" (Sidebar-Nav)           | → `/mermaid-converter`                                                          | ✓       | ✓     | ?                                                                  | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ |
| 7  | Layout   | Link             | "Library" (Sidebar-Nav)                     | → `/library`                                                                    | ✓       | ✓     | ?                                                                  | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ |
| 8  | Layout   | Link             | "Logout"                                    | → `/logout`                                                                     | ✓       | ✓ (`hover:text-neo-text`) | ?                                                            | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ |
| 9  | Layout   | Button           | Mobile-Sidebar-Toggle (Hamburger)           | Sidebar ein/ausblenden (mobile)                                                 | ✓       | ?     | ?                                                                  | n/a      | n/a     | n/a   | n/a     | n/a   | n/a (Desktop-Viewport) |
| 10 | Header   | Heading (kein interaktives Element) | "Document to Text Converter" + Untertitel | rein dekorativ                                          | ✓       | n/a   | n/a                                                                | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ |
| 11 | Form     | Drop-Zone (Div)  | "Drop file here or click to browse / PDF, DOCX, PPTX, EML, HTML -- Max 100MB" | `onclick` triggert `#document_file.click()`; `dragover/dragleave/drop` Handler in JS | ✓ | ✓ (`.c-drop-zone:hover`) | **✗ kein Tab-Stop** (kein `tabindex`, kein `role`, kein `aria-label`) | n/a | ✗ kein Loading-State während Upload | ✗ kein Error-State (z.B. unsupported drag-Type) | ? Drag-Active-Klasse `.drop-zone-active` wird gesetzt, aber visueller Ring ist im Computed-Style `rgba(0,0,0,0)` → live kaum unterscheidbar von default | ✓ (default = "leer") | ✓ statisch + ✓ dragover/dragleave Events; ↯ Drag-Active-Highlight im Code definiert, live nicht visuell erkennbar |
| 12 | Form     | File-Input (hidden) | `#document_file` (`required`, `name=document_file`, `class=hidden`) | hält File-Selection für Form-Submit                                       | n/a (immer hidden) | n/a | **✗ nicht fokussierbar (display:none)** | n/a | n/a | ? — HTML5-Validation-Message "Wähle eine Datei aus." (de-DE) wird gesetzt, aber Browser-Popover hat keinen sichtbaren Anker → ↯ silent fail (siehe Bemerkungen) | n/a | n/a | ↯ Validation-Message existiert, aber unsichtbar; siehe Bemerkungen |
| 13 | Form     | Container        | `#file-info` (Selected-File-Indicator)      | wird nach `change`/`drop` von `hidden` → sichtbar (`removeClass('hidden')`)     | ✓ (initial `hidden`) | n/a | n/a                                                       | n/a      | n/a     | n/a   | ✓ (zeigt Filename + Size)  | ✓ initial via `hidden`-Klasse | ✓ erscheint nach Datei-Pick |
| 14 | Form     | Span (innerhalb #13) | `#file-name` Text "filename.ext (X.X MB)" | Anzeige Name + Größe (in MB, immer auf 1 Nachkommastelle gerundet)              | ✓       | n/a   | n/a                                                                | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ — bei 222-byte-Datei wird "0.0 MB" angezeigt (kein KB-Fallback) |
| 15 | Form     | Button           | Clear-File "×" (`#clear-file`, title="Clear file") | Setzt `fileInput.value=''` + versteckt `#file-info`                       | ✓       | ✓ (`hover:text-neo-text`) | ✓ Browser-default outline `2.74px` rgb(131,149,167)         | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ — entfernt Datei korrekt; **↯ ändert NICHT** den `#result-area` (alter Result bleibt sichtbar) und **NICHT** den `#save-btn`-Stale-State |
| 16 | Form     | Button (Submit)  | "Transform to Text" (`#convert-btn`, `c-btn--primary`) | submit → `POST /transform-document` (multipart) → Response als md       | ✓       | ✓ (`.c-btn:hover` translateY) | ✓ (`.c-btn:focus-visible` Ring)                            | ✓ (`.c-btn:disabled` opacity 0.4, cursor not-allowed) | ✓ Text → "Converting..." + `disabled=true` (kein Spinner-Icon) | ✓ wirft `Error` aus catch → Alert in `#alert-container`; Button restored auf "Transform to Text" | ✓ implizit (Result erscheint, Button restored) | n/a | ✓ alle states außer Spinner/Progress |
| 17 | Feedback | Container        | `#alert-container`                          | Inline-Container für JS-injizierte `.c-alert--danger` Banner                    | ✓ (leer initial) | n/a | n/a                                                       | n/a      | n/a     | ✓ (HTML wird per `innerHTML=` gefüllt) | n/a (kein success-Pfad) | ✓ leer | ✓ Banner erscheint bei 500er; **kein Auto-Dismiss, kein Close-Button** |
| 18 | Feedback | Inline-Alert (`.c-alert.c-alert--danger`) | dynamisch erzeugt mit `err.message` (z.B. "Conversion failed (500)" oder "Simulated backend error" oder "Session expired …") | erscheint bei fetch-Fehler, fail-Status, oder safeJSON-Login-Redirect | ✓ Styling | n/a | n/a | n/a | n/a | ✓ rendert | n/a | n/a | ✓ — nur `c-alert--danger` Pfad; `c-alert--success/warning/info` werden hier nie verwendet |
| 19 | Feedback | Server-Flash (Jinja `{% include _partials/flash_messages.html %}`) | rendert `get_flashed_messages()` mit category-class | nur server-rendered; in `documents.py` werden allerdings keine `flash()`-Calls abgesetzt → de facto immer leer | ✓ Template eingebunden | n/a | n/a | n/a | n/a | n/a | n/a | ✓ leer | ✓ — leer in allen Walkthrough-Schritten |
| 20 | Result   | Container        | `#result-area` (`hidden` initial)           | enthält Heading + Action-Buttons + `<pre>`-Inhalt                                | ✓ initial `hidden` | n/a | n/a                                                       | n/a      | n/a     | n/a   | ✓ (`.remove('hidden')`) | ✓ `hidden` initial (kein Empty-Placeholder, einfach unsichtbar) | ✓ erscheint nach Erfolg; **persistiert nach `clear-file`-Click** (siehe Bemerkungen) |
| 21 | Result   | Heading          | "Conversion Result"                         | rein dekorativ                                                                  | ✓       | n/a   | n/a                                                                | n/a      | n/a     | n/a   | n/a     | n/a   | ✓ |
| 22 | Result   | Button           | "Download .md" (`onclick="downloadResult()"`) | Erzeugt Blob aus `lastResult.content`, klickt programmatic Anchor-Element     | ✓       | ✓ (`.c-btn:hover`) | ✓ (`.c-btn:focus-visible`)                                     | n/a (nie disabled — wenn `lastResult` null, no-op silent) | ✗ kein Loading-State (instant) | ✗ kein Error-State (kein try/catch) | n/a (kein "downloaded"-Feedback) | n/a | ✓ Button erscheint, klickbar |
| 23 | Result   | Button           | "Save to Library" (`#save-btn`, `class=save-library-btn`, `onclick="saveToLibrary()"`) | `POST /api/conversions` mit Content+Metadata; bei 200 → "Saved!" + `.saved`-Klasse | ✓ | ✓ (`.save-library-btn:hover`) | ? (kein eigenes `:focus-visible`) | ✓ während des Calls (`disabled=true`, Text "Saving...") | ✓ Text "Saving..." + disabled | ✓ Catch-Block: `alert(...)` (Browser-native, **nicht** das in-page `c-alert`-System) + Text bleibt "Save to Library" + **`.saved`-Klasse wird NICHT entfernt** → grün-getönter Button trotz Failure | ✓ Text "Saved!" + grüne `.saved`-Klasse + bleibt `disabled` | n/a | ✓ alle states; ↯ stale-`.saved`-Klasse-Bug verifiziert |
| 24 | Result   | `<pre>` Content  | `#result-content` (max-h 500px, scroll, monospace) | zeigt extrahierten Markdown als Plaintext                                | ✓       | n/a   | ✗ (kein `tabindex=0`, kein `role`/`aria-label`) | n/a | n/a | n/a | n/a | n/a | ✓ — Inhalt sichtbar; **scrollt nicht automatisch in View nach Result-Anzeige**; lange Inhalte werden auf 500px gekappt mit eigenem Scroll |
| 25 | Browser-native | Validation Popup | "Wähle eine Datei aus." (de-DE) | Browser-eigenes Popover bei Submit ohne Datei | n/a | n/a | n/a | n/a | n/a | ↯ Message *existiert* (`fi.validationMessage` = "Wähle eine Datei aus."), aber File-Input ist `display:none` → **Popover hat keinen visuellen Anker**, Submit-Klick ohne Datei wirkt nach außen wie ein No-Op | n/a | n/a | ↯ kritisch: leere Submission visuell silent |

---

## Zusammenfassung

- **Gesamtzahl interaktive Elemente:** 24 (ohne #19 inaktive Flash-Includes; ohne #25 Browser-native Popup)
- **Im Code identifizierte fehlende States (✗):**
  - Drop-Zone hat **keinen Tab-Stop / Keyboard-Pfad** — weder `tabindex`, noch `role="button"`, noch `aria-label`. Die einzige Möglichkeit, den Datei-Picker per Tastatur zu öffnen, wäre, Tab auf das (versteckte!) `#document_file`-Input zu legen — was unmöglich ist, weil `display:none`.
  - Drop-Zone hat keinen Loading- oder Error-State während Upload (z.B. wenn unsupported drag type).
  - "Download .md" Button hat keinen Loading-, Error- oder Success-State (kein Toast/Bestätigung; kein try/catch).
  - "Save to Library" Failure-Pfad nutzt Browser-native `alert()` statt das vorhandene `c-alert--danger`-System → inkonsistent mit Conversion-Failure-Pfad (#18).
  - `#result-content` hat keine a11y-Annotations (`role`, `aria-label`, `tabindex`) — Screenreader können den Output nicht gezielt fokussieren.
  - Kein Empty-State im engeren Sinne — `#result-area` ist einfach `hidden`, kein Placeholder à la "Bisher noch keine Konversion".

- **Live-verifizierte Differenzen Code↔live (↯):**
  - Drop-Zone-Active (`.drop-zone-active`) Klasse wird beim Dragover hinzugefügt, aber das im CSS deklarierte `0 0 0 3px rgba(168,180,216,0.3)` Ring-Highlight rendert im Computed-Style als `rgba(0,0,0,0) 0px 0px 0px 0px`. Live ist visuell **kaum** ein Drag-Over-Feedback erkennbar.
  - HTML5-Validation-Message bei leerem Submit existiert technisch, ist aber unsichtbar (Anker-Element ist `display:none`).
  - Save-Library-Button verbleibt nach Failure mit `.saved`-Klasse (grüner Hintergrund), Text "Save to Library" → visuell suggeriert "fertig gespeichert", obwohl Save fehlschlug.
  - Save-Library-Button verbleibt nach `clear-file`-Click ebenfalls mit `.saved`-Klasse, obwohl der Result-Kontext theoretisch frischer Data harrt.
  - `#result-area` bleibt nach `clear-file`-Click sichtbar mit dem alten Conversion-Resultat → potentiell stale.

- **Unverifizierbare States (?):**
  - `:focus-visible`-Style auf einigen Sidebar-Links und auf dem Theme-Toggle — Code zeigt keine expliziten Regeln; Browser-default-Outline kommt zur Anwendung, war aber nicht visuell hervorgehoben in den Screenshots.
  - "Sehr große Datei (>500 MB)" — nicht getestet, da kein einfacher Trigger und Container hat keinen Hartcheck im Frontend (kein `accept`, kein Size-Check); Backend-Verhalten bei Flask-`MAX_CONTENT_LENGTH` ist nicht aus dem Code von `documents.py` ableitbar.
  - "Backend killen mid-upload" — nicht durchgeführt, weil `docker stop markdown-converter-web` den User-Login-State zerstören und andere parallel laufende Operationen abbrechen würde. Stattdessen via fetch-Override ein 500-Response simuliert (Pfad #18 verifiziert).

---

## Bemerkungen / mögliche Bugs (separat — nicht in Tabelle gemischt)

Diese Auffälligkeiten gehen **über fehlende States hinaus** und sind echte Implementierungs-Befunde, die im Walkthrough auftauchten. Stage 2 muss entscheiden, welche davon Heuristik-Findings werden und welche separate Bug-Tickets:

1. **Empty-Submit ist visuell silent.** Klick auf "Transform to Text" ohne Datei: HTML5-Validation-Message ist gesetzt, aber unsichtbar (Anker = display:none-Input). Im Walkthrough sah man **gar nichts** passieren — kein Toast, kein Banner, kein Highlight an der Drop-Zone, keine Browser-Popover-Bubble. Ein User klickt erneut und denkt der Server ist tot.

2. **Drop-Zone-Active ohne sichtbares Highlight.** Die CSS-Regel deklariert ein blau-tinted Ring-Highlight, aber im Computed-Style ist die Ringfarbe transparent (`rgba(0,0,0,0)`). Verifiziert: Klasse wird hinzugefügt, Effekt fehlt. Möglicherweise ein Variable-Resolution-Bug oder eine Übersteuerung an anderer Stelle in style.css. **Nicht weiter analysiert** auf Stage-1-Auftrag.

3. **`Save to Library` Stale-Visual-State.** Nach erfolgreichem Speichern bekommt der Button die `.saved`-Klasse (grün) und bleibt disabled mit Text "Saved!". Bei einer **neuen** Conversion direkt danach setzt der Code zwar `disabled=false` und text="Save to Library" zurück — aber **entfernt nicht** die `.saved`-Klasse. Resultat: grüner Button mit Text "Save to Library" (statt neutralem Default-Look). Identisches Verhalten nach `clear-file`-Click und nach Save-Failure-Pfad. (Code-Stelle: [static/js/document_converter.js:71-74](static/js/document_converter.js:71) und [:120-129](static/js/document_converter.js:120))

4. **Inkonsistentes Error-Reporting für Save-Library.** Conversion-Failures gehen in den page-internen `#alert-container` (`c-alert--danger`-Banner), Save-Failures in einen Browser-native `alert()`-Dialog. Zwei verschiedene Error-UX-Pfade für dasselbe Feature. (Code-Stelle: [static/js/document_converter.js:129](static/js/document_converter.js:129))

5. **Drop-Zone Keyboard-unzugänglich.** Drop-Zone hat keinen `tabindex`, keine `role="button"`, kein `aria-label`. Tab-Reihenfolge überspringt sie komplett. Da das versteckte File-Input ebenfalls nicht erreichbar ist (`display:none`), gibt es **keinen rein-Keyboard-Pfad**, eine Datei zu wählen. (Single-User-Kontext relativiert das, aber als Befund dokumentieren.)

6. **Frontend-vs-Backend-Format-Mismatch.** Drop-Zone-Label sagt "PDF, DOCX, PPTX, EML, HTML". Der File-Input hat **kein `accept`-Attribut**, der Backend `partition()` (unstructured) frisst aber auch `.txt`, `.md`, `.html` und im Walkthrough sogar `.xyz` mit Plaintext-Inhalt. Eine `.xyz`-Datei wurde scheinbar erfolgreich konvertiert. Der UI-Hint ist also entweder zu restriktiv (verschweigt unterstützte Typen) oder unehrlich (lässt unsupported zu). Stage-2-Entscheidung.

7. **Result-Area persistiert nach `clear-file`.** User klickt "×" → Datei verschwindet → das alte Conversion-Result bleibt sichtbar mit den alten Action-Buttons. Suggeriert: "die alte Conversion ist noch aktuell, ich kann sie weiter speichern." Tatsächlich ist `lastResult` aber noch im Memory und Save-to-Library schickt die alten Daten zum Server. Nicht offensichtlich für User.

8. **Kein Auto-Scroll-into-View** beim Erscheinen von `#result-area`. Bei langen Sidebars / kleinen Viewports liegt das Ergebnis möglicherweise außerhalb des Sichtbereichs.

9. **Filename-Size-Anzeige immer in MB.** `(file.size / 1024 / 1024).toFixed(1)` zeigt für 222-byte-Dateien `0.0 MB`. Kein KB/B-Fallback. Gilt im Single-User-Kontext eher als Kosmetik.

---

**Hinweis:** Diese Stufe-1-Datei enthält bewusst **keine** Pattern- oder Microcopy-Vorschläge. Diese folgen in Stufe 3 nach dem Heuristik-Review (Stufe 2).
