# UX-Heuristik-Findings: document_converter (2026-05-03)

**Methodik:** Stufe 2 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Heuristisches Review der strukturierten Inventur aus Stufe 1.
**Quelle:** [docs/ui_inventory_document_converter_2026-05.md](ui_inventory_document_converter_2026-05.md)
**Heuristiken:** Nielsen H1 (Sichtbarkeit des Systemzustands), H4 (Konsistenz und Standards), H6 (Wiedererkennen statt Erinnern), H9 (Fehlermeldungen / Hilfe bei Fehlern)
**Produkt-Kontext:** Single-User (Oliver), LAN-only, login-protected. Technisch versierter Nutzer; H4 bekommt **höheres** Gewicht (mentales Modell der eigenen Konverter), H6 bekommt **geringeres** Gewicht (App ist vertraut). a11y-Findings nicht null gewichtet — Tastaturbedienung wird genutzt.

---

## Findings (sortiert absteigend nach Schweregrad)

| #   | Element / Befund | Problem (1–2 Sätze) | Heuristik | Schweregrad (1–4) | Begründung für Single-User-Kontext |
|-----|------------------|---------------------|-----------|-------------------|--------------------------------------|
| F1  | Submit ohne Datei (Bemerkung B1; Inventur #16/#25) | Klick auf "Transform to Text" ohne ausgewählte Datei wirkt wie ein No-Op: HTML5-Validation-Message ist gesetzt, aber das Anker-Element (`#document_file`) ist `display:none` — kein Popover, kein Banner, kein Drop-Zone-Highlight. Der User bekommt **keinerlei** Rückmeldung, dass sein Klick erkannt wurde. | H1 | 4 | Auch Oliver klickt manchmal vor dem Drop. Ohne Feedback klickt er ein zweites Mal und vermutet Server-/Worker-Problem — die primäre Aufgabe (hochladen → konvertieren) ist blockiert ohne erkennbaren Grund. |
| F2  | Submit ohne Datei (siehe F1) | Der vorhandene Fehlerzustand (Required-Validation) wird nicht zugänglich kommuniziert. Es gibt weder eine sichtbare Meldung noch einen Recovery-Hinweis ("bitte zuerst eine Datei auswählen"). | H9 | 4 | Selbe Disposition wie F1 — H9 ergänzt H1, weil die Recovery-Anleitung fehlt. Single-User ändert nichts, weil das Problem das Fehlen jeglicher Rückmeldung ist, nicht die Verständlichkeit einer Meldung. |
| F3  | Result-Area persistiert nach `clear-file` (Bemerkung B7; Inventur #15/#20) | Klick auf "×" entfernt die Datei aus dem Input, lässt aber `#result-area` mit dem alten Conversion-Resultat sichtbar. UI-State und Memory-State (`lastResult`) divergieren von der User-Intention "Reset". | H1 | 3 | Oliver erwartet beim Clear einen sauberen Neustart. Das alte Result bleibt visuell als "aktueller Zustand" stehen — System kommuniziert einen Zustand, der nicht mehr stimmt. Auch im Single-User-Setup täuschend. |
| F4  | Result-Area persistiert nach `clear-file` (siehe F3) | Da der alte Result + die "Save to Library"-Buttons sichtbar bleiben, kann Oliver versehentlich die alten Daten speichern und glauben, es seien die der gerade geclearten Datei. Wiedererkennung des aktuellen Kontexts schlägt fehl. | H6 | 3 | Mentales Modell ("ich habe gerade die Datei entfernt") wird durch die UI widerlegt. Oliver muss sich erinnern, dass ein Clear nur den Input, nicht das Result reset — Reibung gegen seine eigene Symmetrie-Erwartung. |
| F5  | "Save to Library" Stale-Visual-State (Bemerkung B3; Inventur #23) | Nach erfolgreichem Speichern bleibt der Button mit der `.saved`-Klasse (grüner Hintergrund). Beim Reset (neue Conversion / Clear / Save-Failure) wird `disabled=false` und der Text auf "Save to Library" zurückgesetzt — die `.saved`-Klasse aber **nicht** entfernt. Der Button signalisiert visuell "fertig gespeichert" mit dem Text "Save to Library". | H1 | 3 | Oliver hat ein klares Mental Model "grüner Button = persistierte Aktion". Der Visual-Style erzählt eine andere Geschichte als der Text — System-Status ist intern korrekt, extern verfälscht. Frustrationspotential bei Mehrfach-Konversionen in Folge. |
| F6  | "Save to Library" Stale-Visual-State (siehe F5) | Visueller Style (grün/saved) und Button-Text ("Save to Library") widersprechen sich — gleicher Button kommuniziert in einem Moment zwei inkompatible Zustände. Bricht die interne Konsistenz des Save-Pattern. | H4 | 3 | Im Single-User-Kontext **wichtig**: Oliver erwartet symmetrische Visual-States (grün ⇄ Saved! ⇄ disabled). Die asymmetrische Reset-Routine bricht diese Erwartung jedes zweite Mal. |
| F7  | "Save to Library" Failure nutzt `alert()` statt `c-alert--danger` (Bemerkung B4; Inventur #23) | Conversion-Failures landen im in-page `#alert-container` als `c-alert--danger`-Banner. Save-Failures dagegen nutzen Browser-native `alert()`-Modal. Zwei verschiedene Error-UX-Pfade in **demselben** Feature für **vergleichbare** Failure-Modi. | H4 | 3 | Genau die Asymmetrie, gegen die H4 in Olivers Kontext stark wiegt: er erwartet, dass alle Fehlerpfade derselben Seite gleich aussehen. `alert()` blockiert zudem den Tab — anderer Stil als das nicht-modale Banner. |
| F8  | "Save to Library" Failure nutzt `alert()` statt `c-alert--danger` (siehe F7) | `alert()` zwingt zum manuellen "OK"-Klick und verschwindet dann spurlos — keine in-page Trace, keine Recovery-Hinweise, keine Möglichkeit, die Meldung erneut anzusehen. Ein `c-alert--danger`-Banner würde stehen bleiben und Detail enthalten. | H9 | 3 | Auch im Single-User-Kontext relevant: Save-Failures sind selten genug, dass Oliver die Fehlermeldung schnell wegklickt und vergisst — anstatt sie zu lesen, was die Recovery erschwert. |
| F9  | Drop-Zone-Label vs. Backend-Akzeptanz (Bemerkung B6; Inventur #11/#12) | Drop-Zone sagt "PDF, DOCX, PPTX, EML, HTML — Max 100MB". Das File-Input hat **kein `accept`**-Attribut, das Backend (`unstructured.partition`) frisst aber u.a. auch `.txt`, `.md`, `.html`, und im Walkthrough sogar eine `.xyz`-Datei mit Plaintext-Inhalt. Label und Realität divergieren. | H4 | 3 | Oliver hat einen technischen Anspruch an Konsistenz von Versprechen und Verhalten. Entweder ist das Label zu restriktiv (`txt` ginge auch) oder zu permissiv (`xyz` sollte abgewiesen werden). Beide Lesarten brechen sein mentales Modell des Tools. |
| F10 | Drop-Zone-Active-Highlight transparent (Bemerkung B2; Inventur #11) | Beim Dragover wird die Klasse `.drop-zone-active` gesetzt. Das im CSS deklarierte Ring-Highlight rendert im Computed-Style aber als `rgba(0,0,0,0)` — Drag-State ist live nicht visuell erkennbar. System reagiert ohne sichtbares Feedback. | H1 | 2 | Drop funktioniert trotzdem (Drop-Event kommt an), aber während des Drags fehlt das "Loslassen ist hier ok"-Signal. Im Single-User-Kontext störend, aber nicht blockierend — alternativer Click-Pfad ist da. Auch ein reiner CSS-Bug (siehe Bug-Sektion). |
| F11 | Drop-Zone-Active-Highlight transparent (siehe F10) | Drag-State wird nicht für die Wiedererkennung dargestellt — der User muss sich auf das *erfolgreiche* Drop verlassen, statt während des Drags zu sehen, dass die Zone aktiv akzeptiert. | H6 | 2 | Geringes Gewicht im Single-User-Kontext (vertraut), aber die fehlende visuelle Bestätigung ist auch hier spürbar, gerade bei langsamen Drag-Bewegungen. |
| F12 | `#alert-container` ohne Close-Button und ohne Auto-Dismiss (Inventur #17) | Wenn ein Fehler-Banner gerendert wurde, bleibt er stehen — es gibt keinen Close-X und kein Timeout. Bei Wiederversuch und Erfolg liegt der alte Fehler-Banner weiterhin sichtbar als irreführender Status. | H1 | 2 | Status-Drift bei iterativem Workflow: Fehler beheben → erneut konvertieren → der alte Fehler steht noch. Single-User räumt manuell auf (Reload), aber Reibung bei jeder zweiten Konversion. |
| F13 | Drop-Zone kein Error-State für unsupported drag-type (Inventur #11) | Wird eine Datei mit unklarem Typ über die Zone gezogen oder gedropped, gibt das Frontend keinen Hinweis ("dieser Typ wird nicht unterstützt"). Erst der Backend-Roundtrip liefert eine generische 500er-Meldung. | H9 | 2 | Frontend-seitige Vorab-Validierung würde Recovery beschleunigen. Im Single-User-Kontext seltener relevant, weil Oliver weiß, was er hochlädt — aber der Failure-Pfad bleibt unnötig spät und unspezifisch. |
| F14 | Drop-Zone hat keine eigene Loading-Indikation während Upload (Inventur #11) | Der Submit-Button zeigt "Converting...", die Drop-Zone selbst bleibt unverändert. Bei großen Dateien fehlt das räumliche Feedback "hier passiert gerade etwas". | H1 | 2 | Indirekt über den Button kommuniziert, aber nicht ortsgleich. Im Single-User-Kontext akzeptabel — kein primärer Pain Point. |
| F15 | Kein Auto-Scroll-into-View beim Erscheinen von `#result-area` (Bemerkung B8; Inventur #20) | Nach erfolgreicher Conversion wird `#result-area` sichtbar gemacht, aber der Viewport scrollt nicht dorthin. Bei längeren Sidebars / kleineren Viewports liegt das Result unter der Falz. | H1 | 2 | Auf Olivers Mintbox-Desktop selten ein Problem, auf einem Laptop-Viewport (Halbierung etc.) jedoch schon. Niedrige bis mittlere Häufigkeit. |
| F16 | Drop-Zone Keyboard-unzugänglich (Bemerkung B5; Inventur #11/#12) | Drop-Zone hat weder `tabindex`, noch `role="button"`, noch `aria-label`. Das versteckte File-Input ist `display:none` und damit auch nicht fokussierbar. Es existiert **kein** rein-Keyboard-Pfad zum Datei-Picker. | H6 | 2 | Sidebar-Links sind tab-bar — Oliver erwartet symmetrisch, dass das primäre Form-Element im Hauptbereich ebenfalls per Tastatur erreichbar ist. Sev 2 (nicht 3), weil Maus-Pfad solide funktioniert. |
| F17 | Filename-Size-Anzeige immer in MB (Bemerkung B9; Inventur #14) | `(file.size / 1024 / 1024).toFixed(1) + " MB"` — eine 222-byte-Datei wird als "0.0 MB" angezeigt. Bricht die Konvention der meisten Datei-Manager (KB/MB-Fallback). | H4 | 1 | Im Single-User-Kontext kosmetisch — Oliver lädt fast nie Sub-MB-Dateien hoch. Konsistenzbruch mit Standard-Pattern aber für H4 dokumentationswürdig. |
| F18 | `#result-content` ohne a11y-Annotations (Inventur #24) | `<pre>` hat kein `tabindex=0`, kein `role`, kein `aria-label`. Das Result ist mit Screenreader / reinem Tastaturfokus nicht gezielt ansteuerbar. | H6 | 1 | Single-User-Kontext: kosmetisch. Oliver nutzt keinen Screenreader, würde den Fokus aber bei Tastaturbedienung manchmal direkt am Output erwarten. |
| F19 | "Download .md" ohne Success-Feedback (Inventur #22) | Click → Blob-Download startet sofort. Kein Toast / kein "✓ heruntergeladen"-Hint. Die Browser-Download-Bar ist die einzige Bestätigung. | H1 | 1 | Browser-Download-UI deckt die Bestätigung redundant ab. Im Single-User-Kontext kaum spürbar. |

---

## Reine Implementierungs-Bugs (kein eigenständiges Heuristik-Finding, separates Ticket-Material)

Diese Befunde sind in den Findings oben **bereits aus UX-Sicht erfasst**, brauchen aber zusätzlich konkrete Code-Fixes:

- **B1: CSS-Override / Variable-Resolution für `.drop-zone-active`.** Die deklarierte Box-Shadow-Farbe `rgba(168,180,216,0.3)` rendert als `rgba(0,0,0,0)`. Vermutlich CSS-Variable-Resolution-Bug oder Override an anderer Stelle in [static/css/style.css](static/css/style.css). → **siehe Findings F10, F11.**
- **B2: `.saved`-Klasse wird beim Reset des Save-Buttons nicht entfernt.** Drei Code-Stellen in [static/js/document_converter.js](static/js/document_converter.js): nach erfolgreicher neuer Conversion (Zeilen ~71–74), nach Save-Failure-Pfad (Zeile ~129) und implizit beim `clear-file`-Click (separater Bug B3). Reset-Routine sollte `saveBtn.classList.remove('saved')` ergänzen. → **siehe Findings F5, F6.**
- **B3: `clear-file`-Handler reseten weder `#result-area`, `#save-btn` noch das in-Memory `lastResult`.** Im Code-Stand fokussiert sich der Handler nur auf `fileInput.value=''` + `#file-info hidden`. Vollständiger Reset müsste `#result-area` ausblenden, `lastResult = null` setzen und den Save-Button-State (inkl. `.saved`-Klasse) zurücksetzen. → **siehe Findings F3, F4 (auch F5, F6 sekundär).**

---

## Zusammenfassung

- **Heuristik-Findings gesamt:** 19
- **Davon Schweregrad 4 (kritisch):** 2 (F1, F2 — beide zum Empty-Submit-Problem)
- **Davon Schweregrad 3:** 7 (F3–F9 — Result-Area-Stale, Save-Button-Stale, Save-Failure-Inkonsistenz, Format-Label-Mismatch)
- **Davon Schweregrad 2:** 7 (F10–F16 — Drag-State unsichtbar, Alert-Stale, fehlende Frontend-Validation, Loading-/Scroll-Feedback, Keyboard-Pfad)
- **Davon Schweregrad 1:** 3 (F17–F19 — Filename-Display, a11y-Annotations, Download-Feedback)
- **Reine Implementierungs-Bugs (mit Ticket-Material):** 3 (B1 CSS-Highlight, B2 Saved-Klasse, B3 Clear-File-Reset)

**Bemerkungen-Disposition (9 Stufe-1-Bemerkungen):**
- Findings only: 6 (B1 Empty-Submit, B4 Save-Failure-Inkonsistenz, B5 Keyboard, B6 Format-Mismatch, B8 Auto-Scroll, B9 Filename-MB)
- Beides (Finding + Bug-Ticket): 3 (B2 Drag-Highlight, B3 Save-Stale-Visual, B7 Result-Area-Persistenz)
- Bugs only: 0
- → 9 Findings, 3 davon mit zusätzlichem Bug-Ticket-Material

**Schweregrad-Skala:**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse)
