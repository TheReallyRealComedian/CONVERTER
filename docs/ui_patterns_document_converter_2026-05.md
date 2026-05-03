# UX-Patterns + Microcopy: document_converter (2026-05-03)

**Methodik:** Stufe 3 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Konkrete Patterns + DE-Microcopy auf Basis der Heuristik-Findings aus Stufe 2. Konsolidiert die 19 Stufe-2-Findings auf 14 Pattern-Blöcke (5 konsolidiert + 9 einzeln); Bug-Tickets B1/B2/B3 sind reine Code-Fixes ohne eigenen Microcopy-Bedarf.
**Quelle Findings:** [docs/ui_findings_document_converter_2026-05.md](ui_findings_document_converter_2026-05.md)
**Quelle Inventur:** [docs/ui_inventory_document_converter_2026-05.md](ui_inventory_document_converter_2026-05.md)
**Komponenten-Basis:** Existierende Neomorphism-Klassen aus [static/css/style.css](../static/css/style.css) — `c-btn`, `c-btn--primary`, `c-input`, `c-card`, `c-alert--danger/success/warning/info`, `c-drop-zone`, `.toast-notification`, `.save-library-btn`, `.saved`. Kein neues Design-System.

---

## Pattern-Blöcke

### Pattern 1: Empty-Submit liefert keinerlei Rückmeldung
**Adressiert Findings:** F1, F2 (Sev 4)

- **Pattern:** Inline-Validierung + Fehler-Banner über dem Form. Klick auf "Transform to Text" ohne ausgewählte Datei → roter `c-alert--danger`-Banner im `#alert-container` **und** roter Ring um die Drop-Zone (`.c-drop-zone--invalid`-State, gleiche Box-Shadow-Logik wie der Hover-State, nur in Danger-Tint). Banner verschwindet automatisch, sobald der User eine Datei auswählt.
- **Microcopy** (DE, situativ):
  - Banner: "Bitte zuerst eine Datei auswählen oder per Drag & Drop hineinziehen."
  - Drop-Zone-aria-Hint: "Pflichtfeld — keine Datei ausgewählt"
- **Gestaltung:** Banner oben (existierender `#alert-container`), Icon ✗, Danger-Tint (rot, passt zur `c-alert--danger`-Palette). Drop-Zone-Ring rot statt blau für ~2 s, Fade-out, sobald `change`-Event auf `#document_file` feuert. Fokus springt nach Banner-Render automatisch auf die Drop-Zone.
- **Aufwand:** S — Submit-Handler ergänzt Pre-Submit-Check, eine kleine CSS-Klasse `c-drop-zone--invalid` neu, ansonsten existierende Komponenten.

---

### Pattern 2: Result-Area persistiert nach Clear-Klick
**Adressiert Findings:** F3, F4 (Sev 3)
**Bug-Voraussetzung:** B3 (clear-file-Handler-Scope)

- **Pattern:** Vollständiger Reset-Sweep. Klick auf das "×" am File-Info räumt zusätzlich `#result-area` (auf `hidden`), `#alert-container`, das in-Memory `lastResult` und den Save-Button-State (Text, `disabled`, `.saved`-Klasse) ab. Visueller Reset spricht für sich — keine Begleit-Microcopy nötig.
- **Microcopy** (DE, situativ):
  - keine sichtbare Meldung. Aria-live-Hint für Screenreader-Pfad: "Auswahl entfernt, Ergebnis verworfen."
- **Gestaltung:** Ortsgleich beim Clear-Klick, kein zusätzlicher Toast (das Verschwinden ist selbsterklärend). Übergang `opacity` 150 ms, damit der Reset nicht hart wirkt.
- **Aufwand:** XS — nur den `clear-file`-Handler in [static/js/document_converter.js](../static/js/document_converter.js) erweitern.

---

### Pattern 3: Save-Button bleibt visuell "gespeichert" mit Text "Save to Library"
**Adressiert Findings:** F5, F6 (Sev 3)
**Bug-Voraussetzung:** B2 (`.saved`-Klasse-Reset)

- **Pattern:** Symmetrischer Status-Lifecycle für `.save-library-btn`. Vier explizite States, jede Reset-Routine räumt **alle** State-Marker (`disabled`, Text, `.saved`-Klasse) gemeinsam ab.
- **Microcopy** (DE, alle Button-Labels max 3 Wörter):
  - default: "In Library speichern"
  - loading: "Speichert …"
  - success: "✓ Gespeichert"
  - error → kehrt zu default zurück, Fehler über Pattern 4
- **Gestaltung:** Position unverändert (Action-Bar im Result-Header). Success-Tint grün (`var(--nm-tint-success)`, bereits in `.save-library-btn.saved`), Loading dezenter Spinner links vom Text. Reset-Trigger: neue erfolgreiche Conversion **und** Clear-Klick (Pattern 2).
- **Aufwand:** XS — Reset-Routine konsolidieren, eine kleine Helper-Funktion `resetSaveBtn()` in [static/js/document_converter.js](../static/js/document_converter.js).

---

### Pattern 4: Save-Failure nutzt Browser-`alert()` statt In-Page-Banner
**Adressiert Findings:** F7, F8 (Sev 3)

- **Pattern:** Inline `c-alert--danger`-Banner im `#alert-container`, identisch zum Conversion-Failure-Pfad. Beide Fehlerpfade derselben Seite verwenden dieselbe Komponente. Banner steht persistent (siehe Pattern 7 für Close + Auto-Dismiss).
- **Microcopy** (DE, max 2 Sätze):
  - generisch: "Speichern in die Library fehlgeschlagen. Bitte erneut versuchen."
  - mit Detail (falls Server-Hint vorhanden): "Speichern in die Library fehlgeschlagen: Titel ist leer. Bitte einen Titel eingeben."
- **Gestaltung:** Position oben (`#alert-container`), Icon ✗, Danger-Tint. Banner bleibt sichtbar bis User schließt oder Auto-Dismiss (Pattern 7) feuert. Save-Button kehrt in den default-State zurück (Pattern 3).
- **Aufwand:** S — `alert()`-Aufruf ersetzen, kleine Helper-Funktion `showAlert(level, msg)` in [static/js/document_converter.js](../static/js/document_converter.js) (oder im geteilten `_utils.js`).

---

### Pattern 5: Drag-Active-Highlight ist live transparent
**Adressiert Findings:** F10, F11 (Sev 2)
**Bug-Voraussetzung:** B1 (CSS-Override für `.drop-zone-active`)

- **Pattern:** Sichtbarer Drop-Active-State + textuelles Drop-Hint-Overlay. Während des Dragovers wechselt die Drop-Zone auf eine deutlichere Box-Shadow-/Border-Variante (Primary-Tint, nicht das transparente `rgba(0,0,0,0)` des Bugs) und blendet einen kurzen Text in die Mitte ein.
- **Microcopy** (DE):
  - Drag-Hint im Overlay: "Loslassen, um hochzuladen"
- **Gestaltung:** Position innerhalb der Drop-Zone, mittig. Farbe: Primary-Akzent (passend zur `c-btn--primary`-Palette), erhöhte Box-Shadow für "lift"-Wirkung — passt zur Neomorphism-Logik (raised statt pressed). Übergang 100 ms.
- **Aufwand:** S — CSS-Variable-Bug fixen (B1), Overlay-Span einblenden via existierender `.drop-zone-active`-Klasse.

---

### Pattern 6: Drop-Zone-Label und tatsächlich akzeptierte Formate divergieren
**Adressiert Findings:** F9 (Sev 3)

- **Pattern:** Ehrliches Label + `accept`-Attribut am File-Input. Frontend kommuniziert exakt die Formate, die das Backend tatsächlich annehmen *soll*; ein restriktives `accept` filtert die System-File-Picker-Liste vor; ein Server-side-Check bleibt als Backstop.
- **Microcopy** (DE):
  - Drop-Zone-Hint (Zeile 1, fett): "Datei hier ablegen oder klicken zum Auswählen"
  - Drop-Zone-Hint (Zeile 2, dezent): "PDF, DOCX, PPTX, EML, HTML, TXT, MD — max. 100 MB"
- **Gestaltung:** Position unverändert (zentrierter Hint in `c-drop-zone`). Sub-Hint dezenter (kleinere Schrift, gedämpfter Text-Tint). `accept=".pdf,.docx,.pptx,.eml,.html,.htm,.txt,.md"` am `#document_file`.
- **Aufwand:** S — Template-Anpassung + `accept`-Attribut + ggf. Backend-Whitelist auf dieselbe Liste verengen, sonst bleibt die Divergenz nur in eine Richtung gelöst.

---

### Pattern 7: Alert-Banner ohne Close-Button und ohne Auto-Dismiss
**Adressiert Findings:** F12 (Sev 2)

- **Pattern:** Close-Button (×) am `c-alert`-Banner + Auto-Dismiss für nicht-kritische Levels. Danger-Banner bleibt persistent (User soll lesen), Success/Info dismissen automatisch nach 6 s.
- **Microcopy** (DE):
  - Close-Button aria-label: "Meldung schließen"
  - keine sichtbare Begleit-Microcopy
- **Gestaltung:** Close-× rechts oben im Banner, Hover-Tint. Fade-out 200 ms beim Auto-Dismiss. Bei iterativem Workflow (neuer Submit) räumt der Submit-Handler den `#alert-container` ohnehin proaktiv ab.
- **Aufwand:** S — `c-alert`-Komponente um Close-× ergänzen (kleines CSS + ein Click-Handler), Helper-Funktion mit Timeout-Param.

---

### Pattern 8: Drop-Zone gibt Frontend-seitig keinen Hint bei unsupported drag-type
**Adressiert Findings:** F13 (Sev 2)

- **Pattern:** Frontend-Vorab-Validierung gegen die `accept`-Liste (Pattern 6). Beim Drop einer unsupported Datei → Banner statt Backend-Roundtrip. Während des Dragovers, wenn Browser den Typ schon kennt, wechselt die Drop-Zone zusätzlich in einen Warning-Tint statt Primary.
- **Microcopy** (DE, max 2 Sätze):
  - Banner: "Dieser Dateityp wird nicht unterstützt. Erlaubt: PDF, DOCX, PPTX, EML, HTML, TXT, MD."
  - Drag-Hint im Warning-State: "Dateityp nicht unterstützt"
- **Gestaltung:** Banner über Pattern 4 / `c-alert--warning` (orange-Tint). Drag-Hint kurz, klein, in der Drop-Zone-Mitte sichtbar.
- **Aufwand:** S — Drop-Handler erweitert um `file.name`-Endung-Check + Warning-Banner-Trigger.

---

### Pattern 9: Drop-Zone selbst hat während Upload keine Loading-Indikation
**Adressiert Findings:** F14 (Sev 2)

- **Pattern:** Skeleton-/Progress-State innerhalb der Drop-Zone. Während `fetch` läuft, ersetzt die Drop-Zone ihren Inhalt durch eine Inline-Status-Box (Spinner + Status-Text). Submit-Button bleibt parallel im "Converting…"-State.
- **Microcopy** (DE):
  - Phase 1: "Datei wird hochgeladen …"
  - Phase 2: "Konvertierung läuft …"
- **Gestaltung:** Spinner zentriert, Status-Text darunter, neutrale graue Tint. Übergang `opacity` 200 ms zum Drop-Zone-Default-State, sobald Result eintrifft. Kein Progress-Prozent, weil Backend keinen liefert.
- **Aufwand:** M — neue Inline-Loading-Komponente innerhalb der Drop-Zone, Phasen-Wechsel auf `xhr.upload`-Progress-Event vs. nach Request-Send. Aufwand `M`, weil die zwei-Phasen-Logik einen kleinen State-Machine-Aufsatz braucht.

---

### Pattern 10: Result-Area scrollt nicht in den Viewport
**Adressiert Findings:** F15 (Sev 2)

- **Pattern:** `scrollIntoView({ behavior: "smooth", block: "start" })` auf `#result-area` direkt nach dem Sichtbar-Schalten. Nur bei erfolgreicher Conversion, nicht bei Fehlern (Banner ist oben).
- **Microcopy** (DE):
  - keine — visuelles Verhalten genügt.
- **Gestaltung:** Smooth-Scroll, Result-Header landet oben am Viewport. Bei sehr kurzen Results (oder Viewports, in denen das Result eh sichtbar ist) ist `scrollIntoView` ein No-Op — kein Spezialfall nötig.
- **Aufwand:** XS — eine Zeile in [static/js/document_converter.js](../static/js/document_converter.js) im Success-Branch.

---

### Pattern 11: Drop-Zone ist nicht per Tastatur erreichbar
**Adressiert Findings:** F16 (Sev 2)

- **Pattern:** Drop-Zone als interaktives Element annotieren (`role="button"`, `tabindex="0"`, `aria-label`) + Keyhandler für Enter/Space, der den File-Picker öffnet. Damit ist der Maus-Pfad symmetrisch zum Tastatur-Pfad.
- **Microcopy** (DE):
  - aria-label: "Datei auswählen — PDF, DOCX, PPTX, EML, HTML, TXT, MD bis 100 MB"
  - sichtbarer Hint unverändert (siehe Pattern 6)
- **Gestaltung:** Sichtbarer Focus-Ring auf der Drop-Zone (Neomorphism-Outline, passend zu `:focus-visible` an `c-btn`). Kein eigener Visual-Hint, weil Focus-Ring + bestehender Hint-Text genügen.
- **Aufwand:** S — Template-Attribute + Keyhandler + ein kleiner CSS-Block für `.c-drop-zone:focus-visible`.

---

### Pattern 12: Filename-Size immer in MB, auch bei Sub-MB-Dateien
**Adressiert Findings:** F17 (Sev 1)

- **Pattern:** Conditional Unit-Display — Bytes < 1 KB als `B`, < 1 MB als `KB`, sonst `MB`. Eine Stelle nach Komma, Komma als Dezimal-Trenner (DE-Locale).
- **Microcopy** (DE, Beispiele):
  - 222 B → "222 B"
  - 4 731 B → "4,6 KB"
  - 1 234 567 B → "1,2 MB"
- **Gestaltung:** Position unverändert (im `#file-info`). Keine zusätzliche Komponente.
- **Aufwand:** XS — kleiner Helper `formatFileSize(bytes)` in [static/js/document_converter.js](../static/js/document_converter.js).

---

### Pattern 13: Result-Content (`<pre>`) ohne a11y-Annotations
**Adressiert Findings:** F18 (Sev 1)

- **Pattern:** `tabindex="0"`, `role="region"`, `aria-label` am Result-`<pre>`. Damit ist das Result per Tastatur direkt fokussierbar und Screenreader-ansagbar.
- **Microcopy** (DE):
  - aria-label: "Konvertierter Markdown-Text"
- **Gestaltung:** Sichtbarer Focus-Ring entsprechend Neomorphism-`:focus-visible`. Keine sichtbare Microcopy.
- **Aufwand:** XS — drei Attribute am Template.

---

### Pattern 14: "Download .md" ohne Success-Feedback
**Adressiert Findings:** F19 (Sev 1)

- **Pattern:** Toast-Notification (existiert als `.toast-notification` im CSS) für 2,5 s nach Klick.
- **Microcopy** (DE, max 3 Wörter):
  - "✓ Markdown heruntergeladen"
- **Gestaltung:** Position unten-rechts (existierende Toast-Defaults), Success-Tint grün, Fade-in/out. Browser-Download-Bar deckt die Bestätigung redundant ab — der Toast ist primär für die in-page-Continuity gedacht (User bleibt im App-Kontext und sieht die Aktion bestätigt).
- **Aufwand:** S — kleine Helper-Funktion `showToast(msg)`, weil aktuell nur die CSS-Klasse existiert, aber keine JS-Logik (Container anlegen, `.show` toggeln, Timeout).

---

## Bug-Tickets (kein Pattern-Bedarf)

- **B1: CSS-Override für `.drop-zone-active`** — siehe Pattern 5 (Bug-Voraussetzung). Code-Fix in [static/css/style.css](../static/css/style.css). Vermutlich CSS-Variable-Resolution oder konkurrierende Spezifität.
- **B2: `.saved`-Klasse-Reset im Save-Btn-Reset** — siehe Pattern 3 (Bug-Voraussetzung). Code-Fix in [static/js/document_converter.js](../static/js/document_converter.js): drei Reset-Stellen ergänzen `saveBtn.classList.remove('saved')`. Pattern 3 zentralisiert das in einer Helper-Funktion und löst den Bug damit strukturell mit.
- **B3: clear-file-Handler-Scope** — siehe Pattern 2 (Bug-Voraussetzung). Code-Fix in [static/js/document_converter.js](../static/js/document_converter.js): Handler räumt `#result-area`, `lastResult` und Save-Button-State zusätzlich zu `fileInput.value` und `#file-info` ab. Pattern 2 ist die UX-Beschreibung des Fix.

---

## Priorisierung

**Aufwand-Gewicht für Impact-Score:** XS=1, S=2, M=4, L=8. Score = Schweregrad × 5 / Aufwand-Gewicht. Höher = besser.

| Rang | Pattern # | Adressiert | Schweregrad | Aufwand | Impact-Score | Quick-Win |
|------|-----------|------------|-------------|---------|--------------|-----------|
| 1 | Pattern 2 | F3, F4 — Result-Persistenz nach Clear | 3 | XS | 15.0 | ★ Top-5 |
| 2 | Pattern 3 | F5, F6 — Save-Btn Stale-Visual | 3 | XS | 15.0 | ★ Top-5 |
| 3 | Pattern 1 | F1, F2 — Empty-Submit silent | 4 | S | 10.0 | ★ Top-5 |
| 4 | Pattern 10 | F15 — Auto-Scroll auf Result | 2 | XS | 10.0 | ★ Top-5 |
| 5 | Pattern 4 | F7, F8 — Save-Failure-Banner | 3 | S | 7.5 | ★ Top-5 |
| 6 | Pattern 6 | F9 — Format-Label-Mismatch | 3 | S | 7.5 | |
| 7 | Pattern 5 | F10, F11 — Drag-Highlight | 2 | S | 5.0 | |
| 8 | Pattern 7 | F12 — Alert-Close + Auto-Dismiss | 2 | S | 5.0 | |
| 9 | Pattern 8 | F13 — Drop-Zone unsupported drag-type | 2 | S | 5.0 | |
| 10 | Pattern 11 | F16 — Drop-Zone Keyboard | 2 | S | 5.0 | |
| 11 | Pattern 12 | F17 — Filename KB/MB-Fallback | 1 | XS | 5.0 | |
| 12 | Pattern 13 | F18 — Result-Content a11y | 1 | XS | 5.0 | |
| 13 | Pattern 9 | F14 — Drop-Zone Loading-Indikation | 2 | M | 2.5 | |
| 14 | Pattern 14 | F19 — Download Success-Toast | 1 | S | 2.5 | |

**Top-5 Quick-Wins:**

1. **Pattern 2 — Vollständiger Clear-Reset** (Score 15.0): UX-Konsistenz "ich habe geclärt → alles weg" mit minimalem Handler-Diff. Schließt B3 mit.
2. **Pattern 3 — Symmetrischer Save-Button-Lifecycle** (Score 15.0): zentrale `resetSaveBtn()`-Helper-Funktion löst die Stale-Visual-Asymmetrie und schließt B2 mit.
3. **Pattern 1 — Empty-Submit-Banner + Drop-Zone-Highlight** (Score 10.0): höchster Schweregrad (4) der gesamten Stage. Existierende `c-alert--danger`-Komponente reuse, eine neue `--invalid`-CSS-Klasse für die Drop-Zone.
4. **Pattern 10 — `scrollIntoView` auf Result-Area** (Score 10.0): einzeiliger Fix mit klar spürbarem Effekt auf kleineren Viewports.
5. **Pattern 4 — Save-Failure-Banner statt `alert()`** (Score 7.5): konsistente Fehler-UX im Feature, blockiert keinen Tab mehr, lesbar nach Klick. Synergiert mit Pattern 7 (Close + Auto-Dismiss).
