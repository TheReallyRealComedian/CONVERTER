# UX-Patterns + Microcopy: audio_converter (2026-05-03)

**Methodik:** Stufe 3 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Konkrete Patterns + DE-Microcopy auf Basis der Heuristik-Findings aus Stufe 2.
**Quelle Findings:** [docs/ui_findings_audio_converter_2026-05.md](ui_findings_audio_converter_2026-05.md)
**Quelle Inventur:** [docs/ui_inventory_audio_converter_2026-05.md](ui_inventory_audio_converter_2026-05.md)
**F-1 Patterns als Referenz:** [docs/ui_patterns_document_converter_2026-05.md](ui_patterns_document_converter_2026-05.md)
**Helper-API:** [static/js/_utils.js](../static/js/_utils.js) — `showAlert(containerEl, level, msg, options?)`, `showToast(msg, options?)`, `formatFileSize(bytes)`, `safeJSON(response)`
**Komponenten-Basis (post-F-1):** `c-btn`, `c-btn--primary`, `c-input`, `c-card`, `c-alert--danger/success/warning/info` (mit Close-Button + Auto-Dismiss aus Cluster C), `c-drop-zone` (mit `--invalid`/`--warning`/`--loading`-States), `.toast-notification`, `.save-library-btn`, `.saved`, `:focus-visible` für `c-btn` / `.c-drop-zone`.

**Konsolidierung:** 32 Findings → 21 Pattern-Blöcke. 13 Cross-Feature-H4-Findings sind zu vier Konvergenz-Blöcken (P5 Alert-Sweep, P6 Save-Lifecycle, P12 DE-Microcopy, P19 formatFileSize) gebündelt; Surprise-5-Familie (Silent-Failure) bleibt aus Fix-Gründen über drei Blöcke (P1, P3, P9) verteilt, im Cluster-Layout aber als kohärente Sub-Gruppe markiert; Surprise 6 (Live-Textarea Append-Race) ist eigener Block P10 mit expliziter Konflikt-Resolution.

---

## Pattern-Blöcke

### Pattern 1: Empty-Submit auf File-Tab liefert keinerlei Rückmeldung
**Adressiert Findings:** F1 (H1 Sev 4), F2 (H9 Sev 4)
**Cross-Feature-Konvergenz:** Reuse von F-1 Pattern 1 (`showAlert` + `c-drop-zone--invalid`)
**Surprise-5-Familie:** zusammen mit P3, P9 (silent-fail-Anti-Pattern)

- **Pattern:** Pre-Submit-Validierung im Submit-Handler von `#transcribe-form`. Klick auf "Datei umwandeln" ohne ausgewählte Datei → roter `c-alert--danger`-Banner im neuen `#audio-alert-container` (siehe P4) **und** `.c-drop-zone--invalid`-Klasse für ~2 s. Banner verschwindet automatisch beim nächsten `change`-Event auf `#file-upload-input`.
- **Microcopy** (DE, Du-Form):
  - Banner: "Bitte zuerst eine Audio-Datei auswählen oder per Drag & Drop hineinziehen."
- **Gestaltung:** Banner oben im File-Pane, Icon ✗, Danger-Tint. Drop-Zone-Ring rot statt blau, Fade-out 200 ms beim Reset. Fokus springt nach Banner-Render auf die Drop-Zone.
- **Aufwand:** S — Submit-Handler ergänzt Pre-Submit-Check, kein neues CSS (`c-drop-zone--invalid` existiert).

---

### Pattern 2: Drag-Drop wird beworben, ist aber nicht implementiert
**Adressiert Findings:** F3 (H6 Sev 4), F4 (H4 Sev 4)
**Bug-Voraussetzung:** B1 (Drag-Drop-Handler fehlen)
**Cross-Feature-Konvergenz:** Reuse von F-1-Baseline (echtes Drag-Drop) + Cluster D (Drag-MIME-Warning, `c-drop-zone--warning`)

- **Pattern:** Echte `dragover/dragleave/drop`-Handler auf der Drop-Zone, analog zu `document_converter.js`. `dragover` aktiviert `.drop-zone-active` (Lift-Tint). Bei `drop`: File in den hidden `#file-upload-input` schreiben + `change`-Event dispatchen → Filename-Anzeige läuft wie Click-Pfad. Best-Effort MIME-Check während `dragover` toggelt `.c-drop-zone--warning` für nicht-`audio/*`-Typen; verlässliche Validierung beim Drop selbst (siehe P13 Backend-Whitelist).
- **Microcopy** (DE, Du-Form):
  - Drop-Zone-Hint default (Zeile 1): "Audio-Datei hier ablegen oder klicken zum Auswählen"
  - Drop-Zone-Hint default (Zeile 2, dezent): "MP3, WAV, M4A, OGG, FLAC, WEBM — max. 500 MB"
  - Drag-Active-Overlay: "Loslassen, um hochzuladen"
  - Drag-Warning-Overlay: "Dateityp nicht unterstützt"
- **Gestaltung:** Drag-Active wie F-1 (Primary-Tint, Lift-Box-Shadow). Warning-Overlay orange-Tint statt blau. Übergang 100 ms.
- **Aufwand:** S — Handler-Set + Drag-Hint-Span + Template-Texte; CSS-States existieren.

---

### Pattern 3: Mic-Permission-Denied silent
**Adressiert Findings:** F5 (H9 Sev 4)
**Bug-Voraussetzung:** B3 (`getUserMedia` ohne `.catch()`)
**Cross-Feature-Konvergenz:** Reuse von `showAlert`
**Surprise-5-Familie:** zusammen mit P1, P9

- **Pattern:** Explizites `.catch()` an `getUserMedia(...)` differenziert nach `error.name`. Bei `NotAllowedError`/`PermissionDeniedError` → persistenter `c-alert--danger`-Banner im neuen `#live-alert-container` (siehe P4) mit Recovery-Hinweis; bei `NotFoundError` → eigener Banner; sonst generischer Fallback. WS wird in jedem Fall geschlossen, Mic-Button kehrt in Idle-State zurück (siehe P7 für Loading-State-Reset).
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - `NotAllowedError`: "Mikrofon-Zugriff blockiert. Erlaube den Zugriff in den Browser-Site-Einstellungen und versuche es erneut."
  - `NotFoundError`: "Kein Mikrofon gefunden. Schließe ein Aufnahmegerät an und versuche es erneut."
  - generisch: "Mikrofon konnte nicht gestartet werden. Browser-Berechtigung und angeschlossenes Gerät prüfen."
- **Gestaltung:** Banner persistent (Danger-Default), schließbar per ×. Kein Auto-Dismiss, weil Recovery-Aktion nötig ist.
- **Aufwand:** S — `.catch()` + drei `error.name`-Branches + Banner-Calls.

---

### Pattern 4: Banner-Mountpoint im Template (struktureller Vorbedingung-Fix)
**Adressiert Findings:** F27 (H4 Sev 2)
**Cross-Feature-Konvergenz:** Reuse von F-1 Cluster B (`#alert-container` als Banner-Mountpoint)

- **Pattern:** Drei Banner-Container ergänzen — pro Tab-Pane einen, damit Banner kontextual angezeigt werden und beim Tab-Wechsel nicht über alle Panes "wandern". `#live-alert-container`, `#file-alert-container`, `#podcast-alert-container` jeweils direkt unter dem Tab-Heading. Strukturelle Voraussetzung für P1, P3, P5, P8.
- **Microcopy** — keine (struktureller Container ohne sichtbare Microcopy).
- **Gestaltung:** Container leer, kein Spacing wenn ohne Inhalt. Banner-Höhe übernimmt der `c-alert`-Block.
- **Aufwand:** XS — drei `<div>`-Tags im Template.

---

### Pattern 5: Alert-Konsolidierung — `alert()`/Inline-Span/raw-`c-alert` ersetzen
**Adressiert Findings:** F6 (H4 Sev 3), F7 (H9 Sev 3), F8 (H4 Sev 3), F11 (H4 Sev 3)
**Bug-Voraussetzung:** B6 (Inline-`innerHTML`-XSS-Risiko durch raw `err.message`)
**Cross-Feature-Konvergenz:** Reuse von `showAlert(containerEl, level, msg, options?)` aus `_utils.js`

- **Pattern:** **Alle** 11+ `alert()`-Call-Sites + die Inline-`<span>`-Variante in `#transcription-result-container` durchgehend auf `showAlert(container, level, message)` umstellen. Container ist der jeweils passende Mountpoint aus P4 (live/file/podcast). Drei Levels: `danger` (Failures, persistent), `warning` (Validation/Empty-Submit, Auto-Dismiss), `info` (Hinweise). Save-Failures landen im selben Container wie der Save-Button.
- **Microcopy** (DE, Du-Form, max 2 Sätze pro Banner):
  - WS/Token-Failure: "Verbindung zur Transkription fehlgeschlagen. Netzwerk und API-Konfiguration prüfen."
  - Token-Fail spezifisch: "API-Token konnte nicht abgerufen werden. Server-Konfiguration prüfen."
  - Empty-Submit Live (Save ohne Inhalt): "Es gibt nichts zu speichern. Erst eine Live-Transkription aufnehmen."
  - Empty-Submit Script-Generation: "Bitte erst Quelltext im Feld oben eintragen."
  - Empty-Submit Podcast: "Bitte erst ein Skript eintragen oder generieren lassen."
  - Script-Parse-Failure: "Skript konnte nicht gelesen werden. Format prüfen: `Sprecher [stil]: Text`."
  - Conversion-Failure (File-Tab): "Transkription fehlgeschlagen. Datei prüfen und erneut versuchen."
  - Save-Failure: "Speichern in die Library fehlgeschlagen. Bitte erneut versuchen."
  - Save-Failure mit Server-Hint: "Speichern in die Library fehlgeschlagen: {hint}."
  - Copy-Failure: "Kopieren in die Zwischenablage fehlgeschlagen."
  - Copy-Empty (siehe P11): "Es gibt nichts zu kopieren."
  - Script-Generation-Failure: "Skript-Generierung fehlgeschlagen. Bitte erneut versuchen."
  - Podcast-Generation-Failure: "Podcast-Generierung fehlgeschlagen. Bitte erneut versuchen."
- **Gestaltung:** Banner persistent für `danger`, Auto-Dismiss (6 s) für `warning`/`info` — Default-Verhalten von `showAlert`. `err.message` wird via `textContent` interpoliert (XSS-sicher per Helper-Konvention).
- **Aufwand:** S — Volumen ist hoch (11+ Call-Sites), aber jede Call-Site ist ein 1:1-Replace.

---

### Pattern 6: Save-Button-Lifecycle (`.saved`-Klasse-Reset)
**Adressiert Findings:** F9 (H1 Sev 3), F10 (H4 Sev 3)
**Bug-Voraussetzung:** B2 (`.saved`-Klasse wird beim Reset nicht entfernt)
**Cross-Feature-Konvergenz:** Reuse von F-1 Cluster A Pattern 3 (`resetSaveBtn(btn, defaultText)` Helper-Logik)

- **Pattern:** Symmetrischer Lifecycle-Helper für beide Save-Buttons (`#save-transcription-btn`, `#save-live-btn`). Vier explizite States, jede Reset-Routine räumt **alle** State-Marker (`disabled`, Text, `.saved`-Klasse) gemeinsam ab. Reset-Trigger: neue Transkription (File-Pfad), neuer Recording-Start (Live-Pfad), Clear-Klick. Der Helper kann inline in `audio_converter.js` leben oder analog zu F-1's Implementation in `_utils.js` wandern, wenn beim DE-Microcopy-Pass (P12) ohnehin angefasst wird.
- **Microcopy** (DE, Buttons max 3 Wörter):
  - default: "In Library speichern"
  - loading: "Speichert …"
  - success: "✓ Gespeichert"
  - error → kehrt zu default zurück, Fehler über P5
- **Gestaltung:** Position unverändert. Success-Tint grün (`.saved` existiert). Reset-Trigger zentralisiert.
- **Aufwand:** XS — Helper extrahieren, drei Reset-Stellen in `audio_converter.js`.

---

### Pattern 7: Mic-Button kein Loading-State zwischen Click und WS-onopen
**Adressiert Findings:** F12 (H1 Sev 3)
**Cross-Feature-Konvergenz:** Reuse-Logik aus F-1 Polish-2 (P9 Drop-Zone-Loading) — Mic-Button braucht analoges Pattern

- **Pattern:** Click → setze `.mic-loading`-Klasse + `disabled=true`. CSS-only Spinner-Overlay (analog F-1 Cluster-C-Pattern). Drei Phasen optisch zusammengefasst (Token-Fetch + WS-Handshake + getUserMedia-prompt). Klasse wird entfernt sobald `socket.onopen` feuert (Übergang in `.recording`) **oder** im `.catch()`-Pfad (siehe P3) **oder** im `socket.onerror`-Pfad (Fallback). Während Loading: `pointer-events: none` verhindert Doppel-Clicks.
- **Microcopy** (DE, Du-Form):
  - sichtbarer Tooltip während Loading: "Verbindung wird aufgebaut …"
  - aria-label während Loading: "Verbindung zur Transkription wird aufgebaut"
- **Gestaltung:** Spinner-Ring um Mic-Icon (Primary-Tint), nicht blockierender Overlay. Übergang 150 ms zum `.recording`-State.
- **Aufwand:** S — neue CSS-Klasse `.mic-loading` + Toggle-Logik an drei Stellen (onopen, onerror, .catch).

---

### Pattern 8: Configuration-Error blockiert disproportional die ganze Seite
**Adressiert Findings:** F13 (H1 Sev 3), F14 (H9 Sev 3)
**Bug-Voraussetzung:** keine

- **Pattern:** Backend-Templating in `app_pkg/audio.py` reicht **zwei separate Flags** an das Template (`deepgram_api_key_set`, `gemini_api_key_set`). Template rendert pro Tab nur dann einen Banner-Hinweis statt Pane-Hide, wenn der jeweilige Service fehlt. Live + File brauchen Deepgram, Podcast braucht Gemini — die Tabs werden unabhängig deaktiviert. Der bisherige globale Page-Block fällt weg.
- **Microcopy** (DE, Du-Form, max 2 Sätze pro Banner):
  - Deepgram fehlt (im Live + File-Pane): "Transkription nicht verfügbar — Deepgram-API-Key fehlt. Setze `DEEPGRAM_API_KEY` in der `.env` und starte den Container neu."
  - Gemini fehlt (im Podcast-Pane): "Podcast-Generierung nicht verfügbar — Gemini-API-Key fehlt. Setze `GEMINI_API_KEY` in der `.env` und starte den Container neu."
  - Tab-Disabled-State (zusätzlich am Tab-Button): aria-disabled + Tooltip "Service nicht konfiguriert"
- **Gestaltung:** Banner im jeweiligen Pane oben, Icon ⚠, Warning-Tint (kein Danger — Recovery ist möglich). Andere Tabs bleiben voll funktional.
- **Aufwand:** S — Backend-Flag-Split + Template-Conditional-Wrap pro Pane + Tab-Disabled-Logik.

---

### Pattern 9: Lang-Buttons sichtbar disabled während Recording
**Adressiert Findings:** F15 (H1 Sev 3)
**Surprise-5-Familie:** zusammen mit P1, P3 (silent-fail-Anti-Pattern)

- **Pattern:** CSS-Rule `.lang-btn:disabled` mit reduzierter Opacity, `cursor: not-allowed` und visuell unterdrücktem Hover-Style. Sichtbares Locking-Symbol oder dezenter Tooltip beim Hover-Versuch macht den Recording-Lock erkennbar.
- **Microcopy** (DE, Du-Form):
  - Tooltip auf disabled-Lang-Btn (während Recording): "Während der Aufnahme gesperrt"
  - aria-label-Suffix: "(während der Aufnahme gesperrt)"
- **Gestaltung:** Opacity 0.5, `cursor: not-allowed`, Hover-Shadow off. Übergang 150 ms zwischen enabled/disabled.
- **Aufwand:** XS — eine CSS-Rule + Tooltip-Attribut bei `disabled=true`-Setzen.

---

### Pattern 10: Live-Textarea Append-Race überschreibt User-Edits (Surprise 6)
**Adressiert Findings:** F16 (H1 Sev 3)
**Bug-Voraussetzung:** B5 (`socket.onmessage` überschreibt User-Edits)

- **Pattern:** **Konflikt-Resolution-Strategie: Stream hat Vorrang während Recording, User-Edit nur nach Stop.** Während aktivem WS ist die Textarea `readonly` — der Stream-Append-Pfad kann konsistent in `baseText + transcript` schreiben, ohne User-Edits zu zerstören. Sobald `stopRecording()` läuft, wird `readonly` entfernt und der User kann nachträglich korrigieren. State-Wechsel mit klarem Visual-Hint.
- **Microcopy** (DE, Du-Form):
  - Placeholder während Recording: "Transkription läuft — bearbeitbar nach Stop"
  - Tooltip auf der disabled Textarea während Recording: "Während der Aufnahme schreibgeschützt"
  - Subtiler Status-Hint unter dem Textarea (z.B. dezent kursiv): "Live-Update läuft — Stop drücken, um zu bearbeiten"
- **Gestaltung:** `readonly`-Style: graue Border-Tint, Cursor unverändert (User soll lesen können). Übergang zum editierbaren State per Border-Color-Pulse 200 ms beim Stop.
- **Aufwand:** S — `readonly`-Toggle in `connectToDeepgram()`/`stopRecording()`, kleiner CSS-Block, Status-Hint-Span.

---

### Pattern 11: Result-Container Copy-Bug (Placeholder wird kopiert)
**Adressiert Findings:** F17 (H4 Sev 3)
**Bug-Voraussetzung:** B4 (Copy-Guard greift nicht beim Placeholder)

- **Pattern:** Sentinel-basierte Empty-Detection. Placeholder-Span bekommt `data-placeholder="true"`. Copy-Handler prüft: wenn Container nur ein Element mit diesem Attribut enthält (oder `.text-neo-faint` ist) → Empty-Pfad mit `showAlert(container, 'warning', 'Es gibt nichts zu kopieren.')`. Bei echter Transkription wird der Placeholder via `textContent`-Replace überschrieben (Sentinel verschwindet).
- **Microcopy** (DE, Du-Form):
  - Empty-Banner (siehe P5): "Es gibt nichts zu kopieren."
- **Gestaltung:** Visuell identisch zum Placeholder vorher (`.text-neo-faint`). Banner statt `alert()` (P5).
- **Aufwand:** XS — `data-placeholder`-Attribut + Guard-Check.

---

### Pattern 12: DE-Microcopy-Pass flächendeckend (inkl. Save-Live-Title-Format)
**Adressiert Findings:** F18 (H4 Sev 3), F30 (H4 Sev 1)
**Cross-Feature-Konvergenz:** Reuse von F-1 Cluster Polish-1 (DE-Microcopy-Konvention, Du-Form, Verb+Objekt-Buttons)

- **Pattern:** Vollständige Übersetzung aller sichtbaren Strings im Template + JS auf DE/Du-Form. Konstanten-Tabelle unten. Save-Live-Title bekommt explizites `Intl.DateTimeFormat('de-DE', { dateStyle: 'short', timeStyle: 'short' })` statt locale-abhängigem `toLocaleDateString()`. Alle Banner-/Toast-Texte aus P5 fallen ohnehin in DE an.
- **Microcopy** (Konstanten-Tabelle, EN → DE):

| Position | EN | DE |
|----------|----|----|
| Tab "Live Transcription" | Live Transcription | Live-Transkription |
| Tab "Transcribe File" | Transcribe File | Datei umwandeln |
| Tab "Text to Podcast" | Text to Podcast | Text zu Podcast |
| Lang-Btn | English | Englisch |
| Lang-Btn | German | Deutsch |
| Live-Textarea-Placeholder | Live transcription will appear here... | Live-Transkription erscheint hier … |
| Live-Btn | Copy | Kopieren |
| Live-Btn (success) | Copied! | ✓ Kopiert |
| Live-Btn | Clear | Leeren |
| Live-Save-Btn (default) | Save to Library | In Library speichern |
| Live-Save-Btn (loading) | Saving... | Speichert … |
| Live-Save-Btn (success) | Saved! | ✓ Gespeichert |
| File-Drop-Zone-Hint | Click to select an audio file or drag and drop | Audio-Datei hier ablegen oder klicken zum Auswählen |
| File-Submit-Btn | Transcribe File | Datei umwandeln |
| File-Submit-Btn (loading) | Transcribing... | Wird umgewandelt … |
| File-Result-Heading | Transcription Result: | Transkriptions-Ergebnis: |
| File-Result-Placeholder | The transcription will appear here after processing. | Transkription erscheint hier nach der Verarbeitung. |
| File-Save-Btn (default) | Save to Library | In Library speichern |
| Podcast-Mode-Monolog | Monolog / Single narrator (Kate) | Monolog / einzelne Stimme (Kate) |
| Podcast-Mode-Dialogue | Dialogue / Conversation (Kate & Max) | Dialog / Gespräch (Kate & Max) |
| Podcast-Lang-Label | Language | Sprache |
| Podcast-Style-Label | Narration Style | Sprech-Stil |
| Podcast-TTS-Label | TTS Model | TTS-Modell |
| Podcast-Source-Label | Source Text | Quelltext |
| Podcast-Generate-Script-Btn | Generate Script from Text Above | Skript aus Quelltext generieren |
| Podcast-Generate-Script-Btn (loading) | Generating... | Generiert … |
| Podcast-Script-Label | Podcast Script | Podcast-Skript |
| Podcast-Toggle | Advanced: Custom Prompt | Erweitert: Eigener Prompt |
| Podcast-Reset-Prompt-Btn | Reset to Default | Auf Standard zurücksetzen |
| Podcast-Generate-Btn | Generate Podcast | Podcast generieren |
| Podcast-Generate-Btn (loading) | Generating... ({n}s) | Generiert … ({n} s) |
| Podcast-Result-Heading | Your Podcast: | Dein Podcast: |
| Podcast-Download-Btn | Download | Herunterladen |
| Save-Live-Title-Format | `Live Transcription DD.MM.YYYY HH:MM` | `Live-Transkription {Intl.DateTimeFormat('de-DE') Datum + Zeit}` |

- **Gestaltung:** keine — reine String-Substitution.
- **Aufwand:** S — Volumen ist hoch, aber jede Stelle ist ein 1:1-Replace. Template + zwei JS-Dateien + ein Helper-Aufruf für Date-Format.

---

### Pattern 13: Backend-Whitelist + DE-Fehler-JSON für unsupported audio file
**Adressiert Findings:** F19 (H9 Sev 2)
**Bug-Voraussetzung:** B9 (Filesize-Pre-Check fehlt — sinnvoll mit P13 zu bündeln)
**Cross-Feature-Konvergenz:** Reuse von F-1 Cluster D (F-006-Fix für `document_converter`)

- **Pattern:** Single-Source-of-Truth `ACCEPTED_AUDIO_EXTENSIONS` in `app_pkg/audio.py` (z.B. `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.webm`). Fließt nach Template (`accept`-Attribut + `window.PageData.acceptedExtensions`) und JS. Backend liefert 400 mit DE-JSON-Body bei unsupported Extension. Frontend macht Pre-Submit-Filesize-Check gegen `MAX_FILE_SIZE_MB=500` und blockt frühzeitig mit Banner statt 500-Roundtrip.
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Backend-400-Antwort: "Dieses Dateiformat wird nicht unterstützt. Erlaubt: MP3, WAV, M4A, OGG, FLAC, WEBM."
  - Frontend-Pre-Submit (Größe): "Datei ist zu groß. Maximum: 500 MB."
  - Frontend-Pre-Submit (Format): "Dieses Dateiformat wird nicht unterstützt. Erlaubt: MP3, WAV, M4A, OGG, FLAC, WEBM."
- **Gestaltung:** Banner via `showAlert(fileAlertContainer, 'danger', msg)`. Drag-Pfad triggert `c-drop-zone--warning` (P2).
- **Aufwand:** S — Whitelist-Konstante + Backend-Check + Frontend-Filesize-Check + Template-`accept`-Attribut.

---

### Pattern 14: Mic-Button `:focus-visible` + a11y-Annotations
**Adressiert Findings:** F20 (H4 Sev 2), F25 (H6 Sev 2)
**Cross-Feature-Konvergenz:** Reuse von F-1 Cluster E (`:focus-visible`-Vokabular)

- **Pattern:** `#mic-button:focus-visible`-CSS (Outline-Ring im Primary-Tint) + `aria-label` + `aria-pressed` (Toggle-Konvention für Idle ↔ Recording) + `aria-hidden="true"` auf dem inneren SVG. `title` bleibt für Maus-Hover.
- **Microcopy** (DE, Du-Form):
  - aria-label idle: "Aufnahme starten"
  - aria-label recording: "Aufnahme stoppen"
  - aria-pressed: `false` (idle) / `true` (recording)
  - title (Hover): "Aufnahme starten" / "Aufnahme stoppen"
- **Gestaltung:** Focus-Ring analog `c-btn:focus-visible`. Kein eigener visueller Effekt — bestehende Outline-Logik wird übernommen.
- **Aufwand:** XS — CSS-Block + drei Attribut-Updates in `connectToDeepgram`/`stopRecording`.

---

### Pattern 15: Mode-Radios keyboard-zugänglich + sichtbarere Selection
**Adressiert Findings:** F21 (H6 Sev 2), F29 (H1 Sev 1)

- **Pattern:** Native Radio-Inputs aus dem `display:none` herausholen (visuell hidden via `.sr-only`-Klasse, aber im Tab-Pfad). Labels werden mit `:focus-within`/`:has(:checked)`-Logik gestylt, sodass der Tab-Pfad **das Input** fokussiert und der bestehende Custom-Card-Style erhalten bleibt. Selected-Visual deutlich verstärken (kräftigerer Inset-Shadow + sichtbarere Accent-Tint).
- **Microcopy** (DE, Du-Form):
  - keine zusätzlichen Strings (Labels schon in P12 übersetzt)
- **Gestaltung:** `:focus-visible`-Outline auf der Label-Card analog Drop-Zone. Selected-Tint von `var(--nm-tint-accent)` auf einen kontrastreicheren Wert anheben (z.B. doubled opacity oder zusätzliche linke Akzent-Border).
- **Aufwand:** S — CSS-Anpassung (`:has`/`:focus-within`-Pattern) + `tabindex`/`aria`-Attribute neu setzen.

---

### Pattern 16: Custom-Prompt-Toggle als echter Button
**Adressiert Findings:** F22 (H6 Sev 2)

- **Pattern:** `<div id="prompt-editor-toggle">` durch `<button type="button">` ersetzen oder mit `role="button"` + `tabindex="0"` + `aria-expanded`-State annotieren. Keydown-Handler für Enter/Space (falls `<div>`-Variante bleibt). `aria-expanded` toggelt synchron mit `.expanded`-Klasse auf `#prompt-editor-content`.
- **Microcopy** (DE, Du-Form):
  - Label collapsed: "Erweitert: Eigener Prompt anzeigen ▼"
  - Label expanded: "Erweitert: Eigener Prompt verbergen ▲"
  - aria-controls: `prompt-editor-content`
- **Gestaltung:** Position unverändert. `:focus-visible`-Outline. Icon-Wechsel bleibt erhalten.
- **Aufwand:** XS — Element-Wechsel + Attribute + Keydown-Handler (3 Zeilen).

---

### Pattern 17: Clear-Aktionen mit Confirmation (Live-Tab + Reset-Prompt)
**Adressiert Findings:** F23 (H9 Sev 2)

- **Pattern:** Native `confirm()` reicht nicht (Browser-Modal, EN-Default-Buttons, blockierend, Stil-Bruch). Stattdessen ein In-Page-Confirmation-Pattern: Klick auf "Leeren" → Button-Text und Style wechseln zu Warnung mit Bestätigung + Abbrechen-Button daneben. Zweiter Klick (innerhalb 5 s) führt aus, Klick außerhalb / Abbrechen-Klick / Timeout setzt zurück. Schwelle nur bei langen Inhalten (>200 Zeichen) — kurzer Inhalt löscht direkt ohne Confirm-Step.
- **Microcopy** (DE, Du-Form, Buttons max 3 Wörter):
  - Stage 1 (default): "Leeren"
  - Stage 2 (confirm): "Wirklich leeren?" + "Abbrechen"
  - aria-live-Announce bei Stage 2: "Bestätigung erforderlich. Erneut auf Leeren klicken zum Bestätigen."
- **Gestaltung:** Stage 2 Warning-Tint, Abbrechen-Btn neben dem Confirm-Btn. 5 s Auto-Reset zurück zu Stage 1. Bei Reset-Prompt-Btn analoger Mechanismus.
- **Aufwand:** S — kleine State-Machine (idle → pending → executed) im Click-Handler. Wenn als zu schwer empfunden, fallback auf `confirm()` mit DE-Microcopy als XS-Variante; Sub-Thread/Overseer entscheidet final.

---

### Pattern 18: Podcast-Polling Cancel-Button
**Adressiert Findings:** F24 (H9 Sev 2)

- **Pattern:** Während Polling läuft, blendet "Generate Podcast" sich aus und ein "Abbrechen"-Button ein. Click setzt ein Cancel-Flag, der Polling-Loop bricht aus, Backend-Job läuft im Hintergrund weiter (kein Backend-Cancel-Endpoint nötig — der Job läuft zu Ende und wird verworfen, was unter Single-User-Bedingungen akzeptabel ist).
- **Microcopy** (DE, Du-Form, Buttons max 3 Wörter):
  - Cancel-Btn: "Abbrechen"
  - Cancel-Banner nach Click: "Generierung abgebrochen. Backend-Job läuft im Hintergrund weiter."
  - Counter-Format (siehe P12): "Generiert … ({n} s)"
- **Gestaltung:** Abbrechen-Btn neben dem (jetzt versteckten) Generate-Btn. Warning-Tint. Counter bleibt sichtbar bis zum Cancel-Klick.
- **Aufwand:** S — Cancel-Flag + Loop-Exit-Branch + Btn-Toggle.

---

### Pattern 19: Filename-Anzeige mit `formatFileSize`
**Adressiert Findings:** F26 (H4 Sev 2)
**Cross-Feature-Konvergenz:** Reuse von `formatFileSize(bytes)` aus `_utils.js`

- **Pattern:** `change`-Listener auf `#file-upload-input` setzt nicht nur `filename`, sondern `${filename} (${formatFileSize(file.size)})`. Bei großen Audio-Dateien (oft 50–500 MB) ist die Größenangabe besonders relevant — sie korreliert mit der erwarteten Backend-Wartezeit.
- **Microcopy** (DE, Du-Form, Beispiele aus `formatFileSize`):
  - "podcast.mp3 (47,3 MB)"
  - "interview.wav (320,5 MB)"
- **Gestaltung:** Position unverändert (`#file-upload-text`). Größenangabe in derselben Zeile, in Klammern.
- **Aufwand:** XS — eine Zeile im `change`-Listener.

---

### Pattern 20: Live-Transcript `aria-live`-Region + `aria-label`
**Adressiert Findings:** F28 (H1 Sev 1)

- **Pattern:** `aria-label` und `aria-live="polite"` auf `#live-transcript-output`. Streaming-Updates werden so für Screenreader angesagt (höflich, nicht aggressiv interrupting). `aria-busy="true"` während Recording.
- **Microcopy** (DE, Du-Form):
  - aria-label: "Live-Transkription, fortlaufend aktualisiert"
- **Gestaltung:** keine.
- **Aufwand:** XS — drei Attribute.

---

### Pattern 21: Download Success-Toast + Audio-Player Error-Fallback
**Adressiert Findings:** F32 (H4 Sev 1), F31 (H1 Sev 1)
**Cross-Feature-Konvergenz:** Reuse von `showToast(msg)` aus `_utils.js`

- **Pattern:** Click auf "Herunterladen" feuert zusätzlich `showToast('✓ Podcast heruntergeladen')`. Audio-Element bekommt `onerror`-Handler, der bei revoked Blob-URL einen Hinweis zeigt ("Audio nicht mehr verfügbar — bitte erneut generieren.") in einer kleinen Status-Zeile unter dem Player.
- **Microcopy** (DE, Du-Form, max 3 Wörter Toast):
  - Toast: "✓ Podcast heruntergeladen"
  - Audio-Error-Hint: "Audio nicht mehr verfügbar — bitte erneut generieren."
- **Gestaltung:** Toast unten-rechts (Helper-Default). Audio-Error-Hint als kleine `c-alert--info`-Zeile unter dem Player.
- **Aufwand:** S — `showToast`-Call + `onerror`-Listener + Hint-Container.

---

## Bug-Tickets (kein Pattern-Bedarf)

- **B1: Drag-Drop-Handler fehlen** — siehe Pattern 2 (Bug-Voraussetzung). Code-Fix in [static/js/audio_converter.js](../static/js/audio_converter.js): drei Listener (`dragover`, `dragleave`, `drop`) plus File-Injection in `#file-upload-input` analog `document_converter.js`.
- **B2: `.saved`-Klasse-Reset im Save-Btn-Reset** — siehe Pattern 6 (Bug-Voraussetzung). Code-Fix: Helper räumt `.saved`-Klasse zusätzlich zu `disabled` und Text ab. Drei Reset-Stellen in [static/js/audio_converter.js](../static/js/audio_converter.js) konsolidieren.
- **B3: `getUserMedia` ohne `.catch()`** — siehe Pattern 3 (Bug-Voraussetzung). Code-Fix: explizites `.catch()` mit `error.name`-Branches und WS-Cleanup.
- **B4: Result-Container Copy-Guard greift nicht beim Placeholder** — siehe Pattern 11 (Bug-Voraussetzung). Code-Fix: Sentinel-Attribut `data-placeholder="true"` + Guard-Check.
- **B5: Live-Transcript Append-Race** — siehe Pattern 10 (Bug-Voraussetzung). Code-Fix: `readonly` während Recording.
- **B6: Inline-`<span>`-XSS-Risiko durch raw `err.message`** — siehe Pattern 5 (Bug-Voraussetzung). Wird durch `showAlert`-Konsolidierung mitgelöst (Helper nutzt `textContent`).
- **B7: Generate-Script-Button-Restore mit hartkodiertem SVG-Markup** — kein Pattern, reines Maintainability-Risiko. Code-Fix: Initial-`innerHTML` in `data-default-html`-Attribut speichern oder Loading-State über CSS-Klasse + Spinner-Span statt Element-Replace lösen. Empfehlung: bei Implementation von P12 (DE-Microcopy-Pass) gleich mit aufräumen, weil der Restore-Text dann ohnehin angefasst wird.
- **B8: WS-Auth-Failure differenziert nicht 401 vs Network vs Handshake** — kein Pattern, ergänzt P5. Code-Fix: WS-onerror/onclose unterscheidet `event.code` und liefert spezifischere DE-Microcopy (siehe P5-Tabelle, "WS/Token-Failure" generisch + Token-Fail spezifisch — ggf. dritte Variante "Verbindung abgebrochen — Netzwerk prüfen.").
- **B9: Filesize-Pre-Check Frontend** — siehe Pattern 13 (zusammen mit Backend-Whitelist gebündelt).

---

## Cluster-Vorbereitung für Implementation

**Zwei-Cluster-Default — Cluster I = 12 Patterns, Cluster II = 9 Patterns.**

### Cluster I (Sev 4 + Sev 3 + struktureller Vorbedingung-Fix)

12 Patterns: **P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12**

Begründung Gruppierung: Cluster I konvergiert auf das in F-1 etablierte App-Standard-Vokabular (`showAlert`, `showToast`, `formatFileSize`, Drag-Drop, Save-Lifecycle, DE-Microcopy). P4 (Banner-Mountpoint) ist strukturelle Voraussetzung für P1, P3, P5, P8 — muss am Anfang. P12 (DE-Microcopy) ist die "Sweep"-Stelle, an der ohnehin viele Templates und JS-Strings angefasst werden — sinnvoll, sie als erstes oder letztes anzubinden, damit Banner-Texte (P5) und Button-Labels (P6) gleich in DE landen.

**Surprise-5-Sub-Gruppe (Silent-Failure-Elimination):** P1 (Empty-Submit), P3 (Mic-Permission-Denied), P9 (Lang-Disabled-Visual). Diese drei adressieren das gemeinsame Anti-Pattern "User-Aktion ohne sichtbare Reaktion" — sollten zusammen verifiziert werden (Live-Smoke-Test: jede der drei Aktionen in der App auslösen, Sichtbarkeit der Reaktion prüfen).

**Cross-Feature-H4-Sub-Gruppe (Konvergenz):** P5 (Alert-Sweep), P6 (Save-Lifecycle), P12 (DE-Microcopy). Diese drei führen `audio_converter` strukturell auf F-1-Niveau zurück — können einzeln gemerget werden, weil sie unabhängige Code-Pfade berühren.

### Cluster II (Sev 2 + Sev 1 — Polish + a11y)

9 Patterns: **P13, P14, P15, P16, P17, P18, P19, P20, P21**

Begründung Gruppierung: Cluster II ist entweder a11y-Polish (P14, P15, P16, P20) oder einzelne UX-Reibungspunkte (P17 Clear-Confirm, P18 Polling-Cancel, P19 Filesize-Display, P21 Download-Toast + Audio-Error). P13 (Backend-Whitelist) liegt am Übergang — Sev 2, aber Backend-Touch + Single-Source-of-Truth-Refactor ist substanziell und gehört thematisch zu Cluster I's "Konvergenz auf F-1-Niveau". Wenn Cluster I voll ist, P13 in Cluster II; wenn Cluster I noch Kapazität hat, kann P13 hochgezogen werden.

### Drei-Cluster-Split (Empfehlung falls Cluster I als zu groß empfunden wird)

12 Patterns in einem Cluster ist die obere Grenze des in F-1 noch funktionierenden Reviews. Wenn Cluster I als zu groß empfunden wird, ist eine 3-er-Aufteilung:

- **Cluster Ia (Foundation Sweep):** P4, P5, P12 — 3 Patterns. Mechanische Großflächen-Refactors. Vorbedingung für alle anderen. Ein PR.
- **Cluster Ib (Critical UX gaps):** P1, P2, P3, P7, P10 — 5 Patterns. Sev 4s + Mic-Loading + Live-Textarea-Readonly. Live-Walkthrough-pflichtig wegen Mic + Drag-Drop + Stream-Behavior.
- **Cluster Ic (State-Lifecycle):** P6, P8, P9, P11 — 4 Patterns. Save-Lifecycle, Config-Error-Split, Lang-Disabled, Copy-Guard. Statisch verifizierbar + Smoke-Test je State.

Cluster II bleibt unverändert. Drei-Cluster-Empfehlung gilt nur, wenn der Overseer nach P5/P12-Volumen-Sicht entscheidet, dass Cluster I sonst zu unhandlich wird.

---

## Priorisierung — Top-5 Quick-Wins

**Aufwand-Gewicht:** XS=1, S=2, M=4, L=8. Score = Sev × 5 / Aufwand-Gewicht. Höher = besser.

| Rang | Pattern # | Adressiert | Sev | Aufwand | Impact-Score | Quick-Win |
|------|-----------|------------|-----|---------|--------------|-----------|
| 1 | Pattern 6 | F9, F10 — Save-Btn-Lifecycle (Cross-Feature) | 3 | XS | 15.0 | ★ Top-5 |
| 2 | Pattern 9 | F15 — Lang-Disabled sichtbar | 3 | XS | 15.0 | ★ Top-5 |
| 3 | Pattern 11 | F17 — Copy-Guard Sentinel | 3 | XS | 15.0 | ★ Top-5 |
| 4 | Pattern 1 | F1, F2 — Empty-Submit File-Tab | 4 | S | 10.0 | ★ Top-5 |
| 5 | Pattern 2 | F3, F4 — Echtes Drag-Drop | 4 | S | 10.0 | ★ Top-5 |
| 6 | Pattern 3 | F5 — Mic-Permission-Denied | 4 | S | 10.0 | |
| 7 | Pattern 4 | F27 — Banner-Mountpoint im Template | 2 | XS | 10.0 | |
| 8 | Pattern 16 | F22 — Custom-Prompt-Toggle als button | 2 | XS | 10.0 | |
| 9 | Pattern 19 | F26 — Filename mit `formatFileSize` | 2 | XS | 10.0 | |
| 10 | Pattern 14 | F20, F25 — Mic-Btn focus-visible + a11y | 2 | XS | 10.0 | |
| 11 | Pattern 5 | F6, F7, F8, F11 — Alert-Konsolidierung | 3 | S | 7.5 | |
| 12 | Pattern 7 | F12 — Mic-Button Loading-State | 3 | S | 7.5 | |
| 13 | Pattern 8 | F13, F14 — Config-Error spezifisch + tab-scoped | 3 | S | 7.5 | |
| 14 | Pattern 10 | F16 — Live-Textarea readonly während Recording | 3 | S | 7.5 | |
| 15 | Pattern 12 | F18, F30 — DE-Microcopy-Pass | 3 | S | 7.5 | |
| 16 | Pattern 13 | F19 — Backend-Whitelist + DE-Fehler-JSON | 2 | S | 5.0 | |
| 17 | Pattern 15 | F21, F29 — Mode-Radios keyboard + sichtbar selected | 2 | S | 5.0 | |
| 18 | Pattern 17 | F23 — Clear-Confirmation | 2 | S | 5.0 | |
| 19 | Pattern 18 | F24 — Podcast-Polling Cancel-Button | 2 | S | 5.0 | |
| 20 | Pattern 20 | F28 — Live-Transcript `aria-live` | 1 | XS | 5.0 | |
| 21 | Pattern 21 | F31, F32 — Download-Toast + Audio-Player-Error | 1 | S | 2.5 | |

**Top-5 Quick-Wins:**

1. **Pattern 6 — Save-Btn-Lifecycle** (15.0): Cross-Feature-Konvergenz, F-1's `resetSaveBtn`-Logik 1:1 auf zwei Save-Buttons anwenden. Schließt B2 mit. Einer der Brot-und-Butter-Bugs der Stage.
2. **Pattern 9 — Lang-Disabled visuell** (15.0): eine CSS-Rule schließt einen Sev-3-Silent-Fail-Pfad. XS-Aufwand mit hohem Hebel.
3. **Pattern 11 — Copy-Guard Sentinel** (15.0): zwei Zeilen Code, schließt einen seit Inventur identifizierten konkreten Bug (B4) und eine Sev-3-Frustration (Placeholder im Clipboard).
4. **Pattern 1 — Empty-Submit-Banner File-Tab** (10.0): höchster Sev der Stage (4), mit existierendem `showAlert`-Helper und `c-drop-zone--invalid`-Klasse. Pflicht-Fix.
5. **Pattern 2 — Echtes Drag-Drop** (10.0): Sev 4, schließt einen offenen Cross-Feature-Konvergenz-Bruch (Drag-Drop-Lüge). Reuse der F-1-Drop-Handler-Logik macht den Aufwand auf S beherrschbar.

Pattern 3 (Mic-Permission-Denied, Score 10.0) ist gleich wichtig wie P1/P2 (alle Sev 4), nur mengenmäßig nicht in den Top-5 — sollte in Cluster I direkt mit P1/P2 zusammen umgesetzt werden, weil alle drei zur Surprise-5-Familie gehören.

---

**Schweregrad-Skala (aus Stufe 2):**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse)
