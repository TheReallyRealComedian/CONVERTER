# UX-Patterns + Microcopy: library_detail (2026-05-09)

**Methodik:** Stufe 3 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Konkrete Patterns + DE-Microcopy auf Basis der Heuristik-Findings aus Stufe 2. Konsolidiert die 17 Stufe-2-Findings auf 14 Pattern-Blöcke (3 konsolidiert + 11 einzeln). Bug-Tickets BT1–BT6 sind in den Patterns ihrer verknüpften Findings mit-adressiert; BT7 (textarea-escape) und BT8 (window.open-noopener) sind pure Bug-Tickets ohne UX-H-Komponente und nicht Teil von F-3.3 (Sammel-Bug-Pass oder mit-genommen wenn nahegelegene Patterns angefasst werden).
**Quelle Findings:** [docs/ui_findings_library_detail_2026-05.md](ui_findings_library_detail_2026-05.md)
**Quelle Inventur:** [docs/ui_inventory_library_detail_2026-05.md](ui_inventory_library_detail_2026-05.md)
**F-1 / F-2 Patterns als Referenz:** [docs/ui_patterns_document_converter_2026-05.md](ui_patterns_document_converter_2026-05.md), [docs/ui_patterns_audio_converter_2026-05.md](ui_patterns_audio_converter_2026-05.md)
**Helper-API:** [static/js/_utils.js](../static/js/_utils.js) — `safeJSON(response)`, `fallbackCopyText(text)`, `showAlert(containerEl, level, msg, options?)`, `showToast(msg, options?)`, `formatFileSize(bytes)`, `confirmIfLong(text, msg, options?)`
**Komponenten-Basis (post-F-1 / post-F-2):** Existierende Neomorphism-Klassen aus [static/css/style.css](../static/css/style.css) — `c-btn`, `c-btn--primary`, `c-btn--danger`, `c-input`, `c-surface`, `c-surface--flat`, `c-alert--danger/success/warning/info` (mit Close-Button + Auto-Dismiss aus F-1 Cluster C), `.c-tag`, `.type-badge`, `.favorite-btn`, `.toast-notification`, `.hidden`, `:focus-visible` für `c-btn`. Kein neues Design-System.

**Microcopy-Regeln:** Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern, Deutsch durchgängig (Du-Form analog F-1 / F-2).
**Aufwand-Skala:** XS / S / M / L (Daumenregel: XS = 1–3 Zeilen, S = ein Handler-Cluster + Microcopy-Sweep, M = Schema-Touch oder neue Mechanik, L = Cross-Feature-Refactor).
**Impact-Score-Formel:** `Score = Sev × 5 / Aufwand-Gewicht` mit Aufwand-Gewichten XS=1, S=2, M=4, L=8. Höher = besser. Bei konsolidierten Patterns wird die höchste Sev der adressierten Findings genommen (analog F-1.3 / F-2.3).

**Smoke-Pflicht-Konvention** (aus Master-Annotation im Sprint-Prompt): Patterns, die Findings aus Cluster 1 (Silent-Failure-Familie F1/F2/F4/F5) oder Cluster 2 (Notion-State-Wipe + UTC F6/F11/F8) adressieren, tragen den Sub-Tag `🔥 Smoke-Pflicht in F3-IMPL`. Vor dem Pattern-Apply muss in F3-IMPL-* per Live-Smoke (DevTools-Network-Throttle für F1/F2/F4/F5; Browser-Reload-Sequenz für F6/F11; Datum-Inspektion in Europe/Berlin für F8) verifiziert werden, dass der Code-deduced-Befund tatsächlich existiert. Wenn der Smoke zeigt, dass der Befund nicht reproduzierbar ist: Pattern-Apply STOP, Master fragen.

---

## Pattern-Blöcke

### Pattern 1: Auto-Save Title/Tags + Favorite silent-fail
**Adressiert Findings:** F1 (H1 Sev 3), F2 (H9 Sev 3)
**Adressiert Bug-Ticket:** BT1 (`updateField` und `toggleFavorite` ohne Error-Handling)
**Cluster:** 1 (Silent-Failure-Familie) + Cross-Feature-H4
**🔥 Smoke-Pflicht in F3-IMPL**

- **Pattern:** `updateField()` und `toggleFavorite()` bekommen vollständige Fehler-Behandlung: `.catch()` + `if (!r.ok)`-Branch. Bei Erfolg bleibt der bisherige Toast (mit korrektem Level via P8). Bei Fehler erscheint ein persistenter `c-alert--danger`-Banner im neuen `#detail-alert-container` (siehe P15) mit Recovery-Hinweis. Im Fehlerfall wird beim Favorite-Toggle der optimistisch geswapte Glyph zurückgerollt; beim Title/Tags wird der Server-Wert nicht im Input ersetzt (User behält die getippte Eingabe und kann erneut versuchen).
- **Visuelle Hinweise:** Banner oben (`#detail-alert-container`), Danger-Tint, Close-× über `showAlert`-Default. Title/Tags-Input bekommt während des Roundtrips keine Loading-Indikation (Roundtrip ist kurz, Pending-State adressiert separat in P2).
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Title-Save-Failure: „Titel konnte nicht gespeichert werden. Verbindung prüfen und erneut versuchen."
  - Tags-Save-Failure: „Tags konnten nicht gespeichert werden. Verbindung prüfen und erneut versuchen."
  - Favorite-Toggle-Failure: „Favorit konnte nicht aktualisiert werden. Verbindung prüfen und erneut versuchen."
  - Generischer Server-Fehler-Suffix (wenn `r.status >= 500`): „… Server-Fehler — bitte später erneut versuchen."
- **Helper-Reuse:** `showAlert(detailAlertContainer, 'danger', msg)` aus [static/js/_utils.js](../static/js/_utils.js) ersetzt den fehlenden Error-Branch. Optional `safeJSON(r)` für Session-Expired-Detection — wenn `safeJSON` `Session expired`-Error wirft, eigene Microcopy (siehe P15-Eintrag „Session-Expired"). Inline-Code-Anker: [static/js/library_detail.js:8-18](../static/js/library_detail.js#L8-L18) `updateField`; [static/js/library_detail.js:20-32](../static/js/library_detail.js#L20-L32) `toggleFavorite`.
- **Aufwand:** S — `.catch` + `r.ok`-Branch in zwei Funktionen, Banner-Container im Template (gemeinsam mit P3 und P7), Microcopy. Kein neues CSS, kein Schema-Touch.
- **Impact-Score:** 3 × 5 / 2 = **7.5**
- **Konsolidierung:** F1 (H1, sichtbarer Status fehlt) und F2 (H9, Recovery-Anleitung fehlt) entstehen aus derselben fehlenden Error-Behandlung in `updateField`. Eine Lösung (showAlert + Recovery-Microcopy) erfüllt beide Heuristiken — Sichtbarkeit per Banner, Recovery per Microcopy-Hinweis. Cross-Feature-Konvergenz auf F-1 Cluster B Pattern 4 (Save-Failure-Banner) und F-2.3 P5 (Alert-Konsolidierung).

---

### Pattern 2: Auto-Save Pending-State sichtbar machen
**Adressiert Findings:** F3 (H1 Sev 3)

- **Pattern:** Title- und Tags-Input bekommen einen `input`-Listener (zusätzlich zu `change`), der die Inputs in einen Dirty-State setzt (`.c-input--dirty`-Klasse + dezente Akzent-Border). Der Dirty-State wird beim erfolgreichen `updateField`-Roundtrip wieder entfernt. Zusätzlich hört das Window auf `beforeunload` (oder Visibility-Change auf `hidden`): wenn Inputs noch dirty sind, feuert `updateField` proaktiv mit dem aktuellen Value. Damit landet eine schnelle Cmd+Tab-Navigation nicht im Datenverlust-Pfad.
- **Visuelle Hinweise:** Dirty-Indikator als linke Akzent-Border (z.B. 3 px `var(--nm-tint-accent)`) oder kleines „*"-Suffix neben dem Title; nicht aufdringlich, eindeutig vom Default-Fokus-Ring zu unterscheiden. Nach erfolgreichem Save Übergang `border-color` 200 ms zurück zum Default.
- **Microcopy** (DE, Du-Form):
  - Tooltip auf dem Title-Input während Dirty: „Ungespeicherte Änderung — Tab oder Klick außerhalb speichert."
  - Tooltip auf dem Tags-Input während Dirty: „Ungespeicherte Änderung — Tab oder Klick außerhalb speichert."
  - aria-live-Hint nach Save-Erfolg (höflich): „Titel gespeichert." / „Tags gespeichert."
- **Helper-Reuse:** keine neuen Helper. Bestehender `updateField`-Pfad wird wiederverwendet, beforeunload-Handler ergänzt im selben Modul. Code-Anker: [static/js/library_detail.js:8-18](../static/js/library_detail.js#L8-L18) für `updateField`-Trigger; [templates/library_detail.html:13](../templates/library_detail.html#L13) und [templates/library_detail.html:115](../templates/library_detail.html#L115) für die Input-Bindings.
- **Aufwand:** S — kleiner CSS-Block (`.c-input--dirty`), zwei `input`-Listener, ein `beforeunload`-Handler, Tooltip-Strings. Kein neuer Helper, keine Schema-Änderung.
- **Impact-Score:** 3 × 5 / 2 = **7.5**

---

### Pattern 3: Delete silent-fail
**Adressiert Findings:** F4 (H1 Sev 3), F5 (H9 Sev 3)
**Adressiert Bug-Ticket:** BT2 (`deleteConversion` ohne Error-Branch)
**Cluster:** 1 (Silent-Failure-Familie)
**🔥 Smoke-Pflicht in F3-IMPL**

- **Pattern:** `deleteConversion()` bekommt einen `else`-Branch nach dem `r.ok`-Check und einen `.catch()` für Network-Fail. Im Fehlerpfad: persistenter `c-alert--danger`-Banner im `#detail-alert-container` (gemeinsam mit P1 und P7). Delete-Button kehrt aus dem (kurzlebigen) Loading-State (siehe Visualisierung unten) zurück in den Default-State. Bei `r.status === 404` extra Pfad: „bereits gelöscht" mit Navigation zur Library, weil das ein erwarteter Race-Case ist.
- **Visuelle Hinweise:** Banner oben, Danger-Tint, Close-× via `showAlert`-Default. Während des DELETE-Roundtrips zeigt der Delete-Button kurz „Lösche …" als Text-Swap (analog Save-Btn-Loading-Pattern aus F-1 P3); kein Spinner-Icon nötig.
- **Microcopy** (DE, Du-Form, max 2 Sätze; Buttons max 3 Wörter):
  - Confirm-Dialog (DE statt EN, ersetzt das aktuelle `confirm('Delete this conversion? …')`): „Diesen Eintrag wirklich löschen? Das kann nicht rückgängig gemacht werden."
  - Delete-Btn default: „Löschen"
  - Delete-Btn loading: „Lösche …"
  - Delete-Failure-Banner generisch: „Löschen fehlgeschlagen. Verbindung prüfen und erneut versuchen."
  - Delete-Failure-Banner Server-Fehler: „Löschen fehlgeschlagen — Server-Fehler. Bitte später erneut versuchen."
  - Delete-Race (404): „Eintrag wurde bereits entfernt. Du wirst zur Library zurückgeleitet." (info-Banner, nicht danger)
- **Helper-Reuse:** `showAlert(detailAlertContainer, 'danger', msg)` für Failure; `showAlert(detailAlertContainer, 'info', msg, { autoDismissMs: 3000 })` für 404-Race vor der Navigation. `safeJSON(r)` optional für Session-Expired-Detection. Inline-Code-Anker: [static/js/library_detail.js:55-60](../static/js/library_detail.js#L55-L60).
- **Aufwand:** S — `else`-Branch + `.catch` + Loading-Text-Swap + drei Microcopy-Strings + `confirm`-DE-Übersetzung. Banner-Container teilen mit P1.
- **Impact-Score:** 3 × 5 / 2 = **7.5**
- **Konsolidierung:** F4 (H1 Sichtbarkeit) und F5 (H9 Recovery) sind dieselbe Mechanik wie P1, nur in Delete-Pfad statt Save-Pfad. Eine Lösung (showAlert + Recovery-Microcopy) erfüllt beide Heuristiken. Cross-Feature-Konvergenz auf F-1 Cluster B Pattern 4 und F-2.3 P3 (Mic-Permission silent — Familie „User-Aktion ohne sichtbare Reaktion").

---

### Pattern 4: Notion-Form State-Preservation bei Re-Toggle und Target-Switch
**Adressiert Findings:** F6 (H1 Sev 3), F11 (H1 Sev 2)
**Adressiert Bug-Ticket:** BT3 (`toggleNotionPanel` und `selectTarget` zerstören User-Inputs)
**Cluster:** 2 (Notion-State-Wipe)
**🔥 Smoke-Pflicht in F3-IMPL**

- **Pattern:** `renderNotionFields()` wird **nicht mehr unbedingt** ausgeführt. `toggleNotionPanel()` ruft `selectTarget(DEFAULT_TARGET)` nur, wenn der `#notion-fields`-Container leer ist (initialer Open). Bei Re-Toggle (`hidden` → sichtbar wieder, Felder bereits gerendert) bleibt der Inhalt unverändert. `selectTarget()` greift erst bei tatsächlichem Target-Wechsel: wenn `currentTarget !== newTarget`, wird ein FormData-Snapshot der aktuellen Felder gemacht, der Container neu gerendert, dann passende Felder aus dem Snapshot in den neuen Target-Layout zurückgeschrieben (Mapping nach Field-Name — `title` → `title`, `tags` → `tags`, `description`/`summary`/`note`/`text` aus dem gemeinsamen Body-Pool).
- **Visuelle Hinweise:** Beim Target-Switch dezenter Übergang `opacity` 150 ms beim Container, damit der Re-Render nicht hart wirkt. Datum-Input behält den User-Wert, falls vorhanden, sonst wird der lokal-Default aus P5 verwendet.
- **Microcopy** (DE, Du-Form):
  - aria-live-Hint nach Target-Switch (höflich): „Ziel gewechselt — passende Felder übernommen."
  - Tooltip auf den Target-Buttons (Hover): „Notion-Ziel wechseln" (single-source, gilt für alle drei).
- **Helper-Reuse:** keine neuen `_utils.js`-Helper. Lokale Hilfsfunktion `collectFieldValues(container)` und `restoreFieldValues(container, snapshot)` innerhalb `library_detail.js`. Inline-Code-Anker: [static/js/library_detail.js:66-75](../static/js/library_detail.js#L66-L75) `toggleNotionPanel`; [static/js/library_detail.js:85-91](../static/js/library_detail.js#L85-L91) `selectTarget`.
- **Aufwand:** M — neue State-Preservation-Logik mit Snapshot/Restore-Mapping, Conditional-Rendering-Branch, Test des State-Cycles (Re-Toggle, Target-Switch, gemischte Sequenzen). Schema-Touch entfällt, aber die Mechanik ist nicht trivial weil das Field-Mapping zwischen den drei Target-Layouts (`meetings`/`notes`/`inbox`) berücksichtigt werden muss.
- **Impact-Score:** 3 × 5 / 4 = **3.75**
- **Konsolidierung:** F6 und F11 haben dieselbe Wurzel: `renderNotionFields()` setzt unbedingt `container.innerHTML = …` ohne State-Preservation. Zwei Trigger-Pfade (Re-Toggle und Target-Switch), eine Lösung. Sprint-Prompt-Logik „Findings die zur selben State-Mechanik gehören → ein Pattern". F11 hat niedrigere Sev (2 statt 3) wegen geringerer Trigger-Häufigkeit (User-Versehen-Klick statt Re-Toggle), aber identische Lösung.

---

### Pattern 5: Datum-Default lokal statt UTC für Notion-Meeting-Field
**Adressiert Findings:** F8 (H1 Sev 3)
**Adressiert Bug-Ticket:** BT4 (`new Date().toISOString().slice(0,16)` ist UTC)
**Cluster:** 2 (Notion-UTC-Default)
**🔥 Smoke-Pflicht in F3-IMPL**

- **Pattern:** `renderNotionFields()` ersetzt den UTC-Default durch eine lokale Datum-Berechnung: `const now = new Date(); const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000); const datetimeLocal = local.toISOString().slice(0, 16);`. Damit zeigt das `<input type="datetime-local">` für einen Europe/Berlin-Browser tatsächlich die lokale „jetzt"-Zeit, und der gesendete Wert landet korrekt in Notion.
- **Visuelle Hinweise:** keine — reine Logik-Änderung im Pre-Population-Pfad. User sieht im Datum-Input das richtige „jetzt", ohne dass eine UI-Beschriftung verändert wird.
- **Microcopy** (DE, Du-Form):
  - keine neue Microcopy. Datum-Field-Label „Datum" (DE-Microcopy-Pass aus P6) bleibt unverändert.
- **Helper-Reuse:** keine neuen `_utils.js`-Helper für die einmalige Verwendung. **Helper-Vorschlag** für künftige Wiederverwendung: `formatDatetimeLocalNow()` in `_utils.js`, wenn weitere Features denselben Default brauchen — siehe Helper-Vorschlag-Sektion am Doc-Ende. Inline-Code-Anker: [static/js/library_detail.js:96](../static/js/library_detail.js#L96).
- **Aufwand:** XS — eine Zeile Datum-Berechnung, drei Code-Zeilen mit Kommentar. Kein neues CSS, keine Microcopy, kein Schema.
- **Impact-Score:** 3 × 5 / 1 = **15.0**

---

### Pattern 6: DE-Microcopy-Pass flächendeckend
**Adressiert Findings:** F7 (H4 Sev 3)
**Cluster:** 3 (Cross-Feature-Helper-Drift)

- **Pattern:** Alle ~18 EN-Strings auf der Seite werden in DE/Du-Form übersetzt — sowohl Template-Strings (Header-Back-Link, Type-Badges, Toolbar-Buttons, Notion-Target-Buttons, Details-Aside-Labels, Tags-Placeholder) als auch JS-Strings (Toast-Texte aus `updateField`, `copyFullContent`, Notion-Submit-Pfad, `confirm()`-Dialog aus `deleteConversion`). Konstanten-Tabelle unten als Single-Source-of-Truth.
- **Visuelle Hinweise:** keine — reine String-Substitution. Layout bleibt identisch.
- **Microcopy** (Konstanten-Tabelle, EN → DE):

| Position | EN | DE |
|----------|----|----|
| Header-Back-Link | ← Back to Library | ← Zurück zur Library |
| Type-Badge document_to_markdown | Document | Dokument |
| Type-Badge audio | Audio | Audio |
| Type-Badge dialogue | Dialogue | Dialog |
| Type-Badge markdown_input | Markdown | Markdown |
| Title-Input-Placeholder | Untitled | Ohne Titel |
| Toolbar-Btn | Copy to Clipboard | Kopieren |
| Toolbar-Btn (Toast Success) | Copied to clipboard | Kopiert |
| Toolbar-Btn (Toast Failure) | Copy failed | Kopieren fehlgeschlagen |
| Toolbar-Btn | Download as .txt | Als .txt herunterladen |
| Toolbar-Btn | Open in Editor | Im Editor öffnen |
| Toolbar-Btn | Delete | Löschen |
| Toolbar-Btn (Loading siehe P3) | — | Lösche … |
| Confirm-Dialog Delete (siehe P3) | Delete this conversion? This cannot be undone. | Diesen Eintrag wirklich löschen? Das kann nicht rückgängig gemacht werden. |
| Notion-Toggle-Header | Save to Notion | An Notion senden |
| Notion-Target-Btn | Meeting | Meeting |
| Notion-Target-Btn | Note | Notiz |
| Notion-Target-Btn | Inbox | Inbox |
| Notion-Submit-Btn default | Send to Notion | An Notion senden |
| Notion-Submit-Btn loading | Sending... | Sende … |
| Notion-Submit-Toast Success | Saved to Notion! | An Notion gesendet |
| Notion-Submit-Toast Status-Error (siehe P7 für Banner) | Error: {detail} | Senden fehlgeschlagen: {detail} |
| Notion-Submit-Toast Network-Failure (siehe P7) | Failed to connect to Notion | Verbindung zu Notion fehlgeschlagen |
| Details-Aside Label | Source File | Quelldatei |
| Details-Aside Label | File Type | Dateityp |
| Details-Aside Label | File Size | Dateigröße |
| Details-Aside Label | Created | Erstellt |
| Details-Aside Label | Updated | Aktualisiert |
| Details-Aside Label | Content Length | Inhaltslänge |
| Tags-Aside Heading | Tags | Tags |
| Tags-Input-Placeholder | comma-separated tags | kommagetrennte Tags |
| Auto-Save Title-Toast (Success) | title updated | Titel gespeichert |
| Auto-Save Tags-Toast (Success) | tags updated | Tags gespeichert |
| Favorite-Btn title | Toggle favorite | Favorit umschalten |

- **Helper-Reuse:** keine. Reine String-Änderung in [templates/library_detail.html](../templates/library_detail.html) und [static/js/library_detail.js](../static/js/library_detail.js). Cross-Feature-Konvergenz auf F-1 Cluster Polish-1 (DE-Microcopy-Konvention) und F-2.3 P12 (DE-Microcopy-Pass).
- **Aufwand:** S — Volumen ist hoch (~18 Strings), aber jede Stelle ist 1:1-Replace. Template + ein JS-Modul. Sweep sinnvoll mit P1/P3/P7/P8 zusammen, weil dieselben Stellen ohnehin angefasst werden.
- **Impact-Score:** 3 × 5 / 2 = **7.5**

---

### Pattern 7: Notion-Submit Persistent Error-Banner
**Adressiert Findings:** F9 (H4 Sev 2)
**Cluster:** 3 (Cross-Feature-Helper-Drift)

- **Pattern:** Notion-Submit-Errors (Status≥400 und Network-Failure) werden statt über `showToast` (auto-dismiss 2.5 s) über `showAlert(notionAlertContainer, 'danger', msg)` als persistenter Banner angezeigt — siehe P15 für die Banner-Container-Struktur. Banner bleibt sichtbar, bis User ihn schließt oder ein erneuter Submit-Versuch startet (Submit-Handler räumt den Container am Anfang ab). Erfolgs-Pfad bleibt beim Toast (positives Feedback braucht keine Persistenz). Status-Detail (z.B. `error.detail` aus dem Server-JSON) wird im Banner-Text mit-angezeigt.
- **Visuelle Hinweise:** Banner direkt unter dem Notion-Toggle-Header (`#notion-alert-container` aus P15), Danger-Tint, Close-× über `showAlert`-Default. Submit-Button kehrt nach dem Fail in den Default-State zurück (`disabled=false`, Text „An Notion senden" aus P6).
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Submit-Status-Error (mit Server-Detail, falls vorhanden): „Senden fehlgeschlagen: {detail}."
  - Submit-Status-Error (ohne Detail): „Senden an Notion fehlgeschlagen. Erneut versuchen oder Server-Konfiguration prüfen."
  - Submit-Network-Failure: „Verbindung zu Notion fehlgeschlagen. Netzwerk und Notion-MCP-Server-Status prüfen."
- **Helper-Reuse:** `showAlert(notionAlertContainer, 'danger', msg)` aus [static/js/_utils.js](../static/js/_utils.js#L39-L78) ersetzt die `showToast`-Calls auf den Fehler-Pfaden. Erfolgs-`showToast` mit `level: 'success'` (siehe auch P8) bleibt erhalten. Inline-Code-Anker: [static/js/library_detail.js:171-177](../static/js/library_detail.js#L171-L177).
- **Aufwand:** S — zwei `showToast`-Calls auf `showAlert` umstellen + Container im Template (gemeinsam mit P1/P3-Container über P15), Microcopy. Kein neues CSS.
- **Impact-Score:** 2 × 5 / 2 = **5.0**

---

### Pattern 8: Toast-Level pro Call-Site korrekt setzen
**Adressiert Findings:** F10 (H4 Sev 2)
**Adressiert Bug-Ticket:** BT5 (Toast-Level-Default überall `success`)

- **Pattern:** Alle `showToast`-Calls in `library_detail.js`, die Fehler-Pfade ansagen, setzen explizit `{ level: 'danger' }` (oder `'warning'` für Validation-Fälle). Erfolgs-Pfade behalten den Default-Level `success`. Damit korreliert visuelle Tönung (grün/rot/orange) mit dem semantischen Inhalt des Strings. Die Notion-Submit-Fehler-Pfade werden ohnehin durch P7 von Toast auf Banner umgestellt — der einzige Toast-Fehler-Pfad, der bleibt, ist „Kopieren fehlgeschlagen" auf der Toolbar.
- **Visuelle Hinweise:** keine neuen Komponenten. Toast-Tint-Logik ist schon im CSS via `.toast-notification--danger`/`--warning`/`--success` definiert.
- **Microcopy** (DE — siehe P6 Konstanten-Tabelle):
  - Copy-Failure (mit `level: 'danger'`): „Kopieren fehlgeschlagen"
  - alle übrigen Toasts behalten ihren Default-Level (Title-Save / Tags-Save / Notion-Success).
- **Helper-Reuse:** `showToast(msg, { level: 'danger' })` und `showToast(msg, { level: 'warning' })` aus [static/js/_utils.js](../static/js/_utils.js#L92-L117) — Helper unterstützt das bereits, in `library_detail.js` wird es heute nur nicht genutzt. Inline-Code-Anker: [static/js/library_detail.js:36-38](../static/js/library_detail.js#L36-L38) Copy-Failure; [static/js/library_detail.js:174](../static/js/library_detail.js#L174) Notion-Status-Error (entfällt durch P7); [static/js/library_detail.js:177](../static/js/library_detail.js#L177) Notion-Network-Failure (entfällt durch P7).
- **Aufwand:** XS — pro Call-Site eine Options-Object-Ergänzung. Effektiv 1–2 Stellen, weil die Notion-Errors über P7 zu Banner werden.
- **Impact-Score:** 2 × 5 / 1 = **10.0**

---

### Pattern 9: Tags-Input mit Chip-Visualisierung
**Adressiert Findings:** F12 (H6 Sev 2)
**Cluster:** 3 (Cross-Feature-Helper-Drift)

- **Pattern:** Tags-Aside zeigt unter dem CSV-Input eine Chip-Reihe, die bei Eingabe synchron gerendert wird. Jeder Chip nutzt die existierende `.c-tag`-Klasse aus dem CSS (Library-List nutzt die schon). Klick auf das „×" eines Chips entfernt den Tag aus der CSV-Liste und triggert `updateField('tags', …)` mit dem neuen Wert. Der Plain-CSV-Input bleibt erhalten als Fallback-/Power-User-Pfad — die Chips sind eine zusätzliche Visualisierung, keine Ersetzung. Tags werden auf Komma + Trim normalisiert (Whitespace-Tags ignoriert, Duplikate entfernt).
- **Visuelle Hinweise:** Chips unterhalb des Inputs in einem Flex-Wrap-Container, mit kleinen Lücken (gap 0.4 rem). Jeder Chip mit `.c-tag` (Akzent-Tint-Background, abgerundete Ecken). „×" als kleines, dezent gefärbtes Glyph rechts im Chip, Hover-Tint auf danger.
- **Microcopy** (DE, Du-Form, max 3 Wörter pro Element):
  - Chip-Remove-Btn aria-label: „Tag entfernen"
  - Empty-State-Hint (max 3 Sätze, wenn Tags-Liste leer): „Noch keine Tags. Mit Komma trennen, um mehrere zu speichern."
  - Tags-Input-Placeholder (aus P6): „kommagetrennte Tags"
- **Helper-Reuse:** keine `_utils.js`-Helper für die Chip-Render-Logik. Lokale Hilfsfunktion `renderTagChips(csvString, container)` in `library_detail.js`. **Helper-Vorschlag** für `_utils.js`, falls Library-List den Render auch konsolidiert nutzen will — siehe Helper-Vorschlag-Sektion am Doc-Ende. Bestehende `.c-tag`-CSS-Klasse wird wiederverwendet. Inline-Code-Anker: [templates/library_detail.html:112-115](../templates/library_detail.html#L112-L115).
- **Aufwand:** M — neue Render-Logik mit Sync-Listener, Chip-Click-Handler, normalisierende CSV-Logik, leerer-Hint-Empty-State. CSS-Klasse existiert, aber Chip-Layout (Flex-Container, Lücken) ist neu. Kein Schema-Touch.
- **Impact-Score:** 2 × 5 / 4 = **2.5**

---

### Pattern 10: Notion-Toggle-Header `aria-expanded` + `aria-controls`
**Adressiert Findings:** F13 (H6 Sev 1)

- **Pattern:** Der Notion-Toggle-Header (`<button>`-Wrapper mit `<h6>` + Icon-Span) bekommt `aria-expanded="false"` initial und `aria-controls="notion-panel"`. `toggleNotionPanel()` synchronisiert `aria-expanded` mit dem `.hidden`-Klassen-Wechsel. Das Glyph-Swap (▾/▴) bleibt visuell, aber a11y-Leser können den Open/Closed-Status korrekt erkennen.
- **Visuelle Hinweise:** keine neuen Komponenten. ARIA-only-Annotation.
- **Microcopy** (DE, Du-Form):
  - keine neuen Strings — Heading-Text aus P6 („An Notion senden") trägt die Sichtbarkeit; Icon ▾/▴ und `aria-expanded` ergänzen sich.
- **Helper-Reuse:** keine. Inline-Code-Anker: [templates/library_detail.html:47-50](../templates/library_detail.html#L47-L50) (Toggle-Header); [static/js/library_detail.js:66-75](../static/js/library_detail.js#L66-L75) `toggleNotionPanel`.
- **Aufwand:** XS — zwei Attribute im Template, eine `setAttribute`-Zeile in `toggleNotionPanel`.
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 11: Sidebar-Active-State auf Detail-Seite
**Adressiert Findings:** F14 (H4 Sev 1)
**Adressiert Bug-Ticket:** BT6 (Endpoint-Match in `base.html` zu strikt)

- **Pattern:** `templates/base.html` erweitert die Active-Markierung am Library-Sidebar-Link von `request.endpoint == 'library'` auf einen Endpoint-Set-Match: `request.endpoint in ('library', 'library_detail')`. Damit bleibt die Library-Nav auch auf der Detail-Seite hervorgehoben. Hinweis: das Projekt nutzt flache Routes ohne Blueprint (siehe [CLAUDE.md](../CLAUDE.md) Architektur-Notes), also kein `request.blueprint`-Match. Wenn künftig weitere Detail-Endpoints dazukommen (z.B. `library_highlight_detail`), Liste entsprechend erweitern.
- **Visuelle Hinweise:** keine neuen Komponenten. Existierende `.neo-nav-active`-Klasse wird nun auch auf der Detail-Seite gesetzt.
- **Microcopy** (DE, Du-Form):
  - keine neuen Strings.
- **Helper-Reuse:** keine. Inline-Code-Anker: [templates/base.html](../templates/base.html) rund um die Library-Nav-Link-Stelle (genaue Zeile per `grep` ermitteln in F3-IMPL-*).
- **Aufwand:** XS — eine Jinja2-Bedingung erweitern.
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 12: File-Size mit KB/B-Fallback (Server-side)
**Adressiert Findings:** F15 (H4 Sev 1)
**Cluster:** 3 (Cross-Feature-Helper-Drift)

- **Pattern:** Die Details-Aside-Reihe „File Size" wird über einen Jinja2-Filter oder eine Custom-Function im Template-Context gerendert, der die `formatFileSize`-Logik aus [static/js/_utils.js](../static/js/_utils.js#L82-L87) auf Python-Seite spiegelt: < 1 KB als `B`, < 1 MB als `KB`, sonst `MB`, jeweils eine Stelle nach Komma, Komma als Dezimal-Trenner (DE). Die Implementation kann via einen Jinja2-Filter (z.B. `{{ conversion.source_size_bytes|file_size }}`) erfolgen; Filter wird im App-Factory beim Template-Engine-Setup registriert. Damit ist „0.0 MB" für sub-MB-Dateien beseitigt.
- **Visuelle Hinweise:** keine neuen Komponenten. Reine Wert-Formatierung.
- **Microcopy** (DE, Beispiele):
  - 222 B → „222 B"
  - 4 731 B → „4,6 KB"
  - 1 234 567 B → „1,2 MB"
- **Helper-Reuse:** Python-Spiegel der `formatFileSize`-Logik. Sprint-Prompt-Konvention: Cluster-3-Pattern referenziert primär `_utils.js`-Helper — hier ist die Funktion auf Server-Seite, nicht JS-erreichbar (Server-Render im Jinja2-Template). Daher Server-side-Spiegel der Helper-Logik, nicht direkt der `_utils.js`-Helper. Inline-Code-Anker: [templates/library_detail.html:79](../templates/library_detail.html#L79).
- **Aufwand:** XS — Jinja2-Filter (~5 Zeilen Python) + Template-Stelle anpassen. Zur sicheren Konvergenz mit der JS-Helper-Logik einen kleinen Doctest oder Charakterisierungstest mit denselben Beispielen wie `formatFileSize` (222 B → „222 B", 4731 B → „4,6 KB", 1 234 567 B → „1,2 MB").
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 13: Page-`<title>` aktualisieren nach Title-Edit
**Adressiert Findings:** F16 (H1 Sev 1)

- **Pattern:** Im Erfolgs-Branch von `updateField('title', value)` wird `document.title = `${value} – Library`` gesetzt. Damit ist der Browser-Tab-Titel synchron mit dem aktuellen DB-Wert. Bei leerem Wert (User löscht den Titel) Fallback auf `Ohne Titel – Library` (DE-Fallback aus P6).
- **Visuelle Hinweise:** keine — reine `document.title`-Aktualisierung.
- **Microcopy** (DE):
  - Browser-Tab-Title default: `{Titel} – Library`
  - Browser-Tab-Title bei leerem Titel: `Ohne Titel – Library`
- **Helper-Reuse:** keine. Inline-Code-Anker: [static/js/library_detail.js:8-18](../static/js/library_detail.js#L8-L18) `updateField`-Success-Branch; [templates/library_detail.html:3](../templates/library_detail.html#L3) `<title>`-Tag (Server-side-Render bleibt für initialen Page-Load).
- **Aufwand:** XS — eine bedingte Zeile im `updateField`-Success-Branch (nur bei `field === 'title'`).
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 14: `loadSuggestions` HTTP-Status-Check via `safeJSON`
**Adressiert Findings:** F17 (H9 Sev 1)

- **Pattern:** `loadSuggestions()` ersetzt den rohen `r.json()`-Pfad durch `safeJSON(r)` aus [static/js/_utils.js](../static/js/_utils.js#L7-L16). Der Helper detektiert HTML-Redirects (Session-Expired) und non-JSON-Response. Zusätzlich `if (!r.ok)`-Branch für 4xx/5xx, der die Suggestions-Cache-Befüllung überspringt und einen dezenten Warning-Banner zeigt (oder still bleibt — Suggestions sind nice-to-have). Heute ist das Backend [app_pkg/integrations/notion.py:80-96](../app_pkg/integrations/notion.py#L80-L96) so geschrieben, dass es immer 200 mit Empty-Object zurückgibt — der Pattern ist Defensive-Code für künftige Backend-Änderungen.
- **Visuelle Hinweise:** keine neuen Komponenten. Im 4xx/5xx-Pfad bleibt der Suggestions-Container leer (datalists ohne Optionen), kein User-Sichtbarkeits-Verlust.
- **Microcopy** (DE, Du-Form):
  - keine neuen Strings im Default-Pfad. Bei Session-Expired (`safeJSON`-Throw): siehe P15 für gemeinsame Session-Expired-Microcopy.
- **Helper-Reuse:** `safeJSON(r)` ersetzt `r.json()`. Inline-Code-Anker: [static/js/library_detail.js:77-83](../static/js/library_detail.js#L77-L83).
- **Aufwand:** XS — `r.json()` → `await safeJSON(r)` + `r.ok`-Branch (3 Zeilen).
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 15: Banner-Mountpoint-Container im Template (struktureller Vorbedingung-Fix)
**Adressiert:** strukturelle Voraussetzung für P1, P3, P7

- **Pattern:** Zwei Banner-Container ergänzen, damit Banner aus P1, P3, P7 einen festen Mountpoint haben. `#detail-alert-container` direkt unter dem Header (für Auto-Save- und Delete-Failures aus P1 und P3); `#notion-alert-container` direkt unter dem Notion-Toggle-Header und über dem Notion-Panel (für Notion-Submit-Failures aus P7). Container leer im Default-State, kein Spacing wenn ohne Inhalt. Strukturelle Vorbereitung für die Banner-Patterns; pro Container-Sektion ist die Banner-Microcopy lokal zur User-Aktion. Optionale Session-Expired-Microcopy (Helper `safeJSON` wirft `Session expired – please reload the page and log in again.` als generische `Error`-Message): wird im Catch-Block der Banner-Patterns abgefangen und mit DE-Microcopy überschrieben — „Sitzung abgelaufen. Seite neu laden und erneut anmelden."
- **Visuelle Hinweise:** Zwei `<div>`-Tags mit den IDs `#detail-alert-container` und `#notion-alert-container` an den genannten Stellen im Template. Container-CSS ist `display: contents` oder einfach kein Default-Style — Banner-Höhe übernimmt der `c-alert`-Block.
- **Microcopy** (DE, Du-Form):
  - Session-Expired (gemeinsamer Fallback aus `safeJSON`): „Sitzung abgelaufen. Seite neu laden und erneut anmelden."
- **Helper-Reuse:** Container-Tags werden beim Aufruf von `showAlert(containerEl, …)` referenziert. Bestehender `showAlert`-Helper trägt Close-Button + Auto-Dismiss-Verhalten.
- **Aufwand:** XS — zwei `<div>`-Tags im Template.
- **Impact-Score:** keine direkte Finding-Adresse, aber Voraussetzung für P1/P3/P7 — implizite Schwelle für Cluster I (siehe Cluster-Vorschlag unten).

---

## Bug-Tickets ohne UX-H-Komponente (nicht in F-3.3)

Aus den Stage-2-Bug-Tickets sind zwei explizit nicht in F-3.3 adressiert (Sprint-Prompt Out-of-scope):

- **BT7: Notion-Field-Textarea-Content nicht escaped.** Heute unausnutzbar (default `value: ''`), reine Defensive-Code-Hygiene. Code-Anker: [static/js/library_detail.js:134-136](../static/js/library_detail.js#L134-L136). Gehört in einen Sammel-Bug-Pass oder wird mit-genommen, wenn `renderNotionFields` für P4 ohnehin angefasst wird. **Pure Bug-Ticket, kein UX-H-Aspekt.**
- **BT8: `window.open(url, '_blank')` ohne `noopener,noreferrer`.** Notion-Domain trusted, defensiv-defaulting trotzdem Best-Practice. Code-Anker: [static/js/library_detail.js:172](../static/js/library_detail.js#L172). Gehört in einen Sammel-Bug-Pass oder wird mit-genommen, wenn der Notion-Submit-Erfolgspfad für P7/P8 ohnehin angefasst wird. **Pure Bug-Ticket, kein UX-H-Aspekt.**

BT1–BT6 sind in den Patterns ihrer verknüpften Findings adressiert (siehe Pattern-Block-Header).

---

## Cluster-Vorbereitung für Implementation

**Drei-Cluster-Default — Cluster I = 6 Patterns + 1 strukturell, Cluster II = 2 Patterns, Cluster III = 6 Patterns.**

### Cluster I (Foundation Sweep + Silent-Failure-Elimination)

Patterns: **P15 (Container-Vorbereitung), P1, P3, P6, P7, P8, P14**

Begründung Gruppierung: Cluster I konsolidiert die Helper-/Microcopy-/Container-Refactors, die für den Banner-Pfad und den DE-Microcopy-Pass gemeinsam sind. P15 ist strukturelle Voraussetzung für P1, P3, P7 — muss am Anfang. P6 (DE-Microcopy) ist die Sweep-Stelle, an der ohnehin viele Strings angefasst werden — sinnvoll, sie früh zu integrieren, damit die Banner- und Toast-Microcopy aus P1/P3/P7/P8 gleich in DE landet. P14 (`safeJSON`-Reuse in `loadSuggestions`) ist Cross-Feature-Helper-Disziplin — XS, gut mit der Sweep zusammen.

**Silent-Failure-Sub-Gruppe (Smoke-Pflicht):** P1 (Auto-Save), P3 (Delete) — beide aus Cluster 1 der Findings. Vor Apply DevTools-Network-Throttle-Smoke: Title-Edit + Tab → Banner-Reaktion sichtbar; Delete → bei 5xx Banner statt stiller Page-Bleibe.

**Cross-Feature-H4-Sub-Gruppe:** P6 (DE-Microcopy), P7 (showAlert-statt-Toast), P8 (Toast-Level), P14 (safeJSON). Vier Patterns, die `library_detail` strukturell auf F-1 / F-2-Niveau zurückführen. Statisch verifizierbar, keine Smoke-Pflicht.

### Cluster II (Notion-Form-Stability)

Patterns: **P4, P5**

Begründung Gruppierung: Beide adressieren Cluster 2 aus den Findings (Notion-State-Wipe + UTC-Default). Beide haben `🔥 Smoke-Pflicht in F3-IMPL`-Tag. P4 ist die größte Mechanik in F-3.3 (M-Aufwand) wegen FormData-Snapshot/Restore. P5 ist XS, lässt sich am besten parallel als zweiten Commit im selben Cluster mit-implementieren, weil beide den `renderNotionFields`-Pfad anfassen.

**Smoke-Sequenz vor Apply:** Notion-Panel öffnen → Felder ausfüllen → Panel zuklappen → wieder aufklappen → erwartet: Felder noch da. Target-Switch → erwartet: Title/Tags/Body-Pool übertragen, sonst Default. Datum-Inspektion: Notion-Panel öffnen → Target Meeting → Datum-Default mit System-Uhr vergleichen (Europe/Berlin).

### Cluster III (Polish + a11y)

Patterns: **P2, P9, P10, P11, P12, P13**

Begründung Gruppierung: kleinere UX- und a11y-Patterns ohne Smoke-Pflicht. P9 (Tag-Chips, M-Aufwand) ist der schwerste Block in Cluster III; restliche sind XS. Reihenfolge frei wählbar, aber sinnvoll: P10 (aria-expanded) und P11 (Sidebar-Active) zuerst, weil sie in `base.html` bzw. Template-Header-Struktur leben und keine Konflikte mit anderen Patterns haben. P12 (File-Size Server-side) braucht Jinja2-Filter-Registrierung im App-Factory — am besten Einzel-Commit. P13 (Page-Title) ist ein Einzeiler im updateField-Success-Branch, kann in Cluster I mit-genommen werden, falls das `updateField`-Modul ohnehin für P1 angefasst wird (Synergie-Hinweis).

### Zwei-Cluster-Empfehlung (falls Cluster I als zu groß empfunden wird)

7 Patterns inklusive P15 in Cluster I ist im oberen Drittel (F-1.3 Cluster Polish-1 hatte 7, F-2.3 Cluster Ia hatte 3). Wenn der F3-IMPL-Sub-Thread Cluster I als zu unhandlich empfindet, sinnvoller Split:

- **Cluster Ia (Foundation Sweep):** P15, P6, P8, P14 — strukturelle Container + DE-Microcopy + Toast-Level + safeJSON-Reuse. Mechanisch, statisch verifizierbar, kein Smoke. Ein Commit.
- **Cluster Ib (Silent-Failure-Banner):** P1, P3, P7 — drei `showAlert`-basierte Banner-Patterns. Zwei davon (P1, P3) Smoke-Pflicht. Eigener Commit.

Cluster II und III bleiben unverändert.

---

## Top-5 Quick-Wins

**Aufwand-Gewicht:** XS=1, S=2, M=4, L=8. Score = Sev × 5 / Aufwand-Gewicht. Höher = besser.

| Rang | Pattern # | Adressiert | Sev | Aufwand | Impact-Score | Quick-Win |
|------|-----------|------------|-----|---------|--------------|-----------|
| 1 | P5 | F8 — Datum-Default lokal statt UTC | 3 | XS | 15.0 | ★ Top-5 |
| 2 | P8 | F10 — Toast-Level pro Call-Site | 2 | XS | 10.0 | ★ Top-5 |
| 3 | P1 | F1, F2 — Auto-Save Failure-Banner | 3 | S | 7.5 | ★ Top-5 |
| 4 | P3 | F4, F5 — Delete Failure-Banner | 3 | S | 7.5 | ★ Top-5 |
| 5 | P6 | F7 — DE-Microcopy-Pass | 3 | S | 7.5 | ★ Top-5 |
| 6 | P2 | F3 — Auto-Save Pending-Indikator | 3 | S | 7.5 | |
| 7 | P7 | F9 — Notion-Submit Persistent Banner | 2 | S | 5.0 | |
| 8 | P10 | F13 — Notion-Toggle aria-expanded | 1 | XS | 5.0 | |
| 9 | P11 | F14 — Sidebar-Active-State | 1 | XS | 5.0 | |
| 10 | P12 | F15 — File-Size KB/B-Fallback | 1 | XS | 5.0 | |
| 11 | P13 | F16 — Page-Title-Update | 1 | XS | 5.0 | |
| 12 | P14 | F17 — loadSuggestions safeJSON | 1 | XS | 5.0 | |
| 13 | P4 | F6, F11 — Notion-Form-State-Preservation | 3 | M | 3.75 | |
| 14 | P9 | F12 — Tags-Chip-Visualisierung | 2 | M | 2.5 | |

**Top-5 Quick-Wins:**

1. **P5 — Datum-Default lokal statt UTC** (15.0): drei Zeilen Code-Diff schließen einen Sev-3-Datenfehler-Pfad. Höchster Impact-Score der Stage. Cluster 2, Smoke-Pflicht.
2. **P8 — Toast-Level pro Call-Site** (10.0): pro Call-Site eine Options-Object-Ergänzung, beseitigt einen visuellen Mismatch (grüner Toast mit „Error"). Effektiv reduziert auf 1–2 Stellen, weil Notion-Errors über P7 zu Bannern werden.
3. **P1 — Auto-Save Failure-Banner** (7.5): höchste tägliche Trefferhäufigkeit unter den Sev-3-Pfaden, weil Title/Tags-Edit Reader-Daily-Usage ist. Cluster 1, Smoke-Pflicht. Bug-Ticket BT1 mit-gelöst.
4. **P3 — Delete Failure-Banner** (7.5): selbe Mechanik wie P1, anderer Pfad. Cluster 1, Smoke-Pflicht. Bug-Ticket BT2 mit-gelöst.
5. **P6 — DE-Microcopy-Pass** (7.5): hoher Volume-Gewinn (~18 Strings), aber jede Stelle 1:1-Replace. Cross-Feature-Konvergenz auf F-1 Cluster Polish-1 / F-2.3 P12. Sinnvoll mit P1/P3/P7/P8 zusammen, weil dieselben Dateien angefasst werden.

P2 (Auto-Save Pending-Indikator), P7 (Notion-Submit Banner) und P4 (State-Preservation) liegen knapp dahinter — alle drei sind Pflicht-Fixes für die jeweiligen Cluster, nur per Aufwand nicht mehr in den Top-5.

---

## Smoke-Pflicht-Übersicht

Patterns mit `🔥 Smoke-Pflicht in F3-IMPL`-Sub-Tag (Cluster 1 und Cluster 2 — Code-only-Befunde, vor Apply per Live-Smoke verifizieren):

| Pattern | Adressiert | Cluster | Smoke-Mechanik |
|---------|-----------|---------|----------------|
| **P1** | F1, F2 — Auto-Save Title/Tags + Favorite | 1 | DevTools-Network-Throttle: PUT auf 5xx → Banner sichtbar |
| **P3** | F4, F5 — Delete | 1 | DevTools-Network-Throttle: DELETE auf 5xx → Banner sichtbar, keine Navigation |
| **P4** | F6, F11 — Notion-Form-State | 2 | Browser-Reload-Sequenz: Panel-Open → Felder ausfüllen → Re-Toggle → Felder noch da. Target-Switch → Title/Tags/Body-Pool übertragen |
| **P5** | F8 — UTC-Datum | 2 | Datum-Inspektion: Notion-Panel → Target Meeting → Datum-Default mit System-Uhr (Europe/Berlin) vergleichen |

**Anzahl Smoke-Pflicht-Patterns:** 4 (von 14). P15 (Container) trägt keinen Smoke-Tag, aber Apply der Banner-Patterns P1/P3 setzt P15 ohnehin voraus.

Cluster 3 (Cross-Feature-Helper-Drift, P6/P7/P12 mit Cluster-3-Markierung) braucht keine Smoke-Pflicht — Helper-Reuse + DE-Microcopy + File-Size-Filter sind statisch-Code-Inspection-verifizierbar, kein Runtime-Befund.

---

## Helper-Vorschläge (für F3-IMPL-* zur Entscheidung)

Beim Pattern-Schreiben sind zwei mögliche neue `_utils.js`-Helper aufgefallen, die für künftige Wiederverwendung sinnvoll wären — **nicht** still im jeweiligen Pattern mit-anlegen, sondern F3-IMPL-* entscheidet, ob die Helper im Pattern-Cluster mit-implementiert oder als separater Helper-Cluster vorgezogen werden:

- **`formatDatetimeLocalNow()`** — gibt den aktuellen Zeitstempel als lokales `YYYY-MM-DDTHH:MM`-String zurück (für `<input type="datetime-local">`-Default). Heute nur in P5 verwendet. Wenn künftig weitere Features Notion-/Calendar-/Meeting-Felder haben (siehe Memory `project_readwise_replacement.md` zu Reader-Layer-Ausbau), wird der Helper Cross-Feature relevant. Bis dahin reicht die Inline-Variante in `library_detail.js`.
- **`renderTagChips(csvString, container, options?)`** — sync-Render der Chip-Reihe aus einem CSV-String, mit `onRemove`-Callback. Heute in P9 nur in `library_detail.js` benötigt. Wenn Library-List-View den Chip-Render künftig wiederverwendet (heute nur lesend mit `.c-tag` ohne Remove-Pfad), wäre der Helper Cross-Feature angemessen. Bis dahin reicht die lokale Hilfsfunktion in `library_detail.js`.

**Disposition:** beide Helper bleiben im jeweiligen Pattern-Block als „Helper-Vorschlag" markiert; F3-IMPL-Sub-Thread entscheidet beim Cluster-Schnitt.

---

**Schweregrad-Skala (aus Stufe 2):**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse)
