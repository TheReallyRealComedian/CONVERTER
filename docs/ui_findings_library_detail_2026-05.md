# UX-Heuristik-Findings: library_detail (2026-05-09)

**Methodik:** Stufe 2 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Heuristisches Review der strukturierten Inventur aus Stufe 1.
**Quelle:** [docs/ui_inventory_library_detail_2026-05.md](ui_inventory_library_detail_2026-05.md)
**Heuristiken:** Nielsen H1 (Sichtbarkeit des Systemzustands), H4 (Konsistenz und Standards), H6 (Wiedererkennen statt Erinnern), H9 (Fehlermeldungen / Hilfe bei Fehlern)
**Produkt-Kontext:** Single-User (Oliver), LAN-only, login-protected. **`library_detail` ist die Reader-Funktion im Readwise-Ersatz-Kontext** — daily-usage. Severity-Bewertung wichtet den täglichen Treffer-Häufigkeits-Faktor mit (siehe Daily-Usage-Hinweis im Sprint-Prompt). Technisch versierter Nutzer; H4 erhöht durch Cross-Feature-Dimension — F-1 (`document_converter`) und F-2 (`audio_converter`) haben App-Konventionen etabliert (`showAlert`, `showToast`, `formatFileSize`, `safeJSON`, DE-Microcopy, Save-Btn-Lifecycle, `:focus-visible`), die `library_detail` teilweise nicht befolgt. H1 stark wegen Auto-Save- und Delete-silent-fail-Pfade. H9 stark wegen fehlender Recovery-Anleitungen. H6 wegen versteckter Tag-Visualisierung und Disclosure-State.

**Live-Walkthrough-Hinweis:** Master hat zwischen F3-PICK und F3-REVIEW keinen Live-Walkthrough nachgereicht. Findings, die auf live-nicht-verifizierten Code-Beobachtungen basieren, sind in der Severity-Spalte mit `⚠️ code-only` gekennzeichnet — Master kann zwischen F3-REVIEW und F3-PATTERNS Walkthrough nachreichen.

---

## Findings (sortiert absteigend nach Schweregrad)

| #   | Element / Befund | Problem (1–2 Sätze) | Heuristik | Schweregrad (1–4) | Inventur-Anker | Disposition |
|-----|------------------|---------------------|-----------|-------------------|----------------|-------------|
| F1  | Auto-Save (Title/Tags) silent-fail (Bem 2; Inventur #11/#29) | `updateField()` zeigt nur den Success-Toast und ignoriert `r.ok===false` und Network-Failures komplett — kein `.catch`, kein `if (!r.ok)`-Branch. User editiert Title, drückt Tab, sieht **keinen** Toast (oder den falschen — siehe F10) und denkt „gespeichert". Tatsächlich nicht. | H1 | **3** ⚠️ code-only | Inventur Bem 2 (#2 in F-3.1-Disposition) | Finding + Bug-Ticket BT1 |
| F2  | Auto-Save (Title/Tags) silent-fail Recovery (siehe F1) | Da der Fehler komplett unkommuniziert ist, fehlt jede Recovery-Anleitung („versuche erneut" / „Verbindung prüfen"). User merkt erst beim nächsten Reload, dass die Änderung weg ist. | H9 | **3** ⚠️ code-only | Inventur Bem 2 | Finding + Bug-Ticket BT1 |
| F3  | Auto-Save Pending-State unsichtbar (Bem 16; Inventur #11/#29) | `onchange="updateField('title', this.value)"` feuert nur bei Blur/Enter. Wenn der User tippt und schnell zur nächsten Tab navigiert (Cmd+Tab) ohne Blur, wird **gar nicht gespeichert** — und es gibt keinen visuellen „pending changes"/Dirty-Indikator, der diesen Zustand kommunizieren würde. | H1 | **3** | Inventur Bem 16 | nur Finding |
| F4  | Delete silent-fail (Bem 3; Inventur #17) | `deleteConversion()` führt `DELETE` aus und navigiert nur bei `r.ok` zur Library. Bei 4xx/5xx oder Network-Fail bleibt der User auf der Detail-Seite **ohne jeden Hinweis**, dass die Aktion fehlschlug. | H1 | **3** ⚠️ code-only | Inventur Bem 3 | Finding + Bug-Ticket BT2 |
| F5  | Delete-Failure Recovery fehlt (siehe F4) | Keine Recovery-Anleitung — der User klickt erneut, ohne zu wissen, ob das Backend down ist, ob die Conversion bereits weg ist, oder ob seine Session abgelaufen ist. Selbe Familie wie F-2.2 F5 (Mic-Permission silent). | H9 | **3** ⚠️ code-only | Inventur Bem 3 | Finding + Bug-Ticket BT2 |
| F6  | Notion-Panel Re-Toggle löscht User-Inputs (Bem 5; Inventur #20/#25) | `toggleNotionPanel()` ruft beim Open immer `selectTarget(DEFAULT_TARGET)`, was `renderNotionFields()` triggert, was `container.innerHTML = …` zuweist. User füllt Felder aus, klappt Panel zu (z.B. um den Title oben zu ändern), klappt wieder auf → leere Felder, kein Hinweis, dass die Eingaben weg sind. | H1 | **3** ⚠️ code-only | Inventur Bem 5 | Finding + Bug-Ticket BT3 |
| F7  | Englische Strings flächendeckend (Bem 4; Inventur Header/Toolbar/Notion/Tags) | ~18 EN-Strings auf der Seite (Toast-Text, Button-Labels, Placeholders, `confirm()`-Dialog, Type-Badge-Labels). **F-1 Cluster Polish-1 hat document_converter auf mehrheitlich DE umgestellt; F-2 hat es als Sev-3-Cross-Feature-Bruch dokumentiert.** library_detail ist auf Pre-F-1-Polish-Stand. | H4 | **3** | Inventur Bem 4 (Pre-Existing-Item) | nur Finding |
| F8  | Datum-Default für Meeting-Notion-Field ist UTC, nicht lokal (Bem 15; Inventur #25) | `new Date().toISOString().slice(0,16)` liefert UTC-Zeit als Default für `<input type="datetime-local">`. Browser interpretiert den Wert als **lokale** Zeit — User in Europe/Berlin sieht einen 1–2h zurück verschobenen Default und sendet diese falsche Zeit nach Notion (Daten-Fehler in der Quelle, nicht nur im Display). | H1 | **3** ⚠️ code-only | Inventur Bem 15 | Finding + Bug-Ticket BT4 |
| F9  | Notion-Submit-Errors via `showToast` statt `showAlert` (Bem 10; Inventur #26) | `showToast` auto-dismissed nach 2.5s — bei 502 oder 4xx mit User-relevanten Fehler-Details hat der User nur 2.5s zum Lesen. **F-1 Cluster B/C hat `showAlert` exakt für diesen Use-Case eingeführt** (persistente Banner mit Close-Button); Helper steht in `_utils.js` bereit, wird in library_detail nicht genutzt. | H4 | **2** | Inventur Bem 10 | nur Finding |
| F10 | Toast-Level falsch für Error-Pfade (Bem 13; Inventur #14/#26/#30) | Alle `showToast`-Calls in `library_detail.js` lassen `opts` weg → default `level='success'` (grün). Auch für „Copy failed", „Error: …", „Failed to connect to Notion". User sieht einen grünen Toast mit „Error" — visuell-vs-textlich widersprüchlich. | H4 | **2** ⚠️ code-only | Inventur Bem 13 | Finding + Bug-Ticket BT5 |
| F11 | Target-Switch löscht User-Inputs (Bem 7; Inventur #22/#23/#24/#25) | Identisch zu F6, aber über den Target-Switch-Pfad: `selectTarget()` ruft `renderNotionFields()`, das `container.innerHTML = …` setzt. User füllt Meeting-Felder aus, switcht aus Versehen auf Note → alles weg. Niedrigere Trigger-Häufigkeit als F6 (User-Versehen-Klick statt Re-Toggle). | H1 | **2** ⚠️ code-only | Inventur Bem 7 | nur Finding |
| F12 | Tags-Input ohne Chip-Visualisierung (Bem 12; Inventur #29) | Plain CSV-Textfeld. CSS hat `.c-tag { background: var(--nm-tint-accent); }` definiert (Library-List-View nutzt es vermutlich). Detail-View lässt die Tags völlig unstyled — kein visueller Hinweis, was als Tag interpretiert wird, kein Add/Remove einzelner Tags. | H6 | **2** | Inventur Bem 12 | nur Finding |
| F13 | Notion-Toggle-Header fehlt `aria-expanded` / `aria-controls` (Bem 6; Inventur #20) | Klassisches Disclosure-Widget ohne ARIA-Annotation. Screenreader/Keyboard-User können den Open/Closed-Status nicht wahrnehmen. | H6 | **1** | Inventur Bem 6 | nur Finding |
| F14 | Sidebar-Active-State fehlt auf Detail-Seite (Bem 1; Inventur #7) | `base.html` markiert die `Library`-Nav nur als aktiv, wenn `request.endpoint == 'library'`. Auf `library_detail` ist der Endpoint anders, daher fehlt der visuelle „du bist im Library-Bereich"-Cue. | H4 | **1** ⚠️ code-only | Inventur Bem 1 | Finding + Bug-Ticket BT6 (conditional auf Walkthrough) |
| F15 | File-Size sub-MB → „0.0 MB" (Bem 11; Inventur #27) | Server-side Template-Render `{{ "%.1f"|format(conversion.source_size_bytes / 1048576) }} MB` liefert für Sub-MB-Dateien „0.0 MB". **F-1 hat denselben Bug per `formatFileSize`-Helper in `_utils.js` gefixt**; library_detail kann den JS-Helper hier nicht direkt nutzen (Server-Render), folgt der Konvention aber trotzdem nicht. | H4 | **1** | Inventur Bem 11 | nur Finding |
| F16 | Page-`<title>`-Tag wird nicht aktualisiert nach Title-Edit (Bem 17; Inventur #11) | Nach `updateField('title', …)` ändert sich das Server-side gerenderte `<title>{{ conversion.title }} - Library</title>` nicht. Browser-Tab zeigt alten Titel — Status-Drift gegenüber dem aktuellen DB-Zustand. | H1 | **1** | Inventur Bem 17 | nur Finding |
| F17 | `loadSuggestions` prüft kein HTTP-Status (Bem 8; Inventur #25) | `fetch('/api/notion/suggestions').then(r => r.json())` — wenn das Backend 4xx/5xx mit JSON-Body liefert, wird der Error-Body als Suggestions interpretiert. Heute backend [app_pkg/integrations/notion.py:80-96](app_pkg/integrations/notion.py#L80-L96) gibt immer 200 mit Empty-Object zurück, also faktisch unkritisch — aber fragiler Code, keine Recovery-Anleitung wenn das Backend künftig 5xx liefert. | H9 | **1** | Inventur Bem 8 | nur Finding |

---

## Cross-Feature-Konventionsbrüche (Subset der obigen Findings — Querverweis-Hilfe für Stufe 3)

Diese Findings existieren ausschließlich oder primär, weil F-1 / F-2 bereits einen App-Standard etabliert haben, dem `library_detail` nicht folgt. **Stage-3-Implikation:** alle dieser Findings sind mit *existing-helper-reuse* (oder Pattern-Konvergenz aus F-1/F-2) lösbar — XS bis S Aufwand bei oft hoher Severity.

| F#  | Konvention aus F-1 / F-2 | F-1 / F-2 Pattern / Helper / Commit-Anker |
|-----|--------------------------|--------------------------------------------|
| F1, F2 | `showAlert(container, level, msg, options)` für Save-Failure-Banner | F-1 Cluster B Pattern 4 (Save-Failure-Banner); Helper in [static/js/_utils.js](../static/js/_utils.js) |
| F4, F5 | `showAlert` für Delete-Failure (selbe Familie) | F-1 Cluster B/C Pattern (`showAlert` für alle destruktiven-Pfad-Fehler); F-2.2 F5/B3 Mic-Permission-Pattern |
| F7 | DE-Microcopy, Du-Form, Verb+Objekt-Buttons | F-1 Cluster Polish-1 (7 Strings DE); F-2.2 F18 als Cross-Feature-H4-Sev-3 dokumentiert |
| F9 | `showAlert` statt `showToast` für persistente Fehler-Banner | F-1 Cluster B Pattern 4; F-2.2 F11 (Save-Failure-via-`alert()`) |
| F12 | `.c-tag`-Chip-Rendering konsistent zwischen Library-List und Detail | App-internes `.c-tag` CSS bereits definiert (Library-List-View nutzt es) |
| F15 | `formatFileSize` Konvention (KB/B-Fallback) | F-1 Cluster Polish-1 (Helper in `_utils.js`); F-2.2 F26 als Cross-Feature-H4-Sub |

**6 Cross-Feature-H4-Findings** von 17 — **Cross-Feature-Konvergenz-Quote 35%** (etwas niedriger als F-2.2's 41%, weil library_detail strukturell weniger Berührungspunkte mit den Konverter-Konventionen hat — keine Drop-Zone, keine `.saved`-Klasse-Pattern, keine Backend-Whitelist). Zusätzlich zwei mit-induzierte Schwächen: F10 (Toast-Level — interne Konsistenz innerhalb library_detail.js, **nicht** Cross-Feature) und F11 (Target-Switch — selber Fix-Pfad wie F6, kein eigenes Konvention-Bruch).

**Stage-3-Hinweis:** Findings F1/F2/F4/F5 lassen sich in einen einzigen „Silent-Failure-Elimination"-Cluster zusammenführen (showAlert-Reuse + `.catch`-Branch + r.ok-Check). Findings F7/F9/F15 in einen „Helper-/Pattern-Konvergenz"-Cluster (DE-Microcopy + showAlert + formatFileSize-Konvention).

---

## Reine Implementierungs-Bugs (ohne eigenständiges Heuristik-Finding, separates Ticket-Material)

Diese Befunde sind in den Findings oben **nicht** als Heuristik-Findings erfasst, weil sie keine direkte UX-Heuristik-Komponente haben (User sieht keinen Unterschied im aktuellen Zustand). Sie brauchen aber konkrete Code-Fixes:

- **BT1: `updateField` und `toggleFavorite` haben kein Error-Handling.** Kein `.catch`, kein `if (!r.ok)`-Branch. → siehe Findings F1, F2. Code-Anker: [static/js/library_detail.js:8-18](../static/js/library_detail.js#L8-L18) `updateField`; [static/js/library_detail.js:20-32](../static/js/library_detail.js#L20-L32) `toggleFavorite`. Reproduktion: Server bei `PUT /api/conversions/<id>` auf 5xx setzen → Frontend zeigt Toast „field updated" oder gar nichts. Vorgeschlagener Fix-Pfad: `.catch` ergänzen + `if (!r.ok)`-Branch + `showAlert(..., 'danger', …)` (siehe Pattern-Sprint F3-PATTERNS).
- **BT2: `deleteConversion` hat kein Error-Handling.** Kein else-Zweig nach `r.ok`-Check. → siehe Findings F4, F5. Code-Anker: [static/js/library_detail.js:55-60](../static/js/library_detail.js#L55-L60). Reproduktion: Server bei `DELETE /api/conversions/<id>` auf 5xx setzen → User klickt Delete, bestätigt confirm(), nichts passiert sichtbar. Vorgeschlagener Fix-Pfad: else-Zweig + `showAlert(..., 'danger', …)`.
- **BT3: `toggleNotionPanel` und `selectTarget` zerstören User-Inputs durch unbedingten `renderNotionFields`-Call.** `toggleNotionPanel` ruft beim Open immer `selectTarget(DEFAULT_TARGET)`; `selectTarget` ruft immer `renderNotionFields()` mit `container.innerHTML = …`. → siehe Finding F6 (und F11 als verwandte Mechanik). Code-Anker: [static/js/library_detail.js:66-75](../static/js/library_detail.js#L66-L75) `toggleNotionPanel`; [static/js/library_detail.js:85-91](../static/js/library_detail.js#L85-L91) `selectTarget`. Reproduktion: Notion-Panel öffnen, Felder ausfüllen, Panel zuklappen, wieder öffnen → Felder leer. Vorgeschlagener Fix-Pfad: `renderNotionFields()` nur ausführen, wenn `container` leer **oder** ein Target-Switch tatsächlich erfolgt; Form-State über `panel.classList.contains('hidden')`-Wechsel hinweg erhalten.
- **BT4: `new Date().toISOString().slice(0,16)` ist UTC für `<input type="datetime-local">`.** → siehe Finding F8. Code-Anker: [static/js/library_detail.js:96](../static/js/library_detail.js#L96). Reproduktion: in Europe/Berlin-Browser Notion-Panel öffnen, Target=Meeting → Datum-Default zeigt 1–2h zurück verschobene Zeit. Vorgeschlagener Fix-Pfad: lokales Datum-Format ohne `.toISOString()` (z.B. `new Date(Date.now() - new Date().getTimezoneOffset()*60000).toISOString().slice(0,16)` oder dedizierter Helper).
- **BT5: Toast-Level wird in allen `showToast`-Calls nicht gesetzt.** Default `level='success'` greift überall. → siehe Finding F10. Code-Anker: [static/js/library_detail.js:36-38](../static/js/library_detail.js#L36-L38) Copy-Failure; [static/js/library_detail.js:174](../static/js/library_detail.js#L174) Notion-Status-Error; [static/js/library_detail.js:177](../static/js/library_detail.js#L177) Notion-Network-Failure. Reproduktion: Copy bei leerem `<pre>` triggern oder Notion-Submit auf 502 setzen → grüner Toast mit „Error: …"-Inhalt. Vorgeschlagener Fix-Pfad: pro Call-Site das passende Level setzen (`{ level: 'danger' }` für Fehler-Pfade); für persistente Banner siehe F9 (showAlert) statt nur Toast-Level-Fix.
- **BT6: Sidebar-Active-State auf Detail-Seite (conditional auf Walkthrough).** Vermutlich fehlend, weil `base.html` nur `request.endpoint == 'library'` als Active markiert. → siehe Finding F14. Code-Anker: nicht eingelesen — `templates/base.html` rund um die Nav-Links. Reproduktion: nach Walkthrough-Verifikation. Vorgeschlagener Fix-Pfad: Endpoint-Check auf `request.endpoint in ('library', 'library_detail')` erweitern, oder sauberer per `request.blueprint`-Match (Hinweis: das Projekt nutzt aber kein Blueprint, sondern flache Routes — daher Endpoint-Liste).
- **BT7: Notion-Field-Textarea-Content nicht escaped (Bem 9; Inventur #25).** `<textarea …>${f.value}</textarea>` ist roh interpoliert; `<input value="${esc(f.value)}">` ist hingegen escaped. Heute faktisch unausnutzbar (`description`/`summary` haben default `value: ''`), aber Code-Hygiene-Lücke. **Aus F-3.2 herausgenommen, weil keine UX-Heuristik-Komponente** (User sieht heute keinen Unterschied; reine Defensive-Code-Hygiene). Code-Anker: [static/js/library_detail.js:134-136](../static/js/library_detail.js#L134-L136). Reproduktion: künftige Erweiterung mit User-Content-Default in `description`-Field könnte Injection ermöglichen — heute nicht erreichbar. Vorgeschlagener Fix-Pfad: `esc()`-Helper auch auf Textarea-Content anwenden.
- **BT8: `window.open(url, '_blank')` ohne `noopener,noreferrer` (Bem 14; Inventur #26/#32).** Standard-Praxis-Lücke. URL kommt aus Notion-API-Response (vertrauenswürdige Quelle), defensiv-defaulting trotzdem Best-Practice. **Aus F-3.2 herausgenommen, weil keine UX-Heuristik-Komponente** (User sieht keinen Unterschied; reine Defensive-Code-Hygiene). Code-Anker: [static/js/library_detail.js:172](../static/js/library_detail.js#L172). Reproduktion: nicht ausnutzbar, weil Notion-Domain trusted. Vorgeschlagener Fix-Pfad: drittes Argument `'noopener,noreferrer'` auf `window.open` (`window.open(url, '_blank', 'noopener,noreferrer')`), zusätzlich `rel`-Attribut wenn ein Anchor-Element zwischen-eingeführt wird.

---

## Disposition-Verteilung

- **Nur Findings (kommen in F3-PATTERNS):** 9 — F3, F7, F9, F11, F12, F13, F15, F16, F17
- **Findings + Bug-Tickets (kommen in F3-PATTERNS **plus** separates Bug-Ticket):** 8 Findings → **6 unique Bug-Tickets** — F1+F2 (BT1), F4+F5 (BT2), F6 (BT3), F8 (BT4), F10 (BT5), F14 (BT6 conditional)
- **Nur Bug-Tickets (kommen **nicht** in F3-PATTERNS):** 2 — BT7 (textarea-escape), BT8 (window.open-noopener)

**Inventur-Befund-Coverage (alle 18 disponiert):**
- #1 → F14 + BT6 (Walkthrough-Bestätigung nötig)
- #2 → F1 (H1) + F2 (H9) + BT1
- #3 → F4 (H1) + F5 (H9) + BT2
- #4 → F7 (Pre-Existing-Item, F-3-IMPL nach Pattern-Sprint)
- #5 → F6 + BT3
- #6 → F13
- #7 → F11 (Bug-Ticket gefoldet in BT3-Erweiterung in F-3-IMPL)
- #8 → F17
- #9 → BT7 (kein UX-H-Aspekt — abweichend von F-3.1-Disposition „nur Finding")
- #10 → F9
- #11 → F15
- #12 → F12
- #13 → F10 + BT5
- #14 → BT8 (kein UX-H-Aspekt — abweichend von F-3.1-Disposition „nur Finding")
- #15 → F8 + BT4
- #16 → F3
- #17 → F16
- #18 → Helper-Reuse-Beobachtung (Inventur-Meta-Sektion); in den Findings F1/F2/F4/F5/F9/F15 als Cross-Feature-Anker absorbiert (siehe Cross-Feature-Sektion oben). Kein eigenständiges Finding.

**Abweichungen von F-3.1-Disposition (begründet):**
- **#9 textarea-escape:** F-3.1 schlug „nur Finding" vor. F-3.2 ordnet als „nur Bug-Ticket" ein, weil **keine UX-Heuristik-Komponente** vorhanden ist (User sieht heute keinen Unterschied — reine defensive Code-Hygiene; H1/H4/H6/H9 treffen alle nicht). Filtert sich beim Heuristik-Filter heraus, gehört in den Sammel-Bug-Pass.
- **#14 window.open-noopener:** F-3.1 schlug „nur Finding" vor. F-3.2 ordnet als „nur Bug-Ticket" ein, weil **keine UX-Heuristik-Komponente** vorhanden ist (User sieht heute keinen Unterschied; reine Sicherheits-Best-Practice gegenüber non-trusted Domain — Notion-Domain ist trusted). Filtert sich beim Heuristik-Filter heraus.
- **#7 Target-Switch-Wipe:** F-3.1 hatte „Finding (Bug-Ticket eher in F-3-IMPL-Sammelpaket)". F-3.2 bestätigt „nur Finding" und folded den Bug-Aspekt explizit in BT3 (Notion-State-Preservation), weil F6 und F11 dieselbe Wurzel haben (`renderNotionFields()` ohne State-Preservation).
- **Inventur-Befund-Aufteilung:** F-3.1 hatte 18 nummerierte Befunde, F-3.2 produziert 17 Findings + 8 Bug-Tickets, weil mehrere Befunde in 2 Heuristik-Reihen aufgespalten wurden (#2 → F1+F2, #3 → F4+F5) und drei Befunde aus dem Findings-Pool fielen (#9, #14 zu Bug-only; #18 als Meta-Beobachtung absorbiert).

---

## Schwerpunkt-Cluster

Drei thematische Cluster, in denen sich die schweren Findings konzentrieren — analog F-1's „Empty-Submit-Silent / Result-Persistenz / Save-Button-Stale-Visual" und F-2's „Drag-Drop-Lüge / 11+ alert() / Config-Error-Global / Englische Strings":

### Cluster 1 — Silent-Failure-Familie (F1, F2, F4, F5; Sev 3 ⚠️ code-only)
Auto-Save (Title/Tags) und Delete sind komplett unbeobachtet. `updateField`, `toggleFavorite` und `deleteConversion` haben kein `.catch`, kein `if (!r.ok)`. **Daily-Usage-Schmerz hoch** für Title/Tags-Edit (Reader-Funktion täglich genutzt). F-1 Cluster B Pattern 4 hat genau dieses Pattern für `document_converter` etabliert (`showAlert`-Banner für Save-Failures); library_detail folgt nicht. Fix-Pfad in Stage 3: einen einzigen Pattern-Cluster „Silent-Failure-Elimination" der `showAlert` + Recovery-Microcopy ergänzt; Bug-Tickets BT1+BT2 unabhängig vorab fixbar.

### Cluster 2 — Notion-Side State-Wipe & UTC-Default (F6, F11, F8; Sev 2–3 ⚠️ code-only)
Drei Mechanismen, die unbemerkt User-Daten zerstören oder verfälschen, alle im Notion-Submit-Pfad:
- **F6 Re-Toggle-Wipe + F11 Target-Switch-Wipe:** `renderNotionFields()` setzt unbedingt `container.innerHTML = …` ohne State-Preservation. User füllt Felder aus → Re-Toggle oder Target-Switch → leer. Selbe Wurzel, zwei Trigger-Pfade.
- **F8 Datum-UTC:** `new Date().toISOString().slice(0,16)` schickt UTC-Zeit ins `<input type="datetime-local">`, das es als lokal interpretiert. User in Europe/Berlin sendet einen 1–2h falschen Meeting-Zeitstempel nach Notion — Datenfehler in der Zielsystem-Quelle, nicht nur im Display.

Notion-Submit ist nicht daily, aber wenn er passiert, sind diese Fehler subtil und schwer zu bemerken (User sieht beim Send-Erfolg den neuen Tab und denkt „passt"). Fix-Pfad in Stage 3: BT3 (State-Preservation in `renderNotionFields`) + BT4 (lokale Datum-Default-Funktion) als unabhängige Bug-Tickets vorab fixbar; Pattern-Cluster „Notion-Form-Stability" konsolidiert UX-Aspekt.

### Cluster 3 — Cross-Feature-Helper-Drift (F7, F9, F12, F15; Sev 1–3)
F-1/F-2 haben App-Konventionen etabliert (`showAlert`, DE-Microcopy, `formatFileSize`, `.c-tag`-Chip-Rendering), die `library_detail` nicht oder nur teilweise befolgt. **Cross-Feature-H4-Quote 35%** (6 von 17 Findings, F1/F2/F4/F5/F7/F9/F12/F15 zählen mit). Fix-Pfad in Stage 3: ein Pattern-Cluster „Helper-/Konvention-Konvergenz" der die Helper-Migration plus DE-Microcopy plus Tag-Chip-Rendering bündelt. **F1/F2/F4/F5 sind technisch auch Cross-Feature** (showAlert-Pattern aus F-1), aber primär unter Cluster 1 gefasst, weil die Silent-Failure-Wurzel tiefer liegt als die Helper-Wahl.

---

## Zusammenfassung

- **Heuristik-Findings gesamt:** 17
- **Davon Schweregrad 4 (kritisch):** 0 — keine Datenverlust-/Blockade-Pfade auf der primären Lese-Aufgabe (Inhalt-anschauen funktioniert; Edit-Pfade haben silent-fail aber nicht „blockierend")
- **Davon Schweregrad 3:** 8 (F1, F2, F3, F4, F5, F6, F7, F8 — Auto-Save-silent-fail-Familie + dirty-indicator + Delete-silent-fail + Notion-Re-Toggle-Wipe + EN-Strings + Datum-UTC)
- **Davon Schweregrad 2:** 4 (F9, F10, F11, F12 — showAlert-vs-showToast + Toast-Level-falsch + Target-Switch-Wipe + Tags-Chip-Rendering)
- **Davon Schweregrad 1:** 5 (F13, F14, F15, F16, F17 — aria-expanded + Sidebar-Active + File-Size-MB + Page-Title-Stale + loadSuggestions-Status)
- **Reine Bug-Tickets (mit Ticket-Material):** 8 (BT1 updateField/toggleFavorite-Errors, BT2 deleteConversion-Error, BT3 Notion-Form-State-Preservation, BT4 Datum-lokal-Default, BT5 Toast-Level-pro-Call-Site, BT6 Sidebar-Active conditional, BT7 textarea-escape, BT8 window.open-noopener) — davon 6 mit Finding-Verknüpfung, 2 pure Bugs ohne H-Aspekt
- **Cross-Feature-H4-Findings:** 6 von 17 explizit (F1/F2 + F4/F5 + F7 + F9 + F12 + F15) — **Cross-Feature-Konvergenz-Quote ~35%**, etwas niedriger als F-2.2's 41%, weil library_detail strukturell weniger Berührungspunkte mit den Konverter-Konventionen hat (keine Drop-Zone, keine `.saved`-Klasse, keine Backend-Whitelist)
- **`⚠️ code-only`-markierte Findings:** 9 (F1, F2, F4, F5, F6, F8, F10, F11, F14) — Master-Walkthrough-Nachreichung empfohlen vor F-3.3 für mindestens F6/F11 (Re-Toggle/Target-Switch-Wipe — Verhalten verifizieren), F8 (Europe/Berlin-Browser-Verschiebung beobachten), F14 (Sidebar-Active conditional auf Endpoint-Match). F1/F2/F4/F5 sind code-evidente silent-fails — Walkthrough nicht zwingend, aber kurze Verifikation per DevTools-Network-Throttle wäre günstig (5 Min Walkthrough). F10 (Toast-Level-Mismatch) lebt im visuellen Wahrnehmungs-Bereich — Walkthrough sehr günstig.

**Schweregrad-Skala:**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse)

**Bemerkungen-Disposition (18 Stufe-1-Bemerkungen):**
- Findings only: 9 (Bem 4, 6, 7, 8, 10, 11, 12, 16, 17)
- Beides (Finding + Bug-Ticket): 6 (Bem 1, 2, 3, 5, 13, 15)
- Bugs only: 2 (Bem 9, 14) — abweichend von F-3.1-Disposition (siehe „Abweichungen von F-3.1-Disposition" oben)
- Meta absorbiert: 1 (Bem 18 Helper-Reuse — in Cross-Feature-Sektion und einzelnen Findings absorbiert)

**Master-Walkthrough-Empfehlung vor F-3.3:** Ja, kurze Walkthrough-Session (15–30 Min) für die `⚠️ code-only`-Sev-3-Findings empfohlen — primär F6 (Re-Toggle), F11 (Target-Switch), F8 (UTC-Datum mit Europe/Berlin-Browser), F14 (Sidebar-Active Endpoint-Match). Auto-Save-silent-fail (F1/F2/F4/F5) per DevTools-Network-Throttle 5-Min-Verifikation. Toast-Level (F10) bei einem absichtlichen Notion-502-Trigger anschauen. Wenn Master-Bandwidth knapp ist: F-3.3 kann auch ohne Walkthrough beginnen, aber dann sollten die Pattern-Vorschläge für Cluster 1 und 2 das `⚠️ code-only`-Risiko explizit in den Phasen-Stops markieren, damit ein Implementierungs-Smoke vor Merge zwingend ist.
