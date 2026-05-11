# Sprint HYG2 — Hygiene-Welle 2: Sammel-Pass Aktive Reminder

**Datum**: 2026-05-11

**Ziel**: Sechs der acht Aktive-Reminder-Items aus dem Steady-State-BACKLOG in einem Sammel-Sprint abarbeiten, intern strukturiert als drei Sub-Batches (Library-Detail-Polish → Cross-Feature-Hygiene → Audio-UX). Die zwei verbleibenden Items (P8-Master-Smoke + F6-IMPL-Master-Smokes P2/P3/P11) sind Browser-Master-Aufgaben und bleiben außerhalb dieses Sprint-Scopes. **Kein Code-Touch außer für die sechs Sprint-Items**.

**Vorbedingung**:
- Pytest 71/71 grün auf `main`. Letzter Code-Touch: WAVE-CLOSE (commit `<wave-close-hash>`, 2026-05-11). Cleanup-Welle 2026-05 strukturell abgeschlossen.
- **Eingabe**: [BACKLOG.md](BACKLOG.md) — Sektion „P3 — Aktive Reminder" mit 8 Items. Sub-Thread liest komplett vor Phase 1.
- **Sechs Sprint-Items** (aus Aktive-Reminder; Sprint-Scope):
  1. **getUserMedia-in-socket.onopen-Bug** (audio_converter Live-Tab, S) — Permission-Prompt vor WS-Handshake statt nach `socket.onopen`.
  2. **BT7 textarea-escape** in `renderNotionFields` (library_detail.js, XS) — HTML-Escape für Textarea-Content.
  3. **BT8 window.open-noopener** im Notion-Submit-Erfolgspfad (library_detail.js, XS) — `noopener,noreferrer` ergänzen.
  4. **Opacity-Übergang 150ms beim Notion-Target-Switch** (library_detail.js + style.css, XS) — Cosmetic Pattern-Doc-Polish aus F-3.3 P4.
  5. **aria-live im `showToast`-Helper** (_utils.js, XS-S) — `role="status"` + `aria-live="polite"` cross-feature für alle Toast-Call-Sites.
  6. **Drei EN-Strings in `app_pkg/markdown.py:137/143/157`** (XS) — Backend-Flash DE-Microcopy-Pass.
- **Zwei Master-Aufgaben außerhalb Sprint-Scope** (bleiben in Aktive-Reminder):
  - P8-Master-Smoke (F-5-IMPL PDF-Gen-Error → flash statt 500-Page).
  - F6-IMPL-Master-Smokes P2/P3/P11 (Copy-Full-Content / Auto-Save-Failure-Banner / Card-Datum DE-Locale).
- **Methodik-Vorlagen** (Multi-Item-Sprints):
  - F-3-IMPL commits `843574b` / `40dd02e` / `b3e666a` (15 Patterns in 3 Sub-Batches, drei separate Commits).
  - F-6-IMPL commits `8049d3e` / `741794a` / `ca41270` (14 Patterns in 3 Sub-Batches mit BT-Folde).
  - F-5-IMPL commits `07e9aa6` / `9e6999c` (13 Patterns mit Schwester-Feature-Übernahme).
  - **HYG-Sprint 2026-05-09** als Hygiene-Welle-Mechanik-Vorlage (F-002/F-007/F-008/F-011/F-012/F-015/F-016/F-017, +5 Tests, 48/48 grün).
- **Test-Coverage**: aktuelle Suite 71/71 grün. Erwartete neue Tests:
  - 0-1 für Item 6 (DE-Microcopy-Backend-Test, optional analog F-3-IMPL-Pattern für DE-Strings).
  - 0-1 für Item 5 (showToast-aria-live-Roundtrip wenn JSDom-fähig — vermutlich nicht in mocked-SDK-Suite).
  - Erwartete Final-Anzahl: **71–73 Tests grün**.
- **Memory-Layer-Pflicht-Lese**: `feedback_no_silent_fixes.md`, `feedback_pragmatic_merge.md`, `feedback_push_is_normal.md`, `feedback_helper_reuse_design_choice.md`, `feedback_smoke_beats_pattern_text.md`, `reference_converter_dep_bump_constraints.md`.

**Out-of-scope**:
- **P8-Master-Smoke und F6-IMPL-Master-Smokes** (Items 7+8 aus Aktive-Reminder) — Browser-Master-Aufgabe, **nicht in Sub-Thread-Scope**. Sub-Thread berichtet im Phase-3-Bericht dass diese als Master-Smoke-Punch-Liste verbleiben.
- Tesseract-NC-33-Workaround — extern, kein CONVERTER-Touch.
- Deferred F-N-Wellen (`mermaid_converter`, `login`, Playwright-UI-Tests) — eigene Sprints wenn Trigger.
- Re-Strukturierung von library_detail.js oder audio_converter.js außerhalb der sechs Sprint-Items.

---

## Master-Annotation (vorab eingebettet)

### 1. Sub-Batch-Strategie 3 Sub-Batches A/B/C mit A1/A2-Fallback

| Sub-Batch | Items | Anzahl | Smoke-Pflicht | Begründung |
|-----------|-------|--------|---------------|------------|
| **A — Library-Detail-Polish** | Items 2, 3, 4 (BT7 + BT8 + Opacity-Übergang) | 3 | — | Alle drei Touches in `library_detail.js` + `style.css`. Verkoppelter Touch-Bereich (renderNotionFields + sendToNotion + selectTarget). Holistic-Apply möglich. **A1/A2-Split-Fallback** wenn doch zu groß: A1 = BT7+BT8 (Bug-Familie) / A2 = Opacity (Polish). |
| **B — Cross-Feature-Hygiene** | Items 5, 6 (aria-live showToast + DE-Microcopy markdown.py) | 2 | — | Item 5 ist cross-feature (alle Toast-Call-Sites prüfen). Item 6 ist isoliert backend. Atomic-Apply beide. |
| **C — Audio-UX** | Item 1 (getUserMedia-in-socket.onopen) | 1 | 🔥 Live-Mic-Master-Smoke | Refactor-Risiko: Permission-Prompt-Timing-Änderung kann Live-Tab brechen. Eigener Sub-Batch, **🔥 Live-Mic-Smoke-Pflicht** vor Apply-Sign-off. |

**Pflicht-Reihenfolge**: A → B → C, **ohne Auslassen**. STOP-Punkt mit Bericht nach jedem Sub-Batch.

**Apply-Reihenfolge in Sub-Batch A**:
1. BT7 zuerst (textarea-escape via `escapeHtml`-Pattern in `renderNotionFields` — wenn `_utils.js` einen `escapeHtml`-Helper hat verwenden, sonst inline).
2. BT8 zweitens (`window.open(url, '_blank', 'noopener,noreferrer')` in Notion-Submit-Erfolgspfad).
3. Opacity-Übergang drittens (`.notion-form { transition: opacity 0.15s ease; }` plus opacity-Toggle in `selectTarget` vor `loadSuggestions`).

**Apply-Reihenfolge in Sub-Batch B**:
1. Item 5 zuerst (aria-live in showToast — `role="status"` und `aria-live="polite"` als CSS-/JS-Attribut beim Toast-Render).
2. Item 6 zweitens (3 DE-Strings in `app_pkg/markdown.py:137/143/157` — `flash()`-Strings übersetzen ohne Logik-Änderung).

### 2. Master-Punch-Liste Items 7+8 außerhalb Sprint-Scope

**Diese zwei Items werden NICHT vom Sub-Thread gemacht** — sind Master-Browser-Smoke-Aufgaben:

- **Item 7: P8-Master-Smoke** (F-5-IMPL PDF-Gen-Error) — Master forciert PDF-Gen-Error via Theme-Datei-Manipulation oder Playwright-Stop, prüft ob flash-Banner statt 500-Page erscheint.
- **Item 8: F6-IMPL-Master-Smokes (P2 + P3 + P11)** — Master prüft:
  - P2: Card mit Content > 200 char → Copy-Btn → Editor-Paste → Full-Content sichtbar?
  - P3: DevTools Offline → Favorite-Toggle → Failure-Banner sichtbar?
  - P11: Card-Datum „Mär 2026" statt „Mar 2026"?

Sub-Thread berichtet im Phase-3-Bericht dass diese als **Master-Punch-Liste** verbleiben. Falls beim HYG2-Code-Reading offensichtlich ein Bug auffällt der Items 7-8 vorab klärt: kurz im Bericht erwähnen, **nicht** silent-fixen (Memory `feedback_no_silent_fixes.md`).

### 3. Item-5 (aria-live im showToast) ist cross-feature

`showToast` aus `_utils.js` wird von mehreren Features genutzt (audio_converter.js, markdown_converter.js, library.js, library_detail.js, podcasts in audio_converter.js, document_converter.js). Sub-Thread:
- Inspiziert via `grep -rn "showToast" static/js/` alle Call-Sites.
- Update am Helper-Render-Path in `_utils.js` (DOM-Element bekommt `role="status"` + `aria-live="polite"`).
- **Kein Call-Site-Touch nötig** weil der Helper-Render-Path zentralisiert ist — Sub-Thread verifiziert das per Code-Reading.
- Bei Toast-Level-Differenzierung (`danger`/`warning`/`info`/`success`): `role="alert"` für `danger`/`warning` (assertive) und `role="status"` für `info`/`success` (polite) — analog F-3.3 P8 Toast-Level-Differenzierung. Sub-Thread entscheidet beim Apply.

### 4. Item-1 (getUserMedia) Refactor-Risiko + Live-Mic-Smoke

Aktuelle Mechanik in audio_converter.js Live-Tab: WS-Handshake startet, dann `socket.onopen` ruft `getUserMedia` → Permission-Prompt erst nach Handshake → wenn User Denial klickt, WS bleibt offen ohne Mic-Stream → kaputter State.

**Master-Wahl-Default**: Permission-Prompt **vor** WS-Handshake. `getUserMedia` zuerst, dann bei Erfolg WS-Handshake, dann bei Stream-Track-End WS-Close. Bei Denial: kein WS-Handshake, sondern direkt Failure-Banner mit DE-Microcopy „Mikrofon-Zugriff verweigert. Bitte in Browser-Einstellungen erlauben."

**🔥 Live-Mic-Master-Smoke-Pflicht** vor Sprint-Sign-off:
- Master öffnet Live-Tab auf converter.smallpieces.de oder localhost:5656.
- Sequenz 1: erstes Mal Mic-Zugriff anfragen → Browser-Permission-Prompt sichtbar → erlauben → WS-Handshake erfolgreich → Transkription läuft.
- Sequenz 2: in Browser-Settings Mic-Zugriff blocken → Live-Tab öffnen → erwartet: Failure-Banner sichtbar, kein WS-Handshake.
- Sequenz 3: Mic-Track manuell beenden (z.B. Browser-Tab-Schließung des Mic-Indikators) → erwartet: WS-Close, Live-Tab zeigt Stopped-State.

Bei einer Sequenz divergierend: **STOP**, nicht silent-fixen, Master entscheidet.

### 5. Tests

Erwartete Test-Coverage-Anpassung minimal:
- Item 6 DE-Microcopy: optional 1 Test in `tests/test_markdown.py` der die DE-Flash-Strings asserted (analog F-3-IMPL Pattern für DE-Konventions-Tests). Sub-Thread entscheidet ob lohnend — die Strings sind statisch im Code, Test-Wert ist gering.
- Items 1-5: keine Suite-Tests sinnvoll (mocked-SDK-Boundary, Frontend-only-Touches).
- **Final-Erwartung**: 71/71 unverändert oder 72/72 wenn Item-6-Test angelegt.

### 6. BACKLOG-Update nach Sprint-Close

Sub-Thread aktualisiert BACKLOG.md am Sprint-Ende:
- **Sechs Sprint-Items entfernen** aus „P3 — Aktive Reminder" (Items 1-6).
- **Zwei Master-Smoke-Items bleiben** in „P3 — Aktive Reminder" (Items 7-8) — sind Master-Aufgaben außerhalb Sprint-Scope.
- Sprint-Eintrag oben in „Erledigt (rolling)" mit den 6 abgearbeiteten Items + 3-Sub-Batch-Commits.

### 7. Memory-Disposition

Erwartung: keine neue Eintrag. Falls beim Item-1-Apply eine Permission-Prompt-Mechanik-Lehre auftaucht die übertragbar ist (z.B. „Browser-Permission-Prompts immer vor async-State-Init"): defensive `feedback_*.md`.

---

## Phase 1 — Implementation (drei Sub-Batches mit STOP-Punkten)

### Pre-Flight (vor Sub-Batch A)

1. `pytest tests/` im Container — muss **71/71 grün** sein. (Container-side per `reference_converter_dep_bump_constraints.md`.)
2. `git status -s` → clean tree erwartet.
3. **BACKLOG.md** Aktive-Reminder-Sektion kurz überfliegen.
4. **`_utils.js`-Helper-Bestand verifizieren**: insbesondere ob `escapeHtml`-Helper schon existiert (für BT7). Wenn nicht: inline-Escape im library_detail.js mit Begründung.

---

### Sub-Batch A — Library-Detail-Polish (Items 2, 3, 4)

**Mechanik (Holistic-Apply empfohlen — alle drei Touches in library_detail.js)**:

1. **BT7 textarea-escape** in `renderNotionFields`:
   - Code-Anker (Pre-F3-IMPL-Linien): [static/js/library_detail.js:134-136](static/js/library_detail.js#L134-L136). Funktion `renderNotionFields` ist der richtige Anker (Zeilen verschoben durch F3-IMPL).
   - User-Input in Textarea wird beim Re-Render in HTML-String konkateniert — HTML-Escape via inline-Helper oder bestehendem `_utils.js`-Helper (wenn vorhanden).
   - Test-Vehikel: Card mit Notion-Form öffnen, Textarea mit `</textarea><script>alert(1)</script>` füllen, Re-Render triggern (z.B. Target-Switch), verifiziere escape-Render.

2. **BT8 window.open-noopener** im Notion-Submit-Erfolgspfad:
   - Code-Anker: [static/js/library_detail.js:172](static/js/library_detail.js#L172). Funktion `sendToNotion` Erfolgspfad-Branch ist der richtige Anker.
   - `window.open(notionPageUrl)` → `window.open(notionPageUrl, '_blank', 'noopener,noreferrer')`.

3. **Opacity-Übergang 150ms beim Notion-Target-Switch**:
   - `static/css/style.css`: `.notion-form-pane { transition: opacity 0.15s ease; }` oder analoge Klasse.
   - `static/js/library_detail.js` `selectTarget`-Funktion: vor `loadSuggestions` Aufruf `pane.style.opacity = '0.5'`; nach Render `pane.style.opacity = '1'`. Oder via CSS-Klassen-Toggle.

4. Tests: keine.
5. `pytest tests/` muss grün bleiben (71/71).

**Live-Smoke nach Sub-Batch A** (Master-Smoke optional, code-evident reicht):

- BT7: Textarea mit HTML-Inject → Re-Render → kein Inject-Effekt.
- BT8: Notion-Submit-Erfolg → neuer Tab öffnet sich mit `noopener` (DevTools-Inspect der `<a>`/`window.open`-Call).
- Opacity-Übergang: Notion-Target-Switch zeigt 150ms fade.

**STOP nach Sub-Batch A** — Bericht: 3 Items durch, Code-Anker aktualisiert (nach F3-IMPL-Linien-Shift), Test-Stand, `escapeHtml`-Helper-Disposition (existierend, neu lokal, neu in _utils.js).

---

### Sub-Batch B — Cross-Feature-Hygiene (Items 5, 6)

**Mechanik**:

1. **Item 5 (aria-live im showToast-Helper)**:
   - `static/js/_utils.js` `showToast`-Funktion: Toast-DOM-Element bekommt `role="status"` + `aria-live="polite"` (default für info/success) oder `role="alert"` + `aria-live="assertive"` für danger/warning. Master-Wahl: differenziert nach Toast-Level.
   - Cross-Feature-Verifikation: `grep -rn "showToast" static/js/` zeigt alle Call-Sites; Sub-Thread bestätigt dass kein Call-Site eine eigene Render-Logik bypassed.

2. **Item 6 (3 EN-Strings in `app_pkg/markdown.py:137/143/157`)**:
   - Zeile 137 „Error: No Markdown content provided…" → DE-Microcopy z.B. „Fehler: Kein Markdown-Inhalt angegeben."
   - Zeile 143 „Error: Invalid filename provided." → DE z.B. „Fehler: Ungültiger Dateiname."
   - Zeile 157 „Warning: Style not found…" → DE z.B. „Hinweis: Theme nicht gefunden — Standard wird verwendet."
   - Sub-Thread formuliert konkret nach Microcopy-Regeln (≤2 Sätze, kein Emoji).

3. Tests: optional 1 für Item 6 (DE-Flash-Strings-Assertion). Sub-Thread entscheidet ob lohnend.
4. `pytest tests/` muss grün bleiben (71-72/71-72 erwartet).

**Live-Smoke nach Sub-Batch B** (Master-Smoke optional):

- aria-live: DevTools-Inspect zeigt `role="status"`/`role="alert"` + `aria-live`-Attribut auf Toast-DOM.
- DE-Microcopy: Markdown-Submit mit Empty-Content / Invalid-Filename / Unknown-Theme → DE-Flash-Banner.

**STOP nach Sub-Batch B** — Bericht: 2 Items durch, showToast-Cross-Feature-Verifikation, Test-Stand, Toast-Level-Differenzierungs-Disposition (alle polite oder mit assertive für danger).

---

### Sub-Batch C — Audio-UX (Item 1)

**Patterns**: Item 1 (getUserMedia-in-socket.onopen-Bug, S, 🔥 Live-Mic-Master-Smoke).

**Pre-Flight für Sub-Batch C**:

Sub-Thread liest `static/js/audio_converter.js` Live-Tab-Sektion komplett (WS-Handshake, `getUserMedia`, MediaRecorder, Stream-End-Handlers). Verstehe aktuelle Mechanik bevor Apply.

**Mechanik**:

1. **`getUserMedia` vor WS-Handshake**:
   - Aktuelle Reihenfolge: WS-Open → `socket.onopen` → `navigator.mediaDevices.getUserMedia` → Permission-Prompt.
   - Neue Reihenfolge: `navigator.mediaDevices.getUserMedia` (zuerst) → bei Erfolg WS-Handshake → `MediaRecorder` → Streaming.
   - Bei `getUserMedia`-Denial: kein WS-Handshake, sondern `showAlert` mit DE-Microcopy „Mikrofon-Zugriff verweigert. Bitte in den Browser-Einstellungen erlauben."
   - Bei `getUserMedia`-Other-Error: Failure-Banner mit Error-Detail-Suffix.
2. **Stream-Track-End-Handling**: wenn User mid-Recording Mic blockiert oder Tab-Mic-Indikator schließt → WS-Close + UI-Reset.

3. **🔥 Live-Mic-Master-Smoke-Pflicht** (Sub-Thread berichtet als Master-Aufgabe vor Sprint-Sign-off — Sub-Thread kann **nicht selbst** smoke-en):
   - Sequenz 1: erstes Mal Mic-Anfrage → Browser-Permission-Prompt → erlauben → WS-Handshake-Erfolg → Transkription läuft.
   - Sequenz 2: in Browser-Settings Mic-blockiert → Live-Tab öffnen → Failure-Banner.
   - Sequenz 3: Mic-Track mid-Recording beenden → WS-Close + UI-Reset.

4. Tests: keine (mocked-SDK-Boundary).
5. `pytest tests/` muss grün bleiben (71-72).

**STOP nach Sub-Batch C** — Bericht: 1 Item durch, Code-Anker für getUserMedia-Refactor, Test-Stand, Live-Mic-Master-Smoke-Punch-Liste an Master.

---

## Phase 2 — Verify (gesamter Sprint)

1. `pytest tests/` im Container final grün (**71-73 erwartet**).
2. `grep -c "alert(" static/js/library_detail.js` und `grep -c "alert(" static/js/library.js` → 0 (sollten schon 0 sein aus F-3/F-6).
3. `grep "showToast" static/js/_utils.js` zeigt `role="status"`+`aria-live` im Helper.
4. `grep -n "Error: No Markdown\|Error: Invalid filename\|Warning: Style not found" app_pkg/markdown.py` → 0 (alle drei EN-Strings raus).
5. **End-to-End-Smoke** (Master-Pflicht für 🔥 Live-Mic + Code-evident für Rest):
   - **Item 1 (Master-Live-Mic-Smoke)**: drei Sequenzen aus Master-Annotation 4.
   - **Items 2-6**: Code-Reading + Container-Smoke + Browser-DevTools-Inspect für aria-live + DE-Microcopy.
6. DevTools-Console final clean.
7. Sub-Batches A/B/C sind alle drei in `git diff` reflektiert.

Nach Phase 2: STOP — Bericht. Final-Test-Anzahl, Master-Punch-Liste (Item 1 Live-Mic-Smoke + Items 7-8 außerhalb Sprint-Scope).

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- **Default: drei Commits, einer pro Sub-Batch** (analog F-3-IMPL / F-6-IMPL). Subjects z.B. „HYG2 Sub-Batch A: library_detail polish (BT7, BT8, opacity)" / „HYG2 Sub-Batch B: cross-feature hygiene (showToast aria-live, markdown DE-Microcopy)" / „HYG2 Sub-Batch C: audio Live-Tab getUserMedia refactor".
- Branch: direkt auf `main` ist OK.
- `git push origin main`. Wenn Auto-Mode-Classifier blockt **oder** `.git/objects/<hash>`-SMB-Permission blockt: Bericht, Master pusht von Hand via SSH zu Mintbox (siehe Memory `feedback_push_is_normal.md`).

---

## Stop-Regel

Nach **jeder Phase** UND **nach jedem Sub-Batch** Bericht an Master, nicht weiter bis Sign-off.

**Zusätzlich für HYG2**:
- Item 1 (getUserMedia) hat Refactor-Risiko — wenn beim Apply Live-Tab-Logik unklar wird: STOP, Master fragen.
- Bei `escapeHtml`-Helper-Existenz-Frage (BT7): wenn `_utils.js` keinen Helper hat und Sub-Thread inline-Escape ohne Begründung wählt vs. neuen Helper anlegen — Memory `feedback_helper_reuse_design_choice.md` greift: keine künstliche Drift, inline-Escape OK wenn single-call-site.
- Bei Master-Punch-Liste-Items (7+8): nicht selbst smoke-en, im Phase-3-Bericht als Master-Aufgabe markieren.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — 6 Items in einem Sprint, 3 Sub-Batches, 1 🔥 Live-Mic-Master-Smoke-Pflicht, evtl. 1 neuer Test, mehrere Code-Bereiche (`static/js/library_detail.js`, `static/js/_utils.js`, `static/js/audio_converter.js`, `static/css/style.css`, `app_pkg/markdown.py`, optional `tests/test_markdown.py`). XS-lastig (XS: 5, S: 1) — kleiner als die UX-Cascade-Implementation-Sprints F-5-IMPL und F-6-IMPL.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Code-Reading von `library_detail.js` (BT7-Anker) Auffälligkeiten in den anderen F-3-Helpers auffallen die F3-IMPL nicht abgedeckt hat: kurz im Bericht aufzählen, **nicht** in den Sprint-Diff.
- Wenn beim `escapeHtml`-Disposition (BT7) ein **echter** Helper-Vorschlag mit zweiter Call-Site-Begründung aufkommt: in Helper-Vorschlags-Sektion am Doc-Ende.
- Wenn beim `showToast`-aria-live-Touch (Item 5) ein Call-Site Render-Logik bypassed: kurz im Bericht aufzählen — könnte F-N-Pattern für Folge-Welle sein.
- Wenn beim getUserMedia-Apply (Item 1) WS-Library-Mechanik unklar wird (z.B. `Socket.io` vs. native WebSocket): kurz im Bericht aufzählen — kein silent-Apply ohne Master-Sign-off.
- Wenn beim DE-Microcopy-Apply (Item 6) weitere EN-Strings auffallen die nicht im BACKLOG dokumentiert sind: als „aufgefallen, nicht gefixt" in den Bericht — Memory `feedback_no_silent_fixes.md`.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „HYG2 ☑ done 2026-05-XX → commits `<hash-A>` (Sub-Batch A library_detail polish), `<hash-B>` (Sub-Batch B cross-feature hygiene), `<hash-C>` (Sub-Batch C audio Live-Tab refactor). Pytest <neue Anzahl>/<neue Anzahl> grün. **6 Aktive-Reminder-Items abgearbeitet** (BT7, BT8, Opacity-Übergang, aria-live showToast, DE-Microcopy markdown.py, getUserMedia-Refactor). Master-Punch-Liste: Item 1 Live-Mic-Smoke (HYG2), Items 7-8 (P8-F5-IMPL und F6-IMPL P2/P3/P11) bleiben Browser-Master-Aufgaben."
- **BACKLOG.md**: 6 Items aus „P3 — Aktive Reminder" entfernen (Items 1-6 dieser Sprint-Liste). Items 7-8 (Master-Smokes) **behalten**. HYG2-Eintrag oben in „Erledigt (rolling)" mit den 6 abgearbeiteten Items + 3-Sub-Batch-Commits.
- **Memory**: nichts erwartet. Falls Permission-Prompt-Mechanik-Lehre aus Item 1 übertragbar: defensive `feedback_*.md`.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — alle sechs Sprint-Items sind klar spec'd, Master-Annotationen verankern Sub-Batch-Strategie + Apply-Reihenfolge + Live-Mic-Smoke-Pflicht für Item 1 + cross-feature-Disziplin für Item 5.)_
