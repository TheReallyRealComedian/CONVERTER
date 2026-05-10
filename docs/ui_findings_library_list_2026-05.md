# UX-Heuristik-Findings: library List-View (2026-05-10)

**Methodik:** Stufe 2 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Heuristisches Review der strukturierten Inventur aus Stufe 1.
**Quelle:** [docs/ui_inventory_library_list_2026-05.md](ui_inventory_library_list_2026-05.md) — 21 echte interaktive Elemente in 30 Tabellenzeilen, 16 nummerierte Befunde, F-3-Korrespondenz-Übersicht am Doc-Ende.
**Heuristiken:** Nielsen H1 (Sichtbarkeit des Systemzustands), H4 (Konsistenz und Standards), H6 (Wiedererkennen statt Erinnern), H9 (Fehlermeldungen / Hilfe bei Fehlern).
**Produkt-Kontext:** Single-User (Oliver), LAN-only, login-protected. **`library` List-View ist zentraler Hotspot des Reader-Ersatzes für Readwise-Replacement** (laut [memory project_readwise_replacement.md](file:///Users/olivergluth/.claude/projects/-Volumes-MintHome-CODE-CONVERTER/memory/project_readwise_replacement.md)). Jede Konvertierung wird von hier aus gestartet bzw. wieder gefunden — **Daily-Usage-Schmerz hoch** für List-Reibung (Filter-Active-Sichtbarkeit, Suche-Live-vs-Submit-Inkonsistenz, Tag-Click-Reibung, Sortier-Active-Marker, Copy-Btn-Daten-Verlust). Severity-Bewertung wichtet den täglichen Treffer-Häufigkeits-Faktor mit (siehe Sprint-Prompt Master-Annotation 1).

**Geschwister-Feature-Hebel (Cross-Feature-Inversion zu F-5.2's Schwester-Feature-Hebel):** `library` List-View ist **Geschwister-Feature zu `library_detail`** (F-3) — selbe `ConversionHistory`-Datenklasse, andere View-Klasse (List statt Detail). F-6.1-Korrespondenz-Übersicht zeigt 6 von 15 F-3-Patterns als direkt anwendbar (P1, P3, P6, P8, P14, P15), 2 als teil-übertragbar (P2, P5), 3 als bereits-erfüllt (P9, P11, P12) und 4 als nicht-anwendbar (P4, P7, P10, P13). **F-3.2 ist primäre Heuristik-Klassifikations-Quelle** für Findings mit F-3-Korrespondenz; Heuristik+Severity werden 1:1 übernommen, **außer** Daily-Usage-Schmerz-Gewichtung der List-View-als-Reader-Hotspot schiebt (siehe Sprint-Prompt Master-Annotation 1). Die in F-6.1 dokumentierte Pattern-Konvergenz von 53% (8 von 15 als direkt+teil) übersetzt sich erwartungsgemäß auf eine Cross-Feature-H4-Finding-Quote im Bereich 35-50% (Pattern-zu-Finding-Übertragung hat Verluste durch list-spezifische Differenzen wie URL-Persistierung statt Helper-Reuse oder list-spezifische State-Klassen Sortierung/Filter/Suche/Pagination/Empty).

**Live-Walkthrough-Hinweis:** F-6.1 ist Code-only-Inventur mit 4 dokumentierten Code↔live-Divergenz-Verdachten und 9 Live-Walkthrough-Lücken. Findings, deren visueller Effekt aus reinem Code-Reading nicht endgültig beurteilbar ist, sind in der Severity-Spalte mit `⚠️ code-only` gekennzeichnet — Master entscheidet vor F6-PATTERNS, ob Walkthrough nachgereicht oder als 🔥 Smoke-Pflicht in F6-IMPL markiert wird (analog F-3-IMPL- und F-5-IMPL-Methodik, siehe Sprint-Prompt Master-Annotation 6).

---

## Findings (sortiert absteigend nach Schweregrad)

| #   | Element / Befund | Problem (1–2 Sätze) | Heuristik | Severity | Inventur-Anker | F-3-Korrespondenz | Disposition |
|-----|------------------|---------------------|-----------|----------|----------------|-------------------|-------------|
| F1  | Copy-Btn kopiert nur 200-char-Preview statt Full-Content (Befund 1; Inventur #23) | `copyContent(id)` liest `card.querySelector('.line-clamp-3').textContent` ([static/js/library.js:19](../static/js/library.js#L19)) — exakt der Server-clipped Inhalt aus `{{ conv.content[:200] }}...` ([templates/library.html:59](../templates/library.html#L59)), nicht der DB-Full-Content. User-Erwartung "Copy" = Full-Content; Card-Toolbar suggeriert keine Preview-Only-Semantik, der Erfolgs-Toast "Content copied" verstärkt die falsche Erwartung. **Daten-Verlust-light**: User merkt es erst beim Paste in Notion/Editor. Daily-Usage-Schmerz hoch (Reader-Replacement-Workflow-Kette). | H9 | **3** | Befund 1 | — (list-spezifisch) | Finding + Bug-Ticket BT3 |
| F2  | `toggleFavorite` silent-fail (Befund 2; Inventur #19) | `toggleFavorite(id, btn)` ([static/js/library.js:3-15](../static/js/library.js#L3-L15)) togglet Klasse + Glyph nur bei `r.ok`, hat **kein** `.catch`, **kein** `if (!r.ok)`-Branch, **kein** `safeJSON`-Wrap. Bei 4xx/5xx oder Network-Fail bleibt der Glyph alt — kein Toast, kein Banner. Bei Session-Expired (Login-Redirect-HTML als 200) würde `r.ok===true` den Glyph fälschlich togglen. Identische Mechanik wie `library_detail.js` aus F-3, identische Lücken. | H1 | **3** ⚠️ code-only | Befund 2 | F-3 P1 + P14 (direkt) | Finding + Bug-Ticket BT1 |
| F3  | `toggleFavorite` Recovery-Anleitung fehlt (siehe F2) | Da der Fehler komplett unkommuniziert ist, fehlt jede Recovery-Anleitung ("erneut versuchen" / "Verbindung prüfen"). User merkt erst beim nächsten Reload, dass der Favorite-Status nicht persistiert wurde. Identisch zu F-3.2 F2 — H9-Reihe der silent-fail-Familie. | H9 | **3** ⚠️ code-only | Befund 2 | F-3 P1 + P14 (direkt) | Finding + Bug-Ticket BT1 |
| F4  | `deleteConversion` silent-fail (Befund 3; Inventur #24) | `deleteConversion(id, btn)` ([static/js/library.js:27-38](../static/js/library.js#L27-L38)) führt `DELETE` aus und animiert die Card-Entfernung nur bei `r.ok` — **kein** else-Zweig, **kein** `.catch`, **kein** `safeJSON`. Bei 4xx/5xx bleibt Card sichtbar ohne Hinweis warum. Bei Session-Expired (Login-Redirect-HTML als 200) würde Card aus dem DOM entfernt **ohne DB-Delete** (visuelle Lüge). Identische Familie wie F-3.2 F4. | H1 | **3** ⚠️ code-only | Befund 3 | F-3 P3 + P14 (direkt) | Finding + Bug-Ticket BT2 |
| F5  | `deleteConversion` Recovery-Anleitung fehlt (siehe F4) | Keine Recovery-Anleitung, kein Hinweis auf Backend-Down / Session-Expired / bereits-gelöscht. Identisch zu F-3.2 F5 — H9-Reihe der Delete-silent-fail-Familie. | H9 | **3** ⚠️ code-only | Befund 3 | F-3 P3 + P14 (direkt) | Finding + Bug-Ticket BT2 |
| F6  | Englische Strings flächendeckend in List-View (Befund 6; Inventur #11–#15, #18, #19, #21, #22, #23, #24, #25–#28, #30) | Sammel-Befund: ~18 EN-Strings im Template + JS — Filter-Optionen/Placeholder/Labels/Buttons/Tooltips/confirm-Text/Toast-Texte/Pagination/Empty-State. **F-1 Cluster Polish-1 hat document_converter auf DE umgestellt; F-2 hat es als Sev-3-Cross-Feature-Bruch dokumentiert; F-3.2 F7 hat es für `library_detail` als H4 Sev 3 katalogisiert.** List-View ist auf Pre-DE-Pass-Stand, **Type-Badge-Texte überlappen 1:1 mit F-3 #12** (einmal fixen genügt). Daily-Usage-Schmerz hoch durch Reader-Replacement-Hotspot. | H4 | **3** | Befund 6 (Sammel) | F-3 P6 (direkt) | nur Finding |
| F7  | Toast-Level für `copyContent`-Failure-Pfad ist `success` (Befund 4; Inventur #23/#29) | `showToast('Copy failed')` ohne `opts.level='danger'` ([static/js/library.js:23](../static/js/library.js#L23)) → grüner Erfolgs-Toast für Fehler. Visuell-vs-textlich widersprüchlich: User sieht grünen Toast mit "Copy failed". Identisches Issue wie F-3.2 F10, identischer Fix-Pfad (Toast-Level pro Call-Site). | H4 | **2** ⚠️ code-only | Befund 4 | F-3 P8 (direkt) | Finding + Bug-Ticket BT4 |
| F8  | Empty-State nicht filter-aware (Befund 12; Inventur #28) | [templates/library.html:106-110](../templates/library.html#L106-L110) zeigt "No saved conversions yet" + "Use Save to Library on any converter page to start building your library." auch wenn der User mit aktivem Filter ("Audio" + "Favorites") keine Treffer hat — der User **hat** Conversions, nur keine im aktuellen Filter. Empty-State täuscht "library leer" statt "Filter zeigt nichts" — Recovery-Pfad (Filter zurücksetzen) fehlt komplett. Daily-Usage-Schmerz mittel (Filter-Anwendung ist häufiger Reader-Workflow). | H9 | **2** | Befund 12 | — (list-spezifisch) | nur Finding |
| F9  | Search-Input nicht auto-submit (UX-Inkonsistenz) (Befund 8; Inventur #12/#15) | [templates/library.html:19-22](../templates/library.html#L19-L22) — Search-Input (#12) hat **kein** `onchange="this.form.submit()"`, **alle anderen Filter** (#11 Type-Select, #13 Favorites-Checkbox, #14 Sort-Select) sind auto-submit. User-Mental-Model "Filter ändern = Liste filtert sich" kollidiert für den Search-Input — entdeckbar via "Search"-Submit-Btn (#15) oder Enter-Key, aber inkonsistent. Internal H4-Bruch innerhalb der Filter-Bar (analog F-5.2 F8 Two-Dark-Modes). | H4 | **2** ⚠️ code-only | Befund 8 | — (list-spezifisch) | nur Finding |
| F10 | Banner-Mountpoint im Template fehlt — Vorbedingung für `showAlert`-Calls (Befund 5; Inventur Helper-Reuse) | [templates/library.html](../templates/library.html) hat **keinen** `#library-alert-container` o.ä. — F2/F3/F4/F5 können `showAlert` nicht aufrufen, ohne erst einen Mountpoint im Template einzuführen. **Strukturelle Vorbedingung** für die silent-fail-Cluster-Patches (BT1, BT2). Identisch zu F-3 P15 (Detail-View brauchte denselben Mountpoint, F-3-IMPL hat ihn eingeführt). | H4 | **1** | Befund 5 | F-3 P15 (direkt) | nur Finding (Vorbedingung-Verzahnung mit BT1+BT2 in F6-IMPL) |
| F11 | Search-Input ohne Live-Search / Debouncing (Befund 7; Inventur #12) | [templates/library.html:19-22](../templates/library.html#L19-L22) — Submit nur via Button oder Enter-Key. Bei Pagination-Cap 20 wäre Live-Search performant erreichbar; Reader-Replacement-Daily-Usage-Pfad würde davon profitieren. Polish-Aspekt; H6 weil Recognition-over-Recall (User sieht beim Tippen nicht sofort, ob die Query Treffer liefert) und nicht primär H1 (System-Status-Pfad ist explizit-submit, nicht missing). | H6 | **1** ⚠️ code-only | Befund 7 | — (list-spezifisch) | nur Finding |
| F12 | Pagination-URL mit leerem `favorites=''`-String-Artifact (Befund 9; Inventur #25–#27) | [templates/library.html:87,92,100](../templates/library.html#L87) — `favorites='1' if current_favorites else ''` produziert bei deaktivierten Favorites einen leeren Query-Param `?...&favorites=&...` in der URL. Funktional unkritisch (`request.args.get('favorites', '') == '1'` → False), URL ist hässlich/uneinheitlich; vom User in der Browser-Adressleiste sichtbar und beim URL-Sharing irritierend. **Disposition-Abweichung von F-3.2-BT7/BT8-Mechanik**: bleibt als Finding (statt Bug-only), weil URL-Output **vom User wahrnehmbar** ist (Adress-Leiste, Bookmark-Sharing) — H4-Konsistenzaspekt im Output. | H4 | **1** | Befund 9 | — (list-spezifisch) | nur Finding |
| F13 | Type-Filter Backend-Validation fehlt (Befund 10; Inventur Backend) | [app_pkg/library.py:43-44](../app_pkg/library.py#L43-L44) — `query.filter_by(conversion_type=conversion_type)` akzeptiert jeden String aus dem URL-Query; bei `?type=nonsense` → leere Liste statt 400 oder Fallback auf "All". `ALLOWED_CONVERSION_TYPES`-Set ist nur im POST-Pfad genutzt ([:91](../app_pkg/library.py#L91)). Kein SQL-Injection-Risk dank SQLAlchemy-Parametrisierung; reiner UX-Aspekt (User landet auf Empty-State ohne Hinweis warum). H9 weil Recovery-Pfad bei Tippfehler im URL-Query fehlt (nicht H1, weil System-Status sichtbar ist — leere Liste). | H9 | **1** | Befund 10 | — (list-spezifisch) | nur Finding |
| F14 | Card-Datum `%b`-Monatskürzel Locale-abhängig (Befund 11; Inventur #21) | [templates/library.html:63](../templates/library.html#L63) — `conv.created_at.strftime('%d %b %Y, %H:%M')` liefert bei Docker-Container-Default-Locale `C` oder `C.UTF-8` englische Monatskürzel ("May" statt "Mai"). Container-Image enthält wahrscheinlich keine `de_DE`-Locale-Bundles. **Andere Mechanik als F-3 P5** (P5 war JS-`<input type="datetime-local">`-Pre-Population — hier Server-side `strftime` im Card-Render-Pfad), aber **selbe Klasse von Lokalisierungs-Problem**. Cross-Feature-H4 mit P6 DE-Microcopy-Welle (DE-Monatsnamen-Map oder Locale-Setting). | H4 | **1** ⚠️ code-only | Befund 11 | F-3 P5 (teil) | nur Finding |
| F15 | Per-Page-Size 20 hardcoded (Befund 13; Inventur Backend) | [app_pkg/library.py:39](../app_pkg/library.py#L39) — `per_page=20` ohne UI-Toggle für "mehr pro Seite zeigen". Bei wachsender Library (Reader-Replacement-Skalierung Richtung mehrere hundert/tausend Einträge) wird das ein Pagination-Klick-Marathon. Recognition-over-Recall (User-Mental-Model "ich kann mehr sehen" wird nicht angeboten). Sev 1 weil Skalierungs-Welle (heute nicht akut), Disposition: Finding für künftige Reader-Ersatz-Skalierung. | H6 | **1** | Befund 13 | — (list-spezifisch) | nur Finding |
| F16 | Keine aria-live für Card-Remove (Befund 14; Inventur #24) | [static/js/library.js:30-36](../static/js/library.js#L30-L36) — Delete-Animation entfernt die Card visuell (`opacity=0` + scale-Transform + nach 200 ms `card.remove()`), aber Screenreader bekommt keinen Hinweis. F-3-Welle hat `#notion-target-status`-Region für Notion-Target-Switches eingeführt — selbes Pattern wäre für Card-Remove anwendbar (z.B. `#library-action-status`-Region mit "Eintrag gelöscht."-Microcopy). a11y-Befund mit Verzahnung zu F-3-P3-Konvergenz (Card-Remove ist Teil des Delete-Pfades). | H1 | **1** ⚠️ code-only | Befund 14 | — (list-spezifisch) | nur Finding |
| F17 | Card-Hover-Lift Animation kollidiert visuell mit Sub-Element-Klicks (Befund 15; Inventur #17/#19/#23/#24) | [static/css/style.css:242-245](../static/css/style.css#L242-L245) — `.c-card:hover { translateY(-2px) }` läuft während die Maus die Card überquert; sowohl Favorite-Btn (#19), Card-Link (#20) als auch Copy/Delete-Buttons (#23/#24) liegen innerhalb der hover-bewegten Card. Klick-Hit-Robustheit ist Browser-abhängig (Pointer-Capture macht das vermutlich unkritisch); ggf. visuelles Flackern bei schnellem Maus-Wechsel zwischen Cards. Recognition-over-Recall (Hover-Lift signalisiert Click-Affordance, gleichzeitig sind die Sub-Element-Buttons primary Targets — Affordance-Konflikt). Live-Walkthrough-Lücke. | H6 | **1** ⚠️ code-only | Befund 15 | — (list-spezifisch) | nur Finding |

---

## Reine Bug-Tickets (Implementations-Anker für Findings — separates Ticket-Material)

Vier Bug-Tickets, alle Finding-linked und in den Findings-Tabellen oben verlinkt. **Keine Pure-Bug-Tickets ohne H-Aspekt** in F-6.2 — analog F-4.2 (0 Pure-Bugs) und im Gegensatz zu F-3.2 (BT7/BT8 textarea-escape, window.open-noopener). Begründung: List-View hat keine vergleichbar User-unsichtbaren Defensive-Code-Lücken; alle gefundenen Code-Bugs haben einen User-wahrnehmbaren UX-Aspekt. **Kein Fix-Pfad-Vorschlag, keine konkrete Microcopy** — das macht F6-PATTERNS bzw. F6-IMPL.

- **BT1: `toggleFavorite` hat kein Error-Handling und keinen `safeJSON`-Wrap.** Kein `.catch`, kein `if (!r.ok)`-Branch, kein Login-Redirect-HTML-Detection. → siehe Findings F2, F3. Code-Anker: [static/js/library.js:3-15](../static/js/library.js#L3-L15). Reproduktion: Server bei `PUT /api/conversions/<id>` auf 5xx setzen → Frontend bleibt im alten State, kein Feedback. Session-Expired-Variante: Login-Redirect-HTML als `r.ok=true` erkannt → Glyph togglet ohne dass etwas gespeichert wurde. **Vorbedingung BT1**: F10 (Banner-Mountpoint) muss vorab im Template eingeführt werden (analog F-3-IMPL Sub-Batch A).

- **BT2: `deleteConversion` hat kein Error-Handling und keinen `safeJSON`-Wrap.** Kein else-Zweig nach `r.ok`-Check, kein Login-Redirect-HTML-Detection. → siehe Findings F4, F5. Code-Anker: [static/js/library.js:27-38](../static/js/library.js#L27-L38). Reproduktion: Server bei `DELETE /api/conversions/<id>` auf 5xx setzen → Card bleibt sichtbar, kein Hinweis. Session-Expired-Variante: Login-Redirect-HTML als `r.ok=true` erkannt → Card aus DOM entfernt **ohne DB-Delete** (visuelle Lüge — Reload zeigt die Conversion wieder). **Vorbedingung BT2**: F10 (Banner-Mountpoint) wie BT1. **Verzahnung mit F16**: aria-live-Region für Card-Remove kann gemeinsam mit BT2-Patches eingeführt werden (gleicher Code-Pfad).

- **BT3: Copy-Btn liest 200-char-Preview statt DB-Full-Content.** `copyContent(id)` liest `card.querySelector('.line-clamp-3').textContent` ([static/js/library.js:19](../static/js/library.js#L19)) — exakt der Server-clipped Inhalt aus `{{ conv.content[:200] }}...` ([templates/library.html:59](../templates/library.html#L59)). → siehe Finding F1. Reproduktion: lange Conversion (>200 chars Content) in der List-View, "Copy" klicken, in Editor pasten — nur die Preview + "..." kommt an. Erwartet wäre `card.dataset.content` oder ein Server-Side-Endpoint `GET /api/conversions/<id>/content` für Full-Content-Fetch (Disposition-Wahl in F6-PATTERNS). **Daten-Verlust-light**: User merkt es erst beim Paste — ohne Recovery-Hinweis.

- **BT4: Toast-Level wird im `copyContent`-Failure-Pfad nicht gesetzt.** Default `level='success'` greift bei `showToast('Copy failed')` ([static/js/library.js:23](../static/js/library.js#L23)). → siehe Finding F7. Identische Mechanik zu F-3.2 BT5 (Toast-Level pro Call-Site in `library_detail.js`). Reproduktion: Clipboard-API verweigern (z.B. via DevTools-Clipboard-Permission-Block oder bei Cross-Origin-Iframe) → grüner Toast mit "Copy failed". Vorgeschlagene Mechanik: pro Call-Site das passende Level setzen (`{ level: 'danger' }` für Fehler-Pfade) — F6-PATTERNS legt Microcopy fest.

---

## Disposition-Verteilung

- **Nur Findings (kommen in F6-PATTERNS):** 11 — F6, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17
- **Findings + Bug-Tickets (kommen in F6-PATTERNS **plus** separates Bug-Ticket):** 6 Findings → **4 unique Bug-Tickets** — F2+F3 (BT1), F4+F5 (BT2), F1 (BT3), F7 (BT4)
- **Nur Bug-Tickets (kommen **nicht** in F6-PATTERNS):** 0 — Disposition-Abweichung von F-3.2/F-5.2 explizit begründet im Bug-Tickets-Sektions-Header oben.

**Inventur-Befund-Coverage (alle 16 disponiert):**

| Inventur-Befund | Finding(s) | Bug-Ticket | Anmerkung |
|-----------------|-----------|------------|-----------|
| #1 Copy-Btn 200-char-Preview | F1 (H9 Sev 3) | BT3 | Master-Annotation 3 1:1 übernommen |
| #2 toggleFavorite silent-fail | F2 (H1 Sev 3) + F3 (H9 Sev 3) | BT1 | F-3.2 F1+F2+BT1 1:1 übernommen |
| #3 deleteConversion silent-fail | F4 (H1 Sev 3) + F5 (H9 Sev 3) | BT2 | F-3.2 F4+F5+BT2 1:1 übernommen |
| #4 Toast-Level Copy-Failure | F7 (H4 Sev 2) | BT4 | F-3.2 F10+BT5 1:1 übernommen |
| #5 Banner-Mountpoint fehlt | F10 (H4 Sev 1) | — | F-3 P15-Verzahnung als Vorbedingung für BT1+BT2 |
| #6 EN-Strings Sammel | F6 (H4 Sev 3) | — | F-3.2 F7 1:1 übernommen |
| #7 Search-no-live/Debouncing | F11 (H6 Sev 1) | — | list-spezifisch, Polish |
| #8 Search-not-auto-submit | F9 (H4 Sev 2) | — | list-spezifische internal H4-Inkonsistenz |
| #9 favorites=''-URL-Artifact | F12 (H4 Sev 1) | — | F-3.2-BT7/BT8-Mechanik geprüft, abgewichen mit Begründung (URL ist user-sichtbar) |
| #10 Type-Filter Backend-Validation | F13 (H9 Sev 1) | — | list-spezifisch |
| #11 %b-Locale Card-Datum | F14 (H4 Sev 1 ⚠️ code-only) | — | F-3 P5 teil-übertragbar |
| #12 Empty-State filter-aware | F8 (H9 Sev 2) | — | list-spezifisch, daily-usage-Schmerz-relevant |
| #13 Per-Page-Size 20 hardcoded | F15 (H6 Sev 1) | — | list-spezifisch, Skalierungs-UX |
| #14 aria-live Card-Remove | F16 (H1 Sev 1 ⚠️ code-only) | — | list-spezifisch, Verzahnung mit BT2 |
| #15 Card-Hover-Lift visuelle Kollision | F17 (H6 Sev 1 ⚠️ code-only) | — | list-spezifisch, ⚠️ Live-Walkthrough-Lücke |
| #16 Helper-Reuse-Beobachtung | — (in Cross-Feature-H4-Sektion absorbiert) | — | Master-Annotation 5: Helper-Reuse-Drift mit begründeter Design-Wahl ist keine H4-Verletzung |

**Abweichungen von F-6.1-Pre-Disposition (begründet):**

- **#9 favorites=''-URL-Artifact:** F-6.1 schlug "Finding (Code-Hygiene)" vor. F6-REVIEW prüft, ob analog F-3.2-BT7/BT8 als Bug-only umzudisponieren wäre. **Entscheidung: bleibt als Finding** (F12, H4 Sev 1), weil URL-Output **vom User wahrnehmbar** ist (Browser-Adressleiste, Bookmark-Sharing) — H4-Konsistenzaspekt im Output ist gegeben. Das unterscheidet sich von F-3.2 BT7 (textarea-escape im JS-DOM, User unsichtbar) und F-3.2 BT8 (window.open-noopener, User unsichtbar). **Kein Pure-Bug-Aspekt** — daher keine BT-Disposition.
- **#16 Helper-Reuse-Beobachtung:** F-6.1 markierte als Meta-Befund mit Disposition-Hinweis "Inventur-Meta-Sektion". Sprint-Prompt Master-Annotation 5 verlangt Absorption in Cross-Feature-H4-Sektion (analog F-3.2-Befund-18-Mechanik). **Übernommen — kein Heuristik-Finding**, sondern Helper-Reuse-Reflexion in Cross-Feature-H4-Sektion (siehe unten).
- **Severity-Übernahme von F-3.2:** F2/F3 (Sev 3), F4/F5 (Sev 3), F6 (Sev 3 EN-Strings), F7 (Sev 2 Toast-Level) sind **identisch** zu F-3.2 F1+F2/F4+F5/F7/F10. **Keine Daily-Usage-Schmerz-Schiebung** — Sub-Thread hat geprüft, dass List-View-Daily-Usage-Schmerz und Detail-View-Daily-Usage-Schmerz für diese Findings vergleichbar liegen (beide Reader-Replacement-Hotspots; List-View ist **etwas** häufiger als Trigger-Pfad, aber die F-3.2-Sev-3-Wertung deckt den Daily-Usage-Aspekt schon ab — kein Hochstufen auf Sev 4 begründbar, weil keine Datenverlust-/Blockade-Pfade auf der primären Lese-Aufgabe).
- **F1 (Copy-200char) Severity-Wahl:** Master-Annotation-3-Vorschlag H9 Sev 3 1:1 übernommen. Sub-Thread hat Sev 4 geprüft (Daten-Verlust-Pfad?), aber abgelehnt, weil (a) der "verlorene" Content noch in der DB ist und über Detail-View / Re-Copy aus Full-Content-Endpoint erreichbar wäre, (b) der User merkt es beim Paste sofort, also kein "stille Datenkorruption". Sev 3 (regelmäßig spürbar, frustrierend) ist die richtige Stufe — Daily-Usage-Schmerz hoch wegen Workflow-Ketten.

---

## Cross-Feature-H4-Sektion (Geschwister-Feature-Konvergenz zu F-3 library_detail)

**Pattern-Konvergenz-Quote (F-6.1 Korrespondenz-Übersicht): 53%** (8 von 15 F-3-Patterns als direkt+teil; 11 von 15 als "relevant inkl. bereits-erfüllt").
**Cross-Feature-H4-Finding-Quote (F-6.2): ~47%** (8 von 17 Findings haben F-3-Korrespondenz: F2, F3, F4, F5, F6, F7, F10, F14). Im erwarteten Master-Bereich 35-50% — höher als F-3.2's 35% (`library_detail` zu nicht-Geschwistern), höher als F-2.2's 41% und F-4.2's 0%, niedriger als F-5.2's 86% (Schwester-Feature mit Helper-Bestand). **Begründung der Position im Erwartungsband:** List-View und Detail-View teilen `ConversionHistory`-Datenklasse und Helper-Set, aber List-View-State-Klassen (Sortierung/Filter/Suche/Pagination/Empty) haben keinen Detail-View-Korrespondenten — daher Pattern-Konvergenz ~50%, nicht ~80% wie bei F-5.2-Konverter-Schwesterpaaren.

### Direkt übertragbare F-3-Patterns (mit library-List-View-Findings)

| F-3-Pattern | F-3.2-Finding-Quelle | List-View-Finding | Heuristik | Severity | Code-Anker |
|-------------|---------------------|-------------------|-----------|----------|------------|
| **P1 — Auto-Save silent-fail / Favorite-silent-fail** | F-3.2 F1 (H1 Sev 3) | F2 (1:1 übernommen) | H1 | 3 ⚠️ code-only | [static/js/library.js:3-15](../static/js/library.js#L3-L15) `toggleFavorite` |
| **P1 — Auto-Save Recovery-Anleitung** | F-3.2 F2 (H9 Sev 3) | F3 (1:1 übernommen) | H9 | 3 ⚠️ code-only | [static/js/library.js:3-15](../static/js/library.js#L3-L15) |
| **P3 — Delete silent-fail** | F-3.2 F4 (H1 Sev 3) | F4 (1:1 übernommen) | H1 | 3 ⚠️ code-only | [static/js/library.js:27-38](../static/js/library.js#L27-L38) `deleteConversion` |
| **P3 — Delete Recovery-Anleitung** | F-3.2 F5 (H9 Sev 3) | F5 (1:1 übernommen) | H9 | 3 ⚠️ code-only | [static/js/library.js:27-38](../static/js/library.js#L27-L38) |
| **P6 — DE-Microcopy flächendeckend** | F-3.2 F7 (H4 Sev 3) | F6 (1:1 übernommen, Type-Badge-Texte überlappen 1:1 mit F-3 #12 — einmal fixen) | H4 | 3 | [templates/library.html](../templates/library.html), [static/js/library.js:21,23,28](../static/js/library.js#L21) |
| **P8 — Toast-Level pro Call-Site** | F-3.2 F10 (H4 Sev 2) | F7 (1:1 übernommen) | H4 | 2 ⚠️ code-only | [static/js/library.js:23](../static/js/library.js#L23) |
| **P14 — `safeJSON` Login-Redirect-Detection** | F-3.2 BT1+BT2 (in Findings F1/F2/F4/F5 Helper-Reuse-Anker) | F2/F3/F4/F5 (mit-Patch in BT1+BT2) | H1+H9 | 3 ⚠️ code-only | [static/js/library.js:3-15](../static/js/library.js#L3-L15), [static/js/library.js:27-38](../static/js/library.js#L27-L38) |
| **P15 — Banner-Mountpoint-Container im Template** | F-3.2-impliziter struktureller Vorbedingung-Fix (in F-3-IMPL umgesetzt) | F10 (Vorbedingung für BT1+BT2) | H4 | 1 | [templates/library.html](../templates/library.html) (Mountpoint einzuführen) |

**Severity-Übereinstimmung:** Alle 7 direkt-übertragbaren Findings haben **identische** Heuristik+Severity wie F-3.2-Quellen. **Keine Daily-Usage-Schmerz-Schiebung** — Begründung in Disposition-Verteilung-Sektion oben dokumentiert.

### Teil-übertragbare F-3-Patterns (mit modifizierten library-List-View-Findings)

| F-3-Pattern | F-3.2-Finding-Quelle | List-View-Finding | Heuristik | Severity | Anpassung |
|-------------|---------------------|-------------------|-----------|----------|-----------|
| **P5 — Datum-Lokalisierung** | F-3.2 F8 (H1 Sev 3, JS-Pre-Population mit UTC-toISOString) | F14 (H4 Sev 1, Server-side `%b`-Locale-Abhängigkeit) | H4 | 1 ⚠️ code-only | **Andere Mechanik**: F-3 war JS-`<input type="datetime-local">`-Pre-Population (Notion-Submit-Daten-Fehler); hier ist es Server-side `strftime` im Card-Render-Pfad (Display-Cosmetic). Severity-Verschiebung Sev 3 → Sev 1, weil hier kein Daten-Fehler in Zielsystem-Quelle (Card-Datum ist nur Display, kein Submit-Pfad). Heuristik-Verschiebung H1 → H4, weil hier Konsistenz-Bruch mit DE-Microcopy-Welle (P6-Konvergenz), nicht System-Status-Bruch. |

**P2 — Auto-Save Pending-State** ist in F-6.1 als "teil-übertragbar" markiert mit Notiz "Aufwand und UX-Wert eher gering — Folde-Kandidat". F6-REVIEW **bestätigt** die Folde-Disposition: kein eigenständiges F-6.2-Finding, weil (a) List-View hat keinen Title/Tags-Edit-Pfad, (b) Favorite-Toggle-Pending-State (ms-zwischen-Klick-und-Server-Response) ist subjektiv kaum wahrnehmbar bei lokaler Latenz, (c) F-3-IMPL P2 (`.c-input--dirty`-inset-shadow) hat keinen passenden Code-Anker auf List-View-Cards. F6-PATTERNS kann das als Folde-Hinweis aufgreifen, ohne eigenständiges Pattern.

### Bereits konvergente F-3-Patterns (List-View erfüllt das F-3-Pattern strukturell — kein Finding nötig)

| F-3-Pattern | Bereits erfüllt durch | Code-Anker |
|-------------|------------------------|------------|
| **P9 — Tags-Input mit Chip-Visualisierung** | List-View rendert Tag-Chips Server-side mit `.c-tag`-Klasse — **interessante Inversion: List-View ist hier voraus, F-3-IMPL P9 hat das Pattern aus der List-View übernommen für Detail-View** | [templates/library.html:67-71](../templates/library.html#L67-L71), [static/css/style.css:1108-1110](../static/css/style.css#L1108-L1110) |
| **P11 — Sidebar-Active-State** | F-3-IMPL hat den path-Match `'/library' in request.path` in `base.html:84` etabliert — deckt sowohl `/library` (List) als auch `/library/<id>` (Detail) ab | [templates/base.html:84](../templates/base.html#L84) |
| **P12 — File-Size mit KB/B-Fallback (Server-side)** | Helper-Existenz `file_size`-Jinja-Filter ist da — auf List-View **n/a für Display** (Cards zeigen kein File-Size). Helper bleibt für Detail-View. | [app_pkg/__init__.py:114-123](../app_pkg/__init__.py#L114-L123) |

### Nicht-anwendbare F-3-Patterns

| F-3-Pattern | Begründung |
|-------------|-----------|
| **P4 — Notion-Form State-Preservation** | List-View hat keine Notion-Form (Notion-Send ist Detail-only). |
| **P7 — Notion-Submit Persistent Error-Banner** | s.o. — kein Notion-Submit-Pfad auf List-View. |
| **P10 — Notion-Toggle aria-expanded/aria-controls** | s.o. — kein Notion-Toggle-Disclosure auf List-View. |
| **P13 — Page-`<title>` aktualisieren nach Title-Edit** | List-View hat keinen Title-Edit-Pfad; `<title>Library</title>` bleibt statisch. |

### Helper-Reuse-Reflexion (Master-Annotation 5)

F-6.1 hat in der Helper-Reuse-Spuren-Sektion **eine wichtige methodische Reflexion** dokumentiert: nicht jede Helper-Reuse-Vermisst-Stelle ist automatisch ein H4-Finding, wenn die "Alternative" eine begründete Design-Wahl ist. Drei Stellen werden hier explizit als **positive Disziplin-Notiz** dokumentiert (kein Heuristik-Finding):

- **`saveViewState/loadViewState` zweite Call-Site: nein — URL-Persistierung ist die etablierte Design-Wahl.** Der gesamte View-State der List-View (Sortierung / Filter / Favorites / Suche / Pagination) wird über URL-Query-Params persistiert. Vorteile: bookmark-bar, sharable, browser-back-restoriert State, kein localStorage-Quota-Risk. F6-PATTERNS könnte argumentieren, dass beim Default-Reload ohne Query-Params trotzdem der letzte User-State wiederhergestellt werden soll (Reader-Mode-Analogie aus markdown_converter F9), aber das ist eine **Design-Entscheidung**, kein Code-deduzierter H4-Befund. → **Keine H4-Verletzung** trotz Helper-Reuse-Drift, weil die Alternative (URL-Persistierung) eine etablierte begründete Design-Wahl ist.
- **`confirmInPlace` aus F-4-IMPL: zweite Call-Site nein — kein Bulk-Delete-Pfad auf List-View.** `confirmInPlace` ist die Idle→Confirm-Pending→Cancelling-State-Machine aus `audio_converter.js` für **mid-flight-cancel** eines laufenden RQ-Jobs, kein Generic-Bulk-Delete-Helper. Per-Card-Delete (#24) verwendet rohes `confirm()` und ist semantisch näher zu F-3 #17 (Detail-View Delete) als zu F-4 Cancel. → **Keine H4-Verletzung**, weil das Helper-Pattern semantisch nicht passt.
- **`confirmIfLong`: n/a — Card-Delete soll jeden Delete bestätigen** (Threshold-Logik passt nicht zur Semantik der Card-Lösch-Aktion). → **Keine H4-Verletzung**.

**Methodische Lehre für künftige Wellen** (Memory-Kandidat, siehe Sprint-Prompt-Abschluss): **Helper-Reuse-Drift bedeutet nicht zwingend H4-Verletzung wenn die Alternative eine begründete Design-Wahl ist.** Beispiele: URL- vs. localStorage-Persistierung; Server-Side-Sort vs. Client-Side-Filter; rohes `confirm()` vs. `confirmInPlace`-State-Machine. F6-REVIEW bestätigt die F-6.1-Reflexion und übernimmt sie in die Cross-Feature-H4-Sektion als positive Disziplin-Notiz.

**Echte H4-Verletzungen** (in der Findings-Tabelle erfasst, **nicht** in der "begründete Design-Wahl"-Liste):

- **`showAlert`-Mountpoint fehlt** ([templates/library.html](../templates/library.html)) → **F10 H4 Sev 1** (echter struktureller Vorbedingung-Befund mit F-3-P15-Korrespondenz). Die Helper-Reuse-Vermisst-Stelle hier ist **keine** begründete Design-Wahl, sondern ein **vergessener Mountpoint** (analog F-3 vor F-3-IMPL).
- **`safeJSON` fehlt in PUT/DELETE-Pfaden** ([static/js/library.js:5,29](../static/js/library.js#L5)) → in BT1+BT2 als Patch-Anker dokumentiert, in den Findings F2/F3/F4/F5 als Helper-Reuse-Aspekt der silent-fail-Familie absorbiert. Keine separate Entkopplung als "echte H4-Verletzung", weil der Helper-Reuse-Aspekt strukturell mit dem Error-Handling-Aspekt zusammenfällt.

---

## List-spezifische Heuristik-Sub-Sektion (Findings ohne F-3-Korrespondent)

**9 von 17 Findings** sind list-spezifisch ohne F-3-Korrespondent — der "neue" Heuristik-Anteil dieses Sprints. Heuristik-Verteilung:

| Heuristik | Findings | Anzahl |
|-----------|----------|--------|
| H1 | F16 (aria-live Card-Remove) | 1 |
| H4 | F9 (Search-not-auto-submit), F12 (favorites='' URL) | 2 |
| H6 | F11 (Search-no-live), F15 (Per-Page-Size hardcoded), F17 (Card-Hover-Lift) | 3 |
| H9 | F1 (Copy-200char), F8 (Empty-State filter-aware), F13 (Type-Filter-Validation) | 3 |

**Severity-Verteilung der list-spezifischen Findings:**

- Sev 3 (1): F1 Copy-Btn 200-char-Preview — daily-usage-Hotspot
- Sev 2 (2): F8 Empty-State-filter-aware, F9 Search-not-auto-submit
- Sev 1 (6): F11, F12, F13, F15, F16, F17 — list-Polish-Long-Tail

**Mapping auf F-6.1 List-View-State-Sub-Sektion:**

| State-Klasse | Findings (list-spezifisch) |
|---------------|----------------------------|
| Sortierung-State | — (kein Finding; Sort-Active-Marker via `selected`-Attribut funktioniert) |
| Filter-State (Type+Favorites) | F12 favorites='' URL (Filter-Output), F13 Type-Filter-Validation (Filter-Input) |
| Suche-State | F9 Search-not-auto-submit (Konsistenz), F11 Search-no-live (Polish) |
| Bulk-Selektion-State | — (existiert nicht, keine Findings) |
| Pagination-State | F15 Per-Page-Size hardcoded |
| Empty-State | F8 Empty-State filter-aware |
| Card-Interaktion (Hover/Click/a11y) | F1 Copy-200char, F16 aria-live Card-Remove, F17 Card-Hover-Lift |

Dass die meisten list-spezifischen Findings auf Sev 1 liegen, ist **kein Indiz für niedrigen Daily-Usage-Schmerz** — die Sev-3-Hotspots (F1 Copy + Silent-Fail-Familie F2-F5 + EN-Strings F6) sind eindeutig daily-usage-getrieben. Der Long-Tail an Sev-1-list-Polish-Items ist normaler Bestand einer Reader-Replacement-Reifung und in F6-PATTERNS / F6-IMPL als Folde-Wellen sortierbar.

---

## Schwerpunkt-Cluster

Vier thematische Cluster, in denen sich die schweren Findings konzentrieren — analog F-3.2's "Silent-Failure / Notion-Wipe / Cross-Feature-Helper-Drift" und F-5.2's "Submit-Resilienz / Helper-Konvergenz / Two-Dark-Modes / Reader-Mode-Polish":

### Cluster 1 — Silent-Failure-Familie (F1, F2, F3, F4, F5; Sev 3, 4 von 5 ⚠️ code-only — F1 ist Code-evident)

Copy-Btn-Daten-Verlust-light + `toggleFavorite`/`deleteConversion` komplett unbeobachtet. Drei verschiedene Mechaniken (Copy-Quelle-falsch + PUT-silent + DELETE-silent), aber alle sitzen im daily-usage-Hotspot der Per-Card-Quick-Actions. **Daily-Usage-Schmerz hoch** wegen Reader-Replacement-Workflow-Ketten. F-3 Cluster 1 hat genau diese Wurzel-Struktur für `library_detail` etabliert (`updateField`/`toggleFavorite`/`deleteConversion`-silent-fails) — F-3-IMPL hat sie via `showAlert`-Banner-Pattern + `.catch`-Branch + `r.ok`-Check + `safeJSON`-Wrap geschlossen. **Inversion auf List-View**: identische Patches strukturell, plus zusätzliche aria-live-Region (F16) im Delete-Pfad. Fix-Pfad in F6-PATTERNS: ein einzelner Pattern-Cluster "Silent-Failure-Elimination" der `showAlert` + `safeJSON` + `r.ok`-Check + Recovery-Microcopy ergänzt; Bug-Tickets BT1+BT2+BT3+BT4 unabhängig vorab fixbar (BT3 Copy-Quelle ist eigene Mechanik, kann separat gefixt werden ohne Mountpoint-Vorbedingung).

### Cluster 2 — Cross-Feature-H4-Helper-Reuse zu F-3 (F2/F3, F4/F5, F6, F7, F10, F14; ~47% Quote)

Sechs Findings (acht Reihen wenn man F2+F3 und F4+F5 doppelt zählt) mit F-3-Korrespondenz: Silent-Fail-Familie + EN-Strings + Toast-Level + Banner-Mountpoint + %b-Locale (teil). **F-3-IMPL hat die Patches für `library_detail` schon durchgezogen**; List-View ist die Inversions-Welle. Helper-Reuse-Drift hat **drei begründete Stellen** (saveViewState, confirmInPlace, confirmIfLong — siehe Cross-Feature-H4-Sektion) und **zwei echte Lücken** (showAlert-Mountpoint fehlt = F10; safeJSON fehlt = in BT1/BT2 absorbiert). Fix-Pfad in F6-PATTERNS: Pattern-Cluster "Helper-/Konvention-Konvergenz zu F-3" der die Helper-Migration (showAlert + safeJSON + Toast-Level + Banner-Mountpoint) plus DE-Microcopy plus DE-Monatsnamen-Map bündelt — sehr strukturell ähnlich zu F-3.2 Cluster 3.

### Cluster 3 — List-View-State-Visibility und Empty-State-Recovery (F8, F9, F11; Sev 1–2)

Drei Findings rund um Filter-/Suche-/Empty-State-Visibility: Empty-State nicht filter-aware (F8 H9 Sev 2 — daily-usage-relevant), Search-not-auto-submit (F9 H4 Sev 2 — internal H4-Bruch innerhalb der Filter-Bar), Search-no-live (F11 H6 Sev 1 — Polish). Alle drei list-spezifisch ohne F-3-Korrespondenz. **Kein Cross-Feature-Anker**, daher eigenständiger List-View-Cluster. Fix-Pfad in F6-PATTERNS: Pattern-Cluster "List-View-State-Recovery" der Empty-State filter-aware macht (Recovery-Pfad "Filter zurücksetzen") + Search-Auto-Submit/Live-Search-Polish-Pfad konsolidiert.

### Cluster 4 — List-Polish-Long-Tail (F12, F13, F14, F15, F16, F17; Sev 1)

Sechs Sev-1-Findings ohne primäre Cluster-Zugehörigkeit: favorites='' URL-Artifact (F12), Type-Filter-Validation (F13), %b-Locale (F14 — auch in Cluster 2 als P5-teil), Per-Page-Size hardcoded (F15), aria-live Card-Remove (F16 — auch in Cluster 1 als BT2-Verzahnung), Card-Hover-Lift (F17). **Skalierungs- und a11y-Polish**, kein daily-usage-akut. F6-PATTERNS kann diese Findings als Mini-Patterns oder als "Folde-in-Sammelpaket"-Disposition bündeln; F6-IMPL kann sie als Sub-Batch C oder spätere Welle einplanen. **Disposition-Hinweis**: F14 (%b-Locale) und F16 (aria-live) haben Doppel-Zugehörigkeit (Cluster 2 / Cluster 1) und können in den jeweiligen Pattern-Cluster-Patches absorbiert werden, statt als eigenständige Folde-Items zu rangieren.

---

## Zusammenfassung

- **Heuristik-Findings gesamt:** 17
- **Davon Schweregrad 4 (kritisch):** 0 — keine Datenverlust-/Blockade-Pfade auf der primären Lese-Aufgabe (Library-Browsing funktioniert; Quick-Actions haben silent-fail aber nicht "blockierend"; Copy-Btn-Preview-Only ist Daten-Verlust-light, aber Full-Content noch in DB erreichbar via Detail-View)
- **Davon Schweregrad 3:** 6 (F1 Copy-200char H9, F2 toggleFavorite-silent H1, F3 toggleFavorite-Recovery H9, F4 deleteConversion-silent H1, F5 deleteConversion-Recovery H9, F6 EN-Strings H4)
- **Davon Schweregrad 2:** 3 (F7 Toast-Level H4, F8 Empty-State-filter-aware H9, F9 Search-not-auto-submit H4)
- **Davon Schweregrad 1:** 8 (F10 Banner-Mountpoint H4, F11 Search-no-live H6, F12 favorites='' URL H4, F13 Type-Filter-Validation H9, F14 %b-Locale H4, F15 Per-Page-Size H6, F16 aria-live H1, F17 Card-Hover H6)
- **Reine Bug-Tickets (mit Ticket-Material):** 4 (BT1 toggleFavorite-Errors, BT2 deleteConversion-Errors, BT3 Copy-200char-Preview-Quelle, BT4 Toast-Level Copy-Failure) — **alle Finding-linked, keine Pure-Bugs ohne H-Aspekt** (Disposition-Abweichung von F-3.2's BT7/BT8-Mechanik begründet im Bug-Tickets-Sektions-Header)
- **Cross-Feature-H4-Findings:** 8 von 17 — **Cross-Feature-Konvergenz-Quote ~47%** (F2/F3 + F4/F5 + F6 + F7 + F10 + F14), im erwarteten Master-Bereich 35-50%, höher als F-3.2's 35% (List+Detail teilen Datenklasse + Helper-Set), niedriger als F-5.2's 86% (Konverter-Schwesterpaar mit Helper-Bestand)
- **`⚠️ code-only`-markierte Findings:** 10 von 17 (F2, F3, F4, F5, F7, F9, F11, F14, F16, F17) — Master-Walkthrough-Nachreichung empfohlen vor F6-PATTERNS für mindestens F1/BT3 (Copy-Source-Verifikation per DevTools-Konsole), F2-F5/BT1+BT2 (DevTools-Network-Throttle 5-Min-Verifikation), F7/BT4 (Toast-Level-Visualisierung), F14 (Docker-Container-Locale-Check via `docker exec ... locale`); F9 (Search-Submit-Verhalten Live-Browser), F17 (Card-Hover-Robustheit via Pointer-Throttle). Wenn Master-Bandwidth knapp ist: F6-PATTERNS kann auch ohne Walkthrough beginnen, aber dann sollten die Pattern-Vorschläge für Cluster 1 und Cluster 2 das `⚠️ code-only`-Risiko in den Phasen-Stops markieren, damit ein Implementierungs-Smoke vor Merge in F6-IMPL zwingend ist.

**Schweregrad-Skala:**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse)

**Bemerkungen-Disposition (16 Stufe-1-Befunde, vollständig disponiert):**
- Findings only: 11 (Befund 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
- Beides (Finding + Bug-Ticket): 4 Inventur-Befunde → 6 Findings (Befund 1 → F1+BT3; Befund 2 → F2+F3+BT1; Befund 3 → F4+F5+BT2; Befund 4 → F7+BT4)
- Bugs only: 0 — Disposition-Abweichung von F-3.2 (BT7/BT8) und F-5.2 (BT3/BT4/BT5) begründet
- Meta absorbiert: 1 (Befund 16 Helper-Reuse — in Cross-Feature-H4-Sektion absorbiert per Sprint-Prompt Master-Annotation 5)

**Master-Walkthrough-Empfehlung vor F6-PATTERNS:** Optional 15-30 Min Walkthrough-Session für die `⚠️ code-only`-Sev-2/3-Findings — primär F1 (Copy-Quelle-Verifikation), F2-F5 (DevTools-Network-Throttle), F7 (Toast-Level visuell), F14 (Container-Locale). Bei knapper Master-Bandwidth: F6-PATTERNS kann ohne Walkthrough beginnen mit Smoke-Pflicht-Marker für F6-IMPL (analog F-3-IMPL und F-5-IMPL-Methodik).
