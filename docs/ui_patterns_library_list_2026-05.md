# UX-Patterns + Microcopy: library List-View (2026-05-10)

**Methodik:** Stufe 3 der Duan-Kaskade (Duan et al., *Heuristic Evaluation with LLMs*, CHI 2024). Konkrete Patterns + DE-Microcopy auf Basis der Heuristik-Findings aus Stufe 2. Konsolidiert die 17 Stufe-2-Findings + 4 Bug-Tickets auf 14 Pattern-Blöcke (3 konsolidiert + 11 einzeln). Bug-Tickets BT1–BT4 sind in den Patterns ihrer verknüpften Findings mit-adressiert (BT1 ↔ P3 Auto-Save, BT2 ↔ P4 Delete, BT3 ↔ P2 Copy-Full-Content, BT4 ↔ P6 Toast-Level). **Keine Pure-Bug-Tickets ohne H-Aspekt** in F-6.2 — siehe Disposition-Verteilung in Findings-Doc.
**Quelle Findings:** [docs/ui_findings_library_list_2026-05.md](ui_findings_library_list_2026-05.md)
**Quelle Inventur:** [docs/ui_inventory_library_list_2026-05.md](ui_inventory_library_list_2026-05.md)
**F-1 / F-2 / F-3 / F-4 / F-5 Patterns als Referenz:** [docs/ui_patterns_document_converter_2026-05.md](ui_patterns_document_converter_2026-05.md), [docs/ui_patterns_audio_converter_2026-05.md](ui_patterns_audio_converter_2026-05.md), [docs/ui_patterns_library_detail_2026-05.md](ui_patterns_library_detail_2026-05.md), [docs/ui_patterns_podcast_flow_2026-05.md](ui_patterns_podcast_flow_2026-05.md), [docs/ui_patterns_markdown_converter_2026-05.md](ui_patterns_markdown_converter_2026-05.md)
**Helper-API:** [static/js/_utils.js](../static/js/_utils.js) — `safeJSON(response)`, `fallbackCopyText(text)`, `showAlert(containerEl, level, msg, options?)`, `showToast(msg, options?)`, `formatFileSize(bytes)`, `formatDatetimeLocalNow()`, `confirmIfLong(text, msg, options?)`, `loadViewState(key, defaultValue?)`, `saveViewState(key, state)`. CSS-Utility `.sr-only` aus [static/css/style.css](../static/css/style.css). Server-side: `file_size`-Jinja-Filter aus F-3-IMPL (auf List-View nicht angezeigt, wird hier nicht eingesetzt).
**Komponenten-Basis:** Existierende Neomorphism-Klassen aus [static/css/style.css](../static/css/style.css) — `c-btn`, `c-btn--primary`, `c-btn--danger`, `c-input`, `c-card`, `c-surface`, `c-alert--danger/success/warning/info` (mit Close-Button + Auto-Dismiss aus F-1 Cluster C), `.c-tag`, `.type-badge`, `.favorite-btn`, `.toast-notification`, `.hidden`, `.sr-only`, `:focus-visible` für `c-btn`. Banner-Mountpoint im List-Template ist heute **nicht vorhanden** — Pattern P1 führt ihn ein als strukturelle Vorbedingung für P3, P4 (analog F-3 P15).

**Microcopy-Regeln:** Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern, Deutsch durchgängig (Du-Form analog F-1 / F-2 / F-3 / F-4 / F-5).
**Aufwand-Skala:** XS / S / M / L (Daumenregel: XS = 1–3 Zeilen, S = ein Handler-Cluster + Microcopy-Sweep, M = neue Mechanik mit State-Refactor oder Server-Touch, L = Cross-Stack-Refactor).
**Impact-Score-Formel:** `Score = Sev × 5 / Aufwand-Gewicht` mit Aufwand-Gewichten XS=1, S=2, M=4, L=8. Höher = besser. Bei konsolidierten Patterns wird die höchste Sev der adressierten Findings genommen (analog F-1.3 / F-2.3 / F-3.3 / F-4.3 / F-5.3).

**Geschwister-Feature-Übernahme aus F-3.3 (zentrale Mechanik dieses Sprints):** F-6.2 hat eine 47% Cross-Feature-H4-Finding-Quote zu F-3 (`library_detail`) ausgewiesen — `library` List-View und `library_detail` teilen die `ConversionHistory`-Datenklasse plus den Helper-Set. F-3.3 hat die Pattern-Mechanik für die korrespondierenden Findings (Auto-Save-Failure-Banner, Delete-Failure-Banner, DE-Microcopy-Sweep, Toast-Level pro Call-Site, Banner-Mountpoint-Vorbedingung, `safeJSON`-Wrap) bereits ausgearbeitet — F-6.3 übernimmt diese Patterns 1:1, **passt nur drei Felder an** (Code-Anker auf `library`-Code statt `library_detail`, Microcopy wo list-spezifisch nötig, Adressiert-Findings auf F-6.2-Nummern). Aufwand wird übernommen außer es gibt einen klaren Begründungs-Grund für Abweichung. F-3-Pattern-Identität bleibt in jeder Pattern-Block-Header-Zeile sichtbar (Feld „F-3-Korrespondenz"). Memory-Anker [feedback_helper_reuse_design_choice.md](file:///Users/olivergluth/.claude/projects/-Volumes-MintHome-CODE-CONVERTER/memory/feedback_helper_reuse_design_choice.md) gilt: Helper-Reuse-Drift mit begründeter Design-Wahl (URL-Persistierung, kein Bulk-Delete, semantisch unpassender Helper) ist keine H4-Verletzung — siehe Cross-Feature-H4-Sektion am Doc-Ende.

**Live-Verifikation-Konvention** (Master-Annotation 3 — kein Master-Walkthrough vor F6-PATTERNS): Patterns für die ⚠️ code-only-Findings (F2, F3, F4, F5, F7, F9, F11, F14, F16, F17) tragen den Sub-Tag `🔥 Smoke-Pflicht in F6-IMPL` mit explizitem Smoke-Mechanik-Hinweis pro Pattern. **Drei Pflicht-Live-Master-Smoke-Patterns** (analog F-5.3 Drei-Pflicht-Kalibrierung) — Master-Annotation 3 explizit übernommen — sind zusätzlich mit `🔥🔥 Pflicht-Live-Master-Smoke`-Marker:
- **P2 (F1+BT3) Copy-Full-Content** — Copy-Paste-Verifikation mit langem Content (>200 char) im Browser.
- **P3 (F2+F3) Auto-Save-Failure-Banner** — DevTools-Network-Throttle Offline → Favorite-Toggle → Banner-Sichtbarkeit (analog F-3-IMPL P1 Smoke-Mechanik).
- **P11 (F14) Card-Datum DE-Lokalisierung** — Browser-Inspektion der Card-Datum-Anzeige (deutsche vs. englische Monatsabkürzung), DE-Erwartung „Mär" statt „Mar".

Andere Smoke-Pflicht-Patterns (P4, P6, P8, P13, P14) sind „code-evident verifiziert im Container" — Smoke per Code-Reading + ggf. Container-Test-Sweep, kein Live-Browser-Master-Smoke nötig.

**List-View-State-Default-Wahl-Konvention** (Master-Annotation 5): Patterns für F8 (Empty-State filter-aware) und F9+F11 (Search-Input UX) enthalten konkrete pragmatische Default-Mechanik (nicht Variante-A/B/C-Liste) — server-side Empty-State-Branching im `library_view`-Render bzw. Submit-Required mit DE-Hint statt Live-Debouncing. „Master-Default-Wahl"-Marker im Pattern-Block. Sub-Thread (F6-IMPL) kann beim Apply abweichen wenn technisches Problem oder bessere Alternative beim Code-Touch auffällt — Bericht-Pflicht; **keine** Variante-A/B/C-Diskussion in F-6.3.

---

## Pattern-Blöcke

### Pattern 1: Banner-Mountpoint-Container im Template (struktureller Vorbedingung-Fix)
**Adressiert Findings:** F10 (H4 Sev 1)
**F-3-Korrespondenz:** F-3.3 P15 (Banner-Mountpoint-Container im Template — direkt übernommen)
**Cluster:** 1 (Silent-Failure-Familie + Cross-Feature-H4 Vorbedingung)
**Live-Verifikation-Status:** — (struktureller Template-Touch, statisch verifizierbar)

- **Pattern:** Aus F-3.3 P15 übernommen — ein Banner-Container ergänzen, damit Banner aus P3 (Auto-Save-Failure) und P4 (Delete-Failure) einen festen Mountpoint haben. `#library-alert-container` direkt unter dem Filter-Bar-`<div class="c-surface ...">` (oben in der Editor-Region, oberhalb der Cards-Grid und der Pagination, sichtbar bei jedem Filter-Stand). Ein Container reicht für die List-View — anders als `library_detail`, das zwei Container braucht (Detail + Notion-Submit), gibt es hier keinen Notion-Submit-Pfad. Container leer im Default-State, kein Spacing wenn ohne Inhalt. Strukturelle Vorbereitung für die Banner-Patterns; pro Container-Sektion ist die Banner-Microcopy lokal zur User-Aktion. Optionale Session-Expired-Microcopy (Helper `safeJSON` wirft `Session expired …` als generische `Error`-Message): wird im Catch-Block der Banner-Patterns abgefangen und mit DE-Microcopy überschrieben — „Sitzung abgelaufen. Seite neu laden und erneut anmelden."
- **Visuelle Hinweise:** Ein `<div id="library-alert-container">`-Tag direkt nach dem Filter-Bar-Block (zwischen Filter-Bar und Cards-Grid in [templates/library.html:35](../templates/library.html#L35)). Container-CSS ist `display: contents` oder einfach kein Default-Style — Banner-Höhe übernimmt der `c-alert`-Block. Optional `margin-bottom: 1rem` damit der Banner sichtbar Abstand zur Cards-Grid hält wenn er aktiv ist.
- **Microcopy** (DE, Du-Form):
  - Session-Expired (gemeinsamer Fallback aus `safeJSON` für P3 + P4): „Sitzung abgelaufen. Seite neu laden und erneut anmelden."
- **Helper-Reuse:** Container-Tag wird beim Aufruf von `showAlert(containerEl, …)` referenziert. Bestehender `showAlert`-Helper trägt Close-Button + Auto-Dismiss-Verhalten (siehe [static/js/_utils.js:39-78](../static/js/_utils.js#L39-L78)).
- **Aufwand:** XS — ein `<div>`-Tag im Template. F-3.3-Aufwand 1:1 übernommen (F-3.3 hatte zwei Container, hier reicht einer — gleicher Aufwand-Bucket).
- **Impact-Score:** 1 × 5 / 1 = **5.0** (die Vorbedingung wirkt indirekt über P3/P4 — Score selbst ist niedrig, Cluster-Wert hoch)
- **Cluster-Hinweis:** Diese Vorbedingung muss in F6-IMPL als erstes greifen (vor P3 + P4), analog F-3-IMPL Sub-Batch A.

---

### Pattern 2: Copy-Btn liefert Full-Content statt 200-char-Preview
**Adressiert Findings:** F1 (H9 Sev 3)
**Adressiert Bug-Ticket:** BT3 (Copy-Quelle ist DOM-Preview-Slice statt DB-Full-Content)
**F-3-Korrespondenz:** — (list-spezifisch — Copy-Btn existiert auf Detail-View nicht in dieser Form, weil Detail-View Full-Content sowieso schon im DOM hat)
**Cluster:** 1 (Silent-Failure-Familie — Daten-Verlust-light)
**🔥 Smoke-Pflicht in F6-IMPL** — **🔥🔥 Pflicht-Live-Master-Smoke**

- **Pattern:** `copyContent(id)` in [static/js/library.js:17-25](../static/js/library.js#L17-L25) liest heute `card.querySelector('.line-clamp-3').textContent` — exakt der Server-clipped Inhalt aus `{{ conv.content[:200] }}...` aus [templates/library.html:59](../templates/library.html#L59). Ergebnis: User klickt „Copy", paste-t in Notion/Editor, bekommt nur 200 Zeichen + „...". Pattern-Mechanik: Full-Content **nicht** über das DOM-Preview-Element lesen, sondern aus einer dedizierten Quelle. **Master-Default-Wahl**: per `data-content`-Attribut auf der Card im Template-Render einbetten (`<div class="c-card" data-id="{{ conv.id }}" data-content="{{ conv.content }}">`), `copyContent(id)` liest dann `card.dataset.content`. Vorteile: kein neuer Backend-Endpoint, kein Roundtrip-Latency-Hit, kein Race auf 5xx-Pfaden. Nachteil: vergrößert das HTML pro Card-Render um den Full-Content-String — bei pagination=20 und durchschnittlich 5–20 KB Content pro Conversion sind das 100–400 KB extra HTML pro Page. Akzeptabel für Single-User-Single-Instance-LAN-only-App; falls F6-IMPL Skalierungs-Bedenken hat (Reader-Replacement Wachstum auf mehrere hundert/tausend Conversions), Alternative wäre ein Fetch-On-Click `GET /api/conversions/<id>/content` mit Plaintext-Body — Aufwand-Anpassung dann S → M. Sub-Thread F6-IMPL kann pragmatisch entscheiden.
- **Visuelle Hinweise:** Keine sichtbare UI-Änderung — Copy-Btn-Tooltip/Label bleibt identisch (DE-Microcopy aus P5: „Inhalt kopieren"). User merkt den Fix nur beim Paste in Editor (Full-Content statt Preview-Slice).
- **Microcopy** (DE, Du-Form, max 3 Wörter pro Element; Toast max 2 Sätze):
  - Copy-Btn Label (aus P5 DE-Sweep): „Kopieren"
  - Copy-Btn title: „Inhalt kopieren"
  - Toast-Success (aus P5 + P6 — Level `success` default): „Inhalt kopiert"
  - Toast-Failure (aus P6 — Level `danger` per opts): „Kopieren fehlgeschlagen"
- **Helper-Reuse:** `fallbackCopyText(text)` aus [static/js/_utils.js:18-34](../static/js/_utils.js#L18-L34) bleibt unverändert; nur die `text`-Quelle wechselt von `card.querySelector('.line-clamp-3').textContent` auf `card.dataset.content`. `showToast` aus [static/js/_utils.js:92-117](../static/js/_utils.js#L92-L117) bleibt; Level-Korrektur kommt aus P6. Code-Anker: [static/js/library.js:17-25](../static/js/library.js#L17-L25) `copyContent`; [templates/library.html:41](../templates/library.html#L41) Card-Wrapper für `data-content`-Attribut.
- **Aufwand:** S — Template-Touch für `data-content`-Attribut + JS-Änderung in `copyContent` (eine Zeile) + Toast-Microcopy (in P5 mit-bearbeitet) + Toast-Level-Fix (in P6 mit-bearbeitet). Backend-Touch entfällt bei der Default-Wahl (Embed). Aufwand wäre XS wenn isoliert betrachtet (eine Zeile JS + ein Attribut Template), aber mit dem Embed-Volumen-Hinweis und der Smoke-Pflicht ist S realistisch.
- **Impact-Score:** 3 × 5 / 2 = **7.5**
- **Smoke-Mechanik:** Eine Conversion mit >200 char Content in der Library haben (alternativ: kurz testweise eine erstellen). „Copy"-Btn klicken. In Editor (Notion / VS Code / Plaintext) pasten. Erwartung: vollständiger Content statt 200-Zeichen-Slice + „...". Toast „Inhalt kopiert" sichtbar.
- **Verzahnung-Hinweis:** BT3 (Copy-Quelle-Bug) wird durch dieses Pattern aufgelöst — kein separater Bug-Ticket-Apply nötig.

---

### Pattern 3: Auto-Save Favorite-Toggle silent-fail
**Adressiert Findings:** F2 (H1 Sev 3), F3 (H9 Sev 3)
**Adressiert Bug-Ticket:** BT1 (`toggleFavorite` ohne Error-Handling und ohne `safeJSON`-Wrap)
**F-3-Korrespondenz:** F-3.3 P1 + P14 (Auto-Save-Failure-Banner + `safeJSON`-Wrap — direkt übernommen, mit Favorite-Toggle-spezifischem Microcopy-String)
**Cluster:** 1 (Silent-Failure-Familie + Cross-Feature-H4)
**🔥 Smoke-Pflicht in F6-IMPL** — **🔥🔥 Pflicht-Live-Master-Smoke**

- **Pattern:** Aus F-3.3 P1 + P14 übernommen — `toggleFavorite(id, btn)` in [static/js/library.js:3-15](../static/js/library.js#L3-L15) bekommt vollständige Fehler-Behandlung: `.catch()` für Network-Fail + `if (!r.ok)`-Branch für 4xx/5xx + `safeJSON(r)`-Wrap für Session-Expired-Detection (Login-Redirect-HTML als 200). Bei Erfolg bleibt der bisherige optimistic Glyph-Toggle (★/☆). Bei Fehler erscheint ein persistenter `c-alert--danger`-Banner im `#library-alert-container` (siehe P1) mit Recovery-Hinweis; der Glyph wird **nicht** geändert (kein optimistic-rollback nötig, weil heute der Toggle erst im `r.ok`-Branch passiert — Pattern behält das, ergänzt nur den else-Pfad). Bei Session-Expired (`safeJSON`-Throw) wird die Session-Expired-Microcopy aus P1 angezeigt.
- **Visuelle Hinweise:** Banner oben (`#library-alert-container` aus P1), Danger-Tint, Close-× über `showAlert`-Default. Favorite-Btn-Glyph bleibt im alten Zustand (kein erfolgreicher Toggle). Kein Pending-Indikator zwischen Klick und Server-Response (Roundtrip ist kurz; Pending-State wäre eine eigene Welle, hier nicht).
- **Microcopy** (DE, Du-Form, max 2 Sätze):
  - Favorite-Toggle-Failure (generisch): „Favorit konnte nicht aktualisiert werden. Verbindung prüfen und erneut versuchen."
  - Favorite-Toggle-Failure mit Server-Detail (`r.status >= 500`): „Favorit konnte nicht aktualisiert werden — Server-Fehler. Bitte später erneut versuchen."
  - Session-Expired (aus P1, gemeinsamer Fallback): „Sitzung abgelaufen. Seite neu laden und erneut anmelden."
- **Helper-Reuse:** `showAlert(libraryAlertContainer, 'danger', msg)` aus [static/js/_utils.js:39-78](../static/js/_utils.js#L39-L78) ersetzt den fehlenden Error-Branch. `safeJSON(r)` aus [static/js/_utils.js:7-16](../static/js/_utils.js#L7-L16) für Session-Expired-Detection. Inline-Code-Anker: [static/js/library.js:3-15](../static/js/library.js#L3-L15) `toggleFavorite`.
- **Aufwand:** S — `.catch` + `r.ok`-Branch + `safeJSON`-Wrap in einer Funktion + Banner-Container-Reuse aus P1 + Microcopy. Helper schon eingebunden (`_utils.js` lädt via `base.html`). F-3.3 P1-Aufwand (S) 1:1 übernommen — identische Mechanik.
- **Impact-Score:** 3 × 5 / 2 = **7.5**
- **Smoke-Mechanik:** DevTools öffnen → Network-Tab → Throttling auf „Offline" (oder einen Service-Worker-Block / Backend-Stop). Favorite-Btn an einer Card klicken. Erwartung: Banner mit „Favorit konnte nicht aktualisiert werden …" oben sichtbar; Glyph bleibt im alten Zustand. Throttle zurück auf „Online" → erneuter Klick → Banner verschwindet (oder wird durch nächsten Banner überschrieben), Glyph togglet erfolgreich.
- **Konsolidierung:** F2 (H1 Sichtbarkeit fehlt) und F3 (H9 Recovery-Anleitung fehlt) entstehen aus derselben fehlenden Error-Behandlung in `toggleFavorite`. Eine Lösung (showAlert + Recovery-Microcopy) erfüllt beide Heuristiken — Sichtbarkeit per Banner, Recovery per Microcopy-Hinweis. Identisch zu F-3.3-Konsolidierungs-Logik (F-3.3 P1 löst F-3.2 F1+F2). BT1 mit-gelöst.

---

### Pattern 4: Delete-Conversion silent-fail
**Adressiert Findings:** F4 (H1 Sev 3), F5 (H9 Sev 3)
**Adressiert Bug-Ticket:** BT2 (`deleteConversion` ohne Error-Handling und ohne `safeJSON`-Wrap)
**F-3-Korrespondenz:** F-3.3 P3 + P14 (Delete-Failure-Banner + `safeJSON`-Wrap — direkt übernommen, mit list-Card-Remove-spezifischer Mechanik)
**Cluster:** 1 (Silent-Failure-Familie + Cross-Feature-H4)
**🔥 Smoke-Pflicht in F6-IMPL** (code-evident verifiziert im Container)

- **Pattern:** Aus F-3.3 P3 + P14 übernommen — `deleteConversion(id, btn)` in [static/js/library.js:27-38](../static/js/library.js#L27-L38) bekommt einen `else`-Branch nach dem `r.ok`-Check und einen `.catch()` für Network-Fail plus `safeJSON(r)`-Wrap. Im Fehlerpfad: persistenter `c-alert--danger`-Banner im `#library-alert-container` (gemeinsam mit P3). Card bleibt im DOM (kein optimistic-remove vor Server-Bestätigung — heute schon korrekt im Code, bleibt). Bei `r.status === 404` extra Pfad: „bereits gelöscht" mit Card-Remove + info-Banner (Race-Case wenn User in zwei Tabs gelöscht hat). Beim erfolgreichen Delete ergänzt P13 (aria-live) die Card-Remove-Mechanik. **Anders als F-3.3 P3**: kein Page-Navigations-Pfad — auf der List-View bleibt der User auf derselben Page, nur die Card verschwindet.
- **Visuelle Hinweise:** Banner oben (`#library-alert-container` aus P1), Danger-Tint, Close-× über `showAlert`-Default. Während des DELETE-Roundtrips kein Loading-Indikator am Delete-Button (Roundtrip kurz; F-3.3 P3 hatte „Lösche …"-Text-Swap, hier optional weil Card-Action ohne Page-Bleibe — F6-IMPL kann pragmatisch entscheiden, ob Text-Swap hinzu kommt).
- **Microcopy** (DE, Du-Form, max 2 Sätze; Buttons max 3 Wörter):
  - Confirm-Dialog (aus P5 DE-Sweep, ersetzt heutiges `confirm('Delete this conversion? This cannot be undone.')`): „Diesen Eintrag wirklich löschen? Das kann nicht rückgängig gemacht werden."
  - Delete-Failure-Banner generisch: „Löschen fehlgeschlagen. Verbindung prüfen und erneut versuchen."
  - Delete-Failure-Banner Server-Fehler (`r.status >= 500`): „Löschen fehlgeschlagen — Server-Fehler. Bitte später erneut versuchen."
  - Delete-Race (404, info-Banner mit Auto-Dismiss): „Eintrag wurde bereits entfernt."
  - Session-Expired (aus P1, gemeinsamer Fallback): „Sitzung abgelaufen. Seite neu laden und erneut anmelden."
  - Delete-Btn Label (aus P5 DE-Sweep): „Löschen"
  - Delete-Btn title: „Eintrag löschen"
- **Helper-Reuse:** `showAlert(libraryAlertContainer, 'danger', msg)` für Failure; `showAlert(libraryAlertContainer, 'info', msg, { autoDismissMs: 4000 })` für 404-Race vor dem Card-Remove. `safeJSON(r)` aus [static/js/_utils.js:7-16](../static/js/_utils.js#L7-L16) für Session-Expired-Detection. Inline-Code-Anker: [static/js/library.js:27-38](../static/js/library.js#L27-L38).
- **Aufwand:** S — `else`-Branch + `.catch` + `safeJSON`-Wrap + Microcopy + DE-`confirm`-Übersetzung (in P5 mit-bearbeitet). Banner-Container teilen mit P3. F-3.3 P3-Aufwand (S) 1:1 übernommen.
- **Impact-Score:** 3 × 5 / 2 = **7.5**
- **Smoke-Mechanik (code-evident im Container):** Code-Reading-Verifikation der Mechanik (analog F-3 P3): `else`-Branch nach `r.ok`-Check, `.catch`-Pfad, `safeJSON`-Aufruf, Microcopy-Strings. Bei Bedarf zusätzlich Container-Test mit gemocktem `fetch`-Fail (in der Test-Suite oder per `pytest`-Charakterisierung). **Kein Pflicht-Live-Master-Smoke** — Mechanik ist identisch zu F-3 P3 und dort schon Live-verifiziert.
- **Konsolidierung:** F4 (H1 Sichtbarkeit) und F5 (H9 Recovery) sind dieselbe Mechanik wie P3, nur in Delete-Pfad statt Save-Pfad. Eine Lösung (showAlert + Recovery-Microcopy) erfüllt beide Heuristiken. Identisch zu F-3.3 (P3 löst F-3.2 F4+F5). BT2 mit-gelöst. **Verzahnung mit P13** (aria-live Card-Remove): aria-live-Hint im Erfolgs-Branch des Delete-Pfades ergänzen — gemeinsamer Code-Touch sinnvoll, F6-IMPL kann beide in einem Sub-Batch zusammenziehen.

---

### Pattern 5: DE-Microcopy-Pass flächendeckend
**Adressiert Findings:** F6 (H4 Sev 3)
**F-3-Korrespondenz:** F-3.3 P6 (DE-Microcopy-Pass flächendeckend — direkt übernommen, mit List-View-Strings angepasst)
**Cluster:** 1 (Cross-Feature-H4-Helper-Reuse zu F-3)
**Live-Verifikation-Status:** — (reine String-Substitution, statisch verifizierbar)

- **Pattern:** Aus F-3.3 P6 übernommen — alle ~18 EN-Strings auf der List-View-Seite werden in DE/Du-Form übersetzt — sowohl Template-Strings (Filter-Bar-Optionen + Placeholder + Labels + Buttons + Tooltips, Type-Badge-Texte, Toolbar-Buttons, Pagination, Empty-State) als auch JS-Strings (Toast-Texte aus `copyContent`, `confirm`-Dialog aus `deleteConversion`). **Type-Badge-Texte überlappen 1:1 mit F-3.3 P6** — wenn beide Patterns in einem Sub-Batch laufen, einmal fixen genügt. Konstanten-Tabelle unten als Single-Source-of-Truth.
- **Visuelle Hinweise:** Keine — reine String-Substitution. Layout bleibt identisch.
- **Microcopy** (Konstanten-Tabelle, EN → DE):

| Position | Anker | EN | DE |
|----------|-------|----|----|
| Filter-Bar Type-Select Default | [library.html:12](../templates/library.html#L12) | All Types | Alle Typen |
| Filter-Bar Type-Option | [library.html:13](../templates/library.html#L13) | Document | Dokument |
| Filter-Bar Type-Option | [library.html:14](../templates/library.html#L14) | Audio | Audio |
| Filter-Bar Type-Option | [library.html:15](../templates/library.html#L15) | Dialogue | Dialog |
| Filter-Bar Type-Option | [library.html:16](../templates/library.html#L16) | Markdown | Markdown |
| Filter-Bar Type-Select title | [library.html:11](../templates/library.html#L11) | Filter by type | Nach Typ filtern |
| Filter-Bar Search-Input placeholder | [library.html:21](../templates/library.html#L21) | Search title, content, tags... | Titel, Inhalt, Tags suchen … |
| Filter-Bar Favorites-Label | [library.html:26](../templates/library.html#L26) | Favorites | Favoriten |
| Filter-Bar Sort-Select title | [library.html:28](../templates/library.html#L28) | Sort order | Sortierung |
| Filter-Bar Sort-Option | [library.html:29](../templates/library.html#L29) | Newest | Neueste zuerst |
| Filter-Bar Sort-Option | [library.html:30](../templates/library.html#L30) | Oldest | Älteste zuerst |
| Filter-Bar Sort-Option | [library.html:31](../templates/library.html#L31) | Title A-Z | Titel A–Z |
| Filter-Bar Submit-Btn | [library.html:33](../templates/library.html#L33) | Search | Suchen |
| Type-Badge document_to_markdown | [library.html:44](../templates/library.html#L44) | Document | Dokument |
| Type-Badge audio_transcription | [library.html:45](../templates/library.html#L45) | Audio | Audio |
| Type-Badge dialogue_formatting | [library.html:46](../templates/library.html#L46) | Dialogue | Dialog |
| Type-Badge markdown_input | [library.html:47](../templates/library.html#L47) | Markdown | Markdown |
| Favorite-Btn title | [library.html:50](../templates/library.html#L50) | Toggle favorite | Favorit umschalten |
| Toolbar-Btn Copy | [library.html:75](../templates/library.html#L75) | Copy | Kopieren |
| Toolbar-Btn Copy title | [library.html:75](../templates/library.html#L75) | Copy content | Inhalt kopieren |
| Toolbar-Btn Delete | [library.html:76](../templates/library.html#L76) | Delete | Löschen |
| Toolbar-Btn Delete title | [library.html:76](../templates/library.html#L76) | Delete | Eintrag löschen |
| Pagination Prev | [library.html:88](../templates/library.html#L88) | Prev | Zurück |
| Pagination Next | [library.html:101](../templates/library.html#L101) | Next | Weiter |
| Empty-State Heading (in P7 ersetzt durch filter-aware Variante) | [library.html:108](../templates/library.html#L108) | No saved conversions yet | Noch keine gespeicherten Einträge |
| Empty-State Body (in P7 ersetzt) | [library.html:109](../templates/library.html#L109) | Use "Save to Library" on any converter page to start building your library. | Über „In Library speichern" auf einer Konverter-Seite beginnen. |
| Page-Title (Tab) | [library.html:3](../templates/library.html#L3) | Library | Library |
| Copy-Toast Success | [library.js:21](../static/js/library.js#L21) | Content copied to clipboard | Inhalt kopiert |
| Copy-Toast Failure (Level via P6) | [library.js:23](../static/js/library.js#L23) | Copy failed | Kopieren fehlgeschlagen |
| Confirm-Dialog Delete (in P4 mit-bearbeitet) | [library.js:28](../static/js/library.js#L28) | Delete this conversion? This cannot be undone. | Diesen Eintrag wirklich löschen? Das kann nicht rückgängig gemacht werden. |

- **Helper-Reuse:** Keine. Reine String-Änderung in [templates/library.html](../templates/library.html) und [static/js/library.js](../static/js/library.js). Cross-Feature-Konvergenz auf F-1 Cluster Polish-1 (DE-Microcopy-Konvention), F-2.3 P12, F-3.3 P6, F-5.3 (DE-Konvention durchgehend).
- **Aufwand:** S — Volumen ist hoch (~30 String-Touches inkl. Tooltips), aber jede Stelle ist 1:1-Replace. Template + ein JS-Modul. Sweep sinnvoll mit P3/P4/P6 zusammen, weil dieselben Stellen ohnehin angefasst werden. F-3.3 P6-Aufwand (S) 1:1 übernommen — identisches Volumen-Profil.
- **Impact-Score:** 3 × 5 / 2 = **7.5**

---

### Pattern 6: Toast-Level pro Call-Site korrekt setzen
**Adressiert Findings:** F7 (H4 Sev 2)
**Adressiert Bug-Ticket:** BT4 (Toast-Level-Default `success` greift im Copy-Failure-Pfad)
**F-3-Korrespondenz:** F-3.3 P8 (Toast-Level pro Call-Site — direkt übernommen)
**Cluster:** 1 (Cross-Feature-H4-Helper-Reuse zu F-3)
**🔥 Smoke-Pflicht in F6-IMPL** (code-evident verifiziert im Container)

- **Pattern:** Aus F-3.3 P8 übernommen — der einzige Toast-Failure-Pfad in `library.js` ist der Copy-Failure ([static/js/library.js:23](../static/js/library.js#L23)). `showToast('Copy failed')` ohne Options bekommt den Default-Level `'success'` (grüner Toast für Fehler). Pattern-Mechanik: pro Call-Site das passende Level setzen — `showToast('Kopieren fehlgeschlagen', { level: 'danger' })` für den Catch-Branch von `copyContent`. Erfolgs-Toast bleibt beim Default-Level `success`. Damit korreliert visuelle Tönung (grün/rot) mit dem semantischen Inhalt des Strings.
- **Visuelle Hinweise:** Keine neuen Komponenten. Toast-Tint-Logik ist schon im CSS via `.toast-notification--danger` / `--warning` / `--success` definiert (siehe [static/js/_utils.js:101](../static/js/_utils.js#L101)).
- **Microcopy** (DE — siehe P5 Konstanten-Tabelle):
  - Copy-Failure (mit `level: 'danger'`): „Kopieren fehlgeschlagen"
  - Copy-Success behält Default-Level `success`: „Inhalt kopiert"
- **Helper-Reuse:** `showToast(msg, { level: 'danger' })` aus [static/js/_utils.js:92-117](../static/js/_utils.js#L92-L117) — Helper unterstützt das bereits, in `library.js` wird es heute nur nicht genutzt. Inline-Code-Anker: [static/js/library.js:23](../static/js/library.js#L23) Copy-Failure.
- **Aufwand:** XS — eine Options-Object-Ergänzung an einer Call-Site. F-3.3 P8-Aufwand (XS) 1:1 übernommen — auf List-View ist nur eine Stelle betroffen (im Gegensatz zu Detail-View, wo F-3.3 P8 mehrere Toast-Pfade sweepte, von denen die meisten durch P7 zu Bannern wurden — auf List-View entfällt das, weil keine Notion-Submit-Toasts existieren).
- **Impact-Score:** 2 × 5 / 1 = **10.0**
- **Smoke-Mechanik (code-evident im Container):** Code-Reading der Toast-Call-Site: `showToast('Kopieren fehlgeschlagen', { level: 'danger' })`. Bei Bedarf zusätzlich Live-Trigger durch DevTools-Clipboard-Permission-Block — aber Master-Annotation 3 stuft das nicht als Pflicht-Live-Master-Smoke ein, weil Toast-Level-Mechanik trivial verifizierbar.

---

### Pattern 7: Empty-State filter-aware mit Recovery-Pfad
**Adressiert Findings:** F8 (H9 Sev 2)
**F-3-Korrespondenz:** — (list-spezifisch — Detail-View hat keinen Empty-State-Pfad)
**Cluster:** 2 (List-View-State-Visibility und Empty-State-Recovery)
**Live-Verifikation-Status:** — (Empty-State-Branch ist live-evident bei Filter-Trigger)
**Master-Default-Wahl** (Master-Annotation 5)

- **Pattern:** Heute zeigt [templates/library.html:106-110](../templates/library.html#L106-L110) den selben Empty-State („No saved conversions yet" + „Use 'Save to Library' …") egal ob die Library leer ist oder ob nur der aktive Filter keine Treffer liefert. Pattern-Mechanik: server-side im `library_view`-Render in [app_pkg/library.py:30-72](../app_pkg/library.py#L30-L72) eine Boolean ergänzen, die unterscheidet, ob die Empty-Liste durch Filter zustande kommt — `has_active_filter = bool(conversion_type or search or favorites)` — und in den Template-Context geben. Im Template wird der Empty-State-Block dann zweistufig: bei `has_active_filter=True` zeigt er die Filter-Mismatch-Variante mit Recovery-Pfad „Filter zurücksetzen" (Link auf `/library` ohne Query-Params); bei `has_active_filter=False` (keine Filter aktiv und Library wirklich leer) zeigt er die ursprüngliche Variante mit Hinweis auf den Save-Pfad. **Master-Default-Wahl**: server-side im Render entscheiden (statt JS-Branch), weil der Render-Context die Filter-Booleans ohnehin schon trägt — kein zusätzlicher Roundtrip, kein JS-Code nötig. **Konstitutiv-Hinweis (Bedenken)**: die Boolean lässt sich aus den `current_*`-Context-Variables ableiten, also kein neuer Backend-Pfad — F6-IMPL muss nur die Boolean berechnen und im Template-Branch nutzen.
- **Visuelle Hinweise:** Empty-State-Block bleibt zentriert im Page-Body (analog heute). Filter-zurücksetzen-Link ist als `c-btn`-Sekundär (nicht-Primary, nicht-Danger) gestaltet — gleiche Styling-Klasse wie Pagination-Buttons. aria-Hint via dezenter Untertitel-Microcopy.
- **Microcopy** (DE, Du-Form, max 3 Sätze für Empty-State; Buttons max 3 Wörter):
  - **Variante A — Filter aktiv, keine Treffer**:
    - Heading: „Keine Treffer mit aktuellen Filtern"
    - Body: „Mit den aktiven Filtern ist nichts gespeichert. Filter zurücksetzen, um die gesamte Library zu zeigen."
    - Btn: „Filter zurücksetzen" (Link auf `/library`)
  - **Variante B — Library wirklich leer**:
    - Heading: „Noch keine gespeicherten Einträge"
    - Body: „Über „In Library speichern" auf einer Konverter-Seite beginnen."
    - (Kein Btn — Link über die Sidebar-Navigation reicht)
- **Helper-Reuse:** Keine. Reine Server-Context-Erweiterung + Template-Branch. Code-Anker: [app_pkg/library.py:33-72](../app_pkg/library.py#L33-L72) `library`-View für Boolean-Berechnung; [templates/library.html:106-110](../templates/library.html#L106-L110) Empty-State-Block für Branch.
- **Aufwand:** S — Backend-Touch (eine Zeile Boolean-Berechnung + Context-Erweiterung) + Template-Branch (Jinja2 if/else mit zwei Microcopy-Varianten + Btn-Link) + DE-Microcopy. Kein neuer Helper, kein neues CSS, kein Schema-Touch.
- **Impact-Score:** 2 × 5 / 2 = **5.0**

---

### Pattern 8: Search-Input Submit-Required mit DE-Hint (statt Live-Debouncing)
**Adressiert Findings:** F9 (H4 Sev 2), F11 (H6 Sev 1)
**F-3-Korrespondenz:** — (list-spezifisch — Detail-View hat keinen Search-Pfad)
**Cluster:** 2 (List-View-State-Visibility und Empty-State-Recovery)
**🔥 Smoke-Pflicht in F6-IMPL** (code-evident verifiziert im Container)
**Master-Default-Wahl** (Master-Annotation 5)

- **Pattern:** F9 (Search-Input nicht auto-submit, anders als die anderen Filter) und F11 (Search-Input ohne Live-Search/Debouncing) sind beide Search-Input-Verhalten-Findings, mit unterschiedlichen Heuristiken (H4 Konsistenz vs. H6 Recognition), aber zu **derselben Mechanik-Frage** verbindbar: wann triggert die Search-Query? **Master-Default-Wahl**: **Submit-Required mit DE-Hint** statt Live-Debouncing. Begründung: der gesamte View-State der List-View (Sortierung / Filter / Favorites / Suche / Pagination) wird über URL-Query-Params persistiert (siehe Cross-Feature-H4-Sektion, Helper-Reuse-Reflexion). Live-Search ohne URL-Update wäre State-Drift zwischen Query-Result und Browser-URL; mit URL-Update wäre History-Pollution bei jedem Tastendruck (jede Tastatur-Eingabe würde einen URL-Change auslösen). Submit-Required ist konsistent mit der URL-Query-Param-Mechanik der anderen Filter (auto-submit funktioniert dort, weil ein einzelner Toggle-/Select-Change ein wohldefinierter URL-Change-Event ist; bei Search ist jedes Tastendruck-Event kein URL-Change-Wert). Pattern-Mechanik: Search-Input bekommt einen `placeholder` mit explizitem Hinweis („Titel, Inhalt, Tags suchen … (Enter)") und der Submit-Btn bleibt sichtbar (heute schon, [library.html:33](../templates/library.html#L33)). Damit wird die Konsistenz-Lücke aus F9 zu einer expliziten Konvention („Search ist explizit-submit, weil URL-State-konsistent") und F11 wird strukturell adressiert (kein Live-Debouncing-Aufwand nötig). Internal H4-Bruch zwischen Search-Input und den anderen Filtern wird damit nicht beseitigt, sondern als bewusste Design-Wahl mit Microcopy-Markierung kommuniziert — analog F-3.3-Pattern „Helper-Reuse-Drift mit begründeter Design-Wahl".
- **Visuelle Hinweise:** Search-Submit-Btn („Suchen" aus P5 DE-Sweep) bleibt sichtbar in der Filter-Bar — kein Display-Switch, kein Hide. Search-Input bekommt einen erweiterten Placeholder.
- **Microcopy** (DE, Du-Form):
  - Search-Input placeholder (ersetzt heutiges „Search title, content, tags..."): „Titel, Inhalt, Tags suchen … (Enter)"
  - Search-Submit-Btn (aus P5 DE-Sweep): „Suchen"
- **Helper-Reuse:** Keine. Reine String-Änderung. Code-Anker: [templates/library.html:21](../templates/library.html#L21) Search-Input-Placeholder; [templates/library.html:33](../templates/library.html#L33) Search-Submit-Btn (bereits vorhanden, kein Touch nötig).
- **Aufwand:** XS — eine Placeholder-String-Änderung + Validation, dass der Submit-Btn sichtbar bleibt. Kein JS-Touch, kein neues CSS, kein Backend-Touch.
- **Impact-Score:** 2 × 5 / 1 = **10.0** (höchste Sev der adressierten Findings: F9 mit Sev 2)
- **Smoke-Mechanik (code-evident im Container):** Code-Reading der Filter-Bar-Mechanik: andere Filter bleiben auto-submit (`onchange="this.form.submit()"`), Search-Input bleibt explizit-submit. Bei Bedarf zusätzlich Live-Browser-Test: Tippen ohne Enter → keine Filter-Änderung; Enter → Suche triggert; Submit-Btn-Klick → Suche triggert. **Kein Pflicht-Live-Master-Smoke** — Mechanik ist trivial verifizierbar.
- **Konsolidierung:** F9 (H4) und F11 (H6) gehören zur selben Mechanik-Frage (Search-Trigger). Eine Lösung (Submit-Required mit explizitem DE-Hint) erfüllt beide: die Konsistenz-Lücke aus F9 wird als bewusste Design-Wahl markiert (Microcopy-Hinweis); die Recognition-Lücke aus F11 wird durch den expliziten Hint im Placeholder addressed (User sieht „Enter" und weiß, dass die Eingabe nicht live wirkt). F8 (Empty-State) bleibt als eigenständiges P7 — andere Mechanik (server-side Render-Branch statt Input-Verhalten).

---

### Pattern 9: `favorites=''`-URL-Artifact in Pagination-Links
**Adressiert Findings:** F12 (H4 Sev 1)
**F-3-Korrespondenz:** — (list-spezifisch — Detail-View hat keine Pagination)
**Cluster:** 3 (List-Polish-Long-Tail)
**Live-Verifikation-Status:** — (URL-Output ist statisch verifizierbar im Browser-Address-Bar)

- **Pattern:** Heute produzieren die Pagination-Links in [templates/library.html:87,92,100](../templates/library.html#L87) bei deaktivierten Favorites einen leeren Query-Param `?...&favorites=&...` in der URL. Funktional unkritisch (`request.args.get('favorites', '') == '1'` → False), URL ist hässlich/uneinheitlich. Pattern-Mechanik: die Jinja2-Inline-Bedingung `favorites='1' if current_favorites else ''` durch eine Konstrukt umbauen, das den Param **ganz weglässt**, wenn er nicht aktiv ist. **Master-Default-Wahl**: das `url_for()` mit Kwargs-Dictionary aufrufen, das nur die aktiv-gesetzten Filter enthält. Beispiel-Skizze: ein Macro oder eine Inline-`{% set %}`-Konstruktion, die die Kwargs-Dict konditional aufbaut und dann `{{ url_for('library', **kwargs) }}` aufruft. Alternativ: eine Hilfsfunktion `library_url(page, **filters)` im Template-Context, die `None`-Werte filtert. F6-IMPL kann pragmatisch zwischen den beiden entscheiden — die Hilfsfunktion ist DRYer (drei Pagination-Links statt drei `{% set %}`-Blöcke), die Inline-Variante kommt ohne Backend-Touch aus.
- **Visuelle Hinweise:** Keine — reine URL-Output-Änderung. User merkt es nur in der Adressleiste / beim URL-Sharing.
- **Microcopy** (DE, Du-Form):
  - Keine — reine URL-Mechanik.
- **Helper-Reuse:** Keine `_utils.js`-Helper. Optional ein Server-side Template-Helper (Jinja2-Custom-Function) im App-Factory analog `file_size`-Filter aus F-3-IMPL. Code-Anker: [templates/library.html:87,92,100](../templates/library.html#L87) Pagination-Links.
- **Aufwand:** XS — drei `url_for`-Aufrufe umbauen (entweder via Inline-`{% set %}` oder via neue Hilfsfunktion). Bei Hilfsfunktion plus Server-Touch: XS+, immer noch im XS-Bucket.
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 10: Type-Filter Backend-Validation
**Adressiert Findings:** F13 (H9 Sev 1)
**F-3-Korrespondenz:** — (list-spezifisch — Detail-View hat keinen Type-Filter)
**Cluster:** 3 (List-Polish-Long-Tail)
**Live-Verifikation-Status:** — (Backend-Branch ist statisch verifizierbar)

- **Pattern:** Heute akzeptiert [app_pkg/library.py:43-44](../app_pkg/library.py#L43-L44) jeden String aus dem URL-Query (`?type=nonsense`) und filtert die DB-Query mit dem unbekannten Type-String → leere Liste. `ALLOWED_CONVERSION_TYPES`-Set ist bereits definiert, wird im POST-Pfad ([app_pkg/library.py:91](../app_pkg/library.py#L91)) für Validation genutzt, im GET-Filter-Pfad aber nicht. Pattern-Mechanik: `if conversion_type and conversion_type in ALLOWED_CONVERSION_TYPES:` als Filter-Branch. Bei unbekanntem Type-Wert wird der Filter still ignoriert (Fallback auf „All Types" — analog dem Default-State der Filter-Bar). Kein 400-Error, weil URL-Query-Tippfehler keine Backend-Fehler-Eskalation rechtfertigen — User bekommt einfach die ungefilterte Liste. **Verzahnung-Hinweis mit P7**: bei `has_active_filter`-Berechnung in P7 sollte der gefilterte Type-Wert nur dann als „aktiv" zählen, wenn er in `ALLOWED_CONVERSION_TYPES` ist (sonst irreführender Empty-State).
- **Visuelle Hinweise:** Keine — reine Backend-Branch-Hardening.
- **Microcopy** (DE, Du-Form):
  - Keine — reine Backend-Mechanik. (Falls F6-IMPL einen User-sichtbaren Hint ergänzen will: „Filter ignoriert" als info-Banner — **nicht im Default**, weil URL-Tippfehler ein Edge-Case ist und der unsichtbare Fallback der bessere UX-Default ist.)
- **Helper-Reuse:** Keine. Code-Anker: [app_pkg/library.py:43-44](../app_pkg/library.py#L43-L44) GET-Filter-Pfad; [app_pkg/library.py:11-16](../app_pkg/library.py#L11-L16) `ALLOWED_CONVERSION_TYPES`-Set (bereits da).
- **Aufwand:** XS — eine Set-Membership-Check-Zeile.
- **Impact-Score:** 1 × 5 / 1 = **5.0**

---

### Pattern 11: Card-Datum DE-Lokalisierung statt `%b`-Locale-Default
**Adressiert Findings:** F14 (H4 Sev 1)
**F-3-Korrespondenz:** F-3.3 P5 (teil — andere Mechanik)
**Cluster:** 3 (List-Polish-Long-Tail) + Cross-Feature-H4 (DE-Welle)
**🔥 Smoke-Pflicht in F6-IMPL** — **🔥🔥 Pflicht-Live-Master-Smoke**

- **Pattern:** Heute nutzt [templates/library.html:63](../templates/library.html#L63) `conv.created_at.strftime('%d %b %Y, %H:%M')` — `%b` liefert die abgekürzte Monatsbezeichnung in der Container-Default-Locale (vermutlich `C` oder `C.UTF-8`, also EN „May" statt DE „Mai"). **Andere Mechanik als F-3.3 P5** (P5 war JS-`<input type="datetime-local">`-Pre-Population mit `formatDatetimeLocalNow`-Helper — hier ist es Server-side `strftime` im Card-Render-Pfad). Pattern-Mechanik: einen kleinen DE-Monatsnamen-Map-Helper im Server-Template-Context bereitstellen (analog dem `file_size`-Jinja-Filter aus F-3-IMPL) — z.B. `format_card_datetime(dt)`, der einen DE-formatierten String liefert wie „10 Mai 2026, 14:30" oder „10. Mai 2026, 14:30". **Master-Default-Wahl**: kompletter Custom-Filter im App-Factory (analog `file_size`), nicht via System-Locale-Setzung (Locale-Setting im Container-Image hätte System-weite Nebenwirkungen und ist fragiler). Ein dezidierter Filter ist explizit, testbar, container-locale-agnostisch.
- **Visuelle Hinweise:** Card-Datum bleibt an der gleichen Position. Format-Wechsel ist visuell minimal (nur Monatsname). Tooltip via `title="{{ conv.created_at.strftime('%Y-%m-%d %H:%M') }}"` bleibt unverändert (technisches Datum für genaue Inspektion).
- **Microcopy** (DE — Format-Beispiele):
  - „10 Mai 2026, 14:30"
  - „01 Jan 2026, 09:05"
  - „30 Dez 2025, 23:59"
  - DE-Monatsnamen-Map (Filter-intern): Jan / Feb / Mär / Apr / Mai / Jun / Jul / Aug / Sep / Okt / Nov / Dez
- **Helper-Reuse:** Server-side Custom Jinja2-Filter im App-Factory analog `file_size` aus F-3-IMPL. Inline-Code-Anker: [templates/library.html:63](../templates/library.html#L63) Card-Datum-Anzeige; [app_pkg/__init__.py](../app_pkg/__init__.py) Filter-Registrierung (nähe `file_size`-Filter).
- **Aufwand:** S — Filter-Funktion (~10 Zeilen Python mit DE-Monatsnamen-Map) + Filter-Registrierung im App-Factory + Template-Stelle anpassen + ggf. Test der Filter-Beispiele. F-3.3 P5-Aufwand (XS) ist hier **nicht direkt übertragbar**, weil die Mechanik anders ist (Server-side Filter statt JS-Helper).
- **Impact-Score:** 1 × 5 / 2 = **2.5**
- **Smoke-Mechanik:** Card-Datum-Anzeige im Browser inspizieren — Erwartung „Mai" statt „May" für 10. Mai-Konvertierungen. Alternativ: Container-`docker exec ... locale` zur Verifikation, dass die Filter-Mechanik unabhängig vom Container-Locale funktioniert.

---

### Pattern 12: Per-Page-Size-Toggle für wachsende Library
**Adressiert Findings:** F15 (H6 Sev 1)
**F-3-Korrespondenz:** — (list-spezifisch — Detail-View hat keine Pagination)
**Cluster:** 3 (List-Polish-Long-Tail)
**Live-Verifikation-Status:** — (UI-Toggle ist live-evident bei Test)

- **Pattern:** Heute ist [app_pkg/library.py:39](../app_pkg/library.py#L39) `per_page=20` hardcoded ohne UI-Toggle. Bei wachsender Library (Reader-Replacement-Skalierung Richtung mehrere hundert/tausend Einträge) wird das ein Pagination-Klick-Marathon. Pattern-Mechanik: `per_page` aus URL-Query-Param lesen mit Whitelist `{20, 50, 100}` (Default 20, alles außerhalb fällt auf Default zurück — gleiche Validation-Mechanik wie P10). UI: ein zusätzliches `<select name="per_page">` in der Filter-Bar mit den drei Optionen, auto-submit (analog Sort-Select). DE-Microcopy: „20 pro Seite" / „50 pro Seite" / „100 pro Seite". **Konstitutiv-Hinweis**: heute ist die Library noch klein (Sev 1, weil aktuell nicht akut spürbar) — Pattern ist primär eine Skalierungs-Vorbereitung. F6-IMPL kann das pragmatisch entscheiden, ob es jetzt mit-implementiert wird oder als Folde-in-späterer-Welle dispositioniert.
- **Visuelle Hinweise:** Neuer `<select>` in der Filter-Bar, gleiche Position wie Sort-Select (rechts neben Favorites-Checkbox). Auto-Submit beim Change.
- **Microcopy** (DE, Du-Form, max 3 Wörter pro Option):
  - Per-Page-Select title: „Pro Seite"
  - Option-Werte: „20 pro Seite" / „50 pro Seite" / „100 pro Seite"
- **Helper-Reuse:** Keine. Server-side Read aus `request.args.get('per_page', 20, type=int)` mit Whitelist-Check. Code-Anker: [app_pkg/library.py:39](../app_pkg/library.py#L39) `per_page`-Setzung; [templates/library.html:9-34](../templates/library.html#L9-L34) Filter-Bar für UI-Insert; [templates/library.html:87,92,100](../templates/library.html#L87) Pagination-Links für Param-Pass-Through (verzahnt mit P9).
- **Aufwand:** M — Backend-Read-Branch + Whitelist-Validation + Filter-Bar-UI-Insert + Pagination-Links-Param-Pass-Through (verzahnt mit P9 — beide Patterns berühren `url_for('library', …)`-Calls in den Pagination-Links). Kein neuer Helper, kein Schema-Touch.
- **Impact-Score:** 1 × 5 / 4 = **1.25**

---

### Pattern 13: aria-live-Region für Card-Remove
**Adressiert Findings:** F16 (H1 Sev 1)
**F-3-Korrespondenz:** — (list-spezifisch — F-3 hat aria-live für Notion-Target-Switches, nicht für Card-Remove)
**Cluster:** 3 (List-Polish-Long-Tail) + Verzahnung mit P4 Delete-Pfad
**🔥 Smoke-Pflicht in F6-IMPL** (code-evident verifiziert im Container)

- **Pattern:** Heute entfernt [static/js/library.js:30-36](../static/js/library.js#L30-L36) die Card-Animation visuell (`opacity=0` + scale-Transform + nach 200 ms `card.remove()`), aber Screenreader bekommt keinen Hinweis. Pattern-Mechanik: eine `<div id="library-action-status" aria-live="polite" class="sr-only">`-Region im Template einführen (analog F-3-IMPL `#notion-target-status`-Region), und im Erfolgs-Branch von `deleteConversion` (siehe P4) den Text setzen: „Eintrag gelöscht."  Region wird nach dem Card-Remove geleert (oder bleibt mit dem Status-Text — Screenreader liest beim Setzen, leeres Setzen ist optional). **Verzahnung mit P4**: gleicher Code-Pfad — beide Patterns berühren den Erfolgs-Branch von `deleteConversion`. F6-IMPL kann sie in einem Sub-Batch zusammenziehen.
- **Visuelle Hinweise:** Keine — `.sr-only`-Region ist visuell unsichtbar, nur für Screenreader.
- **Microcopy** (DE, Du-Form):
  - Card-Remove-Status-Text (aria-live `polite`): „Eintrag gelöscht."
  - Optional bei P4-404-Race-Pfad: „Eintrag bereits entfernt." (gemeinsam mit info-Banner aus P4)
- **Helper-Reuse:** Keine. `.sr-only`-Klasse aus [static/css/style.css](../static/css/style.css) bereits vorhanden. Code-Anker: [templates/library.html](../templates/library.html) für Region-Insert (z.B. nach `#library-alert-container`); [static/js/library.js:30-36](../static/js/library.js#L30-L36) Erfolgs-Branch von `deleteConversion` für Status-Text-Setzung.
- **Aufwand:** XS — eine `<div>`-Region im Template + zwei Zeilen JS im Erfolgs-Branch (Status-Text setzen + ggf. nach Card-Remove leeren).
- **Impact-Score:** 1 × 5 / 1 = **5.0**
- **Smoke-Mechanik (code-evident im Container):** Code-Reading der DOM-Annotations: Region ist `aria-live="polite"`, Status-Text wird im Erfolgs-Branch gesetzt. Bei Bedarf zusätzlich Screenreader-Test (VoiceOver / NVDA) — aber nicht Pflicht-Live-Master-Smoke.

---

### Pattern 14: Card-Hover-Lift Affordance-Konflikt entschärfen
**Adressiert Findings:** F17 (H6 Sev 1)
**F-3-Korrespondenz:** — (list-spezifisch — Detail-View hat keine Card-Hover-Mechanik)
**Cluster:** 3 (List-Polish-Long-Tail)
**🔥 Smoke-Pflicht in F6-IMPL** (code-evident verifiziert im Container)

- **Pattern:** Heute läuft [static/css/style.css:242-245](../static/css/style.css#L242-L245) `.c-card:hover { translateY(-2px) }` während die Maus die Card überquert; Sub-Element-Buttons (Favorite, Card-Link, Copy, Delete) liegen alle innerhalb der hover-bewegten Card. Affordance-Konflikt: Hover-Lift signalisiert Click-Affordance auf der gesamten Card, gleichzeitig sind die Sub-Element-Buttons primary Targets (mit eigener Hover-Affordance). Pattern-Mechanik (mehrere Optionen, F6-IMPL wählt pragmatisch beim Apply):
  - **Option A (Master-Default)**: Hover-Lift behalten, aber `pointer-events`-Ordering / `position: relative; z-index`-Hierarchie sicherstellen, damit Sub-Element-Klicks robust hit-getargetet werden. Visuelles Flackern bleibt minimal-akzeptabel, wenn der Lift-Übergang flüssig ist (`transition: transform 150ms ease`).
  - **Option B**: Hover-Lift entfernen — minimaler CSS-Touch (`translateY` raus), aber visueller Reichtum der Liste sinkt. Sub-Element-Affordance dominiert dann uneingeschränkt.
  - **Option C**: Hover-Lift nur auf den Card-Link-Bereich beschränken (statt der ganzen Card) — saubererer Affordance-Match, aber mehr CSS-Refactor (Layer-Klassen-Aufteilung).
  - **F6-IMPL wählt pragmatisch beim Apply**, basierend auf einem kurzen Hover-Test im Browser. Default-Vorschlag: Option A (minimal-invasiv) plus `transition`-Glättung wenn nicht schon da.
- **Visuelle Hinweise:** Bei Option A keine sichtbaren Layout-Änderungen. Bei Option B verschwindet die `translateY`-Lift-Animation. Bei Option C ist die Lift-Animation auf den Card-Link-Bereich beschränkt (Toolbar-Buttons unten bleiben statisch).
- **Microcopy** (DE, Du-Form):
  - Keine — reine CSS-/Layer-Mechanik.
- **Helper-Reuse:** Keine. Reine CSS-Modifikation. Code-Anker: [static/css/style.css:242-245](../static/css/style.css#L242-L245) `.c-card:hover`-Block; [templates/library.html:41-79](../templates/library.html#L41-L79) Card-Struktur für Layer-Inspection.
- **Aufwand:** XS — bei Option A oder B ein-Zeilen-CSS-Touch. Bei Option C: S (Layer-Refactor mit neuer CSS-Klasse für den Lift-Bereich).
- **Impact-Score:** 1 × 5 / 1 = **5.0** (für Default-Option A)
- **Smoke-Mechanik (code-evident im Container):** Code-Reading der CSS-Klassen + Card-Struktur. Bei Bedarf Live-Browser-Test mit schnellem Maus-Wechsel zwischen Cards (Hover-Robustheit) + DevTools-Inspect der z-index-Hierarchie. **Kein Pflicht-Live-Master-Smoke** — Polish-Befund mit Live-Walkthrough-Lücke; Master-Annotation 3 stuft das nicht als Pflicht-Live-Master ein.

---

## Cluster-Vorschlag für F6-IMPL

**Default-Empfehlung — 1-Sprint mit 3 Sub-Batches** (analog F-5-IMPL Sub-Batch-Schnitt):

### Sub-Batch A — Silent-Failure-Familie + Cross-Feature-H4-Welle (Cluster 1)

Patterns: **P1 (Banner-Mountpoint), P2 (Copy-Full-Content), P3 (Auto-Save-Banner), P4 (Delete-Banner), P5 (DE-Microcopy), P6 (Toast-Level)**

Begründung Gruppierung: Cluster 1 konsolidiert die Helper-/Microcopy-/Container-Refactors, die für den Banner-Pfad und den DE-Microcopy-Pass gemeinsam sind. P1 ist strukturelle Voraussetzung für P3 + P4 — muss am Anfang. P2 (Copy-Full-Content) hat eigene Mechanik (Template-Touch für `data-content`-Attribut), aber gehört in den Sub-Batch wegen identischem Daily-Usage-Hotspot-Cluster und Smoke-Pflicht-Bündel. P5 (DE-Microcopy) ist die Sweep-Stelle, an der ohnehin viele Strings angefasst werden — sinnvoll, sie zusammen mit den Banner- und Toast-Patches zu integrieren, damit die neue Banner- und Toast-Microcopy aus P3/P4/P6 gleich in DE landet. P6 (Toast-Level) ist XS, gut mit der Sweep zusammen.

**Smoke-Pflicht-Sub-Gruppe** (P2, P3, P4, P6 — vier Patterns mit `🔥 Smoke-Pflicht in F6-IMPL`-Tag, davon P2 + P3 als `🔥🔥 Pflicht-Live-Master-Smoke`): vor Apply per Live-Smoke verifizieren — Copy-Paste-Test mit langem Content (P2), DevTools-Network-Throttle PUT/DELETE auf 5xx (P3, P4-code-evident), Toast-Level-Code-Reading (P6).

### Sub-Batch B — List-View-State-Visibility (Cluster 2)

Patterns: **P7 (Empty-State filter-aware), P8 (Search Submit-Required mit DE-Hint)**

Begründung Gruppierung: zwei Patterns mit Master-Default-Wahl-Marker (Master-Annotation 5). Beide adressieren List-View-State-Visibility — Empty-State-Recovery (P7) und Search-Input-UX-Konsistenz (P8). Beide haben keinen F-3-Korrespondenten. P7 ist S (Backend-Touch + Template-Branch + Microcopy), P8 ist XS (Placeholder-Microcopy). Sub-Batch ist klein, kann pragmatisch in einem Commit landen.

### Sub-Batch C — List-Polish-Long-Tail (Cluster 3)

Patterns: **P9 (favorites='' URL), P10 (Type-Filter-Validation), P11 (Card-Datum DE-Lokalisierung), P12 (Per-Page-Size-Toggle), P13 (aria-live), P14 (Card-Hover-Lift)**

Begründung Gruppierung: sechs Sev-1-Patterns ohne Daily-Usage-Akut-Trigger. Reihenfolge frei wählbar. P9 + P12 berühren beide die Pagination-Links (`url_for('library', …)`) — sinnvoll als Pärchen. P11 (Card-Datum) ist S wegen Server-side Custom-Filter, ist aber ein eigenständiger Code-Touch. P13 (aria-live) verzahnt mit P4 (Delete-Pfad) — kann pragmatisch in Sub-Batch A mit-genommen werden, wenn das `deleteConversion`-Modul ohnehin angefasst wird (Synergie-Hinweis). P14 (Card-Hover-Lift) ist optional wenn F6-IMPL Sub-Batch C als zu groß empfindet — könnte in eine spätere Polish-Welle ausgelagert werden.

**Pragmatischer Synergie-Hinweis P13 → Sub-Batch A**: P13 (aria-live für Card-Remove) ist eng verzahnt mit P4 (Delete-Failure-Banner) — derselbe Erfolgs-Branch von `deleteConversion` wird angefasst. Wenn F6-IMPL Sub-Batch A schreibt, kann P13 mitgenommen werden (XS-Aufwand, Synergie). Verbleibender Sub-Batch C reduziert sich dann auf 5 Patterns.

### Zwei-Cluster-Empfehlung (falls Cluster 1 als zu groß empfunden wird)

6 Patterns + 1 Microcopy-Sweep in Sub-Batch A ist im oberen Drittel der bisherigen Sub-Batch-Größen (F-3-IMPL Sub-Batch A hatte 4, F-5-IMPL Cluster Ia hatte 4). Wenn der F6-IMPL-Sub-Thread Sub-Batch A als zu unhandlich empfindet, sinnvoller Split:

- **Sub-Batch A1 (Foundation Sweep):** P1 (Mountpoint), P5 (DE-Microcopy), P6 (Toast-Level) — strukturelle Container + DE-Microcopy + Toast-Level. Mechanisch, weitgehend statisch verifizierbar. Ein Commit.
- **Sub-Batch A2 (Silent-Failure-Banner + Copy-Full-Content):** P2, P3, P4 — drei Patterns mit Smoke-Pflicht (P2 + P3 als Pflicht-Live-Master-Smoke). Eigener Commit.

Sub-Batches B und C bleiben unverändert.

---

## Top-5 Quick-Wins

**Aufwand-Gewicht:** XS=1, S=2, M=4, L=8. Score = Sev × 5 / Aufwand-Gewicht. Höher = besser.

| Rang | Pattern # | Adressiert | Sev | Aufwand | Impact-Score | Quick-Win |
|------|-----------|------------|-----|---------|--------------|-----------|
| 1 | P6 | F7 — Toast-Level pro Call-Site | 2 | XS | 10.0 | ★ Top-5 |
| 2 | P8 | F9, F11 — Search Submit-Required mit DE-Hint | 2 | XS | 10.0 | ★ Top-5 |
| 3 | P2 | F1 — Copy-Full-Content (BT3) | 3 | S | 7.5 | ★ Top-5 |
| 4 | P3 | F2, F3 — Auto-Save Failure-Banner (BT1) | 3 | S | 7.5 | ★ Top-5 |
| 5 | P4 | F4, F5 — Delete Failure-Banner (BT2) | 3 | S | 7.5 | ★ Top-5 |
| 6 | P5 | F6 — DE-Microcopy-Sweep | 3 | S | 7.5 | |
| 7 | P1 | F10 — Banner-Mountpoint-Container | 1 | XS | 5.0 | (Vorbedingung für P3/P4) |
| 8 | P7 | F8 — Empty-State filter-aware | 2 | S | 5.0 | |
| 9 | P9 | F12 — favorites='' URL | 1 | XS | 5.0 | |
| 10 | P10 | F13 — Type-Filter-Validation | 1 | XS | 5.0 | |
| 11 | P13 | F16 — aria-live Card-Remove | 1 | XS | 5.0 | |
| 12 | P14 | F17 — Card-Hover-Lift | 1 | XS | 5.0 | |
| 13 | P11 | F14 — Card-Datum DE-Lokalisierung | 1 | S | 2.5 | |
| 14 | P12 | F15 — Per-Page-Size-Toggle | 1 | M | 1.25 | |

**Top-5 Quick-Wins:**

1. **P6 — Toast-Level pro Call-Site** (10.0): eine Options-Object-Ergänzung an einer Call-Site, beseitigt einen visuellen Mismatch (grüner Toast mit „Copy failed"). F-3.3 P8-Übernahme — identische Mechanik wie auf Detail-View, nur eine Stelle in `library.js`. Cluster 1, Smoke-Pflicht code-evident.
2. **P8 — Search Submit-Required mit DE-Hint** (10.0): ein Placeholder-String-Touch mit DE-Hint („Enter zum Suchen"), löst zwei Findings (F9 H4 Konsistenz + F11 H6 Recognition) als bewusste Design-Wahl-Markierung. Cluster 2, Master-Default-Wahl. Konsolidierungs-Quote-Treiber.
3. **P2 — Copy-Full-Content** (7.5): Daten-Verlust-light-Beseitigung im Daily-Usage-Hotspot. Template-Touch für `data-content`-Attribut + JS-Quelle-Wechsel. BT3 mit-gelöst. Cluster 1, **Pflicht-Live-Master-Smoke**.
4. **P3 — Auto-Save Failure-Banner** (7.5): F-3.3 P1+P14-Übernahme. Höchste tägliche Trefferhäufigkeit unter den Sev-3-Pfaden (Favorite-Toggle ist häufige Reader-Action). BT1 mit-gelöst. Cluster 1, **Pflicht-Live-Master-Smoke**.
5. **P4 — Delete Failure-Banner** (7.5): F-3.3 P3+P14-Übernahme. Selbe Mechanik wie P3, anderer Pfad. Verzahnt mit P13 (aria-live). BT2 mit-gelöst. Cluster 1, Smoke-Pflicht code-evident.

P5 (DE-Microcopy-Sweep, 7.5) liegt knapp dahinter — Cluster-1-Sweep-Stelle, sinnvoll mit P3/P4/P6 zusammen, deshalb von Position 6 in den Sub-Batch-A-Verbund mit-genommen.

---

## Smoke-Pflicht-Übersicht

Patterns mit `🔥 Smoke-Pflicht in F6-IMPL`-Sub-Tag (ehe ⚠️ code-only-Findings adressiert werden, vor Apply per Live-Smoke verifizieren). **Drei Pflicht-Live-Master-Smoke** (Master-Annotation 3) zusätzlich mit `🔥🔥`-Marker:

| Pattern | Adressiert | Cluster | Smoke-Mechanik | Pflicht-Live-Master |
|---------|-----------|---------|----------------|---------------------|
| **P2** | F1 — Copy-Full-Content | 1 | Conversion mit >200 char Content → „Copy"-Btn → in Editor pasten → erwartet: Full-Content statt Preview-Slice | **🔥🔥 Ja** |
| **P3** | F2, F3 — Auto-Save Favorite-Toggle | 1 | DevTools-Network-Throttle Offline → Favorite-Btn klicken → Banner sichtbar, Glyph bleibt im alten Zustand | **🔥🔥 Ja** |
| **P4** | F4, F5 — Delete | 1 | Code-Reading: `else`-Branch + `.catch` + `safeJSON` + Microcopy. Optional Container-Test mit Mock-Fail | code-evident |
| **P6** | F7 — Toast-Level | 1 | Code-Reading der Call-Site: `{ level: 'danger' }` gesetzt. Optional DevTools-Clipboard-Permission-Block für Live-Visual | code-evident |
| **P8** | F9, F11 — Search Submit-Required | 2 | Code-Reading: andere Filter bleiben auto-submit, Search-Input bleibt explizit-submit. Optional Live-Browser: Tippen ohne Enter → keine Filter-Änderung | code-evident |
| **P11** | F14 — Card-Datum DE-Lokalisierung | 3 | Card-Datum-Anzeige im Browser inspizieren — Erwartung „Mai" statt „May" für Mai-Konvertierungen. Optional `docker exec ... locale` für Container-Locale-Verifikation | **🔥🔥 Ja** |
| **P13** | F16 — aria-live Card-Remove | 3 | Code-Reading: Region ist `aria-live="polite"`, Status-Text wird im Erfolgs-Branch gesetzt. Optional Screenreader-Test | code-evident |
| **P14** | F17 — Card-Hover-Lift | 3 | Code-Reading der CSS-Klassen + Card-Struktur. Optional Live-Browser: schneller Maus-Wechsel + DevTools-z-index-Inspect | code-evident |

**Anzahl Smoke-Pflicht-Patterns:** 8 (von 14). **Davon Pflicht-Live-Master-Smoke:** 3 (P2, P3, P11). **Code-evident verifiziert im Container:** 5 (P4, P6, P8, P13, P14). P1 (Banner-Mountpoint) trägt keinen Smoke-Tag, aber Apply der Banner-Patterns P3/P4 setzt P1 ohnehin voraus.

Cluster 3-Patterns ohne Smoke-Pflicht (P9, P10, P12) sind statisch verifizierbar (URL-Output, Backend-Branch, Filter-Bar-UI-Touch) — kein Runtime-Befund.

---

## Cross-Feature-H4-Sektion (Geschwister-Feature-Konvergenz zu F-3 library_detail)

**Direkt übertragbare F-3-Patterns mit library-List-View-Patterns:**

| F-3-Pattern (Quelle) | F-6.3-Pattern | List-View-Code-Anker | Konvergenz-Typ |
|----------------------|----------------|----------------------|-----------------|
| **F-3.3 P15 (Banner-Mountpoint)** | P1 (Banner-Mountpoint) | [templates/library.html:35](../templates/library.html#L35) (nach Filter-Bar) | direkt 1:1, 1 Container statt 2 |
| **F-3.3 P1 + P14 (Auto-Save-Failure-Banner + safeJSON)** | P3 (Auto-Save Favorite-Toggle) | [static/js/library.js:3-15](../static/js/library.js#L3-L15) `toggleFavorite` | direkt 1:1 |
| **F-3.3 P3 + P14 (Delete-Failure-Banner + safeJSON)** | P4 (Delete-Conversion) | [static/js/library.js:27-38](../static/js/library.js#L27-L38) `deleteConversion` | direkt 1:1, plus aria-live-Verzahnung mit P13 |
| **F-3.3 P6 (DE-Microcopy-Sweep)** | P5 (DE-Microcopy) | [templates/library.html](../templates/library.html), [static/js/library.js](../static/js/library.js) | direkt 1:1, ~30 Strings inkl. Filter-Bar / Pagination |
| **F-3.3 P8 (Toast-Level)** | P6 (Toast-Level) | [static/js/library.js:23](../static/js/library.js#L23) | direkt 1:1, eine Call-Site |

**Konvergenz-Quote:** 5 von 14 F-6.3-Patterns mit F-3.3-Korrespondenz = **36%**. Im erwarteten Master-Bereich 30-50% (Master-Annotation 3 + Findings-Doc-Erwartung). Niedriger als die F-6.2-Findings-Quote von 47%, weil zwei F-3-korrespondierende Findings-Paare in F-6.3 zu jeweils einem Pattern konsolidiert wurden (F2+F3 → P3 zählt als 1 Pattern, F4+F5 → P4 zählt als 1 Pattern). Höher als reine Einzel-Pattern-Konvergenz, weil F-3-Mechaniken die Sub-Batch-A-Mehrheit dominieren.

**Teil-übertragbare F-3-Patterns** (mit modifizierter Mechanik):

| F-3-Pattern (Quelle) | F-6.3-Pattern | Anpassung |
|----------------------|----------------|-----------|
| **F-3.3 P5 (Datum-Lokalisierung — JS-Pre-Population mit `formatDatetimeLocalNow`)** | P11 (Card-Datum DE-Lokalisierung) | **Andere Mechanik**: P5 war JS-Helper (Notion-Submit-Daten); P11 ist Server-side Custom Jinja2-Filter (Card-Display-Render). Heuristik-Verschiebung von H1 (Daten-Fehler) auf H4 (Display-Konsistenz mit DE-Welle). Aufwand-Verschiebung XS → S wegen Server-side-Filter-Aufbau analog `file_size`-Filter aus F-3-IMPL. |

### Bereits konvergente F-3-Patterns (List-View erfüllt das F-3-Pattern strukturell — kein F-6.3-Pattern nötig)

| F-3-Pattern | Bereits erfüllt durch | Code-Anker |
|-------------|------------------------|------------|
| **P9 — Tags-Input mit Chip-Visualisierung** | List-View rendert Tag-Chips Server-side mit `.c-tag`-Klasse — interessante Inversion: List-View war hier voraus, F-3-IMPL P9 hat das Pattern aus der List-View für Detail-View übernommen | [templates/library.html:67-71](../templates/library.html#L67-L71), [static/css/style.css](../static/css/style.css) `.c-tag` |
| **P11 — Sidebar-Active-State** | F-3-IMPL hat den path-Match `'/library' in request.path` in `base.html` etabliert — deckt sowohl `/library` (List) als auch `/library/<id>` (Detail) ab | [templates/base.html:84](../templates/base.html#L84) |
| **P12 — File-Size mit KB/B-Fallback (Server-side)** | Helper-Existenz `file_size`-Jinja-Filter ist da — auf List-View **n/a für Display** (Cards zeigen kein File-Size). Helper bleibt für Detail-View. | [app_pkg/__init__.py](../app_pkg/__init__.py) `file_size`-Filter |

### Nicht-anwendbare F-3-Patterns

| F-3-Pattern | Begründung |
|-------------|-----------|
| **P4 — Notion-Form State-Preservation** | List-View hat keine Notion-Form (Notion-Send ist Detail-only). |
| **P7 — Notion-Submit Persistent Error-Banner** | s.o. — kein Notion-Submit-Pfad auf List-View. |
| **P10 — Notion-Toggle aria-expanded/aria-controls** | s.o. — kein Notion-Toggle-Disclosure auf List-View. |
| **P13 — Page-`<title>` aktualisieren nach Title-Edit** | List-View hat keinen Title-Edit-Pfad; `<title>Library</title>` bleibt statisch. |

### Helper-Reuse-Reflexion (übernommen aus F-6.2 Master-Annotation 5)

F-6.2 hat in der Cross-Feature-H4-Sektion eine wichtige methodische Reflexion dokumentiert: nicht jede Helper-Reuse-Vermisst-Stelle ist automatisch ein H4-Finding, wenn die „Alternative" eine begründete Design-Wahl ist. F-6.3 übernimmt diese drei Stellen als **positive Disziplin-Notiz** (kein Pattern, kein Finding):

- **`saveViewState/loadViewState` zweite Call-Site: nein — URL-Persistierung ist die etablierte Design-Wahl.** Der gesamte View-State der List-View (Sortierung / Filter / Favorites / Suche / Pagination) wird über URL-Query-Params persistiert. Vorteile: bookmark-bar, sharable, browser-back-restoriert State, kein localStorage-Quota-Risk. Pattern P8 (Search Submit-Required) macht diese Design-Wahl explizit (Konsistenz mit URL-State-Mechanik). → **Keine H4-Verletzung**, weil URL-Persistierung bewusste Design-Wahl ist. Memory-Anker: [feedback_helper_reuse_design_choice.md](file:///Users/olivergluth/.claude/projects/-Volumes-MintHome-CODE-CONVERTER/memory/feedback_helper_reuse_design_choice.md).
- **`confirmInPlace` aus F-4-IMPL: zweite Call-Site nein — kein Bulk-Delete-Pfad auf List-View.** `confirmInPlace` ist die Idle→Confirm-Pending→Cancelling-State-Machine aus `audio_converter.js` für **mid-flight-cancel** eines laufenden RQ-Jobs, kein Generic-Bulk-Delete-Helper. Per-Card-Delete (siehe P4) verwendet rohes `confirm()` und ist semantisch näher zu F-3 Detail-Delete als zu F-4-Cancel. → **Keine H4-Verletzung**, weil das Helper-Pattern semantisch nicht passt.
- **`confirmIfLong`: n/a — Card-Delete soll jeden Delete bestätigen** (Threshold-Logik passt nicht zur Semantik der Card-Lösch-Aktion — Threshold ist für „Leeren"-Buttons auf Transcripts/Prompt-Editors). → **Keine H4-Verletzung**.

**Echte H4-Verletzungen** (in F-6.3-Patterns adressiert, **nicht** in der „begründete Design-Wahl"-Liste):

- **`showAlert`-Mountpoint fehlt** ([templates/library.html](../templates/library.html)) → **F10 H4 Sev 1, P1**. Vergessener Mountpoint, struktureller Vorbedingungs-Fix analog F-3-IMPL.
- **`safeJSON` fehlt in PUT/DELETE-Pfaden** ([static/js/library.js:5,29](../static/js/library.js#L5)) → in P3 und P4 adressiert (F-3.3 P14-Übernahme).

### Methodische Lehre (bereits im Memory verankert)

[feedback_helper_reuse_design_choice.md](file:///Users/olivergluth/.claude/projects/-Volumes-MintHome-CODE-CONVERTER/memory/feedback_helper_reuse_design_choice.md): „fehlende Helper-Call-Site ≠ automatisch H4-Verletzung wenn Alternative bewusste Design-Wahl ist (URL-Persistierung, semantisch unpassender Helper); Präzedenzfall F-6.2." F-6.3 bestätigt die Lehre — drei Helper-Reuse-Vermisst-Stellen werden als positive Disziplin-Notiz dokumentiert, statt sie zu künstlichen H4-Patterns aufzublasen.

---

## Helper-Vorschläge (für F6-IMPL-Sub-Thread zur Entscheidung)

**Erwartung erfüllt: keine neuen `_utils.js`-Helper.** Alle benötigten Helper sind bereits etabliert (siehe Helper-API-Header oben). Beim Pattern-Schreiben ist kein echter Helper-Vorschlag mit zweiter Call-Site-Begründung aufgekommen, der eine generische `_utils.js`-Extraktion rechtfertigen würde — das Memory `feedback_helper_reuse_design_choice.md` greift (keine künstliche Drift, keine Extraktion ohne zweite Call-Site).

**Server-side Helper-Vorschlag (App-Factory, nicht `_utils.js`)** — als Hinweis für F6-IMPL, nicht still mit-anlegen:

- **`format_card_datetime` Jinja2-Filter** für P11 (Card-Datum DE-Lokalisierung). Spiegelt das Mechanik-Muster vom `file_size`-Filter aus F-3-IMPL: explizit-implementiertes Format mit DE-Monatsnamen-Map, container-locale-agnostisch. Heute nur in P11 verwendet (eine Template-Stelle). Wenn künftig weitere Card-Listen-Views (z.B. Highlights-Layer aus Reader-Replacement-Roadmap, siehe Memory `project_readwise_replacement.md`) den selben Datums-Format brauchen, wird der Filter Cross-Feature relevant. Bis dahin reicht die Ein-Call-Site-Verwendung — Filter ist ohnehin im App-Factory zu registrieren, also kein „Drift-Risiko".

**Disposition:** Vorschlag bleibt im jeweiligen Pattern-Block als Server-Helper-Hinweis markiert; F6-IMPL-Sub-Thread entscheidet beim Cluster-Schnitt.

---

**Schweregrad-Skala (aus Stufe 2):**
1. kosmetisch (kaum spürbar)
2. gering (nur in Edge-Cases störend)
3. mittel (regelmäßig spürbar, frustrierend)
4. kritisch (verhindert/verfälscht die primäre Aufgabe oder produziert falsche Ergebnisse)
