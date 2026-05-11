# Sprint F6-PATTERNS — F-6.3 Patterns + Microcopy `library` List-View

**Datum**: 2026-05-10

**Ziel**: Stufe 3 (Patterns + Microcopy) der dreistufigen UX-Cascade-Methodik für die `library` List-View. Aus den 17 Findings + 4 Bug-Tickets aus F-6.2 konkrete UI-Pattern-Blöcke entwickeln, mit deutscher Microcopy, Aufwandsschätzung XS/S/M/L, Top-N-Quick-Wins per Impact-Score und Cluster-Vorschlag für F6-IMPL. **Geschwister-Feature-Hebel** (47% Cross-Feature-H4-Finding-Quote, 53% Pattern-Konvergenz zu F-3): Pattern-Übernahme aus F-3.3 ist die zentrale Mechanik für die Cross-Feature-H4-Konvergenz-Patterns — F-3-Patterns mit Korrespondenz wandern 1:1 mit angepasstem Code-Anker und ggf. Microcopy. **Kein Code-Touch**.

**Vorbedingung**:
- Pytest 66/66 grün auf `main`. Letzter Code-Touch: F6-REVIEW (commit `816d527`, 2026-05-10).
- **Eingabe**: [docs/ui_findings_library_list_2026-05.md](docs/ui_findings_library_list_2026-05.md) (Sub-Thread liest komplett vor Phase 1).
  - **17 Findings**: Sev 4: 0 / Sev 3: 6 (F1 Copy-200char-Preview H9, F2+F3 toggleFavorite-silent H1+H9, F4+F5 deleteConversion-silent H1+H9, F6 EN-Strings H4) / Sev 2: 3 (F7 Toast-Level H4, F8 Empty-State H1, F9 Search-no-auto-submit H1) / Sev 1: 8 (F10 Banner-Mountpoint H4, F11 Search-no-live H6, F12 favorites='' H4, F13 Type-Filter-Validation H9, F14 %b-Locale H4, F15 Per-Page-Size H1, F16 aria-live H6, F17 Card-Hover-Lift H6).
  - **4 Bug-Tickets** (alle Finding-linked, keine pure): BT1 (toggleFavorite-Errors, ↔F2/F3) / BT2 (deleteConversion-Errors, ↔F4/F5) / BT3 (Copy-200char-Quelle, ↔F1) / BT4 (Toast-Level Copy-Failure, ↔F7).
  - **Cross-Feature-H4-Finding-Quote 47%** (8 von 17 Findings: F2/F3 + F4/F5 + F6 + F7 + F10 + F14) — Schwester-Feature-Hebel zu F-3 bestätigt im Master-Erwartungsband 35-50%.
  - **9 ⚠️ code-only-Findings**: F2-F5, F7, F9, F11, F14, F16, F17.
  - **Bereits-konvergente F-3-Patterns** (3): P9 Tag-Chips Server-side, P11 Sidebar-Active via `base.html`-path-Match, P12 file_size-Filter (n/a für List-View). **Keine Patterns nötig**, in Cross-Feature-H4-Sektion als positives Inventar erwähnen.
  - **Nicht-anwendbare F-3-Patterns** (4): P4 / P7 / P10 Notion-only, P13 Title-Edit-only.
  - **Helper-Reuse-Reflexion aus F-6.2** (Cross-Feature-H4-Sektion): `saveViewState/loadViewState` zweite Call-Site nein (URL-Persistierung Design-Wahl), `confirmInPlace` zweite Call-Site nein (kein Bulk-Delete), `confirmIfLong` semantisch unpassend. Memory-Eintrag verankert: [feedback_helper_reuse_design_choice.md](file:///Users/olivergluth/.claude/projects/-Volumes-MintHome-CODE-CONVERTER/memory/feedback_helper_reuse_design_choice.md) — Helper-Reuse-Drift mit begründeter Design-Wahl ist keine H4-Verletzung.
  - **4 Schwerpunkt-Cluster** aus F-6.2:
    - **Cluster 1: Silent-Failure-Familie** (F1-F5, Sev 3, Daily-Usage-Hotspot — Copy-Btn + Auto-Save + Delete).
    - **Cluster 2: Cross-Feature-H4-Helper-Reuse zu F-3** (F2/F3, F4/F5, F6, F7, F10, F14 — überlappt mit Cluster 1).
    - **Cluster 3: List-View-State-Visibility und Empty-State-Recovery** (F8, F9, F11).
    - **Cluster 4: List-Polish-Long-Tail** (F12-F17).
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - **F-3.3 Patterns-Doc**: [docs/ui_patterns_library_detail_2026-05.md](docs/ui_patterns_library_detail_2026-05.md) — **primäre Pattern-Übernahme-Quelle** (Geschwister-Feature). Pattern-Beschreibungen, Microcopy und Mechanik-Skizzen für P1/P3/P6/P8/P14/P15 sind dort bereits ausgearbeitet.
  - F-1.3: [docs/ui_patterns_document_converter_2026-05.md](docs/ui_patterns_document_converter_2026-05.md) — 14 Patterns.
  - F-2.3: [docs/ui_patterns_audio_converter_2026-05.md](docs/ui_patterns_audio_converter_2026-05.md) — 21 Patterns.
  - F-4.3: [docs/ui_patterns_podcast_flow_2026-05.md](docs/ui_patterns_podcast_flow_2026-05.md) — 12 Patterns + Live-verifiziert-vs-Smoke-Pflicht-Markierung.
  - F-5.3: [docs/ui_patterns_markdown_converter_2026-05.md](docs/ui_patterns_markdown_converter_2026-05.md) — 13 Patterns + Schwester-Feature-Übernahme-Disziplin als Vorlage für analoge Geschwister-Feature-Übernahme.
- **Helper-Bestand in `_utils.js`** (alle bereits etabliert, **keine neuen Helper erwartet**): `showAlert`, `showToast`, `formatFileSize`, `safeJSON`, `formatDatetimeLocalNow`, `confirmIfLong`, `fallbackCopyText`, `saveViewState/loadViewState`, `attachAutoDismissToServerBanners` (lokal in `markdown_converter.js`), `confirmInPlace` (lokal in `audio_converter.js`), `.sr-only`-Utility. Server-side: `file_size`-Jinja-Filter.
- **Microcopy-Regeln** (für DE-Texte): Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern.
- **Impact-Score-Formel**: `Score = Sev × 5 / Aufwand-Gewicht`. XS=1, S=2, M=4, L=8.

**Out-of-scope**:
- Implementation — eigener Folge-Sprint `F6-IMPL`.
- Code-Änderungen jeglicher Art.
- **BT1-BT4 finding-linked**: hier nicht eigenständig adressieren, werden via die Patterns ihrer Findings mit-gelöst (BT1 ↔ Auto-Save-Pattern aus F2+F3, BT2 ↔ Delete-Pattern aus F4+F5, BT3 ↔ Copy-Full-Content-Pattern aus F1, BT4 ↔ Toast-Level-Pattern aus F7).
- DE-Pass für englische Strings (F6): wird **innerhalb** der F-6.3-Patterns mit-gemacht (DE-Microcopy-Sweep aus F-3.3 P6-Übernahme). **Kein** separater DE-Pass-Pattern.
- Andere Features (`mermaid_converter`, `login`) — eigene Folge-Wellen.

---

## Master-Annotation (vorab eingebettet)

### 1. Geschwister-Feature-Pattern-Übernahme aus F-3.3 als zentrale Mechanik

**Analog F-5.3's F-1.3-Übernahme-Disziplin**: 47% der Findings haben F-3-Korrespondenz, und die F-3.3-Patterns-Doc hat die Pattern-Mechanik schon ausgearbeitet (Beschreibung, Visuelle Hinweise, Microcopy-Strings, Helper-Reuse, Aufwand). Diese Arbeit nochmal zu machen ist Verschwendung — und schlimmer: würde Pattern-Drift erzeugen wo Konvergenz das Ziel ist.

**Methodik-Konsequenz**:
- **F-3.3-Patterns mit Korrespondenz wandern 1:1 mit**. Pattern-Block-Inhalt wird übernommen, **nur** drei Felder werden angepasst:
  - **Code-Anker**: auf `library`-Code statt `library_detail` (z.B. `static/js/library.js` statt `static/js/library_detail.js`, `templates/library.html` statt `templates/library_detail.html`).
  - **Microcopy-Anpassung wo list-spezifisch nötig**: z.B. statt „Notiz konnte nicht gespeichert werden" eher „Favorit konnte nicht aktualisiert werden" / „Konvertierung konnte nicht gelöscht werden". Die meisten F-3-Strings sind direkt brauchbar, weil Helper-Reuse-Mechanik identisch ist.
  - **Adressiert-Findings**: F-6.2-Finding-Nummer statt F-3.2-Finding-Nummer.
- **Aufwand wird übernommen**, außer Sub-Thread sieht klaren Grund für Abweichung.
- **Pattern-Nummern-Konvention**: Sub-Thread vergibt F-6.3-eigene P1, P2, … nach thematischer Cluster-Reihenfolge. **Nicht** F-3-Nummern reproduzieren. F-3-Korrespondenz wird in einer expliziten Sub-Zeile pro Pattern-Block ausgewiesen (z.B. „**F-3-Korrespondenz**: P1 Auto-Save-Failure-Banner-Übernahme").

**Konvergenz-Items die F-3-Übernahme nutzen** (Sub-Thread mappt im Pre-Flight):
- F2+F3 (toggleFavorite-silent H1+H9 Sev 3) → F-3.3 P1 + P14 (Auto-Save-Failure-Banner + safeJSON-Wrap) konsolidieren.
- F4+F5 (deleteConversion-silent H1+H9 Sev 3) → F-3.3 P3 + P14 (Delete-Failure-Banner + safeJSON-Wrap) konsolidieren.
- F6 (EN-Strings H4 Sev 3) → F-3.3 P6 DE-Microcopy-Sweep übernehmen.
- F7 (Toast-Level Copy-Failure H4 Sev 2) → F-3.3 P8 Toast-Level pro Call-Site übernehmen.
- F10 (Banner-Mountpoint H4 Sev 1) → F-3.3 P15 struktureller Vorbedingung-Block übernehmen.
- F14 (Locale `%b` H4 Sev 1) — **abweichend** weil andere Mechanik (Server-side `strftime` statt JS-Helper). Eigenes list-spezifisches Pattern. F-3.3 P5 (`formatDatetimeLocalNow`) ist nicht direkt anwendbar.

### 2. Befund 1 (Copy-Btn 200-char-Preview) als eigenes list-spezifisches Pattern

Master-Annotation aus F-6.2 wird hier **als Pattern-Block ausgeformt**. Pattern-Mechanik:
- **Adressierte Findings**: F1 (Copy-200char-Preview H9 Sev 3 ⚠️) + BT3 (Copy-Quelle-Bug).
- **UI-Pattern**: Statt `content[:200]`-Slice → Full-Content aus `ConversionHistory.content` (oder analoger Spalte) für Copy-Aktion. Backend ggf. neuer Endpoint `api_copy_content/<id>` oder Embed im Page-Render. Sub-Thread inspiziert beim Pattern-Schreiben die Datenquelle in `app_pkg/library.py`.
- **Microcopy**: deutsch, max 2 Sätze, ohne Emoji. „Vollständiger Text kopiert" oder ähnlich konkret formulieren (Sub-Thread entscheidet).
- **Helper-Reuse**: bestehender `fallbackCopyText` aus `_utils.js` plus `showToast`-Success-Level.
- **Aufwand**: S (Backend-Touch wenn neuer Endpoint nötig, sonst XS wenn Embed im Page-Render reicht).
- **Live-Verifikation-Status**: `🔥 Smoke-Pflicht in F6-IMPL` (Copy-Paste-Test mit langem Content >200 char).
- **Verzahnung-Hinweis**: BT3 wird durch dieses Pattern aufgelöst, kein separater Bug-Ticket-Apply nötig.

Sub-Thread kann Pattern-Nummer und Cluster-Zuordnung wählen — Erwartung: gehört in **Cluster 1 (Silent-Failure-Familie)** oder eigenständig.

### 3. Smoke-Pflicht-Kalibrierung — Drei 🔥-Pflicht-Live-Master-Smoke-Patterns von ~9

9 ⚠️ code-only-Findings (F2-F5, F7, F9, F11, F14, F16, F17) ist viel. Sub-Thread soll Smoke-Disziplin analog F-5.3 kalibrieren — **drei Pflicht-Live-Master-Smoke-Patterns**, andere als „code-evident verifiziert im Container":

**🔥-Pflicht-Live-Master-Smoke-Empfehlung** (Sub-Thread kann pragmatisch abweichen):
- **F1+BT3 Copy-Full-Content-Pattern** — Copy-Paste-Verifikation mit langem Content (>200 char) im Browser.
- **F2+F3 Auto-Save-Failure-Banner** — DevTools-Network-Throttle Offline → Favorite-Toggle → Banner-Sichtbarkeit. Analog F-3-IMPL P1 Smoke-Mechanik.
- **F14 Locale-`%b`-Display** — Browser-Inspektion der Card-Datum-Anzeige (deutsche vs. englische Monatsabkürzung), DE-Erwartung „Mär" statt „Mar".

**Code-evident-verifiziert** (Container-side):
- F4+F5 Delete-Failure-Banner — Code-Reading-Verifikation der Mechanik (analog F-3 P3).
- F7 Toast-Level — Code-Reading der Toast-Call-Sites.
- F9 + F11 Search-Live/Submit — Code-Reading der Search-Handler.
- F16 aria-live — Code-Reading der DOM-Annotations.
- F17 Card-Hover-Lift — Code-Reading der CSS-Klassen + JS-Click-Handler.

Sub-Thread berichtet pro 🔥-Pflicht-Pattern Smoke-Mechanik im Pattern-Block.

### 4. Konsolidierungs-Erwartung ~15-25% mit klaren Konsolidaten

Konsolidierungs-Quote-Erwartung **15-25%** (analog F-3.3's 21%, F-4.3's 25%, F-5.3's 13%):
- **F2+F3 → 1 Pattern** (Auto-Save-Failure-Familie, identischer Code-Pfad `toggleFavorite` + identische F-3.3-P1/P14-Korrespondenz).
- **F4+F5 → 1 Pattern** (Delete-Failure-Familie, identischer Code-Pfad `deleteConversion` + identische F-3.3-P3/P14-Korrespondenz).
- **F8 + F11 oder F8 + F9 + F11** ggf. konsolidieren (List-View-State-Sub-Cluster mit Search+Empty-State-Übergang) — Sub-Thread entscheidet pragmatisch.
- **F-3-Konvergenz-Pattern-Identität bleibt erkennbar**: F6 / F7 / F10 / F14 bleiben **eigene Pattern-Blöcke** (analog F-5.3-Disziplin) — F-3-Pattern-Anker werden in der F-3-Korrespondenz-Sub-Zeile dokumentiert, aber nicht zu Cluster-Bündeln zusammenschmelzen.

Erwartete Pattern-Anzahl: **13-15 Patterns** aus 17 Findings + 4 Bug-Tickets (~15-25% Konsolidierung).

### 5. List-View-State-Cluster (F8, F9, F11) — pragmatische Master-Default-Wahl pro Pattern

F8 (Empty-State filter-aware), F9 (Search-no-auto-submit), F11 (Search-no-live) sind List-View-State-Findings die eine pragmatische Default-Wahl im Pattern brauchen (analog F-5.3's Reader-Mode-Default-Wahl):

- **F8 Empty-State filter-aware**: Default-Vorschlag — server-side im `library_view`-Render entscheiden welcher Empty-State angezeigt wird (Filter-aktiv → „Keine Treffer mit aktuellen Filtern" + Filter-zurücksetzen-Link; Filter-leer → „Library leer — erste Konvertierung starten" + Link zur Konvertierungs-Seite). Begründung im Pattern.
- **F9 Search-no-auto-submit + F11 Search-no-live**: Default-Vorschlag — **Submit-Required mit DE-Hint** („Enter zum Suchen") statt Live-Debouncing, weil URL-Persistierung-Design-Wahl gilt (Live-Search ohne URL-Update wäre State-Drift; mit URL-Update wäre History-Pollution bei jedem Tastendruck). Submit-Required ist konsistent mit der URL-Query-Param-Mechanik. Begründung im Pattern.

**Sub-Thread folgt Defaults** beim Pattern-Schreiben. Kann abweichen wenn beim Apply technische Probleme oder bessere Alternative — Bericht-Pflicht. **Keine** Variante-A/B/C-Diskussion.

### 6. Helper-Vorschlags-Disposition: keine neuen Helper

**Erwartung**: keine neuen `_utils.js`-Helper. Alle benötigten Helper sind bereits etabliert (siehe Helper-Bestand). **Memory `feedback_helper_reuse_design_choice.md`-Geist greift hier**: keine künstliche Helper-Drift, keine Extraktion ohne zweite Call-Site.

Sub-Thread kann abweichen wenn beim Pattern-Schreiben ein **echter** Helper-Vorschlag aufkommt mit zweiter Call-Site-Begründung (z.B. ein generischer `escapeHtml`-Helper wenn mehrere Patterns Template-Injection-Pfade berühren) — in Helper-Vorschlags-Sektion am Doc-Ende sammeln mit Begründung.

---

## Phase 1 — Patterns + Microcopy

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Findings-Doc + Inventur-Doc komplett lesen**.
3. **Methodik-Vorlagen lesen**: F-3.3 (primäre Übernahme-Quelle, **muss komplett gelesen werden**), F-5.3 (Geschwister-Feature-Übernahme-Disziplin-Vorlage), F-1.3, F-2.3, F-4.3.
4. **`_utils.js`-Helper-Bestand verifizieren**: `grep -n "^function\|window\." static/js/_utils.js`. Patterns sollen vorhandene Helper nutzen — neue nur mit Begründung am Doc-Ende.
5. **F-3-Korrespondenz-Mapping vorbereiten**: für jeden F-6.2-Finding mit F-3-Korrespondenz das passende F-3-Pattern in F-3.3 nachschlagen und Übernahme-Notiz vorbereiten.

**Pattern-Aufgabe**:

Für jeden Finding (oder konsolidierte Finding-Gruppe) ein Pattern-Block mit:

- **Pattern-Nummer** (P1, P2, …) nach thematischer Cluster-Reihenfolge.
- **Adressiert Findings**: Liste mit H + Sev + linked BTs.
- **F-3-Korrespondenz**: Pattern-Code aus F-3.3 (z.B. „F-3.3 P1+P14") wenn übertragen, oder „—" wenn list-spezifisch.
- **Live-Verifikation-Status**: `🔥 Smoke-Pflicht in F6-IMPL` für Patterns die ⚠️ code-only-Findings adressieren mit explizitem Smoke-Mechanik-Hinweis (siehe Master-Annotation 3); sonst leer.
- **UI-Pattern-Beschreibung**: 2-4 Sätze, konkrete Mechanik. Bei Übernahme: kurz vermerken „aus F-3.3 P-X übernommen" + ggf. list-spezifische Anpassung.
- **Visuelle Hinweise**: vorhandene CSS-Klassen aus Neomorphism + `.sr-only` aus F-3-IMPL nutzen.
- **Microcopy** (deutsch): exakte Strings für Banner, Status-Labels, Tooltips. Nach Microcopy-Regeln. Bei Übernahme: F-3.3-Strings recyclen wo passend; bei list-spezifischen Findings konkret formulieren (insbesondere für Befund-1/F1-Toast und F8 Empty-State-Microcopy).
- **Helper-Reuse**: welche `_utils.js`-Helper. **Erwartung hoch** wegen 47% H4-Quote.
- **Aufwand**: XS / S / M / L mit Begründung. Bei Übernahme: F-3.3-Aufwand verifizieren oder Abweichung begründen.
- **Impact-Score**: `Sev × 5 / Aufwand-Gewicht`.
- **Konsolidierung-Hinweis** falls mehrere Findings adressiert.

**Konsolidierungs-Logik** (analog F-3.3 / F-5.3):

- **F2+F3 (toggleFavorite-silent) → 1 Pattern** mit F-3.3-P1+P14-Übernahme.
- **F4+F5 (deleteConversion-silent) → 1 Pattern** mit F-3.3-P3+P14-Übernahme.
- **F8 + F9/F11 ggf. konsolidieren** falls List-View-State-Cluster eng verkoppelt.
- **F-3-Konvergenz-Patterns** bleiben als **eigene Pattern-Blöcke** (analog F-5.3-Disziplin): F6 / F7 / F10 / F14 — F-3-Pattern-Identität soll erkennbar bleiben.

**Output-Doc**: `docs/ui_patterns_library_list_2026-05.md`. Struktur 1:1 wie F-3.3 / F-5.3 + Geschwister-Feature-Sektion:

1. Header mit Findings-Quelle, Sprint-Datum, Methodik-Hinweis, **Geschwister-Feature-Übernahme-Notiz** (F-3.3 als primäre Pattern-Übernahme-Quelle, Memory `feedback_helper_reuse_design_choice.md`-Pointer).
2. **Pattern-Blöcke** P1 bis Pn (nach Cluster-Reihenfolge).
3. **Cluster-Vorschlag für F6-IMPL**: 1-3 Implementations-Cluster vorgeschlagen. **Default-Empfehlung**: 1-Sprint mit 2-3 Sub-Batches analog F-5-IMPL (Cluster 1 Silent-Failure-Familie inkl. Cross-Feature-H4 → Cluster 2 List-View-State-Visibility → Cluster 3 List-Polish-Long-Tail). Bei <13 Patterns ist 1-Sprint pragmatisch.
4. **Top-5 Quick-Wins**: Tabelle nach Impact-Score absteigend.
5. **Smoke-Pflicht-Übersicht**: Liste der Patterns mit `🔥 Smoke-Pflicht in F6-IMPL`-Sub-Tag plus Smoke-Mechanik pro Pattern. Drei-Pflicht-Live-Master-Smoke-Markierung (Master-Annotation 3).
6. **Cross-Feature-H4-Sektion** (Geschwister-Feature-strukturiert analog F-5.3):
   - **Direkt übertragbare Konvergenz-Items**: Liste der Patterns mit F-3-Korrespondenz, Code-Anker auf `library.js` / `library.html` / `library.py`.
   - **Bereits konvergente F-3-Patterns**: P9 Tag-Chips / P11 Sidebar-Active / P12 file_size-Filter als positives Inventar.
   - **Nicht-anwendbare F-3-Patterns**: P4 / P7 / P10 / P13.
   - **Helper-Reuse-Reflexion** (übernommen aus F-6.2 Master-Annotation 5): `saveViewState/loadViewState`-URL-Persistierungs-Begründung als positive Disziplin-Notiz mit Pointer auf Memory `feedback_helper_reuse_design_choice.md`. `confirmInPlace`-kein-Bulk-Delete-Begründung.
   - **Konvergenz-Quote**: Cross-Feature-H4-Pattern-Quote (X von Y Patterns mit F-3-Korrespondenz). Erwartung 30-50%.
7. **Helper-Vorschlags-Sektion** am Doc-Ende: erwartet leer (Master-Annotation 6). Falls Sub-Thread doch einen Helper-Vorschlag mit zweiter Call-Site begründet sieht: dokumentieren ohne still anzulegen.

Nach Phase 1: STOP — Bericht. Statistik (Pattern-Anzahl, Konsolidierungs-Quote, Aufwand-Verteilung, Smoke-Pflicht-Anzahl mit Drei-Pflicht-Live-Master-Kalibrierung, F-3-Übernahme-Anzahl, Cluster-Vorschlag, Top-5-Quick-Wins, Befund-1-Pattern-Disposition).

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Pattern-Doc nochmal und prüft:

1. **Vollständigkeit**: jeder der 17 Findings ist adressiert. BT1+BT2+BT3+BT4 in Patterns erwähnt (F2+F3-Pattern für BT1, F4+F5-Pattern für BT2, F1-Pattern für BT3, F7-Pattern für BT4). Keine Pure-Bug-Tickets dispositioniert weil alle 4 Finding-linked.
2. **F-3-Übernahme-Disziplin**: Patterns mit F-3-Korrespondenz tragen explizit „F-3-Korrespondenz: P-X" + Übernahme-Notiz; Code-Anker stimmen (`library` statt `library_detail`); Microcopy ist deutsch und folgt Microcopy-Regeln.
3. **Live-Verifikation-Status-Konsistenz**: Patterns für die 9 ⚠️ code-only-Findings tragen `🔥 Smoke-Pflicht in F6-IMPL` mit Smoke-Mechanik-Hinweis. Drei-Pflicht-Live-Master-Smoke-Markierung (Master-Annotation 3) explizit gesetzt.
4. **List-View-State-Default-Wahl**: Patterns für F8/F9/F11 enthalten konkrete Default-Mechanik (server-side Empty-State-Branching + Submit-Required-mit-DE-Hint statt Live-Debouncing) — Master-Annotation 5.
5. **Befund-1-Pattern**: F1+BT3 in einem Pattern-Block mit Full-Content-Copy-Mechanik, S Aufwand (oder XS wenn Embed-Pfad), Smoke-Pflicht. Master-Annotation 2 aufgegriffen.
6. **Konsolidierungs-Quote**: 15-25% Bereich oder Abweichung begründet.
7. **Microcopy-Regeln**: Stichprobe — Banner-Strings (≤2 Sätze), Status-Labels (≤3 Wörter), Empty-States (≤3 Sätze).
8. **Impact-Score-Konsistenz**: Top-5-Tabelle stimmt mit Pattern-Block-Score-Angaben überein.
9. **Cross-Feature-H4-Sektion**: bereits-konvergente F-3-Patterns gelistet (3 Items), nicht-anwendbare gelistet (4 Items), Helper-Reuse-Reflexion mit Memory-Pointer dokumentiert.
10. **Helper-Disposition**: keine neuen Helper still angelegt; falls Vorschläge dann in Helper-Vorschlags-Sektion mit Begründung.
11. **Disziplin**: keine konkreten Code-Diffs, kein Bug-Fix, keine Implementations-Schritte.

Nach Phase 2: STOP — Bericht. „Pattern-Doc konsistent" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-6.3 / Stufe 3: patterns + microcopy of library list view".
- Body: Statistik (Pattern-Anzahl, Konsolidierungs-Quote, Aufwand-Verteilung, Smoke-Pflicht-Anzahl mit Drei-Pflicht-Live-Master-Kalibrierung, F-3-Übernahme-Anzahl, Cluster-Vorschlag, Top-5-Quick-Wins, List-View-State-Default-Wahl-Hinweis).
- Branch: direkt auf `main` ist OK.
- `git push origin main`. Wenn Auto-Mode-Classifier blockt **oder** `.git/objects/<hash>`-SMB-Permission blockt (analog F-5.3 / F-6.1 / F-6.2): im Phase-3-Bericht erwähnen, Master committet/pusht von Hand via SSH zu Mintbox.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — eine Output-Datei (`docs/ui_patterns_library_list_2026-05.md`), Pattern-Blöcke + Microcopy + Aufwandsschätzung + Cluster-Vorschlag, kein Code-Touch, keine Tests, kein Smoke. Geschwister-Feature-Übernahme aus F-3.3 senkt den Pattern-Konstruktions-Aufwand vs. F-4.3, aber F-3.3-Cross-Read und Per-Pattern-Übernahme-Disziplin bleiben Aufwand.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim F-3-Pattern-Übernehmen ein **Aufwand-Mismatch** entsteht (F-3-Pattern war XS weil Helper schon im `library_detail.js` eingebunden, hier ist es XS weil identisch — sollte überwiegend keine Abweichung sein): kurz im Pattern-Block notieren.
- Wenn beim Pattern-Schreiben ein **Helper-Vorschlag** mit zweiter Call-Site-Begründung aufkommt: in Helper-Vorschlags-Sektion am Doc-Ende sammeln, nicht still mit-anlegen. **Memory `feedback_helper_reuse_design_choice.md` greift**: keine künstliche Drift.
- Wenn ein **List-View-State-Default-Vorschlag** aus Master-Annotation 5 sich beim Pattern-Schreiben als technisch problematisch erweist (z.B. server-side Filter-aware Empty-State braucht eine Filter-Boolean im Context die noch nicht da ist): kurz Bedenken im Pattern-Block vermerken — F6-IMPL muss dann eine Alternative finden, **nicht** F-6.3 die Diskussion auflösen.
- Wenn beim Code-Reading neue Findings auffallen die nicht in F-6.2 dokumentiert sind: als „aufgefallen, nicht gefixt" in den Bericht — siehe Memory `feedback_no_silent_fixes.md`.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F6-PATTERNS ☑ done 2026-05-XX → commit `<hash>`. Patterns-Doc unter `docs/ui_patterns_library_list_2026-05.md`. <N> Patterns (XS: a, S: b, M: c, L: d), <K> Cluster vorgeschlagen, Top-5-Quick-Wins-Score-Range <X.0-Y.0>, <S>× 🔥 Smoke-Pflicht für F6-IMPL davon <M>× Pflicht-Live-Master-Smoke, F-3-Übernahme-Anzahl: <U> von <T> H4-Findings. Verbleibende Sequenz: F6-IMPL → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F6-PATTERNS" raus → Erledigt-Liste; Master fügt F6-IMPL als Position 1 beim nächsten Dispatch hinzu — Sub-Thread fügt es **nicht** selbst hinzu (Master-Edit-Zone).
- **Memory**: nichts erwartet — Pattern-Methodik etabliert, Geschwister-Feature-Übernahme analog F-5.3 schon in `feedback_helper_reuse_design_choice.md` indirekt verankert. Falls beim Apply eine **neue** übertragbare Lehre auftaucht: defensiv `feedback_*.md` schreiben.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Pattern-Methodik etabliert, Geschwister-Feature-Übernahme + Befund-1-Pattern + Smoke-Pflicht-Kalibrierung + Konsolidierungs-Erwartung + List-View-State-Default-Wahl + Helper-Disposition in Master-Annotationen oben verankert.)_
