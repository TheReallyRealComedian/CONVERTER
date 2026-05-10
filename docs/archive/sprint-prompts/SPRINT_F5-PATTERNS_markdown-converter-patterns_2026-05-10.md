# Sprint F5-PATTERNS — F-5.3 Patterns + Microcopy `markdown_converter`

**Datum**: 2026-05-10

**Ziel**: Stufe 3 (Patterns + Microcopy) der dreistufigen UX-Cascade-Methodik für `markdown_converter`. Aus den 15 Findings + 5 Bug-Tickets aus F-5.2 konkrete UI-Pattern-Blöcke entwickeln, mit deutscher Microcopy, Aufwandsschätzung XS/S/M/L, Top-N-Quick-Wins per Impact-Score und Cluster-Vorschlag für F5-IMPL. **Schwester-Feature-Hebel** (47% Cross-Feature-H4-Findings, 86% Pattern-Konvergenz zu F-1): Pattern-Übernahme aus F-1.3 ist die zentrale Mechanik — F-1-Patterns mit Korrespondenz wandern 1:1 mit angepasstem Code-Anker und angepasster Microcopy. **Kein Code-Touch**.

**Vorbedingung**:
- Pytest 65/65 grün auf `main`. Letzter Code-Touch: F5-REVIEW (commit `83deeb5`, 2026-05-10).
- **Eingabe**: [docs/ui_findings_markdown_converter_2026-05.md](docs/ui_findings_markdown_converter_2026-05.md) (Sub-Thread liest komplett vor Phase 1).
  - **15 Findings**: Sev 4: 0 / Sev 3: 3 (F1 H4 Save-`alert()`, F2 H9 Save-Recovery, F3 H9 PDF-Gen-Error-Re-Render ⚠️) / Sev 2: 6 (F4 H1, F5 H9 ⚠️, F6 H1 ⚠️, F7 H1 ⚠️, F8 H4 ⚠️, F9 H6) / Sev 1: 6 (F10 H1, F11 H6 ⚠️, F12 H6 ⚠️, F13 H6, F14 H1, F15 H9 ⚠️).
  - **5 Bug-Tickets**: BT1 (PDF-Gen-Re-Render-Context, ↔F3) / BT2 (Sample-Merge-Template, ↔F6) / BT3 (`updateStyle()` doppelt, pure) / BT4 (Inline-`<style>`, pure) / BT5 (toter `<link>`, pure).
  - **Pattern-Konvergenz-Quote 86%** zu F-1.3 (12/14 F-1-Patterns übertragbar oder bereits-erfüllt) — Schwester-Feature-Inversion zu F-4.3's 0% bestätigt.
  - **Cross-Feature-H4-Findings 7/15 (47%)** — höher als F-2.2 (~41%), F-3.2 (~35%), F-4.2 (0%).
  - **Bereits-konvergente F-1-Patterns** (5): P3 setTimeout-Reset, P6 SEC-Sprint Format-Label, P8 SEC-Sprint Frontend-Vorab-Check (Backend-Whitelist mit DE-Microcopy), P10 Result-Scroll, P11 native File-Input-Visibility. **Keine Patterns nötig**, in Cross-Feature-H4-Sektion als positives Inventar erwähnen.
  - **Nicht-anwendbare F-1-Patterns** (2): P2 Result-Area, P5 Drag-Active-Highlight.
  - **8 ⚠️ code-only-Findings**: F3, F5, F6, F7, F8, F11, F12, F15 — siehe Master-Annotation 3 unten zur Walkthrough-Disposition.
  - **4 Schwerpunkt-Cluster** aus F-5.2:
    - **Cluster 1: Cross-Feature-H4-Helper-Reuse zu F-1** (F1, F2, F14, F13, F10) — günstigster Konvergenz-Pfad.
    - **Cluster 2: Reader-Mode-State und Visual-Layout** (F9, F11, F12, F8) — H6/H4-dominiert, Daily-Usage-Schmerz mittel-hoch.
    - **Cluster 3: Error-Recovery-Pfade** (F3, F5, F6, F2) — H9-dominiert, alle ⚠️ code-only.
    - **Cluster 4: Async-Pre-Check und Loading-Visibility** (F4, F7) — H1-Reibung niedrig.
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - F-1.3: [docs/ui_patterns_document_converter_2026-05.md](docs/ui_patterns_document_converter_2026-05.md) — **primäre Pattern-Übernahme-Quelle** (Schwester-Feature). Pattern-Beschreibungen, Microcopy und Mechanik-Skizzen sind dort bereits ausgearbeitet.
  - F-2.3: [docs/ui_patterns_audio_converter_2026-05.md](docs/ui_patterns_audio_converter_2026-05.md) — 21 Patterns + Cluster-I/II-Vorbereitung.
  - F-3.3: [docs/ui_patterns_library_detail_2026-05.md](docs/ui_patterns_library_detail_2026-05.md) — 15 Patterns + 3-Sub-Batch-Cluster.
  - F-4.3: [docs/ui_patterns_podcast_flow_2026-05.md](docs/ui_patterns_podcast_flow_2026-05.md) — 12 Patterns + Live-verifiziert-vs-Smoke-Pflicht-Markierung.
- **Helper-Bestand in `_utils.js`** (Cross-Feature-H4-Reuse-Quelle, hohe Erwartung wegen 47% H4-Quote): `showAlert`, `showToast`, `formatFileSize`, `safeJSON`, `formatDatetimeLocalNow`, `confirmIfLong`, `confirmInPlace` (lokal in `audio_converter.js` aus F-4.3, wartet auf zweite Call-Site), `.sr-only`-Utility, `safeGet` (falls vorhanden). Server-side: `file_size`-Jinja-Filter aus F-3-IMPL.
- **Microcopy-Regeln** (für DE-Texte): Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern.
- **Impact-Score-Formel**: `Score = Sev × 5 / Aufwand-Gewicht`. XS=1, S=2, M=4, L=8.

**Out-of-scope**:
- Implementation — eigener Folge-Sprint `F5-IMPL`.
- Code-Änderungen jeglicher Art.
- **BT3 / BT4 / BT5** (pure Bug-Tickets ohne UX-H-Komponente): nicht im Pattern-Sprint, separater Bug-Sweep oder mit F5-IMPL mit-genommen wenn nahegelegene Patterns berührt werden (siehe Master-Annotation 4).
- **BT1 / BT2** (finding-linked): hier nicht eigenständig adressieren, werden via die Patterns ihrer Findings (P-X für F3, P-Y für F6) mit-gelöst.
- DE-Pass für Englisch-Strings (Befund 8 aus F-5.1 ausgenommen): wird **innerhalb** der F-5.3-Patterns mit-gemacht — wo ein Pattern Microcopy spezifiziert, ist sie deutsch. **Kein** separater DE-Pass-Pattern.
- Andere Features.

---

## Master-Annotation (vorab eingebettet)

### 1. Schwester-Feature-Pattern-Übernahme aus F-1.3 als zentrale Mechanik

**Erwartung-Verschiebung gegenüber F-4.3**: F-4.3 hatte 0% H4-Quote und musste alle 12 Patterns aus eigenen Findings konstruieren. Hier ist die Lage umgekehrt — 47% der Findings haben F-1-Korrespondenz, und die F-1.3-Pattern-Doc hat die Pattern-Mechanik schon ausgearbeitet (Beschreibung, Visuelle Hinweise, Microcopy-Strings, Helper-Reuse, Aufwand). Diese Arbeit nochmal zu machen ist Verschwendung — und schlimmer: würde Pattern-Drift erzeugen wo Konvergenz das Ziel ist.

**Methodik-Konsequenz**:
- **F-1-Patterns mit Korrespondenz wandern 1:1 mit**. Pattern-Block-Inhalt wird übernommen, **nur** drei Felder werden angepasst:
  - **Code-Anker**: auf `markdown_converter`-Code statt `document_converter` (z.B. `static/js/markdown_converter.js` statt `static/js/document_converter.js`, `templates/markdown_converter.html` statt `templates/document_converter.html`).
  - **Microcopy-Anpassung wo markdown-spezifisch nötig**: z.B. statt „Konvertierung läuft …" eher „Konvertierung läuft …" oder „PDF wird erstellt …" je nach Pattern. Die meisten F-1-Strings sind direkt brauchbar, weil File-Upload-/Submit-/Save-Workflow strukturgleich ist.
  - **Adressiert-Findings**: F-5.2-Finding-Nummer statt F-1.2-Finding-Nummer.
- **Aufwand wird übernommen**, außer Sub-Thread sieht klaren Grund für Abweichung (z.B. F-1-Pattern war XS weil Helper schon vorhanden, hier ist es S weil Helper für `markdown_converter` neu eingebunden werden muss — dann Begründung im Pattern-Block).
- **Pattern-Nummern-Konvention**: Sub-Thread vergibt F-5.3-eigene P1, P2, … nach thematischer Cluster-Reihenfolge. **Nicht** F-1-Nummern reproduzieren — die Pattern-Reihenfolge folgt der F-5.2-Cluster-Struktur (Cluster 1 zuerst, dann Cluster 2, etc.). F-1-Korrespondenz wird in einer expliziten Spalte/Sub-Zeile pro Pattern-Block ausgewiesen (z.B. „**F-1-Korrespondenz**: P4 Save-Failure-`alert()`-Konvergenz").

**Konvergenz-Items aus F-5.2 die F-1-Übernahme nutzen** (Sub-Thread mappt im Pre-Flight):
- F1 (Save-`alert()` H4 Sev 3) → F-1.3 P4 Save-Failure-`alert()`-Konvergenz übernehmen.
- F2 (Save-Recovery H9 Sev 3) → F-1.3 P4-Verzahnungs-Aspekt oder eigenständig mit F-1-Recovery-Pattern.
- F4 (Empty-Submit H1 Sev 2) → F-1.3 P1 Empty-Submit-Pattern, mit Sev-2-Anpassung wegen Server-Flash-Roundtrip.
- F7 (Submit-Loading H1 Sev 2 ⚠️) → F-1.3 P9 Submit-Loading-Pattern, mit teil-übertragbar-Hinweis auf CSRF-Refresh-Roundtrip.
- F10 (File-Info-Display H1 Sev 1) → F-1.3 P12-Adaption: nicht KB/MB-Filename-Fallback sondern **Display-Aufbau insgesamt** (markdown_converter zeigt File-Info gar nicht — mehr als nur ein Konvention-Bruch).
- F13 (a11y Iframe H6 Sev 1) → F-1.3 P13 a11y-Annotations übernehmen, Iframe-spezifische Anker (`tabindex` / `role` / `aria-label`).
- F14 (Alert-Auto-Dismiss H1 Sev 1) → F-1.3 P7 Auto-Dismiss-Pattern, mit Sev-1-Anpassung wegen vorhandener Close-×-Button.

**Cross-Feature-H4-Sektion in der Patterns-Doc** wird strukturiert wie F-5.2's Cross-Feature-H4-Sektion: direkt-übertragbare Konvergenz-Items + bereits-konvergente F-1-Patterns + nicht-anwendbare + Konvergenz-Quote.

### 2. Befund 16 / F3 PDF-Gen-Error-Recovery als eigenes Pattern

Master-Annotation aus F-5.2 wird hier **als Pattern-Block ausgeformt**. Pattern-Mechanik:
- **Adressierte Findings**: F3 (PDF-Gen-Error-Re-Render H9 Sev 3 ⚠️) + BT1 (Re-Render-Context-Lücke).
- **UI-Pattern**: Statt `render_template('markdown_converter.html', markdown_text=...)` im Error-Branch von [app_pkg/markdown.py:246](app_pkg/markdown.py#L246) → `flash(<DE-Microcopy>, 'danger')` + `redirect(url_for('markdown_converter'))`. User landet auf der Markdown-Seite mit sichtbarem Error-Banner statt 500-Page.
- **Microcopy**: deutsch, max 2 Sätze, ohne Emoji. Sub-Thread formuliert konkret.
- **Helper-Reuse**: Flask `flash()` + bestehender Banner-Render im Template.
- **Aufwand**: XS (1-2 Zeilen Backend, kein Frontend-Touch).
- **Live-Verifikation-Status**: `🔥 Smoke-Pflicht in F5-IMPL` (PDF-Gen-Error forcieren via z.B. Theme-Datei-Manipulation oder Playwright-mock).
- **Verzahnung-Hinweis**: BT1 wird durch dieses Pattern aufgelöst, kein separater Bug-Ticket-Apply nötig.

Sub-Thread kann Pattern-Nummer und Cluster-Zuordnung wählen — Erwartung: gehört in **Cluster 3 (Error-Recovery-Pfade)**.

### 3. Walkthrough-Disposition: kein expliziter Master-Walkthrough vor F5-PATTERNS

Master-Entscheidung: F5-PATTERNS startet ohne Live-Walkthrough. Die 8 ⚠️ code-only-Findings tragen `🔥 Smoke-Pflicht in F5-IMPL`-Sub-Tag, F5-IMPL-Sub-Thread verifiziert vor Pattern-Apply (analog F-3-IMPL- und F-4-IMPL-Methodik).

**Begründung**:
- F-3-IMPL und F-4-IMPL haben gezeigt: Smoke-Pflicht im Implementation-Sprint funktioniert sauber, weil Sub-Thread direkten Zugriff auf den Container hat und schneller verifiziert als Master-Walkthrough+Bericht-Roundtrip.
- Sev-Verteilung im aktuellen Sprint: **kein Sev 4**, nur 3× Sev 3 (F1 / F2 / F3) — niedrigere Walkthrough-Dringlichkeit als F-4.3 (2× Sev 4 Cancel-Lüge).
- F3 (PDF-Gen-Error) hat ohnehin 🔥 Smoke-Pflicht via Pattern-Block (Master-Annotation 2).

**Konsequenz für Pattern-Vorschläge**: Patterns für F5/F6/F7/F8/F11/F12/F15 tragen `🔥 Smoke-Pflicht in F5-IMPL` mit explizitem Smoke-Mechanik-Hinweis pro Pattern (analog F-4.3 für F5/F8/F16):
- **F5 Theme-CSS-Fetch-silent (H9 Sev 2)**: Smoke via DevTools 404-Block für Theme-CSS-URL.
- **F6 Sample-Merge-Bug (H1 Sev 2)**: Smoke nach BT2-Apply via Page-Load auf Empty-Markdown-State.
- **F7 Submit-Loading-kurz (H1 Sev 2)**: Smoke via DevTools-Network-Throttle für CSRF-Roundtrip-Sichtbarkeit.
- **F8 Two-Dark-Modes (H4 Sev 2)**: Smoke via Theme-Toggle ↔ Reader-Dark-Reihenfolge mit unterschiedlichen Reihenfolgen.
- **F11 Width-Buttons-Initial (H6 Sev 1)**: Smoke via Reader-Mode-Toggle + Width-Visual-Vergleich.
- **F12 Reader-Mode-Persistenz (H6 Sev 1)**: Smoke via Reload-nach-Reader-Active.
- **F15 CSRF-Refresh-Race (H9 Sev 1)**: Smoke via DevTools-CSRF-Manipulation oder lange-Pausen-Reproduktion.

### 4. Reader-Mode-Designentscheidungen — pragmatische Default-Wahl im Pattern

F8 (Two-Dark-Modes), F9 (Reader-Mode-Persistenz), F11 (Width-Buttons-Initial), F12 (Theme/Reader-Persistenz) sind alle Reader-Mode-Visual-/State-Findings ohne F-1-Korrespondenz und ohne Live-Verifikation. Der Sub-Thread soll für jedes Pattern eine **pragmatische Default-Mechanik** vorschlagen (eine konkrete Lösung, nicht „Variante A oder B"):

- **F8 Two-Dark-Modes**: Default-Vorschlag — Reader-Dark übersteuert Theme-Dark wenn Reader aktiv (lokaler Scope-Sieg). Begründung im Pattern.
- **F9 Reader-Mode-Persistenz**: Default-Vorschlag — `localStorage` für Reader-On/Off-State plus Width-Auswahl plus Reader-Theme. Default-Mechanik analog F-4.3 P4 Browser-Reload-Recovery (existiert bereits als Helper-Pattern).
- **F11 Width-Buttons-Initial**: Default-Vorschlag — bei Reader-Aktivierung der zuletzt-genutzte Width oder „Mittel" als Default mit `aria-pressed`-Markierung; vor Reader-Aktivierung sind Buttons disabled+grau (analog F-4.3 P12 Tab-Disabled).
- **F12 Theme/Reader-Persistenz**: Default-Vorschlag — Theme + Reader-State aus `localStorage` rehydrieren beim Page-Load. Verzahnt mit F9 (gemeinsamer `localStorage.markdown.viewState`-Key).

Diese Default-Vorschläge werden im Pattern-Block als „Master-Default-Wahl" ausgewiesen — Sub-Thread kann abweichen wenn Begründung gut, aber **soll keine Variante-A/B/C-Liste** schreiben.

---

## Phase 1 — Patterns + Microcopy

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Findings-Doc + Inventur-Doc komplett lesen**.
3. **Methodik-Vorlagen lesen**: F-1.3 (primäre Übernahme-Quelle, **muss komplett gelesen werden**), F-2.3, F-3.3, F-4.3.
4. **`_utils.js`-Helper-Bestand verifizieren**: Patterns sollen vorhandene Helper nutzen — neue nur mit Begründung am Doc-Ende.
5. **F-1-Korrespondenz-Mapping vorbereiten**: für jeden F-5.2-Finding mit F-1-Korrespondenz das passende F-1-Pattern in F-1.3 nachschlagen und Übernahme-Notiz vorbereiten.

**Pattern-Aufgabe**:

Für jeden Finding (oder konsolidierte Finding-Gruppe) ein Pattern-Block mit:

- **Pattern-Nummer** (P1, P2, …) nach thematischer Cluster-Reihenfolge.
- **Adressiert Findings**: Liste mit H + Sev + linked BTs.
- **F-1-Korrespondenz**: Pattern-Code aus F-1.3 (z.B. „F-1.3 P4") wenn übertragen, oder „—" wenn markdown-spezifisch.
- **Live-Verifikation-Status** (NEU für F-5.3 wegen ⚠️ code-only-Findings): `🔥 Smoke-Pflicht in F5-IMPL` für Patterns die F3/F5/F6/F7/F8/F11/F12/F15 adressieren mit explizitem Smoke-Mechanik-Hinweis (siehe Master-Annotation 3); sonst leer (außer es gäbe live-verifizierte Patterns — hier nicht der Fall).
- **UI-Pattern-Beschreibung**: 2-4 Sätze, konkrete Mechanik. Bei Übernahme: kurz vermerken „aus F-1.3 P-X übernommen" + ggf. markdown-spezifische Anpassung.
- **Visuelle Hinweise**: vorhandene CSS-Klassen aus Neomorphism + `.sr-only` aus F-3-IMPL nutzen.
- **Microcopy** (deutsch): exakte Strings für Banner, Status-Labels, Tooltips. Nach Microcopy-Regeln. Bei Übernahme: F-1.3-Strings recyclen wo passend; bei markdown-spezifischen Findings konkret formulieren (insbesondere für Befund-16/F3-Banner).
- **Helper-Reuse**: welche `_utils.js`-Helper. **Erwartung hoch** wegen 47% H4-Quote — Patterns für Cluster 1 (F1, F2, F14, F13, F10) bauen auf etablierten Helpern (`showAlert`, `showToast`, `safeJSON`, `formatFileSize`).
- **Aufwand**: XS / S / M / L mit Begründung. Bei Übernahme: F-1.3-Aufwand verifizieren oder Abweichung begründen.
- **Impact-Score**: `Sev × 5 / Aufwand-Gewicht`.
- **Konsolidierung-Hinweis** falls mehrere Findings adressiert.

**Konsolidierungs-Logik** (analog F-1.3 / F-2.3 / F-3.3 / F-4.3):

- **F1 (Save-`alert()` H4) + F2 (Save-Recovery H9)**: stark konvergent — selber Code-Pfad in `saveMarkdownToLibrary`. Konsolidierung **wahrscheinlich ein Pattern** mit zwei Heuristik-Aspekten in der Beschreibung.
- **F11 (Width-Buttons-Initial) + F12 (Reader-Mode-Persistenz) + F9 (Reader-Mode-Persistenz)**: Reader-Mode-State-Cluster — vermutlich **ein großes Pattern** „Reader-Mode-State-Persistenz und Visual-Defaults" oder zwei (Persistenz + Initial-Active separat).
- **F8 (Two-Dark-Modes)**: kann eigenständig oder mit F9/F11/F12 konsolidiert werden — Sub-Thread entscheidet pragmatisch.
- **F-1-übernommene Patterns** (P4-Korrespondenz F1+F2, P7-Korrespondenz F14, P13-Korrespondenz F13, P12-Adaption F10) bleiben als **eigene Pattern-Blöcke**, nicht zu Cluster-Bündeln zusammenschmelzen — F-1-Pattern-Identität soll erkennbar bleiben.

**Erwartete Pattern-Anzahl**: 9-12 Patterns aus 15 Findings + 5 BTs (~20-30% Konsolidierung — höher als F-4.3 25% wegen F1/F2-Konvergenz und Reader-Mode-Cluster, aber niedriger als F-2.3 50% wegen Schwester-Feature-Übernahme die 1:1 statt zu konsolidieren bedeutet).

**Output-Doc**: `docs/ui_patterns_markdown_converter_2026-05.md`. Struktur 1:1 wie F-1.3 / F-2.3 / F-3.3 / F-4.3 plus Schwester-Feature-Sektion:

1. Header mit Findings-Quelle, Sprint-Datum, Methodik-Hinweis, **Schwester-Feature-Übernahme-Notiz**.
2. **Pattern-Blöcke** P1 bis Pn (nach Cluster-Reihenfolge).
3. **Cluster-Vorschlag für F5-IMPL**: 1-3 Implementations-Cluster vorgeschlagen. **Default-Empfehlung**: 1-Sprint mit 2-3 Sub-Batches (Cluster 1 H4-Helper-Reuse → Cluster 2 Reader-Mode-State → Cluster 3 Error-Recovery + Cluster 4 Pre-Check), gestützt auf F-3-IMPL-Lehre (15 Patterns in 3 Sub-Batches funktionierte) und F-4-IMPL-B-Lehre (10 Patterns in einem Sweep funktionierte wenn verkoppelt). Bei <13 Patterns ist 1-Sprint pragmatisch.
4. **Top-5 Quick-Wins**: Tabelle nach Impact-Score absteigend.
5. **Smoke-Pflicht-Übersicht**: Liste der Patterns mit `🔥 Smoke-Pflicht in F5-IMPL`-Sub-Tag (8 Findings: F3, F5, F6, F7, F8, F11, F12, F15) plus Smoke-Mechanik pro Pattern.
6. **Cross-Feature-H4-Sektion** (analog F-5.2-Struktur): direkt-übertragbare Konvergenz-Items mit F-1.3-Pattern-Code, bereits-konvergente F-1-Patterns (5 Items: P3 / P6 / P8 / P10 / P11) als positives Inventar, nicht-anwendbare F-1-Patterns (2 Items: P2 / P5).
7. **Helper-Vorschlags-Sektion** am Doc-Ende: falls beim Pattern-Schreiben fehlende Helper auffallen (z.B. ein generischer `saveViewState(key, state)`-Helper für Reader-Mode-Persistenz + andere Local-Storage-Persistenzen). Nicht still mit-anlegen — F5-IMPL-Sub-Thread entscheidet beim Cluster-Schnitt.

Nach Phase 1: STOP — Bericht. Statistik (Pattern-Anzahl, Konsolidierungs-Quote, Aufwand-Verteilung, Smoke-Pflicht-Anzahl, F-1-Übernahme-Anzahl, Cluster-Vorschlag, Top-5-Quick-Wins).

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Pattern-Doc nochmal und prüft:

1. **Vollständigkeit**: jeder der 15 Findings ist adressiert. BT1+BT2 in Patterns erwähnt (F3-Pattern für BT1, F6-Pattern für BT2), BT3+BT4+BT5 als „pure Bug-Tickets nicht in F-5.3" notiert.
2. **F-1-Übernahme-Disziplin**: Patterns mit F-1-Korrespondenz tragen explizit „F-1-Korrespondenz: P-X" + Übernahme-Notiz; Code-Anker stimmen (`markdown_converter` statt `document_converter`); Microcopy ist deutsch und folgt Microcopy-Regeln.
3. **Live-Verifikation-Status-Konsistenz**: 8 Patterns für F3/F5/F6/F7/F8/F11/F12/F15 tragen `🔥 Smoke-Pflicht in F5-IMPL` mit Smoke-Mechanik-Hinweis. Übersichts-Sektion und Pattern-Block-Tags stimmen überein.
4. **Reader-Mode-Default-Wahl**: Patterns für F8/F9/F11/F12 enthalten konkrete Default-Mechanik (nicht Variante-A/B/C-Liste) — siehe Master-Annotation 4.
5. **Befund-16-Pattern**: F3+BT1 in einem Pattern-Block mit `redirect()` + `flash()`-Mechanik, XS Aufwand, Smoke-Pflicht. Master-Annotation 2 aufgegriffen.
6. **Microcopy-Regeln**: Stichprobe — Banner-Strings (≤2 Sätze), Status-Labels (≤3 Wörter), Empty-States (≤3 Sätze).
7. **Impact-Score-Konsistenz**: Top-5-Tabelle stimmt mit Pattern-Block-Score-Angaben überein.
8. **Cross-Feature-H4-Sektion**: 5 bereits-konvergente F-1-Patterns gelistet, 2 nicht-anwendbare gelistet, Konvergenz-Quote spiegelt F-5.2 (86%).
9. **Disziplin**: keine konkreten Code-Diffs, kein Bug-Fix, keine Implementations-Schritte.

Nach Phase 2: STOP — Bericht. „Pattern-Doc konsistent" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-5.3 / Stufe 3: patterns + microcopy of markdown_converter".
- Body: Statistik (Pattern-Anzahl, Konsolidierungs-Quote, Aufwand-Verteilung, Smoke-Pflicht-Anzahl, F-1-Übernahme-Anzahl, Cluster-Vorschlag, Top-5-Quick-Wins, Reader-Mode-Default-Wahl-Hinweis).
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints. Wenn der Auto-Mode-Classifier blockt: Master pusht von Hand. (Siehe Memory `feedback_push_is_normal.md`.)

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — eine Output-Datei (`docs/ui_patterns_markdown_converter_2026-05.md`), Pattern-Blöcke + Microcopy + Aufwandsschätzung + Cluster-Vorschlag, kein Code-Touch, keine Tests, kein Smoke. Schwester-Feature-Übernahme aus F-1.3 senkt den Pattern-Konstruktion-Aufwand vs. F-4.3, aber F-1.3-Cross-Read und Per-Pattern-Übernahme-Disziplin bleiben Aufwand.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim F-1-Pattern-Übernehmen ein **Aufwand-Mismatch** entsteht (F-1-Pattern war XS weil Helper schon im `document_converter.js` eingebunden, hier ist es S weil neuer Helper-Import nötig): kurz im Pattern-Block notieren, neue Aufwand-Stufe begründen.
- Wenn beim Pattern-Schreiben ein **Helper-Vorschlag** sich als sinnvoll für mehrere Patterns zeigt (z.B. `saveViewState(key, state)` für F9+F11+F12): in Helper-Vorschlags-Sektion am Doc-Ende sammeln, nicht still mit-anlegen.
- Wenn ein **Reader-Mode-Default-Vorschlag** aus Master-Annotation 4 sich beim Pattern-Schreiben als technisch problematisch erweist (z.B. localStorage-Race-Condition mit Theme-CSS-Fetch): kurz Bedenken im Pattern-Block vermerken — F5-IMPL muss dann eine Alternative finden, **nicht** F-5.3 die Diskussion auflösen.
- Wenn ein BT (BT3/BT4/BT5) sich beim Pattern-Schreiben als nahegelegen herausstellt (z.B. BT3 `updateStyle()` doppelt wäre nahegelegen wenn ein Reader-Mode-Pattern den Style-Setter-Pfad anfasst): in der Pattern-Doc kurz „BT-X-Verzahnung" notieren, F5-IMPL entscheidet ob mit-fixen oder separater Bug-Pass.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F5-PATTERNS ☑ done 2026-05-XX → commit `<hash>`. Patterns-Doc unter `docs/ui_patterns_markdown_converter_2026-05.md`. <N> Patterns (XS: a, S: b, M: c, L: d), <K> Cluster vorgeschlagen, Top-5-Quick-Wins-Score-Range <X.0–Y.0>, <S>× 🔥 Smoke-Pflicht für F5-IMPL, F-1-Übernahme-Anzahl: <U> von <T> H4-Findings. Verbleibende Sequenz: F5-IMPL → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F5-PATTERNS" raus → Erledigt-Liste; Master fügt F5-IMPL als Position 1 beim nächsten Dispatch hinzu — Sub-Thread fügt es **nicht** selbst hinzu (Master-Edit-Zone).
- **Memory**: nichts erwartet — Pattern-Methodik etabliert. Falls Schwester-Feature-Übernahme-Lehren auftauchen (z.B. „F-1-Pattern-1:1-Übernahme-Methodik bei Schwester-Feature-Wellen"): `feedback_*.md` schreiben.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Pattern-Methodik etabliert, Schwester-Feature-Übernahme + Befund-16-Pattern + Walkthrough-Disposition + Reader-Mode-Default-Wahl in Master-Annotationen oben verankert.)_
