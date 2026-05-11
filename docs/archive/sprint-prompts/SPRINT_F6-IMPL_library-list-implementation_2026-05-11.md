# Sprint F6-IMPL — F-6 Implementation `library` List-View (alle 14 Patterns, 3 Sub-Batches)

**Datum**: 2026-05-11

**Ziel**: Alle 14 Patterns aus F6-PATTERNS in einem Sprint implementieren, intern strukturiert als drei Sub-Batches (Silent-Failure-Familie+Cross-Feature-H4 → List-View-State → Polish-Long-Tail). Aufwand-Verteilung XS-lastig (XS=8 / S=5 / M=1 / L=0) wegen Geschwister-Feature-Übernahme aus F-3.3. Die Sub-Batch-Mechanik dient als Schutz vor Überforderung — nach jedem Sub-Batch ein STOP-Punkt mit Bericht. **A1/A2-Split-Fallback** für Sub-Batch A (7 Patterns inkl. P13-Synergie) wenn Sub-Thread Verkopplung als zu groß einschätzt.

**Vorbedingung**:
- Pytest 66/66 grün auf `main`. Letzter Code-Touch: F6-PATTERNS (commit `d920822`, 2026-05-11).
- **Eingabe**: [docs/ui_patterns_library_list_2026-05.md](docs/ui_patterns_library_list_2026-05.md). Sub-Thread liest **vor jedem Sub-Batch** die zugehörigen Pattern-Blöcke nochmal.
  - **14 Pattern-IDs P1–P14**.
  - **8 Smoke-Pflicht-Patterns**: P2, P3, P4, P6, P8, P11, P13, P14 (alle 9 ⚠️ code-only-Findings aus F-6.2 adressiert plus P2 list-spezifisch).
  - **Drei Pflicht-Live-Master-Smoke-Patterns** (🔥🔥-Marker in F-6.3): P2 Copy-Full-Content / P3 Auto-Save Favorite-Toggle / P11 Card-Datum DE-Lokalisierung.
  - **F-3-Übernahme** 5 direkt (P1/P3/P4/P5/P6) + 1 teil (P11) = 6 Patterns mit F-3.3-Korrespondenz.
  - **Konsolidaten**: P3 = F2+F3 (Auto-Save-Familie), P4 = F4+F5 (Delete-Familie), P8 = F9+F11 (Search Submit-Required).
  - **Helper-Vorschlag (nicht still angelegt)**: `format_card_datetime`-Jinja2-Filter analog `file_size` aus F3-IMPL für P11 — F6-IMPL entscheidet beim Apply (Master-Annotation 4 unten).
  - **List-View-State-Default-Wahl** aus F-6.3 in P7 (server-side filter-aware Empty-State) und P8 (Submit-Required-mit-DE-Hint statt Live-Debouncing) verankert.
- **Methodik-Vorlagen** (Multi-Pattern-Sprints):
  - F-3-IMPL commits `843574b` / `40dd02e` / `b3e666a` (15 Patterns in 3 Sub-Batches, drei separate Commits via temp-Backup→revert→stepwise-replay-Mechanik).
  - F-4-IMPL-B commit `3ef7f9e` (10 Patterns in einem Sweep ohne Sub-Batch-Split).
  - F-5-IMPL commits `07e9aa6` (A+B bundled) + `9e6999c` (C) (13 Patterns mit Schwester-Feature-Übernahme — **direkte Vorlage** für Geschwister-Feature-Übernahme-Mechanik).
  - F-3.3 als **Pattern-Übernahme-Quelle** für die 6 F-3-übernommenen Patterns (P1/P3/P4/P5/P6/P11).
- **Test-Coverage**: aktuelle Suite 66/66 grün. Erwartete neue Tests:
  - 1-2 für P3 (Auto-Save-Failure-Backend-Contract — `test_api_update_favorite_failure_returns_5xx` o.ä.).
  - 1-2 für P4 (Delete-Failure-Backend-Contract — analog F-3-IMPL `test_api_delete_conversion_404_for_other_users_conversion`).
  - 0-1 für P11 (Filter-Roundtrip / Datum-Format-Test wenn `format_card_datetime`-Jinja-Filter extrahiert).
  - 0-1 für P8 (Search-Submit-Backend-Contract falls server-side-Änderung).
  - Erwartete Final-Anzahl: **67–70 Tests grün**.
- **Memory-Layer-Pflicht-Lese**: `feedback_no_silent_fixes.md` (Bugs als Findings, nicht inline fixen), `feedback_pragmatic_merge.md` (Risiko-Kalibrierung), `feedback_push_is_normal.md` (Push direkt nach Commit OK), `feedback_helper_reuse_design_choice.md` (Helper-Reuse-Drift mit begründeter Design-Wahl ≠ H4-Verletzung — keine künstliche Drift), `feedback_smoke_beats_pattern_text.md` (Smoke gewinnt bei Pattern-Text-Spannung), `reference_converter_dep_bump_constraints.md` (pytest im Container).

**Out-of-scope**:
- Weitere F-N-Wellen für andere Features (`mermaid_converter`, `login`).
- WAVE-CLOSE.
- **F-3.2 BT7+BT8** (textarea-escape, window.open-noopener) — gehören zu library_detail, nicht zu library-List-View.
- **library_detail-Code-Refactors** — F-3 ist abgeschlossen für die Detail-View; library-List-View ist eigenes Feature. Wenn beim Code-Reading library_detail-Auffälligkeiten erscheinen: nur kurz im Bericht erwähnen, nicht in den Sprint-Diff.
- F-N-Backlog-Items (P3-Reminder): drei pre-existierende EN-Strings in `app_pkg/markdown.py`, P8-Master-Smoke (PDF-Gen-Error) — andere Welle.

---

## Master-Annotation (vorab eingebettet)

### 1. Sub-Batch-Strategie 3 Sub-Batches A/B/C, A1/A2-Fallback

| Sub-Batch | Patterns | Anzahl | Smoke-Pflicht | Begründung |
|-----------|----------|--------|---------------|------------|
| **A — Silent-Failure-Familie + Cross-Feature-H4** | P1, P2, P3, P4, P5, P6 (+ P13 Synergie wenn nahegelegen) | 6 (+1) | P2, P3, P4, P6 | Schafft Banner-Mountpoint-Vorbedingung (P1) + behebt Daily-Usage-Hotspot (Copy-Btn, Auto-Save, Delete) + DE-Microcopy-Sweep + Toast-Level-Konvergenz. P13-Synergie weil aria-live-Region beim Banner-Mountpoint-Touch nahegelegen. **A1/A2-Split-Fallback** wenn 6+1 zu groß: A1 = P1+P3+P4 (Banner-Familie mit P15-Vorbedingung) / A2 = P2+P5+P6 (+P13). |
| **B — List-View-State** | P7, P8 | 2 | P8 (code-evident) | Eng gescoped (Empty-State + Search-Submit-Required). P7 server-side filter-aware, P8 Frontend-DE-Hint. Master-Default-Wahl aus F-6.3 verankert. |
| **C — Polish-Long-Tail** | P9, P10, P11, P12, P13, P14 | 6 | P11, P13, P14 | Kosmetisch + a11y + Locale. P11 Pflicht-Live-Master-Smoke. **P13 kann aus A vorgezogen werden** wenn Sub-Batch-A-Banner-Touches die aria-live-Region ohnehin anfassen — Sub-Thread berichtet im A-Bericht ob P13 mit-gefoldet oder verbleibt für C. |

**Pflicht-Reihenfolge**: A → B → C, **ohne Auslassen**. STOP-Punkt mit Bericht nach jedem Sub-Batch — Master kann zwischen Sub-Batches korrigieren oder Sprint abbrechen.

**Holistic vs. sequenziell pro Sub-Batch**: Sub-Thread entscheidet pragmatisch.
- **Sub-Batch A (6+1 Patterns, Helper-Reuse + Banner-Mountpoint + Konsolidaten)**: holistic empfohlen mit P1 zuerst (struktureller Vorbedingung-Block, analog F-3-IMPL Sub-Batch A P15 → P1+P3).
- **Sub-Batch B (2 Patterns, eng gescoped)**: atomic — beide in einem Apply-Schritt.
- **Sub-Batch C (6 Patterns, lose verkoppelt)**: sequenziell, P11 zuerst als 🔥🔥-Pflicht-Live-Master-Smoke-Anchor.

### 2. Apply-Reihenfolge-Empfehlung innerhalb Sub-Batch A

Sub-Thread folgt **default**, kann pragmatisch abweichen mit Begründung im Bericht (analog F-3-IMPL-Mechanik):

1. **P1 zuerst** (struktureller Vorbedingung-Block — Banner-Mountpoint `<div id="library-alert-container">` ins Template wo F-3-IMPL P15-Pattern ihn auch hat). P3+P4 brauchen den Mountpoint.
2. **P5 zweitens** (DE-Microcopy-Sweep der 2 EN-Strings in `library.js` plus weiterer EN-Strings die beim Code-Reading auffallen). Schafft DE-Foundation für nachfolgende Patterns.
3. **P3 + P4 drittens** holistic (Auto-Save + Delete Failure-Banner mit `showAlert` + `safeJSON`-Wrap aus F-3.3-Übernahme). Identischer Code-Pfad in `library.js`.
4. **P6 viertens** (Toast-Level pro Call-Site für Copy-Failure — kleine Anpassung).
5. **P2 fünftens** (Copy-Full-Content via `data-content`-Attribut-Embed — list-spezifisches Pattern, eigenes Apply-Schritt; siehe Master-Annotation 5 für Trade-off-Re-Check).
6. **P13 sechstens** (aria-live-Region) — **wenn nahegelegen** beim Banner-Mountpoint-Touch aus P1 mit-folden. Sonst nach C verlegen.

### 3. Smoke-Pflicht-Kalibrierung — Drei 🔥🔥-Pflicht-Live-Master-Patterns von 8

8 Smoke-Pflicht-Patterns (P2, P3, P4, P6, P8, P11, P13, P14) ist viel. F-6.3 hat **Drei-Pflicht-Live-Master-Smoke** kalibriert:

**🔥🔥-Pflicht-Live-Master-Smoke** (vor Apply oder direkt nach Apply, Sub-Thread entscheidet):
- **P2 (Copy-Full-Content)** — Copy-Paste-Verifikation mit langem Content (>200 char) im Browser. Test: Card mit langer Conversion auswählen, Copy-Btn klicken, in Editor pasten, prüfen ob Full-Content (nicht 200-char-Preview).
- **P3 (Auto-Save Favorite-Toggle Failure-Banner)** — DevTools-Network-Throttle Offline → Favorite-Toggle-Klick → Erwartung: Failure-Banner statt silent-fail.
- **P11 (Card-Datum DE-Lokalisierung)** — Browser-Inspektion der Card-Datum-Anzeige (deutsche Monatsabkürzung „Mär" statt englischer „Mar"). Direkter Browser-Test, nicht nur Server-Locale-Inspektion.

**Code-evident-verifiziert** (Container-side ohne Live-Browser-Smoke):
- P4 (Delete-Failure-Banner) — Code-Reading + Container-Smoke (analog F-3 P3 mit DevTools-Throttle code-evident verifizierbar).
- P6 (Toast-Level Copy-Failure) — Code-Reading der Toast-Call-Sites.
- P8 (Search-Submit-Required mit DE-Hint) — Code-Reading der Search-Handler + Template-Render-Smoke.
- P13 (aria-live) — Code-Reading der DOM-Annotations.
- P14 (Card-Hover-Lift) — Code-Reading der CSS-Klassen + JS-Click-Handler.

Sub-Thread berichtet pro 🔥🔥-Pflicht-Pattern Smoke-Ergebnis im Sub-Batch-Bericht.

### 4. Helper-Vorschlag `format_card_datetime`-Jinja2-Filter

F-6.3 hat den **Server-side Helper-Vorschlag** `format_card_datetime`-Jinja2-Filter analog `file_size` aus F3-IMPL dokumentiert. F6-IMPL entscheidet beim Apply:

**Master-Empfehlung**: **extrahieren** wenn:
- P11-Apply die Datum-Logik mehr als 2-3 Zeilen Jinja2-Template-Code wird, ODER
- Die Logik braucht Python-DateTime-Manipulation die im Template-`strftime`-Pfad ungelegen ist, ODER
- Analog `file_size`-Pattern: server-side Helper mit klarer Single-Responsibility ist sauberer als Inline-`strftime` mit Locale-Switch.

**Alternative**: Inline Jinja2 `{{ created_at.strftime('%d. %b %Y') }}` mit serverseitiger Locale-Setzung wenn 1-Zeile-Pfad reicht.

Single-Call-Site (Card-Datum in `library.html`) plus klare Server-Side-Reuse-Mechanik = Extraktion analog `file_size`-Filter OK. **Memory `feedback_helper_reuse_design_choice.md`** greift hier: extrahieren ist berechtigt wegen Methodik-Parität mit `file_size`-Pattern, nicht „zweite Call-Site"-Regel-Bruch.

Sub-Thread berichtet im P11-Apply-Bericht ob Extraktion gemacht oder Inline-Pfad gewählt mit Begründung.

### 5. `data-content`-Attribut-Embed für P2 — Trade-off-Pre-Flight-Check

F-6.3 hat **`data-content`-Attribut-Embed im Template-Render** als Default-Wahl für P2 (Copy-Full-Content) verankert, mit Skalierungs-Trade-off-Notiz: bei 20-Items/Page mit Full-Content im HTML könnte das HTML aufblähen.

**Master-Empfehlung Pre-Flight-Check**:
1. Sub-Thread inspiziert vor P2-Apply die typische `ConversionHistory.content`-Größe in der DB:
   - `docker exec converter-web python -c "from app_pkg.models import db, ConversionHistory; print('avg:', db.session.query(db.func.avg(db.func.length(ConversionHistory.content))).scalar(), 'max:', db.session.query(db.func.max(db.func.length(ConversionHistory.content))).scalar())"`
2. **Default-Wahl bleibt `data-content`-Embed** wenn typisches Content < ~10 KB und max < ~50 KB (Page mit 20 Items hat dann max ~1 MB HTML — vertretbar).
3. **Pragmatische Alternative**: neuer Backend-Endpoint `GET /api/library/<id>/content` mit On-Click-Fetch — wenn typisches Content > 50 KB oder max > 200 KB. Sub-Thread entscheidet beim Apply mit Bericht.

### 6. BT-Folde-Disposition

Alle 4 Bug-Tickets sind Finding-linked und werden via die Patterns mit-gelöst:
- **BT1 ↔ P3** (toggleFavorite-Errors) — Auto-Save-Pattern löst Banner-Anzeige + safeJSON-Wrap. Mit-gelöst.
- **BT2 ↔ P4** (deleteConversion-Errors) — Delete-Pattern löst Banner-Anzeige + safeJSON-Wrap. Mit-gelöst.
- **BT3 ↔ P2** (Copy-200char-Quelle) — Copy-Full-Content-Pattern löst Daten-Quelle-Bug. Mit-gelöst.
- **BT4 ↔ P6** (Toast-Level Copy-Failure) — Toast-Level-Pattern löst Level-Konvergenz. Mit-gelöst.

**Keine separate BT-Apply-Disposition** nötig.

### 7. List-View-State-Default-Wahl-Validierung

F-6.3 Master-Annotation 5 hat Defaults verankert für P7 (server-side filter-aware Empty-State) und P8 (Submit-Required-mit-DE-Hint statt Live-Debouncing wegen URL-Persistierung-Design-Wahl). F6-IMPL-Sub-Thread:
- **Folgt Defaults beim Apply** als Default-Pfad.
- **Kann abweichen** wenn beim Apply technische Probleme auftreten (z.B. server-side `has_active_filter`-Boolean braucht Context-Erweiterung die der Template-Render nicht trivial passt).
- **Bericht-Pflicht** bei Abweichung mit Begründung. **Nicht** Variante-A/B/C-Diskussion eröffnen — bei Problem konkrete pragmatische Alternative wählen und im Bericht dokumentieren (Memory `feedback_smoke_beats_pattern_text.md`-Geist greift hier analog).

### 8. Memory-Disziplin

Sub-Thread liest **vor Pre-Flight** die fünf Memory-Einträge:
- `feedback_no_silent_fixes.md` — Aufgefallen-nicht-gefixt-Disziplin für Code-Reading-Befunde außerhalb der 14 Patterns.
- `feedback_pragmatic_merge.md` — Risiko-Kalibrierung beim Sub-Batch-Schnitt.
- `feedback_push_is_normal.md` — Push direkt nach Commit OK, kein Sign-off-Gate.
- `feedback_helper_reuse_design_choice.md` — keine künstliche Helper-Drift; `format_card_datetime`-Extraktion ist berechtigt wegen `file_size`-Methodik-Parität.
- `feedback_smoke_beats_pattern_text.md` — Smoke gewinnt bei Pattern-Text-Spannung; bei Reader-Mode-Default-analoger Spannung pragmatische Alternative wählen.

---

## Phase 1 — Implementation (drei Sub-Batches mit STOP-Punkten)

### Pre-Flight (vor Sub-Batch A)

1. `pytest tests/` im Container — muss **66/66 grün** sein. (Container-side per `reference_converter_dep_bump_constraints.md`.)
2. `git status -s` → clean tree erwartet.
3. **Pattern-Doc + Findings-Doc + Inventur-Doc** kurz überfliegen.
4. **F-3.3 Patterns-Doc** lesen für Sub-Batch-A-Übernahme-Vorbereitung — die F-3-Korrespondenz-Spalte in F-6.3 zeigt welche F-3.3-Pattern-Blöcke 1:1 mit Code-Anker-Anpassung übernommen werden.
5. **`_utils.js`-Helper-Bestand verifizieren**: `grep -n "^function\|window\." static/js/_utils.js` — erwartet alle etablierten Helper plus `saveViewState`/`loadViewState` aus F5-IMPL.
6. **`P2 Pre-Flight-Check`**: typische `ConversionHistory.content`-Größe in DB inspizieren (siehe Master-Annotation 5). Default-Wahl `data-content`-Embed wenn typisches < 10 KB.

---

### Sub-Batch A — Silent-Failure-Familie + Cross-Feature-H4 (6+1 Patterns)

**Patterns**: P1 (Banner-Mountpoint — F-3.3 P15-Übernahme), P2 (Copy-Full-Content — list-spezifisch + BT3), P3 (Auto-Save Favorite-Toggle Failure-Banner — F-3.3 P1+P14-Übernahme + BT1), P4 (Delete-Failure-Banner — F-3.3 P3+P14-Übernahme + BT2), P5 (DE-Microcopy-Sweep — F-3.3 P6-Übernahme), P6 (Toast-Level Copy-Failure — F-3.3 P8-Übernahme + BT4). **P13 (aria-live) wenn nahegelegen mit-folden**.

**Mechanik (Holistic-Rewrite empfohlen)**:

1. **P1 zuerst** (Banner-Mountpoint-Vorbedingung): `<div id="library-alert-container" class="…"></div>` ins Template `library.html` an Stelle die `showAlert` aus `_utils.js` als Anchor erwartet (analog F-3-IMPL `<div id="detail-alert-container">`).
2. **P5 zweitens** (DE-Microcopy-Sweep): 2 EN-Strings in `library.js` (BACKLOG-Reminder) plus weitere EN-Strings die beim Code-Reading auffallen, DE-übersetzen. F-3.3-P6-Strings recyclen wo passend.
3. **P3 + P4 holistic** (Auto-Save + Delete Failure-Banner): `toggleFavorite` und `deleteConversion` mit `safeJSON`-Wrap + `r.ok`-Check + `showAlert(container, 'danger', …)`-Aufruf für Failure-Pfad. F-3.3-Code-Strings 1:1 wiederverwenden, Code-Anker auf `library.js` umlegen.
4. **P6 viertens** (Toast-Level Copy-Failure): `showToast`-Call-Site für Copy-Failure von `success`/Default-Level auf `danger`-Level umstellen analog F-3.3 P8.
5. **P2 fünftens** (Copy-Full-Content): `data-content`-Attribut im Card-Template-Render einbauen (Pre-Flight-Check entscheidet ob Embed oder Backend-Endpoint), Copy-Handler in `library.js` liest `event.target.dataset.content` statt `content[:200]`-Slice.
6. **P13 sechstens** (aria-live-Region für Card-Remove + Banner-Updates): wenn beim Banner-Mountpoint-Touch aus P1 nahegelegen mit-folden (mit `aria-live="polite"` auf `#library-alert-container`). Sonst nach Sub-Batch C verlegen — Sub-Thread berichtet.

7. **🔥🔥-Pflicht-Live-Master-Smoke**:
   - **P3 (Master-Smoke vor Apply oder direkt nach Apply)**: DevTools-Network-Throttle Offline → Favorite-Toggle → Erwartung: Failure-Banner sichtbar (nicht silent-fail). Bei nicht reproduzierbar (z.B. weil Favorite-Pfad sowieso `r.ok`-checked ist): STOP, Master fragen.
   - **P2 (Master-Smoke nach Apply)**: Card mit langem Content > 200 char auswählen, Copy-Btn klicken, in Editor pasten, Full-Content verifizieren.

8. Tests: 1-2 neue Tests für P3 + P4 in `tests/test_library.py` (Backend-Contract-Tests für Failure-Branches analog F-3-IMPL `test_api_update_conversion_rejects_non_dict_body` / `test_api_delete_conversion_404_for_other_users_conversion`).
9. `pytest tests/` muss grün bleiben (67-68 erwartet).

**Live-Smoke nach Sub-Batch A**:

- DE-Microcopy: alle EN-Strings im Browser sichtbar.
- Auto-Save-Failure-Pfad: DevTools-Throttle → Banner statt silent-fail.
- Delete-Failure-Pfad: code-evident verifiziert (Container-Smoke + Code-Reading).
- Copy-Full-Content: lange Conversion → Copy → Paste → Full-Content.
- showAlert-Reuse: keine raw `alert()`-Calls in `library.js` (`grep -c "alert(" static/js/library.js` = 0).
- P13-Disposition: Sub-Thread berichtet ob mit-gefoldet oder nach C verlegt.
- Toast-Level Copy-Failure: code-evident geprüft.

**STOP nach Sub-Batch A** — Bericht: welche der 6+1 Patterns durch, F-3-Übernahme-Disziplin (welche F-3.3-Code-Strings 1:1 wiederverwendet), Test-Stand, 🔥🔥-Smoke-Ergebnisse für P2+P3, P13-Disposition (mit-gefoldet oder verlegt), `data-content`-vs-Backend-Endpoint-Entscheidung für P2 mit Pre-Flight-Größen-Daten.

---

### Sub-Batch B — List-View-State (2 Patterns)

**Patterns**: P7 (Empty-State filter-aware — server-side im `library_view`-Render), P8 (Search Submit-Required mit DE-Hint — F9+F11 konsolidiert).

**Mechanik (atomic — beide Patterns in einem Apply-Schritt)**:

1. **P7 zuerst** (server-side filter-aware Empty-State): `library_view`-Render in `app_pkg/library.py` mit `has_active_filter`-Boolean (true wenn `request.args` Filter-Keys enthält). Template `library.html` rendert je nach `has_active_filter` entweder „Keine Treffer mit aktuellen Filtern. [Filter zurücksetzen]" oder „Library leer — erste Konvertierung starten".
2. **P8 zweitens** (Search-Submit-Required mit DE-Hint): Search-Input-Element in `library.html` mit `placeholder="Titel, Inhalt, Tags suchen … (Enter)"` und ohne `onchange`/`input`-Live-Handler. Form-Submit-Event triggert URL-Query-Update analog Filter-Pills. Live-Debouncing **nicht** implementieren (URL-Persistierung-Design-Wahl).
3. **🔥-Smoke (Container-Smoke)** für P8: Code-Reading der Search-Handler-Removal + Template-Render-Smoke (Placeholder-String sichtbar, kein Live-Submit).
4. Tests: optional 1 für P8 (Search-Submit-Backend-Contract wenn server-side-Änderung in `library_view` nötig) oder 1 für P7 (Filter-Roundtrip-Test).
5. `pytest tests/` muss grün bleiben (67-69 erwartet).

**Live-Smoke nach Sub-Batch B** (Master-Smoke optional, code-evident reicht):

- P7 Empty-State filter-aware: URL `?type=docx` mit leerem DB → „Keine Treffer mit aktuellen Filtern" mit Reset-Btn.
- P7 Empty-State global: leere DB ohne Filter → „Library leer".
- P8 Search Submit-Required: Eingabe ohne Enter → keine Filter-Anwendung, Hint sichtbar.

**STOP nach Sub-Batch B** — Bericht: beide Patterns durch, Master-Default-Wahl-Validierung (P7 server-side Branching + P8 Submit-Required) mit Code-Anker, Test-Stand, ob abweichende Default-Wahl mit Begründung.

---

### Sub-Batch C — Polish-Long-Tail (6 Patterns)

**Patterns**: P9 (favorites='' URL-Bereinigung — F12, XS), P10 (Type-Filter-Validation — F13, XS), P11 (Card-Datum DE-Lokalisierung — F14, S, 🔥🔥-Pflicht-Live-Master-Smoke + Helper-Vorschlag-Entscheidung), P12 (Per-Page-Size aus Query-Param — F15, M), P13 (aria-live — F16, XS, **falls nicht aus A vorgezogen**), P14 (Card-Hover-Lift-Animation — F17, XS).

**Pre-Flight für Sub-Batch C**:

Sub-Thread liest Patterns P9–P14 nochmal komplett und entscheidet:
- **Default**: sequenziell P11 zuerst (🔥🔥-Pflicht-Live-Master-Smoke + Helper-Extraktion-Entscheidung), dann P9 → P10 → P12 → P13 → P14.
- **P13-Disposition**: wenn aus Sub-Batch A nicht mit-gefoldet, hier mit aria-live-Annotations.

**Mechanik (sequenziell, P11 zuerst)**:

1. **P11 zuerst** (Card-Datum DE-Lokalisierung):
   - Sub-Thread entscheidet beim Apply: `format_card_datetime`-Jinja2-Filter extrahieren analog `file_size` (Master-Empfehlung wenn > 2-3 Zeilen Inline) ODER Inline `{{ created_at.strftime('%d. %b %Y') }}` mit Server-Locale-Setzung.
   - **🔥🔥 Master-Smoke**: Browser-Inspektion der Card-Datum-Anzeige — Erwartung „Mär 2026" statt „Mar 2026".
   - 1 optionaler Test (Filter-Roundtrip oder Datum-Format-Test).

2. **P9 zweitens** (favorites='' URL-Bereinigung): `library_view`-Render in `app_pkg/library.py` prüft auf `favorites=''`-Edge-Case und filtered aus URL-Query-Args raus (`favorites='1'` bleibt erhalten).
3. **P10 drittens** (Type-Filter-Validation): Backend-Allowlist für `type`-Filter-Werte analog F-013 (existierende Allowlists in app_pkg/audio.py). Invalide Werte → 400 mit DE-Microcopy.
4. **P12 viertens** (Per-Page-Size aus Query-Param): `per_page`-URL-Param mit Default 20 und Allowlist {10, 20, 50, 100}. Backend-Validation, Frontend-Dropdown.
5. **P13 fünftens** (aria-live wenn nicht aus A vorgezogen): aria-live="polite" auf relevanten Mountpoints + role="status" für Card-Remove-Announcements.
6. **P14 sechstens** (Card-Hover-Lift): CSS-Klasse für Hover-Visual-Lift + JS-Click-Affordance-Konsistenz. Code-Reading-Verifikation.

7. **🔥-Smoke-Mapping**:
   - **Master-Live-Smoke**: P11 (Card-Datum DE-Locale).
   - **Container-Smoke**: P13 (Code-Reading), P14 (Code-Reading).

8. Tests: 0-1 für P10 (Type-Filter-Allowlist-Test), 0-1 für P12 (per_page-Allowlist-Test). Final-Anzahl 67-70.
9. `pytest tests/` muss grün bleiben.

**Live-Smoke nach Sub-Batch C**:

- P11 (Master): Card-Datum DE-Locale.
- P9/P10/P12: code-evident verifiziert (Container-Smoke).
- P13 (Code-Reading): aria-live-Annotations sichtbar in DOM-Inspect.
- P14 (Code-Reading): Hover-Lift-CSS-Klasse + JS-Click-Handler konsistent.

**STOP nach Sub-Batch C** — Bericht: alle 6 Patterns durch, P11-Master-Smoke-Ergebnis, `format_card_datetime`-Helper-Entscheidung mit Begründung, Test-Final-Anzahl, ob beim Code-Reading neue Findings auffielen die nicht in F-6.2 dokumentiert waren.

---

## Phase 2 — Verify (gesamter Sprint)

1. `pytest tests/` im Container final grün (**67-70 erwartet**).
2. `grep -c "alert(" static/js/library.js` → 0 (alle `alert()`-Calls durch `showAlert` ersetzt).
3. `grep -n "showAlert\|safeJSON\|showToast\|saveViewState" static/js/library.js` zeigt Helper-Imports und Call-Sites.
4. **End-to-End-Smoke** (Master-Pflicht für die drei 🔥🔥-Pflicht-Live-Master-Smoke-Patterns):
   - **P2 (Master-Smoke)**: Card mit langem Content > 200 char auswählen → Copy-Btn → in Editor pasten → Full-Content verifizieren.
   - **P3 (Master-Smoke)**: DevTools-Throttle Offline → Favorite-Toggle → Failure-Banner sichtbar.
   - **P11 (Master-Smoke)**: Card-Datum DE-Locale sichtbar.
5. DevTools-Console final clean.
6. Sub-Batches A/B/C sind alle drei in `git diff` reflektiert — keine Sub-Batch-Auslassung.
7. F-3-Übernahme-Disziplin: `grep` zeigt dieselben Helper-Patterns wie `library_detail.js` (wo F-3 dort lebt).

Nach Phase 2: STOP — Bericht. Liste der gesmokten Pfade, Final-Test-Anzahl, etwaige Drift-Befunde.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- **Default: drei Commits, einer pro Sub-Batch** (analog F-3-IMPL und F-5-IMPL). Subjects z.B. „F-6 Sub-Batch A: Silent-Failure-Familie und Cross-Feature-H4 (P1-P6 plus P13)" / „F-6 Sub-Batch B: List-View-State (P7, P8)" / „F-6 Sub-Batch C: Polish-Long-Tail (P9-P14)".
- Falls Sub-Thread alle Sub-Batches in einem Commit bündeln will: Bericht-Pflicht, Default ist drei Commits.
- Bei A1/A2-Split: vier Commits.
- Branch: direkt auf `main` ist OK.
- `git push origin main`. Wenn Auto-Mode-Classifier blockt **oder** `.git/objects/<hash>`-SMB-Permission blockt (analog F-5.3 / F-6.1 / F-6.2 / F-6.3): Bericht, Master pusht von Hand via SSH zu Mintbox bzw. macOS-side (siehe Memory `feedback_push_is_normal.md`).

---

## Stop-Regel

Nach **jeder Phase** UND **nach jedem Sub-Batch** Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für F6-IMPL**:
- Wenn der Sub-Thread während Sub-Batch A merkt, dass die 6+1-Verkopplung zu groß ist: **A1/A2-Split-Fallback aktivieren** und nach A1 STOP, Master fragen.
- Wenn ein 🔥🔥-Pflicht-Pattern (P2/P3/P11) sich als nicht reproduzierbar erweist: STOP, **nicht** silent-fixen — Master entscheidet ob Pattern aus Scope fällt oder Befund neu zu bewerten ist.
- Wenn beim Code-Reading weitere Findings auffallen die nicht in F-6.2 dokumentiert sind: als „aufgefallen, nicht gefixt" in den Bericht — Memory `feedback_no_silent_fixes.md`.
- Wenn List-View-State-Default-Wahl aus F-6.3 Master-Annotation 5 beim Apply technische Probleme zeigt: konkrete pragmatische Alternative wählen, **nicht** Variante-A/B/C-Diskussion eröffnen, im Bericht begründen (Memory `feedback_smoke_beats_pattern_text.md`-Geist).

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**L** — 14 Patterns in einem Sprint, 3 Sub-Batches (mit A1/A2-Fallback), 8 Smoke-Pflicht-Patterns mit Drei-Pflicht-Live-Master-Kalibrierung, ggf. 1 neuer Jinja2-Filter (`format_card_datetime`), 2-4 neue Tests, mehrere Code-Bereiche (`static/js/library.js`, `templates/library.html`, `static/css/style.css`, `app_pkg/library.py`, ggf. `app_pkg/__init__.py` für Jinja-Filter, `tests/test_library.py`). Aufwand-Verteilung XS-lastig (XS=8 / S=5 / M=1) wegen Geschwister-Feature-Übernahme aus F-3.3 — strukturell vergleichbar mit F-5-IMPL (XS-lastig wegen Schwester-Feature-Übernahme).

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Code-Reading von `app_pkg/library.py` Auffälligkeiten in den **anderen** Routen auffallen (`api_*`-Endpoints die F-3-IMPL nicht abgedeckt hat): kurz im Bericht aufzählen, **nicht** in den Sprint-Diff.
- Wenn ein Helper-Vorschlag aus F-6.3 im Verlauf des Sprints überflüssig wird: kurz im Bericht aufzählen + nicht anlegen.
- Wenn beim Live-Smoke ein Befund aus F-6.1 plötzlich anders aussieht als beschrieben: im Bericht aufnehmen — code-deduced-Inventur kann irren.
- Wenn `format_card_datetime` beim Apply als überflüssig erkannt wird (z.B. Inline-Pfad reicht weil Locale-Switch trivial): Helper-Vorschlag verwerfen mit Begründung im Bericht.
- Wenn `data-content`-Embed beim P2-Apply doch problematisch wird (z.B. typisches Content > 50 KB nach Pre-Flight-Check): pragmatische Alternative (Backend-Endpoint) wählen mit Bericht.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F6-IMPL ☑ done 2026-05-XX → commits `<hash-A>` (Sub-Batch A), `<hash-B>` (Sub-Batch B), `<hash-C>` (Sub-Batch C). Pytest <neue Anzahl>/<neue Anzahl> grün. Live-Smoke clean für P2 + P3 + P11 (Master-Smoke). 14 Patterns implementiert + BT1/BT2/BT3/BT4 alle mit-gelöst + ggf. 1 neuer Jinja-Filter `format_card_datetime`. **F-6 strukturell abgeschlossen** für `library` List-View. Verbleibende Sequenz: F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F6-IMPL" raus → Erledigt-Liste mit allen 14 Pattern-IDs zur Traceability. Sektion „2. F-N…" rückt auf Position 1 mit Hinweis dass die nächste F-N-Welle ein Feature anpicken muss (`mermaid_converter` oder `login` — Master-Entscheidung) oder Master die Sequenz schließt mit WAVE-CLOSE.
- **Memory**: nur wenn übertragbare Lehren auftauchen. Defensiv. Geschwister-Feature-Übernahme-Disziplin ist schon in `feedback_helper_reuse_design_choice.md` indirekt verankert.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — alle Patterns sind in F-6.3 vollständig spezifiziert mit List-View-State-Default-Wahl + `data-content`-Embed-Default + `format_card_datetime`-Helper-Vorschlag, Microcopy steht, Aufwandsschätzung ist da, Sub-Batch-Strategie ist im Sprint-Prompt verankert, Smoke-Pflicht-Kalibrierung ist über Drei-Pflicht-Live-Master-Patterns aufgelöst.)_
