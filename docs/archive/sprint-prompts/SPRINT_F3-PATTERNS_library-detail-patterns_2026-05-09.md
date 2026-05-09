# Sprint F3-PATTERNS — F-3.3 Patterns + Microcopy `library_detail`

**Datum**: 2026-05-09

**Ziel**: Stufe 3 (Patterns + Microcopy) der dreistufigen UX-Cascade-Methodik für `library_detail`. Aus den 17 Findings + 6 finding-linked Bug-Tickets aus F-3.2 konkrete UI-Pattern-Blöcke entwickeln, mit deutscher Microcopy, Aufwandsschätzung XS/S/M/L, Top-N-Quick-Wins per Impact-Score und Cluster-Vorschlag für F3-IMPL-*. **Kein Code-Touch** — das passiert in F3-IMPL-*.

**Vorbedingung**:
- Pytest 48/48 grün auf `main`. Letzter Code-Touch: F3-REVIEW (commits `82f6fd8` + `58759a7`, 2026-05-09).
- **Eingabe**: [docs/ui_findings_library_detail_2026-05.md](docs/ui_findings_library_detail_2026-05.md) (Sub-Thread liest komplett vor Phase 1).
  - 17 Findings: Sev 4: 0, Sev 3: 8, Sev 2: 4, Sev 1: 5.
  - 8 Bug-Tickets BT1–BT8: 6 finding-linked, 2 pure.
  - Cross-Feature-H4-Quote ~35% (6/17 Findings).
  - 9 Findings mit `⚠️ code-only`-Marker — Smoke-Pflicht-Vehikel für F3-IMPL-* (siehe unten).
  - 3 Schwerpunkt-Cluster:
    - **Cluster 1: Silent-Failure-Familie** (F1+F2+F4+F5 — Auto-Save-silent + Delete-silent, daily-usage-Schmerz hoch).
    - **Cluster 2: Notion-Side State-Wipe + UTC-Default** (F6+F11+F8 — User-Daten-Verlust unbemerkt).
    - **Cluster 3: Cross-Feature-Helper-Drift** (F7+F9+F12+F15 — F-1/F-2-Konventionen nicht eingehalten).
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - F-1.3: [docs/ui_patterns_document_converter_2026-05.md](docs/ui_patterns_document_converter_2026-05.md) — 14 Pattern-Blöcke (5 konsolidiert + 9 einzeln), höchster Aufwand M.
  - F-2.3: [docs/ui_patterns_audio_converter_2026-05.md](docs/ui_patterns_audio_converter_2026-05.md) — 21 Pattern-Blöcke (konsolidiert aus 32 Findings, 11 Konsolidierungen) plus Cluster-I/II-Vorbereitung.
- **Helper-Bestand in `_utils.js`** (Cross-Feature-H4-Lösungs-Vehikel): `showAlert(msg, type, options)`, `showToast(msg, type)`, `formatFileSize(bytes)`, `safeJSON(response)`. Patterns für Cross-Feature-H4-Findings (~6 Patterns aus Cluster 3) sollen primär Helper-Reuse vorschlagen, nicht neue Helpers anlegen.
- **Microcopy-Regeln** (für DE-Texte in Pattern-Vorschlägen):
  - Fehler max 2 Sätze.
  - Empty-State max 3 Sätze.
  - Buttons max 3 Wörter.
  - Keine Emojis bei Fehlern.
  - Deutsch durchgängig.
- **Impact-Score-Formel**: `Score = Sev × 5 / Aufwand-Gewicht`. Aufwand-Gewicht: XS=1, S=2, M=4, L=8. Höher = besser. Top-5 Quick-Wins per Impact-Score (analog F-1.3 / F-2.3).

**Out-of-scope**:
- Implementation — eigene Folge-Sprints `F3-IMPL-*`.
- Code-Änderungen jeglicher Art.
- Bug-Tickets **BT7** (textarea-escape) und **BT8** (window.open-noopener): pure Bug-Tickets ohne UX-H-Komponente, gehören **nicht** in den Pattern-Sprint. Werden in einem Bug-Sweep oder mit-genommen wenn nahegelegene Patterns angefasst werden.
- Bug-Tickets **BT1–BT6** (finding-linked): hier nicht eigenständig adressieren — sie werden via die Patterns ihrer verknüpften Findings mit-gelöst. Im Pattern-Block kurz vermerken welche BTx adressiert werden.
- Konstitutive Befunde aus F-3.1 (api_create_conversion-Strict-Validation, Notion-MCP-String-Doppelung): bleiben aus dem Scope, gehören zur `library`-Welle bzw. Notion-Konsolidierung.
- Andere Features.

---

## Master-Annotation (vorab eingebettet)

**Smoke-Pflicht-Marker für Cluster 1 + 2**: User hat entschieden, dass kein Master-Live-Walkthrough vor F-3.3 erfolgt. Stattdessen tragen Patterns für **Cluster 1 (Silent-Failure-Familie)** und **Cluster 2 (Notion-State-Wipe + UTC)** im Output-Doc ein zusätzliches Sub-Tag:

> **🔥 Smoke-Pflicht in F3-IMPL**: Befund ist `⚠️ code-only` aus F-3.1 — vor Pattern-Apply per Live-Smoke (DevTools-Network-Throttle für F1/F2/F4/F5; Browser-Reload-Sequenz für F6/F11; Datum-Inspektion für F8) verifizieren dass der Befund tatsächlich existiert. Wenn Smoke zeigt dass der Code-deduced-Befund nicht reproduzierbar ist: Pattern-Apply STOP, Master fragen.

Dieses Sub-Tag muss bei jedem Cluster-1-/Cluster-2-Pattern-Block sichtbar im Output-Doc stehen, **nicht** nur in einer Übersichts-Sektion — der F3-IMPL-Sub-Thread wird die einzelnen Pattern-Blöcke durchgehen, nicht die Übersicht.

**Cluster 3 (Cross-Feature-Helper-Drift)** braucht keine Smoke-Pflicht-Markierung — Helper-Reuse ist Code-Inspection-verifizierbar, kein Runtime-Befund.

---

## Phase 1 — Patterns + Microcopy

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Findings-Doc + Inventur-Doc komplett lesen**.
3. **Methodik-Vorlagen lesen**: F-1.3 + F-2.3 Pattern-Docs (Output-Format 1:1).
4. **Helper-Bestand in `_utils.js` auf aktuellen Stand verifizieren**: `grep -n "^function\|^export function\|window\\." static/js/_utils.js` zeigt verfügbare Helper. Patterns sollen vorhandene Helper nutzen, nur in begründeten Fällen neue Helper vorschlagen.

**Pattern-Aufgabe**:

Für jeden Finding (oder konsolidierte Finding-Gruppe) ein Pattern-Block mit:

- **Pattern-Nummer** (lauf, P1, P2, …).
- **Adressiert Findings**: Liste der Findings (mit H + Sev) und ggf. linked Bug-Tickets (BTx).
- **Smoke-Pflicht-Marker** falls Cluster 1 oder 2 (siehe Master-Annotation oben).
- **UI-Pattern-Beschreibung**: konkrete UI-Mechanik in 2–4 Sätzen. Welche Komponente, welcher State, welche Interaktion.
- **Visuelle Hinweise**: welche existierenden CSS-Klassen aus dem Neomorphism-System nutzen (`c-btn`, `c-alert`, `c-drop-zone`, etc.). Nur in begründeten Fällen neue CSS vorschlagen.
- **Microcopy** (deutsch): exakte Strings für Fehler-Banner, Empty-States, Button-Labels, Tooltips. Nach den Microcopy-Regeln oben.
- **Helper-Reuse**: welche `_utils.js`-Helper nutzt das Pattern. Wenn Inline-Code in `library_detail.js` vorhanden ist der durch Helper ersetzt wird: Code-Anker (`file:line`).
- **Aufwand**: XS / S / M / L. Begründung in einem Satz (Schema-Touch ja/nein, neue CSS, Test-Welle).
- **Impact-Score**: berechnet nach Formel.
- **Konsolidierung-Hinweis** falls mehrere Findings adressiert: warum die Findings ein gemeinsames Pattern teilen.

**Konsolidierungs-Logik** (analog F-1.3 / F-2.3):

- Findings die zur selben User-Action gehören (z.B. F1+F2 = Auto-Save-silent für `updateField` + `toggleFavorite`) → **ein** Pattern.
- Findings die zur selben State-Mechanik gehören (z.B. F6+F11 = State-Wipe-bei-Re-Toggle + State-Wipe-bei-Target-Switch) → **ein** Pattern.
- Findings die durch denselben Helper-Reuse gelöst werden (z.B. F7+F9 = beide via `showAlert`-statt-`alert()`) → **ein** Pattern.
- Findings mit unterschiedlichem User-Outcome (z.B. F8 UTC-Datum vs. F11 State-Wipe) → **separate** Patterns auch wenn sie im selben Cluster liegen.

**Erwartete Pattern-Anzahl**: 12-17 Patterns aus 17 Findings + 6 finding-linked BTs (~30% Konsolidierung wie F-1.3, weniger als F-2.3's ~50% wegen geringerer struktureller Überlappung der Findings).

**Output-Doc**: `docs/ui_patterns_library_detail_2026-05.md`. Struktur (1:1 wie F-1.3 / F-2.3):

1. Header mit Findings-Quelle, Sprint-Datum, Methodik-Hinweis (Microcopy-Regeln, Aufwand-Skala, Impact-Score-Formel).
2. **Pattern-Blöcke**: P1 bis Pn nach obigem Format.
3. **Cluster-Vorschlag für F3-IMPL-***: 1-3 Implementations-Cluster vorgeschlagen, jeweils mit Pattern-Nummern und Begründung. Cross-Feature-Konvergenz-Patterns (Cluster 3 aus Findings) bevorzugt in einem dedizierten Cluster bündeln (analog F-2.3 wo „Cluster Ia Foundation Sweep" alle Cross-Feature-Konvergenz-Patterns trug).
4. **Top-5 Quick-Wins**: Tabelle nach Impact-Score absteigend, mit Pattern-Nummer, adressiertem Finding, Sev, Aufwand, Score.
5. **Smoke-Pflicht-Übersicht**: Liste der Patterns mit `🔥 Smoke-Pflicht in F3-IMPL`-Sub-Tag — als Quick-Reference, aber das Sub-Tag bleibt **zusätzlich** in jedem betroffenen Pattern-Block stehen.

Nach Phase 1: STOP — Bericht. Statistik (Pattern-Anzahl, Konsolidierungs-Quote, Aufwand-Verteilung XS/S/M/L, Smoke-Pflicht-Anzahl, Cluster-Vorschlag mit Pattern-Nummern, Top-5-Quick-Wins).

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Pattern-Doc nochmal mit Distanz und prüft:

1. **Vollständigkeit**: jeder der 17 Findings ist in einem Pattern-Block adressiert (entweder einzeln oder im Konsolidat). BT7 + BT8 sind explizit als „pure Bug-Ticket, nicht in F-3.3" notiert. BT1–BT6 sind je in ihrem verknüpften Finding-Pattern erwähnt.
2. **Smoke-Pflicht-Konsistenz**: jeder Cluster-1-/Cluster-2-Pattern-Block trägt das `🔥 Smoke-Pflicht in F3-IMPL`-Sub-Tag. Smoke-Pflicht-Übersichts-Sektion und Pattern-Block-Sub-Tags stimmen überein.
3. **Helper-Reuse-Disziplin**: jedes als „Cross-Feature-H4" markierte Pattern (~6 erwartet aus Cluster 3) referenziert konkret einen `_utils.js`-Helper. Wo neue Helper vorgeschlagen werden: separate Begründung, nicht still mit-anlegen.
4. **Microcopy-Regeln**: Stichprobe — drei Fehler-Strings (≤2 Sätze), drei Button-Labels (≤3 Wörter), drei Empty-States (≤3 Sätze). Keine Emojis bei Fehlern.
5. **Impact-Score-Konsistenz**: Top-5-Quick-Wins-Tabelle stimmt mit den Pattern-Block-Score-Angaben überein. Aufwand-Gewichtung korrekt (XS=1, S=2, M=4, L=8).
6. **Disziplin**: keine konkreten Code-Diffs im Doc, kein Bug-Fix, keine Implementations-Schritte (das passiert in F3-IMPL-*).

Nach Phase 2: STOP — Bericht. „Pattern-Doc konsistent, alle Findings adressiert, Smoke-Pflicht-Marker durchgängig, Helper-Reuse-Disziplin gewahrt" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-3.3 / Stufe 3: patterns + microcopy of library_detail".
- Body: kurze Statistik (Pattern-Anzahl, Konsolidierungs-Quote, Aufwand-Verteilung, Smoke-Pflicht-Anzahl, Cluster-Vorschlag, Top-5-Quick-Wins).
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints. Wenn der Auto-Mode-Classifier blockt: im Phase-3-Bericht erwähnen, Master pusht von Hand.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — eine Output-Datei (`docs/ui_patterns_library_detail_2026-05.md`), Pattern-Blöcke + Microcopy + Aufwandsschätzung + Cluster-Vorschlag, kein Code-Touch, keine Tests, kein Smoke. Wenn die Konsolidierungs-Quote überraschend hoch wird (>50%) und weniger Patterns entstehen als 12: kein Re-Skopung-Trigger, normaler Sprint-Verlauf.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Pattern-Schreiben **fehlende Helper-Funktionen** in `_utils.js` auffallen, die für Cluster 3 sinnvoll wären (z.B. ein generischer `confirmAction(msg)` oder `formatDateBerlinTZ(iso)`-Helper): als „Helper-Vorschlag" am Doc-Ende sammeln, **nicht** im einzelnen Pattern still mit-anlegen — F3-IMPL-* entscheidet ob die Helper im Pattern-Cluster mit-implementiert oder als separater Helper-Cluster vorgezogen werden.
- Wenn beim Pattern-Schreiben Findings überlappen, die unterschiedlichen Schwerpunkt-Clustern zugeordnet sind: konsolidieren ist OK, Cluster-Zuordnung dann „Cluster X+Y" markieren.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F3-PATTERNS ☑ done 2026-05-XX → commit `<hash>`. Patterns-Doc unter `docs/ui_patterns_library_detail_2026-05.md`. <N> Patterns (XS: a, S: b, M: c, L: d), <K> Cluster vorgeschlagen, Top-5-Quick-Wins-Score-Range <X.0–Y.0>, <S> Smoke-Pflicht-Patterns. Verbleibende Sequenz: F3-IMPL-* → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F3-PATTERNS" raus → Erledigt-Liste; Sektion „2. F3-IMPL-*" rückt auf Position 1 mit Pattern-Anzahl als Hint für Cluster-Schnitt. Folge-Sprint-Nummern -1.
- **Memory**: nichts erwartet — Pattern-Methodik ist seit F-1.3/F-2.3 etabliert. Falls überraschend doch (z.B. „library_detail-Patterns brauchen ein neues Microcopy-Genre für Reader-Aktionen"): `feedback_*.md` schreiben.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Pattern-Methodik ist seit F-1.3 / F-2.3 klar etabliert, Vorlagen und Output-Format vorhanden. Master-Walkthrough wurde bewusst nicht durchgeführt — Smoke-Pflicht-Marker tragen die Code-only-Verifikation in F3-IMPL-* durch.)_
