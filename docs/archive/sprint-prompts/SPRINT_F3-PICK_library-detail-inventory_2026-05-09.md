# Sprint F3-PICK — F-3.1 UX-Inventur `library_detail`

**Datum**: 2026-05-09

**Ziel**: Stufe 1 (Inventur) der dreistufigen UX-Cascade-Methodik für das Feature `library_detail`. Alle interaktiven Elemente kartieren, vorhandene + fehlende States dokumentieren, Code↔live-Divergenzen flaggen. **Kein Bewerten, kein Bug-Fix, kein Pattern-Vorschlag** — das kommt in F-3.2 (Heuristik-Review) bzw. F-3.3 (Patterns + Microcopy).

**Vorbedingung**:
- Pytest 48/48 grün auf `main`. Letzter Code-Touch: CVE-DG-Sprint (commit `74589e3`, 2026-05-09). Stage-7-CVE-Block durch (5/5).
- Touch-Pfade des Features:
  - **Route-Modul**: [app_pkg/library.py](app_pkg/library.py) — enthält `library_detail`-Route plus API-Endpoints (`api_update_conversion`, `api_delete_conversion`, `api_send_to_notion`).
  - **Template**: [templates/library_detail.html](templates/library_detail.html).
  - **JS**: [static/js/library_detail.js](static/js/library_detail.js).
  - **Shared**: [static/css/style.css](static/css/style.css) für Neomorphism-States, [static/js/_utils.js](static/js/_utils.js) für die Helper (`showAlert`, `showToast`, `formatFileSize`, `safeJSON`).
- Bekannte Pre-Existing-Items aus BACKLOG-P3:
  - 6 englische UI-Strings in `static/js/library_detail.js` (namentlich aufgelistet beim F-1 Polish-1-Sub-Thread). Die werden **in der Inventur nur erwähnt**, nicht gefixt — das passiert in F-3-IMPL-* nach Patterns-Sprint.
- Methodik-Vorlagen (sehr wichtig — gleiche Struktur wie F-1.1 und F-2.1):
  - F-1.1 Inventur: [docs/ui_inventory_document_converter_2026-05.md](docs/ui_inventory_document_converter_2026-05.md) — 24 Elemente, 6 fehlende States im Code, 5 Code↔live-Divergenzen, 9 separate Befunde (3 als Bug-Tickets geflaggt).
  - F-2.1 Inventur: [docs/ui_inventory_audio_converter_2026-05.md](docs/ui_inventory_audio_converter_2026-05.md) — 47 Elemente, 13 fehlende States, 7 Divergenzen, 6 unverifizierbare States, 6 audio-spezifische States.
- Kontext für die Methodik (aus CLAUDE.md): Single-User-App, LAN-only, login-protected. Primäre `library_detail`-Aufgabe: Conversion-Eintrag lesen, Highlights/Notizen verwalten, Notion-Export. Readwise-Ersatz-Kontext (siehe Memory `project_readwise_replacement.md`).

**Out-of-scope**:
- Heuristik-Review (Stufe 2) — eigener Folge-Sprint `F3-REVIEW`.
- Patterns + Microcopy (Stufe 3) — eigener Folge-Sprint `F3-PATTERNS`.
- Implementation — eigene Folge-Sprints `F3-IMPL-*`.
- Code-Änderungen jeglicher Art. Selbst wenn ein offensichtlicher Bug auffällt: dokumentieren als „separater Befund" (Bug-Ticket-Kandidat), **nicht** fixen — siehe Memory `feedback_no_silent_fixes.md`.
- Andere Features (`library`, `markdown_converter`, `mermaid_converter`, `login`, podcast-flow) — eigene Folge-Wellen.

---

## Phase 1 — Inventur (read-only)

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Methodik-Vorlagen lesen**: F-1.1 + F-2.1 Inventur-Docs (Pfade in Vorbedingung). Die Output-Struktur dort ist die zu reproduzierende Form.
3. **Touch-Pfade lesen** (Pflicht in genau dieser Reihenfolge):
   - `templates/library_detail.html` zuerst — DOM-Struktur, Form-Felder, Button-Labels, datenbindende Stellen (`{{ … }}`-Variablen).
   - `static/js/library_detail.js` zweitens — Event-Handler, AJAX-Calls, State-Mutations, Helper-Reuse vs. Inline-Code.
   - `app_pkg/library.py` drittens — Route-Handler-Signature, Daten-Lade-Pfad, API-Endpoints und ihre Response-Shapes.
   - `static/css/style.css` letztens — relevante Selektoren für `library_detail`-spezifische Komponenten (Suche nach `c-`-Prefix-Klassen aus dem Template).

**Inventur-Aufgabe** (kein Bewerten, nur Mapping):

Für jedes interaktive Element auf der Seite kartieren:

| Spalte | Was rein |
|--------|----------|
| `#` | Lauf-Nummer |
| `Element-Typ` | Button / Input / Textarea / Drop-Zone / Tab / Dropdown / Modal / Link / etc. |
| `Label` (im Template) | Text wie er im DOM steht (deutsch oder englisch markieren) |
| `Aktion` | Was passiert beim Klick/Submit/Change (Endpoint? State-Change? Navigation?) |
| `Vorhandene States` | default, hover, focus, disabled, loading, error, success, empty — die im Code/CSS belegt sind |
| `Fehlende States` | dieselbe Liste — die im Code **nicht** belegt sind |
| `Notizen` | Auffälligkeiten, Code↔live-Divergenzen, mögliche Bugs (als „Befund Nr. X" markiert) |

**Ergänzungs-Sektionen** (analog F-1.1 / F-2.1):

- **Code↔live-Divergenzen**: Stellen wo Template, JS und Route-Handler nicht zusammenpassen (z.B. JS spricht einen Endpoint mit POST an, Route-Handler erwartet PUT; Template-Label sagt eine Sache, JS überschreibt vor Anzeige; CSS-State-Klasse wird gesetzt aber CSS-Regel existiert nicht).
- **Separate Befunde** (nummeriert): mögliche Bugs, Inkonsistenzen, Out-of-Scope-Beobachtungen. Jeder Befund mit:
  - Code-Anker (`file:line` oder Selektor)
  - Beschreibung in 1-2 Sätzen
  - Vermuteter Schweregrad-Hinweis (rein subjektiv — wird in F-3.2 final eingeordnet)
  - Disposition-Vorschlag: „nur Finding" / „Finding + Bug-Ticket" / „nur Bug-Ticket"
- **Live-Walkthrough-Lücken**: wenn der Sub-Thread keinen Browser-Access hat (analog CVE-DG-Lehre), Code-only-Inventur ist 80–90% des Werts. States die nur runtime sichtbar sind (CSS `:hover`, JS-injected `loading`-Klassen, transient Error-Banner) als „Code-deduced, nicht live verifiziert" markieren. Master macht ggf. Live-Walkthrough-Nachreichung in F-3.2 oder zwischen den Sprints.

**Output-Doc**: `docs/ui_inventory_library_detail_2026-05.md`. Struktur 1:1 wie F-1.1 / F-2.1 Inventur-Docs — Tabelle als Hauptteil, dann Divergenzen, dann separate Befunde, dann Live-Walkthrough-Lücken-Hinweis.

Nach Phase 1: STOP — Bericht. Kurz: Element-Anzahl, fehlende-States-Anzahl, Divergenzen-Anzahl, Befunde-Anzahl mit Disposition-Verteilung, ob Live-Walkthrough-Lücken bestehen.

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Inventur-Doc nochmal mit Distanz und prüft:

1. **Vollständigkeit**: jedes interaktive Element aus dem Template ist in der Tabelle (Stichprobe: alle `<button>`, `<input>`, `<textarea>`, `<a class="...">`-Action-Links).
2. **Konsistenz**: jeder Befund hat Code-Anker. Keine Befunde ohne `file:line`.
3. **Disziplin**: kein Pattern-Vorschlag im Doc (passiert nur in F-3.3). Keine Severity-Bewertung (passiert in F-3.2). Kein Bug-Fix.
4. **Helper-Reuse-Spuren**: in den Notizen markieren wo `library_detail.js` schon `_utils.js`-Helper nutzt vs. wo es Inline-Code dupliziert (relevant für F-3.3 Cross-Feature-H4-Konvergenz). Aber keine Empfehlung formulieren — nur Beobachtung.

Nach Phase 2: STOP — Bericht. „Inventur-Doc konsistent, alle Befunde mit Code-Ankern, kein Pattern-/Severity-Drift" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit für den Inventur-Output. Subject z.B. „F-3.1 / Stufe 1: UI inventory of library_detail".
- Body: kurze Statistik (Element-Anzahl, fehlende States, Divergenzen, Befunde) plus Notiz ob Live-Walkthrough-Lücken bestehen.
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints (Single-User-Single-Instance-Repo).

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für Inventur-Sprints**: wenn beim Code-Reading ein Bug auffällt, der akut wirkt (z.B. Crash-Pfad, Datenverlust-Risiko): im Bericht **flaggen** mit „akut" — Master entscheidet ob Hot-Fix-Sprint vorgezogen wird oder als Befund mit-läuft. Der Sub-Thread fixt **nicht selber**.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — eine Output-Datei (`docs/ui_inventory_library_detail_2026-05.md`), Code-Reading + Mapping, kein Code-Touch, keine Tests, kein Smoke. Wenn das Feature überraschend groß ist (z.B. Element-Anzahl >50, viele Sub-Tabs): Bericht-Eintrag, Master sieht ggf. Sprint-Re-Skopung — aber default ist S.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Code-Reading von `app_pkg/library.py` Auffälligkeiten in den **anderen** Routen auffallen (`library`, `api_*`-Endpoints), die **nicht** zu `library_detail` gehören: kurz im Bericht aufzählen, **nicht** in die Inventur-Doc — die ist strikt `library_detail`. Master fold-et bei der `library`-Welle.
- Englische UI-Strings (die 6 bekannten + neue Befunde): erwähnen in den Notizen, **nicht** als separater Befund — das ist ein erwartetes Pre-Existing-Item.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F3-PICK ☑ done 2026-05-XX → commit `<hash>`. Inventur-Doc unter `docs/ui_inventory_library_detail_2026-05.md`. <Element-Anzahl> Elemente, <fehlende> States, <Divergenzen> Divergenzen, <Befunde> Befunde. Verbleibende Sequenz: F3-REVIEW → F3-PATTERNS → F3-IMPL-* → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F3-PICK" raus → Erledigt-Liste; Sektion „2. F3-REVIEW" rückt auf Position 1, alle Folge-Sprint-Nummern -1.
- **Memory**: nichts erwartet — Inventur-Methodik ist seit F-1/F-2 etabliert, keine neue Lehre wahrscheinlich. Falls überraschend doch (z.B. „library_detail hat ein UX-Pattern X, das in F-1/F-2 noch nicht gesehen wurde, könnte für `library`-Welle relevant werden"): `feedback_*.md` schreiben.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Inventur-Methodik ist seit F-1.1 / F-2.1 klar etabliert, Vorlagen und Output-Format vorhanden.)_
