# Sprint F2-CII — F-2 audio_converter Cluster II

**Datum**: 2026-05-09

**Ziel**: F-2 (UX-Welle für `audio_converter`) strukturell abschließen, indem Cluster II (9 Patterns, Sev 2+1) implementiert wird. Nach diesem Sprint ist `audio_converter` auf demselben UX-Niveau wie `document_converter` nach F-1.

**Vorbedingung**:
- Pytest 37/37 grün auf `main`. Letzter Code-Touch: F-2 Cluster I (commit `ef78508`, 2026-05-03).
- Cluster I (P1–P12, Sev 4+3) ist live-smoked und durch.
- Helper aus [static/js/_utils.js](static/js/_utils.js) (`showAlert`, `showToast`, `formatFileSize`, `safeJSON`) sind etabliert und werden in Cluster I durchgängig genutzt.
- Pattern-Vorlage für diesen Sprint: [docs/ui_patterns_audio_converter_2026-05.md](docs/ui_patterns_audio_converter_2026-05.md), Sektion „Cluster II (Sev 2 + Sev 1 — Polish + a11y)".

**Out-of-scope**:
- Cluster I (P1–P12) — bereits durch, nicht erneut anfassen.
- `getUserMedia`-in-`socket.onopen`-Bug (Permission-Prompt erst nach WS-Handshake) — als Out-of-Scope respektieren, kommt in einem späteren Sprint.
- Andere Features (`document_converter`, `markdown_converter`, `library`, …) — nicht angefasst.
- F-006-Backend-Whitelist für andere Endpoints (markdown, audio-für-andere-Tabs) — die kommen im nächsten Sprint `SEC`. **Aber:** P13 in diesem Sprint *ist* die Backend-Whitelist für `audio_converter`'s File-Upload-Pfad (Sev 2 in Cluster II), die ziehen wir hier mit.

---

## Phase 1 — Implementation

Pre-Flight:

1. `pytest tests/` — muss 37/37 grün sein. Wenn nicht: stoppen, an Master melden.
2. Patterns durchlesen: [docs/ui_patterns_audio_converter_2026-05.md](docs/ui_patterns_audio_converter_2026-05.md), die 9 Pattern-Blöcke P13 bis P21. Jeder Block hat Microcopy, technischen Plan und Aufwandsschätzung.
3. Cluster I als Mechanik-Referenz: commit `ef78508` (Foundation-Sweep + Critical-UX + State-Lifecycle holistic). Insbesondere wie dort `showAlert`/`showToast`/`formatFileSize` gerufen werden.
4. F-1 Cluster D als Vorlage für P13 (Backend-Whitelist-Pattern): commit `e68b6dd` und [app_pkg/documents.py:30](app_pkg/documents.py#L30) (precomputed `accept`-String, fließt nach Template + `window.PageData`, Backend liefert 400+DE-JSON für unsupported extensions).
5. F-1 Polish-1 als Vorlage für P19 (`formatFileSize`-Reuse): commit `ea9db78`.

**Patterns dieses Sprints** (Pflicht alle 9, sofern nicht in Phase-Ende-Bericht begründet warum einer rausfällt):

| Pattern | Was | Adressiert |
|---------|-----|------------|
| P13 | Backend-Whitelist + DE-Fehler-JSON für unsupported audio file | F19 H9 Sev 2 |
| P14 | Mic-Button `:focus-visible` + a11y-Annotations | F20 + F25 Sev 2 |
| P15 | Mode-Radios keyboard-zugänglich + sichtbarere Selection | F21 Sev 2 + F29 Sev 1 |
| P16 | Custom-Prompt-Toggle als echter Button | F22 Sev 2 |
| P17 | Clear-Aktionen mit Confirmation (Live-Tab + Reset-Prompt) | F23 Sev 2 |
| P18 | Podcast-Polling Cancel-Button | F24 Sev 2 |
| P19 | Filename-Anzeige mit `formatFileSize` (Cross-Feature-Reuse) | F26 Sev 2 |
| P20 | Live-Transcript `aria-live`-Region + `aria-label` | F28 Sev 1 |
| P21 | Download Success-Toast + Audio-Player Error-Fallback | F32 + F31 Sev 1 |

**Mechanik-Leitplanken**:

- **Helper-Reuse zwingend**: keine eigenen Toast/Alert/Filesize-Implementierungen. Wenn ein Pattern einen Helper braucht, der noch nicht existiert (z.B. „showConfirm" für P17), in `_utils.js` ergänzen — nicht inline.
- **Microcopy deutsch**: Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern.
- **Keine `alert()`-Calls** im Frontend-JS. `showAlert` oder `showToast` benutzen. Cluster I hat alle 11+ alert-Sites entfernt — nicht reintroducen.
- **a11y-Patterns wieder reuse aus F-1 Cluster E**: `:focus-visible`-Vokabular von `c-btn`, fokussierbare `region` mit aria-label.
- **Holistic-Rewrite vs. sequentiell**: Cluster I-Erfahrung — bei stark verkoppelten Patterns ist Holistic-Rewrite effizienter. Cluster II ist eher additiv (a11y-Polish + einzelne Reibungspunkte). Sequenziell pro Pattern ist OK; Sub-Batches optional. Bei Unklarheit: pragmatisch entscheiden, kurz im Bericht erwähnen.
- **P17 Clear-Confirmation**: ein neuer interaktiver Pattern (kein Helper bisher). Zwei Optionen: (a) `window.confirm()` als pragmatischer Default, (b) eigener Confirm-Modal-Helper. Empfehlung: **(a) `window.confirm()`** in diesem Sprint, weil ein Modal-System ein eigener Scope wäre. Wenn der Sub-Thread (b) für saubere fühlt, vorher rückfragen.

**Erwartete Files (grobe Erwartung, kann sich leicht verschieben)**:

```
static/js/audio_converter.js              # EDIT — größter Touch
static/js/_utils.js                       # EDIT — ggf. Helper-Ergänzung (showConfirm? siehe oben)
templates/audio_converter.html            # EDIT — accept-Whitelist, aria-Attribute, Mountpoints
app_pkg/audio.py                          # EDIT — Backend-Whitelist (P13), 400+DE-JSON-Pfad
static/css/style.css                      # EDIT — :focus-visible-Regeln, evtl. neue States
tests/test_audio.py (oder test_audio_routes.py)  # EDIT — 1 neuer Test analog F-1 Cluster D (test_transcribe_audio_unsupported_extension_returns_400)
```

**Code-Quality-Gates**:

- `pytest tests/` grün. Erwartung: vorher 37, nachher 38 (neuer Test für P13 analog F-1 Cluster D).
- `grep -c "alert(" static/js/audio_converter.js` → 0.
- Keine englischen UI-Strings in `static/js/audio_converter.js` oder `templates/audio_converter.html`.
- Helper aus `_utils.js` wiederverwendet (`showAlert`, `showToast`, `formatFileSize`, ggf. neuer `showConfirm`).

Nach Phase 1: STOP — Bericht an Master. Kurzer Status: welche der 9 Patterns durch, welche offen/begründet ausgelassen, ob Sub-Batches gebildet wurden, ob `_utils.js` erweitert wurde.

---

## Phase 2 — Verify

1. `pytest tests/` grün (38 erwartet).
2. **Live-Smoke** auf `localhost:5656` (Docker-Stack hochfahren falls nicht läuft: `docker compose up --build`):
   - File-Tab: unsupported Extension hochladen → 400+DE-JSON-Banner (P13).
   - File-Tab: Filename-Display zeigt Größe via `formatFileSize` (P19).
   - Live-Tab: Mic-Button mit Tab navigierbar, `:focus-visible`-Ring sichtbar (P14).
   - Mode-Radios: Tab-Navigation funktioniert, Auswahl visuell deutlich (P15).
   - Custom-Prompt-Toggle: ist `<button>`, nicht `<a href="#">` (P16).
   - Clear-Aktionen (Live-Tab + Reset-Prompt): Confirmation-Dialog erscheint (P17).
   - Podcast-Generation: Cancel-Button während Polling sichtbar und funktionsfähig (P18).
   - Live-Transcript: Screen-Reader-Tools melden `aria-live`-Region (P20) — alternativ DevTools-Inspect auf `aria-live="polite"` und `aria-label` verifizieren.
   - Download-Pfad: Success-Toast nach Download (P21). Fehlerhafter Audio-Player-Source: Error-Fallback (P21).
3. Kein DevTools-Console-Fehler auf irgendeiner der drei Tabs (File, Live, Podcast).
4. Cluster-I-Pfade unverändert: `showAlert`, `showToast`, Save-Btn-Lifecycle, Drag-Drop, Mic-Permission-Denied — alle weiter funktional (Regression-Smoke).

Nach Phase 2: STOP — Bericht. Liste der gesmokten Pfade, etwaige Auffälligkeiten.

---

## Phase 3 — Commit

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Default: ein Commit für Cluster II. Wenn Sub-Batches gebildet wurden, separate Commits in derselben Branch.
- Branch: direkt auf `main` ist OK (Single-User-Repo, keine PR-Pflicht). Wenn der Sub-Thread lieber Branch + Merge will: kurzer Hinweis im Bericht, Master entscheidet.
- Push erst nach Master-Sign-off.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**M** — UI-Touch in einem Feature, ein Backend-Endpoint angefasst (P13), ein neuer Test, mehrere a11y-Patterns. Cluster I (12 Patterns Sev 4+3) wurde als „gerade noch handhabbar" beschrieben; Cluster II ist 9 Patterns Sev 2+1, weniger verkoppelt — sollte komfortabler laufen.

---

## Konstitutiv mit-genommen, falls berührt

- Falls der Sub-Thread während der a11y-Patterns weitere DevTools-Console-Warnings sieht, die mit den Cluster-II-Patterns thematisch zusammenhängen (z.B. fehlende `aria-label`-Stellen in benachbarten Komponenten): kurz dokumentieren, im Sprint-Bericht erwähnen, **nicht** still mitfixen (siehe Memory `feedback_no_silent_fixes.md`).
- Falls beim Lesen von `audio_converter.js` ein offensichtlicher Bug auffällt der nicht in der Pattern-Liste steht: Issue-Format („gefunden, beschrieben, **nicht** gefixt") in den Bericht.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F-2 Cluster II ☑ done 2026-05-XX → commit `<hash>`. F-2 strukturell abgeschlossen. Nächster Sprint: SEC."
- **BACKLOG.md**: Sektion „1. F2-CII" raus → Erledigt-Liste; alle Sprint-Codes nachrücken (Sequenz bleibt, nur Nummer schiebt sich um eins).
- **Memory**: nur wenn eine *übertragbare* Lehre auftaucht (z.B. „a11y-Polish-Sweeps brauchen X" als `feedback_*.md`). Defensiv: lieber nichts schreiben als Trivialitäten persistieren.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Cluster II ist klar gescoped via Pattern-Doc.)_
