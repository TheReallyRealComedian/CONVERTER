# Sprint F4-PICK — F-4.1 UX-Inventur `podcast-flow`

**Datum**: 2026-05-09

**Ziel**: Stufe 1 (Inventur) der dreistufigen UX-Cascade-Methodik für das Feature `podcast-flow`. Cross-cutting Async-Pipeline: Frontend-Tab innerhalb `audio_converter.html`, Backend-Routes in `app_pkg/podcasts.py`, Worker-Pfad in `tasks.py`. Alle interaktiven Elemente kartieren, vorhandene + fehlende States dokumentieren, Code↔live-Divergenzen flaggen, Async-spezifische State-Übergänge (queued → started → finished/failed) explizit als eigene State-Klasse. **Kein Bewerten, kein Bug-Fix, kein Pattern-Vorschlag** — das kommt in F-4.2 / F-4.3.

**Vorbedingung**:
- Pytest 51/51 grün auf `main`. Letzter Code-Touch: F3-IMPL (commits `843574b` / `40dd02e` / `b3e666a` / `5ba29c1`, 2026-05-09). F-3 strukturell abgeschlossen für library_detail.
- Touch-Pfade des Features (cross-cutting):
  - **Frontend Template**: [templates/audio_converter.html](templates/audio_converter.html) — Tab-Pane `#podcast-pane`, plus Tab-Button `data-tab="podcast"`. **Nur podcast-spezifische Sub-Bereiche kartieren** — der Rest des audio_converter wurde in F-2 schon abgehandelt.
  - **Frontend JS**: [static/js/audio_converter.js](static/js/audio_converter.js) — podcast-relevante Funktionen (Generate-Klick-Handler, Status-Polling-Loop, Cancel-Button, Download-Trigger, Mode-Radio-Handler, Skript-Edit-Handler). Der audio-File-Tab und live-Tab sind bereits durch F-2 gemappt.
  - **Backend Routes**: [app_pkg/podcasts.py](app_pkg/podcasts.py) — 4 Routes:
    - `POST /generate-podcast` (legacy Google-TTS-Pfad — verifizieren ob noch genutzt oder dead).
    - `POST /generate-gemini-podcast` (RQ-Queue-Pfad, primärer Workflow).
    - `GET /podcast-status/<job_id>` (Polling, F-001-narrow-except-Pfad).
    - `GET /podcast-download/<job_id>` (Download mit `Path.is_relative_to`-Schutz aus SEC).
  - **Worker**: [tasks.py](tasks.py) — `generate_podcast_task` (RQ-Background-Job).
  - **Shared**: [static/css/style.css](static/css/style.css), [static/js/_utils.js](static/js/_utils.js) (Helper inkl. neue `formatDatetimeLocalNow` + `.sr-only`-Utility aus F-3-IMPL).
- **Was F-2 schon erledigt hat (NICHT erneut behandeln)**:
  - P15 (audio_converter.html `#podcast-alert-container` Banner-Mountpoint) — bereits vorhanden.
  - P18 (Podcast-Polling Cancel-Button) — bereits implementiert in F-2 Cluster II.
  - DE-Microcopy für podcast-Tab Strings (Mode-Radio, Language-Select, Quelltext-Placeholder, Skript-Placeholder) — bereits aus F-2 DE-Pass.
  - showAlert/showToast/formatFileSize-Reuse für audio_converter.js — bereits aus F-2 Cluster I.
  - F-001-Critical-Fix (NoSuchJobError vs. transport-error distinguish) — schon im Code, F-3.1-/F-3.2-Findings haben das bestätigt.
  - F-005 Path-Traversal in `podcast_download` — durch SEC-Sprint mit `Path.is_relative_to` ersetzt.
  - F-013 Input-Allowlist für Gemini-Podcast-Parameter (`narration_style`, `script_length`, `language`, `num_speakers`) — durch SEC-Sprint.
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - F-1.1 Inventur: [docs/ui_inventory_document_converter_2026-05.md](docs/ui_inventory_document_converter_2026-05.md) — 24 Elemente.
  - F-2.1 Inventur: [docs/ui_inventory_audio_converter_2026-05.md](docs/ui_inventory_audio_converter_2026-05.md) — 47 Elemente, inkl. 6 audio-spezifische States (Mic-Loading, WS-Connection, Permission-Denied, Recording-Active, etc.). **Wichtig**: dort ist der Podcast-Tab aus Cross-Tab-Perspektive berührt — Sub-Thread liest nochmal um zu wissen was bereits-in-F-2 dokumentiert ist.
  - F-3.1 Inventur: [docs/ui_inventory_library_detail_2026-05.md](docs/ui_inventory_library_detail_2026-05.md) — 21 Elemente, Live-Walkthrough-Lücken-Sektion am Doc-Ende.
- **Async-spezifische State-Klasse** (NEU für podcast-flow, neue Erwartung gegenüber F-1/F-2/F-3):
  - **Queued**: Job in Redis enqueued, Worker hat ihn noch nicht aufgegriffen.
  - **Started**: Worker hat den Job gepickt, ist mitten in Generierung.
  - **Stage-Progress**: optional — gibt es in der Pipeline mehrere Stages (Skript-Generierung → TTS → Chunk-Konkatenation), und sind die im UI sichtbar?
  - **Finished**: Job fertig, Result verfügbar.
  - **Failed**: Worker-Exception, Job in failed-Queue.
  - **Cancelled**: vom User per Cancel-Button abgebrochen.
  - Plus: Polling-Frequenz, Polling-Timeout (Stale-Job-Erkennung), Network-Drop während Polling.
- **Kontext für die Methodik**: Single-User-App, LAN-only, login-protected. Primäre `podcast-flow`-Aufgabe: Quelltext eingeben (oder Skript direkt schreiben) → Mode wählen (monolog/dialogue) → Generieren → warten (Long-Running, mehrere Minuten möglich) → Download. **Long-running async** ist der zentrale Reibungspunkt — da konzentriert sich vermutlich der meiste UX-Schmerz.

**Out-of-scope**:
- Heuristik-Review (Stufe 2) — eigener Folge-Sprint `F4-REVIEW`.
- Patterns + Microcopy (Stufe 3) — eigener Folge-Sprint `F4-PATTERNS`.
- Implementation — eigene Folge-Sprints `F4-IMPL-*`.
- Code-Änderungen jeglicher Art. Bugs als „separater Befund" dokumentieren, nicht fixen — siehe Memory `feedback_no_silent_fixes.md`.
- audio-File-Tab und live-Tab innerhalb `audio_converter.html` — bereits durch F-2 gemappt.
- Andere Features (`library`, `markdown_converter`, `mermaid_converter`, `login`) — eigene Folge-Wellen.

---

## Phase 1 — Inventur (read-only)

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Methodik-Vorlagen lesen**: F-1.1 + F-2.1 + F-3.1 Inventur-Docs (Pfade in Vorbedingung). Plus: F-2.1 `audio_converter`-Inventur **gezielt** auf den Podcast-Tab-Bereich durchgehen, um zu sehen welche Elemente dort bereits dokumentiert sind und nicht erneut auftauchen sollen.
3. **Touch-Pfade lesen** (Pflicht in genau dieser Reihenfolge):
   - `templates/audio_converter.html` — nur `#podcast-pane` und Tab-Button. Form-Felder, Buttons, Mode-Radios, Status-Display-Bereich.
   - `static/js/audio_converter.js` — podcast-relevante Funktionen. `grep -n "podcast" static/js/audio_converter.js` zeigt die Aufschlagspunkte.
   - `app_pkg/podcasts.py` — 4 Routes mit ihren Response-Shapes (besonders `/podcast-status/<job_id>` JSON-Schema).
   - `tasks.py` — `generate_podcast_task`-Signature und Job-Meta-Updates (falls vorhanden — z.B. `job.meta['stage'] = ...`).

**Inventur-Aufgabe** (kein Bewerten, nur Mapping):

Für jedes interaktive Element + jeden async State kartieren:

| Spalte | Was rein |
|--------|----------|
| `#` | Lauf-Nummer |
| `Element-Typ` | Button / Input / Textarea / Mode-Radio / Tab-Pane / Status-Display / Polling-Loop / Async-Trigger / etc. |
| `Label` (im Template) | Text wie er im DOM steht (deutsch oder englisch markieren) |
| `Aktion` | Was passiert (Endpoint? RQ-Enqueue? Polling-Start? Status-Render? Cancel?) |
| `Vorhandene States` | default, hover, focus, disabled, **queued, started, stage-progress, finished, failed, cancelled** (async-Klasse), error, success, empty — die im Code/CSS belegt sind |
| `Fehlende States` | dieselbe Liste — die nicht belegt sind |
| `Notizen` | Auffälligkeiten, Code↔live-Divergenzen, Async-Race-Conditions, mögliche Bugs (als „Befund Nr. X" markiert) |

**Ergänzungs-Sektionen** (analog F-1.1 / F-2.1 / F-3.1):

- **Code↔live-Divergenzen**: Stellen wo Template, JS, Route-Handler und Worker nicht zusammenpassen.
- **Async-Pipeline-Mapping**: separate Sektion am Doc-Anfang oder -Ende mit Sequenz-Diagramm-Style:
  - Frontend: User-Click → JS-Action → Backend-Endpoint → Response.
  - Polling: Frontend-Loop → `/podcast-status/<job_id>` → JSON-Response → State-Update.
  - Worker: RQ-Pick → tasks.generate_podcast_task → Stages → Result-Write.
  - Download: User-Click → `/podcast-download/<job_id>` → File-Stream → Cleanup.
  - **Wichtig**: was passiert beim Browser-Reload während laufendem Job? Was bei Network-Drop während Polling? Was wenn `job.id` im LocalStorage stale ist?
- **Separate Befunde** (nummeriert): Bugs, Inkonsistenzen, Race-Conditions. Jeder Befund mit Code-Anker, Beschreibung, Disposition-Vorschlag (`nur Finding` / `Finding + Bug-Ticket` / `nur Bug-Ticket`).
- **Live-Walkthrough-Lücken**: Async-States sind besonders schwer aus Code abzuleiten (z.B. „Wie sieht das UI aus wenn ein Job 8 Min dauert? Gibt es einen Stale-Indicator nach 5 Min ohne Fortschritt?"). **Mit einer Live-Job-Generierung** (15-30 Min Browser-Walkthrough) könnte Master diese Lücken besser füllen — Sub-Thread markiert als „Live-Walkthrough wäre besonders wertvoll" mit konkreter Test-Anleitung. Code-only-Inventur deckt 60-70% (niedriger als F-1/F-2/F-3 wegen Async-Komplexität).

**Output-Doc**: `docs/ui_inventory_podcast_flow_2026-05.md`. Struktur 1:1 wie F-1.1 / F-2.1 / F-3.1 Inventur-Docs — Async-Pipeline-Mapping ist neue Zusatz-Sektion.

Nach Phase 1: STOP — Bericht. Element-Anzahl, fehlende-States-Anzahl (besonders async-State-Lücken), Divergenzen-Anzahl, Befunde-Anzahl mit Disposition-Verteilung, Async-Pipeline-Mapping-Status, Live-Walkthrough-Lücken-Empfehlung.

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Inventur-Doc nochmal mit Distanz und prüft:

1. **Vollständigkeit**: jedes podcast-spezifische interaktive Element ist in der Tabelle. Plus: jeder async State ist als eigene Spalte/Zeile erfasst (queued/started/stage/finished/failed/cancelled).
2. **Cross-Reference mit F-2.1**: Elemente die in F-2.1 schon dokumentiert sind, sind nicht doppelt aufgenommen — entweder im Header als „bereits durch F-2 gemappt, nicht erneut" verwiesen, oder als „durch F-2-Pattern bereits erfüllt" markiert.
3. **Konsistenz**: jeder Befund hat Code-Anker. Async-Race-Conditions haben sequenz-spezifische Anker (z.B. „polling-Tick T+5s während User Cancel klickt").
4. **Disziplin**: kein Pattern-Vorschlag, keine Severity-Bewertung, kein Bug-Fix.
5. **Helper-Reuse-Spuren**: in den Notizen markieren wo `audio_converter.js`-podcast-Funktionen schon `_utils.js`-Helper nutzen vs. wo Inline-Code dupliziert (analog F-3.1).
6. **Async-Pipeline-Mapping** ist vollständig (Frontend → Backend → Worker → Polling → Download), nicht nur Frontend-zentriert.

Nach Phase 2: STOP — Bericht. „Inventur-Doc konsistent" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit für den Inventur-Output. Subject z.B. „F-4.1 / Stufe 1: UI inventory of podcast-flow".
- Body: Statistik (Element-Anzahl, async-State-Coverage, Divergenzen, Befunde, Async-Pipeline-Mapping-Vollständigkeit, Live-Walkthrough-Empfehlung).
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints. Wenn Auto-Mode-Classifier blockt: im Phase-3-Bericht erwähnen.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute.

**Zusätzlich für Inventur-Sprints**: bei akut wirkendem Bug (Crash-Pfad, Datenverlust-Risiko, Worker-Boot-Loop) im Bericht **flaggen** mit „akut" — Master entscheidet ob Hot-Fix-Sprint vorgezogen wird oder als Befund mit-läuft. Sub-Thread fixt **nicht selber**.

**Async-spezifische STOP-Trigger**: wenn beim Code-Reading von `tasks.generate_podcast_task` oder Worker-Pfaden eine Race-Condition auffällt die zu Datenverlust führen könnte (z.B. zwei Worker pickup desselben Jobs, oder Cleanup-Pfad löscht laufenden Job): **akut**-Flag pflicht.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — eine Output-Datei (`docs/ui_inventory_podcast_flow_2026-05.md`), Code-Reading + Mapping cross-cutting (Frontend + Backend + Worker), kein Code-Touch, keine Tests, kein Smoke. Wenn das Async-Pipeline-Mapping überraschend groß wird (z.B. mehrere Stage-Übergänge die nicht im UI sichtbar sind aber im Worker drin): Bericht-Eintrag, Master sieht ggf. Sprint-Re-Skopung — aber default ist S.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim Code-Reading von `tasks.py` oder `services/gemini/`-Submodulen Auffälligkeiten auffallen die **nicht** zu podcast-flow-UX gehören (z.B. Gemini-Service-interne Hygiene-Items): kurz im Bericht aufzählen, **nicht** in die Inventur-Doc — die ist strikt podcast-flow-Frontend-+-Backend-Pipeline.
- Wenn der legacy `/generate-podcast`-Pfad (Google-TTS) als nicht mehr genutzt erkannt wird (z.B. kein Frontend-Trigger): in den Notizen notieren als „dead code candidate" — Master entscheidet ob Removal in einer Hygiene-Welle nachgereicht wird.
- Englische UI-Strings: erwähnen wenn welche durchrutschen, **nicht** als separater Befund — F-2 DE-Pass sollte das eigentlich abgedeckt haben, aber wenn doch welche im Podcast-Bereich übrig sind, ist das ein erwartetes Pre-Existing-Item.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F4-PICK ☑ done 2026-05-XX → commit `<hash>`. Inventur-Doc unter `docs/ui_inventory_podcast_flow_2026-05.md`. <Element-Anzahl> Elemente, <fehlende> States (davon <X> async-spezifisch), <Divergenzen> Divergenzen, <Befunde> Befunde, Async-Pipeline-Mapping <vollständig/teilweise>. Verbleibende Sequenz: F4-REVIEW → F4-PATTERNS → F4-IMPL-* → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F-N…" (oder wie auch immer der Sub-Thread das umbenennt) raus → Erledigt-Liste mit Eintrag für F4-PICK; Sektion für F4-REVIEW wird auf Position 1 angelegt mit Verweis auf die neue Inventur-Doc.
- **Memory**: nichts erwartet — Inventur-Methodik ist seit F-1.1/F-2.1/F-3.1 etabliert. Falls überraschend doch (z.B. „Async-Pipeline-Inventur braucht eigene State-Diagramm-Konvention für künftige async Features"): `feedback_*.md` schreiben.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Inventur-Methodik klar, Async-Pipeline-Erwartung im Sprint-Prompt verankert.)_
