# Sprint F4-IMPL-B — F-4 Implementation Cluster II + III (Async-Visibility + Polish + Defensiv)

**Datum**: 2026-05-10

**Ziel**: Cluster II + III aus F-4.3 implementieren — die 10 verbleibenden Patterns nach F4-IMPL-A. Schwerpunkte: **Async-State-Visibility** (queued/started/stage-progress unterscheidbar machen, Architektur-Hebel job.meta), **Polling-Robustheit** (Edge-Cases, Browser-Reload-Recovery, Service-Gate), **Polish + Skript-Format-Hilfe**. Niedriger Risiko-Profil als F4-IMPL-A (keine Sev-4, höchstes Item M-Aufwand auf P3).

**Vorbedingung**:
- Pytest 60/60 grün im Container. Letzter Code-Touch: F4-IMPL-A (commits `e6e9443` + `9d5070d`, 2026-05-10).
- **Eingabe**: [docs/ui_patterns_podcast_flow_2026-05.md](docs/ui_patterns_podcast_flow_2026-05.md). Sub-Thread liest **vor jedem Pattern-Apply** den zugehörigen Pattern-Block.
- **Patterns dieses Sprints**: P3, P4, P5, P6, P7, P8, P9, P10, P11, P12 (10 Patterns aus Cluster II + Cluster III von F-4.3).
- **Empfohlene Apply-Reihenfolge** (vom F4-IMPL-A-Sub-Thread vorgeschlagen, Quick-Wins-zuerst-Logik):
  - **P5** (queued/started-Konflation, F6, Sev 2, XS) — **Drift-Check**: F4-IMPL-A hat stopped/canceled/failed-mit-meta→cancelled-Mapping schon implementiert. Restscope ist nur die queued/started-Differenzierung.
  - **P6** (Download-Toast-Guard, F10, Sev 2, XS) — **BT4-Folde** (Audio-Blob-URL-Revoke ist die natürliche Mit-Fix-Stelle in der Download-Mechanik).
  - **P12** (Tab-Disabled-Konsistenz, F9, Sev 2, XS).
  - **P4** (Browser-Reload-LocalStorage, F5, Sev 3, S) — **🔥 Smoke-Pflicht**: Browser-Reload mid-Job-Polling, Job-ID aus localStorage wiederherstellen.
  - **P7** (Polling-Visibility-Konsolidierung, F7+F16, Sev 1-2, S) — **🔥 Smoke-Pflicht**.
  - **P11** (Skript-Textarea-Readonly, F12, Sev 1, XS).
  - **P8** (Polling-Edge-Cases, F8+F15, Sev 1-2, S) — **🔥 Smoke-Pflicht**. **Drift-Befund**: F15 ist seit F4-IMPL-A live verifizierbar (rq-stopped-Status taucht jetzt tatsächlich auf nach Cancel).
  - **P9** (Service-Gate, F13+F14, Sev 1-2, S).
  - **P3** (Stage-Progress, F3, Sev 3, M) — **Architektur-Hebel**: existierende Logger-Lines in `services/gemini/synthesis.py` als job.meta-Source. **Plan-B-Vermerk**: bei Subprocess-Sync-Risiko Redis-Key `podcast:stage:{job_id}` als Fallback.
  - **P10** (kleinste Polish, niedriger Aufwand).
- **Live-verifiziert** durch F4-IMPL-A-Smoke (kein `⚠️ code-only` mehr): F15 (rq-stopped-Status real), F-001-Pattern-Pfade (Cancel-Endpoint-Recherche im IMPL-A hat das bestätigt).
- **Smoke-Pflicht in F4-IMPL-B** für: P4 (Browser-Reload-Recovery), P7 (Polling-Konsolidierung), P8 (Polling-Edge-Cases) — alle erben den `🔥`-Status aus F-4.3 Master-Annotation.
- **Methodik-Vorlagen**:
  - F-3-IMPL holistic Sub-Batches als Multi-Pattern-Mechanik (commits `843574b` / `40dd02e` / `b3e666a` / `5ba29c1`).
  - F4-IMPL-A für Async-spezifische Pattern-Mechanik (commit `e6e9443`).
- **Helper-Bestand in `_utils.js`**: `showAlert`, `showToast`, `formatFileSize`, `safeJSON`, `formatDatetimeLocalNow`, `.sr-only`. **Kein** `confirmInPlace` (lokal in audio_converter.js geblieben aus F4-IMPL-A — wenn F4-IMPL-B eine zweite Call-Site findet: jetzt nach `_utils.js` extrahieren).
- **Memory-Layer-Pflicht-Lese**:
  - `reference_converter_dep_bump_constraints.md` — neuer rq-2.x-Cancel-Block aus F4-IMPL-A.
  - `feedback_no_silent_fixes.md`.
  - `feedback_pragmatic_merge.md`.

**Out-of-scope**:
- F4-IMPL-A Patterns (P1+P2) — bereits durch.
- WAVE-CLOSE.
- Konstitutive Befunde (Legacy `/generate-podcast`, F-2.1-Doc-Korrektur).

---

## Phase 1 — Implementation

Pre-Flight (vor jedem Pattern-Apply):

1. `pytest tests/` — muss 60/60 grün sein im Container.
2. `git status -s` → clean tree erwartet.
3. **Pattern-Block in der Patterns-Doc lesen** vor jedem Apply.
4. **Drift-Check für teil-erfüllte Patterns** (P5): `app_pkg/podcasts.py` `/podcast-status`-Branch lesen, prüfen welche Status-Mappings F4-IMPL-A schon eingebaut hat. Restscope dokumentieren bevor Apply.

### Apply-Reihenfolge mit Mechanik-Hinweisen

**P5 — queued/started-Konflation aufheben** (F6, Sev 2 H1, XS):
- Drift: F4-IMPL-A hat stopped/canceled/failed→cancelled-Mapping. P5-Restscope: nur queued vs. started differenzieren in JSON-Response von `/podcast-status`.
- Frontend: Status-Display unterscheidet „In Warteschlange" vs. „Wird generiert".

**P6 — Download-Toast-Guard + BT4-Folde** (F10, Sev 2 H9, XS):
- Backend `/podcast-download` führt File-Cleanup nach Stream — wenn Stream fehlschlägt: Toast statt silent.
- BT4 mit-fix: Audio-Blob-URL-Revoke nach Download oder bei Page-Unload.

**P12 — Tab-Disabled-Konsistenz** (F9, Sev 2 H6, XS):
- Tab-Buttons müssen aria-disabled sein wenn Pane nicht aktivierbar (Service-Gate).

**P4 — Browser-Reload-LocalStorage** (F5, Sev 3 H1, S, **🔥 Smoke-Pflicht**):
- Frontend speichert active job_id in localStorage beim Polling-Start.
- Bei Page-Reload: prüfen ob localStorage einen aktiven Job hat → Polling-Loop wieder aufnehmen oder Status-Display rekonstruieren.
- **Smoke**: Browser-Reload mid-Job → erwartet Polling-Loop läuft weiter, Status sichtbar. Wenn Smoke zeigt nicht reproduzierbar: STOP.

**P7 — Polling-Visibility-Konsolidierung** (F7+F16, Sev 1-2 H1, S, **🔥 Smoke-Pflicht**):
- Inline Status-Display-Code als Helper konsolidieren (lokal oder `_utils.js`).
- **Smoke**: Status-Übergänge (queued → started → finished) visuell verifizieren.

**P11 — Skript-Textarea-Readonly** (F12, Sev 1 H6, XS):
- Während aktivem Job: Skript-Textarea readonly (User soll nicht edit-en während Worker generiert).

**P8 — Polling-Edge-Cases** (F8+F15, Sev 1-2 H9, S, **🔥 Smoke-Pflicht**):
- F15-Drift: cancelled-Status existiert jetzt real seit F4-IMPL-A. P8 verifiziert dass Polling-Loop ihn korrekt handhabt.
- Network-Drop während Polling: Retry-Logik oder Banner.
- **Smoke**: Network-Throttle „Offline" mid-Polling → erwartet Banner + Reconnect.

**P9 — Service-Gate** (F13+F14, Sev 1-2 H4+H6, S):
- F-013-Allowlist-Konsistenz-Gap: Frontend-Validation für narration_style/script_length/num_speakers analog zur Backend-Allowlist aus SEC.
- Service-down-Banner-Wirkung präzisieren (Pane bleibt sichtbar mit disabled-Controls).

**P3 — Stage-Progress** (F3, Sev 3 H1, M, Architektur-Hebel):
- **Default**: existierende Logger-Lines in `services/gemini/synthesis.py` zusätzlich in `job.meta['stage'] = ...` schreiben. Status-Endpoint reichen `meta['stage']` durch.
- **Plan-B bei Subprocess-Sync-Risiko**: Redis-Key `podcast:stage:{job_id}` mit `redis.setex(... ex=3600)`. Sub-Thread evaluiert während Apply ob job.meta-Default funktioniert; bei Race-Conditions oder Worker-Subprocess-Isolation auf Plan-B umstellen mit Begründung im Bericht.
- Frontend: neue Stage-Anzeige zwischen „Wird generiert" und Progress-Bar.

**P10 — kleinste Polish** (Sev 1, XS):
- Sub-Thread checkt Patterns-Doc für genauen Inhalt.

### Mechanik-Leitplanken

- **DE-Microcopy** für alle neuen Strings.
- **Helper-Reuse** prüfen: `confirmInPlace` ist lokal in audio_converter.js — wenn P-Pattern eine zweite Confirmation braucht, jetzt nach `_utils.js` ziehen.
- **F-001-Pattern**: alle Status-/Cancel-Endpoints behalten narrow-except (`NoSuchJobError` → 404, andere → 500).
- **Container-pytest**: alle Tests im Container laufen lassen, nicht lokal.
- **Erwartete Final-Test-Anzahl**: 60 + 3-5 neue = **63-65 Tests grün**.

### Sub-Batch-Strategie (optional, pragmatisch)

10 Patterns ist innerhalb der F-2-Cluster-I-Schmerzgrenze (12). Default: alle 10 in einem Sprint, sequenziell in obiger Reihenfolge. Wenn der Sub-Thread merkt dass die Verkopplung höher ist als gedacht (z.B. P3 zieht Worker-Refactor nach sich): kann in zwei Sub-Batches splitten — A=P5,P6,P12,P11 (XS-Polish), B=P4,P7,P8,P9,P3,P10 (Smoke-Pflicht + M-Aufwand). STOP-Punkt zwischen Sub-Batches.

**Erwartete Files**:

```
app_pkg/podcasts.py                    # P5 status-mapping, P3 status-meta-stage durchreichen
tasks.py                               # P3 job.meta['stage'] writes (oder Plan-B redis-key writes)
services/gemini/synthesis.py           # P3 Stage-Markers an Logger-Calls hängen
static/js/audio_converter.js           # P5+P6+P11+P4+P7+P8+P9+P10+P12 Frontend-Touches
templates/audio_converter.html         # eventuell EDIT — Tab-Disabled-Markup, Stage-Display-Mountpoint
static/js/_utils.js                    # eventuell EDIT — confirmInPlace nur wenn zweite Call-Site
tests/test_podcasts.py                 # 3-5 neue Tests
```

Nach Phase 1: STOP — Bericht. Welche Patterns durch, ob P3-Plan-A oder Plan-B gewählt, ob Sub-Batches gebildet wurden, neue Test-Anzahl, Mid-Sprint-Drifts.

---

## Phase 2 — Verify

1. `docker exec markdown-converter-web python -m pytest tests/` final grün (63-65 erwartet).
2. **Live-Smoke für die 3 🔥-Patterns**:
   - **P4 Browser-Reload-Recovery**: Job starten → Browser F5 → Status-Display kommt wieder, Polling läuft weiter.
   - **P7 Polling-Visibility**: Job durchlaufen → queued/started/stage-progress/finished alle sichtbar im Status-Display.
   - **P8 Polling-Edge-Cases**: Network-Throttle „Offline" mid-Polling → Banner + Reconnect; Cancel-Status (jetzt real seit F4-IMPL-A) wird korrekt angezeigt.
3. **Cluster-I-Regression-Smoke**: Cancel-Disk-Forensik nochmal mit den IMPL-B-Änderungen — Worker stoppt weiterhin, kein orphaned WAV. Sicherstellen dass IMPL-B-Refactor keine IMPL-A-Mechanik gebrochen hat.
4. **Stage-Progress-Smoke (P3)**: Job durchlaufen → Stage-Display zeigt Übergänge („Skript wird generiert" → „TTS-Synthese" → „Audio-Konkatenation").
5. Server-Logs auf neue Warnings/Errors checken.

Nach Phase 2: STOP — Bericht. Live-Smoke-Resultate, IMPL-A-Regression-Status.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Default: **ein Commit für F4-IMPL-B**. Falls Sub-Batches gefahren: zwei Commits.
- Branch: direkt auf `main` ist OK.
- `git push origin main`. Wenn Auto-Mode-Classifier blockt: Master pushed von Hand.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

**Zusätzlich für F4-IMPL-B**:
- Bei rotem Test-Run: STOP.
- Bei P3-Architektur-Hebel-Sackgasse (Logger-Lines können nicht synchron in job.meta schreiben): STOP, Master entscheidet ob Plan-B oder P3-Re-Skopung.
- Bei Smoke-Pflicht-Pattern-Befund nicht reproduzierbar (P4/P7/P8): STOP, nicht silent-fixen.
- Bei Cluster-I-Regression im Verify-Smoke (Cancel funktioniert plötzlich nicht mehr): STOP, IMPL-A-Mechanik darf nicht gebrochen werden.

---

## Größe

**M-L** — 10 Patterns aber meist niedriger Aufwand (5× XS, 4× S, 1× M auf P3). Höchstes Risiko-Item P3 mit Architektur-Hebel + Plan-B.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn beim P5/P3-Touch der `/podcast-status`-Endpoint substantiell wächst (mehrere Status-Quellen: rq-Status + meta + Cancel-Flag + Stage): kurz prüfen ob Helper-Funktion sich aufdrängt. Default: lokal, F4-IMPL-B macht keine Service-Refactor-Welle.
- Wenn beim P3-Logger-to-job.meta-Hebel zusätzliche Stages auffallen die nicht im Patterns-Doc gelistet sind: notieren, **nicht** spontan exposen — Patterns-Doc ist Spec.
- Wenn beim Frontend-Touch der Helper `confirmInPlace` für eine zweite Stelle nützlich wäre: nach `_utils.js` extrahieren mit Begründung.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F4-IMPL-B ☑ done 2026-05-XX → commit `<hash>`. P3-P12 implementiert. Pytest <neue Anzahl>/<neue Anzahl> grün im Container. Live-Smoke clean inkl. P4 Browser-Reload + P7 Polling-Visibility + P8 Polling-Edge-Cases + P3 Stage-Progress + Cluster-I-Regression. P3-Mechanik: <job.meta-Default / Redis-Key-Plan-B>. **F-4 strukturell abgeschlossen** für podcast-flow. Verbleibende Sequenz: F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F4-IMPL-B" raus → Erledigt-Liste mit allen 10 Pattern-IDs. Sektion „2. F-N…" rückt auf Position 1 mit Hinweis auf neue Feature-Wahl-Entscheidung. Folge-Sprint-Nummern -1.
- **Memory**: nur wenn Async-spezifische Lehren aufgetaucht (z.B. „job.meta-Stage-Progress-Pattern für künftige async Features" oder „rq-Subprocess-Sync-Risiko für künftige Worker-Tasks"). Defensiv.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Patterns vollständig spezifiziert, Drift-Befunde aus F4-IMPL-A in Apply-Reihenfolge integriert, Architektur-Hebel mit Plan-B-Mechanik im Pattern-Block.)_
