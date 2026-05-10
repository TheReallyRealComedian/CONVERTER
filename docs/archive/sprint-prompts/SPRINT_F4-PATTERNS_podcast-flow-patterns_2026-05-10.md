# Sprint F4-PATTERNS — F-4.3 Patterns + Microcopy `podcast-flow`

**Datum**: 2026-05-10

**Ziel**: Stufe 3 (Patterns + Microcopy) der dreistufigen UX-Cascade-Methodik für `podcast-flow`. Aus den 16 Findings + 3 finding-linked Bug-Tickets aus F-4.2 konkrete UI-Pattern-Blöcke entwickeln, mit deutscher Microcopy, Aufwandsschätzung XS/S/M/L, Top-N-Quick-Wins per Impact-Score und Cluster-Vorschlag für F4-IMPL. **Async-Schwerpunkt** (62.5% Findings sind Async-spezifisch, H1+H9 dominieren): Pattern-Strukturierung folgt den 5/6 unzureichenden Async-State-Klassen. **Kein Code-Touch**.

**Vorbedingung**:
- Pytest 51/51 grün auf `main`. Letzter Code-Touch: F4-REVIEW (commit `3bde227`, 2026-05-10).
- **Eingabe**: [docs/ui_findings_podcast_flow_2026-05.md](docs/ui_findings_podcast_flow_2026-05.md) (Sub-Thread liest komplett vor Phase 1).
  - 16 Findings: Sev 4: 2 (F1+F2 Cancel-Lüge), Sev 3: 3 (F3 Stage-Progress, F4 File-Cleanup, F5 Browser-Reload), Sev 2: 6, Sev 1: 5.
  - 4 Bug-Tickets BT1–BT4: 3 finding-linked, 1 pure (BT4 Blob-URL-Revoke).
  - Cross-Feature-H4-Quote: **0%** — F-2 hat audio_converter.js-podcast-Block schon konvergiert. Keine künstlichen H4-Findings konstruiert.
  - Async-Verteilung: 62.5% Findings Async-spezifisch. Async-Heuristik-Sub-Sektion in Findings-Doc mappt Findings auf Async-State-Klassen.
  - 6 Findings mit `⚠️ code-only`-Marker: F1, F2, F4, F5, F8, F16. **Davon F1+F2+F4 nach Master-Walkthrough live verifiziert (siehe Master-Annotation unten) — Marker entfällt.** Verbleibende code-only: F5, F8, F16.
  - 4 Schwerpunkt-Cluster:
    - **Cluster 1: Cancel-und-Cleanup-Recovery** (F1+F2+F4, Sev 3-4) — live-verifizierte Sev 4 Cancel-Lüge.
    - **Cluster 2: Async-State-Visibility** (F3+F5+F6+F10+F16, Sev 1-3) — Stage-Progress-Fehlen, Architektur-Hebel via Worker-Logger-Lines.
    - **Cluster 3: Polling- und Defensiv-Robustheit** (F8+F13+F14+F15, Sev 1-2) — kleine Defensiv-Lücken.
    - **Cluster 4: Speaker-Format-Hilfe und Edit-Verhalten** (F11+F12, Sev 1-2 H6).
- **Methodik-Vorlagen** (Output-Format 1:1 reproduzieren):
  - F-1.3: [docs/ui_patterns_document_converter_2026-05.md](docs/ui_patterns_document_converter_2026-05.md) — 14 Patterns.
  - F-2.3: [docs/ui_patterns_audio_converter_2026-05.md](docs/ui_patterns_audio_converter_2026-05.md) — 21 Patterns + Cluster-I/II-Vorbereitung.
  - F-3.3: [docs/ui_patterns_library_detail_2026-05.md](docs/ui_patterns_library_detail_2026-05.md) — 15 Patterns + 3-Sub-Batch-Cluster.
- **Helper-Bestand in `_utils.js`** (Cross-Feature-H4-Reuse, niedrige Erwartung wegen 0%-Quote): `showAlert`, `showToast`, `formatFileSize`, `safeJSON`, `formatDatetimeLocalNow`, `.sr-only`-Utility.
- **Microcopy-Regeln** (für DE-Texte): Fehler max 2 Sätze, Empty-State max 3 Sätze, Buttons max 3 Wörter, keine Emojis bei Fehlern.
- **Impact-Score-Formel**: `Score = Sev × 5 / Aufwand-Gewicht`. XS=1, S=2, M=4, L=8.

**Out-of-scope**:
- Implementation — eigener Folge-Sprint `F4-IMPL`.
- Code-Änderungen jeglicher Art.
- **BT4** (pure Bug-Ticket Blob-URL-Revoke, kein UX-H-Aspekt): nicht im Pattern-Sprint, separater Bug-Sweep oder mit F4-IMPL mit-genommen wenn nahegelegene Patterns berührt werden.
- **BT1–BT3** (finding-linked): hier nicht eigenständig adressieren, werden via die Patterns ihrer Findings mit-gelöst.
- Befund 3 (Legacy `/generate-podcast` Dead-Code) — Hygiene-Welle.
- F-2.1-Doc-Korrektur (Service-Gate-Verhalten) — Doc-Hygiene-Welle.

---

## Master-Annotation (vorab eingebettet)

### Master-Walkthrough-Ergebnis 2026-05-10 — Cancel-Disk-Forensik

User hat Cancel-Disk-Forensik durchgeführt. Verifikation für die zwei Sev-4-Befunde + verzahntes File-Cleanup-Issue:

**Test-Lauf** (Job `9bf48e0a-7fb1-46bb-baac-5d0f88ef12c5`, monolog, deutsch):
- Job-Start: 04:00:55. User klickte Cancel mid-job.
- Worker-Verhalten: **lief 1:12 Min nach Cancel-Klick weiter, „Successfully completed" um 04:02:08**.
- WAV-File: `tmp_rgn3y1o.wav`, **11.7 MB im Volume `/var/lib/docker/volumes/converter_podcast_data/_data/` geschrieben**.
- TTS-Token: vollständig verbraucht (gemini-2.5-flash-preview-tts, 1:12 Min Generierung).
- Frontend-UI: **„unauffällig" — sieht aus, als wäre Cancel erfolgreich. Variante 1 aus dem Heuristik-Review = aktive Frontend-Lüge.**

**Live-verifiziert (Marker `⚠️ code-only` entfällt für diese drei Findings)**:
- **Befund 9 / F1 (Cancel-Lüge — Worker läuft weiter)**: BESTÄTIGT Sev 4. Cost-Verlust real (CPU + TTS-Token).
- **Befund 9 / F2 (Frontend-Status zeigt erfolgreichen Cancel trotz weiterlaufendem Worker)**: BESTÄTIGT Sev 4. Aktive Frontend-Lüge.
- **Befund 18 / F4 (File-Cleanup-vs-Re-Download)**: BESTÄTIGT Sev 3 + verzahnt mit F1/F2 — orphaned WAV bleibt im Volume liegen, weil Cleanup nur beim Download-Pfad triggert. Nach abgebrochenen Jobs: Disk-Wachstum.

**Drei verbindliche Pattern-Sub-Mechaniken für Cluster 1** (Cancel-und-Cleanup-Recovery):

1. **Cancel-Button muss tatsächlich Worker-Stop triggern**. RQ-`job.cancel()` allein reicht in rq 2.x nicht — der Job muss `cooperative cancel checks` haben (z.B. zwischen TTS-Chunks `if redis.exists(f"cancel:{job_id}"): raise CancelledError`), oder Worker-`SIGINT`/`Worker.kill_horse()`-Pfad. Echter Backend-Stop, nicht nur Status-Flip.
2. **Frontend-Lüge entfernen**. UI darf den Status nicht als „abgebrochen" zeigen, bevor der Worker tatsächlich `cancelled`/`failed` reportet. Confirmation-Dialog beim Cancel-Klick (DE-Microcopy: „Generierung abbrechen? TTS-Token werden teilweise schon verbraucht sein."), Polling läuft weiter mit Zwischenstate „wird abgebrochen …", endgültiger Status erst bei Worker-Confirmation.
3. **Orphaned-File-Cleanup**. Edge-case: wenn Worker trotz Cancel-Signal durchläuft (z.B. weil mid-TTS nicht clean stoppbar), das geschriebene WAV-File entweder beim Cancel-Backend-Pfad löschen, oder via periodischen Cleanup-Job (cron-Style, „lösche WAVs älter als X ohne Download-Trigger") aufräumen. Disk-Wachstum-Schutz.

**Restliche `⚠️ code-only`-Findings** (F5 Browser-Reload, F8 Polling-Edge-Cases, F16 Async-State): Walkthrough nicht durchgeführt. Pattern-Vorschläge tragen `🔥 Smoke-Pflicht in F4-IMPL`-Sub-Tag, F4-IMPL-Sub-Thread verifiziert vor Pattern-Apply.

### Architektur-Hebel für Cluster 2 (Async-State-Visibility)

Aus F-4.1 Out-of-Scope-Befund: `services/gemini/synthesis.py` hat 50+ Logger-Lines mit Stage-Markers (Skript-Generierung → TTS-Calls pro Speaker → Chunk-Konkatenation). Diese **schon-existierenden Log-Lines** könnten als `job.meta`-Source für Stage-Progress dienen — keine neue Telemetrie, nur die bestehenden Logger-Calls **zusätzlich** in `job.meta['stage'] = ...` schreiben. Pattern-Vorschlag für Cluster 2 soll diesen Hebel nutzen.

---

## Phase 1 — Patterns + Microcopy

Pre-Flight:

1. `git status -s` → clean tree erwartet.
2. **Findings-Doc + Inventur-Doc komplett lesen**.
3. **Methodik-Vorlagen lesen**: F-1.3 + F-2.3 + F-3.3 Pattern-Docs (Output-Format 1:1).
4. **`_utils.js`-Helper-Bestand verifizieren**: erwartet `showAlert`, `showToast`, `formatFileSize`, `safeJSON`, `formatDatetimeLocalNow`, `.sr-only`. Patterns sollen vorhandene Helper nutzen — neue nur mit Begründung am Doc-Ende.

**Pattern-Aufgabe**:

Für jeden Finding (oder konsolidierte Finding-Gruppe) ein Pattern-Block mit:

- **Pattern-Nummer** (P1, P2, …).
- **Adressiert Findings**: Liste mit H + Sev + linked BTs.
- **Live-Verifikation-Status** (NEU für F-4.3 wegen Master-Walkthrough): `✅ live-verifiziert` für Patterns die F1/F2/F4 adressieren; `🔥 Smoke-Pflicht in F4-IMPL` für Patterns die F5/F8/F16 adressieren; sonst leer.
- **UI-Pattern-Beschreibung**: 2-4 Sätze, konkrete Mechanik. Async-Patterns explizit über die State-Übergänge sprechen (queued→started→stage-progress→finished/failed/cancelled).
- **Visuelle Hinweise**: vorhandene CSS-Klassen aus Neomorphism + `.sr-only` aus F-3-IMPL nutzen.
- **Microcopy** (deutsch): exakte Strings für Confirmation-Dialoge, Status-Labels, Error-Banner, Tooltips. Nach Microcopy-Regeln.
- **Helper-Reuse**: welche `_utils.js`-Helper. **Erwartung sehr niedrig** wegen 0% Cross-Feature-H4 — nur wo Inline-Code in audio_converter.js dupliziert ist (kommt selten vor laut F-4.1 Befund 20).
- **Aufwand**: XS / S / M / L mit Begründung.
- **Impact-Score**: `Sev × 5 / Aufwand-Gewicht`.
- **Konsolidierung-Hinweis** falls mehrere Findings adressiert.

**Konsolidierungs-Logik** (analog F-1.3 / F-2.3 / F-3.3):

- Findings die zur selben Async-State-Übergang gehören → **ein** Pattern.
- Findings die zur selben Pipeline-Stage gehören → **ein** Pattern.
- Findings die durch denselben Architektur-Hebel gelöst werden (z.B. job.meta-Stage-Progress) → **ein** Pattern.
- **Cluster 1 (Cancel-und-Cleanup-Recovery)**: vermutlich **ein großes Pattern** mit den drei Sub-Mechaniken aus Master-Annotation (Worker-Stop + Frontend-Lüge-Fix + Cleanup), oder zwei (Cancel-Mechanik + Cleanup separat). Sub-Thread entscheidet pragmatisch — wenn die drei Sub-Mechaniken kohärenter Code-Touch sind: ein Pattern; wenn der Cleanup-Pfad strukturell separat ist: zwei.

**Erwartete Pattern-Anzahl**: 12-14 Patterns aus 16 Findings + 3 finding-linked BTs (~15-25% Konsolidierung wegen Async-thematischer Bündelung in Cluster 1+2).

**Output-Doc**: `docs/ui_patterns_podcast_flow_2026-05.md`. Struktur 1:1 wie F-1.3 / F-2.3 / F-3.3:

1. Header mit Findings-Quelle, Sprint-Datum, Methodik-Hinweis.
2. **Pattern-Blöcke** P1 bis Pn.
3. **Cluster-Vorschlag für F4-IMPL**: 1-3 Implementations-Cluster vorgeschlagen. **F-3-IMPL-Lehre**: 1-Sprint mit 3-Sub-Batches funktioniert wenn Helper-/Cluster-Reuse-Disziplin gewahrt; 12-Pattern-Schmerzgrenze gilt aber. Bei <13 Patterns ist 1-Sprint pragmatisch, bei ≥13 Sub-Batch-Strategie verankern.
4. **Top-5 Quick-Wins**: Tabelle nach Impact-Score absteigend.
5. **Smoke-Pflicht-Übersicht**: Liste der Patterns mit `🔥 Smoke-Pflicht in F4-IMPL`-Sub-Tag (Findings F5, F8, F16). Plus Liste mit `✅ live-verifiziert` für die drei Cluster-1-Patterns aus Master-Annotation.

Nach Phase 1: STOP — Bericht. Statistik (Pattern-Anzahl, Konsolidierungs-Quote, Aufwand-Verteilung, Smoke-Pflicht- und Live-verifiziert-Anzahl, Cluster-Vorschlag, Top-5-Quick-Wins).

---

## Phase 2 — Konsistenz-Check

Read-only. Sub-Thread liest die eigene Pattern-Doc nochmal und prüft:

1. **Vollständigkeit**: jeder der 16 Findings ist adressiert. BT1-BT3 in Patterns erwähnt, BT4 als „pure Bug-Ticket nicht in F-4.3" notiert.
2. **Live-Verifikation-Status-Konsistenz**: Patterns für F1/F2/F4 tragen `✅ live-verifiziert` mit Verweis auf Master-Walkthrough. Patterns für F5/F8/F16 tragen `🔥 Smoke-Pflicht in F4-IMPL`. Übersichts-Sektion und Pattern-Block-Tags stimmen überein.
3. **Architektur-Hebel-Konsistenz**: Cluster-2-Pattern für Stage-Progress nutzt explizit den Worker-Logger-Lines→job.meta-Hebel (siehe Master-Annotation), nicht eine alternative Telemetrie-Lösung.
4. **Helper-Reuse-Disziplin**: Cross-Feature-H4-Quote 0% bedeutet kein Helper-Reuse-Druck — falls trotzdem Helper-Reuse-Patterns entstanden, Begründung im Doc.
5. **Microcopy-Regeln**: Stichprobe — Confirmation-Dialog-Strings (≤2 Sätze), Status-Labels (≤3 Wörter), Empty-States (≤3 Sätze).
6. **Impact-Score-Konsistenz**: Top-5-Tabelle stimmt mit Pattern-Block-Score-Angaben überein.
7. **Disziplin**: keine konkreten Code-Diffs, kein Bug-Fix, keine Implementations-Schritte.

Nach Phase 2: STOP — Bericht. „Pattern-Doc konsistent" oder Liste der Korrekturen.

---

## Phase 3 — Commit + Push

- Plain-prose Commit-Message, mehrere `-m`-Flags, keine Backticks, keine Unicode-Pfeile, keine HEREDOCs.
- Ein Commit. Subject z.B. „F-4.3 / Stufe 3: patterns + microcopy of podcast-flow".
- Body: Statistik (Pattern-Anzahl, Konsolidierungs-Quote, Aufwand-Verteilung, Live-verifiziert-vs-Smoke-Pflicht-Verteilung, Cluster-Vorschlag, Top-5-Quick-Wins, Architektur-Hebel-Hinweis).
- Branch: direkt auf `main` ist OK.
- `git push origin main` direkt nach Commit ist Teil des Sprints. Wenn der Auto-Mode-Classifier blockt: Master pushed von Hand.

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off.

Bei Frust-Signal oder „mach jetzt einfach": einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — eine Output-Datei (`docs/ui_patterns_podcast_flow_2026-05.md`), Pattern-Blöcke + Microcopy + Aufwandsschätzung + Cluster-Vorschlag, kein Code-Touch, keine Tests, kein Smoke.

---

## Konstitutiv mit-genommen, falls berührt

- Wenn das Cluster-1-Pattern (Cancel-Mechanik) RQ-spezifische Recherche erfordert (z.B. „kann rq 2.8.0 cooperative cancel via redis-key-poll, oder müssen wir Worker.kill_horse() emulieren?"): kurz im Pattern-Block notieren, **nicht** im F-4.3-Sprint final auflösen — F4-IMPL macht die Recherche.
- Wenn beim Pattern-Schreiben fehlende Helper auffallen die für mehrere Patterns sinnvoll wären (z.B. ein generischer `confirmAction(msg, options)`-Helper für Cancel-Confirmation + andere Confirms): in „Helper-Vorschlag"-Sektion am Doc-Ende sammeln, nicht still mit-anlegen.
- Wenn ein Architektur-Hebel sich als Sackgasse erweist (z.B. job.meta-Stage-Progress-Idee scheitert weil Logger-Lines im Worker-Subprocess nicht synchron zugreifbar sind): kurz Bedenken vermerken — F4-IMPL muss dann eine Alternative finden.

Alles andere aus dem BACKLOG bleibt liegen.

---

## BACKLOG- und STATUS-Updates nach Abschluss

Sub-Thread pflegt am Ende:

- **STATUS.md**: „F4-PATTERNS ☑ done 2026-05-XX → commit `<hash>`. Patterns-Doc unter `docs/ui_patterns_podcast_flow_2026-05.md`. <N> Patterns (XS: a, S: b, M: c, L: d), <K> Cluster vorgeschlagen, Top-5-Quick-Wins-Score-Range <X.0–Y.0>, <V> Live-verifiziert + <S> Smoke-Pflicht. Verbleibende Sequenz: F4-IMPL → F-N… → WAVE-CLOSE."
- **BACKLOG.md**: Sektion „1. F4-PATTERNS" raus → Erledigt-Liste; Sektion „2. F4-IMPL" rückt auf Position 1 mit Pattern-Anzahl als Hint für Sprint-Schnitt-Entscheidung.
- **Memory**: nichts erwartet — Pattern-Methodik etabliert. Falls Async-spezifische Lehren auftauchen (z.B. „job.meta-Stage-Progress-Pattern für künftige async UX-Wellen"): `feedback_*.md` schreiben.

---

## Phase-0-Entscheidungen

_(Phase 0 in diesem Sprint nicht aktiviert — Pattern-Methodik etabliert, Master-Walkthrough-Annotation oben verankert die drei Cluster-1-Sub-Mechaniken vorab, Architektur-Hebel für Cluster 2 spec'd.)_
