# OVERSEER HANDOFF — CONVERTER UX-Backlog

**Erstellt:** 2026-05-03 vor Maschinen-Wechsel von Mintbox auf neuen Host.
**Empfänger:** Neuer Claude-Code-Agent, der die Overseer-Rolle für den UX-Backlog übernimmt.
**Lese-Reihenfolge:** Dieses Dokument zuerst, dann `CLAUDE.md`, dann die `docs/`-Inhalte (siehe unten).

---

## TL;DR — Was du sofort tun musst

1. **Lese dieses Dokument vollständig.** Es ist selbsterklärend gestaltet, du brauchst keinen Vorwissen-Kontext aus der vorigen Session. Anschließend `CLAUDE.md` für den Live-Status der Stages.
2. **Schreibe keine Code-Änderungen selbst.** Du bist Overseer — du draftest Sub-Thread-Prompts, mergst deren Output, pflegst Status. Code-Arbeit machen frische Sub-Threads in eigenen Worktrees.
3. **Drei offene Aufgaben warten:**
   - (a) **`document_converter` ist broken** (von Oliver gemeldet, Ursache unbekannt) → Diagnose-Stage drafften, sobald Oliver da ist
   - (b) **F-2 Cluster I Live-Smoke** noch nicht durchgeführt — Container muss rebuilt + manuell geklickt werden bevor Cluster II startet
   - (c) **F-2 Cluster II** (P13–P21, Polish + a11y) — drafften nach Smoke-Confirmation

---

## 1. Wer du bist — die Overseer-Rolle

Du bist **Backlog-Master / Overseer** in einem Sub-Thread-getriebenen Arbeitsmodus, den Oliver "Overseer-Pattern" nennt. Konkret:

- **Du draftest Stage-Prompts** für frische Claude-Code-Sessions. Diese Prompts sind self-contained — der Sub-Thread bekommt keinen Kontext aus dem Hauptthread, also muss alles im Prompt stehen oder durch Datei-Verweise ins Repo zugänglich sein.
- **Du arbeitest die Sub-Thread-Reports ab:** ff-merge in `main`, Status in `CLAUDE.md` aktualisieren, ggf. neuen Stage-Prompt drafften.
- **Du machst keine Code-Änderungen** außer minimaler Bookkeeping (CLAUDE.md-Status-Updates, Merging, gelegentlich kleinste Helper-Korrekturen). Implementations-Arbeit ist Sub-Thread-Aufgabe.
- **Du wartest aktiv auf Oliver** für:
  - Auswahl-Entscheidungen (welches Feature als nächstes? welcher Implementierungs-Pfad bei Optionen?)
  - Live-Smoke-Bestätigungen (Browser-Tests die du nicht machen kannst)
  - Push-Aktionen (Credentials sind nicht in der Session)

**Was du NICHT bist:** Implementierer. Pair-Programmer. Code-Reviewer-im-Detail. Du operierst auf Plan-Ebene und bewegst Patterns/Cluster durch den Workflow.

---

## 2. Was CONVERTER ist (Repo-Kontext)

Flask-Webapp für Multimedia-Konversion: Markdown→PDF, Document→Markdown (PDF/DOCX/PPTX/EML/HTML/TXT/MD), Audio-Transkription (Deepgram), AI-Podcast-Generierung (Google Gemini TTS). Läuft im Docker-Compose mit Redis/RQ-Worker.

**Nutzer:** Single-User-App — nur Oliver. LAN-only auf Port 5656, login-protected. Kein Multi-User, kein öffentlicher Zugang. Das ist wichtig für UX-Bewertungen: Konsistenz (Nielsen H4) hat hohes Gewicht (Olivers mentales Modell), Wiedererkennung (H6) hat geringeres Gewicht (App ist vertraut).

**Architektur-Highlights** (vollständig in `CLAUDE.md`):
- `app.py` Bootstrap-Shim → `app_pkg/` mit per-Feature `register(app)`-Modulen (kein Flask Blueprint)
- `services/gemini/` Package, `services/deepgram_service.py`, etc.
- `tasks.py` + `worker.py` für RQ-Background-Jobs (Podcast-Generierung)
- `tests/` mit Charakterisierungs-Tests (37 grün, ~6 s) — mocken auf SDK-Singleton-Boundary, **nicht** auf UI/Browser

---

## 3. Methodik — Cascade nach Duan et al. CHI 2024

Pro Feature läuft eine **3-Stufen-Kaskade** (Stufen-Outputs sind Inputs der nächsten Stufe; manuelle Korrektur dazwischen ist gewollt):

1. **F-X.1 Inventur** (Rolle: UI-Architekt) — alle interaktiven Elemente kartieren, Element-Typ + Label + Aktion + States (default/hover/focus/disabled/loading/error/success/empty + audio-spezifische). Kein Bewerten. Output: `docs/ui_inventory_<feature>_2026-05.md`.

2. **F-X.2 Heuristik-Review** (Rolle: Senior UX-Experte, Nielsen) — Inventur durch Nielsen H1/H4/H6/H9 filtern, Schweregrad 1–4. Cross-Feature-H4 ab F-2 explizit (audio bricht F-1's etablierte Konventionen). Output: `docs/ui_findings_<feature>_2026-05.md`.

3. **F-X.3 Patterns + Microcopy** (Rolle: UX-Microcopy-Designer) — pro Finding konkretes UI-Pattern, deutsche Microcopy (Du-Form), Aufwand-Schätzung XS/S/M/L. Aggressiv konsolidieren (mehrere Findings → ein Pattern). Output: `docs/ui_patterns_<feature>_2026-05.md`.

Danach: **Implementation in Sprint-Clustern.**

**Cluster-Strategie:**
- F-1 (`document_converter`) hat 7 kleine Cluster gefahren (Cluster A bis Polish-2). Im Nachgang als zu kleinteilig empfunden.
- F-2 (`audio_converter`) fährt **2 große Cluster** auf Olivers Wunsch ("weniger dafür größere Sprints"):
  - Cluster I = Sev 4 + Sev 3 Patterns
  - Cluster II = Sev 2 + Sev 1 + Microcopy-Polish
- **Live-Smoke-Gate zwischen I und II ist obligatorisch** — verhindert das F-1-Problem, dass alle 7 Cluster un-smoked übereinander gestapelt wurden und subtile Regressionen nicht gefangen wurden (siehe Sektion 8 unten — `document_converter` ist genau dadurch broken geworden, vermutlich).

**Microcopy-Konventionen** (aus F-1 etabliert):
- Du-Form, deutsch
- Buttons max 3 Wörter, Verb + Objekt, aktiv ("In Library speichern")
- Fehler max 2 Sätze, Empty-State max 3 Sätze
- Keine Emojis in Fehlermeldungen; dezente Unicode-Glyphs (✓ ⚠ ⓘ ✗) erlaubt
- Keine Error-Codes, keine SDK-Namen, keine technischen Begriffe
- Filename-Format DE-Locale: "222 B" / "4,6 KB" / "1,2 MB" (Komma als Dezimaltrenner)
- Datums-Format DE-Locale: "dd.mm.yy hh:mm" via `Intl.DateTimeFormat('de-DE', {dateStyle:'short', timeStyle:'short'})`

---

## 4. Tools-Constraints auf neuer Maschine

**KRITISCH:** Auf der neuen Maschine sind die Browser-MCP-Tools **NICHT verfügbar**:
- `mcp__Claude_in_Chrome__*` — Browser-Automation
- `mcp__Claude_Preview__*` — Headless Preview

**Konsequenz für Sub-Thread-Prompts:**

Bei Inventur-Stages (F-X.1) bisher hatten Sub-Threads via Chrome-MCP echte Live-Walkthroughs gemacht (Permission-Prompts triggern, Drag-Drop simulieren, States provozieren). Das geht nicht mehr.

**Adaptation:** Inventur-Prompt-Vorlage anpassen — Live-Walkthrough wird zur **Oliver-manuell-Aufgabe**. Der Sub-Thread:
- Macht statische Code-Analyse (Pflicht, wie bisher)
- Liefert eine **konkrete Smoke-Pfad-Liste** als Output (klick-für-klick)
- Markiert in der Tabelle alle States als `?` wenn Live-Verifikation erforderlich wäre
- Oliver klickt manuell durch und reicht Befunde nach
- Inventur wird in einem zweiten Commit ergänzt (Live-Spalte gefüllt, oder Notation "live verifiziert von Oliver")

Das verlangsamt Stage F-X.1 leicht, ist aber sauber. Heuristik-Review (F-X.2) und Patterns (F-X.3) brauchen keine Live-Walkthroughs — die laufen unverändert.

**Implementation-Stages:** ohne Browser-Tooling ist Live-Smoke der Sub-Threads sowieso oft zurückgehalten worden (siehe F-1-Geschichte). Sub-Threads machen Static-Verifikation (`pytest`, `node --check`, grep), Live-Smoke macht Oliver. Das war auch auf Mintbox so — kein echter Verlust.

---

## 5. Aktueller Stand (Stand 2026-05-03)

### F-1 `document_converter` — strukturell durch, aber ⚠️ broken in Production

- **Cascade:** alle drei Stages ☑
- **Implementation:** alle 7 Cluster ☑ (A, B, C, D, E, Polish-1, Polish-2)
  - 14 Patterns implementiert, 3 Bug-Tickets (B1–B3) strukturell mitgelöst
  - Pytest 37/37 grün durchgängig
- **⚠️ PRODUCTION-REGRESSION:** Oliver berichtet `document_converter` ist aktuell nicht funktional. Keine Detail-Diagnose verfügbar — die Information kam erst beim Maschinen-Wechsel.
  - **Vermutung:** eine der 7 Cluster-Implementierungen hat eine Regression eingeführt, die `pytest` nicht fängt. Pytest mockt SDK-Boundaries (`app.deepgram_service`, `app.gemini_service`, `app.task_queue`) und testet HTTP-Boundary, **nicht** UI/JS/Browser-Verhalten.
  - **Wahrscheinliche Kandidaten:**
    - Cluster D Backend-Whitelist (`ACCEPTED_EXTENSIONS` in `app_pkg/documents.py`) hat eine Datei-Endung blockiert die früher funktionierte — z.B. wenn Oliver eine `.htm`/`.html` (oder anderes) submittet
    - Cluster A `resetSaveBtn()` Helper hat Save-Btn-Selektor zerstört
    - Cluster B/C `showAlert`-Migration hat Container-ID falsch
    - Polish-1 DE-String hat Selektor in JS getroffen (z.B. `if (button.textContent === 'Save to Library')` jetzt nicht mehr matcht)
  - **Empfehlung erste Aufgabe:** Diagnose-Sub-Thread (Prompt-Vorlage in Sektion 11 unten). Oliver beschreibt Symptome, Sub-Thread reproduziert + bisect.

### F-2 `audio_converter` — Cluster I done, Live-Smoke ausstehend

- **Cascade:** alle drei Stages ☑
- **Implementation:**
  - **Cluster I** (P1–P12, Sev 4+3): ☑ done — commit `ef78508`
    - 12 Patterns implementiert, 8 Bugs (B1–B8) strukturell mitgelöst
    - Pytest 37/37 grün
    - Helper-Reuse durchgängig (`showAlert`/`showToast`/`formatFileSize` aus `_utils.js`); 935 LOC in `audio_converter.js` umgebaut
    - **⚠️ Live-Smoke noch nicht durchgeführt** — obligatorisches Gate vor Cluster II
  - **Cluster II** (P13–P21, Sev 2+1): ☐ awaiting Live-Smoke-Confirmation

### F-3..F-N — queued

Wahl ausstehend. Kandidaten:
- `markdown_converter` (166 LOC Template — Editor + Preview + PDF + Reader-Mode)
- `library_detail` (133 LOC — View/Edit/Delete/Notion-Integration)
- `library` (118 LOC — Liste/Filter/Empty-State)
- `mermaid_converter` (79 LOC)
- `login` (64 LOC)
- podcast-generation flow (cross-template async polling)

### Cleanup-Plan (historisch, archiviert)

Vor F-1 lief eine separate Code-Cleanup-Welle (Stages 0–7) — komplett abgeschlossen, archiviert nach `docs/cleanup_plan.md`. Enthält 18 Findings (F-001 bis F-018), Cleanup-Stage-Outputs, deferred CVE-Upgrade-Items. Architektonische Entscheidungen daraus (`register(app)`-Pattern, Service-Singleton-Pattern) sind in `CLAUDE.md` § Architecture Notes gespiegelt.

**F-006 Status (aus Cleanup-Stage 4):** "alle drei Upload-Endpoints validieren weder Extension noch Content-Type". F-1 Cluster D hat das **für `document_converter` geschlossen** (`ACCEPTED_EXTENSIONS` + Backend-Whitelist + 400-Pfad). **Offen:** für `markdown_converter` und `audio_converter` — wird in deren F-N-Stages mit-adressiert wenn passend.

---

## 6. Konkrete nächste Schritte (in priorisierter Reihenfolge)

### Schritt 1 — `document_converter` Regression diagnostizieren

Sobald Oliver wieder am Rechner ist:
- Frag ihn: **welche Symptome zeigt `document_converter`?** (Crash? Stille? Falsche Daten? Welcher Pfad — Submit, Save, Clear?)
- Dann drafte einen Diagnose-Sub-Thread-Prompt (Vorlage in Sektion 11). Output: `docs/regression_document_converter_2026-05.md` mit Befund + Reproduktion + vermuteter Cluster.
- Basierend auf Befund: Hot-Fix-Stage drafften (Microscope auf einen Cluster) ODER Revert-Stage falls die Regression diffus ist.

### Schritt 2 — F-2 Cluster I Live-Smoke

Wenn der document_converter-Befund das nicht blockiert, Oliver bittet manuell zu smoken:

```bash
docker compose -f /home/oliver/CODE/CONVERTER/docker-compose.yml up -d --build markdown-converter
timeout 20 docker logs -f markdown-converter-web 2>&1 | tail -50
```

(Beachte: Service heißt `markdown-converter`, Container heißt `markdown-converter-web`. Beide gemeint.)

Smoke-Pfad (13 Steps) ist im letzten Overseer-Commit-Body und in der Conversation-History. Hier die Kurzform:

**File-Tab:** Empty-Submit (Banner) → Drag-Drop (real funktional jetzt) → Submit → Save → "✓ Gespeichert" → neue Datei → Save-Btn neutral → Copy-ohne-Result (Warning statt Placeholder)

**Live-Tab:** Mic-Klick → Permission-Deny (Recovery-Hinweis) → Allow (Loading-Spinner) → Recording → Sprache umschalten (visuell disabled) → Live-Textarea readonly mit Status-Hint → Stop → Textarea editierbar → Save

**Tab-Scoping:** Wenn ein API-Key fehlen würde, andere Tabs bleiben funktional (P8)

**DE-Microcopy quer:** alle UI-Strings deutsch

### Schritt 3 — F-2 Cluster II drafften

Nach erfolgreichem Smoke (oder dokumentiertem Hot-Fix für etwaige Befunde):
- Lies `docs/ui_patterns_audio_converter_2026-05.md` § Pattern-Blöcke P13–P21 + § Cluster-Vorbereitung
- Drafte einen self-contained Implementation-Prompt nach dem Muster aus den vorigen Cluster-Prompts (Vorlage in Sektion 11)
- Microcopy-Konventionen (Sektion 3) bleiben

### Schritt 4 — F-3 Auswahl

Nach F-2 Cluster II + Live-Smoke + Status-Update:
- Frag Oliver welches Feature als nächstes
- Drafte F-3.1 Inventur-Prompt — **mit der angepassten "Static + Oliver-manuell"-Strategie** (siehe Sektion 4)

---

## 7. Workflow — wie Sub-Threads laufen

### Sub-Thread spawnen

1. Oliver öffnet eine **frische Claude-Code-Session in diesem Repo** (Claude Code legt automatisch einen neuen Worktree unter `.claude/worktrees/<adjective-name>/` an, ausgecheckt auf `claude/<adjective-name>` Branch).
2. Oliver paste'n den vom Overseer gedraftete **Stage-Prompt** wörtlich in den Sub-Thread.
3. Sub-Thread arbeitet ab — liest Repo-Files, schreibt Output, committed auf seinen Worktree-Branch.
4. Sub-Thread postet Report im **Overseer-Thread** (zurück zu dir): "Stage X done — commit `<sha>` auf `<branch>`" + Per-Item-Status.

### Output mergen

Du als Overseer:

```bash
# 1. Optional: fetch falls Sub-Thread auf neuer Maschine arbeitete
git -C /home/oliver/CODE/CONVERTER fetch origin

# 2. ff-merge des Sub-Thread-Branchs
git -C /home/oliver/CODE/CONVERTER merge --ff-only claude/<adjective-name>

# 3. CLAUDE.md Status-Zeile updaten (Edit-Tool)
# 4. Bookkeeping-Commit
git -C /home/oliver/CODE/CONVERTER add CLAUDE.md
git -C /home/oliver/CODE/CONVERTER commit -m "<Stage> status: ☑ done — <kurze Zusammenfassung>"
```

**ff-merge ist immer das Ziel** — Sub-Thread-Branches sind aus dem aktuellen `main` ausgecheckt, Oliver committet keinen Code parallel auf `main`. Wenn ff scheitert: Stop. Vermutlich hat Oliver dazwischen was gemacht oder ein Worktree-Branch ist verschoben.

### Push

```bash
git -C /home/oliver/CODE/CONVERTER push origin main
```

**Auth-Hinweis:** auf Mintbox-Setup war HTTPS-Remote ohne Credential-Helper, kein `gh auth`. Oliver musste interaktiv pushen. **Auf neuer Maschine prüfen:**

```bash
git -C /home/oliver/CODE/CONVERTER remote -v   # → HTTPS oder SSH?
gh auth status                                 # → ist gh authenticated?
```

Wenn Push aus deiner Session nicht geht: Oliver pusht manuell. Sag's ihm im Report ("X commits ahead of origin, push wenn du magst").

### CLAUDE.md pflegen

Lebende Status-Datei. **Sparsam mit neuen Sektionen.** Nach Cleanup-Plan-Archivierung ist die Struktur:

```
# CONVERTER - Multimedia Converter & Podcast Generator
## What is this? / Tech Stack / Key Files / Running / Gemini Models Used / Architecture Notes
## Historical (Pointer auf docs/cleanup_plan.md)
## UX Review Plan (Methodik + Status-Block)
### F-1: document_converter (Status-Zeilen)
### F-2: audio_converter (Status-Zeilen)
### F-3..F-N: queued
## How to launch a UX cascade step in a fresh thread (Footer)
```

Pro Stage-Abschluss: Status-Zeile ändern (☐ → ☑ + Kurz-Zusammenfassung), ggf. nächste Stage in der Queue ergänzen. **Bookkeeping-Commits sollten klein sein** (1–3 Zeilen Diff in CLAUDE.md), aber konsequent — sie sind die Audit-Trail.

---

## 8. Lessons Learned aus F-1 (was du anders machen sollst)

Die F-1-Erfahrung hat fünf konkrete Lehren produziert:

1. **Live-Smoke nicht akkumulieren.** F-1 hat 7 Cluster un-smoked übereinander gestapelt, weil Sub-Threads keine Container-Manipulation machen wollten und Oliver es nach hinten verschoben hat. Result: Production-Regression sichtbar erst Tage später, Ursache nicht mehr eindeutig zuordenbar. **F-2 macht es daher** mit obligatorischem Smoke-Gate zwischen Cluster I und II.

2. **Größere Cluster — wenn Patterns gekoppelt sind.** F-1's 7 Mikro-Cluster waren Bookkeeping-lastig (jeder Cluster braucht Prompt + Spawn + Report + Merge + Status). F-2 hat 12 Patterns in einem Cluster zusammengefasst, war "gerade noch handhabbar" mit Drei-Sub-Batch-Strategie (Foundation → Critical-UX → State-Lifecycle). Wenn ein Cluster mehr als 3–4 unabhängige Code-Pfade berührt: 3-Cluster-Split lieber.

3. **Cross-Feature-H4 ist hochwertiger Hebel.** Sobald F-1 etabliert war (showAlert/showToast/formatFileSize/resetSaveBtn-Pattern/Drop-Zone-Pattern), waren ~41% der F-2-Findings durch existing-helper-reuse lösbar — XS-Aufwand bei oft Sev 3+. Achte darauf in F-3+ dass Sub-Threads Helper reusen statt neu bauen. **In Implementation-Prompts explizit fordern.**

4. **Pytest reicht nicht.** Tests bleiben grün während UI bricht. Pytest deckt SDK-Boundary + HTTP-Boundary, nicht JS/Browser/UI. **Smoke-Verifikation ist nicht optional**, auch wenn Pytest grün ist.

5. **Sub-Thread-Berichte ehrlich auswerten.** Wenn ein Sub-Thread sagt "Live-Smoke nicht durchgeführt, weil ..." — das ist ein Signal an dich, dass Smoke explizit gepusht werden muss. Nicht abwinken mit "passt schon".

---

## 9. Memory-System (User-Profil + Feedback-Knowledge)

Persistente Memory unter `/home/oliver/.claude/projects/-home-oliver-CODE-CONVERTER/memory/`. Auf neuer Maschine vermutlich anderer Pfad — Oliver muss synchronisieren oder die Inhalte liegen hier inline (Sicherheits-Backup):

### `user_overseer_pattern.md` (User-Memory)
> Für nicht-triviale mehrstufige Arbeiten arbeitet Oliver als Overseer im Hauptthread: ich entwerfe einen self-contained Stage-Prompt, er öffnet eine frische Claude-Code-Session im Repo und paste'n den Prompt rein, der Sub-Thread arbeitet ab und meldet zurück, der Overseer-Thread aktualisiert Status und entwirft die nächste Stage. Stages werden in CLAUDE.md gepflegt.
>
> **How to apply:** Wenn die Aufgabe sich natürlich in unabhängige Stages zerlegen lässt → Overseer-Pattern vorschlagen. Stages mit klaren Acceptance-Bars und Guardrails entwerfen, sodass Sub-Threads ohne Scope-Creep abschließen können. **Nicht** anwenden für One-Shot-Tasks oder kleine Bugfixes.

### `feedback_no_silent_fixes.md` (Feedback-Memory)
> Bugs während eines Refactor als Finding dokumentieren, nicht inline fixen — bewahrt Auditierbarkeit.
>
> **How to apply:** Wenn ein Sub-Thread während Implementierung einen Bug entdeckt, der über Scope hinausgeht: als Notiz im Commit-Body / im Stage-Report aufnehmen, **nicht** im selben Diff fixen.

### `feedback_pragmatic_merge.md` (Feedback-Memory)
> Pragmatische Risiko-Kalibrierung bei Merges: pure-extraction Refactors mergen ohne paranoiden UI-Smoke; "if you skip"-Analyse statt default-Warnung.
>
> **How to apply:** Bei reinen Code-Bewegungen (Refactor ohne Verhaltens-Änderung) keinen großen Smoke-Aufwand triggern. Bei Verhaltens-Änderungen Smoke nicht abwinken (siehe Lesson Learned 4).

### `reference_notion_mcp_restart.md` (Reference-Memory)
> Bei Notion-MCP-Session-Errors auf Mintbox: `docker restart notion-mcp-server` + `/mcp` reconnect.
>
> **Mintbox-spezifisch — auf neuer Maschine vermutlich nicht relevant.**

### Memory-Sync-Empfehlung

Wenn Oliver beide Maschinen nutzt:

```bash
# Von Mintbox:
rsync -av --exclude='._*' \
  /home/oliver/.claude/projects/-home-oliver-CODE-CONVERTER/memory/ \
  user@neue-maschine:~/.claude/projects/-home-oliver-CODE-CONVERTER/memory/
```

(Hinweis: macOS-Sync mit AppleDouble-Files — die `._*`-Files sollten ausgeschlossen werden, sind reine Filesystem-Metadaten ohne Inhalt.)

---

## 10. Repo-Constraints + Quirks (was du wissen musst, ohne den ganzen Code zu lesen)

- **`.gitignore` enthält `.claude/worktrees/`** seit Commit `6505479` — Worktree-Verzeichnisse landen nicht im Repo. Falls auf neuer Maschine `.claude/worktrees/`-Inhalt im `git status` auftaucht, wurde die Regel nicht übernommen → checken.
- **8 lebende Stage-Branches** unter `claude/<adjective-name>` aus den F-1- und F-2-Sub-Threads. Die sind alle ff-gemerged in main, aber die Branches und Worktrees existieren noch. Können bei Gelegenheit aufgeräumt werden:
  ```bash
  git -C /home/oliver/CODE/CONVERTER worktree list   # zeigt alle Worktrees
  # für jeden veralteten Worktree:
  git -C /home/oliver/CODE/CONVERTER worktree remove .claude/worktrees/<name>
  git -C /home/oliver/CODE/CONVERTER branch -D claude/<name>
  ```
  Nicht dringend.
- **Push-Auth war auf Mintbox HTTPS+interactive.** Auf neuer Maschine vermutlich anders — checken (siehe Sektion 7).
- **Compose-Service-Name** ist `markdown-converter`, **Container-Name** ist `markdown-converter-web`. Beide Begriffe bezeichnen dasselbe Web-Container. Worker-Service ist separat (`worker`), läuft `tasks.py` für Podcast-Generierung — für UX-Stages auf File/Audio/Document/Library nicht relevant, kein Restart nötig.
- **Container hat keinen Source-Bind-Mount** — Code-Änderungen brauchen `docker compose up --build` oder `docker cp` zum Verifizieren. Sub-Threads haben das auf Mintbox immer respektiert (kein automatisches Touch der Production-Container). Auf neuer Maschine genauso fortführen.

---

## 11. Sub-Thread Prompt-Vorlagen (häufige Patterns)

Diese Vorlagen sind die etablierten Strukturen aus F-1 und F-2. Du kannst sie adaptieren — Microcopy/Files/Scope ändern sich pro Stage, Skelett bleibt.

### Vorlage A — Inventur-Stage (F-X.1)

```markdown
ROLLE
Du bist UI-Architekt. Deine einzige Aufgabe ist, die Oberfläche eines bestehenden Features vollständig zu kartieren — ohne zu bewerten.

KONTEXT
Repo: CONVERTER (Flask, Docker, localhost:5656). Single-User (Oliver). Diese Aufgabe ist Stufe 1 der UX-Kaskade nach Duan et al. CHI 2024 — siehe CLAUDE.md § UX Review Plan und OVERSEER_HANDOFF.md.

Feature: <feature_name> — Route /<route>. <kurze Aufgaben-Beschreibung>.

Quellen (Pflicht-Read):
- templates/<feature>.html
- static/js/<feature>.js
- app_pkg/<feature>.py
[+ relevante Services]

AUFGABE Schritt A — Statische Analyse
[Element-Tabelle wie in den existierenden Inventur-Files]

AUFGABE Schritt B — Smoke-Pfad-Liste für Oliver-manuelle Verifikation
**Auf dieser Maschine sind Browser-MCP-Tools nicht verfügbar.** Statt selber Live-Walkthrough zu machen, liefere eine konkrete klick-für-klick-Smoke-Pfad-Liste (10–15 Steps), die Oliver später manuell durchgeht. In der Element-Tabelle alle States, die nicht aus dem Code allein herleitbar sind, mit `?` markieren.

OUTPUT
Datei: docs/ui_inventory_<feature>_2026-05.md
[Format wie F-1/F-2 — Tabelle, Zusammenfassung, Bemerkungen]

Branch: vom System angelegter Worktree-Branch. Ein Commit. Title: "F-X.1 / Stufe 1: UI inventory of <feature>".

ABSCHLUSS
Report im Overseer-Thread: Zahlen aus der Zusammenfassung + Auffälligkeiten + die Smoke-Pfad-Liste prominent. Dann stoppen.
```

### Vorlage B — Heuristik-Review (F-X.2)

Siehe F-2.2-Prompt in der Conversation-History — strukturell identisch, anpassen an Feature + Inventur-Datei. Cross-Feature-H4 ab F-2 explizit fordern (audio bricht F-1's Konventionen, F-3 wird F-1+F-2 brechen).

### Vorlage C — Patterns + Microcopy (F-X.3)

Siehe F-2.3-Prompt. Anpassen an Findings-Datei. Konsolidierungs-Hinweis und Cluster-Vorbereitung explizit fordern (sub-thread soll Cluster-I/II-Vorschlag in Output ablegen).

### Vorlage D — Implementation-Cluster (F-X Cluster I/II/...)

Siehe F-2 Cluster I-Prompt. Anpassen an Pattern-Liste. Drei-Sub-Batch-Strategie empfehlen wenn >6 Patterns. Microcopy-Konventionen + Helper-Reuse-Disziplin verbindlich machen.

### Vorlage E — Regression-Diagnose (NEU, für `document_converter`)

Diese Vorlage hat noch keine Präzedenz — ich entwerfe sie hier:

```markdown
ROLLE
Du bist Senior Frontend/Backend-Engineer mit Bisect-Erfahrung. Deine Aufgabe ist, eine Regression in einem Feature zu reproduzieren und zu lokalisieren — keine Reparatur, nur Diagnose.

KONTEXT
Repo: CONVERTER (Flask, Docker, localhost:5656). Single-User (Oliver). F-1 (document_converter) hat eine UX-Kaskade durchlaufen mit 7 Implementierungs-Clustern (A bis Polish-2, alle commits zwischen `153d418` und `37f8420`). Pytest war durchgängig grün, aber Live-Smoke wurde nie durchgeführt.

Symptom: <Olivers Beschreibung des broken Verhaltens — bitte konkret>

Vermutete Kandidaten (Cluster):
- Cluster A (`153d418`): Save-Btn-Lifecycle, Clear-Reset, Auto-Scroll
- Cluster B (`1242e48`): Empty-Submit-Banner, Save-Failure-Banner, showAlert-Helper
- Cluster C (`a96eb93`): Drag-Highlight + Alert-Close + Conversion-Error-Fold
- Cluster D (`e68b6dd`): Backend-Whitelist + Drop-Zone-Hint + accept-Attribut
- Cluster E (`990d1d3`): Drop-Zone-Keyboard + Result-Pre-a11y
- Polish-1 (`ea9db78`): Filename-Format + showToast + DE-Strings
- Polish-2 (`37f8420`): Drop-Zone-Loading-State

AUFGABE
1. Container starten falls nicht laufend: `docker compose up -d --build markdown-converter`.
2. Symptom reproduzieren — beschreiben was tatsächlich passiert (Network-Tab? Console-Errors? Visueller State?). Screenshot wenn via Tooling möglich, sonst beschreibend.
3. Bisect via `git checkout`: zwischen den Cluster-Commits halbieren bis der erste broken Commit gefunden ist.
   ```bash
   git bisect start
   git bisect bad <aktueller HEAD>
   git bisect good <commit vor F-1, z.B. 19cd936>
   git bisect run <test-script>   # oder manuell zwischen good/bad markieren
   ```
4. Wenn der Übeltäter-Commit gefunden: den Diff anschauen und die konkrete Zeile(n) identifizieren die brechen.
5. **KEINE Reparatur in dieser Stage.** Nur Diagnose dokumentieren.

OUTPUT
Datei: `docs/regression_document_converter_2026-05.md`

```markdown
# Regression-Diagnose: document_converter (2026-05-XX)

## Symptom (von Oliver gemeldet)
<Beschreibung>

## Reproduktion
<klick-für-klick wie der Bug ausgelöst wird>

## Bisect-Ergebnis
- Letzter funktionierender Commit: `<sha>` (`<title>`)
- Erster broken Commit: `<sha>` (`<title>`)
- Geänderte Datei(en) im broken Commit: ...

## Root-Cause
<welche Code-Zeile(n) brechen, warum>

## Fix-Vorschlag (für Hot-Fix-Stage, nicht hier umgesetzt)
<konkreter Vorschlag>
```

Branch: vom System angelegter Worktree-Branch. Ein Commit. Title: "F-1 regression diagnosis: document_converter".

ABSCHLUSS
Report im Overseer-Thread: Symptom + Übeltäter-Commit + Root-Cause + Fix-Vorschlag in 3–5 Sätzen. Overseer drafted Hot-Fix-Stage basierend darauf.
```

---

## 12. Quick-Reference: Helper-API + Pattern-Konventionen aus F-1

Diese sind App-Standard ab F-1 — alle künftigen Features sollen sie reusen.

### `static/js/_utils.js` — geteilte Helper

```javascript
// Alert-Banner (XSS-safe via textContent)
window.showAlert(containerEl, level, message, options = {})
// level: 'danger' | 'success' | 'warning' | 'info'
// options.closable: default true (× Button)
// options.autoDismissMs: default 6000 für non-danger, null für danger

// Singleton-Toast (unten-rechts, success-Tint default)
window.showToast(message, options = {})
// options.level: default 'success'
// options.durationMs: default 2500

// Filename-Format DE-Locale
window.formatFileSize(bytes)
// → "222 B" / "4,6 KB" / "1,2 MB"

// JSON-Response sicher parsen (login-Redirect erkennen)
window.safeJSON(response)
```

### Etablierte Komponenten + State-Klassen

```css
/* in static/css/style.css */
.c-btn / .c-btn--primary / .c-btn--secondary
.c-input
.c-card
.c-alert / .c-alert--danger/--success/--warning/--info
  .c-alert__close (× Button)

.c-drop-zone
  .drop-zone-active (während Dragover)
  .c-drop-zone--invalid (Submit-Fehler)
  .c-drop-zone--warning (unsupported drag-type)
  .c-drop-zone--loading (während Conversion)
  .c-drop-zone__overlay (Drag-Hint)
  .c-drop-zone__loading (Spinner-Overlay)

.toast-notification

.save-library-btn
  .saved (success-State)
```

### `:focus-visible`-Vokabular

Alle interaktiven Elemente bekommen `:focus-visible`-Outline analog `c-btn`. Nicht den Browser-Default — passt nicht zur Neomorphism-Optik.

### Backend-Pattern: `ACCEPTED_EXTENSIONS`-Konstante

In `app_pkg/<feature>.py`:
```python
ACCEPTED_EXTENSIONS = ('pdf', 'docx', ...)
```
Single-Source-of-Truth. Template rendert `accept`-Attribut + `window.PageData.acceptedExtensions` aus dem Tuple. JS prüft Drop-Files dagegen. Backend liefert 400 + DE-JSON-Body bei nicht-erlaubter Extension. Beispiel: `app_pkg/documents.py`.

---

## 13. Bekannte offene Backlog-Items (außer F-2 Cluster II + F-3+)

- **`document_converter` Regression** (NEU — siehe Schritt 1 in Sektion 6)
- **Architektur-Quirk:** `getUserMedia` wird in `audio_converter.js` innerhalb `socket.onopen` aufgerufen — Permission-Prompt erscheint erst nach erfolgreichem WS-Handshake. Vom F-2 Cluster I Sub-Thread als Out-of-Scope respektiert. Wenn Oliver einen "Architektur-Review"-Stage will, das ist ein Kandidat.
- **Englische Strings in `library.js` und `library_detail.js`** — vom F-1 Polish-1 Sub-Thread benannt aber nicht gefixt (waren out-of-scope: Polish-1 betraf nur `document_converter`). Werden in F-N für `library`/`library_detail` mit-adressiert.
- **F-006 (Cleanup-Plan-Finding) noch offen** für `markdown_converter` und `audio_converter` Endpoints — Backend-Whitelist + Validierung. Wird in deren UX-Stages mit-adressiert wenn passend.
- **CVE-Upgrade-Stage** (aus Cleanup-Plan) — 5 Pakete, 8 Advisories. Niedrigste-Risiko-Reihenfolge: Pygments → requests → Flask → pdfminer.six → unstructured. Oliver hat das bisher nicht gestartet, ist eigener Stage außerhalb UX-Backlog.
- **8 lebende Stage-Branches + Worktrees** unter `.claude/worktrees/` — können aufgeräumt werden bei Gelegenheit (siehe Sektion 10).

---

## 14. Kontakt / Eskalation

- **Repository:** https://github.com/TheReallyRealComedian/CONVERTER
- **Oliver's Email:** ogluth@gmail.com (in Git-Config konfiguriert)
- **Live-App:** läuft auf der Maschine wo Docker-Compose läuft, immer auf Port 5656
- **Bei Unklarheiten:** frag Oliver. Du bist nicht allein — der Overseer-Pattern lebt von Olivers Entscheidungen an Verzweigungen.

---

## 15. Letzte Worte

Du übernimmst eine **eingespielte Maschinerie** (3 Cascade-Stages × 2 Features + 7+1 Implementation-Cluster sind durchgelaufen). Sub-Threads sind diszipliniert mit Scope umgegangen, Reports waren ehrlich, Bookkeeping ist konsistent. Die Methodik trägt — du musst sie nicht neu erfinden, nur kontinuieren.

**Wenn du etwas zum ersten Mal entscheidest** (z.B. "wie viel Live-Smoke ist genug bei einem F-3-Feature mit 5 Findings?"), schau in die F-1- und F-2-Geschichte (CLAUDE.md + Git-Log). Das Muster ist wahrscheinlich da.

**Wenn etwas wirklich neu ist** (z.B. ein Sub-Thread-Bericht zeigt eine Architektur-Frage die niemand vorher hatte): drafte deine Empfehlung mit dem Trade-off, gib Oliver die Entscheidung. Nicht selbst-entscheiden bei Strategie-Fragen.

Viel Erfolg.

---

*Erstellt von der Mintbox-Overseer-Session am 2026-05-03 vor Maschinen-Wechsel. Erste Aufgabe: Sektion 6 Schritt 1.*
