# Sprint HYG3 — Hygiene-Block (pip-timeout · pyc-untrack · live-dev-mount · dual-reset)

**Datum**: 2026-06-01

**Ziel**: Vier disjunkte, aufgestaute Hygiene-Items in einem Aufwasch — drei davon aus R2-B/READER-FIX-B aufgefallen, eines (pip-timeout) seit MAC1 offen. Kein Feature, keine Architektur — reine Aufräum- und Dev-Experience-Arbeit.

**Vorbedingung**: HEAD `e15c2e6`, lokal+remote synchron. Pytest **151/151 grün**. READER-FIX-B ☑ done und im echten Vordergrund-Browser bestätigt (Markieren über Block-Grenzen funktioniert). Mac-Dev-Stack über `docker-compose.override.yml`.

**Out-of-scope**:
- **R2-C Lifecycle** — Folge-Sprint.
- **Voll-Template-Auto-Reload** (Templates ohne `restart` live) — bräuchte einen App-Config-Touch (`TEMPLATES_AUTO_RELOAD` aus Env lesen in `create_app`); nicht in einem Hygiene-Sprint. Hier reicht `docker compose restart` für Template-Iteration.
- **MAC1-FOLLOWUP-B Image-Slim** (NVIDIA-CUDA-Wheels) — bleibt eigenes M-Item.
- Alles andere im BACKLOG.

---

## Item-Übersicht (Master-grounded 2026-06-01)

| # | Item | Was genau | Commit? | Größe |
|---|------|-----------|---------|-------|
| 1 | pip-timeout | `Dockerfile:23` `RUN pip install --no-cache-dir -r requirements.txt` → `… --no-cache-dir --timeout=600 --retries=5 -r …` | ja (Dockerfile tracked) | XS |
| 2 | pyc-untrack | 7 Altlast-`.pyc` untracken: `git rm -r --cached services/pdf_extraction/__pycache__/`. **`.gitignore` hat `__pycache__/`+`*.pyc` bereits** — kein gitignore-Touch. | ja | XS |
| 3 | live-dev-mount | `static/` (+`templates/`) in `docker-compose.override.yml` mounten. **Override ist gitignored → lokal-only, KEIN Commit.** | nein | XS |
| 4 | dual-reset | `library.html:58` Microcopy: Chip-Row-Reset cleart nur den Tag, Label impliziert fälschlich „alle Filter". Umbenennen. | ja | XS |

---

## Phase 0 — entfällt

Vier klar geschnittene, Master-grounded XS-Items, keine offene Mechanik-Wahl. **Direkt Phase 1.**

---

## Phase 1 — Infra/Repo-Hygiene + Live-Dev-Mount

Pre-Flight: `pytest tests/` grün (151).

### Item 1 — pip-timeout (`Dockerfile:23`)
`RUN pip install --no-cache-dir -r requirements.txt` → `RUN pip install --no-cache-dir --timeout=600 --retries=5 -r requirements.txt`. Reine Resilienz gegen den MAC1-ReadTimeout am 145-MB-NVIDIA-Wheel. **Inspektions-verifizierbar** (Standard-pip-Flags) — ein voller Rebuild nur zum Prüfen ist unnötig teuer; Olivers nächstes reguläres `docker compose up --build` exerziert es ohnehin.

### Item 2 — pyc-untrack
`git rm -r --cached services/pdf_extraction/__pycache__/` (nur Index, Files bleiben auf Disk). Danach: `git ls-files | grep -E "pyc|__pycache__"` muss **leer** sein, `git status` zeigt die `.pyc` nicht mehr (greift jetzt die bestehende `.gitignore`-Regel). Container unberührt (eigenes FS, plus `PYTHONDONTWRITEBYTECODE 1` im Dockerfile → schreibt eh keine).

### Item 3 — live-dev-mount (lokal-only, KEIN Commit)
In `docker-compose.override.yml` unter `services.markdown-converter` einen `volumes`-Block ergänzen:
```yaml
    volumes:
      - ./static:/app/static
      - ./templates:/app/templates
```
App liegt unter `/app` (Dockerfile `WORKDIR /app` + `COPY . .`). Effekt: `static/`-Edits (JS/CSS) sind nach Browser-Reload live (gunicorn serviert static frisch von Disk); `templates/`-Edits nach `docker compose restart markdown-converter` (Sekunden, kein `--build`) — gunicorn läuft ohne Auto-Reload, cached Jinja-Templates bis Worker-Restart. **Das löst genau den READER-FIX-B-`docker cp`-Schmerz.** Da der Override gitignored ist, wird hier **nichts committet** — Verifikation per Verhalten.

### Quality-Gates + Verify Phase 1
- `pytest tests/` grün (151 — keiner dieser Touches berührt getesteten Code).
- `git status`: keine getrackten `.pyc` mehr; `docker-compose.override.yml` weiterhin untracked.
- **Mount-Verifikation**: Stack neu starten (`docker compose up -d` reicht — Override-Volumes greifen beim Recreate), dann einen harmlosen sichtbaren String in `static/js/library.js` o.ä. ändern → Browser-Reload zeigt die Änderung **ohne** Rebuild → wieder zurückändern. Damit ist der Mount belegt und Phase 2 kann ohne `docker cp` smoken.
- **Commit** (nur Item 1+2, plain-prose, mehrere `-m`):
  `git commit -m "HYG3-infra: pip-timeout-Resilienz plus getrackte pyc-Altlast entfernt" -m "Dockerfile pip install mit --timeout=600 --retries=5 (MAC1-FOLLOWUP-A), services/pdf_extraction/__pycache__ aus dem Index entfernt (gitignore-Regel bestand schon)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`
  Push direkt. Bei Push-Auth-Block → an Master melden.

**STOP — Bericht an Master (inkl. Mount-Verifikation). Nicht in Phase 2 bis Sign-off.**

---

## Phase 2 — Dual-Reset-Polish (`templates/library.html`)

Pre-Flight: `pytest tests/` grün.

**Diagnose** (Master-grounded): Auf der Library-List existieren zwei „Filter zurücksetzen"-Affordances mit **gleichem Label, unterschiedlichem Scope**:
- **Chip-Row** (`library.html:57-58`): `href` = `pagination_args(1, …, '')` → cleart **nur den Tag**, behält type/search/favorites/sort/per_page. Label: „× Filter zurücksetzen" — **impliziert fälschlich „alle Filter".**
- **Empty-State** (`library.html:144`): `href` = `url_for('library')` → cleart **alles**. Label: „Filter zurücksetzen" — korrekt.

Auf einem 0-Treffer-Screen mit aktivem Tag erscheinen beide nebeneinander → verwirrend (genau der Befund aus dem R2-B-Smoke).

**Fix — reiner Microcopy/Label-Change, KEIN Verhaltens-Change:**
- `library.html:58` Chip-Row-Reset auf den echten Scope umbenennen, z.B. **„× Tag-Filter aufheben"** (oder „× Tag entfernen") — macht klar, dass nur der Tag fällt.
- `library.html:144` Empty-State zur Eindeutigkeit auf **„Alle Filter zurücksetzen"** schärfen.
- DE-Microcopy (Buttons ≤ 3 Wörter wo möglich, keine Emojis). `href`/Logik unangetastet.

### Quality-Gates + Verify Phase 2
- `pytest tests/` grün (151).
- **Live-Smoke über den neuen Mount** (Phase 1): `library.html` editieren → `docker compose restart markdown-converter` → `localhost:5656/library`:
  - Tag aktiv (`?tag=…`): Chip-Row zeigt jetzt „× Tag-Filter aufheben"; Klick entfernt **nur** den Tag, übrige Filter bleiben.
  - 0-Treffer mit Tag + z.B. Typ-Filter: Empty-State „Alle Filter zurücksetzen" cleart alles; die zwei Affordances sind nicht mehr verwechselbar.
- **Commit** (plain-prose): `HYG3-dualreset: Chip-Row-Reset-Label auf echten Scope (nur Tag), Empty-State auf "Alle Filter zuruecksetzen"`. Push direkt.

**STOP — Bericht an Master. Nicht in Phase 3 bis Sign-off.**

---

## Phase 3 — Verify + Wrap-up

1. `pytest tests/` final grün (151).
2. **STATUS.md**: HYG3 als „Aktueller Sprint"-Block mit beiden Commit-Hashes; READER-FIX-B → „Vorgänger".
3. **BACKLOG.md**:
   - HYG3 ☑ done 2026-06-01 im „Erledigt"-Block.
   - **MAC1-FOLLOWUP-A pip-timeout** aus P1 entfernen (erledigt). MAC1-FOLLOWUP-B (Image-Slim) bleibt.
   - Drei P3-Items entfernen: `.pyc`-Hygiene, R2-B-Dual-Reset-Polish, `static/`-Volume-Mount.
   - **Hinweis beim static-mount-Item**: als „done (lokal, im gitignored Override — nicht im Repo)" vermerken, damit klar ist, warum kein Commit existiert.
4. **Memory**: kein neuer Eintrag erwartet (alles triviale/bekannte Patterns). Nur falls etwas Unerwartetes auftrat.
5. Push bestätigen (alle HYG3-Commits auf `origin/main`).

**STOP — Schluss-Bericht an Master.**

---

## Stop-Regel

Nach **jeder** Phase Bericht an Master, nicht weiter bis Sign-off. Master = Dispatch, Sub-Session = Execute. Bei „mach jetzt einfach"/Frust: einmal nachfragen, dann der Antwort folgen.

---

## Größe

**S** — vier disjunkte XS-Items, kein Schema-/Backend-/getesteter-Code-Touch (pytest bleibt 151), zwei Commits + ein lokal-only Override-Edit. Risikoarm: Item 1 fügt nur pip-Resilienz hinzu, Item 2 untrackt regenerierbare Caches, Item 3 ist lokal, Item 4 ist reine Microcopy ohne Logik-Change.

---

## BACKLOG- und STATUS-Updates nach Abschluss

- ✓ Sprint HYG3 durch (2026-06-01), zwei Commit-Hashes + ein lokal-only Override-Edit.
- MAC1-FOLLOWUP-A (P1) + drei P3-Items (.pyc, Dual-Reset, static-mount) entfernt.
- 📋 evtl. Follow-up: Voll-Template-Auto-Reload via `TEMPLATES_AUTO_RELOAD`-Env (P3, XS, App-Config-Touch) — falls das `restart` als zu umständlich empfunden wird.
- STATUS.md / BACKLOG.md wie Phase 3.
