# Sprint LEARN-UP — Lernkarten/Review auf Anki-Niveau (L/XL, 6 Phasen)

> **Executor-Doc.** Ein großer Sprint (Oli-Wahl), aber **in Phasen geschnitten** — nach **jeder** Phase **Stop + Bericht**, auf Sign-off warten. Pre-Flight: `pytest tests/` grün (Baseline aus STATUS, aktuell **663**). Arbeitsverzeichnis Mac `/Users/olivergluth/CODE/CONVERTER` = Source-of-Truth. Jede Phase selbst committen + pushen (`feat(LEARN-UP): … (Pn)`). Web-`/review` + Backend (dort lernt Oli; iOS-Lern-Port ist v2). Vollständige Vorschläge/Recherche: [docs/learning_upgrade_proposals.md](docs/learning_upgrade_proposals.md).
>
> **Ziel:** die vier Oli-Lücken schließen — (1) Tags-Clutter, (2) Study-Set-Komposition, (3) Reihenfolge wirkt wie Eintrittsdatum, (4) keine Daily-Limits/Stats. **Hebel:** FSRS' **Retrievability (R)** + ein **Desired-Retention**-Regler geben Ankis gefühlte Raffinesse bei ~20 % der Konfig.
>
> **Oli-Entscheidungen (gelockt):** Tags **ganz raus** aus dem Lern-Picker (bleiben in `/tags`-Manager + Organisieren/Agent-Taggen). Limit-Defaults **10 new / 200 reviews** (an P3 justierbar). Stats = die 4er-Menge (+ optional Workload-Sim, an P4 entscheidbar).

## Master-gegroundete Fakten (Ist-Zustand — darauf aufsetzen)

- **Reihenfolge:** `GET /api/review-state` ([app_pkg/cards.py:483-488](app_pkg/cards.py)) → `.join(Card.review).filter(Review.due<=now).order_by(Review.due.asc()).all()`. **Kein `.limit()`.** Neue Karten `due=now` bei `POST /api/cards` ([cards.py:330](app_pkg/cards.py)), nie umgeplant bis 1. Rating → Rückstand = Erstellungsreihenfolge. **Docstring cards.py:454-455 „newest-due first" ist falsch** (Code `.asc()`) → mitfixen.
- **Scope:** `?tag=<id>` (subtree via `Tag.subtree_ids`) und `?collection=<id>` (single), beide chainen auf dasselbe `base` → **AND**. `_parse_owned` ([cards.py:111-123](app_pkg/cards.py)) parst **eine** id, foreign→404.
- **Scheduler:** `services/scheduler/` — `get_scheduler()` (`FSRSScheduler` default, `SCHEDULER_ENGINE`/`FSRS_DESIRED_RETENTION` env-only, **nicht** in config.py). `FSRSScheduler` ([fsrs_scheduler.py:30-75](services/scheduler/fsrs_scheduler.py)) wrappt py-fsrs (`fsrs==6.3.1`), `enable_fuzzing=False`, `_reconstruct` baut `State.Review` aus stability. **Exponiert R NICHT**; `apply_rating` one-card-one-step.
- **Modelle:** `Card` ([models.py:281](models.py)) — kein `position`, `state ∈ {ok,wackelt}` (kein new/learning/suspended). „Neu" = `Review.stability IS NULL`. `Review` ([models.py:351](models.py)) — `due` **indexed**, `stability`/`difficulty` nullable, `reps`/`lapses`, `rating_history` (JSON-Text). `Collection` ([models.py:407](models.py)) flach, M2M `card_collections`, case-erhaltend, **wenige**. `Tag` ([models.py:162](models.py)) hierarchisch (`parent_id`), geteiltes Vokabular, **viele** — die Clutter-Achse.
- **Review-UI:** `/review` ([templates/review.html](templates/review.html) + [static/js/review.js](static/js/review.js)) — ein `<select id="review-scope-select">` mit `Alles fällig` + optgroup Tags (**kompletter Forest als eingerückte options**, `review.js:317-333`, ungefiltert → Clutter) + optgroup Sammlungen. `loadScopeOptions` fetcht `/api/tags` + `/api/collections`. Tag-Optgroup + Card-Tag-Chips (`review.js:96-102`) sind **display-only, nicht load-bearing** — entfernbar ohne Collection-Study zu brechen.
- **Kein** Daily-Limit, **keine** Stats/Forecast/R-Readout, `desired_retention` nie an UI.
- **Migration:** neue Spalte → `_run_pending_migrations` (get_columns-Check + `ALTER TABLE ADD COLUMN`, idempotent). Memory [[reference_inline_sqlite_migration]]. Prod-DB vor Deploy sichern.

## Phase 0 — Mechanik-Confirm (kurz, kein Feature-Code)

Drei Mechanik-Punkte am Code/Lib verifizieren + Empfehlung bestätigen:
1. **R aus py-fsrs 6.3.1:** wie kommt man an die Retrievability einer rekonstruierten Karte? (`Scheduler.get_card_retrievability(card, now)` o.ä. — an der installierten Lib verifizieren; sonst die R-Formel `R=(1+FACTOR·t/S)^DECAY` mit den Lib-Konstanten). → ein dünner `FSRSScheduler.retrievability(state, now) -> float` Zusatz.
2. **Settings-Store:** Empfehlung **`User.settings_json` TEXT** (ein JSON-Blob, `_run_pending_migrations`-ALTER) — extensibel für ordering_mode (P1) + Limits (P3) + surfaced desired_retention (P4) **ohne** weitere Migration. (Alternative: typisierte Spalten — verworfen, weil wir mehrere Keys ergänzen.) Ein `GET/PUT /api/learn/settings` (Session) liest/schreibt den Blob mit validierten Defaults.
3. **Forecast:** Empfehlung **Bucketing** — `Review.due` der nächsten N Tage per group-by-Tag zählen (kein Forward-Sim); überfälliger Rückstand als eigener Bucket. (Ankis „wenn du nicht lernst"-Graustufe ist Kür.)

**Stop + Bericht: die 3 Bestätigungen + gewählte py-fsrs-R-Mechanik. Sign-off vor P1.**

## Phase 1 — Reihenfolge + Settings-Scaffold (Gap 3)

1. **`User.settings_json`** (TEXT, nullable) via `_run_pending_migrations` + `GET/PUT /api/learn/settings` (Session-authed, validiert; Keys diese Phase: `ordering_mode ∈ {smart,random}` default `smart`). Helper `get_user_settings(user)` mit Defaults.
2. **`FSRSScheduler.retrievability(state, now)`** (aus P0) — R für eine Karte mit stability≠NULL; für stability=NULL (neu) gibt es kein R.
3. **`review-state`-Reihenfolge umbauen:** die fällige Menge in **Review** (stability≠NULL) und **Neu** (stability=NULL) trennen. `smart`-Modus (default): Review nach **R aufsteigend** (die wackeligsten zuerst) + **Zufalls-Tiebreak** (sekundärer random key → nie versteckte Datums-Ordnung); Neu **shuffeln**. Dann **interleaven** (Neu gleichmäßig unter die Reviews streuen, **nicht** vorne stapeln). `random`-Modus: voller Shuffle des ganzen fälligen Sets. Docstring cards.py:454-455 fixen.
4. **UI:** kleiner Modus-Toggle im Review (smart/random), persistiert über `PUT /api/learn/settings`. Minimal.
5. **Tests:** R-Sortierung (Mock-Reviews mit gestellten stability/elapsed → erwartete R-Ordnung), Zufalls-Tiebreak bricht Datums-Ordnung (zwei Karten gleicher R, gemischte created_at → nicht deterministisch nach created_at), Neu-Karten interleaved+geshuffelt statt front-loaded, `random`-Modus = voller Shuffle, Settings-Roundtrip (`PUT`→`GET`, Default `smart`, Validierung ungültiger Werte). Determinismus in Tests via geseedetem RNG.
6. `pytest` grün. **Schema-Touch** (`settings_json`) — Auto-Migrate verifizieren.

**Stop + Bericht.**

## Phase 2 — Study-Set-Picker + Tags raus (Gaps 1 + 2)

1. **Backend Multi-Collection-Union:** `review-state` akzeptiert **mehrere** Collection-Ids (z.B. `?collection=1,2,3` oder wiederholt) → **OR/Union** (`Card.collections.any(Collection.id.in_(ids))`, alle owned). `Alles fällig` (kein Param) bleibt. **Tag-Scope-Param bleibt im Backend** (nicht load-bearing, nur aus der UI genommen) — oder entfernen, falls sauberer; **im Bericht begründen**.
2. **Per-Collection-Fällig-Zähler:** ein Endpoint/Feld, das je Collection die Zahl **fälliger** Karten liefert (ein group-by, **nicht** N Queries) — für die Badges.
3. **UI-Launcher:** das `<select>` ersetzen durch eine **Checkbox-Liste der Collections** mit Fällig-Badges; mehrere ankreuzen → Union lernen; „Alles fällig" als Option. **Tag-Optgroup RAUS** (Oli-Entscheidung) + die display-only Card-Tag-Chips (`review.js:96-102`) raus aus dem Lern-View. `/api/tags`-Fetch im Review entfällt.
4. **Bewusst NICHT anfassen:** `/tags`-Manager, Tag-Modell, Agent-Tag-Schreibpfade, Highlight/Conversion-Tags — Tags leben überall sonst weiter.
5. **Tests:** Multi-Collection-Union (2 Collections → Vereinigung der fälligen, dedupliziert), owned-Scope (fremde Collection-Id → 404/ignoriert), Per-Collection-Zähler korrekt, `Alles fällig` unverändert; UI-seitig Live-Smoke (pytest fängt kein Template — s. CLAUDE.md Test-Suite-Limit).
6. `pytest` grün. **Live-Smoke** (dark+light): Tag-Clutter weg, Multi-Collection-Auswahl lernt die Union, Badges stimmen.

**Stop + Bericht.**

## Phase 3 — Daily-Limits (Gap 4a)

1. **Settings erweitern** (`settings_json`, **keine** neue Migration): `daily_new_limit` (default **10**), `daily_review_limit` (default **200**). `PUT /api/learn/settings` + UI-Felder.
2. **Cap-Logik in `review-state`** (nach Ordering + Scope): Pools getrennt — **Reviews** (stability≠NULL, R-sortiert) auf `daily_review_limit` cappen; **Neu** (stability=NULL, geshuffelt) auf `daily_new_limit` cappen **und nur einstreuen, solange der Review-Cap nicht erschöpft ist** (neue respektieren den Review-Cap — sane default). „Erst shuffeln/sortieren, dann cappen" = **zufällige N aus dem Set** (schließt den Bogen zu Gap 3). ⚠️ **Tages-Zählung:** der Cap gilt pro Tag — heute schon gemachte Reviews/Neue müssen gegen den Cap zählen (aus `rating_history`/`last_reviewed` des heutigen Tages ableiten), sonst cappt es die *angezeigte* Menge, nicht die *tägliche*. Im Bericht die Zähl-Semantik nennen.
3. **UI:** Limits setzbar; Launcher zeigt „X neu verfügbar, Y reviews fällig" (nach Cap).
4. **Tests:** Review-Cap greift (mehr fällig als Cap → nur Cap-viele), New-Cap greift, Neu respektiert Review-Cap (voller Review-Tag → 0 neue), Tages-Zählung (heute schon N gemacht → nur Rest), Defaults 10/200, Settings-Validierung (negativ/nicht-int → abgelehnt).
5. `pytest` grün.

**Stop + Bericht.**

## Phase 4 — Statistik (Gap 4b)

1. **`GET /api/learn/stats`** (Session) rechnet aus Card/Review/`rating_history`:
   - **Heute:** fällig-Zahl, neu-verfügbar-Zahl (nach Cap), grobe Zeitschätzung optional.
   - **Future-Due-Forecast:** `Review.due` der nächsten ~4 Wochen per Tag gebucketet; **überfälliger Rückstand als eigener Marker** (nicht verstecken).
   - **Reifegrad-Zähler:** neu (stability NULL) / lernend / jung (interval <21 d) / reif (≥21 d) — interval aus `due - last_reviewed` bzw. stability ableiten (im Bericht die Klassifikation nennen).
   - **Ist- vs. Ziel-Retention:** True-Retention aus `rating_history` (erste Review pro Karte pro Tag; `again`=fail, sonst pass) über ein Fenster, gegen `desired_retention` (aus Settings/env). Die FSRS-Rückkopplung.
2. **`desired_retention` surface-n** (heute env-only) — in Settings sichtbar/optional editierbar.
3. **Stats-View** (`/review`-Sektion oder eigene Seite) — die 4 Zahlen/der Forecast-Chart, deutsch, schlicht.
4. **Optional (nur falls Oli am P4-Bericht ja sagt):** Workload-Simulator (new/Tag + Retention → Steady-State-reviews/Tag).
5. **Tests:** Forecast-Bucketing (gestellte due-Daten → korrekte Tages-Buckets + Rückstand-Bucket), Reifegrad-Klassifikation (Grenzfall 21 d), True-Retention-Rechnung (bekannte rating_history → erwartete Rate), Retention-Fenster.
6. `pytest` grün.

**Stop + Bericht.**

## Phase 5 — Live-Verify + Wrap

1. **Live-Smoke** (nach Deploy, oder lokal gegen echte Daten): kompletter Review-Flow — Multi-Collection-Auswahl, Reihenfolge fühlt sich nicht mehr nach Datum an, Limits greifen, Stats plausibel. Ordering-Modi durchschalten.
2. **Wrap:** BACKLOG (LEARN-UP ☑ + die 4 geschlossenen Gaps) · STATUS (pytest-Zahl) · **CLAUDE.md** (Learning-Abschnitt: R-Ordering, Multi-Collection-Study, Tags-raus-aus-Review, Daily-Limits via `settings_json`, Stats-Endpoint; `desired_retention` jetzt surfaced) · **Memory** bei übertragbarer Lehre (Kandidat: „FSRS-R exponieren für Ordering + Forecast/Retention aus rating_history; ein `settings_json`-Blob statt Spalten-Wildwuchs"). **Bullet-Guard** `grep -nE '(- \*\*.*){2,}' BACKLOG.md` vor Commit.
3. **Deploy-Notiz:** **Prod-DB VOR Deploy sichern** (neue Spalte `settings_json` via `_run_pending_migrations`), dann Mintbox `git pull` + `up -d --build`. Kein Dep erwartet (py-fsrs schon da).

**Stop + Schluss-Bericht.**

## Bewusst NICHT

- **Keine** Query-DSL/Filtered-Decks/Custom-Study-Presets (Anki-Overkill) — der Multi-Collection-Picker ersetzt sie.
- **Kein** Anfassen des Tag-Modells / `/tags`-Managers / Agent-Tag-Schreibpfade — Tags fliegen **nur** aus dem Lern-Picker.
- **Kein** ease-/interval-basiertes Ordering (SM-2-Konzepte, FSRS verwirft sie), keine 8 Anki-Sortier-Optionen — nur `smart` (R) + `random`.
- **Kein** per-Collection/per-Deck-Limit-Preset (Multi-Deck-Maschinerie) — zwei globale Zahlen.
- **Kein** Umbau des `apply_rating`-Ratings-Pfads oder des Scheduler-Vertrags (nur additiver `retrievability`-Read).
- **Kein** iOS-Lern-Port (v2; nutzt später dieselbe neue API).

## Akzeptanz

- [ ] **P0**: 3 Mechanik-Bestätigungen (py-fsrs-R, settings_json, Forecast-Bucketing) + Sign-off.
- [ ] **P1**: R-Sortierung + Neu-Shuffle/Interleave + random-Modus; `settings_json`+Settings-API; Docstring-Fix; Datums-Ordnung strukturell weg; pytest grün.
- [ ] **P2**: Multi-Collection-Union + Fällig-Badges-Launcher; Tags **raus** aus dem Lern-Picker (Manager unberührt); Live-Smoke.
- [ ] **P3**: Daily-Limits (10/200, editierbar), neue respektieren Review-Cap, **tages-korrekte** Zählung; pytest grün.
- [ ] **P4**: Stats-Endpoint + View (Heute · Forecast+Rückstand · Reifegrad · Ist-vs-Ziel-Retention); `desired_retention` surfaced; pytest grün.
- [ ] **P5**: Live-Verify kompletter Flow; Docs/CLAUDE/Memory/Wrap + Bullet-Guard; Deploy-Notiz (DB-Backup + `settings_json`).
