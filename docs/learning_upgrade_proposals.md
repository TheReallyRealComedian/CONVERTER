# Learning/SR-Upgrade — Vorschläge (Anki-Niveau)

**Ziel:** das Lernkarten-/Review-Feature auf „Anki-Niveau" heben, auf Basis der vier Oli-Lücken (2026-07-13). Zwei Recherchen: (1) Ist-Zustand-Code-Map, (2) Anki-/FSRS-Modell.

**Roter Faden (der Hebel):** FSRS liefert pro Karte eine **Retrievability (R)** + einen einzelnen **Desired-Retention**-Regler. Damit erreicht man Ankis *gefühlte* Raffinesse (nicht-monotone Reihenfolge, selbst-kalibrierendes Scheduling, Workload-Planung) bei ~20 % von Ankis Konfigurations-Wust — genau richtig für Single-User. **Scope:** Web-`/review` + Backend (dort lernt Oli). iOS-Lern-Port ist v2 und würde dieselbe neue API wiederverwenden.

---

## Ist-Zustand (gegroundet — die tragenden Fakten)

- **Reihenfolge = deterministisch.** `GET /api/review-state` sortiert `ORDER BY Review.due ASC` ([app_pkg/cards.py:486](app_pkg/cards.py)). Neue Karten kriegen `due = now` bei `POST /api/cards` ([cards.py:330](app_pkg/cards.py)) und werden bis zur ersten Bewertung nie umgeplant → **Rückstand ungelernter Karten = Erstellungsreihenfolge**. (Irreführender Kommentar cards.py:454-455 „newest-due first" — Code `.asc()` ist autoritativ.)
- **Multi-Collection: nein.** Nur **ein** `?collection=<id>` (Single-Select-UI). `?tag=` + `?collection=` = **AND**. Kein Union-/Mehrfach-Pfad.
- **Daily-Limit: keins.** Due-Query läuft `.all()` ohne `.limit()` ([cards.py:483-488](app_pkg/cards.py)). Kein new/review-Cap in Code/Env/UI.
- **Statistik: keine.** Nur `due_count`/`total_count`. FSRS berechnet R, **exponiert es aber nie**; `desired_retention` ist env-only, nie an Template/Endpoint gereicht.
- **Tags-Clutter (Quelle bestätigt):** der Review-Scope ist **ein** `<select>` ([review.html:15-22](templates/review.html)), das den **kompletten Tag-Forest** als eingerückte `<option>`s rendert ([review.js:317-333](static/js/review.js)) — **ungefiltert** (alle Tags, auch ohne Karten; Tags sind geteiltes Vokabular über Highlights+Conversions+Cards, auto-gemintet → sprawlen). Das ist der Clutter.
- **„Lernpfad" = `Collection`** (kuratiert, flach, wenige, case-erhaltende Namen — [models.py:407](models.py)). **Tags = die sprawlende geteilte Achse** (lowercased, auto-gemintet, viele). Mapping bestätigt.
- **Tags sind aus der Review-UI entfernbar, ohne Collection-Study zu brechen** — unabhängige Code-Pfade; einzige geteilte Abhängigkeit ist das eine `<select>`-Widget.
- **Card-Modell:** kein `position`, kein new/learning/review/suspended-State — nur `state ∈ {ok, wackelt}`. „Neu" = `Review.stability IS NULL`. FSRS via py-fsrs, `enable_fuzzing=False` (deterministisch), `apply_rating` ist one-card-one-step; kein Batch/Forecast/R-Readout.

---

## Vorschlag 1 — Reihenfolge: R-Sortierung + Shuffle statt Eintrittsdatum (Gap 3) · **S/M · zuerst**

- **Ziel:** **Review-Karten** (stability≠NULL) nach **Retrievability aufsteigend** (die wackeligsten zuerst — lernpsychologisch optimal, FSRS-native Form von „relative overdueness") mit **Zufalls-Tiebreak**; **neue Karten** (stability=NULL) **shuffeln + einstreuen** (interleave) statt vorne nach Datum stapeln. Optional ein **„Rein zufällig"-Modus** (voller Shuffle des Sets) — deckt dein „mix it up" wörtlich ab.
- **Warum das strukturell reicht:** R ist eine glatte Funktion der individuellen Lernhistorie → kann *nicht* nach Erstellungsdatum wirken. Ersetzt Ankis 8 Sortier-Optionen durch 2 und ist der bessere Default.
- **Impl:** py-fsrs berechnet R (Stabilität + verstrichene Zeit) — unser `FSRSScheduler` wrappt es schon, also ein dünner `retrievability(state, now)`-Zusatz; `review-state` berechnet R pro fälliger Review-Karte + sortiert, neue Karten separat shuffeln + interleaven; irreführenden Docstring fixen. **Kein Schema.**
- **Kopplung:** „random auswahl aus dem set" hängt mit Gap 4a zusammen — bei mehr Karten als Session gilt **erst shuffeln, dann cappen** = „zufällige N aus dem Set" (statt erste N nach Datum). Der Shuffle hier liefert schon vor Limits Wert.
- **Warum zuerst:** kleinster Eingriff, größte tägliche Erleichterung, **kein UI-Entscheid nötig**.

## Vorschlag 2 — Study-Set-Picker + Tags raus (Gaps 1 + 2) · **M**

- **Ziel:** ein **Lern-Launcher** — Checkbox-Liste der **Collections** mit **Fällig-Badges** je Collection; mehrere ankreuzen → **Union** der fälligen Karten („die 3, nicht die 2"). „Alles fällig" bleibt. **Tags raus aus dem Study-Picker** (bleiben überall sonst: Organisieren, Agent-Taggen, `/tags`-Manager). Ein **„Vorlernen/mehr"-Toggle** (ignoriert das Tageslimit, zieht die nächst-fälligen) als einzige Custom-Study-Fluchttür — später.
- **Impl:** `review-state` akzeptiert **mehrere** Collection-Ids → OR/Union (`Card.collections.any(id.in_(ids))`); UI ersetzt das `<select>` durch Multi-Select-Launcher + per-Collection-Fällig-Zähler (ein group-by statt N Queries); Tag-Optgroup entfernen. **Kein Schema.**
- **Entscheidung (deine):** Tags **ganz raus** aus dem Study-Picker (empfohlen — Collections sind die Study-Achse) vs. als **sekundärer, einklappbarer** Filter behalten.

## Vorschlag 3 — Daily-Limits (Gap 4a) · **M**

- **Ziel:** zwei globale Ganzzahlen — **new/Tag** (Default **10**, nicht Ankis 20 — nachhaltiger für Solo) und **max reviews/Tag** (Default **200**); **neue respektieren den Review-Cap** (kein new-Stapel an schweren Tagen). UI zum Setzen; Fortschritt gegen die Caps in der Session. „X neu verfügbar, Y reviews fällig" am Launcher.
- **Impl:** kleiner **User-Settings-Store** (neue Mini-Tabelle **oder** JSON-Blob auf `User` — UI-editierbar), Cap-Logik in `review-state` (Pools trennen: reviews = stability≠NULL fällig, gecappt; new = stability=NULL, gecappt + nur unter Review-Cap). **Kleiner Schema-Touch** (Settings-Store, Auto-Migrate).
- **Entscheidung (deine):** Default-Zahlen (10/200); Settings-Speicher (env vs. DB — DB, weil UI-editierbar).

## Vorschlag 4 — Statistik (Gap 4b) · **M/L (größter — eigene Fläche)**

- **Ziel (die entscheidungs-nützliche Handvoll, Rest ist Deko):**
  1. **Heute-Zusammenfassung** (am Launcher, nicht Extra-Seite): „N fällig, M neu verfügbar, ~T min".
  2. **Future-Due-Forecast** (nächste ~2–4 Wochen) — die beste Planungs-Stat; zeigt Spitzen/Täler zum Neu-Karten-Dosieren. **Überfälligen Rückstand als eigenen Marker** (Anki versteckt ihn — hier nicht wiederholen).
  3. **Reifegrad-Zähler** — neu / lernend / jung (interval <21d) / reif (≥21d). Ein Blick = „wie viel meiner Sammlung ist wirklich haltbar".
  4. **Ist- vs. Ziel-Retention** — die FSRS-Rückkopplung; driftet die gemessene True-Retention deutlich unter deine Desired-Retention, ist FSRS neu zu optimieren oder die Ratings sind schief. Die eine Zahl, die sagt, dass der *Algorithmus selbst* gesund ist.
  - **Optional (nicht launch-kritisch):** **Workload-Simulator** — gegeben new/Tag + Desired-Retention, projiziere Steady-State-reviews/Tag. Macht „welche Retention?" aus Raten zu „90 % ≈ 40/Tag, 95 % ≈ 80/Tag für dich". Community-Sweet-Spot **85–92 %** (Default 90); über ~97 % explodiert die Last.
- **Impl:** Stats-Endpoint (rechnet aus Review/Card/`rating_history`) + Stats-View; Forecast iteriert Fälligkeiten, Reifegrad klassifiziert per stability/interval, Retention analysiert `rating_history` (erste Review pro Tag). `desired_retention` sichtbar machen (heute env-only). **Kein Schema** (liest Bestand).
- **Entscheidung (deine):** die 4er-Stat-Menge vs. Teilmenge; Workload-Simulator gewünscht?

---

## Empfohlene Sequenz

1. **Reihenfolge-Fix** (S/M) — sofortige Erleichterung, kein UI-Entscheid, kein Schema. Bester erster Schritt.
2. **Study-Set-Picker + Tags raus** (M) — deine Gaps 1+2, die UI-Aufräumung. Kein Schema.
3. **Daily-Limits** (M) — kleiner Schema-Touch (Settings).
4. **Statistik** (M/L) — kein Schema, aber die größte neue Fläche.

Jeder Schritt ist ein eigener Sprint (Master schreibt den Prompt, Sub-Thread führt aus). 1 und 2 könnten gebündelt werden (beide touchen `review-state` + `review.js`), aber getrennt ist 1 ein sauberer sofort-shipbarer Gewinn ohne UI-Entscheid.

## Offene Entscheidungen (Oli)

- **Reihenfolge/Start:** Reihenfolge-Fix zuerst (empf.) vs. Picker zuerst vs. ein großer Lern-Sprint.
- **Tags:** ganz raus aus dem Study-Picker (empf.) vs. sekundär/einklappbar.
- **Limit-Defaults:** 10 new / 200 reviews (empf.) — oder andere.
- **Stats-Umfang:** die 4 + optional Workload-Sim.
