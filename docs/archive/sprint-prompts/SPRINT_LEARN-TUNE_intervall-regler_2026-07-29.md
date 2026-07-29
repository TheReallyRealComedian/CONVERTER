# SPRINT LEARN-TUNE — Intervall-Obergrenze als Regler, Retention dokumentiert

**Größe**: S (2 Phasen) · **Datum**: 2026-07-29 · **Parallel dispatchbar** mit LEARN-RATE (disjunkte Dateien: dieser Sprint fasst `services/scheduler/` + `tests/` + `.env.example`, LEARN-RATE fasst `templates/` + `static/`)

## Warum

Olis 40 TCE-Karten stehen nach drei Runden 1–7 Wochen auf Halde (`due` 30.07.–13.09., Stabilität 11,7–56,6 Tage), obwohl über die Hälfte der Antworten „Schwer" war.

**Die FSRS-Mathematik ist dabei nicht das Problem** — das ist gemessen, nicht vermutet. Dieselbe Rating-Folge, einmal mit Olis echtem 17-Tage-Verzug, einmal pünktlich am `due` bewertet:

| Muster | mit 17-Tage-Lücke | pünktlich |
|---|---|---|
| hard/hard/hard | 15,2 Tage | **2,1 Tage** |
| hard/hard/good | 22,3 Tage | **3,0 Tage** |

Pünktlich macht FSRS aus dreimal „Schwer" genau die zwei Tage, die man erwartet. Die `w`-Parameter sind unberührte py-fsrs-Defaults und **brauchen keine Korrektur**. Was die Intervalle aufbläst, ist der Verzug — FSRS verbucht „nach 17 Tagen noch gewusst" korrekterweise als Stabilitätsbeweis.

Deshalb baut dieser Sprint **keinen** Eingriff in die Mathematik (kein Deckeln von `elapsed_days`: py-fsrs bietet dafür keinen Hook, es würde dem Modell dauerhaft falsche Daten unterschieben, und die Prämisse „die App hat den Verzug verursacht" ist widerlegt — die Tages-Caps kamen mit LEARN-UP am 18.07., die Lücke war 01.07.→18.07.).

Stattdessen zwei ehrliche Betriebsparameter, beide von Oli gewählt: **Retention 0,92** und **Intervall-Deckel 60 Tage**.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

- `FSRSScheduler.__init__` ([services/scheduler/fsrs_scheduler.py:34](services/scheduler/fsrs_scheduler.py)) reicht heute `desired_retention`, `enable_fuzzing`, `learning_steps`, `relearning_steps` durch. **`maximum_interval` wird nicht gesetzt** → py-fsrs-Default **36500**.
- Verifizierte py-fsrs-6.3.1-Signatur: `Scheduler(parameters, desired_retention=0.9, learning_steps, relearning_steps, maximum_interval: int = 36500, enable_fuzzing=True)`.
- `get_scheduler()` ([services/scheduler/__init__.py](services/scheduler/__init__.py)) liest `SCHEDULER_ENGINE` und `FSRS_DESIRED_RETENTION`; `_parse_retention` fällt bei allem Ungültigen auf den Default zurück.
- **`FSRS_DESIRED_RETENTION` ist in Olis `.env` gar nicht gesetzt** → läuft auf 0,9.
- `simulate_workload` ([fsrs_scheduler.py:126](services/scheduler/fsrs_scheduler.py)) berechnet `interval_days()` mit einer **eigenen Formel** aus `engine.parameters` — die Engine clampt dort **nicht** mit. Ein `maximum_interval` an der Engine würde die Projektion also **nicht** erreichen.
- Gemessene Wirkung des Deckels (Muster good/good/good): `36500` → 58 Tage, `21` → 21 Tage. Der Deckel greift am Intervall, **nicht** an der Stabilität (die bleibt 58,4) — genau richtig, das Modell bleibt unverfälscht.
- Gemessene Wirkung der Retention (Muster hard/hard/good): 0,90 → 22 d · 0,92 → 16 d · 0,95 → 9 d. Auch hier bleibt die Stabilität identisch.

## Gesperrte Entscheidungen (Oli, 2026-07-29)

1. **Retention 0,92** — nicht 0,95. Die Simulation rechnet bei Olis aktuellen 24 neuen Karten/Tag: 0,90 → 120 Reviews/Tag, 0,92 → 141, 0,95 → **210** und damit über sein eigenes Review-Limit von 200.
2. **Deckel 60 Tage** — nicht die im Ursprungsbericht vorgeschlagenen 21. 21 Tage zwängen jede reife Karte für immer in einen Drei-Wochen-Takt und machen genau die Ersparnis kaputt, für die FSRS existiert. 60 fängt die Ausreißer.

---

# Phase 1 — Der Regler

## 1.1 `FSRS_MAXIMUM_INTERVAL`

- `FSRSScheduler.__init__` bekommt `maximum_interval=36500` und reicht es an die Engine durch.
- `get_scheduler()` liest `FSRS_MAXIMUM_INTERVAL` über einen `_parse_max_interval`-Helfer, **gebaut wie `_parse_retention`**: alles Ungültige (nicht-int, ≤ 0) fällt still auf den Default zurück, nichts wirft.
- **Ohne gesetzte Env-Variable ist der Sprint verhaltensneutral** — der Default ist exakt der py-fsrs-Default. Das ist die Abnahmebedingung.
- **SM-2 bleibt unberührt** (kennt kein `maximum_interval`). Im Docstring benennen, damit niemand später Symmetrie herstellt, die es nicht gibt.

## 1.2 Der Simulator muss mitziehen

`simulate_workload` clampt heute nicht. Wenn der echte Scheduler bei 60 Tagen deckelt und die Projektion bis 58+ rechnet, driften Vorhersage und Realität auseinander — genau die Art Drift, die LEARN-UP mit `capped_session_counts` bewusst ausgeschlossen hat.

`interval_days()` muss auf denselben Wert clampen, mit dem der echte Scheduler läuft. `desired_retention` bleibt dabei **reiner What-if-Input** (der Simulator ist eine Projektion), `maximum_interval` dagegen ist ein **Betriebsparameter** und kommt aus der Env — dieselbe Quelle wie `get_scheduler()`. Wie du das verdrahtest (Env-Lesen im Modul wie `get_scheduler()`, oder Default-Argument), ist deine Wahl; die Invariante ist: **Simulation und Scheduler benutzen denselben Deckel.**

## 1.3 Tests

- Ohne Env-Variable: die Engine läuft mit **36500** → verhaltensneutral.
- `FSRS_MAXIMUM_INTERVAL=60` → ein Intervall, das ungedeckelt über 60 Tage läge, kommt auf 60 heraus; die **Stabilität ist unverändert** (der Deckel verfälscht das Modell nicht — explizit festnageln).
- Müll-Werte (`abc`, `0`, `-5`, leer) → Default, kein Wurf.
- **Sentinel wie bei LEARN-STEP**: der py-fsrs-Default für `maximum_interval` ist **36500**. Ein Bump, der ihn ändert, muss laut werden statt still das Scheduling zu verschieben — genau die Falle, die LEARN-STEP schon einmal gestellt hat (Memory `reference_fsrs_learning_steps_default_trap`).
- Simulator: mit Deckel sinkt die erwartete Last gegenüber ohne Deckel.

## Stop
`pytest tests/` grün, Testzahl vorher/nachher (Baseline **762**). **Commit + Push** `feat(LEARN-TUNE): FSRS_MAXIMUM_INTERVAL als Env-Regler (P1)`. Dann warten.

---

# Phase 2 — Wrap + Deploy-Anleitung

- **`.env.example`**: beide Schlüssel mit gemessener Wirkung dokumentieren — `FSRS_DESIRED_RETENTION` (Default 0,9; 0,92 ≈ 27 % kürzere Intervalle bei ~18 % mehr Last) und `FSRS_MAXIMUM_INTERVAL` (Default 36500 = aus).
- **CLAUDE.md**, Learning-Abschnitt: die zwei Regler, die Messwerte, und der ausdrückliche Satz, **warum kein Eingriff in `elapsed_days` gebaut wurde**.
- **STATUS.md** + **BACKLOG.md** (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein).
- **Memory**: *Verhaltensregler, die eine Library nativ anbietet, als Env durchreichen statt die Mathematik zu umgehen — und den Library-Default per Sentinel festnageln.* Verlinken auf `[[reference_fsrs_learning_steps_default_trap]]` und `[[reference_swappable_scheduler_interface]]`.

**Schlussbericht muss zwei Dinge unmissverständlich enthalten:**

1. **Die exakten Env-Zeilen für die Mintbox** (`FSRS_DESIRED_RETENTION=0.92`, `FSRS_MAXIMUM_INTERVAL=60`) plus den Hinweis, dass die `.env` dort **nicht** aus dem Repo kommt und von Hand ergänzt werden muss.
2. ⚠️ **Beide Regler wirken ausschließlich auf künftige Bewertungen.** Sie setzen **keine** bestehende Karte um. Olis 40 TCE-Karten bleiben nach dem Deploy exakt dort stehen, wo sie stehen (30.07.–13.09.) — die Regler ändern daran nichts, und LEARN-MOREs Vorziehen reicht nur 7 Tage weit. Wer nach dem Deploy sofortige Wirkung erwartet, wird sie wochenlang nicht sehen. Das ist kein Fehler dieses Sprints, aber es muss im Bericht stehen.

## Nicht-Ziele

- **Kein** Eingriff in `elapsed_days`, `last_review` oder sonst eine FSRS-Eingangsgröße.
- **Kein** Optimizer-Lauf. Der py-fsrs-Optimizer bräuchte torch (das IMG-SLIM gerade aus dem Image geworfen hat) und braucht Größenordnung 1000+ Bewertungen — bei 122 wäre das Kurvenanpassung an Rauschen.
- **Kein** Umsetzen bestehender Karten (siehe oben — eigenes Thema, eigener Sprint, falls Oli ihn will).
- **Keine** Per-User-Einstellung. Beide Werte bleiben Env, wie `desired_retention` es heute schon ist.
