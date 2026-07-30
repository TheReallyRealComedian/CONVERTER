# SPRINT LEARN-BACK — vergiftete Karten zurücksetzen

**Größe**: S (2 Phasen) · **Datum**: 2026-07-30 · **Vorgänger**: LEARN-RATE (`7f75742`) + LEARN-TUNE (`431b866`), beide gelandet **und deployed** (Oli bestätigt 2026-07-30)

## Warum

Olis 40 TCE-Karten tragen einen Zeitplan, der auf ungültigen Daten steht. Alle 122 Bewertungen an diesen Karten entstanden **vor** LEARN-RATE, als „Schwer" für ihn *„kaum bis gar nicht gewusst"* hieß — der Scheduler bekam `hard` (bestanden, Stabilität wächst), wo `again` gehört hätte (Lapse, Stabilität bricht ein). Ihre Stabilität von 11,7–56,6 Tagen ist deshalb nicht bloß zu hoch, sie ist **erfunden**.

Weder LEARN-RATE noch LEARN-TUNE hilft ihnen: der Rating-Fix wirkt erst bei der nächsten Bewertung, und die zwei Regler wirken ausschließlich auf künftige Bewertungen. Die Karten stehen weiter auf `due` 30.07.–13.09., und LEARN-MOREs Vorziehen reicht nur 7 Tage weit.

**Oli hat entschieden: zurücksetzen, nicht umdatieren.** Umdatieren würde die Fälligkeiten zusammenziehen, aber das Phantom-Modell (erfundene Stabilität, erfundene Difficulty) in die Zukunft mitschleppen. Zurücksetzen ist ehrlich: die Karten laufen mit korrekten Ratings frisch an. Preis sind 40 Wiedereinführungen — bewusst akzeptiert.

## Gegroundeter Ist-Zustand (Master — nicht neu herleiten)

- `Review` ([models.py](models.py)): `due` · `stability` · `difficulty` · `last_reviewed` · `reps` · `lapses` · `rating_history` (JSON-Liste, beim Raten angehängt).
- **`initial_review_state()`** ([services/scheduler/base.py](services/scheduler/base.py)) ist die **kanonische** Definition von „neu": `due=now`, `stability/difficulty/last_reviewed=None`, `reps/lapses=0`. Beide Engines geben sie aus `new_card_state()` zurück, und `POST /api/cards` schreibt genau diese Zeile.
- „Neu" wird überall über **`stability IS NULL`** erkannt — `order_due_cards` ([app_pkg/learn.py](app_pkg/learn.py)), `maturity_counts`, iOS' `CardReview.isNew`. Es gibt keine zweite Definition.
- **CLI-Präzedenz**: `@app.cli.command('create-user')` in `_register_cli_commands` ([app_pkg/__init__.py:333](app_pkg/__init__.py)).
- `count_done_today` klassifiziert neu-vs-Review am **ersten** `rating_history`-Eintrag; `true_retention` zählt `again` als Fehlschlag über ein 30-Tage-Fenster.

## Gesperrte Entscheidungen (Master)

1. **`initial_review_state()` wiederverwenden**, nicht die Feldliste nachbauen. Es gibt genau eine Definition von „neu"; eine zweite würde bei der nächsten Scheduler-Änderung still auseinanderlaufen.
2. **`rating_history` wird geleert.** Drei Gründe: (a) die Prämisse des ganzen Sprints ist, dass diese Einträge ungültige Semantik tragen; (b) `count_done_today` liest den **ersten** Eintrag — bliebe die Historie stehen, zählte eine wiedereingeführte Karte gegen das **Review**-Budget statt gegen das Neu-Budget; (c) `true_retention` würde die falsch gemeinten Bewertungen noch ~18 Tage weiterzählen. Preis ist der Audit-Trail; akzeptiert, weil sein Wert durch die Semantik ohnehin zerstört ist.
3. **CLI-Kommando, kein Endpoint, keine UI.** Das ist eine einmalige Korrektur vergifteter Daten, keine wiederkehrende Bedienhandlung. Ein Per-Karte-Reset-Knopf im Review wäre ein eigener, bedarfsgetriebener Sprint — **nicht vorbauen**.
4. **Dry-run ist Default**, echtes Schreiben nur mit `--apply`. Hauspattern (TAG-CLEANUP, `scripts/cleanup_tags.py --apply`).
5. **Scope: eine Sammlung.** Kein „alles zurücksetzen".

---

# Phase 1 — Das Kommando

## 1.1 `flask reset-collection`

Neues CLI-Kommando neben `create-user` in `_register_cli_commands`:

- Argument: die Sammlung (Name **oder** id — Name ist am Prompt bequemer, id eindeutig; beides zu akzeptieren ist billig).
- `--apply` schreibt, ohne den Flag wird nur berichtet.
- Ausgabe in beiden Modi: wie viele Karten betroffen wären/waren, und die Sammlung im Klartext, damit ein Vertipper vor dem `--apply` auffällt.
- Sammlung nicht gefunden → klare Meldung, Exit ≠ 0, **nichts** geschrieben.

## 1.2 Was genau passiert

Für jede betroffene Karte: `Review`-Zeile auf **`initial_review_state()`** setzen **plus** `rating_history = None`.

**Auswahl**: nur Karten der Sammlung, deren `stability IS NOT NULL` — also nur bereits bewertete. Damit ist das Kommando **per Konstruktion idempotent**: der zweite Lauf findet 0 Zeilen (dieselbe Eigenschaft wie beim Daten-Migrations-Präzedenzfall, Memory `reference_data_migration_idempotency`).

Karten in **mehreren** Sammlungen: sie werden zurückgesetzt, wenn die Zielsammlung dabei ist — das ist die einzige sinnvolle Semantik, aber **im Bericht benennen**, falls Überschneidungen existieren.

## 1.3 Tests

- Zurückgesetzte Zeile entspricht **feldweise** `initial_review_state()`, `rating_history` ist `None`, `due` liegt „jetzt".
- **Dry-run schreibt nichts** — Zeile vorher/nachher byte-gleich (der wichtigste Test).
- Idempotenz: zweiter Lauf meldet 0.
- **Scope-Isolation**: Karten anderer Sammlungen sind unverändert (auch die, die dieselben Tags tragen).
- Unbewertete Karten der Zielsammlung bleiben unberührt.
- Unbekannte Sammlung → Fehler, kein Schreiben.
- Nach dem Reset gilt die Karte überall als neu: `stability IS NULL` ⇒ `order_due_cards` steckt sie in den `fresh`-Topf, `maturity_counts` zählt sie als `neu`.

## Stop
`pytest tests/` grün, Testzahl vorher/nachher (Baseline **777**). **Commit + Push** `feat(LEARN-BACK): flask reset-collection (P1)`. Dann warten.

---

# Phase 2 — Runbook + Wrap

⚠️ **Du führst das Kommando NICHT auf Prod aus.** Es ist destruktiv und nicht umkehrbar (`rating_history` ist danach weg). Der Lauf gehört Oli.

Liefere im Schlussbericht ein **kopierfertiges Runbook** in dieser Reihenfolge:

1. **Prod-DB sichern** — `docker cp` aus dem Volume ins Home, exakter Befehl aus Memory `reference_mintbox_prod_db_backup` (`markdown-converter-web`, `/app/data/converter.db`, Ziel `~/converter.db.pre-learn-back-2026-07-30`).
2. **Deploy** des neuen Kommandos (`git pull` + `up -d --build`).
3. **Dry-run** auf `TCE / CD3-Bispecifics` — die gemeldete Zahl muss **40** sein. Weicht sie ab, **stoppen** und melden.
4. **`--apply`**, dann Gegenprobe: `list_collections` muss für die Sammlung `due_count == 40` zeigen (alle sofort fällig, weil `due=now`).

Dazu im Wrap:
- **CLAUDE.md**, Learning-Abschnitt: das Kommando, die zwei gesperrten Entscheidungen (kanonisches `initial_review_state()`, geleerte Historie mit Begründung), und der Satz, dass es **kein** wiederkehrendes Werkzeug ist.
- **STATUS.md** + **BACKLOG.md** (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein).
- **Memory**: *Wenn die Semantik der Eingabedaten ungültig war, ist Zurücksetzen ehrlicher als Umdatieren — und der Reset muss die kanonische „neu"-Definition wiederverwenden statt sie nachzubauen.* Verlinken auf `[[reference_rating_scale_outcome_label]]` und `[[reference_data_migration_idempotency]]`. **Nach dem Schreiben mit `ls` prüfen**, dass die Datei liegt und MEMORY.md genauso viele Index-Zeilen hat wie es Dateien gibt — bei LEARN-RATE ist ein Memory-Write stillschweigend nicht angekommen.
- **Im Bericht benennen**: die 40 Karten kommen als neue Karten zurück und laufen damit gegen Olis `daily_new_limit` — bei ~24/Tag verteilt sich die Wiedereinführung über zwei Tage. Das ist korrekt, aber er soll es erwarten.

## Nicht-Ziele

- **Kein** Endpoint, **keine** UI, **keine** Agent-/Token-Fläche.
- **Kein** Umdatieren-Modus — Oli hat sich entschieden, beides zu bauen wäre unbenutzte Fläche.
- **Kein** Anfassen der Karteninhalte (`front`/`back`/`front_svg`/…), der Sammlungen oder der Tags. Nur die `Review`-Zeile.
- **Kein** Prod-Lauf durch dich.
