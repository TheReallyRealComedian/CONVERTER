# SPRINT LEARN-STEP — Ein Lernschritt statt zwei

**Größe**: S (1 Phase) · **Datum**: 2026-07-25 · **Parallel erlaubt zu**: LEARN-COUNT (disjunkt — der fasst nur `static/js/review.js` an, dieser nur `services/scheduler/` + Tests)

## Warum

Oli: *„ich dürfte eigentlich nicht eine wiederholung der gerade gemachten durchführen (es sei denn die waren schwer) — sprich wir müssen echte tagesgrenzen haben."*

Heute wird **jede neue Karte an ihrem ersten Tag systematisch zweimal gezeigt** — die zweite Vorlage 10 Minuten nach der ersten, **auch nach einem korrekten „Gut"**. Das ist kein Randfall, sondern trifft jede einzelne neue Karte und ist genau die Verletzung, die Oli beschreibt.

## Gegroundete Diagnose (Master, gemessen — nicht neu herleiten)

[services/scheduler/fsrs_scheduler.py:35](services/scheduler/fsrs_scheduler.py) konstruiert `FSRSEngine` **nur** mit `desired_retention` + `enable_fuzzing`. `learning_steps`/`relearning_steps` bleiben damit auf py-fsrs' Defaults — verifiziert gegen den Pin `fsrs==6.3.1`:

```
learning_steps   = (1 min, 10 min)
relearning_steps = (10 min,)
```

**Ist-Verhalten (gemessen):**

| neue Karte | wieder fällig | | graduierte Karte | wieder fällig |
|---|---|---|---|---|
| Nochmal | 1 min ↩︎ | | Nochmal | 10 min ↩︎ ✅ gewollt |
| Schwer | 5:30 ↩︎ | | Schwer | 8 Tage ✅ |
| **Gut** | **10 min ↩︎ ❌** | | Gut | 11 Tage ✅ |
| Einfach | 8 Tage ✅ | | | |

Die **Wiederholungskarten sind bereits korrekt** — nur „Nochmal" kommt zurück. Kaputt ist ausschließlich die Neu-Karten-Spalte.

**Warum genau zweimal, nie öfter:** [`_reconstruct`](services/scheduler/fsrs_scheduler.py) baut jede Karte mit gesetzter `stability` als **`State.Review`, `step=None`** wieder auf (State/Step werden bewusst nicht persistiert). Die Lernschritt-Leiter greift daher **nur beim allerersten Rating** — danach ist die Karte für py-fsrs eine Review-Karte. Erste Vorlage → 10 min → zweite Vorlage → normales Mehrtage-Intervall.

**Soll-Verhalten mit EINEM Lernschritt (ebenfalls gemessen):**

| neue Karte | wieder fällig | |
|---|---|---|
| Nochmal | 10 min ↩︎ | gewollt — war schwer |
| Schwer | 15 min ↩︎ | gewollt — war schwer |
| **Gut** | **2 Tage ✅** | weg für heute |
| Einfach | 8 Tage ✅ | weg für heute |

Wortwörtlich Olis Regel. Und **kein Qualitätsverlust**: die Karte landet nach einem „Gut" im selben Zustand wie vorher nach zwei „Gut" — nur ohne die überflüssige zweite Vorlage.

---

# Phase 1 — Ein Lernschritt

## 1.1 Scheduler explizit konfigurieren

[services/scheduler/fsrs_scheduler.py](services/scheduler/fsrs_scheduler.py): `FSRSEngine` bekommt **explizit** `learning_steps=(timedelta(minutes=10),)` und `relearning_steps=(timedelta(minutes=10),)`.

⚠️ **Beide explizit setzen, auch wenn `relearning_steps` dem Default entspricht.** Der ganze Bug entstand daraus, dass eine verhaltensbestimmende Einstellung stillschweigend von der Library geerbt wurde. Ein py-fsrs-Bump, der die Defaults ändert, darf das Lernverhalten nicht erneut unbemerkt verschieben.

Docstring-Kommentar mit der Begründung: **ein** Schritt, weil ein korrekt beantwortetes „Gut" die Karte für den Tag erledigen soll; der Schritt bleibt für „Nochmal"/„Schwer" bestehen, weil eine gepatzte Karte im selben Durchgang nochmal drankommen **soll**. Die zwei Messtabellen oben mit aufnehmen — sie sind die Begründung.

**Keine neue Env-Variable.** `desired_retention` ist env-gesteuert, das hier nicht: es ist eine Lern-Doktrin, kein Betriebsparameter, und niemand hat einen Regler dafür verlangt.

## 1.2 Bestandsdaten

**Keine Migration, kein Backfill.** Karten mit gesetzter `stability` werden ohnehin als `State.Review` rekonstruiert — die Änderung greift ausschließlich beim ersten Rating einer noch nie bewerteten Karte. Kurz im Commit-Text festhalten, damit niemand nach Migrationsbedarf sucht.

## 1.3 Tests

`tests/` erweitern (bestehende Scheduler-Tests als Vorbild). Die Zeitgrenze „kommt heute nochmal" ist der Prüfstein — als Schwelle etwas wie „< 12 h = kommt heute wieder" verwenden, nicht auf exakte Minutenwerte festnageln (die kommen aus der Library und dürfen sich bei einem Bump leicht verschieben):

| Fall | Erwartung |
|---|---|
| neu + Gut | Intervall **> 12 h** — die Regressionsversicherung für diesen Sprint |
| neu + Einfach | > 12 h |
| neu + Nochmal | < 12 h (kommt wieder — gewollt) |
| neu + Schwer | < 12 h (kommt wieder — gewollt) |
| graduiert + Nochmal | < 12 h (Relearning — unverändert) |
| graduiert + Gut/Schwer/Einfach | > 12 h (unverändert) |

Plus ein **Sentinel** analog zu den nh3-/Flask-WTF-Sentinels: ein Test, der die konfigurierten `learning_steps`/`relearning_steps` am Engine-Objekt festnagelt, damit ein py-fsrs-Bump mit geänderten Defaults **laut** wird statt still das Lernverhalten zu verschieben.

## 1.4 Verifikation

`pytest tests/` grün. Ein Live-Smoke ist **nicht** nötig (reine Scheduler-Logik, voll unit-testbar) — aber im Bericht die Vorher/Nachher-Intervalle für die vier Neu-Karten-Ratings ausweisen.

## Stop
`pytest tests/` grün, Testzahl vorher/nachher, die vier Intervalle belegt. **Commit + Push** `fix(LEARN-STEP): ein Lernschritt — korrekt beantwortete neue Karten sind für den Tag erledigt`. Danach STATUS.md/BACKLOG.md-Wrap (Bullet-Guard `grep -nE '(- \*\*.*){2,}' BACKLOG.md` muss leer sein), CLAUDE.md-Notiz im Learning-Bullet, und ein Memory `reference_fsrs_learning_steps_default_trap.md` (Kern: verhaltensbestimmende Library-Defaults nie implizit erben; `_reconstruct`-State-Verlust macht die Leiter einstufig-wirksam; Sentinel gegen Bumps). Schlussbericht mit Deploy-Schritten.

## Nicht-Ziele

- **Kein** „Mehr lernen"-Knopf — das ist der Folge-Sprint LEARN-MORE (fasst `review.js` an, muss auf LEARN-COUNT warten).
- **Kein** Anfassen der Cap-/Ordering-Logik in `app_pkg/learn.py`, der Tagesgrenzen-Zählung oder von `desired_retention`.
- **Kein** neuer Env-Regler, keine Settings-UI.
- **Kein** Anfassen von `static/js/review.js` — dort arbeitet parallel LEARN-COUNT.
