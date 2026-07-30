"""FSRS scheduler — the default engine, via py-fsrs (PyPI ``fsrs``), plus the
LEARN-UP workload simulator (module function, engine-specific).

Maps our persisted ``Review`` dict onto an ``fsrs.Card`` and back. py-fsrs owns
the interval/stability/difficulty math; we own ``reps``/``lapses`` (FSRS-6's
Card no longer tracks them).

**Documented simplification.** The locked ``Review`` schema carries no column for
FSRS's internal ``state``/``step``, so a previously-reviewed card is
reconstructed in the ``Review`` state (graduated). The stability/difficulty-driven
interval math — the part that matters — is fully preserved; only the sub-day
learning/relearning *step ladder* is collapsed after the first rating. Since
LEARN-STEP the ladder is deliberately a single 10-min step anyway (see
``FSRSScheduler.__init__``), so nothing meaningful is lost. Fuzzing is disabled
so scheduling is deterministic.
"""
import os
from datetime import datetime, timedelta, timezone

from fsrs import Card as FSRSCard
from fsrs import Rating, State
from fsrs import Scheduler as FSRSEngine

from .base import RATINGS, Scheduler, as_aware_utc, initial_review_state

_RATING_MAP = {
    'again': Rating.Again,
    'hard': Rating.Hard,
    'good': Rating.Good,
    'easy': Rating.Easy,
}

# py-fsrs 6.3.1's own ``Scheduler(maximum_interval=...)`` default — pinned here
# so the un-set env case is provably behaviour-neutral (sentinel in
# tests/test_scheduler.py). 36500 days ≈ 100 years == "no cap".
DEFAULT_MAXIMUM_INTERVAL = 36500


def _parse_max_interval(raw):
    """Parse ``FSRS_MAXIMUM_INTERVAL`` — built like ``_parse_retention``.

    Anything invalid (non-int, ``<= 0``) falls back to the library default
    silently; this never raises. Lives here rather than next to
    ``_parse_retention`` in ``__init__`` because ``simulate_workload`` below
    needs the same parser and cannot import from the package ``__init__``
    (circular) — and because the cap is FSRS-specific (SM-2 has no such knob).
    """
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAXIMUM_INTERVAL
    if value <= 0:
        return DEFAULT_MAXIMUM_INTERVAL
    return value


class FSRSScheduler(Scheduler):
    def __init__(self, desired_retention=0.9, enable_fuzzing=False,
                 maximum_interval=DEFAULT_MAXIMUM_INTERVAL):
        """ONE learning step, set explicitly (LEARN-STEP).

        A correctly answered new card is done for the day; the step ladder only
        exists so an again/hard card comes back within the same session. With
        py-fsrs' inherited default of TWO steps (1 min, 10 min), every new card
        was systematically shown twice on day one — even after a correct "Gut":
        because ``_reconstruct`` rebuilds any card with a stability as
        ``State.Review``/``step=None``, the ladder only fires on the very first
        rating, so first show → 10 min → second show → multi-day interval.

        Measured against ``fsrs==6.3.1``, first rating of a new card:

        ==========  ===================  ==================
        rating      default (1m, 10m)    one step (10 min)
        ==========  ===================  ==================
        Nochmal     1 min  ↩              10 min ↩  (wanted)
        Schwer      5:30   ↩              15 min ↩  (wanted)
        Gut         10 min ↩  ← the bug   2 d   ✅
        Einfach     8 d    ✅             8 d   ✅
        ==========  ===================  ==================

        Graduated cards were already correct (again 10 min ↩, hard/good/easy
        multi-day) and are unchanged. ``relearning_steps`` matches the current
        library default but is pinned anyway: a verhaltensbestimmende setting
        must never be inherited silently again — a py-fsrs bump that changes
        the defaults has to fail the sentinel in ``tests/test_scheduler.py``,
        not silently shift the learning behaviour. Deliberately no env knob:
        this is Lern-Doktrin, not an operating parameter.

        ``maximum_interval`` (LEARN-TUNE), in contrast, IS an operating
        parameter and comes from ``FSRS_MAXIMUM_INTERVAL`` via
        ``get_scheduler()``. It caps the scheduled INTERVAL only — measured
        against ``fsrs==6.3.1`` on good/good/good: 36500 → 58 d, 21 → 21 d,
        with the stability identical (58.4) either way. The model stays
        unfalsified; we just stop acting on the far end of its curve. **SM-2 has
        no equivalent knob and deliberately stays untouched** — do not "restore
        symmetry" that does not exist.
        """
        # Fuzzing off → deterministic intervals (predictable for the user and
        # for the tests). desired_retention is the FSRS target recall (~0.9).
        self._engine = FSRSEngine(desired_retention=desired_retention,
                                  enable_fuzzing=enable_fuzzing,
                                  maximum_interval=maximum_interval,
                                  learning_steps=(timedelta(minutes=10),),
                                  relearning_steps=(timedelta(minutes=10),))

    def new_card_state(self):
        return initial_review_state()

    def apply_rating(self, review_state, rating):
        if rating not in RATINGS:
            raise ValueError(f"invalid rating {rating!r}; expected one of {RATINGS}")
        now = datetime.now(timezone.utc)
        card = self._reconstruct(review_state)
        updated, _log = self._engine.review_card(
            card, _RATING_MAP[rating], review_datetime=now)
        reps = int(review_state.get('reps') or 0) + 1
        # "again = lapse" — we count lapses ourselves regardless of FSRS state.
        lapses = int(review_state.get('lapses') or 0) + (1 if rating == 'again' else 0)
        return {
            'due': as_aware_utc(updated.due),
            'stability': updated.stability,
            'difficulty': updated.difficulty,
            'last_reviewed': now,
            'reps': reps,
            'lapses': lapses,
        }

    def retrievability(self, review_state, now=None):
        """Current recall probability from the FSRS forgetting curve (LEARN-UP).

        ``None`` for a brand-new card (stability NULL — no curve yet).
        py-fsrs truncates elapsed time to whole days, so R is day-granular:
        cards reviewed today read 1.0, and equal (stability, elapsed-days)
        pairs tie exactly — callers ordering by R need their own tiebreak.
        """
        if review_state.get('stability') is None:
            return None
        now = as_aware_utc(now) or datetime.now(timezone.utc)
        return float(self._engine.get_card_retrievability(
            self._reconstruct(review_state), now))

    def _reconstruct(self, review_state):
        """Rebuild an ``fsrs.Card`` from the persisted dict."""
        stability = review_state.get('stability')
        due = as_aware_utc(review_state.get('due')) or datetime.now(timezone.utc)
        if stability is None:
            # Brand-new card — let FSRS start it in Learning/step 0.
            return FSRSCard(due=due)
        # Previously reviewed → reconstruct as graduated (Review state). See the
        # module docstring for why state/step aren't persisted.
        return FSRSCard(
            state=State.Review,
            step=None,
            stability=stability,
            difficulty=review_state.get('difficulty'),
            due=due,
            last_review=as_aware_utc(review_state.get('last_reviewed')),
        )


def simulate_workload(desired_retention, new_per_day, horizon_days=365,
                      tail_days=90, maximum_interval=None):
    """Steady-state workload ESTIMATE (LEARN-UP P4): expected reviews/day.

    py-fsrs 6.3.1 ships no simulator (checked: ``Scheduler`` has none, the
    Optimizer's ``_simulate_cost`` is private parameter-fitting cost), so this
    is a simple expected-value cohort simulation — a what-if PROJECTION whose
    ``desired_retention`` input never touches the real scheduler:

    * Every introduced card follows ONE deterministic trajectory: pass/fail
      at each review collapse into the expected stability
      ``S' = r·S_good + (1-r)·S_again`` (branches merged, sub-day learning
      steps collapsed — mirrors the production simplification above).
    * Reviews happen exactly at due (R == r there by construction); the next
      interval comes from the FSRS curve ``I(S) = S/FACTOR·(r^(1/DECAY) − 1)``
      with ``DECAY = −parameters[20]`` and ``FACTOR = 0.9^(1/DECAY) − 1`` —
      derived from the PUBLIC ``parameters`` (how the engine builds its own
      ``_FACTOR``/``_DECAY`` in 6.3.1; re-verify the derivation on a bump).
    * With one intro cohort per day, day ``d`` sees ``new_per_day`` reviews
      per trajectory offset ≤ ``d``; the result is the mean over the last
      ``tail_days`` of the horizon. An estimate, not a promise.

    ``desired_retention`` stays a pure what-if INPUT, but the interval cap is an
    OPERATING parameter (LEARN-TUNE): ``maximum_interval=None`` resolves it from
    ``FSRS_MAXIMUM_INTERVAL`` — the same env key, through the same parser, that
    ``get_scheduler()`` uses. The invariant is that simulation and the real
    scheduler cap at the same value; otherwise the projection would promise
    intervals the scheduler never hands out (the drift LEARN-UP ruled out for
    the today-counts via ``capped_session_counts``). Note the direction: a cap
    RAISES the projected load, because shorter intervals mean more reviews per
    card per year. That is the price of the cap, not a bug.

    Returns expected reviews/day (float, intro ratings not counted — they are
    the "new" slots, not reviews).
    """
    if new_per_day <= 0:
        return 0.0
    if maximum_interval is None:
        maximum_interval = _parse_max_interval(
            os.environ.get('FSRS_MAXIMUM_INTERVAL'))
    engine = FSRSEngine(desired_retention=desired_retention, enable_fuzzing=False)
    decay = -engine.parameters[20]
    factor = 0.9 ** (1.0 / decay) - 1.0
    t0 = datetime(2000, 1, 1, tzinfo=timezone.utc)  # virtual clock

    def interval_days(stability):
        return max(1, min(maximum_interval,
                          round(stability / factor
                                * (desired_retention ** (1.0 / decay) - 1.0))))

    def expected_next_stability(stability):
        due = t0 + timedelta(days=interval_days(stability))

        def rate(rating):
            card = FSRSCard(state=State.Review, step=None, stability=stability,
                            difficulty=5.0, due=due, last_review=t0)
            updated, _log = engine.review_card(card, rating, review_datetime=due)
            return updated.stability

        return (desired_retention * rate(Rating.Good)
                + (1.0 - desired_retention) * rate(Rating.Again))

    intro, _log = engine.review_card(FSRSCard(due=t0), Rating.Good,
                                     review_datetime=t0)
    stability = intro.stability
    offsets, cum = [], 0
    while True:
        cum += interval_days(stability)
        if cum > horizon_days:
            break
        offsets.append(cum)
        stability = expected_next_stability(stability)

    tail = range(max(0, horizon_days - tail_days), horizon_days)
    total = sum(new_per_day * sum(1 for o in offsets if o <= day) for day in tail)
    return total / max(1, len(tail))
