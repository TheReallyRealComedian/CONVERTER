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


class FSRSScheduler(Scheduler):
    def __init__(self, desired_retention=0.9, enable_fuzzing=False):
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
        """
        # Fuzzing off → deterministic intervals (predictable for the user and
        # for the tests). desired_retention is the FSRS target recall (~0.9).
        self._engine = FSRSEngine(desired_retention=desired_retention,
                                  enable_fuzzing=enable_fuzzing,
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
                      tail_days=90):
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

    Returns expected reviews/day (float, intro ratings not counted — they are
    the "new" slots, not reviews).
    """
    if new_per_day <= 0:
        return 0.0
    engine = FSRSEngine(desired_retention=desired_retention, enable_fuzzing=False)
    decay = -engine.parameters[20]
    factor = 0.9 ** (1.0 / decay) - 1.0
    t0 = datetime(2000, 1, 1, tzinfo=timezone.utc)  # virtual clock

    def interval_days(stability):
        return max(1, round(stability / factor
                            * (desired_retention ** (1.0 / decay) - 1.0)))

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
