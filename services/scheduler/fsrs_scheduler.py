"""FSRS scheduler — the default engine, via py-fsrs (PyPI ``fsrs``).

Maps our persisted ``Review`` dict onto an ``fsrs.Card`` and back. py-fsrs owns
the interval/stability/difficulty math; we own ``reps``/``lapses`` (FSRS-6's
Card no longer tracks them).

**Documented simplification.** The locked ``Review`` schema carries no column for
FSRS's internal ``state``/``step``, so a previously-reviewed card is
reconstructed in the ``Review`` state (graduated). The stability/difficulty-driven
interval math — the part that matters — is fully preserved; only the sub-day
learning/relearning *step ramp* (the 1-min/10-min micro-schedule for brand-new
cards) is collapsed. Fuzzing is disabled so scheduling is deterministic.
"""
from datetime import datetime, timezone

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
        # Fuzzing off → deterministic intervals (predictable for the user and
        # for the tests). desired_retention is the FSRS target recall (~0.9).
        self._engine = FSRSEngine(desired_retention=desired_retention,
                                  enable_fuzzing=enable_fuzzing)

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
