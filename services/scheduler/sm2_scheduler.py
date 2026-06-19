"""SM-2 fallback scheduler — the classic SuperMemo-2 algorithm behind the same
interface as FSRS, so the engine is genuinely swappable (config picks one).

Column reuse (the schema is shared with FSRS, no extra columns): ``stability``
holds the **interval in days**, ``difficulty`` holds the **easiness factor**
(EF, starts 2.5, floored at 1.3). ``reps`` ticks every review; ``lapses`` ticks
on ``again``. Quality mapping: again=0, hard=3, good=4, easy=5. A small
per-rating nudge keeps hard < good < easy on the same step.
"""
from datetime import datetime, timedelta, timezone

from .base import RATINGS, Scheduler, initial_review_state

_QUALITY = {'again': 0, 'hard': 3, 'good': 4, 'easy': 5}
_DEFAULT_EF = 2.5
_MIN_EF = 1.3


class SM2Scheduler(Scheduler):
    def new_card_state(self):
        return initial_review_state()

    def apply_rating(self, review_state, rating):
        if rating not in RATINGS:
            raise ValueError(f"invalid rating {rating!r}; expected one of {RATINGS}")
        now = datetime.now(timezone.utc)
        q = _QUALITY[rating]

        ef = review_state.get('difficulty')
        ef = _DEFAULT_EF if ef is None else ef
        interval = review_state.get('stability')
        interval = 0.0 if interval is None else interval
        reps = int(review_state.get('reps') or 0)
        lapses = int(review_state.get('lapses') or 0)

        # SM-2 EF update (applied every review), floored.
        ef = max(_MIN_EF, ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))

        if q < 3:  # again → lapse, relearn this session
            lapses += 1
            new_interval = 0.0
        else:
            if interval < 1:
                new_interval = 1.0
            elif interval < 6:
                new_interval = 6.0
            else:
                new_interval = round(interval * ef, 2)
            # Differentiate hard/good/easy on the same step.
            if rating == 'hard':
                new_interval = max(1.0, round(new_interval * 0.8, 2))
            elif rating == 'easy':
                new_interval = round(new_interval * 1.3, 2)

        reps += 1
        return {
            'due': now + timedelta(days=new_interval),
            'stability': new_interval,   # interval-in-days (column reuse)
            'difficulty': ef,            # easiness factor (column reuse)
            'last_reviewed': now,
            'reps': reps,
            'lapses': lapses,
        }
