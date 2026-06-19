"""Swappable spaced-repetition scheduler interface (R4-LEARN).

A *Scheduler* turns a card's persisted review state + a user rating into the
next review state. Two implementations sit behind this interface — FSRS (the
default, via py-fsrs) and a classic SM-2 fallback — so the engine can be swapped
via config without touching the rate endpoint or the schema.

The interface deals in a plain ``dict`` mirroring the persisted ``Review``
columns, not the ORM row, so it stays storage-agnostic and trivially unit
testable::

    state = scheduler.new_card_state()              # FSRS-"new"
    state = scheduler.apply_rating(state, 'good')   # advance one review

``apply_rating`` returns a dict with: ``due`` (aware-UTC datetime),
``stability``, ``difficulty``, ``last_reviewed`` (aware-UTC), ``reps``,
``lapses``. Ratings are ``again|hard|good|easy``. There is **no auto-grading**:
the rating always comes from the user in the review UI.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone

RATINGS = ('again', 'hard', 'good', 'easy')


def as_aware_utc(value):
    """Coerce None / ISO-string / naive / aware datetime → aware UTC (or None).

    Persisted datetimes come back from SQLite **naive** (UTC wall-clock, the
    dialect drops tzinfo on write); FSRS requires aware-UTC inputs, so the
    schedulers normalise everything on the way in."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def initial_review_state():
    """The shared FSRS-"new" state both engines hand back from
    ``new_card_state`` — due now, nothing learned yet. Mirrors the row
    ``POST /api/cards`` already writes (due=now, reps/lapses 0, rest NULL)."""
    return {
        'due': datetime.now(timezone.utc),
        'stability': None,
        'difficulty': None,
        'last_reviewed': None,
        'reps': 0,
        'lapses': 0,
    }


class Scheduler(ABC):
    """The swap boundary. Both engines implement exactly these two methods."""

    @abstractmethod
    def new_card_state(self) -> dict:
        """Return the initial review state for a brand-new card."""

    @abstractmethod
    def apply_rating(self, review_state: dict, rating: str) -> dict:
        """Given the current review state + a rating, return the next state."""
