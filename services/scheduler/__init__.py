"""Spaced-repetition scheduling — swappable FSRS / SM-2 engines (R4-LEARN).

``get_scheduler()`` returns the active engine, chosen by config (default FSRS).
The rate endpoint calls it per request; constructing a scheduler is cheap.

Config (env):
* ``SCHEDULER_ENGINE`` — ``fsrs`` (default) | ``sm2``
* ``FSRS_DESIRED_RETENTION`` — float in (0, 1), default 0.9
"""
import os

from .base import RATINGS, Scheduler
from .fsrs_scheduler import FSRSScheduler
from .sm2_scheduler import SM2Scheduler

DEFAULT_DESIRED_RETENTION = 0.9


def _parse_retention(raw):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_DESIRED_RETENTION
    if not 0.0 < value < 1.0:
        return DEFAULT_DESIRED_RETENTION
    return value


def get_scheduler():
    """Return the active scheduler instance per the env config (default FSRS)."""
    engine = (os.environ.get('SCHEDULER_ENGINE') or 'fsrs').strip().lower()
    if engine == 'sm2':
        return SM2Scheduler()
    retention = _parse_retention(os.environ.get('FSRS_DESIRED_RETENTION'))
    return FSRSScheduler(desired_retention=retention)


__all__ = ['Scheduler', 'FSRSScheduler', 'SM2Scheduler', 'get_scheduler', 'RATINGS']
