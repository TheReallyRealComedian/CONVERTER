"""R4-LEARN Phase 3 — the swappable scheduler engines (FSRS default + SM-2).

Pure-logic tests on the scheduler interface: both engines behave through the
same ``new_card_state`` / ``apply_rating`` contract, FSRS math moves ``due``
forward and orders again<good<easy, ``again`` is a lapse, and the SM-2 fallback
produces a plausible future ``due`` behind the identical interface.
"""
from datetime import datetime, timezone

import pytest

from services.scheduler import (DEFAULT_DESIRED_RETENTION, FSRSScheduler,
                                RATINGS, SM2Scheduler, get_scheduler,
                                _parse_retention)

_STATE_KEYS = {'due', 'stability', 'difficulty', 'last_reviewed', 'reps', 'lapses'}


def _aware(dt):
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --- new_card_state: both engines return the FSRS-"new" shape ----------------

@pytest.mark.parametrize('sched', [FSRSScheduler(), SM2Scheduler()])
def test_new_card_state_shape(sched):
    s = sched.new_card_state()
    assert set(s) == _STATE_KEYS
    assert s['stability'] is None and s['difficulty'] is None
    assert s['last_reviewed'] is None
    assert s['reps'] == 0 and s['lapses'] == 0
    assert s['due'] is not None


@pytest.mark.parametrize('sched', [FSRSScheduler(), SM2Scheduler()])
def test_apply_rating_rejects_bad_rating(sched):
    with pytest.raises(ValueError):
        sched.apply_rating(sched.new_card_state(), 'bogus')


# --- FSRS math ---------------------------------------------------------------

def test_fsrs_new_card_first_rating_moves_due_forward():
    sched = FSRSScheduler()
    before = datetime.now(timezone.utc)
    out = sched.apply_rating(sched.new_card_state(), 'good')
    assert _aware(out['due']) > before          # due moved forward
    assert out['reps'] == 1
    assert out['lapses'] == 0
    assert out['stability'] is not None         # FSRS learned something
    assert out['last_reviewed'] is not None


def test_fsrs_again_is_a_lapse_and_ticks_counters():
    sched = FSRSScheduler()
    s = sched.apply_rating(sched.new_card_state(), 'good')   # into a reviewed state
    out = sched.apply_rating(s, 'again')
    assert out['reps'] == s['reps'] + 1
    assert out['lapses'] == s['lapses'] + 1     # again = lapse


def test_fsrs_interval_ordering_again_lt_good_lt_easy():
    sched = FSRSScheduler()
    new = sched.new_card_state()
    now = datetime.now(timezone.utc)

    def interval(rating):
        return _aware(sched.apply_rating(dict(new), rating)['due']) - now

    assert interval('again') < interval('good') < interval('easy')


def test_fsrs_good_after_graduation_is_multi_day():
    sched = FSRSScheduler()
    s = sched.apply_rating(sched.new_card_state(), 'good')
    s = sched.apply_rating(s, 'good')           # graduate
    now = datetime.now(timezone.utc)
    out = sched.apply_rating(s, 'good')
    assert _aware(out['due']) - now > _aware(out['last_reviewed']) - now  # in the future
    assert (_aware(out['due']) - now).days >= 1  # day-scale interval, not minutes


# --- SM-2 fallback behind the same interface ---------------------------------

def test_sm2_fallback_plausible_due_same_interface():
    sched = SM2Scheduler()
    now = datetime.now(timezone.utc)
    out = sched.apply_rating(sched.new_card_state(), 'good')
    assert set(out) == _STATE_KEYS              # identical contract
    assert _aware(out['due']) > now             # plausible future due
    assert out['reps'] == 1
    # again is a lapse here too
    lapsed = sched.apply_rating(out, 'again')
    assert lapsed['lapses'] == 1


def test_sm2_interval_ordering():
    sched = SM2Scheduler()
    new = sched.new_card_state()
    now = datetime.now(timezone.utc)

    def interval(rating):
        return _aware(sched.apply_rating(dict(new), rating)['due']) - now

    assert interval('again') < interval('good') <= interval('easy')


# --- config / factory --------------------------------------------------------

def test_get_scheduler_picks_engine_by_config(monkeypatch):
    monkeypatch.delenv('SCHEDULER_ENGINE', raising=False)
    assert isinstance(get_scheduler(), FSRSScheduler)       # default FSRS
    monkeypatch.setenv('SCHEDULER_ENGINE', 'sm2')
    assert isinstance(get_scheduler(), SM2Scheduler)
    monkeypatch.setenv('SCHEDULER_ENGINE', 'FSRS')          # case-insensitive
    assert isinstance(get_scheduler(), FSRSScheduler)


def test_parse_retention_clamps_and_defaults():
    assert _parse_retention('0.85') == 0.85
    assert _parse_retention(None) == DEFAULT_DESIRED_RETENTION
    assert _parse_retention('2.0') == DEFAULT_DESIRED_RETENTION   # out of (0,1)
    assert _parse_retention('garbage') == DEFAULT_DESIRED_RETENTION


def test_ratings_constant():
    assert RATINGS == ('again', 'hard', 'good', 'easy')
