"""R4-LEARN Phase 3 — the swappable scheduler engines (FSRS default + SM-2).

Pure-logic tests on the scheduler interface: both engines behave through the
same ``new_card_state`` / ``apply_rating`` contract, FSRS math moves ``due``
forward and orders again<good<easy, ``again`` is a lapse, and the SM-2 fallback
produces a plausible future ``due`` behind the identical interface.
"""
from datetime import datetime, timedelta, timezone

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


# --- LEARN-STEP: one learning step — a correct answer ends the day -----------
# Threshold semantics, not exact minutes: "< 12 h" == the card comes back
# today, "> 12 h" == it is done for the day. The exact sub-day values come
# from the library and may drift slightly on a bump.

_SAME_DAY = timedelta(hours=12)


def _first_rating_interval(sched, rating):
    now = datetime.now(timezone.utc)
    return _aware(sched.apply_rating(sched.new_card_state(), rating)['due']) - now


@pytest.mark.parametrize('rating', ['good', 'easy'])
def test_fsrs_new_card_correct_answer_is_done_for_today(rating):
    # THE LEARN-STEP regression insurance: with py-fsrs' inherited 2-step
    # default a new card answered "good" came back after 10 minutes — every
    # new card was systematically shown twice on day one.
    assert _first_rating_interval(FSRSScheduler(), rating) > _SAME_DAY


@pytest.mark.parametrize('rating', ['again', 'hard'])
def test_fsrs_new_card_missed_answer_returns_today(rating):
    # A flubbed new card SHOULD come back within the same session.
    assert _first_rating_interval(FSRSScheduler(), rating) < _SAME_DAY


def test_fsrs_graduated_again_returns_today():
    # Relearning unchanged: a lapsed graduated card comes back today.
    sched = FSRSScheduler()
    s = sched.apply_rating(sched.new_card_state(), 'good')
    now = datetime.now(timezone.utc)
    assert _aware(sched.apply_rating(s, 'again')['due']) - now < _SAME_DAY


@pytest.mark.parametrize('rating', ['hard', 'good', 'easy'])
def test_fsrs_graduated_non_again_stays_multi_day(rating):
    # Graduated cards were already correct before LEARN-STEP — pin that they
    # stay multi-day even for the youngest possible graduate (one 'good').
    sched = FSRSScheduler()
    s = sched.apply_rating(sched.new_card_state(), 'good')
    now = datetime.now(timezone.utc)
    assert _aware(sched.apply_rating(s, rating)['due']) - now > _SAME_DAY


def test_fsrs_step_config_sentinel():
    # Sentinel (cf. the nh3/Flask-WTF sentinels): exactly ONE 10-min learning
    # step and ONE 10-min relearning step, configured explicitly. The original
    # bug was silently inheriting py-fsrs' (1 min, 10 min) default — a bump
    # that changes step handling must fail HERE, loudly, instead of silently
    # shifting the learning behaviour.
    engine = FSRSScheduler()._engine
    assert engine.learning_steps == (timedelta(minutes=10),)
    assert engine.relearning_steps == (timedelta(minutes=10),)


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


# --- retrievability (LEARN-UP): FSRS exposes R, SM-2 stays None --------------

def _reviewed_state(stability, days_ago, now):
    return {
        'due': now,
        'stability': stability,
        'difficulty': 5.0,
        'last_reviewed': now - timedelta(days=days_ago),
    }


def test_fsrs_retrievability_at_stability_is_090():
    # By construction of the FSRS curve, R(t=S) == 0.9 exactly.
    now = datetime.now(timezone.utc)
    r = FSRSScheduler().retrievability(_reviewed_state(10.0, 10, now), now)
    assert r == pytest.approx(0.9)


def test_fsrs_retrievability_decreases_with_elapsed():
    now = datetime.now(timezone.utc)
    sched = FSRSScheduler()
    r_fresh = sched.retrievability(_reviewed_state(10.0, 2, now), now)
    r_stale = sched.retrievability(_reviewed_state(10.0, 30, now), now)
    assert 0.0 < r_stale < r_fresh <= 1.0


def test_fsrs_retrievability_day_granular_same_day_is_one():
    # py-fsrs truncates elapsed to whole days → reviewed-today reads 1.0.
    # This tie behaviour is why the smart ordering needs a random tiebreak.
    now = datetime.now(timezone.utc)
    state = {'due': now, 'stability': 10.0, 'difficulty': 5.0,
             'last_reviewed': now - timedelta(hours=3)}
    assert FSRSScheduler().retrievability(state, now) == 1.0


def test_fsrs_retrievability_none_for_new_card():
    sched = FSRSScheduler()
    assert sched.retrievability(sched.new_card_state()) is None


def test_sm2_retrievability_is_always_none():
    # SM-2 reuses `stability` as interval-days — no forgetting curve, so the
    # base-class default (None) must win even for a "reviewed" state.
    now = datetime.now(timezone.utc)
    assert SM2Scheduler().retrievability(_reviewed_state(6.0, 3, now), now) is None


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
