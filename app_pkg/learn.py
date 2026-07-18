"""Learn settings + review-queue ordering (Sprint LEARN-UP).

Settings live in ONE JSON blob on the user row (``User.settings_json``, TEXT)
rather than typed columns: the sprint adds keys across phases (``ordering_mode``
now, daily limits and desired retention later), and one blob means one
migration for all of them. ``get_user_settings`` merges stored values over the
defaults and drops anything invalid, so a stale or hand-edited blob can never
break the review flow — the read side is lenient, the PUT endpoint is strict.

Ordering (``order_due_cards``) happens Python-side per fetch — the client
pulls the queue once and walks it. ``smart`` (default) sorts previously
reviewed cards by FSRS retrievability ascending (the shakiest first) with a
random tiebreak, shuffles brand-new cards, and interleaves them evenly into
the review stream. ``random`` is one full shuffle. Either way the old hidden
creation-date order is structurally gone.
"""
import json
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import jsonify, request
from flask_login import current_user, login_required

from models import Card, Review, User, db

# The learn features count and bucket per USER-local day (single-user app,
# Oliver sits in Berlin) — a UTC day would reset the daily limits mid-evening.
# Shared by the P3 daily caps and the P4 stats bucketing via local_day_bounds.
LOCAL_TZ = ZoneInfo('Europe/Berlin')

# Defaults double as the key whitelist: a settings key exists iff it is here.
LEARN_SETTINGS_DEFAULTS = {
    'ordering_mode': 'smart',  # 'smart' (R asc + interleave) | 'random'
    'daily_new_limit': 10,     # new cards introduced per local day
    'daily_review_limit': 200, # reviews per local day
}

ORDERING_MODES = ('smart', 'random')
MAX_DAILY_LIMIT = 10000


def _valid_ordering_mode(value):
    return value if value in ORDERING_MODES else None


def _valid_limit(value):
    # bool is an int subclass — reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= MAX_DAILY_LIMIT else None


# Per-key validators: return the normalized value, or None for "invalid".
_VALIDATORS = {
    'ordering_mode': _valid_ordering_mode,
    'daily_new_limit': _valid_limit,
    'daily_review_limit': _valid_limit,
}


def local_day_bounds(now=None):
    """Aware-UTC ``[start, end)`` of the Berlin-local day containing ``now``."""
    if now is None:
        now = datetime.now(timezone.utc)
    local = now.astimezone(LOCAL_TZ)
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return (start_local.astimezone(timezone.utc),
            (start_local + timedelta(days=1)).astimezone(timezone.utc))


def get_user_settings(user):
    """Effective learn settings for ``user`` — defaults overlaid with the
    stored blob; unknown keys and invalid values are silently dropped."""
    settings = dict(LEARN_SETTINGS_DEFAULTS)
    raw = getattr(user, 'settings_json', None)
    if not raw:
        return settings
    try:
        stored = json.loads(raw)
    except ValueError:
        return settings
    if not isinstance(stored, dict):
        return settings
    for key, validator in _VALIDATORS.items():
        if key in stored:
            value = validator(stored[key])
            if value is not None:
                settings[key] = value
    return settings


def count_done_today(user_id, now=None):
    """``(reviews_done, new_done)`` for the current Berlin-local day.

    Feeds the daily caps: a cap must count what was ALREADY studied today,
    not merely trim the displayed queue. Distinct cards, not ratings — an
    again-loop on one card burns one review slot, like Anki.

    Derivation: every rating bumps ``last_reviewed``, so the cards touched
    today are exactly those with ``last_reviewed`` in today's bounds — only
    that small set has its ``rating_history`` parsed. A card whose FIRST-ever
    history entry falls in today counts as "new introduced today"; any other
    card touched today counts as "review done today". Fallback for a missing/
    corrupt history: ``reps == 1`` (exactly one rating ever, and it was today)
    classifies as new.
    """
    start, end = local_day_bounds(now)
    # SQLite holds naive UTC wall-clock — compare with naive bounds.
    start_naive, end_naive = (b.replace(tzinfo=None) for b in (start, end))
    rows = (Review.query.join(Card, Review.card_id == Card.id)
            .filter(Card.user_id == user_id,
                    Review.last_reviewed >= start_naive,
                    Review.last_reviewed < end_naive)
            .all())
    reviews_done = new_done = 0
    for rev in rows:
        first_at = None
        if rev.rating_history:
            try:
                history = json.loads(rev.rating_history)
                if isinstance(history, list) and history:
                    first_at = datetime.fromisoformat(history[0]['reviewed_at'])
            except (ValueError, TypeError, KeyError):
                first_at = None
        if first_at is not None:
            if first_at.tzinfo is None:  # legacy naive = UTC wall-clock
                first_at = first_at.replace(tzinfo=timezone.utc)
            introduced_today = start <= first_at.astimezone(timezone.utc) < end
        else:
            introduced_today = (rev.reps or 0) == 1
        if introduced_today:
            new_done += 1
        else:
            reviews_done += 1
    return reviews_done, new_done


def order_due_cards(due_cards, mode, scheduler, rng=None, now=None,
                    review_budget=None, new_budget=None):
    """Order the due queue for one session fetch, then apply the daily caps.

    Ordering — ``smart``: reviewed cards (stability set) by retrievability
    ascending with a random tiebreak; brand-new cards (stability NULL)
    shuffled and evenly interleaved — never front-loaded. ``random``: both
    pools shuffled, merged, shuffled again = one full shuffle.

    Caps (LEARN-UP P3, ``None`` = uncapped) apply AFTER ordering, so a cap
    keeps the shakiest N (smart) or a random N (random) — never the oldest N.
    ``new_budget`` additionally respects the review cap: new cards only fill
    the headroom the review load leaves under ``review_budget``, so a day
    drowning in due reviews introduces nothing new (sane default).
    """
    rng = rng or random.Random()
    now = now or datetime.now(timezone.utc)
    cards = list(due_cards)
    reviewed = [c for c in cards if c.review.stability is not None]
    fresh = [c for c in cards if c.review.stability is None]
    # Shuffle BEFORE the stable sort → equal sort keys keep the shuffled
    # order = the random tiebreak. py-fsrs computes R at day granularity, so
    # same-day/equal-stability ties are common — without the tiebreak the
    # queue would fall back to hidden insertion order.
    rng.shuffle(reviewed)
    if mode != 'random':
        reviewed.sort(key=lambda c: _review_sort_key(scheduler, c.review, now))
    rng.shuffle(fresh)

    headroom = None
    if review_budget is not None:
        reviewed = reviewed[:max(0, review_budget)]
        headroom = max(0, review_budget - len(reviewed))
    if new_budget is not None:
        cap = new_budget if headroom is None else min(new_budget, headroom)
        fresh = fresh[:max(0, cap)]

    if mode == 'random':
        merged = reviewed + fresh
        rng.shuffle(merged)
        return merged
    return _interleave_evenly(reviewed, fresh)


def _review_sort_key(scheduler, review, now):
    r = scheduler.retrievability({
        'due': review.due,
        'stability': review.stability,
        'difficulty': review.difficulty,
        'last_reviewed': review.last_reviewed,
    }, now)
    if r is None:
        # Engine without an R concept (SM-2) → degrade to due-ascending;
        # the pre-sort shuffle still tiebreaks equal timestamps.
        due = review.due or now
        return (1, due.timestamp())
    return (0, r)


def _interleave_evenly(reviewed, fresh):
    """Spread ``fresh`` evenly through ``reviewed`` (both keep their order).

    Fractional-position merge: each stream is spread over (0,1) and the two
    are merged by position — works for any size ratio. Reviews win ties, so a
    new card never lands at the very front while reviews remain.
    """
    if not reviewed or not fresh:
        return reviewed + fresh
    keyed = [((i + 0.5) / len(reviewed), 0, c) for i, c in enumerate(reviewed)]
    keyed += [((i + 0.5) / len(fresh), 1, c) for i, c in enumerate(fresh)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [c for _, _, c in keyed]


def register(app):
    @app.route('/api/learn/settings', methods=['GET'])
    @login_required
    def api_get_learn_settings():
        return jsonify(get_user_settings(current_user))

    @app.route('/api/learn/settings', methods=['PUT'])
    @login_required
    def api_put_learn_settings():
        # Partial update: provided keys are validated strictly (unknown key or
        # invalid value → 400, nothing written), the rest keeps its current
        # effective value. The stored blob is always the full merged dict.
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400
        unknown = set(data) - set(_VALIDATORS)
        if unknown:
            return jsonify({'error': f"Unbekannte Einstellung: {', '.join(sorted(unknown))}."}), 400
        settings = get_user_settings(current_user)
        for key, value in data.items():
            validated = _VALIDATORS[key](value)
            if validated is None:
                return jsonify({'error': f"Ungültiger Wert für '{key}'."}), 400
            settings[key] = validated
        user = db.session.get(User, current_user.id)
        user.settings_json = json.dumps(settings)
        db.session.commit()
        return jsonify(settings)
