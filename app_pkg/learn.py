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
from datetime import datetime, timezone

from flask import jsonify, request
from flask_login import current_user, login_required

from models import User, db

# Defaults double as the key whitelist: a settings key exists iff it is here.
LEARN_SETTINGS_DEFAULTS = {
    'ordering_mode': 'smart',  # 'smart' (R asc + interleave) | 'random'
}

ORDERING_MODES = ('smart', 'random')


def _valid_ordering_mode(value):
    return value if value in ORDERING_MODES else None


# Per-key validators: return the normalized value, or None for "invalid".
_VALIDATORS = {
    'ordering_mode': _valid_ordering_mode,
}


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


def order_due_cards(due_cards, mode, scheduler, rng=None, now=None):
    """Order the due queue for one session fetch.

    ``smart``: reviewed cards (stability set) by retrievability ascending with
    a random tiebreak; brand-new cards (stability NULL) shuffled and evenly
    interleaved — never front-loaded. ``random``: one full shuffle.
    """
    rng = rng or random.Random()
    now = now or datetime.now(timezone.utc)
    cards = list(due_cards)
    if mode == 'random':
        rng.shuffle(cards)
        return cards
    reviewed = [c for c in cards if c.review.stability is not None]
    fresh = [c for c in cards if c.review.stability is None]
    # Shuffle BEFORE the stable sort → equal sort keys keep the shuffled
    # order = the random tiebreak. py-fsrs computes R at day granularity, so
    # same-day/equal-stability ties are common — without the tiebreak the
    # queue would fall back to hidden insertion order.
    rng.shuffle(reviewed)
    reviewed.sort(key=lambda c: _review_sort_key(scheduler, c.review, now))
    rng.shuffle(fresh)
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
