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
random tiebreak, keeps brand-new cards in creation order (LEARN-QUEUE), and
interleaves them evenly into the review stream. ``random`` is one full
shuffle. For REVIEWS the old hidden creation-date order is structurally gone
(that was LEARN-UP's point); for NEW cards the author's order IS the
didactics, so it is deliberately preserved.
"""
import json
import os
import random
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import jsonify, request
from flask_login import current_user, login_required
from sqlalchemy import text

from models import Card, Review, User, db
from services.scheduler import _parse_retention
from services.scheduler.fsrs_scheduler import simulate_workload

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


def local_day_end(now=None, days_ahead=0):
    """Aware-UTC end of the Berlin-local day ``days_ahead`` days after the one
    containing ``now`` (0 = today's end, 1 = tomorrow's end, …).

    LEARN-MORE anchors on this in two places: ``?ahead=<n>`` moves the
    review-state due horizon here, and the response's ``day_end`` field hands
    today's boundary to the client so the JS never does timezone arithmetic —
    the wall-clock day math (DST-safe via ZoneInfo) lives server-side only.
    """
    start, _end = local_day_bounds(now)
    start_local = start.astimezone(LOCAL_TZ)
    return (start_local + timedelta(days=1 + days_ahead)).astimezone(timezone.utc)


def write_settings_keys(user, updates):
    """Merge ``updates`` into the user's raw settings blob ATOMICALLY, preserving
    the rest.

    ``User.settings_json`` is shared by features with disjoint key spaces
    (learn keys flat, DOC-API under the ``document_api`` namespace key). Every
    writer must go through this merge: a plain ``json.dumps(own_keys)`` would
    silently drop the other feature's settings on each save.

    LOST-UPDATE: the merge is ONE UPDATE with SQLite's ``json_patch`` (RFC
    7396) — it happens in the database under its write lock, so two writers
    of different namespaces cannot overwrite each other (the former in-memory
    ``dict.update`` over the loaded row lost a write in 125 of 200 concurrent
    rounds and VANISHED a whole namespace in 30 of 30 first writes). No read,
    no version, no retry. Two semantic differences to ``dict.update``,
    deliberate and documented:

    * a ``None`` value DELETES its key (RFC 7396 null) instead of storing
      null — observably the same for every reader (stored null = "invalid →
      default", absent = "absent → default"); no caller passes None today.
    * objects merge RECURSIVELY: ``{'document_api': {...}}`` overlays the
      stored namespace instead of replacing it, so sub-keys not in the update
      survive. Both callers write their FULL key set, so the known keys end
      up identical; only unknown sub-keys now persist where they were wiped
      (readers drop unknown keys anyway).

    Lenient on a missing/corrupt/non-object blob (starts fresh, as before).
    ``user`` must be a session-attached User (the in-memory attribute is
    expired so it re-reads the merged blob). Does NOT commit — the caller
    owns the transaction.
    """
    db.session.execute(
        text('UPDATE "user" SET settings_json = json_patch('
             "CASE WHEN json_valid(settings_json) AND json_type(settings_json) = 'object' "
             "THEN settings_json ELSE '{}' END, :updates) WHERE id = :uid"),
        {'updates': json.dumps(updates), 'uid': user.id})
    db.session.expire(user, ['settings_json'])


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


def capped_session_counts(n_reviewed, n_fresh, review_budget=None, new_budget=None):
    """Shared cap arithmetic (``None`` = uncapped): how many reviews and new
    cards today's session shows. ``order_due_cards`` (P3) slices with these
    numbers and the stats' "Heute" block (P4) reports them — ONE source, so
    the stats can never drift from the launcher.
    """
    if review_budget is None:
        shown_reviews = n_reviewed
    else:
        shown_reviews = min(n_reviewed, max(0, review_budget))
    if new_budget is None:
        shown_new = n_fresh
    else:
        cap = new_budget
        if review_budget is not None:
            # New cards only fill the headroom the review load leaves.
            cap = min(cap, max(0, review_budget - shown_reviews))
        shown_new = min(n_fresh, max(0, cap))
    return shown_reviews, shown_new


def order_due_cards(due_cards, mode, scheduler, rng=None, now=None,
                    review_budget=None, new_budget=None):
    """Order the due queue for one session fetch, then apply the daily caps.

    Ordering — ``smart``: reviewed cards (stability set) by retrievability
    ascending with a random tiebreak; brand-new cards (stability NULL) in
    CREATION order and evenly interleaved — never front-loaded. ``random``:
    both pools shuffled, merged, shuffled again = one full shuffle.

    Caps (LEARN-UP P3, ``None`` = uncapped) apply AFTER ordering, so the
    review cap keeps the shakiest N (smart) or a random N (random) — never
    the oldest N. The new-card cap in ``smart`` mode DOES take the oldest N,
    which is the point of the creation order: chapter 4 finishes before
    chapter 5 starts. ``new_budget`` additionally respects the review cap:
    new cards only fill the headroom the review load leaves under
    ``review_budget``, so a day drowning in due reviews introduces nothing
    new (sane default).
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
    if mode == 'random':
        rng.shuffle(fresh)
    else:
        reviewed.sort(key=lambda c: _review_sort_key(scheduler, c.review, now))
        # LEARN-QUEUE: new cards keep the order their author wrote them in — a
        # shuffle here would re-roll the queue on every fetch and throw away
        # the didactic sequence. `id` is the tiebreak because a batch write
        # collides on `created_at`; without it the order would not be total,
        # and a non-total order is not reproducible either.
        fresh.sort(key=_fresh_sort_key)

    n_reviews, n_new = capped_session_counts(len(reviewed), len(fresh),
                                             review_budget, new_budget)
    reviewed = reviewed[:n_reviews]
    fresh = fresh[:n_new]

    if mode == 'random':
        merged = reviewed + fresh
        rng.shuffle(merged)
        return merged
    return _interleave_evenly(reviewed, fresh)


def _fresh_sort_key(card):
    """Creation order for brand-new cards: ``created_at`` asc, ``id`` asc.

    ``created_at`` is nullable (defaulted, but a hand-written row could miss
    it) — a NULL sorts last via the leading flag, and the flag also keeps the
    tuple comparison from ever putting ``None`` next to a datetime.
    """
    created = getattr(card, 'created_at', None)
    return (created is None, created, card.id)


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


# --- P4: stats building blocks -----------------------------------------------

MATURE_INTERVAL_DAYS = 21          # Anki-Konvention: reif ab 21-Tage-Intervall
RETENTION_WINDOW_DAYS = 30
FORECAST_DAYS = 28


def forecast_buckets(user_id, days=FORECAST_DAYS, now=None):
    """Future-due forecast: per-Berlin-day counts + the overdue backlog.

    ``overdue`` = due <= now (exactly the raw queue definition). ``days[0]``
    is the REST of today (due after now, before local midnight), then one
    bucket per following Berlin day. Bucketing happens in Python via
    ``local_day_bounds``/astimezone — DST-safe, and the fetched range is
    small (only dues inside the window).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    start, _end = local_day_bounds(now)
    window_end = (start + timedelta(days=days)).replace(tzinfo=None)
    now_naive = now.astimezone(timezone.utc).replace(tzinfo=None)

    overdue = (Review.query.join(Card, Review.card_id == Card.id)
               .filter(Card.user_id == user_id, Review.due <= now_naive)
               .count())
    rows = (db.session.query(Review.due)
            .join(Card, Review.card_id == Card.id)
            .filter(Card.user_id == user_id,
                    Review.due > now_naive, Review.due < window_end)
            .all())
    start_local_date = start.astimezone(LOCAL_TZ).date()
    buckets = [0] * days
    for (due,) in rows:
        due_local = due.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
        idx = (due_local.date() - start_local_date).days
        if 0 <= idx < days:
            buckets[idx] += 1
    return {
        'overdue': overdue,
        'days': [{'date': (start_local_date + timedelta(days=i)).isoformat(),
                  'count': buckets[i]} for i in range(days)],
    }


def maturity_counts(user_id):
    """Reifegrad-Zähler. Classification (LEARN-UP P4):

    * ``neu``  — stability NULL (never rated).
    * ``jung`` — rated, interval ``due − last_reviewed`` < 21 days. The
      schema persists no FSRS learning state (the step ramp is collapsed —
      see fsrs_scheduler), so Anki's "lernend" bucket collapses into jung.
    * ``reif`` — interval ≥ 21 days (Anki convention).

    Fallback when ``due``/``last_reviewed`` is missing: ``stability`` as the
    interval proxy (at R=0.9 the FSRS interval equals the stability).
    """
    rows = (db.session.query(Review.stability, Review.due, Review.last_reviewed)
            .join(Card, Review.card_id == Card.id)
            .filter(Card.user_id == user_id)
            .all())
    counts = {'neu': 0, 'jung': 0, 'reif': 0}
    for stability, due, last_reviewed in rows:
        if stability is None:
            counts['neu'] += 1
            continue
        if due is not None and last_reviewed is not None:
            interval_days = (due - last_reviewed).total_seconds() / 86400.0
        else:
            interval_days = stability
        counts['reif' if interval_days >= MATURE_INTERVAL_DAYS else 'jung'] += 1
    return counts


def true_retention(user_id, window_days=RETENTION_WINDOW_DAYS, now=None):
    """Ist-Retention aus ``rating_history`` über ein Fenster.

    Counted: per card per Berlin day the chronologically FIRST rating
    (later same-day ratings are relearn noise); ``again`` = fail, else pass.
    The card's introduction day (its first-ever rating) is EXCLUDED — that
    answer is the "new" slot, not a recall of something scheduled. Returns
    ``{'pass': p, 'fail': f, 'rate': p/(p+f) | None}``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)
    rows = (db.session.query(Review.rating_history)
            .join(Card, Review.card_id == Card.id)
            .filter(Card.user_id == user_id,
                    Review.rating_history.isnot(None))
            .all())
    passed = failed = 0
    for (raw,) in rows:
        try:
            history = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(history, list) or len(history) < 2:
            continue  # nur die Intro-Bewertung → nichts zu zählen
        per_day = {}
        first_day = None
        for i, entry in enumerate(history):
            try:
                at = datetime.fromisoformat(entry['reviewed_at'])
            except (KeyError, TypeError, ValueError):
                continue
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
            day = at.astimezone(LOCAL_TZ).date()
            if i == 0:
                first_day = day
            per_day.setdefault(day, (at, entry.get('rating')))
            if at < per_day[day][0]:
                per_day[day] = (at, entry.get('rating'))
        for day, (at, rating) in per_day.items():
            if day == first_day or at < window_start:
                continue
            if rating == 'again':
                failed += 1
            else:
                passed += 1
    total = passed + failed
    return {'pass': passed, 'fail': failed,
            'rate': (passed / total) if total else None}


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
        # Merge-write: the blob also carries other features' namespaces
        # (DOC-API) — a plain dumps(settings) would drop them (see
        # write_settings_keys).
        write_settings_keys(user, settings)
        db.session.commit()
        return jsonify(settings)

    @app.route('/api/learn/stats', methods=['GET'])
    @login_required
    def api_learn_stats():
        # The four LEARN-UP stats. "Heute" reuses the P3 building blocks
        # (count_done_today + capped_session_counts + get_user_settings), so
        # its numbers are BY CONSTRUCTION the launcher's numbers — never a
        # second computation that could drift. desired_retention is the
        # read-only env value the real scheduler runs with (surfacing it
        # editable would require the scheduler to read per-user settings =
        # the contract rework this sprint forbids).
        now = datetime.now(timezone.utc)
        now_naive = now.replace(tzinfo=None)
        settings = get_user_settings(current_user)
        reviews_done, new_done = count_done_today(current_user.id, now=now)
        pools = (db.session.query(Review.stability)
                 .join(Card, Review.card_id == Card.id)
                 .filter(Card.user_id == current_user.id, Review.due <= now_naive)
                 .all())
        n_fresh = sum(1 for (s,) in pools if s is None)
        reviews_due, new_available = capped_session_counts(
            len(pools) - n_fresh, n_fresh,
            review_budget=max(0, settings['daily_review_limit'] - reviews_done),
            new_budget=max(0, settings['daily_new_limit'] - new_done))
        return jsonify({
            'today': {
                'reviews_due': reviews_due,
                'new_available': new_available,
                'reviews_done': reviews_done,
                'new_done': new_done,
            },
            'forecast': forecast_buckets(current_user.id, now=now),
            'maturity': maturity_counts(current_user.id),
            'retention': {
                **true_retention(current_user.id, now=now),
                'window_days': RETENTION_WINDOW_DAYS,
                'desired': _parse_retention(os.environ.get('FSRS_DESIRED_RETENTION')),
            },
        })

    @app.route('/api/learn/simulate', methods=['GET'])
    @login_required
    def api_learn_simulate():
        # Workload-Simulator (P4): hypothetical retention as what-if INPUT —
        # a projection only, the real scheduler never sees this value.
        try:
            retention = float(request.args.get('retention', ''))
        except ValueError:
            return jsonify({'error': "Parameter 'retention' muss eine Zahl sein."}), 400
        if not 0.5 <= retention <= 0.99:
            return jsonify({'error': "Parameter 'retention' muss zwischen 0.5 und 0.99 liegen."}), 400
        raw_new = request.args.get('new_per_day')
        if raw_new is None:
            new_per_day = get_user_settings(current_user)['daily_new_limit']
        else:
            try:
                new_per_day = int(raw_new)
            except ValueError:
                return jsonify({'error': "Parameter 'new_per_day' muss int sein."}), 400
            if not 0 <= new_per_day <= 1000:
                return jsonify({'error': "Parameter 'new_per_day' muss zwischen 0 und 1000 liegen."}), 400
        reviews_per_day = simulate_workload(retention, new_per_day)
        return jsonify({
            'retention': retention,
            'new_per_day': new_per_day,
            'reviews_per_day': round(reviews_per_day, 1),
            'estimate': True,  # Kohorten-Schätzung, keine Präzision
        })
