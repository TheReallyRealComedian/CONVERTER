"""LEARN-BACK — ``flask reset-collection``: vergiftete Karten zuruecksetzen.

Die Bewertungen vor LEARN-RATE trugen eine andere Semantik ("Schwer" hiess
"kaum gewusst"), also ist die daraus gewachsene Stabilitaet erfunden. Das
Kommando setzt die Review-Zeilen einer Sammlung auf die KANONISCHE
"neu"-Definition zurueck (``initial_review_state()`` — nicht nachgebaut) und
leert die ``rating_history``.

Der wichtigste Test hier ist ``test_dry_run_writes_nothing``: das Kommando ist
destruktiv und nicht umkehrbar, der Dry-run ist der einzige Schutz vor dem
Vertipper. Danach: Idempotenz (Auswahl ``stability IS NOT NULL`` ⇒ zweiter Lauf
findet 0), Scope-Isolation und dass die zurueckgesetzte Karte UEBERALL als neu
gilt (``order_due_cards``-``fresh``-Topf, ``maturity_counts``).
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from app_pkg.learn import maturity_counts, order_due_cards
from models import Card, Collection, Review, Tag, User, db
from services.scheduler import FSRSScheduler
from services.scheduler.base import initial_review_state

RATED_FIELDS = ('due', 'stability', 'difficulty', 'last_reviewed', 'reps',
                'lapses', 'rating_history')


# --- helpers -----------------------------------------------------------------

def _make_user(app, username='alice'):
    with app.app_context():
        user = User(username=username)
        user.set_password('hunter2hunter2')
        db.session.add(user)
        db.session.commit()
        return user.id


def _make_collection(app, user_id, name):
    with app.app_context():
        coll = Collection(user_id=user_id, name=name)
        db.session.add(coll)
        db.session.commit()
        return coll.id


def _make_card(app, user_id, collection_ids=(), rated=True, tag_name=None):
    """A card in the given collections. ``rated`` = the poisoned shape: a grown
    stability/difficulty, a due weeks out, and a rating history."""
    with app.app_context():
        now = datetime.now(timezone.utc)
        card = Card(user_id=user_id, type='atomic', front='Q', back='A')
        # Add BEFORE appending to the M2M sides — Collection.cards is a dynamic
        # backref, and appending from a session-less card drops the row (same
        # reason POST /api/cards adds the card first, app_pkg/cards.py).
        db.session.add(card)
        for cid in collection_ids:
            card.collections.append(db.session.get(Collection, cid))
        if tag_name:
            tag = Tag.query.filter_by(user_id=user_id, name=tag_name).first()
            if tag is None:
                tag = Tag(user_id=user_id, name=tag_name)
                db.session.add(tag)
            card.tags.append(tag)
        if rated:
            card.review = Review(
                due=now + timedelta(days=42),
                stability=42.5,
                difficulty=6.25,
                last_reviewed=now - timedelta(days=5),
                reps=3,
                lapses=1,
                rating_history=json.dumps([
                    {'rating': 'hard', 'reviewed_at': (now - timedelta(days=5)).isoformat()},
                ]),
            )
        else:
            card.review = Review(due=now, reps=0, lapses=0)
        db.session.commit()
        return card.id


def _snapshot(app, card_id):
    with app.app_context():
        review = Review.query.filter_by(card_id=card_id).one()
        return tuple(getattr(review, field) for field in RATED_FIELDS)


def _invoke(app, *args):
    return app.test_cli_runner().invoke(args=['reset-collection', *args])


@pytest.fixture
def poisoned(app):
    """One collection ('TCE'), two rated cards + one never-rated card, plus a
    rated card in a DIFFERENT collection that shares the same tag."""
    user_id = _make_user(app)
    target_id = _make_collection(app, user_id, 'TCE')
    other_id = _make_collection(app, user_id, 'Andere')
    return {
        'user_id': user_id,
        'target_id': target_id,
        'other_id': other_id,
        'rated': [_make_card(app, user_id, [target_id], tag_name='tce'),
                  _make_card(app, user_id, [target_id], tag_name='tce')],
        'unrated': _make_card(app, user_id, [target_id], rated=False),
        'outside': _make_card(app, user_id, [other_id], tag_name='tce'),
    }


# --- the reset itself --------------------------------------------------------

def test_apply_resets_row_to_canonical_new_state(app, poisoned):
    result = _invoke(app, str(poisoned['target_id']), '--apply')
    assert result.exit_code == 0, result.output
    assert '2 Karten zurueckgesetzt' in result.output

    canonical = initial_review_state()
    with app.app_context():
        for card_id in poisoned['rated']:
            review = Review.query.filter_by(card_id=card_id).one()
            for field, value in canonical.items():
                if field == 'due':
                    continue  # compared below (a timestamp, not a constant)
                assert getattr(review, field) == value, field
            assert review.rating_history is None
            # due = "jetzt" in NAIVE UTC (the column convention). Doubles as the
            # guard on the `_naive_utc` pass-through: the scheduler hands back
            # an aware datetime and SQLite drops the tzinfo silently, so a zone
            # other than UTC would land here as wall-clock — a Berlin-zoned
            # `initial_review_state()` would show up as a 2 h drift.
            assert review.due.tzinfo is None
            drift = abs((review.due - datetime.now(timezone.utc).replace(tzinfo=None))
                        .total_seconds())
            assert drift < 120


def test_dry_run_writes_nothing(app, poisoned):
    """The single most important test: without --apply the row stays byte-equal."""
    before = [_snapshot(app, cid) for cid in poisoned['rated']]
    result = _invoke(app, str(poisoned['target_id']))
    assert result.exit_code == 0, result.output
    assert 'DRY-RUN' in result.output
    assert 'Bewertete Karten: 2 von 3' in result.output
    assert 'Nichts geschrieben' in result.output
    assert [_snapshot(app, cid) for cid in poisoned['rated']] == before


def test_second_apply_is_a_no_op(app, poisoned):
    assert _invoke(app, 'TCE', '--apply').exit_code == 0
    result = _invoke(app, 'TCE', '--apply')
    assert result.exit_code == 0, result.output
    # Selection is `stability IS NOT NULL` → the reset cards are out of scope.
    assert 'Bewertete Karten: 0 von 3' in result.output
    assert 'Nichts zu tun' in result.output


def test_unrated_card_in_target_is_untouched(app, poisoned):
    before = _snapshot(app, poisoned['unrated'])
    assert _invoke(app, 'TCE', '--apply').exit_code == 0
    assert _snapshot(app, poisoned['unrated']) == before


def test_other_collections_are_untouched_even_sharing_a_tag(app, poisoned):
    before = _snapshot(app, poisoned['outside'])
    assert _invoke(app, 'TCE', '--apply').exit_code == 0
    assert _snapshot(app, poisoned['outside']) == before


def test_card_in_several_collections_is_reset_and_the_overlap_is_named(app, poisoned):
    shared = _make_card(app, poisoned['user_id'],
                        [poisoned['target_id'], poisoned['other_id']])
    result = _invoke(app, 'TCE')
    assert 'Davon auch in anderen Sammlungen: "Andere" (1)' in result.output

    assert _invoke(app, 'TCE', '--apply').exit_code == 0
    with app.app_context():
        assert Review.query.filter_by(card_id=shared).one().stability is None


# --- resolution + failure modes ----------------------------------------------

def test_resolves_by_name_and_by_id(app, poisoned):
    by_name = _invoke(app, 'TCE')
    by_id = _invoke(app, str(poisoned['target_id']))
    assert by_name.exit_code == 0 and by_id.exit_code == 0
    assert 'Bewertete Karten: 2 von 3' in by_name.output
    assert by_id.output == by_name.output


def test_unknown_collection_errors_and_writes_nothing(app, poisoned):
    before = [_snapshot(app, cid) for cid in poisoned['rated']]
    result = _invoke(app, 'Gibt-Es-Nicht', '--apply')
    assert result.exit_code != 0
    assert 'nicht gefunden' in result.output
    assert [_snapshot(app, cid) for cid in poisoned['rated']] == before


def test_ambiguous_name_across_users_errors_instead_of_guessing(app, poisoned):
    bob = _make_user(app, username='bob')
    _make_collection(app, bob, 'TCE')
    before = [_snapshot(app, cid) for cid in poisoned['rated']]
    result = _invoke(app, 'TCE', '--apply')
    assert result.exit_code != 0
    assert 'mehrdeutig' in result.output
    assert [_snapshot(app, cid) for cid in poisoned['rated']] == before


# --- "counts as new" everywhere ----------------------------------------------

def test_reset_card_counts_as_new_downstream(app, poisoned):
    assert _invoke(app, 'TCE', '--apply').exit_code == 0
    with app.app_context():
        # maturity_counts: 2 reset + 1 never-rated = 3 neu, the outside card
        # stays 'reif'/'jung' (rated).
        counts = maturity_counts(poisoned['user_id'])
        assert counts['neu'] == 3

        cards = (Card.query.join(Card.review)
                 .filter(Card.id.in_(poisoned['rated'])).all())
        scheduler = FSRSScheduler()
        # `stability IS NULL` puts them in the `fresh` pool, so the NEW budget
        # governs them — with 0 new allowed the queue comes back empty even
        # though the review budget is wide open.
        assert order_due_cards(cards, 'smart', scheduler,
                               review_budget=10, new_budget=0) == []
        assert len(order_due_cards(cards, 'smart', scheduler,
                                   review_budget=10, new_budget=2)) == 2
