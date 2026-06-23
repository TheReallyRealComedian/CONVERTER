"""Backfill degenerierter Conversion-Titel aus der ersten #-Überschrift.

TITLE-FIX-Folge: PDF→Markdown-Decks beginnen mit einem Seiten-Marker
(``<!-- Seite 1 -->``); die alte Titel-Heuristik „erste Zeile" machte daraus
den Titel ``<!-- Seite 1 -->`` — in der Library und im list_conversions-Finder
sind die Decks dann per Titel ununterscheidbar. TITLE-FIX P1 lehrt beide POST-
Endpoints, bei degeneriertem Titel aus dem Content abzuleiten; dieses Script
zieht den BESTAND nach: jede Row mit degeneriertem Titel, deren Content eine
bessere Ableitung liefert, bekommt diesen Titel — der Content bleibt unberührt.

derive_title + _is_degenerate_title werden IMPORTIERT (nie reimplementiert),
damit Backfill und Live-Endpoints nicht auseinanderdriften können (Memory
reference_tag_vocab_central_gate_plus_backfill_script).

Dry-run ist Default und durchläuft denselben Code-Pfad mit Rollback am Ende —
die Vorhersage entspricht damit garantiert dem Apply. Idempotent: ein zweiter
--apply findet nichts mehr, weil die Row dann einen nicht-degenerierten Titel
trägt und gar nicht mehr als Kandidat auftaucht.

Konservativ („falsch ist schlimmer als der Marker"): geschrieben wird NUR, wenn
die Ableitung sich vom alten Titel unterscheidet UND nicht selbst degeneriert
ist (Content ohne Heading/Text ⇒ derive_title liefert ``Untitled`` ⇒ Skip).

Lauf im Container:
    docker compose exec markdown-converter python scripts/backfill_titles.py          # dry-run
    docker compose exec markdown-converter python scripts/backfill_titles.py --apply
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, or_  # noqa: E402

from app import app  # noqa: E402
from models import Conversion, db  # noqa: E402
from services.markdown_sections import derive_title, _is_degenerate_title  # noqa: E402


def _degenerate_candidates():
    """Rows whose title is degenerate. The SQL prefilter is a safe superset of
    ``_is_degenerate_title`` (trim-aware so leading whitespace can't hide a
    ``<!--`` marker or a blank); ``_is_degenerate_title`` is then the
    authoritative gate, so the query can only over-match, never miss."""
    prefilter = or_(
        Conversion.title.is_(None),
        func.trim(Conversion.title).like('<!--%'),
        func.lower(func.trim(Conversion.title)).in_(('', 'untitled', 'untitled markdown')),
    )
    rows = (Conversion.query.filter(prefilter)
            .order_by(Conversion.id).all())
    return [c for c in rows if _is_degenerate_title(c.title)]


def run(apply_changes):
    mode = 'APPLY' if apply_changes else 'DRY-RUN'
    rows = _degenerate_candidates()
    print(f'Titel-Backfill ({mode}) — degenerierte Titel: {len(rows)}')

    changes = 0
    for c in rows:
        new_title = derive_title(c.content)[:255]
        if new_title == c.title:
            continue  # nichts gewonnen — Skip
        if _is_degenerate_title(new_title):
            # Content liefert nichts Besseres (kein Heading/Text) — Marker behalten.
            print(f"  [id {c.id}] {c.title!r} → skip (Ableitung selbst degeneriert: {new_title!r})")
            continue
        changes += 1
        print(f"  [id {c.id}] {c.title!r} → {new_title!r}")
        c.title = new_title

    print(f'{changes} Row(s) zu re-titeln.')
    if apply_changes:
        db.session.commit()
        print('APPLY — Änderungen geschrieben.')
    else:
        db.session.rollback()
        print('DRY-RUN — nichts geschrieben. Echter Lauf: --apply')

    return changes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Degenerierte Conversion-Titel aus der ersten #-Überschrift nachziehen (TITLE-FIX).')
    parser.add_argument('--apply', action='store_true',
                        help='Änderungen wirklich schreiben (Default: dry-run).')
    args = parser.parse_args()
    with app.app_context():
        run(args.apply)
