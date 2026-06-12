"""One-off-Bereinigung des Tag-Bestands nach der R2-E-Normalisierungs-Härtung.

Die NL1/NL2-Newsletter haben Tags mit Markdown-Artefakten in die DB gespült
('** [anthropic', '** ai-agenten', …). Tag.normalize_name (models.py) härtet
seit R2-E jeden Schreibpfad — dieses Script zieht den BESTAND nach: jeder
Tag-Name, der sich unter der Normalisierung ändern würde, wird umbenannt;
kollidiert das Ergebnis mit einem bestehenden Tag desselben Users, werden die
Junction-Rows (conversion_tags UND highlight_tags) umgehängt und das Duplikat
gelöscht; normalisiert ein Name zu '', wird der Tag samt Junction-Rows
gelöscht (gleiche Drain-Mechanik wie api_delete_tag in app_pkg/tags.py).

Dry-run ist Default und durchläuft dieselben Code-Pfade mit Rollback am Ende —
die Vorhersage entspricht damit garantiert dem Apply, auch bei Kollisionen
innerhalb des Laufs (zwei kaputte Tags, die auf denselben neuen Namen
normalisieren). Idempotent: ein zweiter --apply findet nichts mehr zu tun.

Lauf im Container:
    docker compose exec markdown-converter python scripts/cleanup_tags.py          # dry-run
    docker compose exec markdown-converter python scripts/cleanup_tags.py --apply
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # noqa: E402
from models import Tag, conversion_tags, db, highlight_tags  # noqa: E402


def _usage_counts(tag_id):
    conv = db.session.query(conversion_tags).filter_by(tag_id=tag_id).count()
    hl = db.session.query(highlight_tags).filter_by(tag_id=tag_id).count()
    return conv, hl


def _drain_junctions(tag_id):
    db.session.execute(highlight_tags.delete().where(highlight_tags.c.tag_id == tag_id))
    db.session.execute(conversion_tags.delete().where(conversion_tags.c.tag_id == tag_id))


def _move_junction_rows(table, parent_col, source_id, target_id):
    """Junction-Rows von source auf target umhängen. Parents, die das Ziel-Tag
    schon tragen, würden die zusammengesetzte PK verletzen — deren Quell-Rows
    werden gelöscht (Duplikat-Paar), der Rest per UPDATE umgehängt."""
    already_tagged = (
        table.select().with_only_columns(table.c[parent_col])
        .where(table.c.tag_id == target_id)
    )
    db.session.execute(
        table.delete()
        .where(table.c.tag_id == source_id)
        .where(table.c[parent_col].in_(already_tagged))
    )
    db.session.execute(
        table.update().where(table.c.tag_id == source_id).values(tag_id=target_id)
    )


def run(apply_changes):
    mode = 'APPLY' if apply_changes else 'DRY-RUN'
    tags = Tag.query.order_by(Tag.user_id, Tag.id).all()
    print(f'Tag-Cleanup ({mode}) — Bestand: {len(tags)} Tags')

    changes = 0
    for tag in tags:
        new_name = Tag.normalize_name(tag.name)
        if new_name == tag.name:
            continue
        changes += 1
        conv, hl = _usage_counts(tag.id)
        label = f"  [user {tag.user_id}] {tag.name!r} (conv {conv}, hl {hl})"

        if not new_name:
            _drain_junctions(tag.id)
            db.session.delete(tag)
            db.session.flush()
            print(f"{label} → deleted (leer nach Normalisierung)")
            continue

        # Lookup NACH jedem flush — findet auch Tags, die in diesem Lauf
        # bereits umbenannt wurden (Intra-Lauf-Kollision → Merge).
        target = Tag.query.filter(
            Tag.user_id == tag.user_id, Tag.name == new_name, Tag.id != tag.id,
        ).first()
        if target is not None:
            _move_junction_rows(conversion_tags, 'conversion_id', tag.id, target.id)
            _move_junction_rows(highlight_tags, 'highlight_id', tag.id, target.id)
            db.session.delete(tag)
            db.session.flush()
            print(f"{label} → merged into {new_name!r} (id {target.id})")
        else:
            tag.name = new_name
            db.session.flush()
            print(f"{label} → renamed {new_name!r}")

    print(f'{changes} Änderung(en).')
    if apply_changes:
        db.session.commit()
        print('APPLY — Änderungen geschrieben.')
    else:
        db.session.rollback()
        print('DRY-RUN — nichts geschrieben. Echter Lauf: --apply')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tag-Bestand normalisieren (R2-E).')
    parser.add_argument('--apply', action='store_true',
                        help='Änderungen wirklich schreiben (Default: dry-run).')
    args = parser.parse_args()
    with app.app_context():
        run(args.apply)
