"""Backfill metadata.recorded_at für audio_transcription-Rows aus dem Dateinamen.

MCP1-FIX-Folge: Olis Diktiergerät benennt Files YYMMDD_NNNN (2-stelliges Jahr +
laufende Nummer, keine Uhrzeit), was der ursprüngliche MCP1-Parser nicht kannte
— die realen Diktate landeten daher mit leerem recorded_at in der DB. MCP1-FIX-P1
hat parse_recorded_at_from_filename diesen Dialekt beigebracht; dieses Script
zieht den BESTAND nach: jede audio_transcription-Row, die noch kein
metadata.recorded_at hat, deren source_filename aber jetzt einen validen
Parser-Treffer liefert, bekommt metadata['recorded_at'] + recorded_at_source=
'filename' — alle übrigen metadata-Felder bleiben erhalten.

Der Runtime-Parser wird IMPORTIERT (nie reimplementiert), damit Backfill und
Live-Ingest-Pfad nicht auseinanderdriften können (Memory
reference_tag_vocab_central_gate_plus_backfill_script).

Dry-run ist Default und durchläuft denselben Code-Pfad mit Rollback am Ende —
die Vorhersage entspricht damit garantiert dem Apply. Idempotent: ein zweiter
--apply findet nichts mehr zu tun (die Rows tragen dann recorded_at).

Konservativ ("falsch ist schlimmer als leer"): eine Row, deren Dateiname nicht
parst oder die schon ein recorded_at hat, bleibt unangetastet.

Lauf im Container:
    docker compose exec markdown-converter python scripts/backfill_recorded_at.py          # dry-run
    docker compose exec markdown-converter python scripts/backfill_recorded_at.py --apply
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # noqa: E402
from app_pkg.library import parse_recorded_at_from_filename  # noqa: E402
from models import Conversion, db  # noqa: E402


def run(apply_changes):
    mode = 'APPLY' if apply_changes else 'DRY-RUN'
    rows = (Conversion.query
            .filter_by(conversion_type='audio_transcription')
            .order_by(Conversion.id)
            .all())
    print(f'recorded_at-Backfill ({mode}) — audio_transcription-Rows: {len(rows)}')

    changes = 0
    for c in rows:
        try:
            metadata = json.loads(c.metadata_json) if c.metadata_json else {}
        except (TypeError, ValueError):
            print(f"  [id {c.id}] {c.source_filename!r} → skip (metadata_json kaputt)")
            continue
        if not isinstance(metadata, dict):
            print(f"  [id {c.id}] {c.source_filename!r} → skip (metadata kein Objekt)")
            continue
        if 'recorded_at' in metadata:
            continue  # schon gesetzt — idempotenter Skip
        parsed = parse_recorded_at_from_filename(c.source_filename)
        if parsed is None:
            continue  # Dateiname liefert nichts — konservativer Skip
        changes += 1
        new_value = parsed.isoformat()
        metadata['recorded_at'] = new_value
        metadata['recorded_at_source'] = 'filename'
        c.metadata_json = json.dumps(metadata)
        print(f"  [id {c.id}] {c.source_filename!r}: recorded_at leer → {new_value}")

    print(f'{changes} Row(s) zu befüllen.')
    if apply_changes:
        db.session.commit()
        print('APPLY — Änderungen geschrieben.')
    else:
        db.session.rollback()
        print('DRY-RUN — nichts geschrieben. Echter Lauf: --apply')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='metadata.recorded_at aus source_filename nachziehen (MCP1-FIX).')
    parser.add_argument('--apply', action='store_true',
                        help='Änderungen wirklich schreiben (Default: dry-run).')
    args = parser.parse_args()
    with app.app_context():
        run(args.apply)
