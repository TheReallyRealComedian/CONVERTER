"""Faithful-narration library routes (NARR-2).

Serves the persisted audio of an ``audio_narration`` Conversion. The WAV lives
at the deterministic, **id-derived** path ``OUTPUT_DIR/narration_<id>.wav`` on
the shared ``podcast_data`` volume (written by the NARR-3 worker). Because the
path is resolved from the id — never from ``metadata.audio_filename`` — there is
no filename-injection surface; a traversal guard against ``OUTPUT_DIR`` is kept
as belt-and-suspenders.

Session-authed (``@login_required``, owner-404) and **persistent**: unlike the
alt-podcast ``podcast_download``, the file is **never deleted on serve**. NARR-3
will add the token POST that creates narrations to this same module.
"""
import json
import logging
import os
import wave
from pathlib import Path

from flask import jsonify, send_file
from flask_login import login_required
from rq.exceptions import NoSuchJobError

from app_pkg.config import OUTPUT_DIR
from app_pkg.library import get_owned_conversion
from models import db
from services.narration_library import (
    NARRATION_STATUS_FAILED,
    NARRATION_STATUS_PENDING,
    NARRATION_STATUS_READY,
    narration_audio_path,
    narration_metadata,
    narration_status,
)

logger = logging.getLogger(__name__)


# --- NARR-3 reconcile: web-side state machine for a DB-free worker -----------
#
# Option B (set architecture): the RQ worker mounts the shared podcast_data
# volume but NOT the SQLite DB, so it renders the WAV and returns — it never
# flips the Conversion. The web side, which *does* have the DB, decides the
# outcome lazily on read (status poll / serve), from two observable facts: does
# the id-derived WAV exist, and what is the RQ job's terminal state.

def _wav_duration_seconds(path):
    """Whole-second duration of a WAV (``frames / framerate``), best-effort.

    Returns ``None`` on a missing / corrupt / headerless WAV rather than
    raising — a render that genuinely produced audio must never be failed just
    because its duration can't be read (the serve smoke-WAVs are deliberately
    headerless). Matches the documented ``duration_seconds: int | null`` shape.
    """
    try:
        with wave.open(path, 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
    except Exception:
        return None
    if not rate:
        return None
    return round(frames / rate)


def _persist_metadata(conversion, metadata):
    """Write the mutated metadata dict back onto the Conversion and commit."""
    conversion.metadata_json = json.dumps(metadata)
    db.session.commit()


def _fail_narration(conversion, metadata, error):
    """Flip a pending narration to ``failed`` with a short error string."""
    metadata['narration_status'] = NARRATION_STATUS_FAILED
    metadata['error'] = error
    _persist_metadata(conversion, metadata)


def reconcile_narration(conversion):
    """Flip a ``pending`` narration to its terminal state on read (NARR-3).

    Idempotent: terminal states (``ready`` / ``failed``) are left untouched, so
    this is safe to call on every status poll and every serve. The transitions:

    * audio file exists       → ``ready`` + ``duration_seconds`` from the WAV.
    * RQ job failed            → ``failed`` + the truncated ``exc_info``.
    * RQ job gone / no job_id   → ``failed`` ("Job nicht mehr auffindbar.").
    * RQ job queued/started     → stays ``pending``.
    * Redis unreachable         → stays ``pending`` (retried on the next poll).
    """
    if narration_status(conversion) != NARRATION_STATUS_PENDING:
        return

    metadata = narration_metadata(conversion)
    audio_path = narration_audio_path(conversion.id)

    # The file is the source of truth: a present WAV means the worker finished,
    # regardless of how Redis later reports the (possibly already-evicted) job.
    if os.path.exists(audio_path):
        metadata['narration_status'] = NARRATION_STATUS_READY
        metadata['duration_seconds'] = _wav_duration_seconds(audio_path)
        metadata['error'] = None
        _persist_metadata(conversion, metadata)
        return

    # No file yet — consult the RQ job to tell "still rendering" from "dead".
    # Late import: tests patch Job / redis_conn on the top-level app.py module.
    import app as _app_module

    job_id = metadata.get('job_id')
    if not job_id:
        _fail_narration(conversion, metadata, 'Job nicht mehr auffindbar.')
        return
    try:
        job = _app_module.Job.fetch(job_id, connection=_app_module.redis_conn)
    except NoSuchJobError:
        # Job expired/evicted from Redis and no file was produced → unrecoverable.
        _fail_narration(conversion, metadata, 'Job nicht mehr auffindbar.')
        return
    except Exception:
        # Transient Redis error — never fail an in-flight render over a blip.
        logger.warning('reconcile_narration: RQ fetch failed for job %s', job_id, exc_info=True)
        return
    if job.is_failed:
        error = (job.exc_info or '')[:500] or 'Vertonung fehlgeschlagen.'
        _fail_narration(conversion, metadata, error)
    # queued / started / deferred → still rendering, stays pending.


def register(app):
    @app.route('/api/narrations/<int:conversion_id>/audio', methods=['GET'])
    @login_required
    def narration_audio(conversion_id):
        """Stream a ready narration's WAV.

        404 on: not owned / missing (``get_owned_conversion``), wrong type
        (no type leak), status ≠ ready (pending/failed have no file), or file
        absent. 403 on a path that escapes ``OUTPUT_DIR``. Never unlinks
        (persistent library element).
        """
        conversion = get_owned_conversion(conversion_id)  # owner-404

        if conversion.conversion_type != 'audio_narration':
            return jsonify({'error': 'Vertonung nicht gefunden.'}), 404

        if narration_status(conversion) != 'ready':
            # pending/failed carry no audio file yet; NARR-5 surfaces the state.
            return jsonify({'error': 'Audio nicht verfügbar.'}), 404

        # id-derived path (no user input); existence-check before the guard.
        file_path = narration_audio_path(conversion_id)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Audio nicht verfügbar.'}), 404

        # Traversal guard like podcast_download — Path.is_relative_to avoids the
        # str.startswith prefix-collision bug.
        real_path = os.path.realpath(file_path)
        if not Path(real_path).is_relative_to(Path(os.path.realpath(OUTPUT_DIR))):
            app.logger.warning(f"Narration path traversal blocked: {file_path}")
            return jsonify({'error': 'Datei außerhalb des Ausgabe-Verzeichnisses.'}), 403

        # Persistent: NO os.unlink (unlike podcast_download).
        return send_file(
            real_path,
            mimetype='audio/wav',
            download_name=f'narration_{conversion_id}.wav',
        )
