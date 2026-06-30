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
import hmac
import json
import logging
import os
import wave
from pathlib import Path

from flask import jsonify, request, send_file
from flask_login import login_required
from rq.exceptions import NoSuchJobError

from app_pkg.config import OUTPUT_DIR, TIMEOUT_RQ_JOB_SECONDS
# Reuse the Ingest auth primitives (same Bearer parse + target-user resolver as
# the Card write surface), so a session-less narration write resolves the SAME
# target user. Only the secret differs — see _authorize_narration_write.
from app_pkg.ingest import _bearer_token, _resolve_target_user
from app_pkg.library import get_owned_conversion
from models import Conversion, db
from services.markdown_sections import _is_degenerate_title, derive_title
from services.narration_library import (
    NARRATION_STATUS_FAILED,
    NARRATION_STATUS_PENDING,
    NARRATION_STATUS_READY,
    build_narration_metadata,
    narration_audio_path,
    narration_metadata,
    narration_status,
    narration_to_markdown,
)
from services.narration_render import DEFAULT_NARRATION_MODEL, validate_turns
from tasks import generate_narration_task

logger = logging.getLogger(__name__)

# Default BCP-47 language for Cloud-TTS narration (the renderer default + the
# app is German). The body's optional ``language`` field overrides it.
DEFAULT_LANGUAGE_CODE = 'de-DE'


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


# --- NARR-3 write: token-auth gate (own secret) ------------------------------

def _authorize_narration_write():
    """Token-auth gate for ``POST /api/narrations`` (NARR-3).

    Mirrors ``_authorize_card_write`` exactly, but reads a **separate**
    ``NARRATION_TOKEN``: a narration render costs real GCP money per call (unlike
    the free DB-write surfaces that share ``CARD_TOKEN``), so it gets an
    independently revocable secret. Fail-closed (503) without the token,
    constant-time Bearer compare (401 on missing/wrong), token never logged; the
    target user is the shared Ingest resolver (``INGEST_USER`` / first()).

    Returns ``(user, None)`` on success or ``(None, (response, status))``.
    """
    expected = os.environ.get('NARRATION_TOKEN')
    if not expected:
        logger.warning('Narration write rejected: NARRATION_TOKEN not configured')
        return None, (jsonify({'error': 'Narration-API nicht konfiguriert.'}), 503)

    provided = _bearer_token()
    if provided is None or not hmac.compare_digest(provided.encode('utf-8'),
                                                   expected.encode('utf-8')):
        reason = 'missing bearer' if provided is None else 'token mismatch'
        logger.warning('Narration write auth failed (%s) from %s', reason, request.remote_addr)
        return None, (jsonify({'error': 'Nicht autorisiert.'}), 401)

    target = _resolve_target_user()
    if target is None:
        logger.error('Narration write rejected: no target user (INGEST_USER=%r)',
                     os.environ.get('INGEST_USER'))
        return None, (jsonify({'error': 'Kein Ziel-Benutzer vorhanden.'}), 503)

    return target, None


def register(app):
    # Late import: tests patch the RQ singletons (``task_queue``, ``Job``,
    # ``redis_conn``) on the top-level app.py module, so look them up at call
    # time rather than capturing imports here.
    import app as _app_module

    @app.route('/api/narrations', methods=['POST'])
    def api_create_narration():
        """Create a pending narration + enqueue the DB-free render (NARR-3).

        Token-authed (``NARRATION_TOKEN``, CSRF-exempt — session-less write).
        The web side owns the DB: it validates the turn contract, creates the
        ``pending`` audio_narration Conversion (flush → id → metadata), enqueues
        ``generate_narration_task`` onto the shared queue, and writes the job_id
        back into metadata (reconcile needs it). The worker renders DB-free; the
        status/serve reads reconcile ``pending`` → ``ready``/``failed`` later.
        """
        target, err = _authorize_narration_write()
        if err:
            return err

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'Ungültiger Request-Body. JSON-Objekt erwartet.'}), 400

        mode = data.get('mode')
        voices = data.get('voices')
        turns = data.get('turns')

        # Contract validation reused verbatim from the renderer (single source
        # of truth): shape of turns/voices + mode↔speaker-count consistency.
        try:
            validate_turns(turns, voices, mode)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        style_prompt = data.get('style_prompt')
        if not isinstance(style_prompt, str):
            style_prompt = None

        language_code = data.get('language')
        if not isinstance(language_code, str) or not language_code.strip():
            language_code = DEFAULT_LANGUAGE_CODE

        tts_model = data.get('tts_model')
        if not isinstance(tts_model, str) or not tts_model.strip():
            tts_model = DEFAULT_NARRATION_MODEL

        # content = speaker-labelled transcript Markdown (NOT NULL, Reader-able).
        # TITLE-FIX: a degenerate posted title is re-derived from that content.
        content = narration_to_markdown(turns)
        posted = data.get('title')
        title = (derive_title(content) if _is_degenerate_title(posted) else posted)[:255]

        # pending Conversion — flush first so the id-derived audio_filename in
        # metadata matches narration_audio_path.
        conversion = Conversion(
            user_id=target.id,
            conversion_type='audio_narration',
            title=title,
            content=content,
            lifecycle_status='inbox',
        )
        db.session.add(conversion)
        db.session.flush()

        metadata = build_narration_metadata(
            conversion.id, status=NARRATION_STATUS_PENDING,
            tts_model=tts_model, speakers=voices, transcript=turns)
        conversion.metadata_json = json.dumps(metadata)
        db.session.commit()

        # Enqueue the DB-free render task (positional task args + RQ job opts).
        job = _app_module.task_queue.enqueue(
            generate_narration_task,
            conversion.id, turns, voices, style_prompt, mode, language_code, tts_model,
            meta={'user_id': target.id, 'conversion_id': conversion.id},
            job_timeout=TIMEOUT_RQ_JOB_SECONDS,
        )

        # job_id back into metadata — reconcile keys "still rendering" vs "gone"
        # on it. (Two commits: the row must exist before the agent can poll.)
        metadata['job_id'] = job.id
        conversion.metadata_json = json.dumps(metadata)
        db.session.commit()

        app.logger.info(f"Narration job {job.id} queued for conversion {conversion.id}")
        return jsonify({
            'narration_id': conversion.id,
            'job_id': job.id,
            'status': NARRATION_STATUS_PENDING,
        }), 202

    @app.route('/api/narrations/<int:conversion_id>', methods=['GET'])
    @login_required
    def api_narration_status(conversion_id):
        """Poll a narration's state — the agent's / UI's poll endpoint (NARR-3).

        Owner-404 (``get_owned_conversion``) + wrong-type-404 (no type leak),
        then ``reconcile_narration`` flips a finished/failed pending render to
        its terminal state before returning ``to_dict()`` (whose ``metadata``
        carries ``narration_status`` / ``duration_seconds`` / ``error``).
        """
        conversion = get_owned_conversion(conversion_id)  # owner-404

        if conversion.conversion_type != 'audio_narration':
            return jsonify({'error': 'Vertonung nicht gefunden.'}), 404

        reconcile_narration(conversion)  # pending → ready/failed if resolvable
        return jsonify(conversion.to_dict())

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

        # NARR-3: reconcile pending → ready/failed BEFORE the ready-gate, so a
        # pending-but-file-already-present element serves immediately.
        reconcile_narration(conversion)

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

    # Session-less, token-authed write has no CSRF cookie → waive CSRF for THIS
    # view only (same posture as Ingest / the Card writes). The session-authed
    # serve route stays under CSRF.
    app.extensions['csrf'].exempt(api_create_narration)
