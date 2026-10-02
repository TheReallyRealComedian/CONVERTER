"""Audio transcription routes (Deepgram-backed).

Since SYNC-FREEZE P3 the file transcription is a JOB on the worker:
``POST /api/transcriptions`` creates the job mark, stores the upload on the
shared volume as ``source_<job>.<ext>``, enqueues
``tasks.transcribe_audio_task`` and then creates the ``pending``
``audio_transcription`` row in one commit; ``GET /api/transcriptions/<id>``
reconciles the row (file-first under the row's own mark, like the document
conversion) and answers status plus, once ready, the transcript. The
synchronous ``POST /transcribe-audio-file`` is gone. Its only WEB caller was
this page's JS — the iOS app still calls it in its capture flow and has been
hitting a 404 there since (BACKLOG IOS-TRANSCRIBE-ROUTE).

JOB-ID-REUSE: no job file hangs on the row id (SQLite reuses the id of a
deleted highest row) — see ``services/transcription_jobs.py`` for the layout.

Why a job (see ``services/transcription_jobs.py`` for the long form): NOT
against the freeze — since P2 a synchronous transcription parks nobody. The
job buys progress (elapsed time instead of a silent request), a closed tab
(the worker finishes, the library row reconciles on the next read) and
repeatability (same file + language → the stored result; a failed job re-runs
by re-submitting). Per construction it also closes the AudioChunker
singleton exposure P2 opened: one job at a time on the worker, one service
instance per job.

The job row is a library element from the start (``lifecycle_status``
``archive``, like document jobs); "In Library speichern" on the page moves
it into the inbox via ``/place`` instead of creating a second row. The
``recorded_at`` capture (MCP1) happens here at submit — filename date beats
the client's ``lastModified``, the precedence of ``POST /api/conversions``.

Live transcription runs browser ↔ Deepgram over a WebSocket; the page gets a
short-lived token for it from ``GET /api/get-deepgram-token`` (Deepgram's
grant, server-set lifetime, fail-closed — SEC-DG-TOKEN), never the API key.
"""
import json
import logging
import os
import tempfile

from flask import jsonify, render_template, request
from flask_login import current_user, login_required
from rq.exceptions import NoSuchJobError
from werkzeug.utils import secure_filename

from app_pkg.config import (DEEPGRAM_LIVE_TOKEN_TTL_SECONDS,
                            TIMEOUT_DEEPGRAM_GRANT_SECONDS, is_job_mark,
                            new_job_mark, transcribe_job_timeout_for)
from app_pkg.decorators import require_service
from app_pkg.library import (_normalize_client_recorded_at,
                             parse_recorded_at_from_filename)
from models import Conversion, db
from services.transcription_jobs import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_READY,
    TRANSCRIPTION_TYPE,
    build_transcription_metadata,
    discard_job_files,
    ensure_transcription_dir,
    file_sha256,
    probe_duration_seconds,
    read_result_file,
    transcription_metadata,
    transcription_result_path,
    transcription_source_path,
    transcription_status,
)
from tasks import transcribe_audio_task

logger = logging.getLogger(__name__)

# Single source of truth for what the transcription accepts. The template
# reads this via the route context for the file-input ``accept`` attribute and
# for ``window.PageData.acceptedAudioExtensions`` (frontend pre-submit check).
ACCEPTED_AUDIO_EXTENSIONS = ('mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm')
MAX_AUDIO_FILE_SIZE_MB = 500

# F-013: enumerated languages that the audio-tab UI offers. Values outside
# this set used to flow through to Deepgram and surface as a 500 from the SDK;
# they get a clean 400 + DE-JSON.
ACCEPTED_TRANSCRIPTION_LANGUAGES = ('en', 'de')


def _recorded_at_for(filename, client_value):
    """MCP1 capture at submit: ``(iso_string, source)`` or ``(None, None)``.

    The device-authoritative filename date beats the client field (the
    upload's ``lastModified`` can be the copy time) — same precedence as
    ``POST /api/conversions``. The form carries the client value as a string
    of epoch milliseconds; an unparseable value is dropped, never a 400.
    """
    parsed = parse_recorded_at_from_filename(filename)
    if parsed is not None:
        return parsed.isoformat(), 'filename'
    if client_value is None or client_value == '':
        return None, None
    value = client_value
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    normalized = _normalize_client_recorded_at(value)
    if normalized is None:
        logger.warning('recorded_at unparseable, ignored: %r', client_value)
        return None, None
    return normalized, 'client'


def _find_duplicate(user_id, source_sha256, language):
    """Idempotency: same user + content hash + language with a pending/ready
    job → that row. ``failed`` never dedups — re-submitting IS the retry.
    Substring prefilter on the JSON text, then exact confirmation (the
    ingest ``_find_by_source_id`` pattern)."""
    candidates = (Conversion.query
                  .filter_by(user_id=user_id, conversion_type=TRANSCRIPTION_TYPE)
                  .filter(Conversion.metadata_json.contains(source_sha256, autoescape=True))
                  .all())
    for candidate in candidates:
        metadata = transcription_metadata(candidate)
        if (metadata.get('source_sha256') == source_sha256
                and metadata.get('language') == language
                and metadata.get('transcription_status') in (STATUS_PENDING, STATUS_READY)):
            return candidate
    return None


# --- reconcile: web-side state machine for the DB-free worker -----------------

def _persist_metadata(conversion, metadata):
    conversion.metadata_json = json.dumps(metadata)
    db.session.commit()


def _fail_transcription(conversion, metadata, error):
    metadata['transcription_status'] = STATUS_FAILED
    metadata['error'] = error
    _persist_metadata(conversion, metadata)


def _take_result(conversion, metadata, job_id, source_ext):
    """File first: if the job's own result is on the volume, settle the row.

    Returns ``True`` iff the row left ``pending``. The name carries the job
    mark — a result can only be the one THIS row's job wrote (JOB-ID-REUSE).
    """
    if not os.path.exists(transcription_result_path(job_id)):
        return False
    payload = read_result_file(job_id)
    if payload is None:
        _fail_transcription(conversion, metadata, 'Ergebnisdatei unlesbar.')
        return True
    transcript = payload.get('transcript') or ''
    conversion.set_content(transcript)  # LOST-UPDATE: content writers bump content_version
    metadata['transcription_status'] = STATUS_READY
    metadata['transcript_length'] = len(transcript)
    if payload.get('file_size_mb') is not None:
        metadata['file_size_mb'] = payload['file_size_mb']
    metadata['error'] = None
    _persist_metadata(conversion, metadata)
    discard_job_files(job_id, source_ext=source_ext)
    return True


def reconcile_transcription(conversion):
    """Flip a ``pending`` transcription to its terminal state on read.

    Idempotent, safe on every poll — the document-conversion reconcile with
    the transcription file layout (files under the row's own job mark):

    * result file parses      → ``ready``; the transcript becomes ``content``.
    * result file unreadable  → ``failed`` (atomic writes → a real defect).
    * RQ job failed           → ``failed`` + the exc_info **tail**.
    * RQ job gone / no job_id → ``failed`` ("Job nicht mehr auffindbar.").
    * RQ job finished, no file → ``failed`` ("Ergebnis nicht auffindbar.") —
                                the worker writes before the job ends.
    * RQ job queued/started   → stays ``pending``.
    * Redis unreachable       → stays ``pending`` (retried on the next poll).
    """
    if transcription_status(conversion) != STATUS_PENDING:
        return

    metadata = transcription_metadata(conversion)
    source_ext = metadata.get('source_format')
    job_id = metadata.get('job_id')
    # metadata_json is client-writable: only a real mark names a file or a job.
    has_mark = is_job_mark(job_id)

    if has_mark and _take_result(conversion, metadata, job_id, source_ext):
        return

    # Another reconcile of this row may have consumed the result (ready
    # committed, THEN the files discarded) while this one still held the row
    # as pending — read the row again before judging it.
    db.session.refresh(conversion)
    if transcription_status(conversion) != STATUS_PENDING:
        return

    if not has_mark:
        _fail_transcription(conversion, metadata, 'Job nicht mehr auffindbar.')
        return

    import app as _app_module  # late: tests patch Job / redis_conn on app.py

    try:
        job = _app_module.fetch_job(job_id)
    except NoSuchJobError:
        _fail_transcription(conversion, metadata, 'Job nicht mehr auffindbar.')
        discard_job_files(job_id, source_ext=source_ext)
        return
    except Exception:
        logger.warning('reconcile_transcription: RQ fetch failed for job %s',
                       job_id, exc_info=True)
        return
    if job.is_failed:
        error = (job.exc_info or '')[-2000:] or 'Transkription fehlgeschlagen.'
        _fail_transcription(conversion, metadata, error)
        discard_job_files(job_id, source_ext=source_ext)
    elif job.is_finished:
        # Finished between our look at the volume and the fetch → the result
        # is there now. Still none: a concurrent reconcile took it (the row
        # is no longer pending) or it never existed.
        if _take_result(conversion, metadata, job_id, source_ext):
            return
        db.session.refresh(conversion)
        if transcription_status(conversion) != STATUS_PENDING:
            return
        _fail_transcription(conversion, metadata, 'Ergebnis nicht auffindbar.')
        discard_job_files(job_id, source_ext=source_ext)
    # queued / started / deferred → still transcribing, stays pending.


def _transcription_response(conversion):
    metadata = transcription_metadata(conversion)
    status = transcription_status(conversion)
    ready = status == STATUS_READY
    return {
        'id': conversion.id,
        'status': status,
        'title': conversion.title,
        'transcript': conversion.content if ready else None,
        'metadata': {
            'language': metadata.get('language'),
            'file_size_mb': metadata.get('file_size_mb'),
            'duration_seconds': metadata.get('duration_seconds'),
            'transcript_length': metadata.get('transcript_length') if ready else None,
            'recorded_at': metadata.get('recorded_at'),
            'recorded_at_source': metadata.get('recorded_at_source'),
        },
        'error': metadata.get('error'),
        'source': {
            'filename': conversion.source_filename,
            'format': metadata.get('source_format'),
            'size_bytes': conversion.source_size_bytes,
        },
        'lifecycle_status': conversion.lifecycle_status,
        'created_at': conversion.created_at.isoformat() if conversion.created_at else None,
    }


def register(app):
    # Late import: tests patch ``app.deepgram_service``, ``app.task_queue`` and
    # ``app.DEEPGRAM_API_KEY`` on the top-level app.py module, so look them
    # up at call time rather than capturing imports here.
    import app as _app_module

    @app.route('/audio-converter')
    @login_required
    def audio_converter():
        return render_template(
            'audio_converter.html',
            deepgram_api_key_set=bool(_app_module.DEEPGRAM_API_KEY),
            accepted_audio_extensions=ACCEPTED_AUDIO_EXTENSIONS,
            accepted_audio_extensions_accept=','.join('.' + ext for ext in ACCEPTED_AUDIO_EXTENSIONS),
            max_audio_file_size_mb=MAX_AUDIO_FILE_SIZE_MB,
        )

    @app.route('/api/get-deepgram-token', methods=['GET'])
    @login_required
    @require_service('deepgram')
    def get_deepgram_token():
        """Short-lived Deepgram token for the page's live WebSocket.

        Answers only with a token minted by Deepgram's grant; the lifetime is
        the server's (``DEEPGRAM_LIVE_TOKEN_TTL_SECONDS``), nothing in the
        request can change it. Fail-closed (SEC-DG-TOKEN): a failed grant is a
        502 — no branch falls back to the API key, which carries account
        rights and does not expire.
        """
        try:
            token, expires_in = _app_module.deepgram_service.grant_live_token(
                ttl_seconds=DEEPGRAM_LIVE_TOKEN_TTL_SECONDS,
                timeout_seconds=TIMEOUT_DEEPGRAM_GRANT_SECONDS,
            )
        except Exception as e:
            # Type and upstream status only: an SDK error can carry response
            # headers and body, and nothing near a credential belongs in a log.
            app.logger.error('Deepgram token grant failed: %s (upstream status %s)',
                             type(e).__name__, getattr(e, 'status_code', None))
            return jsonify({'error': 'Transkriptions-Token konnte nicht erstellt werden. '
                                     'Bitte erneut versuchen.'}), 502
        response = jsonify({'deepgram_token': token, 'expires_in': expires_in})
        # A credential response is never stored — a cached answer would also
        # be an expired token on the next recording.
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.route('/api/transcriptions', methods=['POST'])
    @login_required
    @require_service('deepgram')
    def api_create_transcription():
        """Submit an audio file for transcription → 202 ``{id, status, job_id}``.

        Multipart ``audio_file`` + ``language`` (+ optional ``recorded_at``
        epoch-ms). Session-authed like the page (CSRF via the inversion; a
        bearer skips it). The configured-ness gate is the Deepgram singleton
        (``require_service``) — the job itself runs on the worker with the
        worker's own key. Same file + language already pending/ready → 200
        with ``deduped: true`` and the stored state, nothing enqueued. A queue
        that cannot take the job → 503, no row, no source left on the volume.
        """
        if 'audio_file' not in request.files:
            return jsonify({"error": 'Kein Datei-Feld "audio_file" im Request.'}), 400
        upload = request.files['audio_file']
        # TRANS-DE-DEFAULT: Oli dictates in German. The default must agree with
        # the module default in static/js/audio_converter.js and the button that
        # carries ``lang-active`` in templates/audio_converter.html.
        language = request.form.get('language', 'de')
        if not upload.filename:
            return jsonify({"error": "Keine Datei ausgewählt."}), 400
        if language not in ACCEPTED_TRANSCRIPTION_LANGUAGES:
            return jsonify({
                "error": "Ungültige Sprache. Erlaubt: "
                         + ", ".join(ACCEPTED_TRANSCRIPTION_LANGUAGES) + "."
            }), 400

        original_filename = upload.filename
        ext = os.path.splitext(secure_filename(original_filename))[1].lstrip('.').lower()
        if ext not in ACCEPTED_AUDIO_EXTENSIONS:
            return jsonify({
                "error": "Dieses Dateiformat wird nicht unterstützt. "
                         "Erlaubt: MP3, WAV, M4A, OGG, FLAC, WEBM."
            }), 400

        # The job mark: created before anything touches the volume. It names
        # the source and the result, is the RQ job id and lands in metadata.
        job_id = new_job_mark()

        # Spool to the shared volume first (same directory as the final path →
        # os.replace stays an atomic same-FS rename), enqueue, then the row.
        job_dir = ensure_transcription_dir()
        tmp_f = tempfile.NamedTemporaryFile(dir=job_dir, suffix='.upload', delete=False)
        tmp_path = tmp_f.name
        tmp_f.close()
        try:
            upload.save(tmp_path)
            size = os.path.getsize(tmp_path)
            if size == 0:
                return jsonify({'error': 'Leere Datei.'}), 400

            source_sha256 = file_sha256(tmp_path)
            duplicate = _find_duplicate(current_user.id, source_sha256, language)
            if duplicate is not None:
                reconcile_transcription(duplicate)  # may have finished meanwhile
                payload = _transcription_response(duplicate)
                payload['deduped'] = True
                return jsonify(payload), 200

            duration = probe_duration_seconds(tmp_path)
            os.replace(tmp_path, transcription_source_path(job_id, ext))
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

        recorded_at, recorded_at_source = _recorded_at_for(
            original_filename, request.form.get('recorded_at'))
        metadata = build_transcription_metadata(
            language=language, source_format=ext, source_sha256=source_sha256,
            duration_seconds=duration,
            file_size_mb=round(size / (1024 * 1024), 2),
            recorded_at=recorded_at, recorded_at_source=recorded_at_source)
        metadata['job_id'] = job_id

        # Enqueue BEFORE the row exists: no open DB write while Redis is
        # asked, and a failed enqueue leaves nothing behind. The task needs
        # no row id — it reads and writes under the mark.
        try:
            _app_module.task_queue.enqueue(
                transcribe_audio_task,
                job_id, ext, language,
                job_id=job_id,
                meta={'user_id': current_user.id},
                job_timeout=transcribe_job_timeout_for(duration),
            )
        except Exception:
            logger.error('Transcription job %s could not be enqueued',
                         job_id, exc_info=True)
            discard_job_files(job_id, source_ext=ext)
            return jsonify({'error': 'Auftrag konnte nicht eingereiht werden. '
                                     'Bitte erneut versuchen.'}), 503

        # One commit, job_id included. If it fails the job still runs — under
        # a mark nobody looks for; the task removes the source itself.
        stem = os.path.splitext(original_filename)[0] or original_filename
        conversion = Conversion(
            user_id=current_user.id,
            conversion_type=TRANSCRIPTION_TYPE,
            title=stem[:255],
            content='',  # fills in on the ready-reconcile
            source_filename=original_filename[:255],
            source_mimetype=upload.mimetype,
            source_size_bytes=size,
            # A job row shelves to the archive; "In Library speichern"
            # moves it into the inbox (the page's existing button).
            lifecycle_status='archive',
            metadata_json=json.dumps(metadata),
        )
        db.session.add(conversion)
        db.session.commit()

        app.logger.info(
            f"Transcription job {job_id} queued for conversion {conversion.id} "
            f"({original_filename}, {size / (1024 * 1024):.1f} MB, "
            f"duration={duration})")
        return jsonify({
            'id': conversion.id,
            'status': STATUS_PENDING,
            'job_id': job_id,
        }), 202

    @app.route('/api/transcriptions/<int:conversion_id>', methods=['GET'])
    @login_required
    def api_transcription_status(conversion_id):
        """Poll a transcription — status and, once ``ready``, the transcript.

        Owner- and type-scoped in one filter → a foreign, missing or
        wrong-type id is an indistinguishable 404. Legacy rows saved by the
        synchronous flow carry no job keys and answer as ``ready``.
        """
        conversion = Conversion.query.filter_by(
            id=conversion_id,
            user_id=current_user.id,
            conversion_type=TRANSCRIPTION_TYPE,
        ).first()
        if conversion is None:
            return jsonify({'error': 'Transkription nicht gefunden.'}), 404

        reconcile_transcription(conversion)
        return jsonify(_transcription_response(conversion))
