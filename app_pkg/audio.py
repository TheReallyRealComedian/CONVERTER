"""Audio transcription routes (Deepgram-backed).

Since SYNC-FREEZE P3 the file transcription is a JOB on the worker:
``POST /api/transcriptions`` stores the upload on the shared volume, creates
a ``pending`` ``audio_transcription`` row and enqueues
``tasks.transcribe_audio_task``; ``GET /api/transcriptions/<id>`` reconciles
the row (file-first, like the document conversion) and answers status plus,
once ready, the transcript. The synchronous ``POST /transcribe-audio-file``
is gone — its only caller was this page's JS.

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

Live transcription (browser ↔ Deepgram WebSocket via the token route) is
untouched.
"""
import json
import logging
import os
import tempfile

from flask import jsonify, render_template, request
from flask_login import current_user, login_required
from rq.exceptions import NoSuchJobError
from werkzeug.utils import secure_filename

from app_pkg.config import transcribe_job_timeout_for
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


def reconcile_transcription(conversion):
    """Flip a ``pending`` transcription to its terminal state on read.

    Idempotent, safe on every poll — the document-conversion reconcile with
    the transcription file layout:

    * result file parses      → ``ready``; the transcript becomes ``content``.
    * result file unreadable  → ``failed`` (atomic writes → a real defect).
    * RQ job failed           → ``failed`` + the exc_info **tail**.
    * RQ job gone / no job_id → ``failed`` ("Job nicht mehr auffindbar.").
    * RQ job queued/started   → stays ``pending``.
    * Redis unreachable       → stays ``pending`` (retried on the next poll).
    """
    if transcription_status(conversion) != STATUS_PENDING:
        return

    metadata = transcription_metadata(conversion)
    source_ext = metadata.get('source_format')

    if os.path.exists(transcription_result_path(conversion.id)):
        payload = read_result_file(conversion.id)
        if payload is None:
            _fail_transcription(conversion, metadata, 'Ergebnisdatei unlesbar.')
            return
        transcript = payload.get('transcript') or ''
        conversion.content = transcript
        metadata['transcription_status'] = STATUS_READY
        metadata['transcript_length'] = len(transcript)
        if payload.get('file_size_mb') is not None:
            metadata['file_size_mb'] = payload['file_size_mb']
        metadata['error'] = None
        _persist_metadata(conversion, metadata)
        discard_job_files(conversion.id, source_ext=source_ext)
        return

    import app as _app_module  # late: tests patch Job / redis_conn on app.py

    job_id = metadata.get('job_id')
    if not job_id:
        _fail_transcription(conversion, metadata, 'Job nicht mehr auffindbar.')
        discard_job_files(conversion.id, source_ext=source_ext)
        return
    try:
        job = _app_module.Job.fetch(job_id, connection=_app_module.redis_conn)
    except NoSuchJobError:
        _fail_transcription(conversion, metadata, 'Job nicht mehr auffindbar.')
        discard_job_files(conversion.id, source_ext=source_ext)
        return
    except Exception:
        logger.warning('reconcile_transcription: RQ fetch failed for job %s',
                       job_id, exc_info=True)
        return
    if job.is_failed:
        error = (job.exc_info or '')[-2000:] or 'Transkription fehlgeschlagen.'
        _fail_transcription(conversion, metadata, error)
        discard_job_files(conversion.id, source_ext=source_ext)
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
        try:
            temp_key = _app_module.deepgram_service.create_temporary_key(ttl_seconds=60)
            return jsonify({"deepgram_token": temp_key})
        except Exception as e:
            app.logger.error(f"Failed to create temporary Deepgram key: {e}", exc_info=True)
            return jsonify({"error": "Failed to create transcription token."}), 500

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
        with ``deduped: true`` and the stored state, nothing enqueued.
        """
        if 'audio_file' not in request.files:
            return jsonify({"error": 'Kein Datei-Feld "audio_file" im Request.'}), 400
        upload = request.files['audio_file']
        language = request.form.get('language', 'en')
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

        # Spool to the shared volume first (same directory as the final path →
        # os.replace stays an atomic same-FS rename), then create the row.
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
            recorded_at, recorded_at_source = _recorded_at_for(
                original_filename, request.form.get('recorded_at'))

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
            )
            db.session.add(conversion)
            db.session.flush()  # id for the id-derived source path

            os.replace(tmp_path, transcription_source_path(conversion.id, ext))

            metadata = build_transcription_metadata(
                language=language, source_format=ext, source_sha256=source_sha256,
                duration_seconds=duration,
                file_size_mb=round(size / (1024 * 1024), 2),
                recorded_at=recorded_at, recorded_at_source=recorded_at_source)
            conversion.metadata_json = json.dumps(metadata)
            db.session.commit()
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

        job = _app_module.task_queue.enqueue(
            transcribe_audio_task,
            conversion.id, ext, language,
            meta={'user_id': current_user.id, 'conversion_id': conversion.id},
            job_timeout=transcribe_job_timeout_for(duration),
        )

        # job_id back into metadata — reconcile keys "still transcribing" vs
        # "gone" on it (two commits, like the document job: the row must
        # exist before the page can poll).
        metadata['job_id'] = job.id
        conversion.metadata_json = json.dumps(metadata)
        db.session.commit()

        app.logger.info(
            f"Transcription job {job.id} queued for conversion {conversion.id} "
            f"({original_filename}, {size / (1024 * 1024):.1f} MB, "
            f"duration={duration})")
        return jsonify({
            'id': conversion.id,
            'status': STATUS_PENDING,
            'job_id': job.id,
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
