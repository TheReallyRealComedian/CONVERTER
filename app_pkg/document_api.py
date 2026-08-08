"""Document-conversion service API (DOC-API) — the conversion becomes a service.

Additive to ``POST /transform-document`` (untouched, the web UI hangs on it):
this module is the **service-facing** surface — JSON in/out, token- or
session-authed, async via the Option-B job mechanic (NARR-3), so another
service can submit a document and poll for structured Markdown.

Endpoints
---------
* ``POST /api/document-conversions`` — submit (multipart field ``file``),
  answers 202 with ``{id, status, job_id}``.
* ``GET  /api/document-conversions/<id>`` — poll; the terminal ``ready``
  answer carries the result **in the same response** (markdown + warnings).
  One endpoint for status *and* result on purpose: the result of a document
  conversion IS JSON-shaped (unlike the narration WAV, which needs its own
  binary serve route), so a second result endpoint would only force a second
  round-trip with zero gain.

Auth — two coequal paths (``_authorize_document_access``):
1. A logged-in user: session cookie or per-user bearer (the MOBILE-AUTH
   ``request_loader`` has already populated ``current_user``).
2. The service token ``DOC_CONVERT_TOKEN`` — an **own** env token, mirrored
   from ``_authorize_narration_write`` (not shared: a conversion can cost
   model money per call, so it must be independently revocable). Fail-closed
   (503 unconfigured), constant-time compare (401), never logged; target user
   via the shared Ingest resolver.

CSRF — deliberately **no** exemption, unlike Ingest/Narration (both predate
the MOBILE-AUTH inversion and are token-only, so their exemption is inert).
Here cookie sessions are a *legitimate* auth path, and an exempted
cookie-authed write would be a real CSRF hole. The inversion already does the
right thing: bearer-header presence skips CSRF (service token + app bearer),
cookie-session mutations keep requiring the CSRF token (the web UI sends it
anyway). Consequence, documented for the contract: a POST with neither an
``Authorization`` header nor a valid CSRF token dies as 400 (CSRF) before
reaching this module's 401/503 — fail-closed either way.

Job mechanic (Option B, extended): the web process validates + stores the
upload on the shared volume, creates the ``pending`` Conversion
(``conversion_type='document_conversion'``, state in ``metadata_json`` — no
schema touch) and enqueues ``tasks.convert_document_task``. The worker is
DB-free: it converts and atomically writes ``result_<id>.json``; reconcile
here *reads* that structured file (markdown + warnings — existence alone
proves nothing, the DOC-API extension over "WAV exists == done") and flips
the row on poll. Transient Redis errors keep ``pending``; terminal states are
idempotent.
"""
import hashlib
import hmac
import json
import logging
import os
import tempfile

from flask import jsonify, request
from flask_login import current_user
from rq.exceptions import NoSuchJobError
from werkzeug.utils import secure_filename

from app_pkg.config import doc_convert_job_timeout_for
from app_pkg.documents import ACCEPTED_EXTENSIONS
# Reuse the Ingest auth primitives (same Bearer parse + target-user resolver
# as the Card/Narration writes); only the secret differs — mirrored, not shared.
from app_pkg.ingest import _bearer_token, _resolve_target_user
from models import Conversion, db
from services.document_conversions import (
    DOC_STATUS_FAILED,
    DOC_STATUS_PENDING,
    DOC_STATUS_READY,
    DOCUMENT_CONVERSION_TYPE,
    build_doc_metadata,
    discard_job_files,
    doc_metadata,
    doc_result_path,
    doc_source_path,
    doc_status,
    ensure_doc_convert_dir,
    read_result_file,
)
from tasks import convert_document_task

logger = logging.getLogger(__name__)

# Per-request upload cap for the document service. Deliberately far below the
# global 500-MB MAX_CONTENT_LENGTH (sized for audio): documents that large are
# not a real use case, and the cap is enforced from the Content-Length header
# BEFORE the multipart body is parsed (1.4 — reject before allocation). The
# multipart framing overhead counts against it; irrelevant at this size.
MAX_DOCUMENT_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

_HASH_CHUNK_BYTES = 1024 * 1024


def _oversize_response():
    """413 if the declared request body exceeds the cap — header-only check.

    Runs BEFORE any ``request.files`` access, i.e. before Werkzeug parses the
    multipart body at all. A missing/lying Content-Length is caught by the
    on-disk backstop after the save.
    """
    length = request.content_length
    if length is not None and length > MAX_DOCUMENT_UPLOAD_BYTES:
        return jsonify({
            'error': f'Datei zu groß. Maximal '
                     f'{MAX_DOCUMENT_UPLOAD_BYTES // (1024 * 1024)} MB.'
        }), 413
    return None


def _file_sha256(path):
    """Chunked sha256 of a file on disk (the P2 idempotency key, stored now)."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_count(path):
    """Cheap page count via PyMuPDF, ``None`` on any failure.

    Feeds the RQ envelope + the source metadata. A broken PDF yields ``None``
    (envelope floors to n=1) and then fails properly inside the task.
    """
    try:
        import fitz
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return None


# --- auth: dual path (session/bearer user OR service token) -------------------

def _authorize_document_access():
    """Resolve the acting user for both endpoints.

    Returns ``(user, None)`` on success or ``(None, (response, status))``.

    Path 1 — a logged-in user (``current_user``): session cookie or per-user
    bearer via the MOBILE-AUTH request_loader. Checked first; a bearer that IS
    a valid ApiToken never reaches the service-token compare.

    Path 2 — ``DOC_CONVERT_TOKEN``, mirrored from ``_authorize_narration_write``
    (own secret, fail-closed 503, constant-time 401, token never logged,
    target user via the shared Ingest resolver). The GET uses the same gate as
    the POST — a service caller must be able to submit *and* poll, which is why
    these routes are hand-authed instead of ``@login_required``.
    """
    if current_user.is_authenticated:
        return current_user, None

    expected = os.environ.get('DOC_CONVERT_TOKEN')
    if not expected:
        logger.warning('Document conversion rejected: DOC_CONVERT_TOKEN not configured')
        return None, (jsonify({'error': 'Dokument-API nicht konfiguriert.'}), 503)

    provided = _bearer_token()
    if provided is None or not hmac.compare_digest(provided.encode('utf-8'),
                                                   expected.encode('utf-8')):
        reason = 'missing bearer' if provided is None else 'token mismatch'
        logger.warning('Document conversion auth failed (%s) from %s',
                       reason, request.remote_addr)
        return None, (jsonify({'error': 'Nicht autorisiert.'}), 401)

    target = _resolve_target_user()
    if target is None:
        logger.error('Document conversion rejected: no target user (INGEST_USER=%r)',
                     os.environ.get('INGEST_USER'))
        return None, (jsonify({'error': 'Kein Ziel-Benutzer vorhanden.'}), 503)

    return target, None


# --- reconcile: web-side state machine for the DB-free worker -----------------

def _persist_metadata(conversion, metadata):
    conversion.metadata_json = json.dumps(metadata)
    db.session.commit()


def _fail_document_conversion(conversion, metadata, error):
    metadata['doc_status'] = DOC_STATUS_FAILED
    metadata['error'] = error
    _persist_metadata(conversion, metadata)


def reconcile_document_conversion(conversion):
    """Flip a ``pending`` document conversion to its terminal state on read.

    Idempotent (terminal states untouched), safe on every poll. Unlike the
    narration reconcile, the success signal is a **structured file read**, not
    file existence: ``result_<id>.json`` carries markdown + warnings.

    * result file parses      → ``ready``; markdown becomes ``content``,
                                warnings land in metadata, scratch files are
                                discarded post-commit.
    * result file unreadable  → ``failed`` (writes are atomic, so this is a
                                real defect; the file is kept for diagnosis).
    * RQ job failed           → ``failed`` + the exc_info **tail** (the
                                exception line lives at the end — NARR-FAIL).
    * RQ job gone / no job_id → ``failed`` ("Job nicht mehr auffindbar.").
    * RQ job queued/started   → stays ``pending``.
    * Redis unreachable       → stays ``pending`` (retried on the next poll).
    """
    if doc_status(conversion) != DOC_STATUS_PENDING:
        return

    metadata = doc_metadata(conversion)
    source_ext = metadata.get('source_format')

    if os.path.exists(doc_result_path(conversion.id)):
        payload = read_result_file(conversion.id)
        if payload is None:
            # Atomic writes make a half-written file impossible — an
            # unparseable result is a defect; keep the file for diagnosis.
            _fail_document_conversion(conversion, metadata, 'Ergebnisdatei unlesbar.')
            return
        conversion.content = payload.get('markdown') or ''
        metadata['doc_status'] = DOC_STATUS_READY
        metadata['warnings'] = [w for w in (payload.get('warnings') or [])
                                if isinstance(w, str)]
        metadata['error'] = None
        _persist_metadata(conversion, metadata)
        # Post-commit: the DB row is the artifact now, the volume files are
        # scratch (source is normally already gone via the task's finally).
        discard_job_files(conversion.id, source_ext=source_ext)
        return

    # No result file yet — consult the RQ job to tell "still converting" from
    # "dead". Late import: tests patch Job / redis_conn on app.py.
    import app as _app_module

    job_id = metadata.get('job_id')
    if not job_id:
        _fail_document_conversion(conversion, metadata, 'Job nicht mehr auffindbar.')
        discard_job_files(conversion.id, source_ext=source_ext)
        return
    try:
        job = _app_module.Job.fetch(job_id, connection=_app_module.redis_conn)
    except NoSuchJobError:
        _fail_document_conversion(conversion, metadata, 'Job nicht mehr auffindbar.')
        discard_job_files(conversion.id, source_ext=source_ext)
        return
    except Exception:
        # Transient Redis error — never fail an in-flight conversion over a blip.
        logger.warning('reconcile_document_conversion: RQ fetch failed for job %s',
                       job_id, exc_info=True)
        return
    if job.is_failed:
        error = (job.exc_info or '')[-2000:] or 'Konvertierung fehlgeschlagen.'
        _fail_document_conversion(conversion, metadata, error)
        discard_job_files(conversion.id, source_ext=source_ext)
    # queued / started / deferred → still converting, stays pending.


# --- response shape (the contract other services read) ------------------------

def _document_conversion_response(conversion):
    """The service-facing answer — deliberately NOT ``Conversion.to_dict()``.

    The library dict carries the app's inner life (lifecycle, tags, favorite);
    the contract carries exactly what a consuming service needs. P2 grows this
    by provenance / degradations / usage — the shape is the sprint's point.
    """
    metadata = doc_metadata(conversion)
    status = doc_status(conversion)
    return {
        'id': conversion.id,
        'status': status,
        'markdown': conversion.content if status == DOC_STATUS_READY else None,
        'warnings': metadata.get('warnings') or [],
        'error': metadata.get('error'),
        'source': {
            'filename': conversion.source_filename,
            'format': metadata.get('source_format'),
            'size_bytes': conversion.source_size_bytes,
            'page_count': metadata.get('page_count'),
        },
        'created_at': conversion.created_at.isoformat() if conversion.created_at else None,
    }


def register(app):
    # Late import: tests patch the RQ singletons on the top-level app.py module.
    import app as _app_module

    @app.route('/api/document-conversions', methods=['POST'])
    def api_create_document_conversion():
        """Submit a document for conversion → 202 ``{id, status, job_id}``.

        Multipart field ``file``. Size is checked from the Content-Length
        header BEFORE the body is parsed (1.4), with an on-disk backstop after
        the save. The upload lands on the shared volume under an id-derived
        name, a ``pending`` Conversion row is created (content fills in on
        reconcile), and the DB-free worker task is enqueued with a
        page-count-scaled RQ envelope.
        """
        target, err = _authorize_document_access()
        if err:
            return err

        oversize = _oversize_response()  # header check — before request.files
        if oversize:
            return oversize

        if 'file' not in request.files:
            return jsonify({'error': 'Kein Datei-Feld "file" im Request.'}), 400
        upload = request.files['file']
        if not upload.filename:
            return jsonify({'error': 'Keine Datei ausgewählt.'}), 400

        original_filename = upload.filename
        ext = os.path.splitext(secure_filename(original_filename))[1].lstrip('.').lower()
        if ext not in ACCEPTED_EXTENSIONS:
            return jsonify({
                'error': 'Dieser Dateityp wird nicht unterstützt. '
                         'Erlaubt: PDF, DOCX, PPTX, EML, HTML, TXT, MD.'
            }), 400

        # Spool to the shared volume first (same directory as the final path →
        # os.replace stays an atomic same-FS rename), verify, then create the
        # row. No DB rollback paths for upload problems.
        convert_dir = ensure_doc_convert_dir()
        tmp_f = tempfile.NamedTemporaryFile(
            dir=convert_dir, suffix='.upload', delete=False)
        tmp_path = tmp_f.name
        tmp_f.close()
        try:
            upload.save(tmp_path)

            size = os.path.getsize(tmp_path)
            if size > MAX_DOCUMENT_UPLOAD_BYTES:
                # Backstop for a missing/lying Content-Length (e.g. chunked).
                return jsonify({
                    'error': f'Datei zu groß. Maximal '
                             f'{MAX_DOCUMENT_UPLOAD_BYTES // (1024 * 1024)} MB.'
                }), 413
            if size == 0:
                return jsonify({'error': 'Leere Datei.'}), 400

            source_sha256 = _file_sha256(tmp_path)
            page_count = _pdf_page_count(tmp_path) if ext == 'pdf' else None

            conversion = Conversion(
                user_id=target.id,
                conversion_type=DOCUMENT_CONVERSION_TYPE,
                title=original_filename[:255],
                content='',  # fills in on the ready-reconcile
                source_filename=original_filename[:255],
                source_mimetype=upload.mimetype,
                source_size_bytes=size,
                # Service jobs are not triage material — the inbox is Oli's
                # working queue; converted documents shelve straight to archive.
                lifecycle_status='archive',
            )
            db.session.add(conversion)
            db.session.flush()  # id for the id-derived source path

            os.replace(tmp_path, doc_source_path(conversion.id, ext))

            metadata = build_doc_metadata(
                status=DOC_STATUS_PENDING, source_format=ext,
                source_sha256=source_sha256, page_count=page_count)
            conversion.metadata_json = json.dumps(metadata)
            db.session.commit()
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

        job = _app_module.task_queue.enqueue(
            convert_document_task,
            conversion.id, ext,
            meta={'user_id': target.id, 'conversion_id': conversion.id},
            job_timeout=doc_convert_job_timeout_for(page_count),
        )

        # job_id back into metadata — reconcile keys "still converting" vs
        # "gone" on it. (Two commits, like narration: the row must exist
        # before the caller can poll.)
        metadata['job_id'] = job.id
        conversion.metadata_json = json.dumps(metadata)
        db.session.commit()

        app.logger.info(
            f"Document conversion job {job.id} queued for conversion {conversion.id}")
        return jsonify({
            'id': conversion.id,
            'status': DOC_STATUS_PENDING,
            'job_id': job.id,
        }), 202

    @app.route('/api/document-conversions/<int:conversion_id>', methods=['GET'])
    def api_document_conversion_status(conversion_id):
        """Poll a conversion — status and, once ``ready``, the result itself.

        Same dual auth as the submit (a service caller must be able to poll).
        Owner- and type-scoped in one filter → a foreign, missing or
        wrong-type id is an indistinguishable 404.
        """
        target, err = _authorize_document_access()
        if err:
            return err

        conversion = Conversion.query.filter_by(
            id=conversion_id,
            user_id=target.id,
            conversion_type=DOCUMENT_CONVERSION_TYPE,
        ).first()
        if conversion is None:
            return jsonify({'error': 'Konvertierung nicht gefunden.'}), 404

        reconcile_document_conversion(conversion)
        return jsonify(_document_conversion_response(conversion))

    # NO csrf exemption here — see the module docstring: bearer presence
    # already skips the CSRF inversion, and cookie sessions (a legitimate auth
    # path on this surface, unlike Ingest/Narration) must keep CSRF.
