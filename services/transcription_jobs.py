"""Persistence + shared-volume helpers for the transcription job (SYNC-FREEZE P3).

Pure module — no Flask context, no SDK. Mirrors ``services/document_conversions``
for ``conversion_type='audio_transcription'`` rows created by the job path:
the job state lives in ``metadata_json`` (no schema touch), the files in an
own namespace directory on the shared ``podcast_data`` volume (the one volume
both containers mount; the files are scratch and cleaned up after reconcile).

Why the transcription is a job at all — honestly: NOT against the freeze.
Since SYNC-FREEZE P2 a synchronous transcription parks nobody (sync views
run on a thread pool). The job buys what a synchronous request cannot:

* **progress** — the page polls and shows elapsed time instead of a request
  that may or may not still be alive behind a spinner;
* **a closed tab survives** — the worker finishes regardless, the library
  row reconciles on the next read (``GET /api/transcriptions/<id>``, and the
  library detail page polls it while the row is pending);
* **repeatability** — the same file re-submitted is answered from the stored
  result (user + content hash + language), a failed job is re-run by simply
  re-submitting (``failed`` never dedups).

And, per construction, it removes the one thread-safety exposure P2 created:
``AudioChunker`` keeps per-request state on the ``DeepgramService`` instance
(``_tmp_path`` …); in the RQ worker one job runs at a time and the task
builds its own service instance — no two transcriptions ever share one.

File layout under ``TRANSCRIPTION_DIR`` (id-derived names, never user input):

* ``source_<id>.<ext>`` — the uploaded audio, written by the web process,
  read (and finally deleted) by the worker.
* ``result_<id>.json``  — the worker's structured result (transcript + facts),
  written atomically (tmp + ``os.replace``), so reconcile can never read a
  half-written file: an unparseable result is a defect, not a race.

metadata_json contract:

  {
    "transcription_status": "pending" | "ready" | "failed",
    "job_id": "<rq job id>",
    "language": "de",
    "source_format": "wav",
    "source_sha256": "<hex>",            # idempotency key (user + hash + language)
    "duration_seconds": 1901.1 | null,   # ffprobe at submit → RQ envelope
    "file_size_mb": 87.03,
    "transcript_length": 26742,          # set on ready
    "recorded_at": "...",                # MCP1 capture when derivable
    "recorded_at_source": "filename" | "client",
    "error": null | "..."                # set when transcription_status == 'failed'
  }

Rows saved by the pre-P3 synchronous flow carry none of these keys; they are
complete transcripts and count as ``ready`` (``transcription_status``).
"""
import hashlib
import json
import os
import subprocess

from app_pkg.config import OUTPUT_DIR

TRANSCRIPTION_TYPE = 'audio_transcription'

STATUS_PENDING = 'pending'
STATUS_READY = 'ready'
STATUS_FAILED = 'failed'

# Namespace directory on the shared volume. Tests monkeypatch THIS module
# global; every path function below reads it at call time.
TRANSCRIPTION_DIR = os.path.join(OUTPUT_DIR, 'transcriptions')

_HASH_CHUNK_BYTES = 1024 * 1024


def ensure_transcription_dir():
    """Create the namespace directory (idempotent). Called before any write."""
    os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
    return TRANSCRIPTION_DIR


def transcription_source_path(conversion_id, source_ext):
    """``source_<id>.<ext>`` — web writes it, the worker derives the SAME path."""
    return os.path.join(TRANSCRIPTION_DIR, f'source_{conversion_id}.{source_ext}')


def transcription_result_path(conversion_id):
    """``result_<id>.json`` — the worker's structured result."""
    return os.path.join(TRANSCRIPTION_DIR, f'result_{conversion_id}.json')


def write_result_file(conversion_id, payload):
    """Atomically write the worker's result JSON (tmp + same-dir ``os.replace``)."""
    ensure_transcription_dir()
    final_path = transcription_result_path(conversion_id)
    tmp_path = final_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, final_path)
    return final_path


def read_result_file(conversion_id):
    """Parsed result JSON, or ``None`` if missing/unreadable/not an object."""
    try:
        with open(transcription_result_path(conversion_id), encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else {}


def discard_job_files(conversion_id, source_ext=None):
    """Best-effort unlink of a job's volume files (result + optionally source).
    The DB row is the artifact; the files are scratch. Never raises."""
    paths = [transcription_result_path(conversion_id)]
    if source_ext:
        paths.append(transcription_source_path(conversion_id, source_ext))
    for path in paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


def file_sha256(path):
    """Chunked sha256 of a file on disk (the idempotency key)."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration_seconds(path):
    """Audio duration via ffprobe (reads headers only), ``None`` on any failure.

    Feeds the RQ envelope at submit time. A file ffprobe cannot read gets the
    single-request envelope and then fails properly inside the task.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'json', path],
            capture_output=True, text=True, timeout=60, check=False)
        if result.returncode != 0:
            return None
        duration = float(json.loads(result.stdout)['format']['duration'])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return None
    return duration if duration > 0 else None


def transcription_metadata(conversion):
    """Parsed ``metadata_json`` as a dict (lenient: garbage → ``{}``)."""
    raw = getattr(conversion, 'metadata_json', None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def transcription_status(conversion):
    """Job status of a row; legacy rows without the key are complete → ready."""
    status = transcription_metadata(conversion).get('transcription_status')
    return status if status in (STATUS_PENDING, STATUS_READY, STATUS_FAILED) else STATUS_READY


def build_transcription_metadata(language, source_format, source_sha256,
                                 duration_seconds, file_size_mb,
                                 recorded_at=None, recorded_at_source=None):
    """The v1 metadata bag of a freshly submitted job (status pending)."""
    metadata = {
        'transcription_status': STATUS_PENDING,
        'job_id': None,
        'language': language,
        'source_format': source_format,
        'source_sha256': source_sha256,
        'duration_seconds': duration_seconds,
        'file_size_mb': file_size_mb,
        'transcript_length': None,
        'error': None,
    }
    if recorded_at is not None:
        metadata['recorded_at'] = recorded_at
        metadata['recorded_at_source'] = recorded_at_source
    return metadata
