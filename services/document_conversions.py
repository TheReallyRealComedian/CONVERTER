"""Persistence + shared-volume helpers for the document-conversion API (DOC-API).

Pure module — no Flask context, no SDK. Mirrors ``services/narration_library.py``
for the new ``conversion_type='document_conversion'`` flow: the job state lives
in ``metadata_json`` (no schema touch), the files live in an own namespace
directory on the shared ``podcast_data`` volume.

Why ``podcast_data`` and not an own volume: it is the one volume already
mounted in BOTH the web and the worker container, the job files here are small
(source document + one result JSON, cleaned up after reconcile — not GB-sized
WAVs), and a subdirectory gives the same namespace isolation an extra volume
would, without touching docker-compose in two places. The misleading volume
*name* is a pre-existing cosmetic debt, not this sprint's.

File layout under ``DOC_CONVERT_DIR`` (id-derived names, never user input):

* ``source_<id>.<ext>``  — uploaded original, written by the web process,
  read (and finally deleted) by the worker.
* ``result_<id>.json``   — the worker's **structured** result. Unlike the
  narration WAV ("file exists" == done), a conversion result carries markdown
  plus warnings, so reconcile must *read* it. Written atomically
  (tmp + ``os.replace``) so the web side can never observe a half-written file:
  an unparseable result file is therefore a real defect, not a race.

metadata_json contract (v1, DOC-API P1 — grows provenance/usage in P2):

  {
    "doc_status": "pending" | "ready" | "failed",
    "job_id": "<rq job id>",
    "source_format": "pdf",              # lowercased extension
    "source_sha256": "<hex>",            # content hash (P2 idempotency key)
    "page_count": 12 | null,             # PDFs only; null elsewhere
    "warnings": ["..."],                 # serializer degradations (ready)
    "error": null | "..."                # set when doc_status == 'failed'
  }
"""
import json
import os

from app_pkg.config import OUTPUT_DIR

DOC_STATUS_PENDING = 'pending'
DOC_STATUS_READY = 'ready'
DOC_STATUS_FAILED = 'failed'
DOC_STATUSES = (DOC_STATUS_PENDING, DOC_STATUS_READY, DOC_STATUS_FAILED)

DOCUMENT_CONVERSION_TYPE = 'document_conversion'

# Namespace directory on the shared volume. Tests monkeypatch THIS module
# global; every path function below reads it at call time.
DOC_CONVERT_DIR = os.path.join(OUTPUT_DIR, 'doc_conversions')


def ensure_doc_convert_dir():
    """Create the namespace directory (idempotent). Called before any write."""
    os.makedirs(DOC_CONVERT_DIR, exist_ok=True)
    return DOC_CONVERT_DIR


def doc_source_path(conversion_id, source_ext):
    """Shared-volume path of the uploaded source: ``source_<id>.<ext>``.

    Derived from the id + the validated extension only — the original filename
    never reaches the filesystem. Web writes it, the worker derives the SAME
    path from the same arguments (single definition, no path drift).
    """
    return os.path.join(DOC_CONVERT_DIR, f'source_{conversion_id}.{source_ext}')


def doc_result_path(conversion_id):
    """Shared-volume path of the worker's structured result: ``result_<id>.json``."""
    return os.path.join(DOC_CONVERT_DIR, f'result_{conversion_id}.json')


def write_result_file(conversion_id, payload):
    """Atomically write the worker's result JSON onto the shared volume.

    tmp file + ``os.replace`` in the same directory (same filesystem →
    atomic rename), so the web-side reconcile can never read a half-written
    JSON. Returns the final path.
    """
    ensure_doc_convert_dir()
    final_path = doc_result_path(conversion_id)
    tmp_path = final_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp_path, final_path)
    return final_path


def read_result_file(conversion_id):
    """Read + parse the worker's result JSON, or ``None`` if unreadable.

    ``None`` covers missing file, invalid JSON and a non-object top level.
    Because writes are atomic, an existing-but-unparseable file is a genuine
    defect (reconcile flips such a row to ``failed`` rather than retrying
    forever).
    """
    try:
        with open(doc_result_path(conversion_id), encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else {}


def discard_job_files(conversion_id, source_ext=None):
    """Best-effort unlink of a job's volume files (result + optionally source).

    Called post-commit by reconcile once the result is persisted in the DB
    (the files are scratch, not the artifact — the narration WAV *is* the
    artifact and stays; a conversion's artifact is the DB row). Never raises.
    """
    paths = [doc_result_path(conversion_id)]
    if source_ext:
        paths.append(doc_source_path(conversion_id, source_ext))
    for path in paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


def build_doc_metadata(*, status=DOC_STATUS_PENDING, source_format=None,
                       source_sha256=None, page_count=None, warnings=None,
                       error=None):
    """Build the ``metadata_json`` dict for a document_conversion Conversion."""
    return {
        'doc_status': status,
        'source_format': source_format,
        'source_sha256': source_sha256,
        'page_count': page_count,
        'warnings': list(warnings or []),
        'error': error,
    }


def doc_metadata(conversion):
    """Parse a Conversion's ``metadata_json`` into a dict, robust to corruption."""
    raw = getattr(conversion, 'metadata_json', None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def doc_status(conversion):
    """The ``doc_status`` from a Conversion's metadata, ``''`` if unset/broken."""
    return doc_metadata(conversion).get('doc_status') or ''
