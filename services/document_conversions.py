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

metadata_json contract (v2, DOC-API P2):

  {
    "doc_status": "pending" | "ready" | "failed",
    "job_id": "<rq job id>",
    "mode": "cloud" | "lokal",           # effective mode of THIS job
    "budget_eur": 1.0,                   # per-job cap, frozen at submit
    "source_format": "pdf",              # lowercased extension
    "source_sha256": "<hex>",            # content hash (idempotency key)
    "page_count": 12 | null,             # PDFs only; null elsewhere
    "provenance_unit": "page"|"document",# set on ready (see below)
    "provenance": ["deterministisch"],   # one entry per unit, in order
    "degradations": [{"code","message","pages"}],  # structured, in-answer
    "usage": {"model_calls",             # set on ready; null = unknown
              "cost_eur"} | null,
    "error": null | "..."                # set when doc_status == 'failed'
  }

Provenance semantics (the contract other services rely on):

* Values: ``deterministisch`` (extracted verbatim, no model involved) <
  ``ocr`` (classic OCR) < ``modell`` (a generative decoder produced or could
  have produced the text). The ordering matters: where a backend cannot
  attribute units individually, the value is **rounded UP** — a consumer must
  never read ``deterministisch`` on text that may be model-generated
  (Bake-off lesson: unmarked mixed provenance is the worst silent failure).
* Unit: ``page`` where the pipeline knows real page boundaries (PDF with a
  readable page count) — one entry per page, list order == page order.
  ``document`` everywhere else — exactly ONE entry covering the whole
  document. Non-PDF formats have no stable page concept in this pipeline
  (unstructured processes an element stream; DOCX "pages" are a renderer
  artifact), so the honest unit is the document. The unit is carried IN the
  answer (``provenance_unit``), never implied by the format.
"""
import json
import os

from app_pkg.config import OUTPUT_DIR

DOC_STATUS_PENDING = 'pending'
DOC_STATUS_READY = 'ready'
DOC_STATUS_FAILED = 'failed'
DOC_STATUSES = (DOC_STATUS_PENDING, DOC_STATUS_READY, DOC_STATUS_FAILED)

DOCUMENT_CONVERSION_TYPE = 'document_conversion'

# Per-job mode (locked decision 1): the caller picks, a CONVERTER setting
# provides the default. German values verbatim from the sprint contract.
MODE_CLOUD = 'cloud'
MODE_LOCAL = 'lokal'
DOC_MODES = (MODE_CLOUD, MODE_LOCAL)

# Provenance values (locked decision 3), ordered by trust: see module docstring.
PROVENANCE_DETERMINISTIC = 'deterministisch'
PROVENANCE_OCR = 'ocr'
PROVENANCE_MODEL = 'modell'
PROVENANCE_VALUES = (PROVENANCE_DETERMINISTIC, PROVENANCE_OCR, PROVENANCE_MODEL)

UNIT_PAGE = 'page'
UNIT_DOCUMENT = 'document'

# Engine generation for the idempotency key (DOC-LOCAL P3). Dedup answers a
# re-submit with a STORED result — that is only honest while the stored
# result is what today's engines would produce. Live-hit 2026-08-16: the
# DOC-QEMU-VERIFY lokal row (legacy text layer, deterministisch×280) masked
# the freshly deployed mineru engine for the identical file, with no user
# path around it. Rows written before this constant existed carry no
# generation and count as 1 → they never match, so every pre-DOC-LOCAL
# result is devalued exactly once, without a migration. ONE global counter,
# deliberately not per format: over-caution is free at this volume, and a
# per-format ledger is bookkeeping nobody maintains.
#
# ⚠️ BUMP THIS on EVERY change to an engine or to a result assembly
# (backend swap, invocation change, serializer/assembly rule change —
# DOC-ROUTE included): if a re-submit today would produce a different
# result, stored rows must stop answering for it.
# Bumps: 2 = DOC-LOCAL (mineru replaces the text layer) · 3 = DOC-WEB (one
# router for both entrances; scan pages named on the fallback; default mode
# ``lokal`` — results of mode-less submits change for every PDF).
DOC_CONVERT_ENGINE_GENERATION = 3

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


def degradation(code, message, pages=None):
    """One structured degradation entry: what could not be done cleanly.

    ``code`` is a stable snake_case slug for machines, ``message`` is German
    for humans (house microcopy), ``pages`` optionally names the affected
    pages (1-based) — ``None`` means "the whole job".
    """
    return {'code': code, 'message': message, 'pages': pages}


# Degradation codes used by P2 (the contract doc lists them with meanings):
DEGRADATION_BUDGET_EXCEEDED = 'budget_exceeded'
DEGRADATION_CLOUD_UNAVAILABLE = 'cloud_unavailable'
DEGRADATION_SERIALIZER = 'serializer'
DEGRADATION_PROVENANCE_DOCUMENT_ONLY = 'provenance_document_only'
# DOC-ENGINE: the format's chosen backend yielded nothing usable and the
# legacy path took over (e.g. trafilatura found no main content in an HTML
# file) — the result is still ready, the switch is named here.
DEGRADATION_BACKEND_FALLBACK = 'backend_fallback'
# DOC-WEB 2.3: on the text-layer fallback a page is a scan — the layer is
# empty there BY NATURE, not by defect; the entry names the pages so the
# empty stretch is explained instead of silently served.
DEGRADATION_SCAN_TEXT_LAYER_EMPTY = 'scan_text_layer_empty'


def build_result_payload(markdown, *, provenance_unit, provenance,
                         degradations=None, usage=None):
    """The ONE result shape a conversion run produces (``result_<id>.json``).

    Shared by the worker task, the paged pipeline (services/document_pipeline)
    and the tests, so whatever backend fills it — today's blackbox or the
    follow-up sprint's router — reconcile reads the identical structure.

    ``usage`` is ``{'model_calls': int, 'cost_eur': float}`` when the run can
    account for itself, or ``None`` when it genuinely cannot (the legacy cloud
    engine neither reports calls nor costs — ``None`` is honest, ``0`` would
    be a claim).
    """
    return {
        'markdown': markdown,
        'provenance_unit': provenance_unit,
        'provenance': list(provenance),
        'degradations': list(degradations or []),
        'usage': usage,
    }


def build_doc_metadata(*, status=DOC_STATUS_PENDING, mode=None, budget_eur=None,
                       source_format=None, source_sha256=None, page_count=None,
                       error=None):
    """Build the ``metadata_json`` dict for a document_conversion Conversion.

    The result fields (provenance_unit / provenance / degradations / usage)
    are absent at submit time and merged in by the ready-reconcile from the
    worker's result payload. ``engine_generation`` stamps the writer's
    generation — part of the dedup key (see the constant above).
    """
    return {
        'doc_status': status,
        'engine_generation': DOC_CONVERT_ENGINE_GENERATION,
        'mode': mode,
        'budget_eur': budget_eur,
        'source_format': source_format,
        'source_sha256': source_sha256,
        'page_count': page_count,
        'provenance_unit': None,
        'provenance': None,
        'degradations': [],
        'usage': None,
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
