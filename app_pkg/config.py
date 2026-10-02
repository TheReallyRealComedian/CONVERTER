"""Shared configuration constants.

Kept dependency-free so tasks.py (worker container) and the web container
can both import from here without pulling in Flask or service SDKs.
"""
import math
import os
import re
import uuid
from zoneinfo import ZoneInfo

from rq.serializers import JSONSerializer

from services.mineru_invocation import (
    MINERU_TIMEOUT_BASE_SECONDS,
    MINERU_TIMEOUT_PER_PAGE_SECONDS,
    mineru_run_timeout_for,
)

# NOTION-TZ: the ONE local zone of this single-user app (Oliver sits in
# Berlin). learn buckets per local day, library reads recorder filenames as
# Berlin wall clock, and the Notion send attaches it to zone-less meeting
# times (Notion reads an offset-free datetime as UTC). Server-fixed on
# purpose — not the device's zone.
LOCAL_TZ = ZoneInfo('Europe/Berlin')

# Shared podcast output directory.
# Must match the docker-compose ``podcast_data`` volume that is mounted
# at the same path in both the web and worker containers.
OUTPUT_DIR = '/app/output_podcasts'

# SEC-REDIS-AUTH (F-7): the ONE serializer every RQ touch point uses — the
# web Queue and ``app.fetch_job`` (app.py), the worker's Queue + Worker
# (worker.py). RQ's default is pickle: whoever can write to Redis could hand
# the worker a job that executes code on unpickling. JSON only carries data.
# Never set a serializer anywhere else — a Queue or Worker left on the
# default fails every job with ``DeserializationError`` (measured against
# rq 2.8.0 + Redis 8.4, see the sprint doc).
RQ_SERIALIZER = JSONSerializer

# JOB-ID-REUSE: the job mark. ``Conversion.id`` is reused by SQLite once the
# highest row is deleted, so no job file on the shared volume may hang on the
# row id alone — two jobs would share one name (the old job's result served
# to the new row; the old job's ``finally`` deleting the new job's source).
# The web side creates the mark (``new_job_mark``) BEFORE anything touches
# the volume, hands it to RQ as the job id and to the task as its first
# argument, and stores it as ``metadata['job_id']``; every job file is named
# after it and the reconcile looks only for the name of its own job.
#
# The reconciles and the delete cleanup read the mark back from
# ``metadata_json`` — which a client can write (``POST /api/conversions``
# takes a metadata bag). So the path helpers of the three job modules build a
# name only from a value that IS a mark (and a source extension that is one):
# no dot, no separator, nothing that could leave the job's namespace.
_JOB_MARK_RE = re.compile(r'[A-Za-z0-9_-]{1,64}')
_JOB_SOURCE_EXT_RE = re.compile(r'[a-z0-9]{1,8}')


def new_job_mark():
    """A fresh job mark — a uuid4, the format RQ would have picked itself."""
    return str(uuid.uuid4())


def is_job_mark(value):
    """Whether ``value`` may become part of a job file name."""
    return isinstance(value, str) and _JOB_MARK_RE.fullmatch(value) is not None


def job_mark(value):
    """``value`` if it is a job mark, else ``ValueError`` (never a path)."""
    if not is_job_mark(value):
        raise ValueError('not a job mark')
    return value


def job_source_ext(value):
    """``value`` if it is a plain lowercase file extension, else ``ValueError``."""
    if not isinstance(value, str) or _JOB_SOURCE_EXT_RE.fullmatch(value) is None:
        raise ValueError('not a job source extension')
    return value


# Upstream timeouts, centralised in one place. Two of them govern faithful
# narration and are deliberately related:
#   (a) TIMEOUT_TTS_SYNTH_SECONDS — the absolute per-call Cloud-TTS gRPC
#       deadline (the actual cap on a wedged synth call), and
#   (b) rq_job_timeout_for(n) — the per-render RQ envelope, scaled from the
#       chunk count and derived *from* (a) with a floor, so raising the
#       deadline can never mid-flight-kill a genuinely-progressing render.
# TIMEOUT_GEMINI_SECONDS and TIMEOUT_DEEPGRAM_SECONDS are independent
# single-call SDK timeouts and stand on their own.
#
# TIMEOUT_DEEPGRAM_SECONDS is the per-request SDK deadline for one
# transcribe_file call. Since DIARIZE it must cover a *single* request of up to
# 90 min audio (MAX_AUDIO_DURATION_SECONDS=5400, up to MAX_FILE_SIZE_MB=500) so
# meetings run as one request with consistent speakers — not the old ≤10-min
# chunk. Deepgram's server-side processing stays fast (<2 min for ~90 min), but
# the deadline must also span the upload of a large file + response. 1200s (20
# min) gives ample headroom and stays well under gunicorn's 1800s CMD timeout,
# so an overrun surfaces as a clean SDK error rather than a gunicorn kill.
TIMEOUT_GEMINI_SECONDS = 300
TIMEOUT_DEEPGRAM_SECONDS = 1200

# Live transcription (SEC-DG-TOKEN): the browser opens its Deepgram WebSocket
# with a short-lived token minted by Deepgram's grant, never with the API key.
# The SERVER sets the lifetime — 30 s is Deepgram's own default, and the page
# fetches the token right before it connects (after the mic permission, so
# the prompt cannot eat the lifetime). The grant runs in a web thread and
# therefore carries its own per-call deadline; the service makes one attempt
# (no SDK retries — their backoff would stretch the deadline several times
# over).
DEEPGRAM_LIVE_TOKEN_TTL_SECONDS = 30
TIMEOUT_DEEPGRAM_GRANT_SECONDS = 10


def _env_positive_float(name, default):
    """Parse a positive float from ``os.environ[name]``, else ``float(default)``.

    A malformed / non-positive value must never brick both containers at import
    time (adversarial #7): missing, junk, or <= 0 all resolve to the default.
    """
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return val if val > 0 else float(default)


# Layer 1 — absolute per-call gRPC deadline for the single unbounded Cloud-TTS
# ``synthesize_speech`` call (the #80 hang: a wedged gRPC call parked the RQ
# work-horse forever). Enforced by grpc's C-core timer, signal-independent; on
# expiry raises ``DeadlineExceeded``, which the renderer already treats as
# retryable — so no new error-handling code.
TIMEOUT_TTS_SYNTH_SECONDS = _env_positive_float('NARRATION_TTS_TIMEOUT_SECONDS', 120.0)

# Renderer retry shape — MUST mirror narration_render._synthesize_with_retry
# (max_retries=2 → 3 attempts total; backoff sleep(1)+sleep(2)=3s between them).
# Used only to size the RQ envelope so it can never be tighter than the
# worst-case per-chunk cost.
_TTS_MAX_RETRIES = 2
_TTS_RETRY_BACKOFF_TOTAL = 3

# Layer 2 — per-render RQ job_timeout scaled from the chunk count instead of a
# flat 600s. BASE = SDK init + WAV concat + shutil.move headroom; per-chunk is
# floored so the default (T=120 → 3*120+3=363) still yields the historical
# n=1 == 600 (behaviour-neutral), and tracks T upward for larger deadlines.
TIMEOUT_RQ_JOB_BASE_SECONDS = 200
_RQ_PER_CHUNK_FLOOR = 400
TIMEOUT_RQ_JOB_PER_CHUNK_SECONDS = max(
    _RQ_PER_CHUNK_FLOOR,
    math.ceil((_TTS_MAX_RETRIES + 1) * TIMEOUT_TTS_SYNTH_SECONDS + _TTS_RETRY_BACKOFF_TOTAL),
)
# Backstop against a pathologically large chunk count (bites only ~n >= 36).
TIMEOUT_RQ_JOB_HARD_CAP = 4 * 3600


def rq_job_timeout_for(n):
    """RQ ``job_timeout`` (seconds) for a render of ``n`` chunks.

    ``min(BASE + PER_CHUNK * n, HARD_CAP)``. PER_CHUNK >= (max_retries+1)*T +
    backoff for every T, so a genuinely-progressing n-chunk render is never
    false-killed; the hard cap only bites for a pathological chunk count.
    """
    scaled = TIMEOUT_RQ_JOB_BASE_SECONDS + TIMEOUT_RQ_JOB_PER_CHUNK_SECONDS * max(n, 0)
    return min(scaled, TIMEOUT_RQ_JOB_HARD_CAP)


# Back-compat export: existing imports + test_narration_write.py:16 still resolve
# this name. == 600 at the default deadline (n=1), so behaviour-neutral.
TIMEOUT_RQ_JOB_SECONDS = rq_job_timeout_for(1)


# --- DOC-API: RQ envelope for a document conversion, scaled from the page
# count (the work-set metric — pages are what the PDF pipeline iterates).
# PER_PAGE mirrors the narration lesson: it must cover the worst-case cost of
# one page, which is a single Gemini-Vision call at its full per-call deadline
# (TIMEOUT_GEMINI_SECONDS) — a genuinely-progressing conversion is never
# false-killed; the shared HARD_CAP backstops pathological page counts.
# Non-PDF formats (and PDFs whose page count can't be read) run as n=1: the
# unstructured 'fast' path is CPU-seconds, so BASE + one page is ample.
TIMEOUT_DOC_JOB_BASE_SECONDS = 300
TIMEOUT_DOC_JOB_PER_PAGE_SECONDS = TIMEOUT_GEMINI_SECONDS

# --- DOC-LOCAL: the lokal-mode envelope rides the measured mineru curve —
# ~61 s fixed model start + ~2.5 s/page (fitted 2..280 pages; 280 pages ran
# 766 s ≈ 13 min). The container deadline is 300 + 10 × n, ~4× margin for GPU
# contention with Olis ComfyUI use; since SEC-SOCKET it lives in
# ``services/mineru_invocation.mineru_run_timeout_for`` — the ONE source this
# envelope is derived from. On top of it: a constant 300 s for everything
# besides the deadline — the launcher's reply margin (copy-in, container
# kill, copy-out, volume cleanup: ``LAUNCHER_REPLY_MARGIN_SECONDS`` = 240 s,
# the worker's extra HTTP wait), a possible text-layer fallback pass and the
# result write. Chain, testnailed: deadline < worker HTTP timeout < envelope
# for every page count the launcher accepts.
# 280 pages: envelope 3400 s (~57 min) vs deadline 3100 s vs measured 766 s.
# The cloud envelope needs no lokal term: 300 s/page ≥ 10 s/page covers any
# mid-flight cloud→mineru switch by construction.
TIMEOUT_DOC_JOB_LOCAL_MARGIN_SECONDS = 300
TIMEOUT_DOC_JOB_LOCAL_BASE_SECONDS = (MINERU_TIMEOUT_BASE_SECONDS
                                      + TIMEOUT_DOC_JOB_LOCAL_MARGIN_SECONDS)
TIMEOUT_DOC_JOB_LOCAL_PER_PAGE_SECONDS = MINERU_TIMEOUT_PER_PAGE_SECONDS


def doc_convert_job_timeout_for(page_count, mode=None):
    """RQ ``job_timeout`` (seconds) for converting a ``page_count``-page document.

    ``min(BASE + PER_PAGE * max(n, 1), HARD_CAP)`` — same shape as
    ``rq_job_timeout_for``; ``None`` / non-int / < 1 all floor to n=1.
    ``mode='lokal'`` (string literal here — the ``MODE_LOCAL`` constant lives
    downstream of config) switches to the mineru curve above; every other
    mode keeps the Gemini-Vision envelope unchanged.
    """
    n = page_count if isinstance(page_count, int) and page_count > 0 else 1
    if mode == 'lokal':
        scaled = mineru_run_timeout_for(n) + TIMEOUT_DOC_JOB_LOCAL_MARGIN_SECONDS
    else:
        scaled = (TIMEOUT_DOC_JOB_BASE_SECONDS
                  + TIMEOUT_DOC_JOB_PER_PAGE_SECONDS * n)
    return min(scaled, TIMEOUT_RQ_JOB_HARD_CAP)


# --- DOC-API P2: per-job cost cap (locked decision 2) --------------------------
#
# Hard cap per job, with degradation to the local path instead of aborting.
# The VALUE is a deliberately conservative placeholder until real document
# sizes have run through (sprint note) — visible here, overridable per env,
# frozen onto each job at submit time. At the bake-off-measured cloud price
# (1.48 ct/page, gemini medium over 492 pages) 1 € buys ~67 pages; a tight
# start is safe because the cap degrades — the caller always gets a result.
DOC_CONVERT_BUDGET_EUR = _env_positive_float('DOC_CONVERT_BUDGET_EUR', 1.0)

# Estimated cloud cost per page, used for the pre-flight budget check while
# the legacy engine cannot account for itself (it reports neither calls nor
# spend). Bake-off measurement as the default; env-overridable so a model
# price change never needs a code change.
DOC_CONVERT_CLOUD_CENT_PER_PAGE = _env_positive_float(
    'DOC_CONVERT_CLOUD_CENT_PER_PAGE', 1.48)


# --- SYNC-FREEZE P3: RQ envelope for a transcription job ------------------------
#
# The transcription runs as a job on the worker (``tasks.transcribe_audio_task``).
# Its work set is the Deepgram request count, which the chunking thresholds of
# ``services.deepgram_service.DeepgramService`` decide — MIRRORED here because
# config must stay SDK-free (tests/test_transcriptions.py pins the mirror to
# the service's class attributes, so a threshold change cannot drift):
#   <= 90 min  → ONE request (consistent speakers across the meeting, DIARIZE)
#   >  90 min  → 30-min chunks overlapping by 5 s, each with up to 2 retries
#               and exponential backoff (2 s + 4 s) — the chunk path has the
#               retries, the single request has none; the envelope uses the
#               worst case of a chunk for both (never false-kill a live job).
AUDIO_SINGLE_REQUEST_MAX_SECONDS = 5400
AUDIO_CHUNK_SECONDS = 1800
AUDIO_CHUNK_OVERLAP_SECONDS = 5
_AUDIO_MAX_RETRIES = 2
_AUDIO_RETRY_BACKOFF_TOTAL = 2 + 4

# BASE covers what is not a Deepgram call: reading a file of up to 500 MB
# from the volume, ffprobe + ffmpeg chunk extraction (streams, seconds), the
# result write. PER_CHUNK = (retries + 1) × the per-request SDK deadline +
# backoff = 3606 s; a genuinely progressing chunk is never killed mid-flight.
# Measured reality is far below (31.7 min ≈ 25–100 s, upload-bound). Named
# property: chunking starts at FOUR chunks (just above 90 min → 5401/1795),
# and 300 + 4 × 3606 already exceeds the shared 4-h HARD_CAP — so every
# single-request file gets 3906 s and every chunked file rides the cap.
TIMEOUT_AUDIO_JOB_BASE_SECONDS = 300
TIMEOUT_AUDIO_JOB_PER_CHUNK_SECONDS = math.ceil(
    (_AUDIO_MAX_RETRIES + 1) * TIMEOUT_DEEPGRAM_SECONDS + _AUDIO_RETRY_BACKOFF_TOTAL)


def audio_chunk_count(duration_seconds):
    """Deepgram requests a recording of ``duration_seconds`` will take.

    Mirrors ``AudioChunker.needs_splitting``'s estimate: one request up to the
    single-request maximum, else ``ceil(duration / (chunk − overlap))``.
    ``None`` / non-numeric / <= 0 (unreadable file) → 1.
    """
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        return 1
    if duration <= 0 or duration <= AUDIO_SINGLE_REQUEST_MAX_SECONDS:
        return 1
    step = AUDIO_CHUNK_SECONDS - AUDIO_CHUNK_OVERLAP_SECONDS
    return max(1, math.ceil(duration / step))


def transcribe_job_timeout_for(duration_seconds):
    """RQ ``job_timeout`` (seconds) for transcribing ``duration_seconds`` of audio.

    ``min(BASE + PER_CHUNK * chunks, HARD_CAP)`` — the shape of
    ``rq_job_timeout_for`` / ``doc_convert_job_timeout_for``.
    """
    scaled = (TIMEOUT_AUDIO_JOB_BASE_SECONDS
              + TIMEOUT_AUDIO_JOB_PER_CHUNK_SECONDS * audio_chunk_count(duration_seconds))
    return min(scaled, TIMEOUT_RQ_JOB_HARD_CAP)


# --- SYNC-FREEZE: SQLite under several gunicorn processes ---------------------
#
# Since SYNC-FREEZE the web app runs as N gunicorn worker PROCESSES (Dockerfile
# CMD) that all open the same SQLite file. Two connection-level settings make
# that safe; both are applied to every new connection by
# ``app_pkg._register_sqlite_pragmas``:
#
# * ``journal_mode=WAL`` — readers never block a writer and a writer never
#   blocks readers; only writers serialise among themselves. The previous
#   state was the rollback journal (``delete``), where one writer locks the
#   whole file against every reader. WAL is persisted IN the database file,
#   so the switch survives restarts; re-issuing the pragma on a file that is
#   already WAL is a read transaction, not a lock.
# * ``busy_timeout`` — how long a connection waits for a lock before raising
#   ``database is locked``. pysqlite's default is 5 s and had never been set
#   explicitly. 10 s: every write transaction of this app is milliseconds (a
#   rating, a highlight, a settings blob; the startup migration is the longest
#   and still sub-second), so 10 s absorbs any burst of N processes writing at
#   once — and a lock held LONGER than that is a bug (a transaction left open
#   around an external call) that should surface as an error in the log
#   rather than park a worker's single request thread even longer.
SQLITE_BUSY_TIMEOUT_SECONDS = 10


# --- SYNC-FREEZE P2: sync views on a thread pool, per process -----------------
#
# ``app_pkg.asgi`` runs every WSGI call on a dedicated ThreadPoolExecutor of
# this size (per gunicorn process) instead of asgiref's single thread. The
# bound is deliberate: each running view may hold one SQLAlchemy connection,
# and the engine's default pool allows 15 per process (pool_size 5 +
# max_overflow 10; a 16th waiter would hit pool_timeout after 30 s) — eight
# threads stay well inside that. Eight is also far more than one user, the
# iOS app and an agent ever have in flight at once (a transcription, two PDF
# renders, a burst of ratings and the pollers fit side by side); beyond it,
# requests queue in the executor instead of failing. Process count
# (Dockerfile CMD) multiplies this.
WEB_SYNC_THREADS = 8
