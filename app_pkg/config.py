"""Shared configuration constants.

Kept dependency-free so tasks.py (worker container) and the web container
can both import from here without pulling in Flask or service SDKs.
"""
import math
import os

# Shared podcast output directory.
# Must match the docker-compose ``podcast_data`` volume that is mounted
# at the same path in both the web and worker containers.
OUTPUT_DIR = '/app/output_podcasts'

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
