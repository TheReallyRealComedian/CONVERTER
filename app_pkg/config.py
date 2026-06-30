"""Shared configuration constants.

Kept dependency-free so tasks.py (worker container) and the web container
can both import from here without pulling in Flask or service SDKs.
"""

# Shared podcast output directory.
# Must match the docker-compose ``podcast_data`` volume that is mounted
# at the same path in both the web and worker containers.
OUTPUT_DIR = '/app/output_podcasts'

# F-015: upstream timeouts. Centralised so the relationship between the
# Gemini SDK timeout, the Deepgram per-request timeout, and the RQ
# job_timeout is visible in one place. Deepgram + RQ are aligned at 600s
# because a long-audio transcription happens inside an RQ job in the
# Gemini-podcast pipeline; Gemini is half that because its TTS calls are
# shorter but still exceed the SDK default.
TIMEOUT_GEMINI_SECONDS = 300
TIMEOUT_DEEPGRAM_SECONDS = 600
TIMEOUT_RQ_JOB_SECONDS = 600
