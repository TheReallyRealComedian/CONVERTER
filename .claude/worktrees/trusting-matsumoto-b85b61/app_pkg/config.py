"""Shared configuration constants.

Kept dependency-free so tasks.py (worker container) and the web container
can both import from here without pulling in Flask or service SDKs.
"""

# Shared podcast output directory.
# Must match the docker-compose ``podcast_data`` volume that is mounted
# at the same path in both the web and worker containers.
OUTPUT_DIR = '/app/output_podcasts'
