# Use the official Playwright Python image that has browsers pre-installed.
# OKTOBER (2026-08-22): noble (Ubuntu 24.04) = Python 3.12 — jammy's 3.10
# reaches EOL 2026-10-04, after which the Google client libraries (texttospeech,
# api_core) ship no further releases for it. The image pins Playwright AND its
# browsers; requirements.txt must carry the SAME playwright version.
FROM mcr.microsoft.com/playwright/python:v1.62.0-noble

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required by unstructured.io.
# ghostscript left with DOC-WEB: camelot was its only user (verified in the
# container — libreoffice/poppler-data merely *Suggest* it, no python package
# references the binary).
RUN apt-get update && apt-get install -y \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# pandoc from the official release deb, NOT jammy's apt (2.9.2.1): the DOCX
# backend (DOC-ENGINE) was measured with 3.10.1, and 2.9's gfm writer predates
# GFM footnotes — it demotes the measured image-footnote-link chain (rule 3)
# to escaped text plus a numbered list, voiding the choice. Arch-aware deb
# (amd64 Mintbox / arm64 local builds both exist upstream).
ARG PANDOC_VERSION=3.10.1
# ARCH-BUILD: sha256 per architecture, checked before dpkg sees the file.
# Source of the hashes: the GitHub release API's per-asset `digest` for
# 3.10.1 (the release ships no SHA256SUMS file), re-computed on both
# downloaded debs on 2026-10-06. A version bump brings two new hashes; an
# unknown architecture fails instead of installing something unchecked.
# (tests/test_build_pins.py pins: curl -> sha256sum -c -> use, two hashes.)
RUN arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
         amd64) sha=b419369915e0f3181be0afdb040ec8ecc6b70e72e5992652a0d83aed9e6bc109 ;; \
         arm64) sha=14add8849fda702051f8f4da7b080dfab91ac7a11144602a9643e065d3b4c206 ;; \
         *) echo "pandoc: no checksum pinned for $arch" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o /tmp/pandoc.deb \
       "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-1-${arch}.deb" \
    && echo "$sha  /tmp/pandoc.deb" | sha256sum -c - \
    && dpkg -i /tmp/pandoc.deb \
    && rm /tmp/pandoc.deb

# CPU-only PyTorch: the CUDA wheels pulled transitively by unstructured (torch +cu130
# plus the nvidia-*-cu13 stack, ~3.9 GB) are dead weight — the container has no GPU
# passthrough and the extraction path is ML-free (partition strategy="fast"; PDFs go via
# Gemini). Pinning the +cpu build BEFORE the requirements install means unstructured finds
# torch already satisfied and never pulls the CUDA variant. Use +cpu explicitly with
# --extra-index-url (not --index-url, which would drop torch's runtime deps from PyPI).
# OKTOBER: constraints.txt freezes ten unpinned transitive packages at their
# last-3.10-build versions (see the file's header) — passed to BOTH installs so
# the torch layer never pulls a networkx/filelock the requirements layer would
# then have to downgrade.
COPY constraints.txt .
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    torch==2.12.1+cpu torchvision==0.27.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c constraints.txt

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=600 --retries=5 -r requirements.txt -c constraints.txt

# Download NLTK assets
# SEC-NONROOT: into a system directory of NLTK's default search path, NOT the
# downloader's default ~/nltk_data — at build time that is /root/nltk_data,
# and /root is 700: as uid 1000 every unstructured partition (TXT/EML/MD and
# the HTML fallback) died on "Resource 'punkt_tab' not found" (measured in the
# prod image; with network, unstructured would instead re-download into the
# container's HOME at every start). The downloader writes its zips 0600, so
# the tree is opened to read-for-all at the end — still root-owned.
RUN python - <<'PY'
import os
import sys
import nltk

NLTK_DIR = '/usr/local/share/nltk_data'

# ARCH-BUILD: fail the build instead of printing. The downloader returns
# False on failure (it does not raise), and the previous block ignored the
# value and swallowed exceptions — a missing resource surfaced only as a
# runtime LookupError in every unstructured partition. Each resource must
# download AND be findable in NLTK_DIR itself (paths= pins the lookup; the
# default search path would also accept a stray /root/nltk_data). TLS stays
# verified: the former unverified-context shim is gone — a TLS failure is a
# build finding, not something to silence. tests/test_build_pins.py executes
# this block against a stub nltk.
# resource id -> what nltk.data.find() resolves. wordnet ships zip-only
# upstream (the reader opens the zip), so its findable name is the zip.
RESOURCES = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet.zip',
    'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
    'words': 'corpora/words',
}

failed = []
for resource, probe in RESOURCES.items():
    try:
        ok = nltk.download(resource, download_dir=NLTK_DIR, quiet=False)
    except Exception as exc:  # network, TLS, disk — every one ends the build
        print(f'nltk: download of {resource} raised {exc!r}')
        ok = False
    if not ok:
        print(f'nltk: download of {resource} FAILED')
        failed.append(resource)
        continue
    try:
        found = nltk.data.find(probe, paths=[NLTK_DIR])
    except LookupError:
        print(f'nltk: {resource} downloaded, but {probe} not found under {NLTK_DIR}')
        failed.append(resource)
        continue
    print(f'nltk: {resource} OK -> {found}')
for root, dirs, files in os.walk(NLTK_DIR):
    for name in dirs:
        os.chmod(os.path.join(root, name), 0o755)
    for name in files:
        os.chmod(os.path.join(root, name), 0o644)
if failed:
    print(f'nltk: {len(failed)} resource(s) failed: {", ".join(failed)}')
    sys.exit(1)
print(f'nltk: all {len(RESOURCES)} resources present under {NLTK_DIR}')
PY

# docker CLI, client binary only (DOC-LOCAL): since SEC-SOCKET only the
# mineru-launcher service uses it — the one container that mounts the host's
# docker socket and starts the mineru sibling container; web and worker carry
# the binary without a socket. It needs the CLI, never a daemon. Static
# binary from download.docker.com (pinned, arch-aware:
# x86_64 Mintbox / aarch64 local builds), ~40 MB instead of the docker.io
# apt package's containerd stack. Placed late on purpose: the layer sits
# BELOW the expensive pip layers, so a CLI bump never rebuilds them.
ARG DOCKER_CLI_VERSION=27.5.1
# ARCH-BUILD: sha256 per architecture, checked before tar opens the file.
# download.docker.com publishes no checksum file for the static bundles —
# trust on first use, 2026-10-06: each tarball downloaded twice (same hash
# both times), and the x86_64 bundle's docker/docker is byte-identical to
# /usr/local/bin/docker in the then-deployed image f9cd92dd81ac. Integrity
# from here on; a version bump brings two new hashes.
RUN arch="$(uname -m)" \
    && case "$arch" in \
         x86_64)  sha=4f798b3ee1e0140eab5bf30b0edc4e84f4cdb53255a429dc3bbae9524845d640 ;; \
         aarch64) sha=e6b53725a73763ab3f988c73f8772eaed429754c1a579db5ff11f21990fd1817 ;; \
         *) echo "docker CLI: no checksum pinned for $arch" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o /tmp/docker.tgz \
       "https://download.docker.com/linux/static/stable/${arch}/docker-${DOCKER_CLI_VERSION}.tgz" \
    && echo "$sha  /tmp/docker.tgz" | sha256sum -c - \
    && tar -xzf /tmp/docker.tgz -C /tmp docker/docker \
    && mv /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker.tgz /tmp/docker

# SEC-NONROOT (F-8): web and worker run as uid:gid 1000:1000, not root.
# Why 1000: what a container process can reach on the host is decided by its
# MOUNTS, not its uid — and the only host paths web/worker see (the exchange
# bind, google-credentials.json) belong to uid 1000 on the Mintbox (oliver)
# and MUST stay reachable; any other uid would need a chown of Oli's files for
# the same reach. The base image's uid 1000 ('ubuntu') is renamed, so exactly
# one passwd entry carries it, and HOME is writable (Chromium wants one).
# The order is load-bearing:
#   1. mkdir + chown the three data mount points BEFORE `COPY . .` and
#      before USER: a fresh named volume takes content AND owner only from a
#      directory that exists in the image — a missing mount point is created
#      root:root and uid 1000 could not write its own volume. (COPY would
#      reset them to root if the build context carried same-named dirs —
#      measured; .dockerignore keeps them out.)
#   2. `COPY . .` without --chown: the code stays root-owned and read-only
#      for the process — it cannot rewrite its own program.
#   3. USER last, right before CMD. The mineru-launcher shares this image and
#      is set back to root in docker-compose.yml: it holds the docker socket,
#      which is root-equivalent whatever uid carries it.
# Operators: `docker exec` into web/worker NEVER with `-u 0` — files root
# leaves in the data directories are out of the process's reach.
RUN usermod --login converter --home /home/converter --move-home ubuntu \
    && groupmod --new-name converter ubuntu \
    && mkdir -p /app/data /app/output_podcasts /app/doclocal_exchange \
    && chown 1000:1000 /app/data /app/output_podcasts /app/doclocal_exchange
ENV HOME=/home/converter

COPY . .

USER 1000:1000

# SYNC-FREEZE: 2 worker PROCESSES, each serving sync views on a thread pool
# of WEB_SYNC_THREADS (app_pkg/asgi.py). Responsiveness no longer depends on
# the process count: since P2 every WSGI call runs on a per-process pool, and
# ONE process answered probes in 6-9 ms while two transcriptions and two PDF
# renders ran inside it (measured: scripts/measure_sync_blocking.py +
# scripts/verify_concurrency.py). Processes are now for two things only:
# (a) surviving a worker restart - gunicorn respawns a crashed or timed-out
# worker, the other process keeps serving meanwhile; (b) the GIL - CPU-bound
# views (Markdown->HTML of a long document, an EPUB build, a 200-card
# review-state JSON) run truly parallel on two cores instead of time-slicing
# one. Four bought nothing beyond that at 2x the memory (200-290 MB RSS per
# process after large uploads). Under P1's single-thread adapter the process
# count was the only lever - that is why it was 4 for one commit; --threads
# never helped because the serialisation sat in asgiref, per process.
# NO --preload: each process builds its own app. (The gRPC channel in
# GoogleTTSService that made a fork after import unsafe left the web process
# with ARCH-NARR5; --preload has not been re-examined since - the Deepgram
# client and the Redis connection are still built at import.) The
# schema bootstrap is serialised by the startup lock in app_pkg/__init__.py,
# and SQLite runs in WAL mode with an explicit busy_timeout (same module) so
# N writers don't trade the freeze for 'database is locked'.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "1800", "--worker-class", "uvicorn.workers.UvicornWorker", "app:asgi_app"]
