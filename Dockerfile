# Use the official Playwright Python image that has browsers pre-installed
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

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
RUN arch="$(dpkg --print-architecture)" \
    && curl -fsSL -o /tmp/pandoc.deb \
       "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-1-${arch}.deb" \
    && dpkg -i /tmp/pandoc.deb \
    && rm /tmp/pandoc.deb

# CPU-only PyTorch: the CUDA wheels pulled transitively by unstructured (torch +cu130
# plus the nvidia-*-cu13 stack, ~3.9 GB) are dead weight — the container has no GPU
# passthrough and the extraction path is ML-free (partition strategy="fast"; PDFs go via
# Gemini). Pinning the +cpu build BEFORE the requirements install means unstructured finds
# torch already satisfied and never pulls the CUDA variant. Use +cpu explicitly with
# --extra-index-url (not --index-url, which would drop torch's runtime deps from PyPI).
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    torch==2.12.1+cpu torchvision==0.27.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=600 --retries=5 -r requirements.txt

# Download NLTK assets
RUN python - <<'PY'
import nltk
import ssl

# Handle SSL issues if they occur
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all resources that unstructured commonly needs
resources_to_download = [
    'punkt',                    
    'punkt_tab',                 
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'stopwords', 
    'wordnet', 
    'maxent_ne_chunker',
    'words',
]

print("Downloading NLTK resources...")
for resource in resources_to_download:
    try:
        nltk.download(resource, quiet=False)
        print(f"✓ Successfully downloaded {resource}")
    except Exception as e:
        print(f"⚠ Failed to download {resource}: {e}")
print("NLTK resource download complete.")
PY

# docker CLI, client binary only (DOC-LOCAL): the worker starts the mineru
# sibling container over the host's docker socket — it needs the CLI, never
# a daemon. Static binary from download.docker.com (pinned, arch-aware:
# x86_64 Mintbox / aarch64 local builds), ~40 MB instead of the docker.io
# apt package's containerd stack. Placed late on purpose: the layer sits
# BELOW the expensive pip layers, so a CLI bump never rebuilds them.
ARG DOCKER_CLI_VERSION=27.5.1
RUN arch="$(uname -m)" \
    && curl -fsSL -o /tmp/docker.tgz \
       "https://download.docker.com/linux/static/stable/${arch}/docker-${DOCKER_CLI_VERSION}.tgz" \
    && tar -xzf /tmp/docker.tgz -C /tmp docker/docker \
    && mv /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker.tgz /tmp/docker

COPY . .

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
# NO --preload: each process builds its own app (the SDK clients created at
# import - a gRPC channel in GoogleTTSService - are not fork-safe); the
# schema bootstrap is serialised by the startup lock in app_pkg/__init__.py,
# and SQLite runs in WAL mode with an explicit busy_timeout (same module) so
# N writers don't trade the freeze for 'database is locked'.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "1800", "--worker-class", "uvicorn.workers.UvicornWorker", "app:asgi_app"]
