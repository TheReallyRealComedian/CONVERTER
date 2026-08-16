# Use the official Playwright Python image that has browsers pre-installed
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required by unstructured.io
RUN apt-get update && apt-get install -y \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    ffmpeg \
    ghostscript \
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

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "1800","--worker-class", "uvicorn.workers.UvicornWorker", "app:asgi_app"]
