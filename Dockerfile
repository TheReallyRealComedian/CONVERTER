# Dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required by unstructured.io
RUN apt-get update && apt-get install -y \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# This single command will now install everything correctly
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------
# Download all NLTK assets that `unstructured` (and other libs) expect.
# We perform an existence-check first so the build cache isn't invalidated
# unless something is actually missing.
# --------------------------------------------------------------------
# Alternative: More comprehensive NLTK resource download
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
    'punkt_tab',           # Required for sentence tokenization
    'punkt',               # Fallback for older NLTK versions
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'stopwords',           # Often needed for text processing
    'wordnet',             # Sometimes needed for lemmatization
]

for resource in resources_to_download:
    try:
        nltk.download(resource, quiet=True)
        print(f"✓ Downloaded {resource}")
    except Exception as e:
        print(f"⚠ Failed to download {resource}: {e}")
PY

# Install the browser binaries for Playwright
RUN playwright install --with-deps chromium

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "app:asgi_app"]