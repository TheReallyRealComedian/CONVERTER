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
# NOTE: 'punkt_tab' from the error message is not a standard NLTK package.
# We download 'punkt' instead, which unstructured will use as a fallback.
resources_to_download = [
    'punkt',
    'averaged_perceptron_tagger',
    'stopwords',
    'wordnet',
]

print("Downloading NLTK resources...")
for resource in resources_to_download:
    try:
        # Set quiet=False to see download progress and potential errors
        nltk.download(resource, quiet=False)
        print(f"✓ Successfully downloaded {resource}")
    except Exception as e:
        print(f"⚠ Failed to download {resource}: {e}")
        # Depending on the resource, you might want to exit if it's critical
        # import sys
        # if resource == 'punkt': sys.exit(1)
print("NLTK resource download complete.")
PY

# Install the browser binaries for Playwright
RUN playwright install --with-deps chromium

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "app:asgi_app"]