# Dockerfile - OPTIMIZED

# Use a specific version for more reproducible builds
FROM python:3.10.13-slim

# Set the working directory
WORKDIR /app

# Set environment variables to prevent __pycache__ and buffer issues
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Tell Playwright where its browsers are stored inside the container
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

# --- STAGE 1: INSTALL DEPENDENCIES ---
# Copy only the requirements file first. The layer below will only be
# re-run if this specific file changes.
COPY requirements.txt .

# Install system dependencies and Python packages in a single, efficient RUN command.
# This creates one cached layer for all our dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Dependencies for unstructured.io
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    pandoc \
    # Dependencies needed by Playwright's browser
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 \
    # Now, install python packages
    && pip install --no-cache-dir -r requirements.txt \
    # Clean up apt cache to keep the image smaller
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- STAGE 2: DOWNLOAD HEAVY ASSETS ---
# These assets depend on the packages installed above.
# We run them as separate layers so they can be cached individually.

# Download NLTK data. This command is idempotent and will be cached.
RUN python -m nltk.downloader punkt averaged_perceptron_tagger stopwords wordnet

# Install Playwright browsers. This is a very heavy step, so we cache it here.
RUN playwright install chromium

# --- STAGE 3: COPY APPLICATION CODE ---
# Now, with all heavy dependencies installed and cached, copy the application code.
# If you only change your Python/HTML/CSS files, only this layer will be re-built,
# which is extremely fast.
COPY . .

# --- FINAL CONFIGURATION ---
# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", \
    "--bind", "0.0.0.0:5000", \
    "--workers", "2", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--log-level", "info", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "app:asgi_app"]