# Dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

# This single command will now install everything correctly
RUN pip install --no-cache-dir -r requirements.txt

# Install the browser binaries for Playwright
RUN playwright install --with-deps chromium

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "app:asgi_app"]