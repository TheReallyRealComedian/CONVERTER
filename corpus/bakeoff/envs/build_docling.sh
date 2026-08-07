#!/bin/bash
# docling >= 2.109.0 (Sprint-Mindestversion: Deutsch-OCR-Fix + PP-OCRv6-Defaults).
# rapidocr-Extra zuerst (Register-Empfehlung), Fallback bleibt tesseract-cli.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv docling
./docling/bin/pip install --quiet --upgrade pip
./docling/bin/pip install --quiet "docling>=2.109.0" || exit 1
./docling/bin/pip install --quiet rapidocr onnxruntime 2>/dev/null || \
  ./docling/bin/pip install --quiet rapidocr-onnxruntime 2>/dev/null || \
  echo "WARN: rapidocr nicht installierbar - Adapter faellt auf tesseract-cli/deu"
./docling/bin/python -c "import docling; print('docling', docling.__version__)"
