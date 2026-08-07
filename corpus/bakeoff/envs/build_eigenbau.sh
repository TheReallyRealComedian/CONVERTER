#!/bin/bash
# Baut das venv fuer die Kandidaten `eigenbau` + `textlayer` und fuers Scoring.
# Pins = exakt die Prod-Pins aus requirements.txt (der Eigenbau soll als der
# Kandidat antreten, der er in Prod ist); rapidfuzz nur fuers CER-Scoring.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv eigenbau
./eigenbau/bin/pip install --quiet --upgrade pip
./eigenbau/bin/pip install --quiet \
    "PyMuPDF==1.24.1" \
    "pdfplumber==0.11.9" \
    "camelot-py==0.11.0" \
    "img2table==1.4.2" \
    "opencv-python-headless==4.10.0.84" \
    "pdfminer.six==20251230" \
    "google-genai>=1.0.0" \
    "rapidfuzz>=3.0"

echo "OK: envs/eigenbau steht. Aufruf: envs/eigenbau/bin/python harness/run_one.py …"
