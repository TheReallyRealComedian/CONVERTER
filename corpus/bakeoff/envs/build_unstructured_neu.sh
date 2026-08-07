#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv unstructured-neu
./unstructured-neu/bin/pip install --quiet --upgrade pip
./unstructured-neu/bin/pip install --quiet "unstructured[all-docs]==0.24.1"
./unstructured-neu/bin/python -c "import unstructured; print('unstructured', unstructured.__version__)"
