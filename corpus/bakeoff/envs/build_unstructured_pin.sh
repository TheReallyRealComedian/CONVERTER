#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv unstructured-pin
./unstructured-pin/bin/pip install --quiet --upgrade pip
./unstructured-pin/bin/pip install --quiet "unstructured[all-docs]==0.18.32"
./unstructured-pin/bin/python -c "import unstructured; print('unstructured', unstructured.__version__)"
