#!/bin/bash
# markitdown + trafilatura — zwei kleine venvs in einem Rutsch.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv markitdown
./markitdown/bin/pip install --quiet --upgrade pip
./markitdown/bin/pip install --quiet "markitdown[all]"
./markitdown/bin/python -c "import markitdown; print('markitdown', getattr(markitdown,'__version__','?'))"
python3 -m venv trafilatura
./trafilatura/bin/pip install --quiet --upgrade pip
./trafilatura/bin/pip install --quiet trafilatura
./trafilatura/bin/python -c "import trafilatura; print('trafilatura', trafilatura.__version__)"
