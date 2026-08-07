#!/bin/bash
# Baut die drei GPU-Kandidaten-Container auf der Mintbox. Einmalig, lange.
# Modelle landen in ~/bakeoff-models (HF-Cache-Volume), 2,3 TB frei.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ~/bakeoff-models

echo "== 1/3 MinerU (offizielles Dockerfile, vllm-Basis) =="
if ! docker image inspect mineru:latest >/dev/null 2>&1; then
    rm -f /tmp/mineru.Dockerfile
    wget -q -O /tmp/mineru.Dockerfile \
        https://raw.githubusercontent.com/opendatalab/MinerU/master/docker/global/Dockerfile
    docker build -t mineru:latest -f /tmp/mineru.Dockerfile /tmp
else
    echo "   mineru:latest existiert schon"
fi

echo "== 2/3 marker v2 =="
docker build -t bakeoff-marker:latest -f Dockerfile.marker .

echo "== 3/3 dots.ocr: vllm-Server-Image + Client =="
docker pull vllm/vllm-openai:latest
docker build -t bakeoff-dotsclient:latest -f Dockerfile.dotsclient .

echo "== Versionen =="
docker run --rm mineru:latest mineru --version || true
docker run --rm bakeoff-marker:latest python -c "import importlib.metadata as m; print('marker-pdf', m.version('marker-pdf'))"
docker run --rm vllm/vllm-openai:latest --version 2>/dev/null | head -1 || \
    docker run --rm --entrypoint python vllm/vllm-openai:latest -c "import vllm; print('vllm', vllm.__version__)"
echo "OK: Container stehen."
