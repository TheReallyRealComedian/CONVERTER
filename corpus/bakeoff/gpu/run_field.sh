#!/bin/bash
# GPU-Feld-Treiber (Mintbox): ein Kandidat, alle PDF-Klassen + Gold-Rollen.
#   bash corpus/bakeoff/gpu/run_field.sh mineru-vlm
# Smoke zuerst (06, 2 Seiten), 12 (280 S.) zuletzt — dort faellt die
# Durchsatz-Messung ab (Modell-Load amortisiert ueber die Seitenzahl).
set -uo pipefail
CAND="${1:?Kandidat fehlt (mineru-vlm|marker2|vlm-dots)}"
cd "$(dirname "$0")/../../.."   # Repo-Root

RUN="python3 corpus/bakeoff/harness/run_one.py --candidate $CAND"
$RUN --class 06 --timeout 3600 || { echo "SMOKE 06 FEHLGESCHLAGEN — Abbruch"; exit 1; }
for cls in 07 05 14 13 03 04 01 02; do
    $RUN --class "$cls" --timeout 7200
done
$RUN --class 01 --role gold --timeout 3600
$RUN --class 07 --role gold --timeout 3600
$RUN --class 12 --timeout 10800
echo "FELD FERTIG: $CAND"
