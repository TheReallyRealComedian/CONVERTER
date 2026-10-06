#!/usr/bin/env sh
# scripts/freeze_image.sh — ARCH-BUILD (2026-10-05): das Protokoll des Builds.
#
# Gibt `pip freeze` eines gebauten converter-app-Images aus, deterministisch
# (LC_ALL=C sortiert, LF, eine Zeile je Paket) und mit einem Kopf, der nur
# von der Image-Identität abhängt (Id + Erstellungszeit des Images, kein
# Wanduhr-Datum) — zwei Läufe auf demselben Image sind byte-gleich.
#
# Warum: der Docker-Layer-Cache war das einzige Lockfile. Was im deployten
# Image läuft, steht seit ARCH-BUILD in docs/build/pip-freeze.txt; das
# Deploy-Ritual ist (s. CLAUDE.md *Running*):
#   1. VOR dem Build:   scripts/freeze_image.sh latest | diff - docs/build/pip-freeze.txt
#                       (muss leer sein — sonst lief ein Build, der nie protokolliert wurde)
#   2. Build + Deploy.
#   3. NACH dem Build:  scripts/freeze_image.sh latest > docs/build/pip-freeze.txt
#                       Diff lesen (jede Bewegung benennen), Datei mit dem Deploy committen.
#
# Läuft dort, wo das Image liegt (Mintbox); vom Mac: ssh mintbox 'sh -s <tag>' < scripts/freeze_image.sh
# Ohne Netz, ohne Schreibzugriff, als der Image-User (uid 1000).
set -eu

tag="${1:-latest}"
image="converter-app:${tag}"

image_id="$(docker image inspect --format '{{.Id}}' "$image")"
created="$(docker image inspect --format '{{.Created}}' "$image")"

printf '%s\n' \
  '# docs/build/pip-freeze.txt — pip freeze des deployten Images (ARCH-BUILD).' \
  "# Image-Id: ${image_id}" \
  "# Image erstellt: ${created}" \
  '# Erzeugt mit: scripts/freeze_image.sh <tag>  (= docker run --rm --network none --entrypoint pip converter-app:<tag> freeze, LC_ALL=C sortiert)'

docker run --rm --network none --entrypoint pip "$image" freeze | LC_ALL=C sort
