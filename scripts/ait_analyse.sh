#!/usr/bin/env bash
# Frozen afterimage analysis + cards over every local runs dir (committed conf runs, X0, all D1/D2 ait dirs).
set -eu
cd "$(dirname "$0")/.."
A=(--runs-dir runs_cluster)
for d in runs_afterimage_x0 runs_ait_*; do [ -d "$d" ] && A+=(--runs-dir "$d"); done
nice .venv/bin/python analysis/afterimage.py "${A[@]}"
nice .venv/bin/python analysis/afterimage_card.py "${A[@]}" > /dev/null
echo "cards: $(ls results/afterimage/cards/*.md | wc -l)"
