#!/usr/bin/env bash
# One runs dir per experiment family (sprint pre-flight). Arm O loads its opening windows from the 18 Qwen2.5 L source
# runs, so each new dir gets symlinks to them. Run on the machine that holds the sources (the cluster checkout).
# Usage: scripts/make_runs_dir.sh <new_runs_dir> [<dir holding the sources, default: runs>]
set -euo pipefail
NEW=$1; SRC=${2:-runs}
mkdir -p "$NEW"
n=0
for d in "$SRC"/s3-qwen25_14b_L_E1_Qwen25-14B-Instruct-AWQ_s*; do
  [ -e "$NEW/$(basename "$d")" ] || ln -s "$(cd "$(dirname "$d")" && pwd)/$(basename "$d")" "$NEW/$(basename "$d")"
  n=$((n + 1))
done
echo "$NEW: $n source runs linked"
