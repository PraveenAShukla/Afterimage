#!/usr/bin/env bash
#SBATCH --job-name=ait
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:30:00
# Afterimage waves on ONE GPU and ONE vLLM server (docs/PREREG_AFTERIMAGE.md). Waves run one after another; the
# configs inside a wave run in parallel, each into its own runs dir (created with the 18 source symlinks).
# Usage: sbatch -p <gpu partition> slurm/ait_waves.sbatch configs/models/<m>.yaml <dir>=<cfg> ... [-- <dir>=<cfg> ...]
# Launch from a clean checkout of the pushed tag. Each `collusion.run` skips completed runs, so resubmitting resumes.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a
source "${VENV_OVERRIDE:-$VENV}/bin/activate"
: "${LOGDIR:?}" "${SOURCES:?set SOURCES to the runs dir that holds the s3-qwen25 source runs}"
SPEC=$1; shift
JOB_MB=${SLURM_MEM_PER_NODE:-49152}
read -r MODEL REV SARGS MAXL < <(python - "$SPEC" <<'PY'
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))
print(s["firm"]["model"], s.get("revision", ""), (s.get("server_args") or "-").replace(" ", "|"), s.get("maxlen", 8192))
PY
)
SARGS=${SARGS//|/ }; [[ "$SARGS" == "-" ]] && SARGS=""
P=$((24000 + SLURM_JOB_ID % 2000)); export VLLM_BASE_URL="http://127.0.0.1:$P"
echo "[$(date -u)] ait job $SLURM_JOB_ID at $(git describe --tags --always) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1): $MODEL@$REV"
SPID=$(MAXLEN=$MAXL VLLM_EXTRA_ARGS="$SARGS" LAUNCH="srun --exact -n1 --gres=gpu:1 -c 6 --mem=$((JOB_MB - 8192))M" \
       serve/start_vllm.sh "$MODEL${REV:+@$REV}" 1 "$P" "$LOGDIR") || { echo "[$(date -u)] SERVER FAILED"; exit 1; }
trap 'kill $SPID 2>/dev/null' EXIT
echo "[$(date -u)] server up (pid $SPID)"
run_wave() {
  local PIDS=() a d c
  for a in "$@"; do
    d=${a%%=*}; c=${a#*=}
    scripts/make_runs_dir.sh "$d" "$SOURCES" > /dev/null
    RUNS_DIR=$d python -m collusion.preflight "$c" > "$LOGDIR/ait_${SLURM_JOB_ID}_preflight_$(basename "$c" .yaml).log" 2>&1 \
      || { echo "[$(date -u)] PREFLIGHT FAILED $c"; continue; }
    python -m collusion.run "$c" --runs-dir "$d" > "$LOGDIR/ait_${SLURM_JOB_ID}_$(basename "$c" .yaml).log" 2>&1 & PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p" || echo "[$(date -u)] a config exited non-zero (pid $p)"; done
}
WAVE=()
for a in "$@"; do
  if [[ "$a" == "--" ]]; then echo "[$(date -u)] wave: ${#WAVE[@]} configs"; run_wave "${WAVE[@]}"; WAVE=(); else WAVE+=("$a"); fi
done
[[ ${#WAVE[@]} -gt 0 ]] && { echo "[$(date -u)] wave: ${#WAVE[@]} configs"; run_wave "${WAVE[@]}"; }
echo "[$(date -u)] ait done"
