#!/usr/bin/env bash
#SBATCH --job-name=sup2cal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a
VENV=${VENV_OVERRIDE:-$VENV}
: "${LOGDIR:?}" "${VENV:?}" "${HF_HOME:?}"
source "$VENV/bin/activate"
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
echo "[$(date)] s1 job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1): $MODEL@$REV"
SPID=$(MAXLEN=$MAXL VLLM_EXTRA_ARGS="$SARGS" LAUNCH="srun --exact -n1 --gres=gpu:1 -c 6 --mem=$((JOB_MB - 8192))M" \
       serve/start_vllm.sh "$MODEL${REV:+@$REV}" 1 "$P" "$LOGDIR") || { echo "[$(date)] SERVER FAILED"; exit 1; }
trap 'kill $SPID 2>/dev/null' EXIT
echo "[$(date)] server up (pid $SPID)"
# (header above: the same server start as slurm/s1_queue.sbatch)
# Supervision v2 B5 calibration: serve ONE model, run the review supervisor over recorded claim-only histories.
# Usage: sbatch ... slurm/sup2_calib.sbatch configs/models/<model>.yaml <qwen3off|qwen25>
python scripts/sup2_calibrate.py --model "$1" --runs-dir runs --out results/supervision_v2/calibration \
  || { echo "[$(date)] CALIBRATION FAILED"; exit 1; }
echo "[$(date)] calibration done"
