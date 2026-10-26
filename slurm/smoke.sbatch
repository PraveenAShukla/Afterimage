#!/usr/bin/env bash
#SBATCH --job-name=collusion-smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# Engineering smoke only: start the vLLM server as a Slurm job step, run preflight against it, stop.
# Runs NO game. Usage (debug QoS): sbatch ... slurm/smoke.sbatch CONFIG MODEL TP
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a
VENV=${VENV_OVERRIDE:-$VENV}   # per-job env (e.g. a newer vLLM for a new architecture); default = live env
: "${LOGDIR:?}" "${VENV:?}" "${HF_HOME:?}"
source "$VENV/bin/activate"
CONFIG=$1; M=$2; TP=$3
P=$((20000 + SLURM_JOB_ID % 20000)); export VLLM_BASE_URL="http://127.0.0.1:$P"
PIDS=(); cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT
echo "[$(date)] gpus: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sort | uniq -c | xargs)"
T0=$(date +%s)
PIDS+=("$(LAUNCH="srun --exact -n1 --gres=gpu:$TP -c 6 --mem=$(( ${SLURM_MEM_PER_NODE:-49152} - 8192 ))M" serve/start_vllm.sh "$M" "$TP" "$P" "$LOGDIR")")
echo "[$(date)] server healthy after $(( $(date +%s) - T0 )) s"
python -m collusion.preflight "$CONFIG"
echo "[$(date)] smoke done (no game was run)"
