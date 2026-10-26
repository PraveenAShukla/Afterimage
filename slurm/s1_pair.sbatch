#!/usr/bin/env bash
#SBATCH --job-name=s1pair
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# Two DIFFERENT models on ONE allocated GPU (our gpu-partition limit is 1 GPU per job): two vLLM servers, each
# capped at 45% of GPU memory, the second as an --overlap job step on the SAME allocation. Slurm assigns the
# device; this script never touches CUDA_VISIBLE_DEVICES. Then all given configs run in parallel.
# Usage: sbatch ... slurm/s1_pair.sbatch configs/models/<m1>.yaml configs/models/<m2>.yaml CFG [CFG ...]
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a
VENV=${VENV_OVERRIDE:-$VENV}
source "$VENV/bin/activate"
S1=$1; S2=$2; shift 2
JOB_MB=${SLURM_MEM_PER_NODE:-49152}; STEP_MB=$(( (JOB_MB - 8192) / 2 ))
spec() { python - "$1" <<'PY'
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))
print(s["firm"]["model"] + ("@" + s["revision"] if s.get("revision") else ""), (s.get("server_args") or "").replace(" ", "|"))
PY
}
read -r M1 A1 < <(spec "$S1"); read -r M2 A2 < <(spec "$S2")
A1=${A1//|/ }; A2=${A2//|/ }
P1=$((26000 + SLURM_JOB_ID % 2000)); P2=$((P1 + 1))
export VLLM_BASE_URL="http://127.0.0.1:$P1" VLLM_BASE_URL_2="http://127.0.0.1:$P2"
SHARE="--gpu-memory-utilization 0.45 --max-num-seqs 24"
echo "[$(date)] pair job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1): $M1 + $M2"
PID1=$(MAXLEN=4096 VLLM_EXTRA_ARGS="$A1 $SHARE" LAUNCH="srun --exact -n1 --gres=gpu:1 -c 3 --mem=${STEP_MB}M" \
       serve/start_vllm.sh "$M1" 1 "$P1" "$LOGDIR") || { echo "[$(date)] SERVER 1 FAILED"; exit 1; }
PID2=$(MAXLEN=4096 VLLM_EXTRA_ARGS="$A2 $SHARE" LAUNCH="srun --overlap -n1 --gres=gpu:1 -c 3 --mem=${STEP_MB}M" \
       serve/start_vllm.sh "$M2" 1 "$P2" "$LOGDIR") || { echo "[$(date)] SERVER 2 FAILED"; kill "$PID1"; exit 1; }
trap 'kill $PID1 $PID2 2>/dev/null' EXIT
echo "[$(date)] both servers up; GPU memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
python -m collusion.preflight "$1" || { echo "[$(date)] PREFLIGHT FAILED"; exit 1; }
PIDS=()
for C in "$@"; do python -m collusion.run "$C" & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p" || echo "[$(date)] a run config exited non-zero (pid $p)"; done
echo "[$(date)] s1 pair done"
