#!/usr/bin/env bash
#SBATCH --job-name=collusion-queue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# One GPU, a queue of open models run back to back. For each model spec: start its vLLM server as a Slurm job step
# (pinned revision, its own server args / context length), run the gated pipeline, stop the server, next model.
# Resubmitting the same queue resumes: finished runs are skipped, interrupted runs resume.
# Usage: sbatch ... slurm/pipeline_queue.sbatch configs/models/a.yaml configs/models/b.yaml ...
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a
VENV=${VENV_OVERRIDE:-$VENV}   # per-job env (e.g. a newer vLLM for a new architecture); default = live env
: "${LOGDIR:?}" "${VENV:?}" "${HF_HOME:?}"
source "$VENV/bin/activate"
JOB_MB=${SLURM_MEM_PER_NODE:-49152}
echo "[$(date)] queue job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1): $*"
i=0
for SPEC in "$@"; do
  i=$((i+1))
  read -r MODEL REV SARGS MAXL < <(python - "$SPEC" <<'PY'
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))
print(s["firm"]["model"], s.get("revision", ""), (s.get("server_args") or "-").replace(" ", "|"), s.get("maxlen", 8192))
PY
)
  SARGS=${SARGS//|/ }; [[ "$SARGS" == "-" ]] && SARGS=""
  P=$((20000 + (SLURM_JOB_ID % 2000) * 10 + i)); export VLLM_BASE_URL="http://127.0.0.1:$P"
  echo "[$(date)] === $SPEC: $MODEL@$REV args='$SARGS' maxlen=$MAXL port=$P"
  SPID=$(MAXLEN=$MAXL VLLM_EXTRA_ARGS="$SARGS" LAUNCH="srun --exact -n1 --gres=gpu:1 -c 6 --mem=$((JOB_MB - 8192))M" \
         serve/start_vllm.sh "$MODEL${REV:+@$REV}" 1 "$P" "$LOGDIR") || { echo "[$(date)] SERVER FAILED for $SPEC; skipping to next model"; continue; }
  echo "[$(date)] server up (pid $SPID)"
  python -m collusion.pipeline "$SPEC" || echo "[$(date)] pipeline exited non-zero for $SPEC"
  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; sleep 5
  echo "[$(date)] === done $SPEC"
done
echo "[$(date)] queue finished"
