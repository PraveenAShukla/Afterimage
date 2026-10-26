#!/usr/bin/env bash
#SBATCH --job-name=s1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# S1 on ONE GPU for ONE model: start its vLLM server (pinned revision / args / context from configs/models/<m>.yaml)
# as a Slurm job step, then run, in order: the one-step maps (map_*), all arm S configs IN PARALLEL, all arm A
# configs IN PARALLEL. Each `collusion.run` skips completed runs, so resubmitting the same job resumes.
# Usage: sbatch ... slurm/s1_queue.sbatch configs/models/<model>.yaml configs/s1/<cfg>.yaml [...]
# RUNS_DIR=<dir> (exported to sbatch) sends runs and preflight to that dir instead of REPO/runs (one dir per experiment).
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
MAPS=(); S=(); A=()
for C in "$@"; do
  case "$(basename "$C")" in map_*) MAPS+=("$C");; S_*|O_*|T_*|Q_*|L_*|STEP_*|IMP_*) S+=("$C");; A_*) A+=("$C");; *) echo "unknown config $C"; exit 2;; esac
done
if [[ ${#S[@]} -gt 0 || ${#A[@]} -gt 0 ]]; then  # maps-only jobs (plan v2 M step) have no game config to preflight
  python -m collusion.preflight "${S[0]:-${A[0]}}" || { echo "[$(date)] PREFLIGHT FAILED"; exit 1; }
fi
for C in "${MAPS[@]}"; do
  echo "[$(date)] map $C"
  python -m collusion.onestep "$C" --workers 48 || echo "[$(date)] map exited non-zero: $C"
done
for GROUP in S A; do
  eval "LIST=(\"\${${GROUP}[@]}\")"
  [[ ${#LIST[@]} -eq 0 ]] && continue
  echo "[$(date)] arm $GROUP: ${#LIST[@]} configs in parallel"
  PIDS=()
  for C in "${LIST[@]}"; do python -m collusion.run "$C" ${RUNS_DIR:+--runs-dir "$RUNS_DIR"} & PIDS+=($!); done
  for p in "${PIDS[@]}"; do wait "$p" || echo "[$(date)] a run config exited non-zero (pid $p)"; done
done
echo "[$(date)] s1 queue done"
