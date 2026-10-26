#!/usr/bin/env bash
#SBATCH --job-name=collusion
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# GPUs, partition, QoS and log paths are passed by slurm/submit.sh (never hard-coded here).
# ONE job = vLLM server(s) + the game client; the job ends when the stage ends (no idle reservations).
# Each server is its own Slurm job step with its own --gres share, so two models never fight over the same GPU
# and Slurm (not us) assigns devices: this script never touches CUDA_VISIBLE_DEVICES.
# Usage (via submit.sh): CONFIG MODEL1 TP1 [MODEL2 TP2]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a      # export everything in site.env (LOGDIR, HF_HOME, ...)
VENV=${VENV_OVERRIDE:-$VENV}   # per-job env (e.g. a newer vLLM for a new architecture); default = live env
: "${LOGDIR:?set LOGDIR in serve/site.env}" "${VENV:?set VENV in serve/site.env}" "${HF_HOME:?set HF_HOME}"
source "$VENV/bin/activate"
CONFIG=$1; M1=$2; TP1=$3; M2=${4:-}; TP2=${5:-}
# job-specific ports so two jobs on one shared node cannot collide
P1=$((20000 + SLURM_JOB_ID % 20000)); P2=$((P1 + 1))
export VLLM_BASE_URL="http://127.0.0.1:$P1" VLLM_BASE_URL_2="http://127.0.0.1:$P2"
# Explicit per-step memory (else a 2nd step waits forever), sized from what THIS job got, leaving 8 GB for the client.
JOB_MB=${SLURM_MEM_PER_NODE:-49152}
if [[ -n "$M2" ]]; then STEP_MEM=$(( (JOB_MB - 8192) / 2 ))M; else STEP_MEM=$(( JOB_MB - 8192 ))M; fi
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT
echo "[$(date)] job $SLURM_JOB_ID config=$CONFIG gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
# Per-model server args for mixed pairs (e.g. GPTQ firm 1 + AWQ firm 2); default = site.env VLLM_EXTRA_ARGS / MAXLEN.
PID1=$(VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS_1-$VLLM_EXTRA_ARGS}" MAXLEN="${MAXLEN_1:-$MAXLEN}" LAUNCH="srun --exact -n1 --gres=gpu:$TP1 -c 6 --mem=$STEP_MEM" serve/start_vllm.sh "$M1" "$TP1" "$P1" "$LOGDIR")
PIDS+=("$PID1")
if [[ -n "$M2" ]]; then
  PID2=$(VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS_2-$VLLM_EXTRA_ARGS}" MAXLEN="${MAXLEN_2:-$MAXLEN}" LAUNCH="srun --exact -n1 --gres=gpu:$TP2 -c 6 --mem=$STEP_MEM" serve/start_vllm.sh "$M2" "$TP2" "$P2" "$LOGDIR")
  PIDS+=("$PID2")
fi
# CONFIG may be a comma-separated list (e.g. FN,FM): all share this job's server(s) and run side by side.
IFS=, read -r -a CFGS <<< "$CONFIG"
for C in "${CFGS[@]}"; do python -m collusion.preflight "$C"; done   # any FAIL -> set -e stops the job here
RUNPIDS=()
for C in "${CFGS[@]}"; do python -m collusion.run "$C" --debug 4 --resume & RUNPIDS+=($!); done
RC=0; for p in "${RUNPIDS[@]}"; do wait "$p" || RC=1; done
for C in "${CFGS[@]}"; do
  STAGE=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["stage"])' "$C")
  python analysis/analyze.py --stage "$STAGE" || true
done
echo "[$(date)] done (runner exit status $RC; resubmit the same job to resume anything unfinished)"
exit $RC
