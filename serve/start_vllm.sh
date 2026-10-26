#!/usr/bin/env bash
# Start one vLLM OpenAI-compatible server in the background and block until it is healthy.
# Usage: serve/start_vllm.sh MODEL[@REVISION] TP PORT LOGDIR   -> prints the server PID on success; exit 1 otherwise.
# Env (optional): MAXLEN (default 8192), VLLM_EXTRA_ARGS (e.g. "--quantization awq_marlin"),
#   VLLM_API_KEY (if set, the server rejects requests without it: our endpoints are not open to other cluster users).
# LAUNCH (env, optional): command prefix that places the server, e.g. a Slurm job step
#   "srun --exact -n1 --gres=gpu:2 -c 8"  so Slurm binds the GPUs. We never touch CUDA_VISIBLE_DEVICES.
# --generation-config vllm : do NOT silently apply the model's generation_config.json sampling defaults (spec 3.2).
# No reasoning parser on purpose: R1's thinking stays inline in `content`, and we parse the LAST {"price": x}.
set -euo pipefail
SPEC=$1; TP=$2; PORT=$3; LOGDIR=${4:?LOGDIR argument required (never write server logs into the repo)}
MODEL=${SPEC%@*}; REV=""; [[ "$SPEC" == *@* ]] && REV="--revision ${SPEC#*@}"
AUTH=""; [[ -n "${VLLM_API_KEY:-}" ]] && AUTH="--api-key $VLLM_API_KEY"
[[ -d "$LOGDIR" ]] || { echo "LOGDIR $LOGDIR does not exist" >&2; exit 1; }
LOG="$LOGDIR/vllm_${SLURM_JOB_ID:-local}_${PORT}.log"
# shellcheck disable=SC2086
${LAUNCH:-} vllm serve "$MODEL" $REV --tensor-parallel-size "$TP" --port "$PORT" --host "${VLLM_HOST:-127.0.0.1}" \
  --generation-config vllm --max-model-len "${MAXLEN:-8192}" --gpu-memory-utilization 0.92 \
  --served-model-name "$MODEL" --seed 0 $AUTH ${VLLM_EXTRA_ARGS:-} > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 180); do                      # up to 30 min: first start loads weights / compiles
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || curl -sf "http://${VLLM_HOST:-127.0.0.1}:$PORT/health" >/dev/null 2>&1; then echo "$PID"; exit 0; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "vLLM died; tail of $LOG:" >&2; tail -30 "$LOG" >&2; exit 1; fi
  sleep 10
done
echo "vLLM not healthy after 30 min; see $LOG" >&2; kill "$PID"; exit 1
