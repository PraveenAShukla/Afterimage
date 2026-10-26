#!/usr/bin/env bash
# Hard spend guard for the Modal vLLM app (Modal must never cross $50).
# Stops the app when ANY of these is true, checked every 2 min:
#   - workspace spend today (modal billing report) >= COST_CAP (default $30; the report lags up to an hour, hence
#     the margin below $50);
#   - the app has been up HARD_HOURS (default 3 h; one H100 at about $4/h, so about $12);
#   - the E5 launcher has finished (no runs left that need the server).
# Server-side backstop without this script: max_containers 1 and a 15-min idle scale-down (serve/modal_vllm.py).
# Usage: setsid nohup serve/modal_watchdog.sh > <log> 2>&1 &
set -u
APP=ait-qwen3-14b
COST_CAP=${COST_CAP:-30}
HARD_S=$(( ${HARD_HOURS:-3} * 3600 ))
T0=$(date +%s)
stop() { echo "$(date -u +%H:%M:%S) STOP: $1"; modal app stop "$APP" --yes 2>&1 | tail -1; exit 0; }
while true; do
  cost=$(modal billing report --for today --json 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("nan"); raise SystemExit
rows = d if isinstance(d, list) else d.get("data", d.get("rows", []))
print(sum(float(r.get("Cost", r.get("cost", 0)) or 0) for r in rows))')
  el=$(( $(date +%s) - T0 ))
  echo "$(date -u +%H:%M:%S) spend_today=\$$cost elapsed=${el}s"
  python3 -c "import sys; c=float('$cost'); sys.exit(0 if c == c and c >= $COST_CAP else 1)" && stop "spend $cost >= $COST_CAP"
  [ "$el" -ge "$HARD_S" ] && stop "up ${el}s >= ${HARD_S}s"
  if [ -f "${DONE_FLAG:-/nonexistent}" ]; then sleep 60; stop "E5 launcher finished"; fi
  sleep 120
done
