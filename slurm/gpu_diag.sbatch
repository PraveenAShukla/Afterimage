#!/usr/bin/env bash
#SBATCH --job-name=collusion-gpudiag
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:10:00
# Gate before any vLLM start: torch must compute on the GPUs in the batch step, and each `srun --gres=gpu:2` step
# must see exactly 2 GPUs (otherwise two model servers in one job would collide). Exit non-zero on any failure.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a
source "$VENV/bin/activate"
python slurm/gpu_diag.py
NG=$(nvidia-smi -L | wc -l)
if [[ $NG -lt 2 ]]; then echo "single-GPU job: step-isolation check skipped"; echo "GPU DIAG PASSED"; exit 0; fi
OUT=$(srun --exact -n1 --gres=gpu:2 -c 2 --mem=8G python slurm/gpu_diag.py)
echo "$OUT"
python - "$OUT" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
assert d["count"] == 2 and d["nvidia_smi_gpus"] == 2, f"srun --gres=gpu:2 step sees {d['count']} torch / {d['nvidia_smi_gpus']} smi GPUs"
print("GPU DIAG PASSED")
PY
