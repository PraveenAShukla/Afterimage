#!/usr/bin/env bash
#SBATCH --job-name=collusion-build
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# CPU job (never the login node): build the Python env and download model weights, then verify both.
# Usage: sbatch -p <cpu partition> -q <cpu qos> -o <LOGDIR>/%x_%j.out slurm/build_env.sbatch MODEL [MODEL ...]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -a; source serve/site.env; set +a
VENV=${VENV_OVERRIDE:-$VENV}   # per-job env (e.g. a newer vLLM for a new architecture); default = live env
: "${VENV:?}" "${HF_HOME:?}" "${LOGDIR:?}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HF_HOME/../uv-cache}"   # keep the package cache off the small home quota
echo "[$(date)] build on $(nproc) CPUs"
# CUDA build must match the node driver: the A100 nodes run driver 570 (CUDA <= 12.8), so the default
# CUDA 13 wheels fail with "driver too old". Override with VLLM_SPEC / TORCH_BACKEND in site.env.
# Measured: torch cu130 -> "driver too old"; torch cu129 -> torch.cuda.is_available() False on driver 570.
# vLLM >= 0.20 ships only cu129/cu130, so we use v0.11.2, the newest release whose default build is CUDA 12.8.
# If this job can see a GPU, pick the build from the node's driver (nvidia-smi "CUDA Version"):
#   >= 13.0 -> latest vLLM (CUDA 13);  12.9 -> vLLM 0.31.0+cu129;  else -> vLLM 0.11.2 (CUDA 12.8). site.env overrides.
DRV=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$" || true)
echo "driver CUDA version: ${DRV:-no GPU visible}"
if [[ -z "${VLLM_SPEC:-}" ]]; then
  if [[ -n "$DRV" ]] && awk "BEGIN{exit !($DRV >= 13.0)}"; then VLLM_SPEC="vllm"; TORCH_BACKEND=${TORCH_BACKEND:-cu130}
  elif [[ "$DRV" == "12.9" ]]; then VLLM_SPEC="vllm @ https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"; TORCH_BACKEND=${TORCH_BACKEND:-cu129}
  else VLLM_SPEC="vllm==0.11.2"; TORCH_BACKEND=${TORCH_BACKEND:-cu128}; fi
fi
TORCH_BACKEND=${TORCH_BACKEND:-cu128}
echo "installing: $VLLM_SPEC (torch backend $TORCH_BACKEND)"
# Each build gets its own PERMANENT versioned dir; "$VENV" is a symlink to the current one. Venvs are not
# relocatable (scripts hard-code their dir), so we never move an env: switching = repointing the link.
FINAL="$VENV"; VENV="${FINAL}-v${SLURM_JOB_ID}"
uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"
uv pip install --torch-backend "$TORCH_BACKEND" "$VLLM_SPEC"
uv pip install -e '.[dev,api]'
uv pip freeze > "$LOGDIR/env_freeze_${SLURM_JOB_ID}.txt"
python - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
if torch.cuda.device_count():
    torch.cuda.init()
    x = torch.ones(512, 512, device="cuda")
    print("GPU check:", torch.cuda.get_device_name(0), "matmul ok", bool((x @ x).sum().item() == 512 ** 3))
PY
python -m pytest -q -p no:cacheprovider tests | tail -1
python -c "import vllm, torch, typing_extensions" || { echo "import check failed; live env left untouched"; exit 1; }
# Switch the link only now that imports, tests and the GPU check passed. A real (non-link) dir at $FINAL is a
# legacy env: set it aside, never delete it here.
if [[ -e "$FINAL" && ! -L "$FINAL" ]]; then mv "$FINAL" "${FINAL}.legacy-$(date +%Y%m%d%H%M%S)"; fi
ln -sfn "$VENV" "${FINAL}.link-tmp" && mv -Tf "${FINAL}.link-tmp" "$FINAL"
echo "[$(date)] env live: $FINAL -> $VENV"
for M in "$@"; do
  echo "[$(date)] downloading $M"
  python - "$M" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, _, rev = sys.argv[1].partition("@")   # MODEL@REVISION pins the exact weights
p = snapshot_download(repo, revision=rev or None, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"])
import os
files = sorted(f for f in os.listdir(p))
size = sum(os.path.getsize(os.path.join(p, f)) for f in files)
print(f"{sys.argv[1]}: {len(files)} files, {size / 1e9:.1f} GB")
PY
done
echo "[$(date)] build done"
