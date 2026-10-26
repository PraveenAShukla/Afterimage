"""GPU diagnostic, run inside a Slurm job (batch step and inside `srun --gres=gpu:K` steps).
Prints one JSON line. Exit 1 if torch cannot use the GPUs it was given."""
import json
import os
import subprocess
import sys

import torch

import glob
import socket

info = {"node": socket.gethostname(), "dev_nvidia": sorted(os.path.basename(x) for x in glob.glob("/dev/nvidia*")),
        "step": os.environ.get("SLURM_STEP_ID"), "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "nvidia_smi_gpus": len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines())}
try:
    try:
        torch.cuda.init()  # raises with the real CUDA error instead of returning False
    except Exception as e:  # noqa: BLE001
        info["init_error"] = repr(e)[:400]
    info["available"] = torch.cuda.is_available()
    info["count"] = torch.cuda.device_count()
    if info["available"]:
        x = torch.ones(1024, 1024, device="cuda")
        info["matmul_ok"] = bool((x @ x).sum().item() == 1024 ** 3)
except Exception as e:  # noqa: BLE001
    info["error"] = repr(e)[:300]
print(json.dumps(info), flush=True)
sys.exit(0 if info.get("matmul_ok") else 1)
