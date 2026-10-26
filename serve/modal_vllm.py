"""Qwen3-14B-AWQ on Modal, served exactly as serve/start_vllm.sh serves it on the cluster :
pinned vLLM 0.11.2, pinned model revision, served name = the HF id, --generation-config vllm, seed 0, max-model-len 4096,
awq_marlin, no auth header (the harness sends none). One container, so all runs share one server process.

  modal deploy serve/modal_vllm.py          # prints the URL; export VLLM_BASE_URL=<url>
  modal app stop ait-qwen3-14b              # stop it as soon as the runs finish (the URL is unauthenticated)
"""
import subprocess

import modal

MODEL = "Qwen/Qwen3-14B-AWQ"
REV = ""
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.11.2", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
hf_cache = modal.Volume.from_name("ait-hf-cache", create_if_missing=True)
app = modal.App("ait-qwen3-14b")


@app.function(image=image, gpu="H100", volumes={"/root/.cache/huggingface": hf_cache}, timeout=6 * 3600,
              scaledown_window=900, max_containers=1)
@modal.concurrent(max_inputs=512)
@modal.web_server(port=8000, startup_timeout=1800)
def serve():
    subprocess.Popen(["vllm", "serve", MODEL, "--revision", REV, "--port", "8000", "--host", "0.0.0.0",
                      "--tensor-parallel-size", "1", "--generation-config", "vllm", "--max-model-len", "4096",
                      "--gpu-memory-utilization", "0.92", "--served-model-name", MODEL, "--seed", "0",
                      "--quantization", "awq_marlin"])
