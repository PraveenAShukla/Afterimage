# Running

## Local checks (no GPU, no money)
```bash
uv venv .venv && uv pip install -e '.[dev,api]'
scripts/check.sh                                        # lint + tests + secret scan
.venv/bin/python -m collusion.run configs/mock/smoke_L.yaml --runs-dir /tmp/mockruns --debug 2
```
Install the git hooks once: `printf '#!/usr/bin/env bash\nexec scripts/check.sh --staged\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`

## On GPUs (Slurm)
1. Copy `serve/site.env.example` to `serve/site.env` and fill in the partition, env path, weights cache and log
   folder. That file is never committed.
2. Build the env and download pinned weights (a GPU job, so the CUDA build matches the node's driver):
   `sbatch -p <partition> --gres=gpu:1 slurm/build_env.sbatch "<model>@<revision>" ...`
3. Run a stage: `slurm/run_stage.sbatch CONFIG MODEL@REV TP`. One job holds the vLLM server and the game runner and
   ends with the stage. `CONFIG` may be a comma-separated list (e.g. FN,FM side by side). The job runs preflight
   first and stops on any FAIL.
4. Resubmitting the same job resumes unfinished games (`--resume`, append-only).
5. `python analysis/analyze.py --stage <stage>` writes `results/` (table, interaction test, punishment plots).

Configs for the RTX 5000 Ada setup are in `configs/rtx/`; the original A100 configs are in `configs/`.

## Paid APIs
1. `.env` (copy `.env.example`, `chmod 600`) holds the keys, caps and rate limits.
2. `python -m collusion.keycheck` checks the keys for free.
3. `python -m collusion.run CONFIG --dry-run` prints the expected and worst-case cost.
4. Run with `--approve-usd <amount>` only after the team agrees on that amount.
5. Watch with `python -m collusion.monitor --watch 60 --auto-stop`; `--stop` is the kill switch.

## Gates
Stage 1: mean Δ < 0.2 → STOP. Stage 2: mean |p − BR| against FN > 0.05 → STOP. Record predictions in
`docs/PREDICTIONS.md` before Stage 1.

## S1 arms on the cluster (plan v2 and later)
- **One model per GPU:** `sbatch -p <partition> --gres=gpu:1 -t 16:00:00 -J <name> -o <logdir>/%x_%j.out slurm/s1_queue.sbatch configs/models/<m>.yaml CFG...`.
  The queue runs every `O_/T_/Q_/S_/L_/STEP_/IMP_` config in parallel, `A_` after, and maps first. Completed runs
  are skipped on resubmit.
- **Two models on one GPU** (cross-model pairs, cross-supervision):
  `sbatch ... slurm/s1_pair.sbatch configs/models/<m1>.yaml configs/models/<m2>.yaml CFG...`.
  Model 1 is served at `$VLLM_BASE_URL`, model 2 at `$VLLM_BASE_URL_2` (a config points the supervisor at it
  with `base_url: $VLLM_BASE_URL_2`).
- **Opening arms cannot resume** (`--resume` refuses prefixed runs). Move partial runs to `runs*/_aborted/<reason>/`
  and rerun with the same seeds, or use `--tag`.
- **Config generators** (each refuses to overwrite and checks seeds against all of configs/):
  `scripts/make_v2_configs.py`, `make_k2_configs.py`, `make_supervision_configs.py`, `make_pressure_configs.py`,
  `make_xsup_configs.py`.
- **Analysis:**
  - Frozen: `analysis/v2.py --phase conf --runs-dir runs_cluster` and `analysis/mech.py`.
  - Supervision and pressure: `analysis/treatments.py {supervision|pressure} --runs-dir runs_cluster --out results/<study>`.
- **Pull cluster runs back append-only:** `rsync -a --ignore-existing <host>:collusion-sprint/runs/<prefix>-* runs_cluster/`.

## Hosted K2 (IFM)
- Provider `ifm` (key `IFM_K2_API_KEY` in `.env`). Pace with `RATE_RPM_IFM`, `RATE_OUTPUT_TPM_IFM` and
  `RATE_MIN_INTERVAL_S_IFM` (0.4 s; IFM allows about 3 requests per second).
- Run from the Mac: `PYTHONPATH=src .venv/bin/python -m collusion.run configs/k2/<cfg>.yaml --runs-dir runs_cluster --approve-usd 1`.
- Details: docs/K2_API.md.
