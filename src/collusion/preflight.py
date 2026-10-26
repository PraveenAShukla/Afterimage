"""Preflight: everything that can be checked BEFORE a real run spends GPU time or money.

  python -m collusion.preflight configs/stage0_smoke.yaml            # free checks + 2 probe calls per vLLM model
  python -m collusion.preflight configs/x.yaml --probe-paid          # also 1 tiny paid call per API model
  python -m collusion.preflight configs/x.yaml --tests               # also run the unit/harness test suite

Each check is PASS / WARN / FAIL. Any FAIL -> exit 1, and the slurm scripts refuse to start the run.
The report is saved to results/preflight/<config>_.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .backends import Sampling, make_backend
from .budget import FREE_PROVIDERS, Budget
from .config import ConfigError, load_config
from .env import load_env
from .parse import parse_price
from .prompts import system_prompt, user_message
from .runlog import REPO, code_version
from .secrets import env_file_problems

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> bool:
        return any(s == FAIL for s, _, _ in self.rows)


def check_vllm(r: Report, spec: dict, sampling: Sampling, prompt: str) -> None:
    be = make_backend(spec)
    info = be.server_info()
    models = info.get("/v1/models")
    if isinstance(models, str):
        r.add(FAIL, f"vllm reachable ({spec['model']})", f"{be.base_url}: {models}")
        return
    served = [m.get("id") for m in models.get("data", [])]
    r.add(PASS if spec["model"] in served else FAIL, "served model id matches config", f"served={served}")
    r.add(PASS if isinstance(info.get("/version"), dict) else WARN, "vLLM version recorded", str(info.get("/version")))
    sys_p, user = system_prompt(prompt), user_message(1, [])
    a = be.complete(sys_p, user, sampling, seed=12345)
    if a.error:
        r.add(FAIL, "probe call", a.error)
        return
    price, status = parse_price(a.text)
    r.add(PASS if status == "ok" else WARN, "probe call parses", f"status={status} price={price} finish={a.finish_reason}")
    r.add(WARN if a.finish_reason == "length" else PASS, "probe not truncated",
          f"{a.output_tokens} output tokens of max {sampling.max_tokens}")
    if a.output_tokens and a.latency_s:
        r.add(PASS, "single-stream speed", f"{a.output_tokens / a.latency_s:.1f} tok/s, {a.latency_s:.1f}s/call")
    b = be.complete(sys_p, user, sampling, seed=12345)
    c = be.complete(sys_p, user, sampling, seed=54321)
    if b.error or c.error:  # never let a dead server read as "outputs differ"
        r.add(FAIL, "seed checks", f"probe call failed: {b.error or c.error}")
        return
    r.add(PASS if b.text == a.text else WARN, "same seed -> same output",
          "deterministic" if b.text == a.text else "differs (batching non-determinism?); log it, reproducibility is approximate")
    r.add(PASS if c.text != a.text else WARN, "different seed -> different output",
          "ok" if c.text != a.text else "identical: is the seed or temperature being ignored?")
    if "deepseek" in spec["model"].lower():
        r.add(PASS if "</think>" in a.text else WARN, "R1 thinking block present",
              "reasoning visible" if "</think>" in a.text else "no </think>: chat template or reasoning parser issue?")


def check_paid(r: Report, spec: dict, sampling: Sampling, prompt: str, budget: Budget, probe: bool) -> None:
    prov, model = spec["provider"], spec["model"]
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "ifm": "IFM_K2_API_KEY"}.get(prov, f"{prov.upper()}_API_KEY")
    if prov == "anthropic":
        key = spec.get("api_key_env", key)  # a config may pick another Anthropic key by variable name
    r.add(PASS if os.environ.get(key) else FAIL, f"{key} set", "(value never printed)")
    try:
        budget.price(model)
        r.add(PASS, f"price known for {model}")
    except Exception as e:  # noqa: BLE001
        r.add(FAIL, f"price known for {model}", str(e))
    cap, spent = budget.cap(prov), budget.spent(prov)
    r.add(PASS if cap > 0 else FAIL, f"BUDGET_USD_{prov.upper()} cap", f"cap=${cap:.2f}, spent so far=${spent:.2f}")
    if probe and not r.failed:
        be = make_backend(spec)
        rid = budget.reserve(prov, model, 400, sampling.max_tokens)
        c = be.complete(system_prompt(prompt), user_message(1, []), sampling, seed=1)
        usd = budget.record(prov, model, "preflight", c.input_tokens, c.output_tokens, rid)
        price, status = parse_price(c.text)
        r.add(FAIL if c.error else (PASS if status == "ok" else WARN), f"paid probe {model}",
              c.error or f"status={status} price={price} in={c.input_tokens} out={c.output_tokens} ${usd:.4f} "
                         f"model_returned={c.model_returned} finish={c.finish_reason}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--probe-paid", action="store_true")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--approve-usd", type=float, default=None, help="needed with --probe-paid (tiny, e.g. 0.20)")
    ap.add_argument("--allow-dirty", action="store_true", help="permit uncommitted code (smoke tests only)")
    a = ap.parse_args(argv)
    load_env()
    r = Report()
    print(f"preflight: {a.config}")
    try:
        cfg = load_config(a.config)
        r.add(PASS, "config valid (strict schema)")
    except ConfigError as e:
        r.add(FAIL, "config valid (strict schema)", str(e))
        return 1
    if a.tests:
        t = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO / "tests")], capture_output=True, text=True)
        r.add(PASS if t.returncode == 0 else FAIL, "test suite", t.stdout.strip().splitlines()[-1] if t.stdout else t.stderr[-300:])
    cv = code_version()
    if cv["dirty"]:
        why = "no commit yet" if not cv["git_sha"] else f"{len(cv['dirty_files'])} uncommitted file(s), e.g. {cv['dirty_files'][:2]}"
        r.add(WARN if a.allow_dirty else FAIL, "code committed", f"{why}: a run must map to a git sha")
    else:
        r.add(PASS, "code committed", cv["git_sha"][:10])
    for prob in env_file_problems(REPO):
        r.add(FAIL, ".env hygiene", prob)
    runs = Path(os.environ.get("RUNS_DIR", REPO / "runs"))  # one runs dir per experiment family (sprint pre-flight)
    runs.mkdir(exist_ok=True)
    free_gb = shutil.disk_usage(runs).free / 1e9
    r.add(PASS if free_gb > 5 else FAIL, "disk space for runs/", f"{free_gb:.0f} GB free")
    if (REPO / "configs" / "seed_assignment.json").exists():
        r.add(PASS, "seed assignment pre-registered")
    if cfg["arm"] in ("P", "D", "Dfrozen", "Dplacebo"):
        from .run import build_jobs
        try:
            build_jobs(cfg, runs)
            r.add(PASS, "source L runs exist and are complete")
        except Exception as e:  # noqa: BLE001
            r.add(FAIL, "source L runs exist and are complete", str(e))
    sampling = Sampling(**cfg["sampling"])
    budget = Budget(runs / "_ledger", REPO / "configs" / "prices.yaml", session_cap_usd=a.approve_usd)
    seen = set()
    for f in ("firm1", "firm2"):
        spec = cfg.get(f)
        if not spec or (spec["provider"], spec["model"], spec.get("base_url")) in seen:
            continue
        seen.add((spec["provider"], spec["model"], spec.get("base_url")))
        if spec["provider"] == "vllm":
            check_vllm(r, spec, sampling, cfg["prompt"])
        elif spec["provider"] not in FREE_PROVIDERS:
            check_paid(r, spec, sampling, cfg["prompt"], budget, a.probe_paid)
    out = REPO / "results" / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(a.config).stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    (out / name).write_text(json.dumps({"config": a.config, "code": cv, "rows": r.rows}, indent=1))
    print(f"preflight {'FAILED' if r.failed else 'passed'} -> results/preflight/{name}")
    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
