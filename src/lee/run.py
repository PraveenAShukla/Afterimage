"""CLI: run a Lee-style cell.   python -m lee.run CONFIG [--runs-dir runs_lee] [--approve-usd X] [--dry-run]

Config (YAML, strict keys):
  stage: str; phase: rep|disc|conf; cell: str (name used in run_id)
  firms: list of n specs {provider, model, extra?}   (or `firm: {...}` + n for a homogeneous cell)
  rounds: 200; seeds: [int]; max_output_tokens: int; est_output_tokens: int; parallel_games: int
  opening (optional): {x, source_template ('...{seed}'), pool: [seeds], perm_seed: int, window: [lo, hi]}
Positions are randomised per run (random.Random(seed).sample), as in Lee. Completed runs (summary.json) are skipped.
Paid providers need --approve-usd (session cap) and BUDGET_USD_<PROVIDER> in .env; unset caps refuse every call.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from collusion.budget import Budget
from collusion.env import load_env
from collusion.runlog import REPO, RunLog, read_jsonl

from .backend import ChatBackend, MockChat
from .game import LeeAgent, play
from .market import NMarket
from .opening import order, shifted
from .validate import write_validation

KEYS = {"stage", "phase", "cell", "firms", "firm", "n", "rounds", "seeds", "max_output_tokens", "est_output_tokens",
        "parallel_games", "opening", "notes"}
FIRM = {"provider", "model", "extra", "strategy"}


def load(path: str) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    errs = [f"unknown key {k!r}" for k in c if k not in KEYS]
    for k in ("stage", "phase", "cell", "rounds", "seeds", "max_output_tokens"):
        if k not in c:
            errs.append(f"missing {k!r}")
    if ("firms" in c) == ("firm" in c):
        errs.append("give exactly one of firms (list) or firm (+ n)")
    if "firm" in c:
        c["firms"] = [c.pop("firm")] * int(c.pop("n", 4))
    for f in c.get("firms", []):
        errs += [f"firm: unknown key {k!r}" for k in f if k not in FIRM]
    if c.get("phase") not in ("rep", "disc", "conf"):
        errs.append("phase must be rep|disc|conf")
    if not (isinstance(c.get("seeds"), list) and c["seeds"] and len(set(c["seeds"])) == len(c["seeds"])):
        errs.append("seeds must be a non-empty list of distinct ints")
    op = c.get("opening")
    if op is not None and (set(op) - {"x", "source_template", "pool", "perm_seed", "window"} or "{seed}" not in op.get("source_template", "")):
        errs.append("opening needs {x, source_template with {seed}, pool, perm_seed, window?}")
    if errs:
        raise ValueError(f"{path}: " + "; ".join(errs))
    c["_config_path"] = path
    return c


def backend(spec: dict, market: NMarket):
    return MockChat(spec["model"], spec.get("strategy", "avg"), market) if spec["provider"] == "mock" else \
        ChatBackend(spec["provider"], spec["model"], spec.get("extra"))


def run_one(c: dict, seed: int, i: int, runs: Path, budget: Budget | None, order_: list[int] | None) -> dict:
    m = NMarket(n=len(c["firms"]))
    rid = f"{c['stage']}_{c['phase']}_{c['cell']}" + (f"_x{c['opening']['x']:.3f}" if c.get("opening") else "") + f"_s{seed}"
    d = runs / rid
    if (d / "summary.json").exists():
        return {"run_id": rid, "skipped": True}
    if d.exists():  # a started, unfinished run (crash or kill): keep it for inspection, never append to it
        q = runs / "_aborted" / f"{rid}__{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
        q.parent.mkdir(parents=True, exist_ok=True)
        d.rename(q)
        print(f"[quarantined] partial run {rid} -> {q.relative_to(runs)}", flush=True)
    pos = random.Random(seed).sample(range(len(c["firms"])), len(c["firms"]))  # position j plays firms[pos[j]]
    specs = [c["firms"][k] for k in pos]
    prefix, info = None, None
    if c.get("opening"):
        op = c["opening"]
        src = op["source_template"].format(seed=order_[i % len(order_)])
        sd = runs / src
        if not (sd / "summary.json").exists() or json.loads((sd / "summary.json").read_text()).get("status") != "complete" \
                or not json.loads((sd / "validation.json").read_text()).get("ok"):
            raise RuntimeError(f"{rid}: opening source {src} is not a complete, validated R0 run (run R0 first)")
        prefix, info = shifted(read_jsonl(runs / src / "rounds.jsonl"), float(op["x"]), *op.get("window", [31, 50]))
        info["source_run"] = src
    log = RunLog(runs, rid)
    agents = [LeeAgent(backend(s, m), j, m.n, c["max_output_tokens"]) for j, s in enumerate(specs)]
    log.write_manifest({"config": {k: v for k, v in c.items() if not k.startswith("_")}, "config_path": c["_config_path"],
                        "seed": seed, "positions": [s["model"] for s in specs], "benchmarks": m.benchmarks(),
                        "system_prompts": [a.system for a in agents], "opening": info})
    s = play(log, m, agents, c["rounds"], seed, prefix, budget)
    log.finish(s)
    errs = write_validation(log.dir)
    print(f"[done] {rid} delta={s.get('delta')} status={s['status']} fallbacks={s['fallbacks']} invalid={len(errs)}", flush=True)
    return {"run_id": rid, **s, "validation_errors": len(errs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--runs-dir", default=str(REPO / "runs_lee"))
    ap.add_argument("--ledger-dir", default=None, help="budget ledger dir (default <runs-dir>/_ledger). One per provider "
                    "process keeps the per-call ledger re-parse fast; the spend-rate tripwire is then per ledger")
    ap.add_argument("--approve-usd", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    load_env()
    c = load(a.config)
    runs = Path(a.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    paid = any(f["provider"] != "mock" for f in c["firms"])
    est = len(c["seeds"]) * c["rounds"] * len(c["firms"])
    print(f"{c['stage']}/{c['cell']}: {len(c['seeds'])} runs x {c['rounds']} rounds x {len(c['firms'])} firms = {est} calls (no retries)")
    if a.dry_run:
        return 0
    if paid and a.approve_usd is None:
        print("REFUSED: paid providers need --approve-usd (a hard session cap)")
        return 2
    budget = Budget(Path(a.ledger_dir) if a.ledger_dir else runs / "_ledger", REPO / "configs" / "prices.yaml",
                    session_cap_usd=a.approve_usd) if paid else None
    ord_ = order(c["opening"]["pool"], c["opening"]["perm_seed"]) if c.get("opening") else None
    rc = 0
    with ThreadPoolExecutor(max_workers=c.get("parallel_games", 4)) as ex:
        for f in as_completed([ex.submit(run_one, c, s, i, runs, budget, ord_) for i, s in enumerate(c["seeds"])]):
            r = f.result()
            rc |= int(r.get("status", "complete") != "complete" or r.get("validation_errors", 0) > 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
