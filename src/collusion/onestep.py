"""One-step drift map (E4): single Agent.decide() calls on injected level-shifted REAL windows, no games.

  python -m collusion.onestep CONFIG [--runs-dir runs] [--workers 32] [--dry-run]

CONFIG (YAML), same firm/sampling/prompt blocks as a run config, plus:
  stage: s1map-qwen3off            # output run id = <stage>_onestep_<mode>_<model>
  mode: diag | grid
  levels: [1.20, 1.275, ...]       # diag: own = rival = x;  grid: every (own, rival) pair of `levels`
  samples: 30                      # per cell
  seed_base: 61001                 # sample k of cell c uses run_seed = seed_base + 1000 * c + k
  opening_source: {template, pool, perm_seed[, cross_model]}

Each sample shows the agent rounds 181-200 of a source L run, level-shifted (opening.shifted_window) to the cell's
(own, rival) means, and asks for round 201. Every attempt is logged in calls.jsonl exactly as in a game (prompt,
raw output, sampling, tokens). samples.jsonl has one line per sample: cell, source run, delta, the visible window
means, the decided price and its status. The same agent code, prompts and parser as the games are used.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from .config import FIRM, SAMPLING, _check
from .game import firm_view
from .market import Market
from .opening import shifted_window, source_order
from .runlog import REPO, RunLog, load_run
from .run import _agent, _short

KEYS = {"stage": str, "prompt": str, "sampling": dict, "firm1": dict, "mode": str, "levels": list, "samples": int,
        "seed_base": int, "opening_source": dict, "market": dict, "notes": str, "prefix_rows": int, "phase": str}


def load(path: str) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    errs: list[str] = []
    _check(c, KEYS, "onestep", errs)
    for k in ("stage", "prompt", "sampling", "firm1", "mode", "levels", "samples", "seed_base", "opening_source"):
        if k not in c:
            errs.append(f"missing {k!r}")
    if not errs:
        _check(c["sampling"], SAMPLING, "sampling", errs)
        _check(c["firm1"], FIRM, "firm1", errs)
        if c["mode"] not in ("diag", "grid"):
            errs.append("mode must be diag or grid")
        if not all(isinstance(v, (int, float)) and 1.0 <= v <= 2.5 for v in c["levels"]):
            errs.append("levels must be numbers in [1, 2.5]")
        if c["samples"] < 1:
            errs.append("samples must be >= 1")
    if errs:
        raise ValueError(f"{path}: " + "; ".join(errs))
    return c


def cells(c: dict) -> list[tuple[float, float]]:
    L = [float(v) for v in c["levels"]]
    return [(x, x) for x in L] if c["mode"] == "diag" else [(o, r) for o in L for r in L]


def _source(c: dict, runs_dir: Path, seed: int) -> dict:
    run = load_run(runs_dir, c["opening_source"]["template"].format(seed=seed))
    if run["summary"] is None or run["summary"].get("status") != "complete":
        raise RuntimeError(f"source {run['manifest']['run_id']} is not complete")
    if run["manifest"]["config"]["arm"] != "L":
        raise RuntimeError("source must be an L run")
    if not c["opening_source"].get("cross_model") and run["manifest"]["config"]["firm1"]["model"] != c["firm1"]["model"]:
        raise RuntimeError("source model differs (set opening_source.cross_model: true for controls)")
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    c = load(a.config)
    runs_dir = Path(a.runs_dir)
    m = Market(**c.get("market", {}))
    order = source_order(c["opening_source"]["pool"], c["opening_source"]["perm_seed"])
    srcs = {s: _source(c, runs_dir, s) for s in order}
    jobs = []
    for ci, (own, riv) in enumerate(cells(c)):
        for k in range(c["samples"]):
            s = order[(ci * c["samples"] + k) % len(order)]
            n_pre = c.get("prefix_rows", 200)  # 200: rounds 181-200 shown, decide at 201 (v1); 20: rounds 1-20, decide at 21 (v2)
            rows, info = shifted_window(srcs[s]["rounds"], own, riv, n_prefix=n_pre)
            hist = [{"t": r["t"], **m.clear(r["p1"], r["p2"])} for r in rows[-20:]]
            jobs.append({"cell": ci, "own": own, "rival": riv, "k": k, "seed": c["seed_base"] + 1000 * ci + k,
                         "source_run": srcs[s]["manifest"]["run_id"], "info": info, "hist": hist})
    rid = f"{c['stage']}_onestep_{c['mode']}{'_' + c['phase'] if c.get('phase') else ''}_{_short(c['firm1']['model'])}"
    print(f"{rid}: {len(cells(c))} cells x {c['samples']} samples = {len(jobs)} decisions")
    if a.dry_run:
        return 0
    log = RunLog(runs_dir, rid)
    log.write_manifest({"config": {**c, "_config_path": a.config}, "kind": "onestep", "market": m.benchmarks(),
                        "opening_source_order": order, "n_jobs": len(jobs)})
    agent = _agent({**c, "firm2": c["firm1"]}, 1)

    def one(j: dict) -> dict:
        h = firm_view(j["hist"], 1)
        t_dec = h[-1]["round"] + 1
        d = agent.decide(t_dec, h, h[-1]["own_price"], j["seed"], 1, None, rid)
        for cl in d.calls:
            log.call({"arm": "onestep", "seed": j["seed"], "round": t_dec, "firm": 1, "cell": j["cell"], **cl,
                      "final_price": d.price, "decision_status": d.status})
        rec = {k: j[k] for k in ("cell", "own", "rival", "k", "seed", "source_run")}
        rec.update({"delta": j["info"]["delta"], "n_clipped": j["info"]["n_clipped"],
                    "win_mean_own": sum(x["own_price"] for x in h) / len(h),
                    "win_mean_rival": sum(x["rival_price"] for x in h) / len(h),
                    "last_own": h[-1]["own_price"], "last_rival": h[-1]["rival_price"],
                    "price": d.price, "status": d.status, "n_attempts": len(d.calls)})
        log._w("samples", rec)
        return rec

    out, fails = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            r = f.result()
            out.append(r)
            fails += str(r["status"]).startswith("fallback")
    summary = {"n": len(out), "status_ok": sum(r["status"] in ("ok", "ok_after_retry") for r in out),
               "fallback_or_fail": fails, "status": "complete"}
    log.finish(summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
