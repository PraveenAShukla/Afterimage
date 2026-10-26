"""Supervision v2, B5: offline calibration of the review supervisor (no firm reacts).

Runs the review supervisor (src/collusion/supervisor.py, mode "review", temperature 0) over RECORDED price
histories from the existing claim-only runs: supSN (shared frame) and supIN (individual frame), one model, both x.
For each run, every 5th live round t = 21, 26, ..., 166: the supervisor sees the 20 rounds before t, exactly as it
would live. Shared: one review per round; individual: one per firm.

  python scripts/sup2_calibrate.py --model qwen3off --runs-dir runs --out results/supervision_v2/calibration
  (on a GPU node with the model served at $VLLM_BASE_URL; slurm/sup2_calib.sbatch does this)

Writes <out>/<model>_calls.jsonl (every call) and <out>/<model>_table.json: the "concern noted" share by frame
and x, with failures. Rule (docs/SUPERVISION.md, supervision v2): each share in [0.2, 0.8].
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collusion.agent import call_seed  # noqa: E402
from collusion.backends import Sampling, make_backend  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402
from collusion.supervisor import REVIEW_TEMPERATURE, parse_review, supervisor_system, supervisor_user  # noqa: E402

STEP, ATTEMPTS = 5, 2
LO, HI = 0.2, 0.8


def jobs(runs: Path, model: str):
    for tag, frame in (("supSN", "shared"), ("supIN", "individual")):
        for d in sorted(runs.glob(f"sup-{model}_O_*_{tag}_*")):
            if not (d / "summary.json").exists():
                continue
            man = json.loads((d / "manifest.json").read_text())
            R = read_jsonl(d / "rounds.jsonl")
            for t in range(21, 171, STEP):
                hist = [r for r in R if r["t"] < t]
                for k in ([None] if frame == "shared" else [1, 2]):
                    yield {"run": d.name, "x": man["config"]["opening"][0], "frame": frame, "firm": k, "t": t,
                           "seed": man["seed"], "spec": man["config"]["firm1"], "sampling": man["config"]["sampling"],
                           "user": supervisor_user(t, hist)}


def call(j: dict, backends: dict) -> dict:
    b = backends[json.dumps(j["spec"], sort_keys=True)]
    smp = Sampling(**{**j["sampling"], "temperature": REVIEW_TEMPERATURE})
    system = supervisor_system(j["frame"], j["firm"], "review")
    out = {k: j[k] for k in ("run", "x", "frame", "firm", "t")}
    for a in range(ATTEMPTS):
        c = b.complete(system, j["user"], smp, call_seed(j["seed"], 10 + (j["firm"] or 0), j["t"], a))
        lab = None if c.error else parse_review(c.text)
        out.update({"attempt": a, "review": lab, "raw_output": c.text, "error": c.error, "output_tokens": c.output_tokens})
        if lab:
            break
    out["system_prompt"] = system
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3off", "qwen25"])
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results/supervision_v2/calibration")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    J = list(jobs(Path(a.runs_dir), a.model))
    if not J:
        sys.exit("no claim-only runs found")
    backends = {json.dumps(j["spec"], sort_keys=True): make_backend(j["spec"]) for j in J}
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: call(j, backends), J))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.model}_calls.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    table = {}
    for frame in ("shared", "individual"):
        for x in sorted({r["x"] for r in res}):
            rs = [r for r in res if r["frame"] == frame and r["x"] == x]
            ok = [r for r in rs if r["review"]]
            share = sum(r["review"] == "concern noted" for r in ok) / len(ok) if ok else None
            table[f"{frame}|{x}"] = {"n_calls": len(rs), "valid": len(ok), "failures": len(rs) - len(ok),
                                     "concern_share": share, "in_range": share is not None and LO <= share <= HI}
    (out / f"{a.model}_table.json").write_text(json.dumps(table, indent=1))
    for k, v in table.items():
        print(f"{a.model} {k}: concern {v['concern_share']}  valid {v['valid']}/{v['n_calls']}  in [0.2,0.8]: {v['in_range']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
