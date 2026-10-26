"""Afterimage CPU pack on the committed logs (EXPLORATORY).

  C10  validation audit: no prompt at t >= 41 shows a history row numbered <= 20 (the planted rows have left).
  C4   sub-monopoly slope: A(W) on the levels {1.55, 1.75} only, and the count of 1.95 games whose 121-170 level ends
       above monopoly (1.925).
  C6   within-cold-start persistence (v2L, 12 runs per model): run-level slope of the 121-170 level on the 41-60
       level, beside A(121-170)/A(41-60) from the planted conf runs.
  C7   natural afterimage (v2T: 20 real openings from the model's own cold starts x 3 seeds): slope of the 121-170
       level on the transplanted round-20 level, opening-cluster bootstrap; beside the planted A(121-170).

  .venv/bin/python analysis/afterimage_cpu.py [--runs-dir runs_cluster] [--skip-c10]
Output: results/afterimage/cpu_pack.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afterimage as AI  # noqa: E402
from v2 import ROOT, SEED, ok, slope  # noqa: E402

from collusion.runlog import read_jsonl  # noqa: E402
from collusion.validate import ROW  # noqa: E402

MONO = 1.925
N_BOOT = 10_000


def runs(rd: Path, prefix: str, arm: str):
    for d in sorted(rd.glob(f"{prefix}*")):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        man = json.loads(mf.read_text())
        if man["config"].get("phase") == "conf" and man["config"].get("arm") == arm and ok(d):
            yield d, man


def level(by: dict, a: int, b: int) -> float | None:
    return float(np.mean([(by[t]["p1"] + by[t]["p2"]) / 2 for t in range(a, b + 1)])) if all(t in by for t in range(a, b + 1)) else None


def c10(rd: Path) -> dict:
    n_calls, bad, n_runs = 0, 0, 0
    for d, _ in runs(rd, "v2O", "O"):
        n_runs += 1
        with open(d / "calls.jsonl") as f:
            for line in f:
                c = json.loads(line)
                if c.get("round", 0) < 41:
                    continue
                n_calls += 1
                bad += any(int(r[0]) <= 20 for r in ROW.findall(c.get("user_prompt") or ""))
    return {"runs": n_runs, "calls_t_ge_41": n_calls, "calls_with_a_row_le_20": bad, "pass": bad == 0}


def c4(rows: list[dict]) -> dict:
    out = {}
    for st in sorted({r["stage"] for r in rows}):
        g = [r for r in rows if r["stage"] == st and r["x"] in (1.55, 1.75)]
        hi = [r for r in rows if r["stage"] == st and r["x"] == 1.95]
        if len({r["x"] for r in g}) < 2:
            continue
        out[st] = {"A_sub_monopoly": {AI.wname(w): slope([r["x"] for r in g], [r["p"][AI.wname(w)] for r in g])
                                      for w in AI.WINDOWS + AI.RULE_WINDOWS if all(r["p"][AI.wname(w)] is not None for r in g)},
                   "n_1.95": len(hi), "n_1.95_late_above_monopoly": sum(r["p"]["121-170"] > MONO for r in hi)}
    return out


def c6(rd: Path, conf: dict, rng) -> dict:
    out = {}
    by_st = defaultdict(list)
    for d, man in runs(rd, "v2L", "L"):
        by = {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}
        e, la = level(by, 41, 60), level(by, 121, 170)
        if e is not None and la is not None:
            by_st[man["config"]["stage"]].append((e, la))
    for st, v in sorted(by_st.items()):
        x, y = np.array([a for a, _ in v]), np.array([b for _, b in v])
        bs = [slope(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(N_BOOT)) if np.ptp(x[i]) > 0]
        m = st.split("-", 1)[1]
        A = conf.get(f"v2O-{m}", {}).get("A", {}).get("p", {})
        ratio = A["121-170"]["A"] / A["41-60"]["A"] if "121-170" in A and A["41-60"]["A"] else None
        out[st] = {"n": len(v), "persistence_slope": slope(x, y), "ci90": [float(np.quantile(bs, .05)), float(np.quantile(bs, .95))],
                   "sd_41_60": float(x.std()), "sd_121_170": float(y.std()), "planted_A_ratio_121_170_over_41_60": ratio}
    return out


def c7(rd: Path, conf: dict, ref: dict, rng) -> dict:
    out = {}
    by_st = defaultdict(list)
    for d, man in runs(rd, "v2T", "T"):
        by = {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}
        z, z20, la = level(by, 20, 20), level(by, 1, 20), level(by, 121, 170)
        if z is not None and la is not None:
            by_st[man["config"]["stage"]].append((man["meta"]["transplant_source_run"], z, la, z20))
    for st, v in sorted(by_st.items()):
        ops = sorted({o[0] for o in v})
        x, y, xm = (np.array([o[k] for o in v]) for k in (1, 2, 3))
        idx = {o: [i for i, oo in enumerate(v) if oo[0] == o] for o in ops}
        bs, bm = [], []
        for _ in range(N_BOOT):
            ii = np.concatenate([idx[o] for o in rng.choice(ops, len(ops))])
            if np.ptp(x[ii]) > 0 and np.ptp(xm[ii]) > 0:
                bs.append(slope(x[ii], y[ii]))
                bm.append(slope(xm[ii], y[ii]))
        m = st.split("-", 1)[1]
        A = conf.get(f"v2O-{m}", {}).get("A", {}).get("p", {}).get("121-170")
        h2 = ref.get("H2", {}).get(st, {})
        out[st] = {"n_runs": len(v), "n_openings": len(ops), "natural_A_121_170": slope(x, y),
                   "ci90_opening_cluster": [float(np.quantile(bs, .05)), float(np.quantile(bs, .95))],
                   "natural_A_on_mean_1_20": slope(xm, y), "ci90_mean_1_20": [float(np.quantile(bm, .05)), float(np.quantile(bm, .95))],
                   "round20_range": [float(x.min()), float(x.max())],
                   "planted_A_121_170": A["A"] if A else None, "planted_ci90_within": A["ci90_within"] if A else None,
                   "icc": h2.get("icc"), "icc_ci95": h2.get("ci95")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs_cluster"))
    ap.add_argument("--out", default=str(ROOT / "results" / "afterimage" / "cpu_pack.json"))
    ap.add_argument("--skip-c10", action="store_true")
    a = ap.parse_args(argv)
    rd = Path(a.runs_dir)
    rng = np.random.default_rng(SEED)
    conf = json.loads((ROOT / "results" / "afterimage" / "conf" / "numbers.json").read_text())["stages"]
    ref = json.loads((ROOT / "results" / "v2" / "conf" / "numbers.json").read_text())
    rows, _ = AI.load([rd])
    res = {"C4": c4([r for r in rows if r["exp"] == "conf"]), "C6": c6(rd, conf, rng), "C7": c7(rd, conf, ref, rng)}
    if not a.skip_c10:
        res["C10"] = c10(rd)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    for k in ("C10",):
        if k in res:
            print(k, res[k])
    for st, e in res["C7"].items():
        print("C7", st, round(e["natural_A_121_170"], 3), [round(v, 3) for v in e["ci90_opening_cluster"]],
              "on mean 1-20", round(e["natural_A_on_mean_1_20"], 3), [round(v, 3) for v in e["ci90_mean_1_20"]], "planted", e["planted_A_121_170"] and round(e["planted_A_121_170"], 3))
    for st, e in res["C6"].items():
        print("C6", st, round(e["persistence_slope"], 3), [round(v, 3) for v in e["ci90"]], "planted ratio", e["planted_A_ratio_121_170_over_41_60"] and round(e["planted_A_ratio_121_170_over_41_60"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
