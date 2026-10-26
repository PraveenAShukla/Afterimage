"""X0, the estimator positive control for the Price Afterimage (rule in docs/AFTERIMAGE.md, committed
before any X0 run). Reads the sigma-0 mock runs (x0-h<h>-sd0, x0long-h<h>-sd0), re-implements the mock partial matcher
in numpy from its spec, and checks the afterimage estimator against the process the mock actually ran.

  mock partial:w, firm i in round t:  p_i,t = fmt4( w * fmt3(p_-i,t-1) + (1 - w) * Nash + eps ),  eps ~ N(0, sigma)
  fmt3 is the 3-decimal history line in the prompt; fmt4 the mock's 4-decimal answer. Because of fmt3, a step
  (1 - w)(p - Nash) below 0.0005 never registers: slow matchers (h >= 200) freeze near their opening. So the target is
  A_q(W), the slope of THIS process, not the analytic w**(t - 20) (also reported, as A_analytic).

  .venv/bin/python analysis/x0_mc.py --runs-dir <dir with the x0 runs> [--R 200] [--n-boot 1000]

Checks (all must pass for X0 PASS; the empirical half on the mock runs themselves is analysis/afterimage.py x0_verdict):
  data path   sigma 0: the re-implementation started from each run's logged rounds 1-20 reproduces every logged price
              (max |diff| <= 1e-9)
  coverage    R synthetic experiments per (h, sigma, length), same plants, fresh noise: the within-level bootstrap 90%
              CI of A(W) covers A_q,sigma(W) (the mean over the R replicates) in >= 80% of replicates, every window
Output: results/afterimage/x0/mc.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afterimage as AI  # noqa: E402
from v2 import ROOT, SEED  # noqa: E402

from collusion.market import Market  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402

NASH = Market().nash()[0]
HS = (10, 50, 200, 800)
SDS = (0.0, 0.002, 0.004, 0.008)
STAGE = re.compile(r"^x0(long)?-h(\d+)-sd0$")


def plants(runs_dirs: list[Path]) -> dict:
    """{(h, long): [run dicts with x, window_id, rounds, logged p1/p2]} from the sigma-0 partial-matcher runs."""
    out = {}
    for rd in runs_dirs:
        for d in sorted(rd.iterdir()):
            mf = d / "manifest.json"
            if not mf.exists():
                continue
            man = json.loads(mf.read_text())
            m = STAGE.match(man["config"].get("stage", ""))
            if not m or not AI.ok(d):
                continue
            R = sorted(read_jsonl(d / "rounds.jsonl"), key=lambda r: r["t"])
            out.setdefault((int(m.group(2)), bool(m.group(1))), []).append(
                {"x": float(man["config"]["opening"][0]), "window_id": man["meta"].get("window_id"), "rounds": R[-1]["t"],
                 "w": float(AI.PARTIAL.match(man["config"]["firm1"]["strategy"]).group(1)),
                 "p1": [r["p1"] for r in R], "p2": [r["p2"] for r in R]})
    return out


def resim_exact(run: dict) -> tuple[list, list]:
    """sigma 0, in Python floats with the mock's own string formatting (bit-for-bit what the mock computes)."""
    w, p1, p2 = run["w"], list(run["p1"][:20]), list(run["p2"][:20])
    for _ in range(20, run["rounds"]):
        a = float(f"{w * float(f'{p2[-1]:.3f}') + (1 - w) * NASH:.4f}")
        b = float(f"{w * float(f'{p1[-1]:.3f}') + (1 - w) * NASH:.4f}")
        p1.append(a)
        p2.append(b)
    return p1, p2


def resim_mc(runs: list[dict], sd: float, R: int, rng) -> np.ndarray:
    """(R, n_runs, rounds) game-mean prices; noise makes rounding ties measure-zero, so np.round matches fmt."""
    w, T = runs[0]["w"], runs[0]["rounds"]
    p1 = np.empty((R, len(runs), T))
    p2 = np.empty((R, len(runs), T))
    p1[:, :, :20] = np.array([r["p1"][:20] for r in runs])[None]
    p2[:, :, :20] = np.array([r["p2"][:20] for r in runs])[None]
    for t in range(20, T):
        e1, e2 = (rng.normal(0, sd, (2, R, len(runs))) if sd else (0.0, 0.0))
        p1[:, :, t] = np.round(w * np.round(p2[:, :, t - 1], 3) + (1 - w) * NASH + e1, 4)
        p2[:, :, t] = np.round(w * np.round(p1[:, :, t - 1], 3) + (1 - w) * NASH + e2, 4)
    return (p1 + p2) / 2


def windows_for(T: int) -> list[tuple[int, int]]:
    return [w for w in AI.WINDOWS + AI.LONG if w[1] <= T]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", action="append", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "afterimage" / "x0" / "mc.json"))
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    P = plants([Path(p) for p in a.runs_dir])
    rng = np.random.default_rng(SEED)
    res = {"R": a.R, "n_boot": a.n_boot, "nash": NASH, "cells": {}, "data_path": {}}
    for (h, long_), runs in sorted(P.items()):
        tag = f"h{h}{'-long' if long_ else ''}"
        dev = 0.0
        for r in runs:
            s1, s2 = resim_exact(r)
            dev = max(dev, max(abs(u - v) for u, v in zip(s1 + s2, r["p1"] + r["p2"], strict=True)))
        res["data_path"][tag] = {"n_runs": len(runs), "max_abs_diff": dev, "pass": dev <= 1e-9}
        x = np.array([r["x"] for r in runs])
        W = windows_for(runs[0]["rounds"])
        for sd in (SDS if not long_ else (0.0,)):
            if sd:
                G = resim_mc(runs, sd, a.R, rng)                                # (R, n, T)
            else:                                                               # exact: the mock's own rounding
                G = np.array([[(np.array(u) + np.array(v)) / 2 for u, v in map(resim_exact, runs)]])
            Y = np.stack([G[:, :, w0 - 1:w1].mean(2) for w0, w1 in W], 2)       # (R, n, n_windows)
            A = np.array([AI.slopes(x, Y[i]) for i in range(Y.shape[0])])       # (R, n_windows)
            A_q = A.mean(0)
            cell = {"windows": [AI.wname(w) for w in W], "A_q": dict(zip(map(AI.wname, W), A_q.tolist(), strict=True)),
                    "A_analytic": {AI.wname(w): AI.a_true(runs[0]["w"], w) for w in W},
                    "A_sd_across_reps": dict(zip(map(AI.wname, W), A.std(0).tolist(), strict=True))}
            if sd:
                cov = np.zeros(len(W))
                for i in range(a.R):
                    bw = AI.boot_within(x, Y[i], rng, a.n_boot)
                    lo, hi = np.quantile(bw, 0.05, 0), np.quantile(bw, 0.95, 0)
                    cov += (lo <= A_q) & (A_q <= hi)
                cell["coverage90"] = dict(zip(map(AI.wname, W), (cov / a.R).tolist(), strict=True))
                cell["coverage_pass"] = bool((cov / a.R >= 0.80).all())
            res["cells"][f"{tag}-sd{sd:g}"] = cell
            print(tag, f"sd{sd:g}", {k: round(v, 3) for k, v in cell["A_q"].items()},
                  "coverage_pass" if cell.get("coverage_pass") else ("" if not sd else "COVERAGE FAIL"))
    res["pass"] = all(v["pass"] for v in res["data_path"].values()) and \
        all(c.get("coverage_pass", True) for c in res["cells"].values()) and bool(res["cells"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print("X0 MC", "PASS" if res["pass"] else "FAIL", "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
