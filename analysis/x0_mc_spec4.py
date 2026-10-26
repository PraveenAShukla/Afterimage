"""X0 Monte Carlo to the amended test specification (L1 + L2), run AFTER the original X0 output was read: POST HOC.
The committed X0 (analysis/x0_mc.py, tag prereg-x0) stays as it is; this adds what the amended specification asks for:
  - h in {10, 50, 125, 200, 480, 800}; partial matcher on both firms, the mock's exact recursion with 3-decimal display,
    4-decimal play and the [1, 2.5] clip, started from the real logged plants (rounds 1-20, by level and window);
  - per-call noise sigma_call tuned per h so the between-game SD of the 121-170 level (within a level) equals
    sigma_eff in {0.05, 0.11};
  - truth = the continuous A_true(W) = mean of w**(t - 20) (noise >= 0.002 dithers the 0.001 display step);
  - R = 1000 synthetic experiments per cell, designs quick 2x12, standard 3x12, full 3x18, conf 3x24 (window i ->
    order[i % 18]), independent noise per run;
  - coverage of the 90% CI for BOTH the within-level bootstrap and the window-cluster bootstrap (1000 draws).
  .venv/bin/python analysis/x0_mc_spec4.py --runs-dir runs_afterimage_x0 [--R 1000] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afterimage as AI  # noqa: E402
from v2 import ROOT, SEED, ok  # noqa: E402

from collusion.market import Market  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402

NASH = Market().nash()[0]
HS = (10, 50, 125, 200, 480, 800)
SIG_EFF = (0.05, 0.11)
DESIGNS = {"quick 2x12": ((1.55, 1.95), 12), "standard 3x12": ((1.55, 1.75, 1.95), 12),
           "full 3x18": ((1.55, 1.75, 1.95), 18), "conf 3x24": ((1.55, 1.75, 1.95), 24)}
WINS = [(41, 60), (81, 100), (121, 170)]
T = 170


def load_plants(rd: Path) -> tuple[dict, list]:
    """{(x, window_id): (p1[0:20], p2[0:20])} and the opening order (window ids in job order) from the x0-h10-sd0 runs."""
    P, order = {}, {}
    for d in sorted(rd.glob("x0-h10-sd0_*")):
        man = json.loads((d / "manifest.json").read_text())
        if not ok(d):
            continue
        R = sorted(read_jsonl(d / "rounds.jsonl"), key=lambda r: r["t"])[:20]
        x, wid = float(man["config"]["opening"][0]), man["meta"]["window_id"]
        P[(x, wid)] = (np.array([r["p1"] for r in R]), np.array([r["p2"] for r in R]))
        order[man["meta"]["index"] if "index" in man["meta"] else len(order)] = wid
    ids = [order[i] for i in sorted(order)][:18]
    return P, ids


def design_runs(P: dict, ids: list, levels, n) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, wid, a, b = [], [], [], []
    for x in levels:
        for i in range(n):
            w = ids[i % 18]
            xs.append(x)
            wid.append(w)
            a.append(P[(x, w)][0])
            b.append(P[(x, w)][1])
    return np.array(xs), np.array(wid), np.array(a), np.array(b)


def simulate(w: float, sd: float, A0: np.ndarray, B0: np.ndarray, R: int, rng) -> np.ndarray:
    """(R, n, T) game-mean prices; noise > 0 everywhere here, so np.round is equivalent to the mock's formatting."""
    n = A0.shape[0]
    p1, p2 = np.empty((R, n, T)), np.empty((R, n, T))
    p1[:, :, :20], p2[:, :, :20] = A0[None], B0[None]
    for t in range(20, T):
        e = rng.normal(0, sd, (2, R, n))
        p1[:, :, t] = np.clip(np.round(w * np.round(p2[:, :, t - 1], 3) + (1 - w) * NASH + e[0], 4), 1.0, 2.5)
        p2[:, :, t] = np.clip(np.round(w * np.round(p1[:, :, t - 1], 3) + (1 - w) * NASH + e[1], 4), 1.0, 2.5)
    return (p1 + p2) / 2


def late_sd(G: np.ndarray, x: np.ndarray) -> float:
    L = G[:, :, 120:170].mean(2)
    return float(np.mean([L[:, x == v].std(1, ddof=1).mean() for v in np.unique(x)]))


def tune(w: float, target: float, A0, B0, x, rng) -> float:
    lo, hi = 1e-5, 0.2
    for _ in range(22):
        mid = (lo * hi) ** 0.5
        (lo, hi) = (mid, hi) if late_sd(simulate(w, mid, A0, B0, 64, rng), x) < target else (lo, mid)
    return (lo * hi) ** 0.5


def counts_within(x: np.ndarray, B: int, rng) -> np.ndarray:
    C = np.zeros((B, len(x)))
    for v in np.unique(x):
        ix = np.where(x == v)[0]
        C[:, ix] = rng.multinomial(len(ix), [1 / len(ix)] * len(ix), size=B)
    return C


def counts_cluster(wid: np.ndarray, B: int, rng) -> np.ndarray:
    u = np.unique(wid)
    k = rng.multinomial(len(u), [1 / len(u)] * len(u), size=B)            # (B, n_ids)
    return k[:, np.searchsorted(u, wid)].astype(float)                    # every run of a drawn id, times its count


def boot_slopes(C: np.ndarray, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Weighted OLS slope per draw for every window column: (B, n_windows)."""
    sw = C.sum(1, keepdims=True)
    xm = (C @ x)[:, None] / sw
    ym = (C @ Y) / sw
    sxy = (C * x) @ Y / sw - xm * ym
    sxx = ((C * x * x).sum(1, keepdims=True) / sw) - xm ** 2
    return sxy / sxx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results" / "afterimage" / "x0" / "mc_spec4.json"))
    a = ap.parse_args(argv)
    P, ids = load_plants(Path(a.runs_dir))
    rng = np.random.default_rng(SEED + 4)
    res = {"status": "POST HOC (run after the original X0 output was read)", "R": a.R, "n_boot": a.n_boot, "cells": {}}
    for h in HS:
        w = 0.5 ** (1 / h)
        truth = {AI.wname(W): AI.a_true(w, W) for W in WINS}
        xc, _, A0c, B0c = design_runs(P, ids, (1.55, 1.75, 1.95), 24)
        for se in SIG_EFF:
            sd = tune(w, se, A0c, B0c, xc, rng)
            for dn, (lv, n) in DESIGNS.items():
                x, wid, A0, B0 = design_runs(P, ids, lv, n)
                G = simulate(w, sd, A0, B0, a.R, rng)
                Y = np.stack([G[:, :, W[0] - 1:W[1]].mean(2) for W in WINS], 2)           # (R, n, W)
                tv = np.array([truth[AI.wname(W)] for W in WINS])
                Ahat = np.array([AI.slopes(x, Y[i]) for i in range(a.R)])
                cov = {"within": np.zeros(len(WINS)), "cluster": np.zeros(len(WINS))}
                for i in range(a.R):
                    for kind, C in (("within", counts_within(x, a.n_boot, rng)), ("cluster", counts_cluster(wid, a.n_boot, rng))):
                        bs = boot_slopes(C, x, Y[i])
                        lo, hi = np.nanquantile(bs, 0.05, 0), np.nanquantile(bs, 0.95, 0)
                        cov[kind] += (lo <= tv) & (tv <= hi)
                cell = {"h": h, "sigma_eff": se, "sigma_call": sd, "design": dn, "truth": truth,
                        "mean_Ahat": dict(zip(truth, Ahat.mean(0).tolist(), strict=True)),
                        "bias": dict(zip(truth, (Ahat.mean(0) - tv).tolist(), strict=True)),
                        "sd_Ahat": dict(zip(truth, Ahat.std(0).tolist(), strict=True)),
                        "coverage90_within": dict(zip(truth, (cov["within"] / a.R).tolist(), strict=True)),
                        "coverage90_cluster": dict(zip(truth, (cov["cluster"] / a.R).tolist(), strict=True))}
                res["cells"][f"h{h}-se{se}-{dn}"] = cell
                print(f"h{h} se{se} {dn}: sd_call {sd:.5f} bias121 {cell['bias']['121-170']:+.3f} "
                      f"cov_w {cell['coverage90_within']['121-170']:.3f} cov_c {cell['coverage90_cluster']['121-170']:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
