"""Mechanism CPU checks on EXISTING logs from the pre-sprint discovery runs (plan v2 addendum N1; descriptive).
Predictions were recorded first (docs/PREDICTIONS.md). Output: results/s1/mech_cpu.json.

  .venv/bin/python analysis/mech_cpu.py
C-a FM trajectories (rival fixed at 1.925): corr(early 1-10, last 50) and the runs that drift up > +0.05.
C-b after a forced cut (D arm, material cuts): in recoveries, does firm 1 raise first and firm 2 follow within 2?
C-c P arm: dp1(t) on max(dp2(t-1), 0) and min(dp2(t-1), 0), t = 21..300, cluster bootstrap over runs.
C-d one-step grid (Discovery-1 v2 maps): own-relative response split by the sign of (rival - own).
C-f K3: Discovery-0 arm S beta with a SEED-cluster bootstrap and a per-group RNG stream.
C-g exact repeat rate p(t) = p(t-1) in Discovery-0 arm S at x = 1.70 (live rounds).
C-i drift identity: OLS dL(t+1) on |G(t)| and BR(L(t)) - L(t) (+ intercept), live rounds 6-100 of arm S;
    A_self = -2 * coef(|G|); mock check first (averager 0, cutmatch 1, raisematch -1); inconclusive if fewer than
    5% of rounds have |G| > 0.005.
C-e (reasoning-text codes) is not run here: its dictionary has not been committed (addendum: commit before
counting). Reported as NOT RUN.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collusion.market import Market  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402

M = Market()
RC = ROOT / "runs_cluster"


def ok(d: Path) -> bool:
    s, v = d / "summary.json", d / "validation.json"
    return s.exists() and json.loads(s.read_text()).get("status") == "complete" and v.exists() and json.loads(v.read_text()).get("ok")


def rng_for(name: str) -> np.random.Generator:  # K3: one RNG stream per group, independent of other groups
    return np.random.default_rng(zlib.crc32(name.encode()))


def runs(pat: str) -> list[list[dict]]:
    return [read_jsonl(d / "rounds.jsonl") for d in sorted(RC.glob(pat)) if ok(d)]


def c_a() -> dict:
    out = {}
    for m, pat in (("Qwen3", "s2-qwen3_14b_off_FM_*"), ("Qwen2.5", "s2-qwen25_14b_FM_*")):
        e, l, up = [], [], 0
        for R in runs(pat):
            a, b = np.mean([r["p1"] for r in R[:10]]), np.mean([r["p1"] for r in R[-50:]])
            e.append(a)
            l.append(b)
            up += b - a > 0.05
        corr = float(np.corrcoef(e, l)[0, 1]) if len(e) > 2 else None
        out[m] = {"n": len(e), "corr_early_late": corr, "n_drift_up_gt_0.05": up, "mean_early": float(np.mean(e)),
                  "mean_late": float(np.mean(l)),
                  "label": "upward pull" if up >= 5 else "self-anchoring" if (corr or 0) > 0.7 and up == 0 else "neither"}
    return out


def c_b() -> dict:
    out = {}
    for m, pat in (("Qwen3", "s3-qwen3_14b_off_D_*"), ("Qwen2.5", "s3-qwen25_14b_D_*")):
        n_rec = n_lead = 0
        for R in runs(pat):
            by = {r["t"]: r for r in R}
            if not (by[201]["p1"] < by[200]["p1"] - 0.02):
                continue
            pre = by[200]["p2"]
            if not any(abs(by[t]["p2"] - pre) <= 0.02 for t in range(203, 211)):
                continue
            n_rec += 1
            t1 = next((t for t in range(202, 211) if by[t]["p1"] > by[t - 1]["p1"] + 0.005), None)
            t2 = next((t for t in range(202, 211) if by[t]["p2"] > by[t - 1]["p2"] + 0.005), None)
            n_lead += t1 is not None and (t2 is None or t1 <= t2) and t2 is not None and t2 - t1 <= 2
        out[m] = {"recoveries": n_rec, "firm1_leads_firm2_follows_within_2": n_lead,
                  "follows_raises": bool(n_rec and n_lead >= n_rec / 2)}
    return out


def c_c() -> dict:
    out = {}
    for m, pat in (("Qwen3", "s3-qwen3_14b_off_P_*"), ("Qwen2.5", "s3-qwen25_14b_P_*")):
        parts = []
        for R in runs(pat):
            X, y = [], []
            for t in range(21, 300):
                d2 = R[t - 1]["p2"] - R[t - 2]["p2"]
                X.append([1.0, max(d2, 0.0), min(d2, 0.0)])
                y.append(R[t]["p1"] - R[t - 1]["p1"])
            parts.append((np.array(X), np.array(y)))
        fit = lambda ps: np.linalg.lstsq(np.vstack([p[0] for p in ps]), np.concatenate([p[1] for p in ps]), rcond=None)[0]  # noqa: E731
        co = fit(parts)
        g = rng_for(f"c_c|{m}")
        bs = np.array([fit([parts[i] for i in g.integers(0, len(parts), len(parts))]) for _ in range(2000)])
        dif = bs[:, 2] - bs[:, 1]
        out[m] = {"n_runs": len(parts), "beta_plus": float(co[1]), "beta_minus": float(co[2]),
                  "ci95_plus": [float(np.quantile(bs[:, 1], .025)), float(np.quantile(bs[:, 1], .975))],
                  "ci95_minus": [float(np.quantile(bs[:, 2], .025)), float(np.quantile(bs[:, 2], .975))],
                  "lb95_minus_minus_plus": float(np.quantile(dif, .05)),
                  "ratchet": bool(co[2] >= 0.5 and co[1] <= 0.2 and np.quantile(dif, .05) > 0)}
    return out


def c_d() -> dict:
    out = {}
    for d in sorted(RC.glob("v2M-*_onestep_grid_disc_*")):
        S = [s for s in read_jsonl(d / "samples.jsonl") if s["status"] in ("ok", "ok_after_retry")]
        res = {}
        for sign, sel in (("rival_above", lambda s: s["rival"] > s["own"] + 1e-9), ("rival_below", lambda s: s["rival"] < s["own"] - 1e-9)):
            g = [s for s in S if sel(s)]
            x = np.array([s["rival"] - s["own"] for s in g])
            y = np.array([s["price"] - s["own"] for s in g])
            X = np.c_[np.ones_like(x), x]
            res[sign] = {"n": len(g), "rival_weight": float(np.linalg.lstsq(X, y, rcond=None)[0][1])}
        out[d.name.split("_onestep")[0]] = res
    return out


def c_f() -> dict:
    out = {}
    for st in ("s1S-qwen3off", "s1S-qwen25", "s1S-gptoss"):
        rows = []
        for d in RC.glob(f"{st}_S_*"):
            if not ok(d):
                continue
            man = json.loads((d / "manifest.json").read_text())
            R = read_jsonl(d / "rounds.jsonl")
            n_pre = man["config"]["rounds"] - man["config"]["window"]
            sel = [r for r in R if 61 <= r["t"] - n_pre <= 100]
            rows.append((man["seed"], man["config"]["opening"][0], float(np.mean([(r["p1"] + r["p2"]) / 2 for r in sel]))))
        seeds = sorted({r[0] for r in rows})
        x, y = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
        sl = lambda xx, yy: float(np.polyfit(xx, yy, 1)[0])  # noqa: E731
        g = rng_for(f"c_f|{st}")
        bs = []
        for _ in range(10_000):  # resample SEEDS as clusters: a seed carries all its levels (same seed + window per level)
            pick = g.choice(seeds, len(seeds))
            ii = np.concatenate([np.where(np.array([r[0] for r in rows]) == s)[0] for s in pick])
            bs.append(sl(x[ii], y[ii]))
        lo, hi = float(np.quantile(bs, .05)), float(np.quantile(bs, .95))
        lab = "opening-determined" if lo > 0.5 else "target-seeking" if hi < 0.2 else "intermediate/threshold"
        out[st] = {"n": len(rows), "n_seeds": len(seeds), "beta": sl(x, y), "lower95": lo, "upper95": hi, "label": lab}
    return out


def c_g() -> dict:
    out = {}
    for st in ("s1S-qwen3off", "s1S-qwen25", "s1S-gptoss"):
        same = tot = 0
        for d in RC.glob(f"{st}_S_*x1.700-1.700*"):
            if not ok(d):
                continue
            R = [r for r in read_jsonl(d / "rounds.jsonl") if not r.get("is_prefix")]
            for a, b in zip(R[:-1], R[1:], strict=True):
                for f in ("p1", "p2"):
                    same += abs(a[f] - b[f]) < 1e-9
                    tot += 1
        out[st] = {"repeat_rate": same / tot if tot else None, "n_pairs": tot}
    return out


def drift_identity(rows_list: list[list[dict]], name: str) -> dict:
    n_gap, n_all, parts = 0, 0, []
    for R in rows_list:
        live = [r for r in R if not r.get("is_prefix")]
        Xr, yr = [], []
        for t in range(5, min(100, len(live) - 1)):
            a, b = live[t], live[t + 1]
            L, G = (a["p1"] + a["p2"]) / 2, a["p1"] - a["p2"]
            Lb = (b["p1"] + b["p2"]) / 2
            Xr.append([1.0, abs(G), M.best_response(L) - L])
            yr.append(Lb - L)
            n_gap += abs(G) > 0.005
            n_all += 1
        parts.append((np.array(Xr), np.array(yr)))
    if not parts or n_all == 0:
        return {"n": 0}
    fit = lambda ps: np.linalg.lstsq(np.vstack([p[0] for p in ps]), np.concatenate([p[1] for p in ps]), rcond=None)[0]  # noqa: E731
    co = fit(parts)
    g = rng_for(f"c_i|{name}")
    bs = np.array([fit([parts[i] for i in g.integers(0, len(parts), len(parts))]) for _ in range(2000)])
    a = -2 * bs[:, 1]
    share = n_gap / n_all
    return {"A_self": float(-2 * co[1]), "ci95": [float(np.quantile(a, .025)), float(np.quantile(a, .975))],
            "coef_br_gap": float(co[2]), "share_gap_gt_0.005": share, "n_rounds": n_all,
            "inconclusive": share < 0.05}


def c_i() -> dict:
    out = {"mock": {}, "llm": {}}
    for pol in ("averager", "cutmatch", "raisematch"):
        out["mock"][pol] = drift_identity(runs(f"mci-{pol}_S_*"), f"mock|{pol}")
    exp = {"averager": 0.0, "cutmatch": 1.0, "raisematch": -1.0}
    out["mock_check_passed"] = all(abs(out["mock"][p].get("A_self", 9) - v) <= 0.15 for p, v in exp.items())
    for st in ("s1S-qwen3off", "s1S-qwen25", "s1S-gptoss"):
        out["llm"][st] = drift_identity(runs(f"{st}_S_*"), st) if out["mock_check_passed"] else {"void": "mock check failed"}
    return out


def main() -> int:
    res = {"C-a": c_a(), "C-b": c_b(), "C-c": c_c(), "C-d": c_d(), "C-e": "NOT RUN (dictionary not committed)",
           "C-f (K3)": c_f(), "C-g": c_g(), "C-i": c_i()}
    p = ROOT / "results" / "s1" / "mech_cpu.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=lambda o: round(o, 3) if isinstance(o, float) else str(o))[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
