"""S1 analysis (E5): is the long-run price set by the opening? Frozen and git-tagged before the decisive runs.

  .venv/bin/python analysis/s1.py --runs-dir runs_cluster [--maps-dir runs_cluster] [--out results/s1]

Groups = run config `stage` (one stage per model or control, e.g. s1S-qwen3off, s1S-ctl-averager, s1A-qwen3off).
Only complete, validated runs (validation.json ok) are used; everything excluded is counted and reported.

Per run (arms S and A; live rounds are t > rounds - window):
  final  = mean of BOTH firms' prices over live rounds 61-100 (the last 40 of the 100 live rounds)
  delta  = collusion index of the mean per-firm profit over the same rounds
Arm S, per group (PRIMARY):
  beta   = OLS slope of final on the opening x. CI: bootstrap resampling runs WITHIN level (10,000); one-sided 95%
           bounds = 5th / 95th percentiles. p_perm = permutation of the x labels across runs (10,000), two-sided,
           H0: beta = 0.
  rule   = "opening-determined" if lower95 > 0.5; "target-seeking" if upper95 < 0.2; else "intermediate/threshold"
           (middle band added after the grim-trigger control).
  secondary: delta(2.10) - delta(1.30) > 0.8 with delta(1.30) < 0. The drift-map prediction is DESCRIPTIVE ONLY
           (it fails on history-dependent controls); its hit count is reported, never used as a criterion.
Arm A, per group: mean |final - (x1 + x2) / 2|, bootstrap CI; averaging falsified if > 0.10 and lower CI > 0.
Drift map (one-step samples.jsonl, E4):
  diag: level means m(x) = E[p | window at x]; drift d(x) = m(x) - x; residuals kept for noise.
  grid: OLS p = c + a * own + m * rival (+ bootstrap CI over samples).
  iterate: x_{t+1} = interp(m)(x_t) + resampled residual, 100 steps from each opening, 2,000 paths; the prediction is
  the mean of x over steps 61-100 (mean and 90% band). Written to predictions.json, to be COMMITTED before arm S.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collusion.market import Market, collusion_index  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402

M = Market()
B = M.benchmarks()
N_BOOT = N_PERM = 10_000
SEED = 20261016
LIVE_LO, LIVE_HI = 61, 100  # live rounds used for `final`
LEVELS_PRIMARY = (1.30, 1.47, 1.70, 1.925, 2.10)


# ---------------------------------------------------------------- loading
def load_runs(runs_dir: Path, arms=("S", "A")) -> tuple[list[dict], dict]:
    rows, excluded = [], defaultdict(int)
    for d in sorted(runs_dir.iterdir()):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        man = json.loads(mf.read_text())
        c = man.get("config", {})
        if c.get("arm") not in arms:
            continue
        summ = d / "summary.json"
        val = d / "validation.json"
        if not summ.exists() or json.loads(summ.read_text()).get("status") != "complete":
            excluded[(c["stage"], "incomplete")] += 1
            continue
        if not val.exists() or not json.loads(val.read_text()).get("ok"):
            excluded[(c["stage"], "invalid")] += 1
            continue
        R = read_jsonl(d / "rounds.jsonl")
        n_pre = c["rounds"] - c["window"]
        live = [r for r in R if r["t"] > n_pre]
        sel = [r for r in live if LIVE_LO <= r["t"] - n_pre <= LIVE_HI]
        if len(sel) != LIVE_HI - LIVE_LO + 1:
            excluded[(c["stage"], "short")] += 1
            continue
        p = np.mean([r["p1"] for r in sel] + [r["p2"] for r in sel])
        pi = np.mean([(r["pi1"] + r["pi2"]) / 2 for r in sel])
        x1, x2 = c["opening"]
        rows.append({"stage": c["stage"], "arm": c["arm"], "model": c["firm1"]["model"], "run_id": man["run_id"],
                     "seed": man["seed"], "x1": x1, "x2": x2, "final": float(p),
                     "delta": float(collusion_index(pi, B["piN"], B["piM"])),
                     "final1": float(np.mean([r["p1"] for r in sel])), "final2": float(np.mean([r["p2"] for r in sel])),
                     "source": man["meta"]["opening_source_run"]})
    return rows, {f"{k[0]}|{k[1]}": v for k, v in excluded.items()}


# ---------------------------------------------------------------- arm S
def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def classify(lo95: float, hi95: float, hi_thr: float = 0.5, lo_thr: float = 0.2) -> str:
    """Decision rule (PREREG_S1, fixed on the scripted controls only): opening-determined if the one-sided
    95% lower bound > 0.5; target-seeking if the upper bound < 0.2; otherwise intermediate/threshold. The middle band
    exists because a two-basin policy (grim trigger at 1.80) has a linear slope of 0.48 and is NOT target-seeking."""
    if lo95 > hi_thr:
        return "opening-determined"
    if hi95 < lo_thr:
        return "target-seeking"
    return "intermediate/threshold"


def slope_test(runs: list[dict], rng) -> dict:
    x = np.array([r["x1"] for r in runs])
    y = np.array([r["final"] for r in runs])
    beta = ols_slope(x, y)
    levels = sorted(set(x))
    idx = [np.where(x == lv)[0] for lv in levels]
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):  # resample runs WITHIN each level: the design (levels, n per level) is fixed
        ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx])
        boot[b] = ols_slope(x[ii], y[ii])
    perm = np.array([ols_slope(rng.permutation(x), y) for _ in range(N_PERM)])
    lo, hi = float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))
    return {"beta": beta, "lower95_one_sided": lo, "upper95_one_sided": hi,
            "ci90_two_sided": [lo, hi], "p_perm_beta0": float(((np.abs(perm) >= abs(beta) - 1e-12).sum() + 1) / (N_PERM + 1)),
            "n": len(runs), "n_per_level": {str(lv): int(len(ix)) for lv, ix in zip(levels, idx, strict=True)},
            "rule": classify(lo, hi)}


def level_table(runs: list[dict], rng) -> list[dict]:
    out = []
    by = defaultdict(list)
    for r in runs:
        by[r["x1"]].append(r)
    for lv in sorted(by):
        f = np.array([r["final"] for r in by[lv]])
        dl = np.array([r["delta"] for r in by[lv]])
        bf = rng.choice(f, (N_BOOT, len(f))).mean(1)
        bd = rng.choice(dl, (N_BOOT, len(dl))).mean(1)
        out.append({"x": lv, "n": len(f), "final_mean": float(f.mean()), "final_sd": float(f.std(ddof=1)) if len(f) > 1 else None,
                    "final_ci95": [float(np.quantile(bf, .025)), float(np.quantile(bf, .975))],
                    "delta_mean": float(dl.mean()), "delta_ci95": [float(np.quantile(bd, .025)), float(np.quantile(bd, .975))],
                    "seeds": sorted(r["seed"] for r in by[lv])})
    return out


def secondary(table: list[dict], pred: dict | None) -> dict:
    t = {round(r["x"], 3): r for r in table}
    out = {}
    if 1.3 in t and 2.1 in t:
        gap = t[2.1]["delta_mean"] - t[1.3]["delta_mean"]
        out["delta_gap_2.10_minus_1.30"] = gap
        out["delta_1.30"] = t[1.3]["delta_mean"]
        out["criterion_delta_gap"] = bool(gap > 0.8 and t[1.3]["delta_mean"] < 0)
    if pred:
        hits = [abs(pred[str(k)]["mean"] - t[k]["final_mean"]) <= 0.08 for k in t if str(k) in pred]
        out["drift_map_hits_descriptive"] = f"{sum(hits)}/{len(hits)}"  # not a criterion (see module docstring)
    return out


# ---------------------------------------------------------------- arm A
def midpoint_test(runs: list[dict], rng) -> dict:
    dev = np.array([abs(r["final"] - (r["x1"] + r["x2"]) / 2) for r in runs])
    b = rng.choice(dev, (N_BOOT, len(dev))).mean(1)
    lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
    per = defaultdict(list)
    for r in runs:
        per[f"{r['x1']}-{r['x2']}"].append(r["final"])
    return {"n": len(dev), "mean_abs_dev_from_midpoint": float(dev.mean()), "ci95": [lo, hi],
            "averaging_falsified": bool(dev.mean() > 0.10 and lo > 0),
            "final_by_opening": {k: float(np.mean(v)) for k, v in per.items()}}


# ---------------------------------------------------------------- drift map
def load_map(d: Path) -> list[dict]:
    S = [s for s in read_jsonl(d / "samples.jsonl") if s["status"] in ("ok", "ok_after_retry")]
    return S


def fit_diag(samples: list[dict]) -> dict:
    by = defaultdict(list)
    for s in samples:
        by[s["own"]].append(s["price"])
    xs = np.array(sorted(by))
    means = np.array([np.mean(by[x]) for x in xs])
    resid = np.concatenate([np.array(by[x]) - np.mean(by[x]) for x in xs])
    return {"levels": xs.tolist(), "mean": means.tolist(), "drift": (means - xs).tolist(),
            "n": [len(by[x]) for x in xs], "resid_sd": float(resid.std(ddof=1)), "_resid": resid}


def fit_grid(samples: list[dict], rng) -> dict:
    X = np.array([[1.0, s["own"], s["rival"]] for s in samples])
    y = np.array([s["price"] for s in samples])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    boot = []
    for _ in range(2000):
        ii = rng.integers(0, len(y), len(y))
        boot.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
    boot = np.array(boot)
    q = lambda j: [float(np.quantile(boot[:, j], .025)), float(np.quantile(boot[:, j], .975))]  # noqa: E731
    return {"c": float(coef[0]), "a_own": float(coef[1]), "m_rival": float(coef[2]), "a_plus_m": float(coef[1] + coef[2]),
            "ci95": {"a_own": q(1), "m_rival": q(2), "a_plus_m": [float(np.quantile(boot[:, 1] + boot[:, 2], .025)),
                                                                  float(np.quantile(boot[:, 1] + boot[:, 2], .975))]},
            "n": len(y)}


def iterate_map(diag: dict, openings=LEVELS_PRIMARY, steps: int = 100, paths: int = 2000, rng=None) -> dict:
    xs, ms, res = np.array(diag["levels"]), np.array(diag["mean"]), diag["_resid"]
    out = {}
    for x0 in openings:
        x = np.full(paths, float(x0))
        keep = []
        for t in range(1, steps + 1):
            x = np.clip(np.interp(x, xs, ms) + rng.choice(res, paths), 1.0, 2.5)
            if t >= LIVE_LO:
                keep.append(x.copy())
        f = np.mean(keep, axis=0)
        out[str(x0)] = {"mean": float(f.mean()), "band90": [float(np.quantile(f, .05)), float(np.quantile(f, .95))]}
    return out


# ---------------------------------------------------------------- figures
def figures(res: dict, rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pN, pM = B["pN"], B["pM"]
    S = [r for r in rows if r["arm"] == "S"]
    stages = sorted({r["stage"] for r in S})
    if S:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        lim = [1.2, 2.2]
        ax.plot(lim, lim, color="#999", lw=1, ls="--", label="45°: opening persists")
        ax.axhline(pN, color="#c0392b", lw=0.8, ls=":")
        ax.axhline(pM, color="#2c7fb8", lw=0.8, ls=":")
        ax.text(lim[1], pN, " Nash", va="center", fontsize=8, color="#c0392b")
        ax.text(lim[1], pM, " monopoly", va="center", fontsize=8, color="#2c7fb8")
        for st in stages:
            g = [r for r in S if r["stage"] == st]
            ctl = "ctl" in st
            lv = sorted({r["x1"] for r in g})
            mu = [np.mean([r["final"] for r in g if r["x1"] == v]) for v in lv]
            ax.plot(lv, mu, marker="o" if not ctl else "x", lw=2.2 if not ctl else 1, ms=5,
                    color=None if not ctl else "#bbbbbb", label=st)
            ax.scatter([r["x1"] for r in g], [r["final"] for r in g], s=8, alpha=0.35, color="#666" if ctl else None)
        ax.set_xlabel("injected opening x (window mean)")
        ax.set_ylabel("final price (mean of both firms, live rounds 61-100)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "fig1_final_vs_opening.png", dpi=200)
        plt.close(fig)
    if res.get("maps"):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.axhline(0, color="#999", lw=1)
        for name, mp in res["maps"].items():
            if "diag" in mp:
                ax.plot(mp["diag"]["levels"], mp["diag"]["drift"], marker="o", ms=3, label=name)
        ax.axvline(pN, color="#c0392b", lw=0.8, ls=":")
        ax.set_xlabel("window level x")
        ax.set_ylabel("one-step drift d(x) = E[p | x] - x")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "fig2_drift_field.png", dpi=200)
        plt.close(fig)
    A = [r for r in rows if r["arm"] == "A"]
    if A:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for st in sorted({r["stage"] for r in A}):
            g = [r for r in A if r["stage"] == st]
            ax.scatter([(r["x1"] + r["x2"]) / 2 for r in g], [r["final"] for r in g], s=14, label=st)
        ax.plot([1.5, 2.0], [1.5, 2.0], color="#999", ls="--", lw=1, label="midpoint")
        ax.set_xlabel("midpoint of the opening (x1 + x2) / 2")
        ax.set_ylabel("final price")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "fig3_asymmetric_vs_midpoint.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------- main
def analyse(runs_dir: Path, maps_dir: Path | None, out: Path, pred_path: Path | None = None) -> dict:
    rng = np.random.default_rng(SEED)
    rows, excluded = load_runs(runs_dir)
    res: dict = {"excluded": excluded, "groups": {}, "maps": {}}
    pred = json.loads(pred_path.read_text()) if pred_path and pred_path.exists() else {}
    for st in sorted({r["stage"] for r in rows}):
        g = [r for r in rows if r["stage"] == st]
        S = [r for r in g if r["arm"] == "S"]
        A = [r for r in g if r["arm"] == "A"]
        e = {"model": g[0]["model"], "n_runs": len(g)}
        if S and len({r["x1"] for r in S}) > 1:
            e["slope"] = slope_test(S, rng)
            e["levels"] = level_table(S, rng)
            e["secondary"] = secondary(e["levels"], pred.get(st))
        if A:
            e["midpoint"] = midpoint_test(A, rng)
        res["groups"][st] = e
    if maps_dir:
        for d in sorted(maps_dir.glob("*_onestep_*")):
            if not (d / "samples.jsonl").exists():
                continue
            name = d.name
            mp = res["maps"].setdefault(name.split("_onestep_")[0] + "|" + name.split("_onestep_")[1].split("_", 1)[1], {})
            S = load_map(d)
            if "_onestep_diag_" in name:
                fd = fit_diag(S)
                mp["diag"] = {k: v for k, v in fd.items() if k != "_resid"}
                mp["prediction"] = iterate_map(fd, rng=rng)
            elif "_onestep_grid_" in name:
                mp["grid"] = fit_grid(S, rng)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s1.json").write_text(json.dumps(res, indent=1))
    (out / "s1_runs.json").write_text(json.dumps(rows, indent=1))
    figures(res, rows, out)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs_cluster"))
    ap.add_argument("--maps-dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "s1"))
    ap.add_argument("--predictions", default=str(ROOT / "results" / "s1" / "predictions.json"))
    a = ap.parse_args(argv)
    res = analyse(Path(a.runs_dir), Path(a.maps_dir) if a.maps_dir else None, Path(a.out), Path(a.predictions))
    for st, e in res["groups"].items():
        if "slope" in e:
            s = e["slope"]
            print(f"{st:<28} n={s['n']:<3} beta {s['beta']:+.3f}  one-sided 95% [{s['lower95_one_sided']:+.3f}, "
                  f"{s['upper95_one_sided']:+.3f}]  p_perm {s['p_perm_beta0']:.4f}  -> {s['rule']}")
            for lv in e["levels"]:
                print(f"    x={lv['x']:.3f} n={lv['n']} final {lv['final_mean']:.3f} delta {lv['delta_mean']:+.3f}")
        if "midpoint" in e:
            m = e["midpoint"]
            print(f"{st:<28} midpoint |dev| {m['mean_abs_dev_from_midpoint']:.3f} {m['ci95']}  falsified={m['averaging_falsified']}")
    for k, mp in res["maps"].items():
        if "prediction" in mp:
            print(k, {x: round(v["mean"], 3) for x, v in mp["prediction"].items()})
        if "grid" in mp:
            g = mp["grid"]
            print(k, f"a {g['a_own']:.3f} m {g['m_rival']:.3f} a+m {g['a_plus_m']:.3f} {g['ci95']['a_plus_m']}")
    if res["excluded"]:
        print("excluded:", res["excluded"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
