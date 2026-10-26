"""Reads runs/ only (never the runner's memory) and writes results/: tables, tests, figures.

  python analysis/analyze.py                 # everything found under runs/
  python analysis/analyze.py --stage stage3  # one stage

Only completed runs (summary.json present) are analysed; incomplete ones are listed, never silently used.
All resampling uses a fixed RNG seed, so results/ is reproducible from runs/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from collusion.market import Market  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402
from collusion.validate import validate_run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
N_BOOT = N_PERM = 10_000
RNG_SEED = 12345


def _safe_name(k: str) -> str:
    """Filename-safe on every OS (no ':', '{', '|', ...)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", k).strip("_")


def load_all(runs_dir: Path, stage: str | None) -> tuple[list[dict], list[str]]:
    runs, incomplete = [], []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (d / "manifest.json").exists():
            continue
        man = json.loads((d / "manifest.json").read_text())
        if stage and man["config"]["stage"] != stage:
            continue
        if not (d / "summary.json").exists():
            incomplete.append(d.name)
            continue
        summ = json.loads((d / "summary.json").read_text())
        errs = validate_run(d)  # re-validate NOW from the raw logs; never trust a stored validation.json
        if summ.get("status") != "complete" or errs:
            incomplete.append(f"{d.name} (status={summ.get('status')}, validation errors={len(errs)})")
            continue
        runs.append({"id": d.name, "manifest": man, "summary": json.loads((d / "summary.json").read_text()),
                     "rounds": read_jsonl(d / "rounds.jsonl")})
    return runs, incomplete


def design_key(c: dict) -> str:
    """Everything that defines a design cell EXCEPT the arm. Runs are only compared within one design key."""
    samp = c["sampling"]
    ident = c.get("identity_label") if (c.get("identity1") or c.get("identity2")) else "none"
    key = (f"{c['stage']}|{c['prompt']}|{c['firm1']['model'].split('/')[-1]}|id:{ident}"
           f"|T{samp['temperature']}|p{samp['top_p']}|max{samp['max_tokens']}|mkt:{json.dumps(c.get('market', {}), sort_keys=True)}")
    sv = c.get("supervision")
    if sv:  # supervision arms (v1 note, v2 review / yoked) never pool with each other
        key += f"|sup:{sv['frame']}:{sv.get('mode', 'note')}:{'A' if sv['agent'] else 'N'}"
    f2 = c.get("firm2", {}).get("model")
    if f2 and f2 != c["firm1"]["model"]:  # cross-model pairs that share a firm-1 model must never pool
        key += f"|f2:{f2.split('/')[-1]}"
    if "opening" in c:  # injected openings: each level and source pool is its own design cell
        key += f"|open:{c['opening'][0]}-{c['opening'][1]}|src:{c['opening_source']['template']}"
    elif "opening_source" in c:  # arm T (transplant): the source pool defines the cell
        key += f"|src:{c['opening_source']['template']}"
    if "phase" in c:  # discovery and confirmation never pool (plan v2 M1)
        key += f"|ph:{c['phase']}"
    if "fixed_price" in c:
        key += f"|q:{c['fixed_price']}"
    if c.get("view"):
        key += f"|view:{json.dumps(c['view'], sort_keys=True)}"
    for k in ("step", "impulse"):  # mechanism arms: each shock size is its own cell
        if k in c:
            key += f"|{k}:{json.dumps(c[k], sort_keys=True)}"
    return key


def group_key(r: dict) -> str:
    c = r["manifest"]["config"]
    return f"{c['arm']}|{design_key(c)}"


def ms(x) -> str:
    x = np.asarray([v for v in x if v is not None], float)
    return "n/a" if len(x) == 0 else f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


def per_arm_table(groups: dict[str, list[dict]]) -> str:
    cols = ["group", "n", "Δ firm1", "Δ avg", "price1", "profit1/πN", "parse fail (calls)", "fallback", "clipped/run",
            "truncated", "|p1−BR|"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for k, rs in sorted(groups.items()):
        S = [r["summary"] for r in rs]
        def g(f, S=S):
            return [s.get(f) for s in S]
        lines.append("| " + " | ".join([k, str(len(rs)), ms(g("delta1")), ms(g("delta_avg")), ms(g("mean_p1")),
                     ms(g("profit_ratio1")), ms(g("parse_fail_rate_calls")), ms(g("fallback_rate_decisions")),
                     ms(g("clipped")), ms(g("truncation_rate")), ms(g("mean_abs_gap_br1"))]) + " |")
    return "\n".join(lines)


def interaction(dL: np.ndarray, dP: np.ndarray, rng: np.random.Generator) -> dict:
    """I = mean Δ_L − mean Δ_P; bootstrap over runs within arm; exact-design permutation test of labels."""
    I = dL.mean() - dP.mean()
    boot = np.array([rng.choice(dL, len(dL)).mean() - rng.choice(dP, len(dP)).mean() for _ in range(N_BOOT)])
    pooled, nL = np.concatenate([dL, dP]), len(dL)
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        x = rng.permutation(pooled)
        perm[i] = x[:nL].mean() - x[nL:].mean()
    b = int((np.abs(perm) >= abs(I) - 1e-15).sum())
    return {"I": float(I), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "p_two_sided": (b + 1) / (N_PERM + 1), "n_L": nL, "n_P": len(dP),
            "ratio_P_over_L": float(dP.mean() / dL.mean()) if dL.mean() else None}


def punishment(D: list[dict], Df: list[dict], L_by_seed: dict, m: Market, rng: np.random.Generator, out: Path) -> dict:
    res = {"runs": []}
    traj = defaultdict(list)
    for r in D:
        rd = {x["t"]: x for x in r["rounds"]}
        k = r["manifest"]["config"]["shock_round"]
        pre = rd[k - 1]["p2"]
        p1k = rd[k]["p1"]
        row = {"run": r["id"],
               "depth_spec": rd[k]["p2"] - m.best_response(p1k),           # spec: p2(201) - BR(p1(201))
               "depth_first_reaction": rd[k + 1]["p2"] - m.best_response(p1k) if k + 1 in rd else None}
        later = [rd[t]["p2"] for t in sorted(rd) if t > k]
        below = [i for i, t in enumerate(sorted(t for t in rd if t > k)) if rd[t]["p2"] < rd[t]["br2"] - 1e-9]
        row["went_below_br"] = bool(below)
        row["recovered"] = bool(below) and any(abs(p - pre) <= 0.02 for p in later[below[0] + 1:])
        res["runs"].append(row)
        for tau in range(0, 11):
            if k + tau in rd:
                traj["D"].append((tau, rd[k + tau]["p2"] - pre))
        Lr = L_by_seed.get(r["manifest"]["seed"])
        if Lr:  # natural no-shock counterfactual: the same L run's own continuation
            lr = {x["t"]: x for x in Lr["rounds"]}
            for tau in range(0, 11):
                if k + tau in lr:
                    traj["L continuation (no shock)"].append((tau, lr[k + tau]["p2"] - lr[k - 1]["p2"]))
    for r in Df:
        rd = {x["t"]: x for x in r["rounds"]}
        k = r["manifest"]["config"]["shock_round"]
        for tau in range(0, 11):
            if k + tau in rd:
                traj["D-frozen"].append((tau, rd[k + tau]["p2"] - rd[k - 1]["p2"]))
    for key in ("depth_spec", "depth_first_reaction"):
        v = np.array([x[key] for x in res["runs"] if x[key] is not None])
        if len(v):
            boot = [rng.choice(v, len(v)).mean() for _ in range(N_BOOT)]
            res[key] = {"mean": float(v.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], "n": len(v)}
    n = len(res["runs"])
    res["frac_below_br_then_recover"] = sum(x["recovered"] for x in res["runs"]) / n if n else None
    _impulse_plot(traj, rng, out)
    return res


def _impulse_plot(traj: dict, rng, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    styles = {"D": ("#c0392b", 2.2), "D-frozen": ("#888888", 1.4), "L continuation (no shock)": ("#bbbbbb", 1.4)}
    for name, pts in traj.items():
        taus = sorted({t for t, _ in pts})
        mean, lo, hi = [], [], []
        for t in taus:
            v = np.array([y for tt, y in pts if tt == t])
            b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            mean.append(v.mean()); lo.append(np.quantile(b, .025)); hi.append(np.quantile(b, .975))
        c, lw = styles.get(name, ("k", 1))
        ax.plot(taus, mean, color=c, lw=lw, label=name, marker="o", ms=3)
        ax.fill_between(taus, lo, hi, color=c, alpha=0.12, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("rounds after firm 1's forced price cut (τ)")
    ax.set_ylabel("firm 2 price − its pre-shock price")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "impulse_response.png", dpi=200); plt.close(fig)


def trajectories(groups: dict, out: Path, b: dict) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, rs in groups.items():
        fig, axes = plt.subplots(1, min(5, len(rs)), figsize=(3 * min(5, len(rs)), 2.6), sharey=True, squeeze=False)
        for ax, r in zip(axes[0], rs[:5], strict=True):
            t = [x["t"] for x in r["rounds"]]
            ax.plot(t, [x["p1"] for x in r["rounds"]], lw=0.8, color="#c0392b", label="firm 1")
            ax.plot(t, [x["p2"] for x in r["rounds"]], lw=0.8, color="#777777", label="firm 2")
            ax.axhline(b["pN"], ls=":", color="k", lw=0.6); ax.axhline(b["pM"], ls="--", color="k", lw=0.6)
            ax.set_title(f"seed {r['manifest']['seed']}", fontsize=8)
        axes[0][0].set_ylabel("price (dotted Nash, dashed monopoly)", fontsize=7)
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.suptitle(k, fontsize=8); fig.tight_layout()
        fig.savefig(out / f"traj_{_safe_name(k)}.png", dpi=150); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--stage")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    runs, incomplete = load_all(Path(a.runs_dir), a.stage)
    m = Market(); b = m.benchmarks()
    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    report = {"n_runs": len(runs), "incomplete_runs": incomplete, "benchmarks": b, "groups": {k: len(v) for k, v in groups.items()}}
    table = per_arm_table(groups)
    by_arm = defaultdict(list)
    for r in runs:
        c = r["manifest"]["config"]
        by_arm[(design_key(c), c["arm"])].append(r)
    for (dk, arm), rs in by_arm.items():
        if arm == "L" and (dk, "P") in by_arm:
            dL = np.array([r["summary"]["delta1"] for r in rs])
            dP = np.array([r["summary"]["delta1"] for r in by_arm[(dk, "P")]])
            report[f"interaction|{dk}"] = interaction(dL, dP, rng)
        if arm == "D":
            Df = by_arm.get((dk, "Dfrozen"), [])
            Ls = {r["manifest"]["seed"]: r for r in by_arm.get((dk, "L"), [])}
            report[f"punishment|{dk}"] = punishment(rs, Df, Ls, m, rng, out)
    trajectories(groups, out, b)
    (out / "table.md").write_text(table + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(table)
    print(json.dumps({k: v for k, v in report.items() if k != "benchmarks"}, indent=1)[:3000])
    if incomplete:
        print(f"WARNING: {len(incomplete)} incomplete runs excluded: {incomplete[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
