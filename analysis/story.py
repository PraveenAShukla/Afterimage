"""Strategist vs follower analyses (task list A1-A5), from existing logs only.

  .venv/bin/python analysis/story.py            # writes results/story.json and prints the tables

A1  follower score, P runs, firm-1 decisions t = 21..300:
      d_match = |p1(t) - p2(t-1)|,  d_BR = |p1(t) - BR(p2(t-1))|;  mean of each, share(d_match < d_BR).
    Also reported on "discriminating" rounds only, where |p2(t-1) - BR(p2(t-1))| > 0.05 (near Nash, matching and
    best-responding coincide). No regression on p2 and BR(p2) (nearly collinear).
A2  FM runs (rival fixed at p^M = 1.925): firm-1 mean price over the last 50 rounds vs BR(1.925); FN gap beside it.
A3  paired L/P tests: P run i replays firm 2 of L run i (manifest replay_map), so pairs are dependent.
      d_i = metric(L_i) - metric(P_i) for profit I (summary delta1, the spec) and price I (summary mean_p1, P9).
      Exact sign-flip test over all 2^n flips if n <= 20 (p = share of flips with |mean| >= |observed|, identity
      included), else 10,000 random flips with p = (b+1)/(m+1). Paired bootstrap 95% CI (10,000, resampling pairs).
A4  firm 1 after its own forced cut: mean p1 over t = 202..210 in D vs D-frozen (rival held at its r200 price),
    paired by seed; D - D-frozen with paired bootstrap CI.
A5  deviation profitability (descriptive): per seed, sum_{t=201..210} pi1(D) - pi1(Dplacebo). A static
    best-responder also makes the deviation unprofitable, so this does not by itself show punishment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from collusion.market import Market  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402
from punishment_paired import analyse as punish_paired  # noqa: E402

M = Market()
RNG_SEED = 20261016
N_BOOT = 10_000
# (label, runs dir, stage prefix, model short name in run ids)
LP_GROUPS = [
    ("Claude Sonnet 5.5", "runs", "s1-sonnet55", "claude-sonnet-5-5"),
    ("Claude Sonnet 5.5 (confirmation)", "runs", "s1-sonnet55_confirm", "claude-sonnet-5-5"),
    ("gpt-6.1-sol", "runs", "s1-gpt61sol", "gpt-61-sol"),
    ("Qwen2.5-14B", "runs_cluster", "s3-qwen25_14b", "Qwen25-14B-Instruct-AWQ"),
    ("Qwen3-14B off", "runs_cluster", "s3-qwen3_14b_off", "Qwen3-14B-AWQ"),
    ("R1-Distill-32B", "runs_cluster", "s3-r1awq", "deepseek-r1-distill-qwen"),
    ("Qwen3.5-35B-A3B", "runs_cluster", "s3-qwen35_a3b_4k", "Qwen35-35B-A3B-GPTQ-Int4"),
]


def _br(p: float, cache: dict = {}) -> float:  # noqa: B006 - deliberate memo
    k = round(p, 6)
    if k not in cache:
        cache[k] = M.best_response(p)
    return cache[k]


def runs_of(base: str, stage: str, arm: str, short: str) -> dict[int, Path]:
    out = {}
    for d in (ROOT / base).glob(f"{stage}_{arm}_E1_{short}*_s*"):
        if (d / "summary.json").exists() and json.loads((d / "summary.json").read_text()).get("status") == "complete":
            out[int(d.name.rsplit("_s", 1)[1])] = d
    return out


def rounds(d: Path) -> dict[int, dict]:
    return {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}


def summ(d: Path) -> dict:
    return json.loads((d / "summary.json").read_text())


def boot_ci(v: np.ndarray, rng) -> list[float]:
    b = rng.choice(v, (N_BOOT, len(v))).mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def sign_flip(d: np.ndarray, rng) -> dict:
    n, obs = len(d), abs(d.mean())
    if n <= 20:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)  # all 2^n sign vectors
        means = np.abs((signs * d).mean(1))
        p, m, kind = float((means >= obs - 1e-12).mean()), len(signs), "exact"
    else:
        s = rng.choice([1, -1], (10_000, n))
        b = int((np.abs((s * d).mean(1)) >= obs - 1e-12).sum())
        p, m, kind = (b + 1) / (10_000 + 1), 10_000, "random"
    return {"p": p, "n_flips": m, "kind": kind}


def hodges_lehmann(d: np.ndarray) -> float:
    """M10: median of the Walsh averages (d_i + d_j)/2, i <= j (paired location estimate, robust to outliers)."""
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def hl_ci(d: np.ndarray, rng, n: int = 5000) -> list[float]:
    b = [hodges_lehmann(rng.choice(d, len(d))) for _ in range(n)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def pooled_perm(a: np.ndarray, b: np.ndarray, rng, n=10_000) -> float:
    obs, x, na = abs(a.mean() - b.mean()), np.concatenate([a, b]), len(a)
    c = 0
    for _ in range(n):
        y = rng.permutation(x)
        c += abs(y[:na].mean() - y[na:].mean()) >= obs - 1e-12
    return (c + 1) / (n + 1)


def a1_follower(P: dict[int, Path], lo: int = 21) -> dict:
    per_run = []
    for d in P.values():
        R = rounds(d)
        dm, db, disc = [], [], []
        for t in range(lo, max(R) + 1):
            if t not in R or t - 1 not in R or R[t].get("source1") != "live":
                continue
            q = R[t - 1]["p2"]
            br = _br(q)
            dm.append(abs(R[t]["p1"] - q))
            db.append(abs(R[t]["p1"] - br))
            disc.append(abs(q - br) > 0.05)
        dm, db, disc = map(np.array, (dm, db, disc))
        per_run.append({"d_match": dm.mean(), "d_BR": db.mean(), "share_match_closer": (dm < db).mean(),
                        "share_match_closer_disc": (dm[disc] < db[disc]).mean() if disc.any() else np.nan,
                        "frac_disc": disc.mean()})
    keys = per_run[0].keys() if per_run else []
    return {"n_runs": len(per_run), **{k: float(np.nanmean([r[k] for r in per_run])) for k in keys}}


def a3_paired(L: dict[int, Path], P: dict[int, Path], rng) -> dict:
    pairs = []
    for ps, d in sorted(P.items()):
        ls = json.loads((d / "manifest.json").read_text())["config"]["replay_map"][str(ps)]
        if ls in L:
            pairs.append((L[ls], d))
    out = {"n_pairs": len(pairs)}
    for name, key in (("profit_I", "delta1"), ("price_I", "mean_p1")):
        a = np.array([summ(x)[key] for x, _ in pairs])
        b = np.array([summ(y)[key] for _, y in pairs])
        dd = a - b
        out[name] = {"mean": float(dd.mean()), "ci95_paired": boot_ci(dd, rng), **sign_flip(dd, rng),
                     "hodges_lehmann": hodges_lehmann(dd), "hl_ci95": hl_ci(dd, rng),
                     "p_pooled_perm": pooled_perm(a, b, rng), "mean_L": float(a.mean()), "mean_P": float(b.mean())}
    return out


def a4_offpath(D: dict, Df: dict, rng) -> dict:
    v = []
    for s in sorted(set(D) & set(Df)):
        rd, rf = rounds(D[s]), rounds(Df[s])
        v.append((np.mean([rd[t]["p1"] for t in range(202, 211)]), np.mean([rf[t]["p1"] for t in range(202, 211)])))
    if not v:
        return {"n": 0}
    a = np.array(v)
    dd = a[:, 0] - a[:, 1]
    return {"n": len(v), "p1_D": float(a[:, 0].mean()), "p1_Dfrozen": float(a[:, 1].mean()),
            "diff": float(dd.mean()), "ci95_paired": boot_ci(dd, rng), **sign_flip(dd, rng)}


def a5_devprofit(D: dict, Dp: dict, rng) -> dict:
    v = []
    for s in sorted(set(D) & set(Dp)):
        rd, rp = rounds(D[s]), rounds(Dp[s])
        v.append(sum(rd[t]["pi1"] - rp[t]["pi1"] for t in range(201, 211)))
    if not v:
        return {"n": 0}
    v = np.array(v)
    return {"n": len(v), "sum_gain_201_210": float(v.mean()), "ci95": boot_ci(v, rng),
            "frac_profitable": float((v > 0).mean())}


def a2_fixed() -> list[dict]:
    rows, br_m = [], _br(1.925)
    groups = {}
    for base in ("runs", "runs_cluster"):
        for d in (ROOT / base).glob("*_FM_E1_*_s*"):
            if (d / "summary.json").exists() and summ(d).get("status") == "complete":
                groups.setdefault((base, d.name.split("_FM_E1_")[0], d.name.split("_FM_E1_")[1].rsplit("_s", 1)[0]), []).append(d)
    for (base, stage, short), ds in sorted(groups.items()):
        p_last = [np.mean([r["p1"] for t, r in R.items() if t > max(R) - 50]) for R in map(rounds, ds)]  # last 50 of each run
        fn = [summ(d)["mean_abs_gap_br1"] for d in (ROOT / base).glob(f"{stage}_FN_E1_{short}_s*")
              if (d / "summary.json").exists() and summ(d).get("status") == "complete"]
        rows.append({"stage": stage, "model": short, "n": len(ds), "fm_p1_last50": float(np.mean(p_last)),
                     "fm_p1_sd": float(np.std(p_last, ddof=1)) if len(p_last) > 1 else None,
                     "br_to_pM": br_m, "fn_gap": float(np.mean(fn)) if fn else None, "n_fn": len(fn)})
    return rows


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    out = {"lp": {}, "fm": a2_fixed()}
    for label, base, stage, short in LP_GROUPS:
        L, P = runs_of(base, stage, "L", short), runs_of(base, stage, "P", short)
        if not P:
            continue
        g = {"stage": stage, "model": short, "n_L": len(L), "n_P": len(P),
             "delta_L": float(np.mean([summ(d)["delta1"] for d in L.values()])) if L else None,
             "A1_P": a1_follower(P), "A1_L": a1_follower(L), "A3": a3_paired(L, P, rng)}
        D, Df, Dp = (runs_of(base, stage, a, short) for a in ("D", "Dfrozen", "Dplacebo"))
        g["A4"] = a4_offpath(D, Df, rng)
        g["A5"] = a5_devprofit(D, Dp, rng)
        if D and Dp:
            pp = punish_paired(ROOT / base, stage, short)
            g["punish"] = {k: pp.get(k) for k in ("n", "depth202", "effect202", "frac_below_br_202", "frac_recover_within_10")}
        out["lp"][label] = g
    (ROOT / "results" / "story.json").write_text(json.dumps(out, indent=1, default=float))

    print("A1  follower score (P runs, firm 1, t=21..300)   [L runs in brackets]")
    print(f"{'model':<34}{'n':>3}{'d_match':>9}{'d_BR':>8}{'share match<BR':>16}{'  (discrim. rounds, frac)':>26}")
    for k, g in out["lp"].items():
        a, b = g["A1_P"], g["A1_L"]
        print(f"{k:<34}{a['n_runs']:>3}{a['d_match']:>9.3f}{a['d_BR']:>8.3f}{a['share_match_closer']:>10.2f} [{b['share_match_closer']:.2f}]"
              f"{a['share_match_closer_disc']:>12.2f} ({a['frac_disc']:.2f})")
    print("\nA2  fixed rival at p^M (FM), firm-1 mean price, last 50 rounds;  BR(1.925) = %.3f" % out["fm"][0]["br_to_pM"] if out["fm"] else "")
    for r in out["fm"]:
        fn = f"{r['fn_gap']:.4f} (n={r['n_fn']})" if r["fn_gap"] is not None else "-"
        print(f"  {r['stage']:<26}{r['model']:<30} n={r['n']:<3} FM p1 {r['fm_p1_last50']:.3f}   FN gap {fn}")
    print("\nA3  paired L-P tests")
    for k, g in out["lp"].items():
        for m in ("price_I", "profit_I"):
            x = g["A3"][m]
            print(f"  {k:<34}{m:<9} n={g['A3']['n_pairs']:<3} {x['mean']:+.3f} [{x['ci95_paired'][0]:+.3f}, {x['ci95_paired'][1]:+.3f}]"
                  f"  sign-flip p={x['p']:.4f} ({x['kind']})  pooled-perm p={x['p_pooled_perm']:.4f}")
    print("\nA4  firm-1 price t=202..210 after its own cut: D vs D-frozen  |  A5 deviation gain sum pi1(D)-pi1(placebo), t=201..210")
    for k, g in out["lp"].items():
        a, b = g["A4"], g["A5"]
        if a["n"]:
            print(f"  {k:<34} n={a['n']:<3} D {a['p1_D']:.3f}  Df {a['p1_Dfrozen']:.3f}  diff {a['diff']:+.3f} "
                  f"[{a['ci95_paired'][0]:+.3f}, {a['ci95_paired'][1]:+.3f}] p={a['p']:.4f} | A5 {b['sum_gain_201_210']:+.3f} "
                  f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] profitable {b['frac_profitable']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
