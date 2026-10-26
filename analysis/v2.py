"""Plan v2 analysis (frozen at tag prereg-s1 before the confirmation). Every number it prints goes to numbers.json.

  .venv/bin/python analysis/v2.py --phase disc --runs-dir runs_cluster [--out results/v2/disc]

Groups are run-config stages: v2O-<m> (arm O), v2Q-<m> (Q), v2A-<m> (A), v2T-<m> (T); filtered by --phase.
Only complete runs with validation ok are used; every exclusion is counted. Outcome of a run ("late price") =
mean of BOTH firms' prices over rounds 121-170 (live rounds 101-150 after the 20-row opening).

H1 (per model): beta_band = OLS slope of the late price on the NOMINAL opening x over the band {1.55, 1.75, 1.95}.
  CI: bootstrap of runs within level (10,000); seeds differ per level by construction (make_v2_configs.seeds_for).
  One-sided p for beta > 0.5 = share of bootstrap draws <= 0.5 (and the mirror for beta < 0.5). Holm over models.
  inherited:     Holm-adjusted p(beta > 0.5) < 0.05
  not inherited: Holm-adjusted p(beta < 0.5) < 0.05
  else inconclusive. Also: x-label permutation p for beta = 0, and the slope on the REALISED last-5-row level.
Secondary: global slope over all levels (target-seeking only if upper 95% < 0.2); edge drift = mean(late - x) at
  1.30 and 2.10; delta(1.95) - delta(1.55); carry-over = slope of the mean price over rounds 41-60 on x;
  diagnostics = read rate (round-21 outputs that mention a number within +-0.03 of x), clip rate, floor rate (1.0).
H2 (Qwens, arm T): ICC(1) of the late price across the 20 transplanted openings (3 seeds each), with a bootstrap
  over openings; the round-20 spread of the openings is reported (H2 is secondary if it is below 0.10).
H3 (arm A): d = |late - midpoint| - |late - min(x1, x2)| > 0 ("lowest opening dominates"); one-sided sign-flip
  test over runs. Order effect = |late(model holds low) - late(other holds low)| from the cross-pair stages.
Q: rho_r = OLS slope of p1 (last 50 rounds) on the fixed rival price q; scripted BR reference = 0.42.
Spread: opening-induced spread of delta within a model (delta at 2.10 minus delta at 1.30) vs the between-model
  spread of cold-start delta (Stage 1 L runs, max - min across models), as in Lee 2610.11256.
"""
from __future__ import annotations

import argparse
import json
import re
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
BAND = (1.55, 1.75, 1.95)
SEED = 20261023
N_BOOT = N_PERM = 10_000
NUM = re.compile(r"(?<![\d.])(\d\.\d+)")
COLD_START = {"Claude Sonnet 5.5": "runs/s1-sonnet55_L_E1_claude-sonnet-5-5_s*",
              "Qwen3-14B off": "runs_cluster/s1-qwen3_14b_off_L_E1_Qwen3-14B-AWQ_s*",
              "Qwen2.5-14B": "runs_cluster/s1-qwen25_14b_L_E1_Qwen25-14B-Instruct-AWQ_s*",
              "gpt-oss-20b": "runs_cluster/s1-gptoss20b_L_E1_gpt-oss-20b_s*",
              "Claude Haiku 5.5": "runs/s1-haiku55_L_E1_claude-haiku-5-5_s*",
              "gpt-6-luna": "runs/s1-gpt6luna_L_E1_gpt-6-luna_s*", "gpt-6.1-sol": "runs/s1-gpt61sol_L_E1_gpt-61-sol_s*",
              "R1-Distill-32B": "runs_cluster/s1-r1awq_L_E1_deepseek-r1-distill-qwen_s*",
              "OLMo-2-13B": "runs_cluster/s1-olmo2_13b_L_E1_OLMo-2-1124-13B-Instruct_s*"}


def ok(d: Path) -> bool:
    s, v = d / "summary.json", d / "validation.json"
    return s.exists() and json.loads(s.read_text()).get("status") == "complete" and v.exists() and json.loads(v.read_text()).get("ok")


def load(runs_dir: Path, phase: str) -> tuple[list[dict], dict]:
    rows, excl = [], defaultdict(int)
    for d in sorted(runs_dir.glob("v2*_*")):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        man = json.loads(mf.read_text())
        c = man["config"]
        if man.get("kind") == "onestep" or c.get("phase") != phase:  # one-step maps are analysed separately (M)
            continue
        if not ok(d):
            excl[f"{c['stage']}|not complete+valid"] += 1
            continue
        R = read_jsonl(d / "rounds.jsonl")
        by = {r["t"]: r for r in R}
        live = [r for r in R if not r.get("is_prefix")]
        n_pre = sum(r.get("is_prefix", False) for r in R)
        rec = {"stage": c["stage"], "arm": c["arm"], "run_id": man["run_id"], "seed": man["seed"],
               "model1": c["firm1"]["model"], "model2": c.get("firm2", {}).get("model"),
               "clip_rate": float(np.mean([r.get("clipped1", False) or r.get("clipped2", False) for r in live])),
               "floor_rate": float(np.mean([min(r["p1"], r["p2"]) <= 1.0 + 1e-9 for r in live]))}
        if c["arm"] == "Q":
            rec.update({"q": c["fixed_price"], "p1_last50": float(np.mean([r["p1"] for r in R[-50:]]))})
        else:
            late = [by[t] for t in range(121, 171) if t in by]
            if len(late) != 50:
                excl[f"{c['stage']}|short"] += 1
                continue
            rec["late"] = float(np.mean([(r["p1"] + r["p2"]) / 2 for r in late]))
            rec["late1"] = float(np.mean([r["p1"] for r in late]))
            rec["late2"] = float(np.mean([r["p2"] for r in late]))
            rec["delta"] = float(collusion_index(np.mean([(r["pi1"] + r["pi2"]) / 2 for r in late]), B["piN"], B["piM"]))
            rec["carry_41_60"] = float(np.mean([(by[t]["p1"] + by[t]["p2"]) / 2 for t in range(41, 61)]))
            rec["round20_level"] = float((by[n_pre]["p1"] + by[n_pre]["p2"]) / 2) if n_pre else None
            meta = man["meta"]
            rec["realised_last5"] = float(np.mean(meta.get("realised_last5", [np.nan])))
            rec["window_id"] = meta.get("window_id")
            if "opening" in c:
                rec["x1"], rec["x2"] = c["opening"]
                calls = [x for x in read_jsonl(d / "calls.jsonl") if x["round"] == n_pre + 1]
                hits = [any(abs(float(v) - rec["x1"]) <= 0.03 for v in NUM.findall(x.get("raw_output") or "")) for x in calls]
                rec["read"] = float(np.mean(hits)) if hits else None
            if c["arm"] == "T":
                rec["opening_id"] = meta["transplant_source_run"]
        rows.append(rec)
    return rows, dict(excl)


def slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def boot_slope_within(x: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    lv = sorted(set(x))
    idx = [np.where(x == v)[0] for v in lv]
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx])
        out[b] = slope(x[ii], y[ii])
    return out


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def h1(rows: list[dict], rng) -> dict:
    res, p_gt, p_lt = {}, {}, {}
    for st in sorted({r["stage"] for r in rows if r["arm"] == "O"}):
        g = [r for r in rows if r["stage"] == st and r["arm"] == "O"]
        band = [r for r in g if r["x1"] in BAND]
        if len({r["x1"] for r in band}) < 2:
            continue
        x, y = np.array([r["x1"] for r in band]), np.array([r["late"] for r in band])
        bb = boot_slope_within(x, y, rng)
        beta = slope(x, y)
        perm = np.array([slope(rng.permutation(x), y) for _ in range(N_PERM)])
        e = {"model": g[0]["model1"], "n_band": len(band), "beta_band": beta,
             "ci90": [float(np.quantile(bb, .05)), float(np.quantile(bb, .95))],
             "ci95": [float(np.quantile(bb, .025)), float(np.quantile(bb, .975))],
             "p_beta_gt_0.5": float((bb <= 0.5).mean()), "p_beta_lt_0.5": float((bb >= 0.5).mean()),
             "p_perm_beta0": float(((np.abs(perm) >= abs(beta) - 1e-12).sum() + 1) / (N_PERM + 1)),
             "beta_band_realised": slope([r["realised_last5"] for r in band], y)}
        xa, ya = np.array([r["x1"] for r in g]), np.array([r["late"] for r in g])
        bg = boot_slope_within(xa, ya, rng)
        e["global_slope"] = slope(xa, ya)
        e["global_ci95"] = [float(np.quantile(bg, .025)), float(np.quantile(bg, .975))]
        e["global_target_seeking"] = bool(np.quantile(bg, .975) < 0.2)
        lvl = defaultdict(list)
        for r in g:
            lvl[r["x1"]].append(r)
        e["levels"] = {f"{k:.3f}": {"n": len(v), "late": float(np.mean([r["late"] for r in v])),
                                    "delta": float(np.mean([r["delta"] for r in v])),
                                    "drift": float(np.mean([r["late"] - k for r in v])),
                                    "read": float(np.nanmean([r["read"] for r in v if r["read"] is not None])) if any(r["read"] is not None for r in v) else None,
                                    "clip": float(np.mean([r["clip_rate"] for r in v])), "floor": float(np.mean([r["floor_rate"] for r in v]))}
                       for k, v in sorted(lvl.items())}
        L = e["levels"]
        if "1.950" in L and "1.550" in L:
            e["delta_1.95_minus_1.55"] = L["1.950"]["delta"] - L["1.550"]["delta"]
        if "2.100" in L and "1.300" in L:
            e["opening_spread_delta"] = L["2.100"]["delta"] - L["1.300"]["delta"]
        e["carry_slope_41_60"] = slope([r["x1"] for r in band], [r["carry_41_60"] for r in band])
        res[st] = e
        p_gt[st], p_lt[st] = e["p_beta_gt_0.5"], e["p_beta_lt_0.5"]
    hg, hl = holm(p_gt), holm(p_lt)
    for st, e in res.items():
        e["holm_p_gt"], e["holm_p_lt"] = hg[st], hl[st]
        e["H1"] = "inherited" if hg[st] < 0.05 else "not inherited" if hl[st] < 0.05 else "inconclusive"
    return res


def h2(rows: list[dict], rng) -> dict:
    out = {}
    for st in sorted({r["stage"] for r in rows if r["arm"] == "T"}):
        g = [r for r in rows if r["stage"] == st]
        by = defaultdict(list)
        for r in g:
            by[r["opening_id"]].append(r["late"])
        grp = [np.array(v) for v in by.values() if len(v) >= 2]
        if len(grp) < 3:
            continue

        def icc(gs):
            k = np.mean([len(x) for x in gs])
            ms_b = np.var([x.mean() for x in gs], ddof=1) * k
            ms_w = np.mean([x.var(ddof=1) for x in gs])
            return float((ms_b - ms_w) / (ms_b + (k - 1) * ms_w))
        bs = [icc([grp[i] for i in rng.integers(0, len(grp), len(grp))]) for _ in range(2000)]
        r20 = [r["round20_level"] for r in g]
        out[st] = {"n_openings": len(grp), "icc": icc(grp), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                   "round20_spread_sd": float(np.std(r20, ddof=1)), "secondary_if_spread_below": 0.10}
    return out


def h3(rows: list[dict], rng) -> dict:
    out = {}
    for st in sorted({r["stage"] for r in rows if r["arm"] == "A"}):
        g = [r for r in rows if r["stage"] == st]
        d = np.array([abs(r["late"] - (r["x1"] + r["x2"]) / 2) - abs(r["late"] - min(r["x1"], r["x2"])) for r in g])
        flips = rng.choice([-1, 1], (N_PERM, len(d)))
        p = float((((flips * d).mean(1) >= d.mean() - 1e-12).sum() + 1) / (N_PERM + 1))
        out[st] = {"n": len(d), "mean_d": float(d.mean()), "p_one_sided": p,
                   "mean_abs_dev_min": float(np.mean([abs(r["late"] - min(r["x1"], r["x2"])) for r in g])),
                   "mean_abs_dev_mid": float(np.mean([abs(r["late"] - (r["x1"] + r["x2"]) / 2) for r in g])),
                   "by_opening": {f"{a}-{b}": float(np.mean([r["late"] for r in g if (r["x1"], r["x2"]) == (a, b)]))
                                  for a, b in sorted({(r["x1"], r["x2"]) for r in g})}}
    return out


def q_ladder(rows: list[dict]) -> dict:
    out = {}
    for st in sorted({r["stage"] for r in rows if r["arm"] == "Q"}):
        g = [r for r in rows if r["stage"] == st]
        out[st] = {"n": len(g), "rho_r": slope([r["q"] for r in g], [r["p1_last50"] for r in g]),
                   "by_q": {f"{q:.3f}": float(np.mean([r["p1_last50"] for r in g if r["q"] == q])) for q in sorted({r["q"] for r in g})},
                   "br_reference": slope([1.473, 1.925], [M.best_response(1.473), M.best_response(1.925)])}
    return out


def cold_start_spread() -> dict:
    d = {}
    for name, pat in COLD_START.items():
        base, glob = pat.split("/", 1)
        v = [json.loads((x / "summary.json").read_text())["delta1"] for x in sorted((ROOT / base).glob(glob)) if ok(x)]
        if v:
            d[name] = float(np.mean(v))
    return {"per_model": d, "between_model_range": float(max(d.values()) - min(d.values())) if d else None,
            "lee_reference_claude_minus_deepseek": 0.79 - 0.24}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["disc", "conf"], required=True)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs_cluster"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(SEED)
    rows, excl = load(Path(a.runs_dir), a.phase)
    res = {"phase": a.phase, "excluded": excl, "n_runs": len(rows), "H1": h1(rows, rng), "H2": h2(rows, rng),
           "H3": h3(rows, rng), "Q": q_ladder(rows), "cold_start": cold_start_spread()}
    out = Path(a.out or ROOT / "results" / "v2" / a.phase)
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    (out / "runs.json").write_text(json.dumps(rows, indent=1))
    print(f"phase {a.phase}: {len(rows)} runs used; excluded {excl}")
    for st, e in res["H1"].items():
        print(f"H1 {st:<16} beta_band {e['beta_band']:+.3f} 90% [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}] "
              f"Holm p>0.5 {e['holm_p_gt']:.4f} p<0.5 {e['holm_p_lt']:.4f} -> {e['H1']}; global {e['global_slope']:+.3f}; "
              f"realised {e['beta_band_realised']:+.3f}")
        for k, v in e["levels"].items():
            print(f"      x={k} n={v['n']:<3} late {v['late']:.3f} drift {v['drift']:+.3f} delta {v['delta']:+.3f} "
                  f"read {v['read'] if v['read'] is None else round(v['read'], 2)} clip {v['clip']:.2f} floor {v['floor']:.2f}")
    for st, e in res["H2"].items():
        print(f"H2 {st:<16} ICC {e['icc']:.3f} {e['ci95']} openings {e['n_openings']} round-20 spread {e['round20_spread_sd']:.3f}")
    for st, e in res["H3"].items():
        print(f"H3 {st:<16} n={e['n']} d {e['mean_d']:+.3f} p {e['p_one_sided']:.4f} |dev| min {e['mean_abs_dev_min']:.3f} mid {e['mean_abs_dev_mid']:.3f}")
    for st, e in res["Q"].items():
        print(f"Q  {st:<16} rho_r {e['rho_r']:+.3f} (BR ref {e['br_reference']:.3f}) {e['by_q']}")
    print("cold-start between-model range", res["cold_start"]["between_model_range"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
