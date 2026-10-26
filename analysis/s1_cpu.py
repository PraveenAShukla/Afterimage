"""S1 CPU checks on EXISTING logs from the pre-sprint S1 and discovery runs: C2 and C3. Descriptive.

  .venv/bin/python analysis/s1_cpu.py     # -> results/s1/cpu_checks.json

C2  across stored L runs per model: OLS slope of the final level (mean of both firms, rounds 251-300) on the early
    level (rounds 21-40). Observational only: the early level is itself an outcome.
C3  from P-arm logs (firm 2 replayed, so the rival path is exogenous to firm 1): the LEVEL coefficient lambda+ in
      p1(t) = c + lambda+ * z(t-1) + kappa * g(t-1),   z = (own + rival) / 2,  g = own - rival   (t = 21..300)
    lambda+ = a + m + g*s is how far a common shift of both last prices carries into the next price.
    a, m and the BR pull are NOT separately identified (rival price and BR(rival) are nearly collinear on 1.5-2.1).
    CI: bootstrap over runs (cluster bootstrap, 5,000).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collusion.runlog import read_jsonl  # noqa: E402

L_GROUPS = {"Qwen3-14B off": ("runs_cluster", "s3-qwen3_14b_off_L_E1_Qwen3-14B-AWQ_s*"),
            "Qwen2.5-14B": ("runs_cluster", "s3-qwen25_14b_L_E1_Qwen25-14B-Instruct-AWQ_s*"),
            "gpt-oss-20b": ("runs_cluster", "s1-gptoss20b_L_E1_gpt-oss-20b_s*"),
            "R1-Distill-32B": ("runs_cluster", "s1-r1awq_L_E1_deepseek-r1-distill-qwen*_s*"),
            "Claude Sonnet 5.5 (T=1)": ("runs", "s1-sonnet55_L_E1_claude-sonnet-5-5_s*")}
P_GROUPS = {"Qwen3-14B off": ("runs_cluster", "s3-qwen3_14b_off_P_E1_Qwen3-14B-AWQ_s*"),
            "Qwen2.5-14B": ("runs_cluster", "s3-qwen25_14b_P_E1_Qwen25-14B-Instruct-AWQ_s*"),
            "gpt-6.1-sol (T=1)": ("runs", "s1-gpt61sol_P_E1_gpt-61-sol_s*"),
            "Claude Sonnet 5.5 (T=1)": ("runs", "s1-sonnet55_P_E1_claude-sonnet-5-5_s*")}


def complete(d: Path) -> bool:
    s, v = d / "summary.json", d / "validation.json"
    return s.exists() and json.loads(s.read_text()).get("status") == "complete" and v.exists() and json.loads(v.read_text()).get("ok")


def c2(rng) -> dict:
    out = {}
    for name, (base, pat) in L_GROUPS.items():
        e, f = [], []
        for d in sorted((ROOT / base).glob(pat)):
            if not complete(d):
                continue
            R = {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}
            if 300 not in R:
                continue
            e.append(np.mean([(R[t]["p1"] + R[t]["p2"]) / 2 for t in range(21, 41)]))
            f.append(np.mean([(R[t]["p1"] + R[t]["p2"]) / 2 for t in range(251, 301)]))
        if len(e) < 3:
            continue
        e, f = np.array(e), np.array(f)
        sl = lambda a, b: float(np.polyfit(a, b, 1)[0])  # noqa: E731
        boot = [sl(e[i], f[i]) for i in (rng.integers(0, len(e), len(e)) for _ in range(5000)) if np.ptp(e[i]) > 0]
        out[name] = {"n": len(e), "slope_final_on_early": sl(e, f), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                     "corr": float(np.corrcoef(e, f)[0, 1]), "early_range": [float(e.min()), float(e.max())],
                     "final_range": [float(f.min()), float(f.max())]}
    return out


def c3(rng) -> dict:
    out = {}
    for name, (base, pat) in P_GROUPS.items():
        per = []
        for d in sorted((ROOT / base).glob(pat)):
            if not complete(d):
                continue
            R = {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}
            X, y = [], []
            for t in range(21, 301):
                if t in R and t - 1 in R and R[t].get("source1") == "live":
                    o, r = R[t - 1]["p1"], R[t - 1]["p2"]
                    X.append([1.0, (o + r) / 2, o - r])
                    y.append(R[t]["p1"])
            if X:
                per.append((np.array(X), np.array(y)))
        if len(per) < 3:
            continue
        fit = lambda parts: np.linalg.lstsq(np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), rcond=None)[0]  # noqa: E731
        coef = fit(per)
        boot = np.array([fit([per[i] for i in rng.integers(0, len(per), len(per))]) for _ in range(5000)])
        out[name] = {"n_runs": len(per), "n_obs": int(sum(len(p[1]) for p in per)), "lambda_plus": float(coef[1]),
                     "lambda_plus_ci95": [float(np.quantile(boot[:, 1], .025)), float(np.quantile(boot[:, 1], .975))],
                     "kappa_gap": float(coef[2]), "intercept": float(coef[0])}
    return out


def main() -> int:
    rng = np.random.default_rng(20261016)
    res = {"C2_final_vs_early_L_runs": c2(rng), "C3_level_coefficient_P_runs": c3(rng),
           "note": "pre-sprint, descriptive; C3 identifies only the level coefficient lambda+"}
    out = ROOT / "results" / "s1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cpu_checks.json").write_text(json.dumps(res, indent=1))
    print("C2 final level (251-300) on early level (21-40), stored L runs")
    for k, v in res["C2_final_vs_early_L_runs"].items():
        print(f"  {k:<26} n={v['n']:<3} slope {v['slope_final_on_early']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  r={v['corr']:+.2f}  early {v['early_range'][0]:.2f}-{v['early_range'][1]:.2f}")
    print("C3 level coefficient lambda+ (P runs, t=21-300)")
    for k, v in res["C3_level_coefficient_P_runs"].items():
        print(f"  {k:<26} runs={v['n_runs']:<3} lambda+ {v['lambda_plus']:.3f} [{v['lambda_plus_ci95'][0]:.3f}, {v['lambda_plus_ci95'][1]:.3f}]  gap coef {v['kappa_gap']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
