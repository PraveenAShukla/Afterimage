"""Paired punishment analysis: forced cut (D) vs no-shock placebo branch from the SAME round-200 state.

  python analysis/punishment_paired.py <stage-prefix> <model-short> [--runs-dir runs]
  e.g. python analysis/punishment_paired.py s1-sonnet55 claude-sonnet-5-5

Per seed: BR to firm 1's forced price at round 201, then:
- depth201 = p2(201) - BR (the spec's measure; positive by construction, because moves are simultaneous);
- depth202 = p2(202) - BR (first round firm 2 can react; < 0 means punishment below best response);
- effect202 = p2_D(202) - p2_placebo(202) (paired causal effect of the cut on firm 2's price);
- effect_avg_202_210 = the same effect averaged over the 9 reaction rounds.
Bootstrap 95% CIs over seeds (10,000 resamples, fixed RNG seed). Reads logs only.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from collusion.market import Market  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402


def analyse(runs: Path, stage: str, short: str) -> dict:
    m = Market()
    rows = []
    for dD in sorted(runs.glob(f"{stage}_D_E1_{short}_s*")):
        seed = dD.name.split("_s")[-1]
        dP = runs / f"{stage}_Dplacebo_E1_{short}_s{seed}"
        if not ((dD / "summary.json").exists() and (dP / "summary.json").exists()):
            continue
        D = {r["t"]: r for r in read_jsonl(dD / "rounds.jsonl")}
        P = {r["t"]: r for r in read_jsonl(dP / "rounds.jsonl")}
        br = m.best_response(D[201]["p1"])
        cut = D[200]["p1"] - D[201]["p1"]
        material = cut > 0.02  # M8: only material cuts count (a "cut" can be a raise when p2 < p^N, M7)
        moved = any(abs(D[t]["p2"] - D[200]["p2"]) > 0.02 for t in range(202, 211))
        rows.append({"seed": seed, "pre_p2": D[200]["p2"], "cut_p1": D[201]["p1"], "br": br,
                     "cut_size": cut, "material": material, "p2_moved": moved,
                     "match_ratio": (D[200]["p2"] - D[202]["p2"]) / cut if material else None,
                     "depth201": D[201]["p2"] - br, "depth202": D[202]["p2"] - br,
                     "effect202": D[202]["p2"] - P[202]["p2"],
                     "effect_avg_202_210": st.mean(D[t]["p2"] - P[t]["p2"] for t in range(202, 211)),
                     "below_br_202": D[202]["p2"] < br - 1e-9,
                     # M8: "recovers" only for material cuts where firm 2 actually moved (else it never left)
                     "recovers": (any(abs(D[t]["p2"] - D[200]["p2"]) <= 0.02 for t in range(203, 211))
                                  if (material and moved) else None)})
    out = {"stage": stage, "model": short, "n": len(rows), "per_seed": rows}
    rng = random.Random(12345)
    for k in ("depth201", "depth202", "effect202", "effect_avg_202_210"):
        v = [r[k] for r in rows]
        if v:
            b = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(10000))
            out[k] = {"mean": st.mean(v), "ci95": [b[249], b[9749]]}
    out["frac_below_br_202"] = sum(r["below_br_202"] for r in rows) / len(rows) if rows else None
    rec = [r["recovers"] for r in rows if r["recovers"] is not None]
    out["frac_recover_within_10"] = sum(rec) / len(rec) if rec else None
    out["n_material_moved"] = len(rec)
    out["n_material"] = sum(r["material"] for r in rows)
    mr = [r["match_ratio"] for r in rows if r["match_ratio"] is not None]
    out["match_ratio_mean"] = st.mean(mr) if mr else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("short")
    ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
    a = ap.parse_args(argv)
    r = analyse(Path(a.runs_dir), a.stage, a.short)
    outdir = Path(__file__).resolve().parents[1] / "results" / a.stage
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "punishment_paired.json").write_text(json.dumps(r, indent=1))
    for k in ("depth201", "depth202", "effect202", "effect_avg_202_210"):
        if k in r:
            print(f"{a.stage} {k:<20} {r[k]['mean']:+.3f}  95% CI [{r[k]['ci95'][0]:+.3f}, {r[k]['ci95'][1]:+.3f}]  n={r['n']}")
    print(f"{a.stage} below BR at r202: {r['frac_below_br_202']} | firm 2 back within 0.02 of pre-shock price by r210: {r['frac_recover_within_10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
