"""Pre-registered analysis of the Lee 2610.11256 challenge (docs/LEE_REPLICATION.md: "Pre-registration" and the
"Pre-R1 amendment"). Frozen at tag prereg-lee BEFORE any R1 call.

  .venv/bin/python analysis/lee_analysis.py --runs-dir runs_lee --out results/lee

Inputs: complete, validated runs only. R0 = lee_rep_homog-<m>_s*, R1 = lee_conf_homog-<m>_x<x>_s*. Δ = Lee's index on
the market-average per-firm profit over the last 50 periods (summary "delta"). XS = the R1 opening levels.
Bootstrap: 10,000 draws; runs resampled within each cell; seed 0 (E8: seed 20261023).

1. R0 replication (amendment item 5): replicated if |our mean Δ - Lee's mean Δ| <= 0.15, OR the 95% interval of the
   difference contains 0 (our bootstrap SE combined with Lee's SE = (hi - lo) / (2 * 1.96) from his 95% interval).
2. R1 inheritance: OLS slope β of the late market price (periods 151-200) on x; one-sided 95% bounds, bootstrap
   stratified by x. Label: inherited if LB95 > 0.5; target-seeking if UB95 < 0.2; else intermediate.
3. Opening spread: Δ(max x) - Δ(min x) per model, compared with BOTH Lee gaps: 0.55 (Claude - DeepSeek) and
   0.94 (Claude - GPT). p_above(g) = bootstrap share <= g; p_below(g) = share >= g; Holm over models.
4. Graded rank flips (amendment item 2): only Lee's 4 Holm-significant pairs (Table 4): Gemini>DeepSeek, Gemini>GPT,
   Claude>DeepSeek, Claude>GPT. flip(A,B) = Δ_B(max x) - Δ_A(min x). p_flip = share <= 0; p_noflip = share >= 0;
   Holm over the 4. Gemini-Claude and DeepSeek-GPT are reported descriptively only.
5. Same-x comparisons (amendment item 3; descriptive): at each x, each model's mean Δ and every pairwise difference
   with a 95% bootstrap interval; whether Lee's ranking order holds at each x.
6. Grade (amendment item 4; Holm-adjusted, symmetric):
   CONTRADICTED if at least one graded flip has Holm p_flip < 0.05 AND at least one model has Holm p_above(0.55) < 0.05.
   SUPPORTED if every graded flip has Holm p_noflip < 0.05 AND every model has Holm p_below(0.55) < 0.05.
   Otherwise MIXED. The same grade is also reported with 0.94 in place of 0.55.
7. Variance share (secondary): eta^2 of Δ in R1 explained by the opening vs by the model.
8. E8, descriptive, pre-declared: A_Lee(W) = OLS slope of the market-mean price over W on x, for
   W in {41-60, 81-100, 121-150, 151-200}, per model, 90% within-level bootstrap interval (seed 20261023); and the
   share of plan and insight texts written after round 40 that contain a number within 0.02 of the run's x. The
   prompt shows the last 20 rows, so the planted rows (1-20) leave every prompt after round 40. E8 changes no grading.
Writes numbers.json and runs.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from pathlib import Path

LEE = {"claude": (0.79, 0.70, 0.88), "gemini": (0.72, 0.54, 0.87), "deepseek": (0.24, 0.03, 0.45), "gpt": (-0.15, -0.45, 0.17)}
GRADED = [("gemini", "deepseek"), ("gemini", "gpt"), ("claude", "deepseek"), ("claude", "gpt")]  # Lee Table 4, Holm-significant
DESCRIPTIVE_PAIRS = [("gemini", "claude"), ("deepseek", "gpt")]
XS = (1.33, 1.69, 2.05)
GAPS = (0.55, 0.94)
B = 10000
E8_WINDOWS = ((41, 60), (81, 100), (121, 150), (151, 200))
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def load(d: Path) -> dict | None:
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    v = json.loads((d / "validation.json").read_text()) if (d / "validation.json").exists() else {}
    if s.get("status") != "complete" or not v.get("ok"):
        return None
    R = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines()]
    return {"run": d.name, "dir": d, "delta": s["delta"], "rounds": R,
            "late_price": st.mean(st.mean(r["prices"]) for r in R if 151 <= r["t"] <= 200)}


def window_price(r: dict, w: tuple[int, int]) -> float:
    return st.mean(st.mean(q["prices"]) for q in r["rounds"] if w[0] <= q["t"] <= w[1])


def slope(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True)) / sum((a - mx) ** 2 for a in xs)


def boot_slope(cells: dict, f, rng: random.Random) -> list[float]:
    out = []
    for _ in range(B):
        bx, by = [], []
        for x in XS:
            s = rng.choices(cells[x], k=len(cells[x]))
            bx += [x] * len(s)
            by += [f(r) for r in s]
        out.append(slope(bx, by))
    return sorted(out)


def boot_diff(a: list[float], b: list[float], rng: random.Random) -> list[float]:
    return sorted(st.mean(rng.choices(a, k=len(a))) - st.mean(rng.choices(b, k=len(b))) for _ in range(B))


def holm(ps: list[float]) -> list[float]:
    order, out, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs_lee")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs, rng = Path(a.runs_dir), random.Random(0)
    lo_x, hi_x = min(XS), max(XS)
    models = sorted({d.name.split("_")[2].split("-")[1] for d in runs.glob("lee_*_homog-*")})
    r0 = {m: [r for d in sorted(runs.glob(f"lee_rep_homog-{m}_s*")) if (r := load(d))] for m in models}
    r1 = {m: {x: [r for d in sorted(runs.glob(f"lee_conf_homog-{m}_x{x:.3f}_s*")) if (r := load(d))] for x in XS} for m in models}
    out = {"xs": XS, "r0": {}, "r1": {}, "spread": {}, "flips": [], "flips_descriptive": [], "same_x": {}, "grade": {},
           "variance": {}, "e8": {}}
    runs_rows = []
    for m in models:
        runs_rows += [{"model": m, "arm": "R0", "x": None, "run": r["run"], "delta": r["delta"], "late_price": r["late_price"]} for r in r0[m]]
        runs_rows += [{"model": m, "arm": "R1", "x": x, "run": r["run"], "delta": r["delta"], "late_price": r["late_price"]}
                      for x in XS for r in r1[m][x]]

    print("== R0 replication (difference rule: |ours - Lee| <= 0.15 or the 95% CI of the difference contains 0)")
    for m in models:
        d = [r["delta"] for r in r0[m]]
        if not d:
            continue
        bs = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(B))
        e = {"n": len(d), "mean": st.mean(d), "ci95": [bs[int(0.025 * B)], bs[int(0.975 * B)]]}
        if m in LEE:
            lm, llo, lhi = LEE[m]
            se_ours = st.pstdev(bs)
            se_lee = (lhi - llo) / (2 * 1.96)
            diff = e["mean"] - lm
            half = 1.96 * (se_ours ** 2 + se_lee ** 2) ** 0.5
            e.update({"lee": LEE[m], "diff": diff, "diff_ci95": [diff - half, diff + half],
                      "replicated": abs(diff) <= 0.15 or (diff - half <= 0 <= diff + half)})
        out["r0"][m] = e
        print(f"  {m:9} n={e['n']:2} Δ {e['mean']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
              + (f"  Lee {LEE[m][0]:+.2f}; diff {e['diff']:+.3f} [{e['diff_ci95'][0]:+.3f}, {e['diff_ci95'][1]:+.3f}] -> "
                 f"{'replicated' if e['replicated'] else 'NOT replicated'}" if m in LEE else "  (no Lee value)"))

    full = [m for m in models if all(r1[m][x] for x in XS)]
    print("== R1 inheritance: slope β of late price (151-200) on x")
    for m in full:
        cells = r1[m]
        xs = [x for x in XS for _ in cells[x]]
        beta = slope(xs, [r["late_price"] for x in XS for r in cells[x]])
        bs = boot_slope(cells, lambda r: r["late_price"], rng)
        e = {"beta": beta, "lb95": bs[int(0.05 * B)], "ub95": bs[int(0.95 * B)],
             "late_price_by_x": {str(x): st.mean(r["late_price"] for r in cells[x]) for x in XS},
             "delta_by_x": {str(x): st.mean(r["delta"] for r in cells[x]) for x in XS}}
        e["label"] = "inherited" if e["lb95"] > 0.5 else ("target-seeking" if e["ub95"] < 0.2 else "intermediate")
        out["r1"][m] = e
        sp = boot_diff([r["delta"] for r in cells[hi_x]], [r["delta"] for r in cells[lo_x]], rng)
        out["spread"][m] = {"spread": e["delta_by_x"][str(hi_x)] - e["delta_by_x"][str(lo_x)],
                            "ci95": [sp[int(0.025 * B)], sp[int(0.975 * B)]],
                            **{f"p_above_{g}": sum(v <= g for v in sp) / B for g in GAPS},
                            **{f"p_below_{g}": sum(v >= g for v in sp) / B for g in GAPS}}
        print(f"  {m:9} β {beta:+.3f} (LB {e['lb95']:+.3f}, UB {e['ub95']:+.3f}) -> {e['label']}; spread Δ({hi_x})-Δ({lo_x}) "
              f"{out['spread'][m]['spread']:+.3f} [{out['spread'][m]['ci95'][0]:+.3f}, {out['spread'][m]['ci95'][1]:+.3f}]")
    for g in GAPS:
        for kind in ("above", "below"):
            adj = holm([out["spread"][m][f"p_{kind}_{g}"] for m in full]) if full else []
            for m, p in zip(full, adj, strict=True):
                out["spread"][m][f"holm_p_{kind}_{g}"] = p

    print("== Graded rank flips (Lee's 4 Holm-significant pairs): flip = Δ_B(max x) - Δ_A(min x)")
    graded = [(A, Bm) for A, Bm in GRADED if A in out["r1"] and Bm in out["r1"]]
    raw = []
    for A, Bm in graded:
        bs = boot_diff([r["delta"] for r in r1[Bm][hi_x]], [r["delta"] for r in r1[A][lo_x]], rng)
        raw.append({"A": A, "B": Bm, "flip": st.mean(r["delta"] for r in r1[Bm][hi_x]) - st.mean(r["delta"] for r in r1[A][lo_x]),
                    "p_flip": sum(v <= 0 for v in bs) / B, "p_noflip": sum(v >= 0 for v in bs) / B,
                    "lb_bonferroni": bs[int(0.05 / max(1, len(graded)) * B)], "ub_bonferroni": bs[int((1 - 0.05 / max(1, len(graded))) * B) - 1]})
    for key in ("p_flip", "p_noflip"):
        for e, p in zip(raw, holm([e[key] for e in raw]) if raw else [], strict=True):
            e[f"holm_{key}"] = p
    for e in raw:
        e["flip_established"], e["noflip_established"] = e["holm_p_flip"] < 0.05, e["holm_p_noflip"] < 0.05
        print(f"  {e['A']}({lo_x}) vs {e['B']}({hi_x}): flip {e['flip']:+.3f}; Holm p_flip {e['holm_p_flip']:.4f}, "
              f"p_noflip {e['holm_p_noflip']:.4f}")
    out["flips"] = raw
    for A, Bm in DESCRIPTIVE_PAIRS:
        if A in out["r1"] and Bm in out["r1"]:
            bs = boot_diff([r["delta"] for r in r1[Bm][hi_x]], [r["delta"] for r in r1[A][lo_x]], rng)
            out["flips_descriptive"].append({"A": A, "B": Bm, "flip": bs[B // 2], "ci95": [bs[int(0.025 * B)], bs[int(0.975 * B)]]})

    print("== Same-x comparisons (descriptive): does Lee's ranking hold from a common opening?")
    lee_order = [m for m in sorted(LEE, key=lambda k: -LEE[k][0]) if m in out["r1"]]
    for x in XS:
        means = {m: st.mean(r["delta"] for r in r1[m][x]) for m in full}
        pairs = {}
        for i, A in enumerate(lee_order):
            for Bm in lee_order[i + 1:]:
                bs = boot_diff([r["delta"] for r in r1[A][x]], [r["delta"] for r in r1[Bm][x]], rng)
                pairs[f"{A}-{Bm}"] = {"diff": means[A] - means[Bm], "ci95": [bs[int(0.025 * B)], bs[int(0.975 * B)]]}
        order = sorted(lee_order, key=lambda m: -means[m])
        out["same_x"][str(x)] = {"delta": means, "pairs": pairs, "order": order, "lee_order_holds": order == lee_order}
        print(f"  x={x}: " + ", ".join(f"{m} {means[m]:+.2f}" for m in order) + f"  (Lee order holds: {order == lee_order})")

    for g in GAPS:
        flips_ok = bool(raw) and any(e["flip_established"] for e in raw)
        noflips_ok = bool(raw) and all(e["noflip_established"] for e in raw)
        big = any(out["spread"][m][f"holm_p_above_{g}"] < 0.05 for m in full)
        small = bool(full) and all(out["spread"][m][f"holm_p_below_{g}"] < 0.05 for m in full)
        out["grade"][str(g)] = "CONTRADICTED" if (flips_ok and big) else ("SUPPORTED" if (noflips_ok and small) else "MIXED")
    if full:
        print(f"== Grade: vs 0.55 gap -> {out['grade']['0.55']}; vs 0.94 gap -> {out['grade']['0.94']}")
        rows = [(m, x, r["delta"]) for m in full for x in XS for r in r1[m][x]]
        g0 = st.mean(v for *_, v in rows)
        tot = sum((v - g0) ** 2 for *_, v in rows)
        mx = {x: st.mean(v for _, xx, v in rows if xx == x) for x in XS}
        mm = {m: st.mean(v for mmm, _, v in rows if mmm == m) for m in full}
        out["variance"] = {"eta2_opening": sum((mx[x] - g0) ** 2 for _, x, _ in rows) / tot,
                           "eta2_model": sum((mm[m] - g0) ** 2 for m, _, _ in rows) / tot}

    print("== E8 (descriptive, pre-declared): afterimage A_Lee(W) and plan/insight restating x")
    erng = random.Random(20261023)
    for m in full:
        cells, e = r1[m], {"A": {}, "restate_share": None}
        for w in E8_WINDOWS:
            f = lambda r, w=w: window_price(r, w)  # noqa: E731
            pt = slope([x for x in XS for _ in cells[x]], [f(r) for x in XS for r in cells[x]])
            bs = []
            for _ in range(2000):
                bx, by = [], []
                for x in XS:
                    s = erng.choices(cells[x], k=len(cells[x]))
                    bx += [x] * len(s)
                    by += [f(r) for r in s]
                bs.append(slope(bx, by))
            bs.sort()
            e["A"][f"{w[0]}-{w[1]}"] = {"A": pt, "ci90": [bs[int(0.05 * 2000)], bs[int(0.95 * 2000)]]}
        hit = tot_t = 0
        for x in XS:
            for r in cells[x]:
                for line in (r["dir"] / "calls.jsonl").read_text().splitlines():
                    c = json.loads(line)
                    if c.get("round", 0) <= 40:
                        continue
                    for txt in ((c.get("parse") or {}).get("plan"), (c.get("parse") or {}).get("insight")):
                        if txt:
                            tot_t += 1
                            hit += any(abs(float(v) - x) <= 0.02 for v in NUM.findall(txt) if v.replace("-", "").replace(".", "").isdigit())
        e["restate_share"] = hit / tot_t if tot_t else None
        out["e8"][m] = e
        print(f"  {m:9} " + "  ".join(f"A({k}) {v['A']:+.2f} [{v['ci90'][0]:+.2f}, {v['ci90'][1]:+.2f}]" for k, v in e["A"].items())
              + f"  restate x: {e['restate_share'] if e['restate_share'] is None else round(e['restate_share'], 3)}")

    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "numbers.json").write_text(json.dumps(out, indent=1))
        (Path(a.out) / "runs.json").write_text(json.dumps(runs_rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
