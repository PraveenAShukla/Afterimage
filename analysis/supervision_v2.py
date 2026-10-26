"""Pre-registered analysis of supervision v2 (docs/SUPERVISION.md, "Supervision v2").
Frozen at tag prereg-sup2 BEFORE any review (R) run.

  .venv/bin/python analysis/supervision_v2.py --runs-dir runs_cluster --c rerun --out results/supervision_v2

Arms per model × x ∈ {1.55, 1.95} × frame ∈ {shared, individual}:
  R  real review supervisor (stage sup2-<m>, tags supSR / supIR)
  Y  yoked review lines (supSY / supIY)
  C  claim only: --c rerun = the concurrent re-run (sup2-<m>, supSN / supIN); --c v1 = the batch (sup-<m>)
Outcome: each run's mean price over rounds 121-170. Only complete, validated runs are used.
Window check: per cell, the opening-window ids of R, Y and C must be identical (asserted; the run stops otherwise).

Primary, per model, pooled over x and frame: D1 = R - Y (real judging), D2 = Y - C (seeing reviews at all).
  Stratified two-sample test: in each cell (x, frame) both arms are centred on the mean of the two arms in that
  cell; the permutation (20,000 draws, seed 0, two-sided) shuffles arm labels WITHIN cells. Hodges-Lehmann shift of
  the centred values with a distribution-free 90% interval. Holm over the 4 primary tests.
  Labels: "real" if Holm p < 0.05 and |HL| >= 0.03; "no effect" if the interval lies inside +-0.03; else
  "inconclusive".
Secondary: frame within R (shared - individual), stratified by x.
Every contrast is also reported per x, with the collusion index Δ per cell. p^M = 1.925: at x = 1.95 a higher
price is not "more collusion"; read Δ.
Descriptive: "concern noted" share per arm (R, Y); review failures; price change p_f(t) - p_f(t-1) in the round a
label was shown, after "concern noted" vs after "no concern", in R and in Y; share of firm outputs mentioning the
supervisor (MENTION regex, fixed here).
Writes numbers.json (all of the above) and runs.json (one row per run).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

LATE = (121, 170)
MIN_EFFECT = 0.03
Z90 = 1.6448536269514722
MODELS = {"qwen3off": "Qwen3-14B", "qwen25": "Qwen2.5-14B"}
XS = (1.55, 1.95)
FRAMES = ("shared", "individual")
MENTION = re.compile(r"supervis|review|concern", re.I)  # fixed before any R run
WIN = re.compile(r"_w(\d+)_s(\d+)")


def load(d: Path) -> dict | None:
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    if s.get("status") != "complete" or not json.loads((d / "validation.json").read_text()).get("ok"):
        return None
    man = json.loads((d / "manifest.json").read_text())
    R = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines()]
    late = [r for r in R if LATE[0] <= r["t"] <= LATE[1]]
    return {"dir": d, "run_id": d.name, "cfg": man["config"], "rounds": R, "summary": s,
            "window": int(WIN.search(d.name).group(1)), "seed": man["seed"],
            "p": st.mean((r["p1"] + r["p2"]) / 2 for r in late), "delta": s["delta_avg"]}


def collect(runs: Path, m: str, c_src: str) -> dict:
    cells = {}
    for x in XS:
        for fr in FRAMES:
            f = fr[0].upper()
            for arm, stage, tag in (("R", "sup2", "R"), ("Y", "sup2", "Y"),
                                    ("C", "sup2" if c_src == "rerun" else "sup", "N")):
                rs = []
                for d in sorted(runs.glob(f"{stage}-{m}_O_*_x{x:.3f}-{x:.3f}_sup{f}{tag}_w*")):
                    r = load(d)
                    if r:
                        rs.append(r)
                cells[(x, fr, arm)] = rs
            w = {arm: sorted(r["window"] for r in cells[(x, fr, arm)]) for arm in "RYC"}
            if not (w["R"] == w["Y"] == w["C"]):
                sys.exit(f"WINDOW CHECK FAILED {m} x={x} {fr}: {w}")
    return cells


def strat(groups: list[tuple[list[float], list[float]]], n: int = 20000):
    """Groups = [(a_cell, b_cell)]; centre each cell on its pooled mean; stratified permutation of mean(a) - mean(b)."""
    ca, cb = [], []
    for a, b in groups:
        mu = st.mean(a + b)
        ca += [v - mu for v in a]
        cb += [v - mu for v in b]
    obs = st.mean(ca) - st.mean(cb)
    rng, hit = random.Random(0), 0
    for _ in range(n):
        pa, pb = [], []
        for a, b in groups:
            pool = a + b
            rng.shuffle(pool)
            mu = st.mean(pool)
            pa += [v - mu for v in pool[:len(a)]]
            pb += [v - mu for v in pool[len(a):]]
        hit += abs(st.mean(pa) - st.mean(pb)) >= abs(obs) - 1e-12
    d = sorted(x - y for x in ca for y in cb)
    m_, n_ = len(ca), len(cb)
    k = max(0, int(math.floor(m_ * n_ / 2 - Z90 * math.sqrt(m_ * n_ * (m_ + n_ + 1) / 12))))
    return {"diff_means": obs, "hl": st.median(d), "ci90": [d[k], d[-k - 1]], "p": (hit + 1) / (n + 1), "n": [m_, n_]}


def holm(ps: list[float]) -> list[float]:
    order, out, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def label(ph: float, hl: float, lo: float, hi: float) -> str:
    if ph < 0.05 and abs(hl) >= MIN_EFFECT:
        return "real"
    if ph >= 0.05 and -MIN_EFFECT <= lo and hi <= MIN_EFFECT:
        return "no effect"
    return "inconclusive"


def reviews(rs: list[dict]) -> dict:
    """concern share, failures, and price change in the round a label was shown (after concern vs no concern)."""
    n = fail = conc = 0
    after = {"concern noted": [], "no concern": []}
    for r in rs:
        R = {q["t"]: q for q in r["rounds"]}
        for t, q in R.items():
            if q.get("is_prefix"):
                continue
            for f in (1, 2):
                lab = q.get(f"review{f}")
                n += 1
                if lab is None:
                    fail += 1
                    continue
                conc += lab == "concern noted"
                if t - 1 in R:
                    after[lab].append(q[f"p{f}"] - R[t - 1][f"p{f}"])
    ok = n - fail
    return {"labels": n, "no_review": fail, "concern_share": conc / ok if ok else None,
            "dp_after_concern": st.mean(after["concern noted"]) if after["concern noted"] else None,
            "dp_after_no_concern": st.mean(after["no concern"]) if after["no concern"] else None}


def mentions(rs: list[dict]) -> float | None:
    n = h = 0
    for r in rs:
        for line in (r["dir"] / "calls.jsonl").read_text().splitlines():
            c = json.loads(line)
            n += 1
            h += bool(MENTION.search(c.get("raw_output") or ""))
    return h / n if n else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs_cluster")
    ap.add_argument("--c", choices=["rerun", "v1"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs = Path(a.runs_dir)
    out = {"c_source": a.c, "cells": {}, "primary": [], "secondary": [], "by_x": [], "descriptive": {}}
    rows, prim = [], []
    for m in MODELS:
        cells = collect(runs, m, a.c)
        if not any(cells[(x, fr, 'R')] for x in XS for fr in FRAMES):
            continue  # model not run
        for (x, fr, arm), rs in cells.items():
            out["cells"][f"{m}|{x}|{fr}|{arm}"] = {"n": len(rs), "price": st.mean(r["p"] for r in rs) if rs else None,
                                                   "delta": st.mean(r["delta"] for r in rs) if rs else None,
                                                   "windows": sorted(r["window"] for r in rs)}
            rows += [{"model": m, "x": x, "frame": fr, "arm": arm, "run_id": r["run_id"], "seed": r["seed"],
                      "window": r["window"], "late_price": r["p"], "delta": r["delta"]} for r in rs]

        def v(x, fr, arm, cells=cells):
            return [r["p"] for r in cells[(x, fr, arm)]]
        for name, A, B in (("D1 real judging: R - Y", "R", "Y"), ("D2 seeing reviews: Y - C", "Y", "C")):
            res = strat([(v(x, fr, A), v(x, fr, B)) for x in XS for fr in FRAMES])
            prim.append({"model": m, "contrast": name, **res})
            for x in XS:
                out["by_x"].append({"model": m, "contrast": name, "x": x,
                                    **strat([(v(x, fr, A), v(x, fr, B)) for fr in FRAMES])})
        sec = strat([(v(x, "shared", "R"), v(x, "individual", "R")) for x in XS])
        out["secondary"].append({"model": m, "contrast": "frame within R: shared - individual", **sec,
                                 "label": label(sec["p"], sec["hl"], *sec["ci90"])})
        for arm in "RY":
            for fr in FRAMES:
                rs = [r for x in XS for r in cells[(x, fr, arm)]]
                out["descriptive"][f"{m}|{arm}|{fr}|reviews"] = reviews(rs)
        for arm in "RYC":
            out["descriptive"][f"{m}|{arm}|mention_share"] = mentions([r for k, rs in cells.items() if k[2] == arm for r in rs])
    for e, ph in zip(prim, holm([e["p"] for e in prim]), strict=True):
        e["p_holm"] = ph
        e["label"] = label(ph, e["hl"], *e["ci90"])
    out["primary"] = prim
    print(f"== supervision v2 (C = {a.c}); late mean price rounds {LATE[0]}-{LATE[1]}; Δ per cell")
    for k, c in out["cells"].items():
        print(f"  {k:<32} n={c['n']:>2}  price {c['price'] if c['price'] is None else round(c['price'], 3)}  "
              f"Δ {c['delta'] if c['delta'] is None else round(c['delta'], 3)}")
    print("== primary (Holm over 4)")
    for e in prim:
        print(f"  {MODELS[e['model']]:<12} {e['contrast']:<26} HL {e['hl']:+.3f} [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}]"
              f"  p {e['p']:.4f}  Holm {e['p_holm']:.4f}  n={e['n']}  -> {e['label']}")
    for e in out["by_x"]:
        print(f"    by x {MODELS[e['model']]:<12} {e['contrast']:<26} x={e['x']:.2f} HL {e['hl']:+.3f} "
              f"[{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}] p {e['p']:.4f}")
    for e in out["secondary"]:
        print(f"  secondary {MODELS[e['model']]:<12} {e['contrast']} HL {e['hl']:+.3f} [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}] p {e['p']:.4f} -> {e['label']}")
    for k, d in out["descriptive"].items():
        print(f"  {k}: {d}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "numbers.json").write_text(json.dumps(out, indent=1))
        (Path(a.out) / "runs.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
