"""Pre-registered analysis of the supervision (docs/SUPERVISION.md) and pressure (docs/PRESSURE.md) arms.

  .venv/bin/python analysis/treatments.py supervision --runs-dir runs_cluster [--out results/supervision]
  .venv/bin/python analysis/treatments.py pressure    --runs-dir runs_cluster [--out results/pressure]

Outcome: the run's mean price over rounds 121-170 minus the baseline mean for the same model and x. Baseline = the
conf O runs whose manifest config equals the treatment config without the treatment key, `stage` and `seeds`.
Two-sample contrasts: Monte-Carlo permutation of run values (two-sided, 20,000 draws, seed 0). One-sample (pressure
P3): exact sign-flip when n <= 20, else Monte Carlo. Hodges-Lehmann shift with a distribution-free 90% interval
(normal approximation to the rank-statistic quantiles). Holm across the 6 primary tests (3 contrasts x 2 models).
"Real" = Holm p < 0.05 and |HL| >= 0.03; "no effect" = not significant and the interval inside +-0.03; else
"inconclusive". Only complete runs (status complete) that validate (validation.json ok) are used.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

LATE = (121, 170)
MIN_EFFECT = 0.03
Z90 = 1.6448536269514722
MODELS = {"qwen3off": "Qwen3-14B", "qwen25": "Qwen2.5-14B", "k2": "K2"}
WARN = re.compile(r"coordinat|collu|cartel|tacit|anticompetitive|warn|investigat|suspicious|lockstep", re.I)
STUDY = {"supervision": ("sup", "supervision", ["SA", "SN", "IA", "IN"]),
         "pressure": ("pr", "pressure", ["G2", "L2", "G1", "L1"])}


def load(d: Path) -> dict | None:
    if not (d / "summary.json").exists() or not json.loads((d / "validation.json").read_text()).get("ok") \
            or json.loads((d / "summary.json").read_text()).get("status") != "complete":
        return None
    man = json.loads((d / "manifest.json").read_text())
    R = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines()]
    late = [r for r in R if LATE[0] <= r["t"] <= LATE[1]]
    return {"dir": d, "cfg": man["config"], "rounds": R, "p": st.mean((r["p1"] + r["p2"]) / 2 for r in late),
            "p1": st.mean(r["p1"] for r in late), "p2": st.mean(r["p2"] for r in late)}


def strip(cfg: dict, key: str) -> str:
    return json.dumps({k: v for k, v in cfg.items() if k not in (key, "stage", "seeds", "_config_path")}, sort_keys=True)


def perm2(a: list[float], b: list[float], n: int = 20000) -> float:
    rng, pool, obs = random.Random(0), a + b, abs(st.mean(a) - st.mean(b))
    hit = 0
    for _ in range(n):
        rng.shuffle(pool)
        hit += abs(st.mean(pool[:len(a)]) - st.mean(pool[len(a):])) >= obs - 1e-12
    return (hit + 1) / (n + 1)


def signflip(x: list[float], n: int = 20000) -> float:
    obs = abs(sum(x))
    if len(x) <= 20:
        tot = hit = 0
        for s in itertools.product((1, -1), repeat=len(x)):
            tot += 1
            hit += abs(sum(a * b for a, b in zip(s, x, strict=True))) >= obs - 1e-12
        return hit / tot
    rng = random.Random(0)
    return (sum(abs(sum(v * rng.choice((1, -1)) for v in x)) >= obs - 1e-12 for _ in range(n)) + 1) / (n + 1)


def hl2(a: list[float], b: list[float]) -> tuple[float, float, float]:
    d = sorted(x - y for x in a for y in b)
    m, n = len(a), len(b)
    k = max(0, int(math.floor(m * n / 2 - Z90 * math.sqrt(m * n * (m + n + 1) / 12))))
    return st.median(d), d[k], d[-k - 1]


def hl1(x: list[float]) -> tuple[float, float, float]:
    w = sorted((x[i] + x[j]) / 2 for i in range(len(x)) for j in range(i, len(x)))
    n = len(x)
    k = max(0, int(math.floor(n * (n + 1) / 4 - Z90 * math.sqrt(n * (n + 1) * (2 * n + 1) / 24))))
    return st.median(w), w[k], w[-k - 1]


def holm(ps: list[float]) -> list[float]:
    order, out, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def label(p_holm: float, hl: float, lo: float, hi: float) -> str:
    if p_holm < 0.05 and abs(hl) >= MIN_EFFECT:
        return "REAL"
    if p_holm >= 0.05 and -MIN_EFFECT <= lo and hi <= MIN_EFFECT:
        return "no effect"
    return "inconclusive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=sorted(STUDY))
    ap.add_argument("--runs-dir", default="runs_cluster")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pre, key, cells = STUDY[a.study]
    runs = Path(a.runs_dir)
    treat, base = defaultdict(list), defaultdict(list)  # (model, x, cell) / (model, x)
    for m in MODELS:
        tr = [r for d in sorted(runs.glob(f"{pre}-{m}_O_*")) if (r := load(d))]
        if not tr:
            continue
        shapes = {strip(r["cfg"], key) for r in tr}
        for d in sorted(runs.glob(f"v2O-{m}_O_*_conf_*")):
            r = load(d)
            if r and strip(r["cfg"], key) in shapes and key not in r["cfg"]:
                base[(m, r["cfg"]["opening"][0])].append(r)
        for r in tr:
            t = r["cfg"][key]
            tag = (t["frame"][0].upper() + ("A" if t["agent"] else "N")) if a.study == "supervision" else \
                (t["frame"][0].upper() + ("2" if t["target"] == "both" else "1"))
            treat[(m, r["cfg"]["opening"][0], tag)].append(r)
    xs = sorted({k[1] for k in treat})
    out = {"study": a.study, "cells": {}, "baseline": {}, "primary": [], "notes": {}}
    print(f"== {a.study}: late mean price (rounds {LATE[0]}-{LATE[1]}), n runs")
    for m in MODELS:
        for x in xs:
            b = base.get((m, x), [])
            if not any((m, x, c) in treat for c in cells):
                continue
            bm = st.mean(r["p"] for r in b) if b else float("nan")
            out["baseline"][f"{m}|{x}"] = {"mean": bm, "n": len(b)}
            row = f"{MODELS[m]:<12} x={x:.2f}  base {bm:.3f} (n={len(b)})"
            for c in cells:
                v = [r["p"] for r in treat.get((m, x, c), [])]
                if v:
                    out["cells"][f"{m}|{x}|{c}"] = {"mean": st.mean(v), "sd": st.pstdev(v), "n": len(v)}
                    row += f"  {c} {st.mean(v):.3f}±{st.pstdev(v):.3f} (n={len(v)})"
            print(row)
    # centred values: run price minus the baseline mean at its (model, x)
    def cen(m, c):
        return [r["p"] - out["baseline"][f"{m}|{r['cfg']['opening'][0]}"]["mean"] for x in xs for r in treat.get((m, x, c), [])]

    def cenb(m):
        return [r["p"] - out["baseline"][f"{m}|{x}"]["mean"] for x in xs for r in base.get((m, x), [])]
    tests = []
    for m in ("qwen3off", "qwen25"):
        if not any((m, x, c) in treat for x in xs for c in cells):
            continue
        if a.study == "supervision":
            N, A = cen(m, "SN") + cen(m, "IN"), cen(m, "SA") + cen(m, "IA")
            S, I_ = cen(m, "SA") + cen(m, "SN"), cen(m, "IA") + cen(m, "IN")
            tests += [(m, "C1 belief: N arms - baseline", N, cenb(m)), (m, "C2 real agent: A - N", A, N),
                      (m, "C3 frame: shared - individual", S, I_)]
        else:
            both = cen(m, "G2") + cen(m, "L2")
            asym = [r["p1"] - r["p2"] for x in xs for c in ("G1", "L1") for r in treat.get((m, x, c), [])]
            tests += [(m, "P1 pressure on both - baseline", both, cenb(m)), (m, "P2 loss - gain (both)", cen(m, "L2"), cen(m, "G2")),
                      (m, "P3 firm1 - firm2 (one-firm cells)", asym, None)]
    raw = []
    for _m, _name, x, y in tests:
        if y is None:
            raw.append((signflip(x), *hl1(x), len(x), 0))
        else:
            raw.append((perm2(x, y), *hl2(x, y), len(x), len(y)))
    adj = holm([r[0] for r in raw])
    print(f"\n== primary contrasts (Holm over {len(raw)}); effect >= {MIN_EFFECT} to count")
    for (m, name, _, _), (p, hl, lo, hi, n1, n2), ph in zip(tests, raw, adj, strict=True):
        lab = label(ph, hl, lo, hi)
        out["primary"].append({"model": m, "contrast": name, "hl": hl, "ci90": [lo, hi], "p": p, "p_holm": ph,
                               "n": [n1, n2], "label": lab})
        print(f"{MODELS[m]:<12} {name:<36} HL {hl:+.3f} 90% [{lo:+.3f}, {hi:+.3f}]  p {p:.4f}  Holm {ph:.4f}  n={n1},{n2}  -> {lab}")
    if a.study == "supervision":  # note content (descriptive)
        print("\n== supervisor notes: share with warning words; mean price change next round after warn vs not")
        for m in MODELS:
            for c in ("SA", "IA"):
                rs = [r for x in xs for r in treat.get((m, x, c), [])]
                if not rs:
                    continue
                notes, dw, dn, miss = 0, [], [], 0
                for r in rs:
                    R = {q["t"]: q for q in r["rounds"]}
                    for t, q in R.items():
                        if q.get("is_prefix") or t + 1 not in R:
                            continue
                        for f in (1, 2):
                            nt = R[t + 1].get(f"note{f}")  # the note read before round t+1
                            if not nt:
                                miss += 1
                                continue
                            notes += 1
                            (dw if WARN.search(nt) else dn).append(R[t + 1][f"p{f}"] - q[f"p{f}"])
                share = len(dw) / max(1, notes)
                out["notes"][f"{m}|{c}"] = {"notes": notes, "missing": miss, "warn_share": share,
                                            "dp_after_warn": st.mean(dw) if dw else None, "dp_after_other": st.mean(dn) if dn else None}
                print(f"{MODELS[m]:<12} {c}: {notes} notes seen, {miss} missing, warn share {share:.2f}, "
                      f"dp after warn {st.mean(dw) if dw else float('nan'):+.4f}, after other {st.mean(dn) if dn else float('nan'):+.4f}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "numbers.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
