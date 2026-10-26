"""Cross-supervision follow-up (pre-registered in docs/SUPERVISION.md, "Cross-supervision follow-up"; ADVICE-NOTE
family, see docs/SUPERVISION.md, supervision v2).

  .venv/bin/python analysis/xsup.py --runs-dir runs_cluster --out results/xsup

Contrast X per firm model: cross-supervisor advice-note runs (xsup-<m>, SA + IA) - same-model advice-note runs
(sup-<m>, SA + IA), pooled over frame and x. Outcome = late mean price (rounds 121-170) minus the conf O baseline mean
at the same x (as in the v1 read). Permutation (two-sided), HL with 90% interval, Holm over 2, the 0.03 rule.
Per-x results, Δ per cell, and the warning share of the notes are reported as well.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treatments import MODELS, WARN, hl2, holm, label, load, perm2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs_cluster")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs, out, tests = Path(a.runs_dir), {"cells": {}, "primary": [], "by_x": [], "warn_share": {}}, []
    for m in ("qwen3off", "qwen25"):
        base, xs, ys = {}, [], []
        for x in (1.55, 1.95):
            b = [r for d in runs.glob(f"v2O-{m}_O_*_conf_x{x:.3f}-{x:.3f}_w*") if (r := load(d)) and "supervision" not in r["cfg"]]
            base[x] = st.mean(r["p"] for r in b)
            xr = [r for d in runs.glob(f"xsup-{m}_O_*_x{x:.3f}-{x:.3f}_sup[SI]A_*") if (r := load(d))]
            sr = [r for d in runs.glob(f"sup-{m}_O_*_x{x:.3f}-{x:.3f}_sup[SI]A_*") if (r := load(d))]
            for k, rs in (("cross", xr), ("same", sr)):
                out["cells"][f"{m}|{x}|{k}"] = {"n": len(rs), "price": st.mean(r["p"] for r in rs),
                                                "delta": st.mean(json.loads((r["dir"] / "summary.json").read_text())["delta_avg"] for r in rs)}
                notes = [q.get(f"note{f}") for r in rs for q in r["rounds"] if not q.get("is_prefix") for f in (1, 2)]
                notes = [n for n in notes if n]
                out["warn_share"][f"{m}|{x}|{k}"] = sum(bool(WARN.search(n)) for n in notes) / max(1, len(notes))
            hl, lo, hi = hl2([r["p"] for r in xr], [r["p"] for r in sr])
            out["by_x"].append({"model": m, "x": x, "hl": hl, "ci90": [lo, hi], "p": perm2([r["p"] for r in xr], [r["p"] for r in sr])})
            xs += [r["p"] - base[x] for r in xr]
            ys += [r["p"] - base[x] for r in sr]
        tests.append((m, xs, ys))
    raw = [(perm2(x, y), *hl2(x, y)) for _, x, y in tests]
    for (m, x, y), (p, hl, lo, hi), ph in zip(tests, raw, holm([r[0] for r in raw]), strict=True):
        out["primary"].append({"model": m, "contrast": "X: cross-supervisor - same-model (advice notes)", "hl": hl,
                               "ci90": [lo, hi], "p": p, "p_holm": ph, "n": [len(x), len(y)], "label": label(ph, hl, lo, hi)})
    for e in out["primary"]:
        print(f"{MODELS[e['model']]:<12} X HL {e['hl']:+.3f} [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}] p {e['p']:.4f} Holm {e['p_holm']:.4f} n={e['n']} -> {e['label']}")
    for e in out["by_x"]:
        print(f"   by x {MODELS[e['model']]:<12} x={e['x']:.2f} HL {e['hl']:+.3f} [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}] p {e['p']:.4f}")
    for k, v in out["cells"].items():
        print(f"   {k:<22} n={v['n']} price {v['price']:.3f} Δ {v['delta']:+.3f} warn {out['warn_share'][k]:.2f}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "numbers.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
