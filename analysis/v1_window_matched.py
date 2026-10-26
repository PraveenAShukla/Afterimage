"""Post-hoc robustness for the v1 supervision and pressure reads (docs/SUPERVISION.md, supervision v2, "Relabel"):
recompute the contrasts that use the conf O baseline against a WINDOW-MATCHED baseline, and report every contrast
by x with the collusion index Δ per cell.

  .venv/bin/python analysis/v1_window_matched.py --runs-dir runs_cluster --out results/v1_window_matched

Window matching: a treatment cell's 12 runs use opening windows order[0..11]. The 24-run baseline uses all 18
windows (order[0..17], then order[0..5] again). Matched baseline = baseline runs whose window id appears in the
treatment cell. Tests, intervals and labels are those of analysis/treatments.py (imported). Holm over the same
6-test family as the original read. This is a robustness re-read, not a new pre-registered test.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from treatments import MODELS, hl2, holm, label, load, perm2  # noqa: E402

WIN = re.compile(r"_w(\d+)_s\d+")
STUDIES = {"supervision": ("sup", ["SA", "SN", "IA", "IN"], ("SN", "IN")),
           "pressure": ("pr", ["G2", "L2", "G1", "L1"], ("G2", "L2"))}


def win(d: Path) -> int:
    return int(WIN.search(d.name).group(1))


def tag(r: dict, study: str) -> str:
    t = r["cfg"]["supervision" if study == "supervision" else "pressure"]
    if study == "supervision":
        return t["frame"][0].upper() + ("A" if t["agent"] else "N")
    return t["frame"][0].upper() + ("2" if t["target"] == "both" else "1")


def delta(r: dict) -> float:
    return json.loads((r["dir"] / "summary.json").read_text())["delta_avg"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs_cluster")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs, out = Path(a.runs_dir), {}
    for study, (pre, cells, primary) in STUDIES.items():
        res = {"cells": {}, "contrast_by_x": [], "pooled": []}
        tests = []
        for m in ("qwen3off", "qwen25"):
            tr = defaultdict(list)
            for d in sorted(runs.glob(f"{pre}-{m}_O_*")):
                r = load(d)
                if r:
                    tr[(r["cfg"]["opening"][0], tag(r, study))].append(r)
            xs = sorted({k[0] for k in tr})
            base = defaultdict(list)
            for d in sorted(runs.glob(f"v2O-{m}_O_*_conf_*")):
                r = load(d)
                if r and "supervision" not in r["cfg"] and "pressure" not in r["cfg"] and r["cfg"]["opening"][0] in xs:
                    base[r["cfg"]["opening"][0]].append(r)
            pooled_t, pooled_b, pooled_tf, pooled_bf = [], [], [], []
            for x in xs:
                tw = {win(r["dir"]) for c in primary for r in tr[(x, c)]}
                bm = [r for r in base[x] if win(r["dir"]) in tw]
                for c in cells + ["base_full", "base_matched"]:
                    rs = tr.get((x, c)) if c in cells else (base[x] if c == "base_full" else bm)
                    if rs:
                        res["cells"][f"{m}|{x}|{c}"] = {"price": st.mean(r["p"] for r in rs), "delta": st.mean(delta(r) for r in rs),
                                                         "n": len(rs), "windows": sorted({win(r["dir"]) for r in rs})}
                t = [r["p"] for c in primary for r in tr[(x, c)]]
                for kind, b in (("matched", bm), ("full", base[x])):
                    bv = [r["p"] for r in b]
                    hl, lo, hi = hl2(t, bv)
                    res["contrast_by_x"].append({"model": m, "x": x, "baseline": kind, "hl": hl, "ci90": [lo, hi],
                                                 "p": perm2(t, bv), "n": [len(t), len(bv)]})
                bmm = st.mean(r["p"] for r in bm)
                bfm = st.mean(r["p"] for r in base[x])
                pooled_t += [v - bmm for v in t]
                pooled_b += [r["p"] - bmm for r in bm]
                pooled_tf += [v - bfm for v in t]
                pooled_bf += [r["p"] - bfm for r in base[x]]
            tests.append((m, "matched", pooled_t, pooled_b))
            tests.append((m, "full", pooled_tf, pooled_bf))
        # Holm within each baseline version over the 2 models of this contrast (the 6-test family's other 4 tests
        # do not use the baseline and are unchanged; they are carried from results/<study>/numbers.json)
        orig = json.loads((Path("results") / study / "numbers.json").read_text())["primary"]
        others = [o["p"] for o in orig if not o["contrast"].startswith(("C1", "P1"))]
        for kind in ("matched", "full"):
            sub = [t for t in tests if t[1] == kind]
            raw = [(perm2(x, y), *hl2(x, y)) for _, _, x, y in sub]
            adj = holm([r[0] for r in raw] + others)[:len(raw)]
            for (m, _, x, y), (p, hl, lo, hi), ph in zip(sub, raw, adj, strict=True):
                res["pooled"].append({"model": m, "baseline": kind, "hl": hl, "ci90": [lo, hi], "p": p, "p_holm": ph,
                                      "n": [len(x), len(y)], "label": label(ph, hl, lo, hi)})
        out[study] = res
        name = "C1 belief (N arms - baseline)" if study == "supervision" else "P1 pressure on both - baseline"
        print(f"\n== {study}: {name}")
        for e in res["pooled"]:
            print(f"  {MODELS[e['model']]:<12} {e['baseline']:<8} HL {e['hl']:+.3f} [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}] "
                  f"Holm {e['p_holm']:.4f} n={e['n']} -> {e['label']}")
        for e in res["contrast_by_x"]:
            print(f"    by x {MODELS[e['model']]:<12} x={e['x']:.2f} {e['baseline']:<8} HL {e['hl']:+.3f} "
                  f"[{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}] p {e['p']:.4f} n={e['n']}")
        print("  cells (price, Δ, n):")
        for k, v in res["cells"].items():
            print(f"    {k:<28} {v['price']:.3f}  Δ {v['delta']:+.3f}  n={v['n']}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "numbers.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
