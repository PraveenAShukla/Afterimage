"""Plan v2 figures, drawn ONLY from analysis outputs (results/v2/<phase>/{numbers,runs}.json) and the scripted
control runs. No statistic is computed here except the plotted level means.

  .venv/bin/python analysis/v2_figures.py --phase disc
Fig 2 (headline): late price vs injected opening per model; band shaded; 45 degree line; Nash and monopoly;
       scripted matcher (averager) and static BR overlays from results/classify (dev runs).
Fig 4: opening-induced spread of delta within each model vs the between-model spread of cold-start delta.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collusion.runlog import read_jsonl  # noqa: E402

PN, PM = 1.4729, 1.925
COL = {"v2O-qwen3off": ("Qwen3-14B (thinking off)", "#d95f02"), "v2O-qwen25": ("Qwen2.5-14B", "#1b9e77"),
       "v2O-gptoss": ("gpt-oss-20b", "#7570b3"), "v2O-olmo2": ("OLMo-2-13B (held out)", "#e7298a")}


def control_curve(pol: str) -> tuple[list, list]:
    xs, ys = [], []
    for x in (1.55, 1.75, 1.95):
        v = []
        for d in (ROOT / "runs_cluster").glob(f"cls-dev-{pol}_O_E1_*_x{x:.3f}-{x:.3f}_*"):
            R = read_jsonl(d / "rounds.jsonl")
            v.append(np.mean([(r["p1"] + r["p2"]) / 2 for r in R if 121 <= r["t"] <= 170]))
        if v:
            xs.append(x)
            ys.append(float(np.mean(v)))
    return xs, ys


def fig2(rows: list[dict], out: Path, phase: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.axvspan(1.55, 1.95, color="#f2f2f2", zorder=0)
    ax.text(1.75, 1.02, "band", ha="center", fontsize=8, color="#777", transform=ax.get_xaxis_transform())
    ax.plot([1.1, 2.2], [1.1, 2.2], color="#bbbbbb", lw=1, ls="--")
    ax.text(2.18, 2.16, "opening persists", fontsize=8, color="#888", ha="right")
    for y, lab in ((PN, "Nash 1.47"), (PM, "monopoly 1.93")):
        ax.axhline(y, color="#999", lw=0.7, ls=":")
        ax.text(1.255, y + 0.008, lab, fontsize=8, color="#666")
    for pol, lab in (("averager", "scripted matcher"), ("br", "scripted best response")):
        xs, ys = control_curve(pol)
        if xs:
            ax.plot(xs, ys, color="#c8c8c8", lw=1.2, zorder=1)
            ax.text(xs[-1] + 0.02, ys[-1], lab, fontsize=7, color="#999", va="center")
    lo = 1.25
    for st, (lab, col) in COL.items():
        g = [r for r in rows if r["stage"] == st and r["arm"] == "O"]
        if not g:
            continue
        lv = sorted({r["x1"] for r in g})
        mu = [np.mean([r["late"] for r in g if r["x1"] == v]) for v in lv]
        lo = min(lo, min(r["late"] for r in g) - 0.03)
        ax.scatter([r["x1"] for r in g], [r["late"] for r in g], s=9, color=col, alpha=0.3, zorder=2)
        ax.plot(lv, mu, color=col, lw=2.4, marker="o", ms=5, label=lab, zorder=3)
    ax.set_xlim(1.25, 2.4)
    ax.set_ylim(lo, 2.2)
    ax.set_xlabel("injected opening price x (rounds 1-20)")
    ax.set_ylabel("late price (both firms, rounds 121-170)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"fig2_late_vs_opening_{phase}.png", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["disc", "conf"], required=True)
    a = ap.parse_args(argv)
    d = ROOT / "results" / "v2" / a.phase
    rows = json.loads((d / "runs.json").read_text())
    fig2(rows, d, a.phase)
    print("figures in", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
