"""Presentation figures for S1, drawn ONLY from the frozen analysis outputs (results/s1/s1_runs.json).
No statistic is computed here; analysis/s1.py (tag s1-prereg-v1) stays untouched.

  .venv/bin/python analysis/s1_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "s1"
PN, PM = 1.4729, 1.925
MODELS = {"s1S-qwen3off": ("Qwen3-14B (thinking off)", "#d95f02"), "s1S-qwen25": ("Qwen2.5-14B", "#1b9e77"),
          "s1S-gptoss": ("gpt-oss-20b", "#7570b3")}
CONTROLS = {"s1S-ctl-averager": "averager", "s1S-ctl-br": "static best response", "s1S-ctl-grim": "grim trigger 1.80",
            "s1S-ctl-hillclimb": "hill-climber"}


def fig1(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot([1.1, 2.15], [1.1, 2.15], color="#bbbbbb", lw=1, ls="--")
    ax.text(2.13, 2.15, "opening persists", fontsize=8, color="#888", ha="right", va="bottom")
    for y, lab in ((PN, "Nash 1.47"), (PM, "monopoly 1.93")):
        ax.axhline(y, color="#999", lw=0.7, ls=":")
        ax.text(1.255, y + 0.008, lab, fontsize=8, color="#666")
    for st, lab in CONTROLS.items():  # controls: thin grey, labelled at the right end
        g = [r for r in rows if r["stage"] == st]
        if not g:
            continue
        lv = sorted({r["x1"] for r in g})
        mu = [np.mean([r["final"] for r in g if r["x1"] == v]) for v in lv]
        ax.plot(lv, mu, color="#c8c8c8", lw=1.1, zorder=1)
        ax.text(lv[-1] + 0.015, mu[-1], lab, fontsize=7, color="#999", va="center")
    for st, (lab, col) in MODELS.items():
        g = [r for r in rows if r["stage"] == st]
        if not g:
            continue
        lv = sorted({r["x1"] for r in g})
        mu = [np.mean([r["final"] for r in g if r["x1"] == v]) for v in lv]
        ax.scatter([r["x1"] for r in g], [r["final"] for r in g], s=10, color=col, alpha=0.35, zorder=2)
        ax.plot(lv, mu, color=col, lw=2.4, marker="o", ms=5, label=lab, zorder=3)
    lo = min([1.25] + [r["final"] for r in rows if r["stage"] in MODELS]) - 0.03
    ax.set_xlim(1.25, 2.45)
    ax.set_ylim(lo, 2.2)
    ax.set_xlabel("injected opening price x")
    ax.set_ylabel("final price (both firms, live rounds 61-100)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_final_vs_opening_clean.png", dpi=200)
    plt.close(fig)


def fig3(rows: list[dict]) -> None:
    A = [r for r in rows if r["arm"] == "A" and "ctl" not in r["stage"]]
    if not A:
        return
    ops = sorted({(r["x1"], r["x2"]) for r in A})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    w = 0.18
    names = {"s1A-qwen3off": ("Qwen3-14B", "#d95f02"), "s1A-qwen25": ("Qwen2.5-14B", "#1b9e77")}
    for i, (st, (lab, col)) in enumerate(names.items()):
        for j, op in enumerate(ops):
            g = [r for r in A if r["stage"] == st and (r["x1"], r["x2"]) == op]
            if g:  # points, not bars: the axis does not start at 0
                xs = j + (i - 0.5) * w * 1.1
                ax.scatter([xs] * len(g), [r["final"] for r in g], s=9, color=col, alpha=0.35)
                ax.plot([xs], [np.mean([r["final"] for r in g])], marker="o", ms=8, color=col, ls="none",
                        label=lab if j == 0 else None)
    for j, (x1, x2) in enumerate(ops):
        ax.hlines([x1, (x1 + x2) / 2, x2], j - 0.35, j + 0.35, colors=["#333", "#999", "#333"], linestyles=["-", "--", "-"], lw=1)
        ax.text(j + 0.37, x1, "low firm's opening", fontsize=7, va="center")
        ax.text(j + 0.37, (x1 + x2) / 2, "midpoint", fontsize=7, va="center", color="#777")
        ax.text(j + 0.37, x2, "high firm's opening", fontsize=7, va="center")
    ax.set_xticks(range(len(ops)), [f"({a:.2f}, {b:.2f})" for a, b in ops])
    ax.set_xlim(-0.5, len(ops) - 0.5 + 0.9)
    ax.set_ylim(1.3, 2.2)
    ax.set_xlabel("asymmetric opening (firm 1, firm 2)")
    ax.set_ylabel("final price (both firms)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_asymmetric_clean.png", dpi=200)
    plt.close(fig)


def main() -> int:
    rows = json.loads((OUT / "s1_runs.json").read_text())
    fig1(rows)
    fig3(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
