"""Mechanism figures (addendum): the direction plane (rho_up vs rho_down) with every scripted control from the
exact mock gate and each LLM with its 90% CI; and the IMP late-market response. Drawn from results/mech/*.json only.

  .venv/bin/python analysis/mech_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mech"
LLM = {"mSTEP-qwen3off": ("Qwen3-14B", "#d95f02"), "mSTEP-qwen25": ("Qwen2.5-14B", "#1b9e77"),
       "mSTEP-sonnet55": ("Sonnet 5.5 (T=1)", "#7570b3")}


def main() -> int:
    gate = json.loads((OUT / "mgate0" / "gate_table.json").read_text())["rows"]
    mech = json.loads((OUT / "mech.json").read_text())
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([-0.1, 1.4], [-0.1, 1.4], color="#bbb", ls="--", lw=1)
    ax.text(1.05, 1.12, "symmetric", color="#999", fontsize=8, rotation=45)
    ax.text(0.12, 1.36, "upper-left corner = ratchet (match cuts, ignore raises)", color="#999", fontsize=8, va="top")
    seen = set()
    for r in gate:
        if r["arm"] != "STEP" or r["policy"] in seen:
            continue
        seen.add(r["policy"])
        if r["policy"] == "averager":  # sits exactly on "match" (1, 1): labelled together
            continue
        if r["rho_up"] > 1.4:  # grim (2.18, 0): off the plot; labelled at the edge
            ax.annotate("grim 1.80 (2.18, 0) →", (1.12, 0.02), fontsize=7, color="#888")
            continue
        ax.scatter(r["rho_up"], r["rho_down"], marker="x", color="#888", s=30)
        lab = "match / averager" if r["policy"] == "match" else "BR / hill-climber" if r["policy"] == "br" else r["policy"]
        if r["policy"] == "hillclimb":
            continue
        ax.annotate(lab, (r["rho_up"], r["rho_down"]), xytext=(5, -10), textcoords="offset points", fontsize=7, color="#888")
    for st, (lab, col) in LLM.items():
        e = mech["STEP"].get(st)
        if not e or "rho_up" not in e:
            continue
        u, d = e["rho_up"], e["rho_down"]
        ax.errorbar(u["mean"], d["mean"], xerr=[[u["mean"] - u["ci90"][0]], [u["ci90"][1] - u["mean"]]],
                    yerr=[[d["mean"] - d["ci90"][0]], [d["ci90"][1] - d["mean"]]], fmt="o", color=col, ms=7, capsize=3,
                    label=f"{lab}: {e['label']}")
    ax.set_xlim(-0.1, 1.4)
    ax.set_ylim(-0.1, 1.4)
    ax.set_xlabel("ρ_up: share of a rival's RAISE passed through")
    ax.set_ylabel("ρ_down: share of a rival's CUT passed through")
    ax.legend(fontsize=7, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    fig.savefig(OUT / "fig3_direction_plane.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    names = []
    for i, st in enumerate(["mIMP-qwen3off", "mIMP-qwen25"]):
        e = mech["IMP"].get(st)
        if not e or "H_down" not in e:
            continue
        names.append(st.split("-", 1)[1])
        for j, (k, col) in enumerate((("H_down", "#c0392b"), ("H_up", "#2c7fb8"))):
            h = e[k]
            ax.errorbar(i + (j - 0.5) * 0.25, h["mean"], yerr=[[h["mean"] - h["ci90"][0]], [h["ci90"][1] - h["mean"]]],
                        fmt="o", color=col, capsize=3, label=("after a one-round CUT" if k == "H_down" else "after a one-round RAISE") if i == 0 else None)
    ax.axhline(0, color="#999", lw=1)
    ax.set_xticks(range(len(names)), ["Qwen3-14B", "Qwen2.5-14B"][: len(names)])
    ax.set_ylabel("lasting share of the shock\n(late market vs placebo) / Δ")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_imp_persistence.png", dpi=200)
    plt.close(fig)
    print("figures in", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
