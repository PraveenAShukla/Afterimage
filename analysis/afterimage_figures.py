"""Figures 1 and 2 for the Price Afterimage write-up, from the JSON outputs only (afterimage.py, afterimage_card.py,
results/lee/numbers.json) plus the two same-seed runs for the inset.

  Fig 1  A(W) over rounds, one line per model; Qwen3 highlighted, the rest grey; dashed line at 40.5 where the last
         planted row leaves the 20-row prompt window; rails at 1 (tracks the rival price) and 0 (best responds).
  Fig 2  Two axes: cold-start collusion index (no plant) vs late afterimage A(121-170); scripted known answers and
         Lee's market (E8) marked separately.
  .venv/bin/python analysis/afterimage_figures.py [--runs-dir runs_cluster] [--out results/afterimage]
Figures go to out/figures/ by default, so a rerun never overwrites the committed results/afterimage/ figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2 import ROOT  # noqa: E402

from collusion.market import Market, collusion_index  # noqa: E402
from collusion.runlog import read_jsonl  # noqa: E402

ACCENT, GREY, INK, MUTED = "#2980B9", "#95A5A6", "#222222", "#6b6b6b"   # figure-fix palette: steel blue = ours, gray = rest
HALO = [pe.withStroke(linewidth=3, foreground="white"), pe.Normal()]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name: str) -> Path:
    p = FIG_OUT / name
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return p


def text_h(ax, pt: float) -> float:
    """Font size in data units on the y axis (figure-fix technique 4); call after set_ylim."""
    h_in = ax.get_figure().get_size_inches()[1] * ax.get_position().height
    return pt / (h_in * 72 / (ax.get_ylim()[1] - ax.get_ylim()[0]))
OUT = ROOT / "results" / "afterimage"   # inputs: numbers files and cards
FIG_OUT = ROOT / "out" / "figures"     # outputs (--out to change)
LINES = [("v2O-qwen3off", "Qwen3-14B"), ("v2O-qwen25", "Qwen2.5-14B"), ("v2O-gptoss", "gpt-oss-20b"),
         ("v2O-k2", "K2"), ("ait-e4-haiku55", "Claude Haiku 5.5"), ("ait-e4-sonnet55", "Claude Sonnet 5.5"),
         ("ait-e4-gpt61sol", "GPT-6.1-sol")]
NAMES = {"qwen3-14b-off": ("Qwen3-14B (off)", (-6, 5)), "qwen3-14b-off-E5": ("Qwen3-14B E5", (-6, -11)),
         "qwen2.5-14b": ("Qwen2.5-14B", (-6, 4)), "gpt-oss-20b": ("gpt-oss-20b", (6, 3)), "haiku-5.5": ("Claude Haiku 5.5", (6, 3)),
         "sonnet-5.5": ("Claude Sonnet 5.5", (6, 3)), "gpt-6.1-sol": ("GPT-6.1-sol", (6, 3))}
LOW = {"haiku-5.5": (-0.2, -0.13), "gpt-6.1-sol": (0.2, 0.12), "lee-gemini": (0.83, 0.13)}   # crowded band near A = 0
PAIR = "runs_cluster/v2Ofig1-qwen3off_O_E1_Qwen3-14B-AWQ_conf_x{x}-{x}_w29283_s936921"


def stages() -> dict:
    out = {}
    for f in sorted(OUT.glob("*/numbers.json")):
        out.update(json.loads(f.read_text()).get("stages", {}))
    return out


def bins(A: dict) -> tuple[list, list]:
    ks = sorted((k for k in A if k.count("-") == 1 and int(k.split("-")[1]) - int(k.split("-")[0]) == 9), key=lambda k: int(k.split("-")[0]))
    return [(int(k.split("-")[0]) + int(k.split("-")[1])) / 2 for k in ks], [A[k]["A"] for k in ks]


def fig1(st: dict, runs_dir: Path) -> Path:
    fig, (ax, ins) = plt.subplots(1, 2, figsize=(8.2, 3.5), gridspec_kw={"width_ratios": [3.3, 1], "wspace": 0.95})
    ax.set_xlim(20, 170)
    ax.set_ylim(-0.16, 1.12)
    edge = blended_transform_factory(ax.transAxes, ax.transData)
    ax.axhline(1, color=MUTED, lw=0.7, ls=":")
    ax.axhline(0, color=MUTED, lw=0.7, ls=":")
    ax.text(0.99, 1.015, "1 = tracks the rival price", transform=edge, fontsize=8, color=MUTED, va="bottom", ha="right",
            path_effects=HALO)
    ax.text(0.01, -0.02, "0 = best responds", transform=edge, fontsize=8, color=MUTED, va="top", ha="left",
            path_effects=HALO)
    ax.axvline(40.5, color=INK, lw=0.9, ls="--")
    ax.text(40.5, 1.13, "planted rows leave the prompt window \u2192", fontsize=8, color=INK, va="bottom", ha="left",
            clip_on=False)
    ends = []
    for s_, name in LINES:
        e = st.get(s_, {}).get("A", {}).get("p")
        if not e:
            continue
        x, y = bins(e)
        hi = s_ == "v2O-qwen3off"
        if hi:
            ks = sorted((k for k in e if k.count("-") == 1 and int(k.split("-")[1]) - int(k.split("-")[0]) == 9), key=lambda k: int(k.split("-")[0]))
            ax.fill_between(x, [e[k]["ci90_within"][0] for k in ks], [e[k]["ci90_within"][1] for k in ks], color=ACCENT, alpha=0.15, lw=0)
        ax.plot(x, y, color=ACCENT if hi else GREY, lw=2.2 if hi else 1.2, zorder=3 if hi else 2)
        ends.append((y[-1], name, hi))
    gap = 1.25 * text_h(ax, 8.5)
    groups = []                                       # lines ending within one label height share ONE label
    for y0, name, hi in sorted(ends):
        if groups and y0 - groups[-1][-1][0] < gap and not hi and not groups[-1][-1][2]:
            groups[-1].append((y0, name, hi))
        else:
            groups.append([(y0, name, hi)])
    last = None
    for g in groups:
        yc = sum(v[0] for v in g) / len(g)
        name = g[0][1] if len(g) == 1 else ",\n".join(", ".join(v[1] for v in g[i:i + 2]) for i in range(0, len(g), 2)) + " (all ≈ 0)"
        hi = g[0][2]
        y1 = yc if last is None else max(yc, last + gap)
        ax.annotate(name, (x[-1], yc), xytext=(1.02, y1), textcoords=edge, fontsize=8.5, va="center",
                    color=INK if hi else MUTED, fontweight="bold" if hi else "normal", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkA=0, shrinkB=2))
        last = y1
    ax.set_xlabel("round (10-round bins)")
    ax.set_ylabel("afterimage $A$ (share of the\nplanted difference still in the price)")
    for xv, col in (("1.550", GREY), ("1.950", ACCENT)):
        d = runs_dir.parent / PAIR.format(x=xv)
        if (d / "rounds.jsonl").exists():
            R = sorted(read_jsonl(d / "rounds.jsonl"), key=lambda r: r["t"])
            ins.plot([r["t"] for r in R], [(r["p1"] + r["p2"]) / 2 for r in R], color=col, lw=1)
    ins.axvline(20.5, color=INK, lw=0.6, ls="--")
    ins.set_title("one Qwen3 seed,\ntwo plants", fontsize=9)
    ins.set_xlabel("round")
    ins.set_ylabel("market price")
    return save(fig, "fig1_afterimage_curve.png")


def fig2() -> Path:
    m = Market()
    b = m.benchmarks()
    d16 = float(collusion_index(m.profits(1.6, 1.6)[0], b["piN"], b["piM"]))
    board = json.loads((OUT / "cards" / "leaderboard.json").read_text())
    cards = {c.stem: json.loads(c.read_text()) for c in (OUT / "cards").glob("*.json") if c.stem != "leaderboard"}
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    texts, xs, ys = [], [], []
    for name, c in cards.items():
        cs = c.get("cold_start")
        if not cs or c["label"].startswith("degenerate") or name.endswith("-E5"):
            continue
        hi = name.startswith("qwen3")
        x, y = cs["delta_mean"], c["A_late"]["A"]
        ax.scatter(x, y, s=48, color=ACCENT if hi else GREY, zorder=6, edgecolor="white", lw=0.8)
        lab = NAMES.get(name, (name,))[0]
        if name in LOW:
            ax.annotate(lab, (x, y), xytext=LOW[name], textcoords="data", fontsize=8.5, color=INK, path_effects=HALO,
                        ha="center", va="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkB=3))
        else:
            texts.append(ax.text(x, y, lab, fontsize=8.5, color=INK, fontweight="bold" if hi else "normal",
                                 path_effects=HALO))
        xs.append(x)
        ys.append(y)
    lee = json.loads((ROOT / "results" / "lee" / "numbers.json").read_text())
    for row in board:
        if row["market"] != "Lee 4-firm":
            continue
        k = row["card"].split("-", 1)[1]
        x0 = lee["r0"].get(k, {}).get("mean")
        if x0 is not None and row["A_late"] is not None:
            ax.scatter(x0, row["A_late"], s=42, facecolor="none", edgecolor=MUTED, lw=1.2, zorder=3)
            if f"lee-{k}" in LOW:
                ax.annotate(f"{k} (Lee's market)", (x0, row["A_late"]), xytext=LOW[f"lee-{k}"], textcoords="data", fontsize=8,
                            color=MUTED, path_effects=HALO, ha="center", va="center",
                            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkB=4))
            else:
                texts.append(ax.text(x0, row["A_late"], f"{k} (Lee's market)", fontsize=8, color=MUTED, path_effects=HALO))
            xs.append(x0)
            ys.append(row["A_late"])
    for x, y, lab, at in ((d16, 1.0, "scripted copier", (d16 - 0.32, 1.0)), (d16, 0.0, "scripted focal 1.60", (0.5, -0.17)),
                          (0.0, 0.0, "scripted best response", (-0.45, 0.09))):   # hand-placed known answers
        ax.scatter(x, y, marker="s", s=30, color=INK, zorder=4)
        ax.annotate(lab, (x, y), xytext=at, textcoords="data", fontsize=8, color=INK, path_effects=HALO, ha="center",
                    va="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkB=4))
        xs.append(x)
        ys.append(y)
    ax.set_xlim(min(xs) - 0.25, max(xs) + 0.45)
    ax.set_ylim(-0.25, 1.12)
    adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.3, 1.8), force_text=(0.6, 1.0),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    ax.set_xlabel("cold-start collusion index $\\Delta$ (no plant; 0 = Nash, 1 = monopoly)")
    ax.set_ylabel("late afterimage $A$(121–170)")
    return save(fig, "fig2_two_axes.png")


RELAY = [  # (stage, label, rival re-emits the plant?) - firm-1 afterimage, rounds 121-170
    ("v2O-qwen3off", "Qwen3 vs Qwen3 (self-play)", True),
    ("ait-e1-copy-qwen3off", "vs a rival that copies its price", True),
    ("ait-d2-sham-qwen3off", "self-play + sham note", True),
    ("ait-e1-br-qwen3off", "vs a rival that best-responds", False),
    ("ait-e1b-nash-qwen3off", "vs a rival fixed at Nash", False),
    ("ait-d2-hint-qwen3off", "self-play + best-response hint", False),
]


def fig3_relay() -> Path:
    """Signature figure: Qwen3 keeps the plant only while its rival keeps re-emitting it."""
    st = stages()
    rows = [(lab, st[s]["A"]["p1"]["121-170"], keep) for s, lab, keep in RELAY if s in st]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    for i, (_lab, a, keep) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, max(a["A"], 0.004), height=0.62, color=ACCENT if keep else GREY)
        lo, hi = a["ci90_within"]
        ax.plot([lo, hi], [y, y], color=INK, lw=0.6, alpha=0.5)
        ax.text(max(a["A"], hi) + 0.02, y, f"{a['A']:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Qwen3's afterimage, rounds 121-170 (higher = more of the plant kept)")
    ax.axvline(1, color=MUTED, lw=0.6, ls=":")
    return save(fig, "fig3_relay_cut.png")


def main(argv=None) -> int:
    global FIG_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs_cluster"))
    ap.add_argument("--out", default=str(FIG_OUT))
    a = ap.parse_args(argv)
    FIG_OUT = Path(a.out)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    print(fig1(stages(), Path(a.runs_dir)), fig2(), fig3_relay())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
