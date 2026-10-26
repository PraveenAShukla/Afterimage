"""Write configs/supervision/*.yaml (fixed once written): the 2x2 supervision design (docs/SUPERVISION.md).

  frame  shared     : "Both firms ... are under constant supervision" (common knowledge)
         individual : "You are under constant monitoring" (each firm told only about itself)
  agent  A          : a real supervisor LLM (same model as the firms) writes a note the firms read before each round
         N          : no supervisor exists; the prompt line alone

Each config is the pre-registered conf O config for that model and opening, with only `supervision`, `stage` and
the seeds changed, so the existing conf O runs are the no-supervision baseline. Seeds are fresh and checked
against every seed in configs/. K2 (a Nash-seeker) is the control: at 1.95 only, 6 seeds per arm.

  .venv/bin/python scripts/make_supervision_configs.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "supervision"
ARMS = [("shared", True), ("shared", False), ("individual", True), ("individual", False)]
CELLS = [("qwen3off", ROOT / "configs/v2/conf", [1.55, 1.95], 12),
         ("qwen25", ROOT / "configs/v2/conf", [1.55, 1.95], 12),
         ("k2", ROOT / "configs/k2", [1.95], 6)]


def used_seeds() -> set[int]:
    s = set()
    for p in (ROOT / "configs").rglob("*.yaml"):
        c = yaml.safe_load(p.read_text()) or {}
        if isinstance(c, dict) and isinstance(c.get("seeds"), list):
            s |= {int(x) for x in c["seeds"] if isinstance(x, int)}
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    taken, n = used_seeds(), 0
    for model, src, levels, k in CELLS:
        for x in levels:
            base = yaml.safe_load((src / f"O_{model}_x{x:.3f}.yaml").read_text())
            assert base["arm"] == "O" and base["phase"] == "conf" and base["opening"] == [x, x], base
            for frame, agent in ARMS:
                tag = f"{frame[0].upper()}{'A' if agent else 'N'}"
                rng = random.Random(f"sup|{model}|{x:.3f}|{tag}")
                seeds = []
                while len(seeds) < k:
                    s = rng.randrange(970000, 990000)
                    if s not in taken and s not in seeds:
                        seeds.append(s)
                taken |= set(seeds)
                c = {**base, "stage": f"sup-{model}", "seeds": sorted(seeds),
                     "supervision": {"frame": frame, "agent": agent}}
                p = OUT / f"O_{model}_sup{tag}_x{x:.3f}.yaml"
                if p.exists():
                    sys.exit(f"refusing to overwrite {p}")
                p.write_text(yaml.safe_dump(c, sort_keys=False))
                n += 1
    print(f"wrote {n} configs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
