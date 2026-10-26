"""Write configs/k2/O_k2_x*.yaml (fixed once written): IFM K2 as a disclosed extra model in the S1 conf O band.
Each config is the gpt-oss conf O config at the same level (same opening pool, perm_seed, rounds, window, prompt) with
only the firm, sampling, parallelism and seeds changed. Seeds are fresh and checked against every seed in configs/.

  .venv/bin/python scripts/make_k2_configs.py
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "configs" / "v2" / "conf", ROOT / "configs" / "k2"
# reasoning_effort low, as gpt-oss: at the default effort K2 reasoned past 16,384 tokens on the round-21 prompt (2/2 calls)
FIRM = {"provider": "ifm", "model": "IFM/K2-Horizon-375B-A23B", "extra_body": {"reasoning_effort": "low"}}
LEVELS = [1.30, 1.47, 1.55, 1.75, 1.95, 2.10]
N_SEEDS = 12


def used_seeds() -> set[int]:
    s = set()
    for p in (ROOT / "configs").rglob("*.yaml"):
        c = yaml.safe_load(p.read_text()) or {}
        if isinstance(c, dict) and isinstance(c.get("seeds"), list):
            s |= {int(x) for x in c["seeds"] if isinstance(x, int)}
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    taken = used_seeds()
    for x in LEVELS:
        c = yaml.safe_load((SRC / f"O_gptoss_x{x:.3f}.yaml").read_text())
        assert c["arm"] == "O" and c["phase"] == "conf" and c["opening"] == [x, x], c
        rng = random.Random(f"k2|O|{x:.3f}")
        seeds = []
        while len(seeds) < N_SEEDS:
            s = rng.randrange(960000, 970000)
            if s not in taken and s not in seeds:
                seeds.append(s)
        taken |= set(seeds)
        c.update(stage="v2O-k2", seeds=sorted(seeds), sampling={"temperature": 0.7, "top_p": 1.0, "max_tokens": 4096},
                 parallel_games=12, est_output_tokens=600, firm1=dict(FIRM), firm2=dict(FIRM))
        p = OUT / f"O_k2_x{x:.3f}.yaml"
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
        p.write_text(re.sub(r"\n(firm1:)", r"\n\1", yaml.safe_dump(c, sort_keys=False)))
    print(f"wrote {len(LEVELS)} configs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
