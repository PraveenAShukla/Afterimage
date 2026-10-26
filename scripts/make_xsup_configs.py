"""Write configs/xsup/*.yaml (fixed once written): external-supervisor follow-up (docs/SUPERVISION.md).

Cross-supervision: Qwen3 firms with a Qwen2.5 supervisor, and Qwen2.5 firms with a Qwen3 supervisor. Real-agent
arms only (supSA, supIA), x in {1.55, 1.95}, 12 fresh seeds each. Each config is the same-model supervision config
with only the supervisor spec, stage and seeds changed. Firms are served at $VLLM_BASE_URL and the supervisor at
$VLLM_BASE_URL_2 (slurm/s1_pair.sbatch: firm model first, supervisor model second). The supervisor keeps its own
model's settings (Qwen3: thinking off).

  .venv/bin/python scripts/make_xsup_configs.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "configs" / "supervision", ROOT / "configs" / "xsup"
PAIRS = {"qwen3off": "qwen25", "qwen25": "qwen3off"}  # firm model -> supervisor model


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
    for firm, sup in PAIRS.items():
        for x in (1.55, 1.95):
            for tag in ("SA", "IA"):
                c = yaml.safe_load((SRC / f"O_{firm}_sup{tag}_x{x:.3f}.yaml").read_text())
                s = yaml.safe_load((SRC / f"O_{sup}_sup{tag}_x{x:.3f}.yaml").read_text())["firm1"]
                rng = random.Random(f"xsup|{firm}|{x:.3f}|{tag}")
                seeds = []
                while len(seeds) < 12:
                    v = rng.randrange(1000000, 1100000)
                    if v not in taken and v not in seeds:
                        seeds.append(v)
                taken |= set(seeds)
                c.update(stage=f"xsup-{firm}", seeds=sorted(seeds))
                c["supervision"] = {**c["supervision"], "supervisor": {**s, "base_url": "$VLLM_BASE_URL_2"}}
                p = OUT / f"O_{firm}_xsup{tag}_x{x:.3f}.yaml"
                if p.exists():
                    sys.exit(f"refusing to overwrite {p}")
                p.write_text(yaml.safe_dump(c, sort_keys=False))
                n += 1
    print(f"wrote {n} configs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
