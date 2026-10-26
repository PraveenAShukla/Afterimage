"""Write configs/sup2/*.yaml and configs/sup2/yoke_pairing.json (fixed once written): supervision v2
(docs/SUPERVISION.md, supervision v2).

Per model (qwen3off, qwen25) × x ∈ {1.55, 1.95} × frame (shared, individual):
  R  supervision {frame, agent: true,  mode: review}                       12 fresh seeds   (tag supSR / supIR)
  Y  supervision {frame, agent: false, mode: yoked, yoke: {seed: R run}}   12 fresh seeds   (tag supSY / supIY)
  C  supervision {frame, agent: false}  (claim only, concurrent re-run)    12 fresh seeds   (tag supSN / supIN)
Each config is the v1 claim-only config for that model, x and frame (configs/supervision/O_<m>_sup{S,I}N_x*.yaml)
with only `supervision`, `stage` (sup2-<m>) and the seeds changed. 12 seeds per cell, so every arm uses opening
windows order[0..11] (window matching).
Yoke pairing: yoked seed i takes the review run of R seed perm[i], perm drawn once from random.Random(PAIR_SEED).
A yoked seed is never its source's seed, and all seeds are disjoint from every seed in configs/.

  .venv/bin/python scripts/make_sup2_configs.py     (needs runs_cluster/ for the opening sources: run ids)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collusion.run import build_jobs  # noqa: E402

SRC, OUT = ROOT / "configs" / "supervision", ROOT / "configs" / "sup2"
RUNS = ROOT / "runs_cluster"
PAIR_SEED = 20261021


def used_seeds() -> set[int]:
    s = set()
    for p in (ROOT / "configs").rglob("*.yaml"):
        c = yaml.safe_load(p.read_text()) or {}
        if isinstance(c, dict) and isinstance(c.get("seeds"), list):
            s |= {int(x) for x in c["seeds"] if isinstance(x, int)}
    return s


def fresh(rng: random.Random, taken: set[int], n: int = 12) -> list[int]:
    out = []
    while len(out) < n:
        v = rng.randrange(1100000, 1300000)
        if v not in taken and v not in out:
            out.append(v)
    taken |= set(out)
    return sorted(out)


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    taken, pairing, n = used_seeds(), {"pair_seed": PAIR_SEED, "pairs": {}}, 0
    for m in ("qwen3off", "qwen25"):
        for x in (1.55, 1.95):
            for frame in ("shared", "individual"):
                f = frame[0].upper()
                base = yaml.safe_load((SRC / f"O_{m}_sup{f}N_x{x:.3f}.yaml").read_text())
                assert base["supervision"] == {"frame": frame, "agent": False}, base["supervision"]
                key = f"{m}|{x:.3f}|{frame}"
                rng = random.Random(f"sup2|{key}")
                cells = {}
                for arm in ("R", "C", "Y"):
                    cells[arm] = {**base, "stage": f"sup2-{m}", "seeds": fresh(rng, taken)}
                cells["R"]["supervision"] = {"frame": frame, "agent": True, "mode": "review"}
                cells["C"]["supervision"] = {"frame": frame, "agent": False}
                rjobs = {j.seed: j.run_id for j in build_jobs(cells["R"], RUNS)}
                perm = list(cells["R"]["seeds"])
                random.Random(f"{PAIR_SEED}|{key}").shuffle(perm)
                yoke = {ys: rjobs[rs] for ys, rs in zip(cells["Y"]["seeds"], perm, strict=True)}
                assert all(not rid.endswith(f"_s{ys}") for ys, rid in yoke.items())
                cells["Y"]["supervision"] = {"frame": frame, "agent": False, "mode": "yoked", "yoke": yoke}
                pairing["pairs"][key] = {str(ys): rid for ys, rid in yoke.items()}
                for arm, c in cells.items():
                    tag = {"R": "R", "Y": "Y", "C": "N"}[arm]
                    p = OUT / f"O_{m}_sup{f}{tag}_x{x:.3f}.yaml"
                    p.write_text(yaml.safe_dump(c, sort_keys=False))
                    n += 1
    (OUT / "yoke_pairing.json").write_text(json.dumps(pairing, indent=1))
    print(f"wrote {n} configs + yoke_pairing.json to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
