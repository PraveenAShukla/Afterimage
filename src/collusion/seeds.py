"""Pre-registered seed assignment (docs/DESIGN.md 3.3). Written once to configs/seed_assignment.json and never
regenerated: if the file exists, make_assignment refuses. The permutation itself is saved so the
randomisation test is exact and auditable.

Interpretation (a team design decision): a pool of 2n seeds is
drawn, randomly permuted, first n -> L, last n -> P. P run i replays firm 2 of L run i (a different run).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path


def make_assignment(path: Path, n_pairs: int = 20, master_seed: int = 20261015, pool_lo: int = 1, pool_hi: int = 10**6) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: the assignment is pre-registered and must not be regenerated")
    rng = random.Random(master_seed)
    pool = rng.sample(range(pool_lo, pool_hi), 2 * n_pairs)
    perm = list(range(2 * n_pairs))
    rng.shuffle(perm)
    L = [pool[i] for i in perm[:n_pairs]]
    P = [pool[i] for i in perm[n_pairs:]]
    out = {"created_unix": time.time(), "master_seed": master_seed, "pool": pool, "permutation": perm,
           "L": L, "P": P, "P_replays_L": {str(p): l for p, l in zip(P, L, strict=True)}}
    path.write_text(json.dumps(out, indent=2))
    return out


def load_assignment(path: Path) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    import sys
    print(json.dumps(make_assignment(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 20), indent=1)[:400])
