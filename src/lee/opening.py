"""Randomised openings for the n-firm market (docs/LEE_REPLICATION.md, arm R1): a REAL window (default rounds
31-50) of a stored cold-start run of the SAME model, shifted by ONE delta for all n firms so that its mean price is
x, clipped to [pmin, pmax], and relabelled rounds 1..20. Source windows follow a recorded permutation.
"""
from __future__ import annotations

import hashlib
import json
import random


def order(pool: list[int], perm_seed: int) -> list[int]:
    if not pool or len(set(pool)) != len(pool):
        raise ValueError("pool must be distinct seeds")
    return random.Random(perm_seed).sample(list(pool), len(pool))


def shifted(src_rounds: list[dict], x: float, lo: int = 31, hi: int = 50, pmin: float = 0.0, pmax: float = 5.0) -> tuple[list[list[float]], dict]:
    by = {r["t"]: r for r in src_rounds if not r.get("is_prefix")}
    win = [by.get(t) for t in range(lo, hi + 1)]
    if any(w is None for w in win):
        raise RuntimeError(f"source run lacks rounds {lo}-{hi}")
    allp = [p for w in win for p in w["prices"]]
    d = x - sum(allp) / len(allp)
    rows, n_clip = [], 0
    for w in win:
        r = []
        for p in w["prices"]:
            v = min(pmax, max(pmin, p + d))
            n_clip += v != p + d
            r.append(v)
        rows.append(r)
    canon = json.dumps([[round(p, 12) for p in r] for r in rows], separators=(",", ":"))
    return rows, {"window": [lo, hi], "delta": d, "n_clipped": n_clip, "sha256": hashlib.sha256(canon.encode()).hexdigest(),
                  "achieved_mean": sum(p for r in rows for p in r) / sum(len(r) for r in rows)}
