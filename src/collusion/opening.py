"""Injected openings (arms S and A): a level-shifted REAL price window as the game's history.

The agent only ever sees the last HISTORY_WINDOW (20) rows. Those rows are rounds 181-200 of a stored L run of the
same model, shifted so that their mean equals the target opening, then clipped to the price bounds:
  - symmetric (x1 == x2, arm S): ONE delta for both firms, so the mean over both firms' 40 prices equals x
    (the rival relationship inside the window is preserved exactly);
  - asymmetric (x1 != x2, arm A): one delta per firm, so firm i's window mean equals x_i.
Rows 1-180 repeat the shifted window, so every one of the 200 prefix rows is consistent with the visible window.
The prefix is re-cleared through the market by game.play (prefix rows are never re-queried), and live play
starts at round 201, exactly like the D branches.
"""
from __future__ import annotations

import random

P_MIN, P_MAX = 1.0, 2.5  # same bounds as parse.clip

WINDOW_LO, WINDOW_HI = 181, 200


def source_order(pool: list[int], perm_seed: int) -> list[int]:
    """Recorded random permutation of the source-run pool: job i uses order[i % len(order)]."""
    if not pool or len(set(pool)) != len(pool):
        raise ValueError("opening_source.pool must be a non-empty list of distinct seeds")
    return random.Random(perm_seed).sample(list(pool), len(pool))


def shifted_window(src_rounds: list[dict], x1: float, x2: float, n_prefix: int = 200,
                   lo: int = WINDOW_LO, hi: int = WINDOW_HI) -> tuple[list[dict], dict]:
    """Return (prefix rows [{t, p1, p2}] for t = 1..n_prefix, info). The last (hi-lo+1) rows are the shifted window."""
    by_t = {r["t"]: r for r in src_rounds if not r.get("is_prefix")}
    win = [by_t.get(t) for t in range(lo, hi + 1)]
    if any(w is None for w in win):
        raise RuntimeError(f"source run lacks rounds {lo}-{hi}")
    k = len(win)
    if n_prefix % k:
        raise ValueError(f"n_prefix={n_prefix} must be a multiple of the window length {k}")
    m1 = sum(w["p1"] for w in win) / k
    m2 = sum(w["p2"] for w in win) / k
    if x1 == x2:
        d1 = d2 = x1 - (m1 + m2) / 2
    else:
        d1, d2 = x1 - m1, x2 - m2
    n_clip = 0
    shifted = []
    for w in win:
        q = []
        for p, d in ((w["p1"], d1), (w["p2"], d2)):
            v = p + d
            c = min(P_MAX, max(P_MIN, v))
            n_clip += c != v
            q.append(c)
        shifted.append(q)
    rows = [{"t": t, "p1": shifted[(t - 1) % k][0], "p2": shifted[(t - 1) % k][1]} for t in range(1, n_prefix + 1)]
    info = {"window": [lo, hi], "src_mean": [m1, m2], "delta": [d1, d2], "n_clipped": n_clip,
            "achieved_mean": [sum(s[0] for s in shifted) / k, sum(s[1] for s in shifted) / k]}
    return rows, info


def transplant_prefix(src_rounds: list[dict], n_prefix: int) -> list[dict]:
    """Arm T: the source run's own rounds 1..n_prefix exactly as played (no shift)."""
    by_t = {r["t"]: r for r in src_rounds if not r.get("is_prefix")}
    if any(t not in by_t for t in range(1, n_prefix + 1)):
        raise RuntimeError(f"source run lacks rounds 1-{n_prefix}")
    return [{"t": t, "p1": by_t[t]["p1"], "p2": by_t[t]["p2"]} for t in range(1, n_prefix + 1)]


def prefix_fingerprint(prefix: list[dict]) -> dict:
    """Recorded with every prefixed run: sha256 of the exact prefix, the realised level of the last 5 rows (what the
    agent last saw), and the within-window OLS slope of the mean price on t over the visible 20 rows."""
    import hashlib
    import json
    canon = json.dumps([[r["t"], round(r["p1"], 12), round(r["p2"], 12)] for r in prefix], separators=(",", ":"))
    win = prefix[-20:]
    m = [(r["p1"] + r["p2"]) / 2 for r in win]
    ts = list(range(len(win)))
    tb, mb = sum(ts) / len(ts), sum(m) / len(m)
    den = sum((t - tb) ** 2 for t in ts)
    slope = sum((t - tb) * (y - mb) for t, y in zip(ts, m, strict=True)) / den if den else 0.0
    last5 = prefix[-5:]
    return {"prefix_sha256": hashlib.sha256(canon.encode()).hexdigest(),
            "realised_last5": [sum(r["p1"] for r in last5) / len(last5), sum(r["p2"] for r in last5) / len(last5)],
            "window_slope": slope}
