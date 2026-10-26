"""Re-derive a Lee run from its logs: market clearing every round, the prompt table vs the played history (2-decimal
display), the parse of every raw output, the injected opening (rebuilt from its source run), and the summary.
Writes validation.json {ok, n_errors, errors}.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from collusion.runlog import read_jsonl

from .game import summarise
from .market import NMarket
from .opening import shifted
from .parse import parse_reply

ROW = re.compile(r"^(\d+) \| ([\d. ]+) \| (-?[\d.]+) \| (-?[\d.]+)$", re.M)  # profit < 0 when price < cost


def validate_run(d: Path) -> list[str]:
    man = json.loads((d / "manifest.json").read_text())
    R = read_jsonl(d / "rounds.jsonl")
    C = read_jsonl(d / "calls.jsonl")
    m = NMarket(n=len(man["positions"]))
    errs = []
    if [r["t"] for r in R] != list(range(1, len(R) + 1)):
        errs.append("rounds not contiguous")
    by = {r["t"]: r for r in R}
    for r in R:
        x = m.clear(r["prices"])
        if any(not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(r["pi"], x["pi"], strict=True)):
            errs.append(f"t={r['t']}: profits do not recompute")
            break
    for c in C:
        rr = parse_reply(c["raw_output"])
        if c["parse"]["status"] != rr["status"] or c["parse"].get("price") != rr.get("price"):
            errs.append(f"t={c['round']} firm={c['firm']}: re-parse differs")
            break
        shown = ROW.findall(c["user_prompt"])
        want = [t for t in range(max(1, c["round"] - 20), c["round"])]
        if [int(s[0]) for s in shown] != want:
            errs.append(f"t={c['round']} firm={c['firm']}: prompt shows rounds {[int(s[0]) for s in shown][:2]}.. not the last 20")
            break
        for t, ps, q, pi in shown:
            h = by[int(t)]
            if (any(abs(float(a) - b) > 0.0051 for a, b in zip(ps.split(), h["prices"], strict=True))
                    or abs(float(q) - h["q"][c["firm"]]) > 0.0051 or abs(float(pi) - h["pi"][c["firm"]]) > 0.0051):
                errs.append(f"t={c['round']} firm={c['firm']}: prompt row {t} disagrees with history")
                break
    op = man.get("opening")
    if op:
        rows, info = shifted(read_jsonl(d.parent / op["source_run"] / "rounds.jsonl"), man["config"]["opening"]["x"], *op["window"])
        pre = [r["prices"] for r in R if r.get("is_prefix")]
        if info["sha256"] != op["sha256"] or any(abs(a - b) > 1e-12 for x, y in zip(pre, rows, strict=True) for a, b in zip(x, y, strict=True)):
            errs.append("opening prefix does not rebuild from its source run")
    sp = d / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        rs = summarise(R, m, {}, s["status"])
        for k in ("delta", "mean_price_last50", "converged_lee"):
            if k in rs and (k not in s or (isinstance(rs[k], float) and not math.isclose(rs[k], s[k], abs_tol=1e-9)) or
                            (not isinstance(rs[k], float) and rs[k] != s[k])):
                errs.append(f"summary {k} does not recompute")
    return errs


def write_validation(d: Path) -> list[str]:
    errs = validate_run(d)
    (d / "validation.json").write_text(json.dumps({"ok": not errs, "n_errors": len(errs), "errors": errs[:50]}, indent=1))
    return errs
