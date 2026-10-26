"""Live spend + run monitor, and the kill switch.

  python -m collusion.monitor                 # one snapshot
  python -m collusion.monitor --watch 60      # refresh every 60 s (with --auto-stop it also pulls the brake)
  python -m collusion.monitor --stop          # create runs/STOP: all paid calls refused, every game aborts next round
  python -m collusion.monitor --resume        # remove runs/STOP

Alerts:
- spend ≥ 80% of a provider cap (auto-stop at ≥ 95%);
- the rolling 10-minute spend ≥ 80% of MAX_USD_PER_10MIN;
- a running game whose heartbeat is older than 10 min (hung);
- a run's fallback or truncation rate above the health limits.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .env import load_env
from .runlog import REPO


def ledger(runs: Path) -> list[dict]:
    p = runs / "_ledger" / "spend.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def snapshot(runs: Path) -> tuple[list[str], list[str], bool]:
    """(report lines, alerts, should_auto_stop)"""
    lines, alerts, stop = [], [], False
    recs, now = ledger(runs), time.time()
    closed = {r.get("id") for r in recs if r.get("type", "settle") == "settle"}
    lim10 = float(os.environ.get("MAX_USD_PER_10MIN", "5"))
    lines.append(f"{'provider':<10} {'cap':>8} {'settled':>9} {'open':>8} {'last10m':>8} {'%cap':>6}")
    for prov in ("anthropic", "openai"):
        cap = float(os.environ.get(f"BUDGET_USD_{prov.upper()}", "0"))
        settled = sum(r["usd"] for r in recs if r["provider"] == prov and r.get("type", "settle") == "settle")
        open_ = sum(r["usd_worst"] for r in recs if r["provider"] == prov and r.get("type") == "reserve" and r["id"] not in closed)
        last10 = sum(r["usd"] for r in recs if r["provider"] == prov and r.get("type", "settle") == "settle" and now - r["ts"] < 600)
        pct = 100 * settled / cap if cap else 0.0
        lines.append(f"{prov:<10} {cap:>8.2f} {settled:>9.4f} {open_:>8.4f} {last10:>8.4f} {pct:>5.1f}%")
        if cap and pct >= 80:
            alerts.append(f"{prov}: {pct:.0f}% of cap spent")
        if cap and pct >= 95:
            stop = True
        if last10 >= 0.8 * lim10:
            alerts.append(f"{prov}: ${last10:.2f} in 10 min (tripwire ${lim10:.2f})")
    active = []
    for hb in sorted(runs.glob("*/heartbeat.json")):
        d = hb.parent
        if (d / "summary.json").exists():
            continue
        h = json.loads(hb.read_text())
        age = now - h["ts"]
        active.append(f"  {d.name[:60]:<60} r{h['round']}/{h['of']} eta {h['eta_s'] / 60:5.1f}m "
                      f"fb={h['fallbacks']} trunc={h['truncated']} ${h['usd']:.3f} (hb {age:.0f}s ago)")
        if age > 600:
            alerts.append(f"{d.name}: no heartbeat for {age / 60:.0f} min (hung?)")
        if h["calls"] and h["fallbacks"] / max(h["round"], 1) > 0.1:
            alerts.append(f"{d.name}: fallback rate high ({h['fallbacks']} in {h['round']} rounds)")
    lines.append(f"active runs: {len(active)}" + ("\n" + "\n".join(active) if active else ""))
    lines.append(f"kill switch: {'ON (runs/STOP exists)' if (runs / 'STOP').exists() else 'off'}")
    return lines, alerts, stop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--watch", type=int, default=0)
    ap.add_argument("--auto-stop", action="store_true", help="create runs/STOP at >=95% of a cap")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    load_env()
    runs = Path(a.runs_dir)
    runs.mkdir(exist_ok=True)
    if a.stop:
        (runs / "STOP").write_text(f"stopped at {time.ctime()}\n")
        print("kill switch ON: paid calls refused, running games abort at their next round")
        return 0
    if a.resume:
        (runs / "STOP").unlink(missing_ok=True)
        print("kill switch off")
        return 0
    while True:
        lines, alerts, stop = snapshot(runs)
        print(time.strftime("%H:%M:%S"), "\n".join(lines), sep="\n")
        for al in alerts:
            print("ALERT:", al)
        if stop and a.auto_stop and not (runs / "STOP").exists():
            (runs / "STOP").write_text(f"auto-stop at {time.ctime()}: {alerts}\n")
            print("AUTO-STOP: kill switch ON")
        if not a.watch:
            return 1 if alerts else 0
        time.sleep(a.watch)


if __name__ == "__main__":
    raise SystemExit(main())
