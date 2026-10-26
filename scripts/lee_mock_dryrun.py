"""Full-scale rehearsal of the Lee replication with NO paid calls: every configs/lee file is run end to end with each
model's firm replaced by a scripted mock agent (same seeds, rounds, parallelism, openings and validator), R0 first and
then R1 (whose openings are cut from the rehearsal's own R0 runs).

  .venv/bin/python scripts/lee_mock_dryrun.py --out <scratch dir>

Checks: every run completes and validates; R1 openings rebuild from their sources; call counts match the cost
model (R0 8,000 + R1 21,600 firm-calls per model); the mock's known behaviour shows in the summary
(a 'hold' agent keeps its opening, so the R1 late price tracks x with slope about 1).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lee.run import main as run_main  # noqa: E402

MOCK = {"claude": "hold", "gemini": "avg", "gpt": "br", "deepseek": "match_min", "qwen": "hold"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    cfgdir, runs = out / "configs", out / "runs"
    cfgdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for phase in ("R0", "R1"):
        for f in sorted((ROOT / "configs" / "lee").glob(f"{phase}_*.yaml")):
            c = yaml.safe_load(f.read_text())
            name = c["cell"].split("-")[1]
            c["firm"] = {"provider": "mock", "model": f"mock-{name}", "strategy": MOCK[name]}
            p = cfgdir / f.name
            p.write_text(yaml.safe_dump(c, sort_keys=False))
            rc = run_main([str(p), "--runs-dir", str(runs)])
            if rc != 0:
                print(f"FAIL {f.name} rc={rc}")
                return 1
    calls, bad, by = {}, [], {}
    for d in sorted(x for x in runs.iterdir() if x.is_dir() and not x.name.startswith("_")):
        v = json.loads((d / "validation.json").read_text())
        s = json.loads((d / "summary.json").read_text())
        if not v["ok"] or s["status"] != "complete":
            bad.append((d.name, s["status"], v["errors"][:2]))
        m = d.name.split("_")[2]
        calls[m] = calls.get(m, 0) + s["calls"]
        if "_x" in d.name:
            x = float(d.name.split("_x")[1].split("_")[0])
            by.setdefault((m, x), []).append(s.get("mean_price_last50"))
    print(f"runs: {sum(1 for _ in runs.iterdir() if _.is_dir() and not _.name.startswith('_'))}, invalid/incomplete: {len(bad)}, "
          f"wall {time.time() - t0:.0f}s")
    for b in bad[:5]:
        print("  BAD", b)
    for m, n in sorted(calls.items()):
        print(f"  {m:16} firm-calls {n}  (expected 29,600)")
    for (m, x), ps in sorted(by.items()):
        print(f"  R1 {m:16} x={x:.2f} late price {st.mean(ps):.3f}")
    return 0 if not bad and all(n == 29600 for n in calls.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
