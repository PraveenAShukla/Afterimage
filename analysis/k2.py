"""K2 (IFM K2, disclosed extra model, outside the frozen H1 family): its own numbers file.

The frozen results/v2/conf/numbers.json predates the K2 runs. K2's figures (beta_band 0.015, 90% CI [-0.006, +0.036])
were computed by analysis/v2.py over runs_cluster after K2 was added, which also reshuffles the shared bootstrap draws
of the other stages. So: conf is verified WITHOUT the K2 runs (EXTRA_STAGES), and K2 is verified here, from v2.py over
the full run set, keeping only the K2 block.

  .venv/bin/python analysis/k2.py            (writes results/v2/k2/numbers.json)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = "v2O-k2"
EXTRA_STAGES = (STAGE,)   # stages the frozen conf numbers must be verified without


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs_cluster"))
    ap.add_argument("--out", default=str(ROOT / "results" / "v2" / "k2"))
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "analysis" / "v2.py"), "--phase", "conf", "--runs-dir", a.runs_dir,
                        "--out", tmp], check=True, capture_output=True)
        full = json.loads((Path(tmp) / "numbers.json").read_text())
    res = {"stage": STAGE, "source": "analysis/v2.py --phase conf over the full conf run set (K2 included); H1 block only",
           "H1": {STAGE: full["H1"][STAGE]}, "excluded": {k: v for k, v in full["excluded"].items() if k.startswith(STAGE)}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    e = res["H1"][STAGE]
    print(f"K2 beta_band {e['beta_band']:.4f} ci90 [{e['ci90'][0]:.4f}, {e['ci90'][1]:.4f}] n_band {e['n_band']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
