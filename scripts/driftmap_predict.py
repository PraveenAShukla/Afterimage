"""Plan v2 `pred-M` step: drift-map predictions from the CONFIRMATION one-step maps, using the FROZEN fit and
iteration in analysis/s1.py (tag prereg-s1). DESCRIPTIVE ONLY (pre-registered as such). Written to
results/v2/conf/driftmap_predictions.json, which must be committed and tagged `pred-M` BEFORE any conf O/T run.

  .venv/bin/python scripts/driftmap_predict.py --runs-dir runs_cluster

The only settings changed for v2 are the horizon (150 live rounds, so steps = 150) and the averaging window
(live rounds 101-150, i.e. rounds 121-170, so s1.LIVE_LO = 101). Seeded RNG; 2,000 paths per opening.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s1  # noqa: E402  (frozen at prereg-s1)

OPENINGS = (1.30, 1.47, 1.55, 1.75, 1.95, 2.10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs_cluster"))
    a = ap.parse_args(argv)
    s1.LIVE_LO = 101
    rng = np.random.default_rng(20261023)
    out = {"note": "descriptive only; frozen s1.fit_diag / s1.iterate_map; steps=150, mean over live steps 101-150",
           "models": {}}
    for d in sorted(Path(a.runs_dir).glob("v2M-*_onestep_diag_conf_*")):
        S = s1.load_map(d)
        fd = s1.fit_diag(S)
        pred = s1.iterate_map(fd, openings=OPENINGS, steps=150, rng=rng)
        out["models"][d.name] = {"n_samples": len(S), "resid_sd": fd["resid_sd"], "prediction": pred}
        print(d.name, {k: round(v["mean"], 3) for k, v in pred.items()})
    for d in sorted(Path(a.runs_dir).glob("v2M-*_onestep_grid_conf_*")):
        g = s1.fit_grid(s1.load_map(d), rng)
        out["models"].setdefault(d.name.replace("_grid_", "_diag_"), {})["grid"] = g
        print(d.name, f"a {g['a_own']:.3f} m {g['m_rival']:.3f} a+m {g['a_plus_m']:.3f}")
    p = ROOT / "results" / "v2" / "conf"
    p.mkdir(parents=True, exist_ok=True)
    (p / "driftmap_predictions.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
