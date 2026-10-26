"""H1 Holm over the PRE-REGISTERED family only (PREREG_S1: the 4 models). Separate from the frozen analysis/v2.py,
whose h1() applies Holm across every arm-O stage present (E0 spot check, Fig-1 pair, Sonnet included), a bug
logged in the team's design notes. This script reuses the frozen code's own UNADJUSTED one-sided p values from
results/v2/conf/numbers.json and only changes the family. Both versions are reported.

  .venv/bin/python analysis/v2_holm_prereg.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from v2 import holm  # noqa: E402  (frozen helper, unchanged)

FAMILY = ["v2O-qwen3off", "v2O-qwen25", "v2O-gptoss", "v2O-olmo2"]


def main() -> int:
    num = json.loads((ROOT / "results" / "v2" / "conf" / "numbers.json").read_text())
    H1 = num["H1"]
    fam = [s for s in FAMILY if s in H1]
    hg = holm({s: H1[s]["p_beta_gt_0.5"] for s in fam})
    hl = holm({s: H1[s]["p_beta_lt_0.5"] for s in fam})
    out = {}
    for s in fam:
        lab = "inherited" if hg[s] < 0.05 else "not inherited" if hl[s] < 0.05 else "inconclusive"
        out[s] = {"beta_band": H1[s]["beta_band"], "ci90": H1[s]["ci90"], "holm_p_gt_prereg": hg[s], "holm_p_lt_prereg": hl[s],
                  "H1_prereg_family": lab, "H1_frozen_code": H1[s]["H1"]}
        print(f"{s:<14} beta {H1[s]['beta_band']:+.3f} 90% [{H1[s]['ci90'][0]:+.3f}, {H1[s]['ci90'][1]:+.3f}]  "
              f"prereg-family Holm p>0.5 {hg[s]:.4f} p<0.5 {hl[s]:.4f} -> {lab}   (frozen-code label: {H1[s]['H1']})")
    (ROOT / "results" / "v2" / "conf" / "h1_prereg_family.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
