"""E2b post-data amendment (tag prereg-afterimage-D3): the luna-high cell, rerun with max_tokens 8192.

At D1, max_tokens 2048 truncated 9.5% of high-effort calls (median 922, p99 at the cap), and 10/24 runs failed the
truncation check. This copies the two D1 high configs UNCHANGED except: stage ait-e2b-luna-high8k (new run ids),
sampling.max_tokens 8192 and est_output_tokens 1400. Same seeds and opening windows, so the contrast with the D1 'none'
cell stays window-paired.
  .venv/bin/python scripts/make_e2b_amend_configs.py
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "configs/afterimage/e2b", ROOT / "configs/afterimage/e2b_amend"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.glob("O_ait-e2b-luna-high_x*.yaml")):
        c = yaml.safe_load(f.read_text())
        c["stage"] = "ait-e2b-luna-high8k"
        c["sampling"] = {**c["sampling"], "max_tokens": 8192}
        c["est_output_tokens"] = 1400
        p = OUT / f.name.replace("luna-high", "luna-high8k")
        p.write_text(yaml.safe_dump(c, sort_keys=False))
        print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
