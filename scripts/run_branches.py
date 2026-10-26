"""Run the punishment branches (D, D-frozen, D-placebo) on a HEADLINE model's Stage-1 L runs.

  python scripts/run_branches.py configs/models/<name>.yaml [--approve-usd X]

Uses the pipeline's own config builder and logging (runs/_pipeline/<name>/pipeline.jsonl), stage s1-<name>, so the
analysis pairs the branches with the L runs they branch from. Shock at round 201, 10-round window.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from collusion.env import load_env  # noqa: E402
from collusion.pipeline import Pipeline  # noqa: E402
from collusion.runlog import REPO  # noqa: E402

load_env()
spec = yaml.safe_load(open(sys.argv[1]))
approve = float(sys.argv[sys.argv.index("--approve-usd") + 1]) if "--approve-usd" in sys.argv else None
p = Pipeline(spec, REPO / "runs", approve, dry_run=False)
seed_rel = str(p.seeds_path.relative_to(REPO))
n1 = spec.get("n", {}).get("stage1", 10)
tmpl = f"s1-{p.name}_L_{spec.get('prompt', 'E1')}_{p.short()}_s{{seed}}"
p.log("branches_start", approve_usd=p.approve, spent_so_far=round(p.spent(), 4))
for arm in ("D", "Dfrozen", "Dplacebo"):
    p.run_cfg(p.cfg("s1", arm, seed_assignment=seed_rel, seeds_from="L", n_runs=n1, source_run_template=tmpl,
                    shock_round=201, window=10), f"s1_{arm}")
p.log("branches_end", spent_usd=round(p.spent(), 4))
