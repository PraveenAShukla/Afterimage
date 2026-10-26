"""E8: the afterimage A_Lee(W) on a mock R1 output: a copier gives A ≈ 1, a best responder A ≈ 0."""
import json
import sys
from pathlib import Path

import yaml

from lee.run import main as run_main

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import lee_analysis as LA  # noqa: E402


def test_e8_copier_one_best_responder_zero(tmp_path):
    runs = tmp_path / "runs"
    for name, strat in (("claude", "hold"), ("gpt", "br")):
        base = {"stage": "lee", "cell": f"homog-{name}", "firm": {"provider": "mock", "model": f"mock-{name}", "strategy": strat},
                "n": 4, "rounds": 200, "max_output_tokens": 200, "parallel_games": 2}
        p = tmp_path / f"R0_{name}.yaml"
        p.write_text(yaml.safe_dump({**base, "phase": "rep", "seeds": [11, 12]}))
        assert run_main([str(p), "--runs-dir", str(runs)]) == 0
        for x in LA.XS:
            p = tmp_path / f"R1_{name}_{x}.yaml"
            p.write_text(yaml.safe_dump({**base, "phase": "conf", "seeds": [int(x * 100) + (1 if name == "gpt" else 0)],
                                         "opening": {"x": x, "source_template": f"lee_rep_homog-{name}_s{{seed}}", "pool": [11, 12],
                                                     "perm_seed": 4301, "window": [31, 50]}}))
            assert run_main([str(p), "--runs-dir", str(runs)]) == 0
    out = tmp_path / "res"
    assert LA.main(["--runs-dir", str(runs), "--out", str(out)]) == 0
    e8 = json.loads((out / "numbers.json").read_text())["e8"]
    for w in ("41-60", "81-100", "121-150", "151-200"):
        assert abs(e8["claude"]["A"][w]["A"] - 1.0) < 0.02, (w, e8["claude"]["A"][w])   # copier: the plant persists
        assert abs(e8["gpt"]["A"][w]["A"]) < 0.02, (w, e8["gpt"]["A"][w])                # best responder: none of it does
    assert e8["claude"]["restate_share"] is not None
