"""Lever 1: the best-response hint line and its matched sham. One line after the history rows,
never at round 1; the prompt is byte-identical when the view key is absent."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from collusion.config import ConfigError
from collusion.market import Market
from collusion.prompts import SHAM_LINE, hint_line, user_message
from collusion.run import main
from collusion.runlog import read_jsonl
from collusion.validate import validate_run
from test_opening import V2, _controls_setup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import afterimage as AI  # noqa: E402

H = [{"round": 1, "own_price": 1.6, "rival_price": 1.7, "own_share": 0.4, "own_profit": 0.2}]


def test_byte_identical_when_off_and_never_at_round_1():
    assert user_message(5, H) == user_message(5, H, {}) == user_message(5, H, None, hint=(1.7, 1.5))
    for kind in ("br_hint", "sham"):
        assert user_message(1, [], {"line": kind}, hint=(1.7, 1.5)).endswith("No history yet.")
    assert user_message(5, H, {"line": "sham"}).endswith("\n" + SHAM_LINE)
    assert user_message(5, H, {"line": "br_hint"}, hint=(1.7, 1.5)).endswith(hint_line("br_hint", (1.7, 1.5)))
    assert "1.700" in hint_line("br_hint", (1.7, 1.5)) and "1.500" in hint_line("br_hint", (1.7, 1.5))


def _run(tmp_path, strategy, line, x=1.55, seeds=(1,), arm="O"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": strategy}  # sources are mock; each call uses its own dir
    cfg = {**V2, "arm": arm, "seeds": list(seeds), "firm1": f, "firm2": f}
    if arm == "O":
        cfg["opening"] = [x, x]
    else:
        for k in ("opening", "opening_source"):
            cfg.pop(k, None)
    if line:
        cfg["view"] = {"line": line}
    p = tmp_path / f"{arm}_{strategy}_{line}_{x}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    tag = {"br_hint": "_vBRH", "sham": "_vSHM", None: ""}[line]
    return [d for d in runs.iterdir() if d.name.startswith(V2["stage"]) and f"x{x:.3f}" in d.name and
            (tag in d.name if tag else "_vBRH" not in d.name and "_vSHM" not in d.name)] if arm == "O" else \
        [d for d in runs.iterdir() if f"_{arm}_" in d.name and tag in d.name and "_s1" in d.name and not d.name.endswith(("_s11", "_s12"))]


@pytest.mark.parametrize("line", ["br_hint", "sham"])
def test_validator_passes_and_catches_a_wrong_line(tmp_path, line):
    (d,) = _run(tmp_path, "hint", line)
    assert validate_run(d) == []
    rows = [json.loads(x) for x in (d / "calls.jsonl").read_text().splitlines()]
    k = next(i for i, c in enumerate(rows) if c.get("round", 0) > 25)
    rows[k]["user_prompt"] = rows[k]["user_prompt"].replace("Note: ", "Note:  ")
    (d / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert any("line" in e for e in validate_run(d))


def test_cold_start_under_the_hint_passes_round_1(tmp_path):
    (d,) = _run(tmp_path, "hint", "br_hint", arm="L")
    assert validate_run(d) == []
    first = [c for c in read_jsonl(d / "calls.jsonl") if c["round"] == 1]
    assert first and all("Note: " not in c["user_prompt"] for c in first)


def _A(tmp_path, strategy, line):
    xs, late = [], []
    for x in (1.55, 1.95):
        for d in _run(tmp_path / f"{strategy}{line}{x}", strategy, line, x=x, seeds=(1, 2)):
            by = {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}
            xs.append(x)
            late.append(np.mean([(by[t]["p1"] + by[t]["p2"]) / 2 for t in range(121, 171) if t in by]))
    return AI.slope(xs, late)


def test_mock_known_answers(tmp_path):
    assert abs(_A(tmp_path, "hint", "br_hint")) < 0.02           # follows the hint -> Nash, forgets x
    assert abs(_A(tmp_path, "hint", None) - 1) < 0.02            # no line -> plays the rival's last price
    assert abs(_A(tmp_path, "averager", "br_hint") - 1) < 0.02   # a policy that ignores the line keeps x
    assert abs(_A(tmp_path, "averager", "sham") - 1) < 0.02
    m = Market()
    assert abs(m.best_response(m.nash()[0]) - m.nash()[0]) < 1e-6


def test_bad_line_rejected(tmp_path):
    f = {"provider": "mock", "model": "mock", "strategy": "averager"}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({**V2, "arm": "O", "opening": [1.55, 1.55], "seeds": [1], "firm1": f, "firm2": f,
                                 "view": {"line": "shout"}}))
    with pytest.raises(ConfigError, match="view"):
        main([str(p), "--runs-dir", str(tmp_path / "r")])
