"""Pressure arms (docs/PRESSURE.md): a personal-financial pressure line, gain vs loss, on both firms or firm 1 only."""
import pytest
import yaml

from collusion.config import ConfigError
from collusion.prompts import PRESSURE, system_prompt
from collusion.run import main
from collusion.runlog import read_jsonl
from collusion.validate import validate_run
from test_opening import V2, _controls_setup


def test_lines_off_by_default_and_matched():
    assert system_prompt("E1") == system_prompt("E1", pressure=None)
    for k in ("gain", "loss"):
        assert system_prompt("E1", pressure=k).endswith(PRESSURE[k])
    g, ls = PRESSURE["gain"].split(), PRESSURE["loss"].split()
    assert abs(len(g) - len(ls)) <= 4 and "family" in PRESSURE["gain"] and "family" in PRESSURE["loss"]
    both = system_prompt("E1", supervision="individual", pressure="loss")
    assert both.index(PRESSURE["loss"]) < both.index("monitoring")


@pytest.mark.parametrize("frame, target", [("gain", "both"), ("loss", "both"), ("gain", "firm1"), ("loss", "firm1")])
def test_four_cells_end_to_end(tmp_path, frame, target):
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": "averager"}
    p = tmp_path / "O.yaml"
    p.write_text(yaml.safe_dump({**V2, "arm": "O", "opening": [1.55, 1.55], "seeds": [1], "firm1": f, "firm2": f,
                                 "pressure": {"frame": frame, "target": target}}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    tag = f"_pr{frame[0].upper()}{'2' if target == 'both' else '1'}"
    (d,) = [x for x in runs.iterdir() if tag in x.name]
    assert validate_run(d) == []
    C = read_jsonl(d / "calls.jsonl")
    for c in C:
        assert (PRESSURE[frame] in c["system_prompt"]) == (target == "both" or c["firm"] == 1)
    # a firm-2 call carrying the line would be caught
    lines = (d / "calls.jsonl").read_text().splitlines()
    import json
    rows = [json.loads(x) for x in lines]
    k = next(i for i, c in enumerate(rows) if c["firm"] == 2)
    rows[k]["system_prompt"] = (rows[k]["system_prompt"].replace(" " + PRESSURE[frame], "") if target == "both"
                                else rows[k]["system_prompt"] + " " + PRESSURE[frame])
    (d / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert any("pressure" in e for e in validate_run(d))


def test_bad_pressure_rejected(tmp_path):
    f = {"provider": "mock", "model": "mock", "strategy": "averager"}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({**V2, "arm": "O", "opening": [1.55, 1.55], "seeds": [1], "firm1": f, "firm2": f,
                                 "pressure": {"frame": "panic", "target": "both"}}))
    with pytest.raises(ConfigError, match="pressure"):
        main([str(p), "--runs-dir", str(tmp_path / "r")])
