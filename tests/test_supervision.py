"""Supervision arms (docs/SUPERVISION.md): monitoring line in the system prompt × real supervisor agent."""
import json

import pytest
import yaml

from collusion.config import ConfigError
from collusion.prompts import SUPERVISION, system_prompt, user_message
from collusion.run import main
from collusion.runlog import read_jsonl
from collusion.supervisor import parse_note, supervisor_system
from collusion.validate import validate_run
from test_opening import V2, _controls_setup


def test_prompt_lines_and_spec_unchanged_when_off():
    assert system_prompt("E1") == system_prompt("E1", supervision=None)
    assert system_prompt("E1", supervision="shared").endswith(SUPERVISION["shared"])
    assert "you and your competitor" in SUPERVISION["shared"] and "competitor" not in SUPERVISION["individual"]
    h = [{"round": 1, "own_price": 1.5, "rival_price": 1.6, "own_share": 0.4, "own_profit": 0.2}]
    assert user_message(2, h) == user_message(2, h, note=None)
    assert user_message(2, h, note="Watching.").endswith("\n1, 1.500, 1.600, 0.400, 0.200\nSupervisor's note: Watching.")


def test_parse_note_and_individual_wording():
    assert parse_note('<think>{"note": "x"}</think>ok {"note": "Prices  are\\nhigh."}') == "Prices are high."
    assert parse_note("no json") is None and parse_note('{"note": ""}') is None
    assert len(parse_note('{"note": "' + "w " * 60 + '"}').split()) == 40
    s = supervisor_system("individual", 2)
    assert "to Firm 2 only" in s and "do not say whether the other firm is supervised" in s
    assert "both firms will read" in supervisor_system("shared")


@pytest.mark.parametrize("frame, agent", [("shared", True), ("shared", False), ("individual", True), ("individual", False)])
def test_four_arms_end_to_end(tmp_path, frame, agent):
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": "averager"}
    p = tmp_path / "O.yaml"
    p.write_text(yaml.safe_dump({**V2, "arm": "O", "opening": [1.95, 1.95], "seeds": [1], "firm1": f, "firm2": f,
                                 "supervision": {"frame": frame, "agent": agent}}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    tag = f"_sup{frame[0].upper()}{'A' if agent else 'N'}"
    (d,) = [x for x in runs.iterdir() if tag in x.name]
    assert validate_run(d) == []
    C = read_jsonl(d / "calls.jsonl")
    assert all(SUPERVISION[frame] in c["system_prompt"] for c in C)
    sp = d / "supervisor.jsonl"
    if not agent:
        assert not sp.exists() and not any("Supervisor's note" in c["user_prompt"] for c in C)
        return
    S = read_jsonl(sp)
    assert {c["to"] for c in S} == ({"both"} if frame == "shared" else {1, 2})
    R = {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}
    assert any(r["note1"] for r in R.values() if not r["is_prefix"])
    if frame == "shared":
        assert all(r["note1"] == r["note2"] for r in R.values() if not r["is_prefix"])
    # a tampered note is caught
    rows = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines()]
    k = next(i for i, r in enumerate(rows) if r.get("note1"))
    rows[k]["note1"] = "forged"
    (d / "rounds.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert any("note" in e for e in validate_run(d))


def test_config_rejects_bad_supervision(tmp_path):
    f = {"provider": "mock", "model": "mock", "strategy": "averager"}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({**V2, "arm": "O", "opening": [1.95, 1.95], "seeds": [1], "firm1": f, "firm2": f,
                                 "supervision": {"frame": "everyone", "agent": True}}))
    with pytest.raises(ConfigError, match="supervision"):
        main([str(p), "--runs-dir", str(tmp_path / "r")])
