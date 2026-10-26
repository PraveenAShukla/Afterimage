"""Resume after a time-limit kill: continue in place, append-only, validator passes, design changes refused."""
import json

import pytest
import yaml

from collusion.run import main
from collusion.runlog import read_jsonl
from collusion.validate import validate_run

CFG = {"stage": "t", "arm": "L", "prompt": "E1", "rounds": 12, "seeds": [1],
       "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 64},
       "firm1": {"provider": "mock", "model": "mock", "strategy": "collude"},
       "firm2": {"provider": "mock", "model": "mock", "strategy": "collude"}}


def _cfg(tmp_path, **kw):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({**CFG, **kw}))
    return str(p)


def _kill_after(d, keep_rounds):
    """Simulate a job killed during round keep_rounds+1: rounds after it never written, calls for it already logged."""
    rounds = (d / "rounds.jsonl").read_text().splitlines()[:keep_rounds]
    (d / "rounds.jsonl").write_text("\n".join(rounds) + "\n")
    calls = [c for c in read_jsonl(d / "calls.jsonl") if c["round"] <= keep_rounds + 1]
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (d / "summary.json").unlink()
    (d / "validation.json").unlink()


def test_resume_continues_and_validates(tmp_path):
    runs = tmp_path / "runs"
    full = tmp_path / "full"
    assert main([_cfg(tmp_path), "--runs-dir", str(full)]) == 0          # reference uninterrupted run
    assert main([_cfg(tmp_path), "--runs-dir", str(runs)]) == 0
    d = runs / "t_L_E1_mock_s1"
    _kill_after(d, 7)
    before = (d / "calls.jsonl").read_text()
    assert main([_cfg(tmp_path), "--runs-dir", str(runs)]) == 1           # without --resume: left untouched
    assert main([_cfg(tmp_path), "--runs-dir", str(runs), "--resume"]) == 0
    assert (d / "calls.jsonl").read_text().startswith(before)               # append-only: old lines kept verbatim
    assert [r["t"] for r in read_jsonl(d / "rounds.jsonl")] == list(range(1, 13))
    assert validate_run(d) == []
    assert any(e["event"] == "resume" and e["from_round"] == 8 for e in read_jsonl(d / "events.jsonl"))
    ref = [(r["p1"], r["p2"]) for r in read_jsonl(full / "t_L_E1_mock_s1/rounds.jsonl")]
    assert [(r["p1"], r["p2"]) for r in read_jsonl(d / "rounds.jsonl")] == ref   # deterministic seeds: same game


def test_resume_refuses_changed_design(tmp_path):
    runs = tmp_path / "runs"
    main([_cfg(tmp_path), "--runs-dir", str(runs)])
    _kill_after(runs / "t_L_E1_mock_s1", 5)
    with pytest.raises(RuntimeError, match="different design"):
        main([_cfg(tmp_path, sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 64}), "--runs-dir", str(runs), "--resume"])
