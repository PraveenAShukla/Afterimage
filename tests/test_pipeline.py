"""Pipeline: full battery on a mock that passes both gates; gate failure stops early; rerun is idempotent."""
import json

import yaml

import collusion.pipeline as pl
from collusion.pipeline import main


def _spec(tmp_path, monkeypatch, name, strategy, battery="full"):
    monkeypatch.setattr(pl, "REPO", tmp_path)                     # seed files go under tmp, not the repo
    (tmp_path / "configs" / "seeds").mkdir(parents=True, exist_ok=True)
    import collusion.run as run
    monkeypatch.setattr(run, "REPO", tmp_path)
    (tmp_path / "configs" / "prices.yaml").write_text("{}\n")
    spec = {"name": name, "seed_master": 4242, "battery": battery, "rounds": 215, "parallel_games": 4,
            "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 64},
            "firm": {"provider": "mock", "model": "mock", "strategy": strategy},
            "n": {"stage1": 4, "stage2": 4, "stage3": 4}}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(spec))
    return str(p)


def _events(runs, name):
    return [json.loads(l) for l in open(runs / "_pipeline" / name / "pipeline.jsonl")]


def test_full_battery_runs_every_stage(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    assert main([_spec(tmp_path, monkeypatch, "mocksmart", "smart"), "--runs-dir", str(runs)]) == 0
    ev = _events(runs, "mocksmart")
    gates = {e["gate"]: e["passed"] for e in ev if e["event"] == "gate"}
    assert gates == {"collusion": True, "competence": True}
    labels = [e["label"] for e in ev if e["event"] == "stage_end"]
    assert labels == ["s1_L", "s2_FN", "s2_FM", "s3_L", "s3_P", "s3_D", "s3_Dfrozen", "s3_Dplacebo"]
    assert all(e["complete_valid"] == 4 for e in ev if e["event"] == "stage_end")
    placebo = next(runs.glob("s3-mocksmart_Dplacebo_*"))
    rows = [json.loads(l) for l in open(placebo / "rounds.jsonl")]
    assert all(r["source1"] in ("prefix", "live") for r in rows)       # placebo: no forced price anywhere


def test_gate_failure_stops_and_rerun_is_idempotent(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    spec = _spec(tmp_path, monkeypatch, "mockbr", "br")                  # best-responders: Δ ≈ 0 -> fails Stage 1
    assert main([spec, "--runs-dir", str(runs)]) == 0
    ev = _events(runs, "mockbr")
    assert [e for e in ev if e["event"] == "gate"][0]["passed"] is False
    assert "Stage 1 gate" in ev[-1]["result"]
    n_dirs = len(list(runs.glob("s1-mockbr_*")))
    assert main([spec, "--runs-dir", str(runs)]) == 0                   # rerun: skips completed runs, same decision
    assert len(list(runs.glob("s1-mockbr_*"))) == n_dirs


def test_headline_battery(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    assert main([_spec(tmp_path, monkeypatch, "mockhl", "collude", battery="headline"), "--runs-dir", str(runs)]) == 0
    labels = [e["label"] for e in _events(runs, "mockhl") if e["event"] == "stage_end"]
    assert labels == ["s1_L", "s1_P"]


def test_competence_battery(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    assert main([_spec(tmp_path, monkeypatch, "mockcomp", "br", battery="competence"), "--runs-dir", str(runs)]) == 0
    ev = _events(runs, "mockcomp")
    assert [e["label"] for e in ev if e["event"] == "stage_end"] == ["s2_FN", "s2_FM"]
    assert [e for e in ev if e["event"] == "gate"][0]["passed"] is True
