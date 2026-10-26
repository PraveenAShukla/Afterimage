"""Every safety check must be shown to FIRE. A check that never triggers in a test is assumed broken."""
import json
import socket

import pytest
import yaml

from collusion.config import ConfigError, load_config, validate
from collusion.health import HealthMonitor, RunAborted
from collusion.run import main
from collusion.runlog import read_jsonl
from collusion.validate import validate_run

BASE = {"stage": "t", "arm": "L", "prompt": "E1", "rounds": 30, "seeds": [1],
        "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 64},
        "firm1": {"provider": "mock", "model": "mock", "strategy": "collude"},
        "firm2": {"provider": "mock", "model": "mock", "strategy": "collude"}}


def _write(tmp_path, cfg, name="c"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)

# ---------- config ----------

def test_valid_base_config():
    assert validate(dict(BASE)) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda c: c["sampling"].update(temprature=0.0), "unknown key 'temprature'"),
    (lambda c: c.update(roundz=5), "unknown key 'roundz'"),
    (lambda c: c["sampling"].pop("max_tokens"), "missing 'max_tokens'"),
    (lambda c: c["sampling"].update(temperature=5), "outside [0, 2]"),
    (lambda c: c.update(arm="Z"), "not in"),  # Q became a real arm (plan v2)
    (lambda c: c.update(rounds="300"), "expected"),
    (lambda c: c.update(seeds=[1, 1]), "duplicates"),
    (lambda c: c["firm2"].update(model="other"), "SAME model"),
    (lambda c: c.update(arm="D"), "source_run_template"),
    (lambda c: c.update(arm="FN"), "firm2 is not an agent"),
    (lambda c: c["firm1"].update(provider="anthropic"), "est_output_tokens"),
])
def test_config_errors_fire(mutate, needle):
    c = json.loads(json.dumps(BASE))
    mutate(c)
    errs = validate(c)
    assert any(needle in e for e in errs), errs


def test_load_config_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, {**BASE, "typo_key": 1}))

# ---------- health ----------

def test_health_aborts_on_fallbacks():
    m = HealthMonitor({"window": 5, "max_fallback_rate": 0.1})
    fake_call = [{"truncated": False, "transport_error": None}]
    with pytest.raises(RunAborted):
        for t in range(1, 10):
            m.observe_decision("fallback_previous", fake_call * 3)
            m.check(t, 1.5, 1.5)


def test_health_aborts_on_transport_errors():
    m = HealthMonitor({"max_consecutive_transport_errors": 3})
    with pytest.raises(RunAborted):
        for t in range(1, 5):
            m.observe_decision("fallback_previous", [{"truncated": False, "transport_error": "HTTP 503"}])
            m.check(t, 1.5, 1.5)


def test_health_warns_on_frozen_prices():
    m = HealthMonitor({"frozen_price_rounds": 4})
    warns = [w for t in range(1, 6) for w in m.check(t, 1.8, 1.8)]
    assert len(warns) == 1 and "identical" in warns[0]


def test_garbage_model_run_is_aborted_and_excluded(tmp_path):
    """End to end: a model that mostly emits no JSON must abort, not produce a 'collusive' Δ."""
    cfg = {**BASE, "firm1": {"provider": "mock", "model": "mock", "strategy": "garbage:0.9"},
           "firm2": {"provider": "mock", "model": "mock", "strategy": "garbage:0.9"}}
    runs = tmp_path / "runs"
    assert main([_write(tmp_path, cfg), "--runs-dir", str(runs)]) == 1   # nonzero exit: run not usable
    s = json.loads((runs / "t_L_E1_mock_s1/summary.json").read_text())
    assert s["status"] == "aborted" and "fallback rate" in s["stopped_early"]

# ---------- validator ----------

def _good_run(tmp_path):
    runs = tmp_path / "runs"
    assert main([_write(tmp_path, BASE), "--runs-dir", str(runs)]) == 0
    return runs / "t_L_E1_mock_s1"


def test_validator_passes_clean_run(tmp_path):
    d = _good_run(tmp_path)
    assert validate_run(d) == []
    assert json.loads((d / "validation.json").read_text())["ok"]


@pytest.mark.parametrize("tamper, needle", [
    (lambda rs, cs: rs[5].update(pi1=rs[5]["pi1"] + 0.01), "pi1 logged"),
    (lambda rs, cs: rs.pop(7), "not contiguous"),
    (lambda rs, cs: cs[10].update(final_price=1.111), "final_price"),
    (lambda rs, cs: cs[12].update(user_prompt=cs[12]["user_prompt"].replace(", 1.", ", 2.", 1)), "disagrees"),
    (lambda rs, cs: cs[3].pop("raw_output"), "missing fields"),
])
def test_validator_catches_tampering(tmp_path, tamper, needle):
    d = _good_run(tmp_path)
    rs, cs = read_jsonl(d / "rounds.jsonl"), read_jsonl(d / "calls.jsonl")
    tamper(rs, cs)
    (d / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cs))
    errs = validate_run(d)
    assert any(needle in e for e in errs), errs[:5]

# ---------- privacy ----------

def test_manifest_has_no_hostname(tmp_path):
    d = _good_run(tmp_path)
    text = (d / "manifest.json").read_text()
    assert socket.gethostname() not in text
    assert "host_hash" in json.loads(text)["env"]
