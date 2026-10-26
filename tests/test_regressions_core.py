"""Regression tests: each bug found in the first adversarial code review must stay fixed."""
import json
import sys

import pytest
import yaml

from collusion.agent import Agent, Decision
from collusion.arms import Fixed, Live
from collusion.backends import MockBackend
from collusion.backends.base import Backend, Completion
from collusion.budget import Budget, BudgetExceeded
from collusion.config import load_config, validate
from collusion.game import _log_decision, play
from collusion.market import Market
from collusion.parse import parse_price
from collusion.run import Job, _agent, build_jobs, main, run_job
from collusion.runlog import REPO, RunLog, read_jsonl
from collusion.validate import validate_run

sys.path.insert(0, str(REPO / "analysis"))


def _cfg(tmp_path, name, **kw):
    base = {"stage": "t", "prompt": "E1", "rounds": 12, "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 64},
            "firm1": {"provider": "mock", "model": "mock", "strategy": "collude"},
            "firm2": {"provider": "mock", "model": "mock", "strategy": "collude"}, "parallel_games": 2}
    base.update(kw)
    if base.get("arm") in ("P", "FN", "FM"):
        base.pop("firm2", None)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(base))
    return str(p)


def _l_run(tmp_path, seed=5, **kw):
    runs = tmp_path / "runs"
    assert main([_cfg(tmp_path, f"L{seed}", arm="L", seeds=[seed], **kw), "--runs-dir", str(runs)]) == 0
    return runs, runs / f"t_L_E1_mock_s{seed}"


# ---------------------------------------------------------------- B1 parser
def test_B1_decimal_comma_is_not_parsed_as_integer():
    # {"price": 1,5} (European decimal) is swallowed by the optional ",..." tail and parsed as 1 -> clipped to 1.0
    price, status = parse_price('{"price": 1,5}')
    assert not (status == "ok" and price == 1.0), (price, status)


# ---------------------------------------------------------------- B2 budget race
def test_B2_reserve_does_not_hold_budget_for_concurrent_callers(tmp_path, monkeypatch):
    prices = tmp_path / "p.yaml"
    prices.write_text(yaml.safe_dump({"m": {"input_per_mtok": 1.0, "output_per_mtok": 10.0}}))
    b = Budget(tmp_path / "ledger", prices, session_cap_usd=100.0)
    monkeypatch.setenv("BUDGET_USD_OPENAI", "0.0165")  # 1.5 x worst case ($0.011)
    r1 = b.reserve("openai", "m", 1000, 1000)          # caller A holds $0.011
    with pytest.raises(BudgetExceeded):
        b.reserve("openai", "m", 1000, 1000)           # caller B must be refused while A is open
    b.record("openai", "m", "r", 100, 100, r1)          # A settles cheaper than worst case
    b.reserve("openai", "m", 100, 100)                  # now there is room again
    assert b.spent("openai") <= 0.0165


# ---------------------------------------------------------------- B3 paid calls lost from calls.jsonl
class _CountingMock(MockBackend):
    n = 0
    def complete(self, *a, **k):
        type(self).n += 1
        return super().complete(*a, **k)


class _FailSecondReserve(Budget):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.k = 0
    def reserve(self, *a, **k):
        self.k += 1
        if self.k == 2:
            raise BudgetExceeded("cap")


def test_B3_calls_made_before_budget_stop_are_logged(tmp_path):
    _CountingMock.n = 0
    log = RunLog(tmp_path, "r")
    agent = Agent(_CountingMock(strategy="garbage:1.0"))
    out = play(log, Market(), Live(agent), Fixed(1.5), 3, 1, "FN", _FailSecondReserve(tmp_path / "led", REPO / "configs/prices.yaml"))
    assert out["status"] == "budget_stop"
    logged = len(read_jsonl(tmp_path / "r" / "calls.jsonl"))
    assert logged == _CountingMock.n, f"{_CountingMock.n} backend call(s) made, {logged} logged"


# ---------------------------------------------------------------- B4 model mismatch check is a bare prefix test
def test_B4_model_mismatch_not_masked_by_prefix(tmp_path):
    log = RunLog(tmp_path, "m")
    call = {"attempt": 0, "parse_status": "ok", "truncated": False, "transport_error": None, "input_tokens": 1,
            "output_tokens": 1, "usd": 0.0, "raw_output": "", "model": "gpt-5", "model_returned": "gpt-5-mini",
            "parsed_price": 1.5, "finish_reason": "stop"}
    d = Decision(1.5, 1.5, "ok", False, [call])
    stats = {"calls": 0, "parse_fail_calls": 0, "fallbacks": 0, "clipped": 0, "truncated": 0, "transport_errors": 0,
             "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "decisions": 0}
    with pytest.raises(RuntimeError):   # strict=True must refuse a different model (gpt-5-mini is not gpt-5)
        _log_decision(log, stats, "L", 1, 1, 1, d, True)


# ---------------------------------------------------------------- B5-B8 validator blind spots
def test_B5_validator_flags_missing_history_at_t_gt_1(tmp_path):
    runs, d = _l_run(tmp_path)
    lines = read_jsonl(d / "calls.jsonl")
    for c in lines:
        if c["round"] == 6:
            c["user_prompt"] = "Round 6. History of the last 20 rounds (oldest first): round, your price, competitor's price, your quantity share, your profit.\nNo history yet."
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in lines))
    assert validate_run(d), "history silently dropped at round 6 and validator reports no error"


def test_B6_validator_flags_wrong_header_round_number(tmp_path):
    runs, d = _l_run(tmp_path)
    lines = read_jsonl(d / "calls.jsonl")
    for c in lines:
        if c["round"] == 6:
            c["user_prompt"] = c["user_prompt"].replace("Round 6.", "Round 7.", 1)
    (d / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in lines))
    assert validate_run(d), "prompt says Round 7 while playing round 6; validator reports no error"


def test_B7_validator_reparses_raw_output(tmp_path, monkeypatch):
    # a parser that picks the FIRST price (the 9.99 draft in <think>) instead of the last must fail validation
    import re
    import collusion.agent as ag
    def first_match(text):
        m = re.search(r'\{"price":\s*([\d.]+)\}', text or "")
        return (float(m.group(1)), "ok") if m else (None, "no_price_json")
    monkeypatch.setattr(ag, "parse_price", first_match)
    runs = tmp_path / "runs"
    assert main([_cfg(tmp_path, "L5", arm="L", seeds=[5]), "--runs-dir", str(runs)]) == 1   # run flagged unusable
    v = json.loads((runs / "t_L_E1_mock_s5/validation.json").read_text())
    assert not v["ok"] and any("re-parsing raw_output" in e for e in v["errors"])


def test_B8_validator_checks_all_summary_numbers(tmp_path):
    runs, d = _l_run(tmp_path)
    s = json.loads((d / "summary.json").read_text())
    s["mean_p1"] += 0.3
    s["profit_ratio1"] *= 2
    s["mean_abs_gap_br1"] = 0.0
    (d / "summary.json").write_text(json.dumps(s))
    assert validate_run(d), "tampered mean_p1 / profit_ratio1 / mean_abs_gap_br1 not detected (only delta1 is re-derived)"


def test_B9_validator_checks_fixed_price_equals_benchmark(tmp_path):
    cfg = load_config(_cfg(tmp_path, "FN", arm="FN", seeds=[1], firm1={"provider": "mock", "model": "mock", "strategy": "br"}))
    job = Job("fn_bad", 1, Live(_agent(cfg, 1)), Fixed(1.60), None, {"arm": "FN", "index": 0})  # 1.60 != p^N = 1.4729
    run_job(job, cfg, tmp_path / "runs", Budget(tmp_path / "led", REPO / "configs/prices.yaml"))
    assert validate_run(tmp_path / "runs/fn_bad"), "FN run with firm 2 at 1.60 (not p^N) validates clean"


# ---------------------------------------------------------------- B10 analysis trusts a stale validation.json
def test_B10_analysis_revalidates_runs(tmp_path):
    import analyze
    runs, d = _l_run(tmp_path)
    rows = read_jsonl(d / "rounds.jsonl")
    rows[-1]["p1"] = 2.0          # tamper AFTER validation.json was written
    (d / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    loaded, incomplete = analyze.load_all(runs, None)
    assert not loaded and incomplete, "tampered run is analysed because validation.json is stale"


# ---------------------------------------------------------------- B11 config validator gaps
def test_B11_arm_L_with_P_seeds_rejected():
    cfg = {"stage": "s", "arm": "L", "prompt": "E1", "rounds": 300, "seed_assignment": "configs/seed_assignment.json",
           "seeds_from": "P", "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 2048},
           "firm1": {"provider": "mock", "model": "a"}, "firm2": {"provider": "mock", "model": "a"}}
    assert validate(cfg), "arm L with seeds_from: P (breaks the L/P randomisation) is accepted"


def test_B12_D_arm_must_match_source_run_config(tmp_path):
    runs, d = _l_run(tmp_path, seed=11, rounds=30)           # source L run uses prompt E1
    cfg = load_config(_cfg(tmp_path, "D", arm="D", seeds=[11], shock_round=21, window=5, rounds=30, prompt="E0",
                           source_run_template="t_L_E1_mock_s{seed}"))
    with pytest.raises(RuntimeError, match="prompt differs"):
        build_jobs(cfg, runs)   # D with prompt E0 branching an E1 L run should be refused


def test_B13_replay_map_int_keys_normalised_on_load(tmp_path):
    p = _cfg(tmp_path, "P", arm="P", seeds=[21], replay_map={21: 11}, source_run_template="x_s{seed}",
             firm1={"provider": "mock", "model": "mock"})
    cfg = load_config(p)
    assert cfg["replay_map"] == {"21": 11}       # int YAML keys become the strings the runner looks up
    bad = {**cfg, "replay_map": {"21": "eleven"}}
    assert any("replay_map" in e for e in validate(bad))


# ---------------------------------------------------------------- B14 analysis groups
def test_B14_interaction_does_not_pool_identity_variants(tmp_path):
    import analyze
    runs = tmp_path / "runs"
    c = lambda **k: _cfg(tmp_path, k.pop("n"), rounds=60, **k)
    assert main([c(n="La", arm="L", seeds=[1]), "--runs-dir", str(runs)]) == 0
    assert main([c(n="Lb", arm="L", seeds=[2], identity1={"self_name": "Bob"}, identity_label="bob"), "--runs-dir", str(runs)]) == 0
    assert main([c(n="P", arm="P", seeds=[3], replay_map={"3": 1}, source_run_template="t_L_E1_mock_s{seed}",
                   firm1={"provider": "mock", "model": "mock", "strategy": "br"}), "--runs-dir", str(runs)]) == 0
    assert analyze.main(["--runs-dir", str(runs), "--out", str(tmp_path / "out")]) == 0
    rep = json.loads((tmp_path / "out/report.json").read_text())
    inter = [v for k, v in rep.items() if k.startswith("interaction|")]
    assert inter and all(v["n_L"] == 1 for v in inter), [v["n_L"] for v in inter]


# ---------------------------------------------------------------- B15 draft price in <think> played as a decision
def test_B15_think_draft_is_used_per_spec_but_flagged():
    """Spec 3.2: parse the LAST JSON in the full output. A draft inside <think> is therefore played (spec kept, see
    a design proposal), but every such call is flagged price_from_think=True so its rate is reported per arm."""
    class B(Backend):
        provider, model = "mock", "mock"
        def __init__(self): pass
        def complete(self, system, user, sampling, seed):
            return Completion(text='<think>maybe {"price": 1.9} or lower</think>I would charge about 1.75.',
                              finish_reason="stop", input_tokens=1, output_tokens=1, model_returned="mock")
    d = Agent(B()).decide(5, [], 1.5, 0, 1)
    assert d.price == 1.9 and d.calls[0]["price_from_think"] is True
