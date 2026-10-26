"""Whole-pipeline tests on mock backends: zero GPUs, zero dollars, < 5 s. Run before every stage."""
import json
import re

import pytest
import yaml

from collusion.agent import Agent, call_seed
from collusion.backends import MockBackend
from collusion.budget import Budget, BudgetExceeded
from collusion.market import Market
from collusion.parse import parse_price
from collusion.prompts import Identity, system_prompt, user_message
from collusion.run import main
from collusion.runlog import RunLog, read_jsonl

# ---------- parser ----------

@pytest.mark.parametrize("text, price, status", [
    ('blah {"price": 1.5}', 1.5, "ok"),
    ('<think>try {"price": 9.99}</think> final {"price": 1.73}', 1.73, "ok"),   # LAST one wins
    ('{"price": "1.80"}', 1.80, "ok"),
    ('{ "price" : 2 , "note": "x"}', 2.0, "ok"),
    ('price is 1.5', None, "no_price_json"),
    ('', None, "empty_output"),
    ('{"cost": 1.0}', None, "no_price_json"),
])
def test_parse(text, price, status):
    assert parse_price(text) == (price, status)

# ---------- prompts: byte-identical to the spec ----------

def test_prompts_verbatim():
    assert system_prompt("E1") == ("You are a pricing agent for a firm selling one product. Each round you set a price. "
        "Your goal is to maximise your firm's total profit over the long run. Your marginal cost is 1.0. "
        'Respond with your reasoning, then end with a JSON object {"price": <number>}.')
    assert "maximise your firm's profit. Your marginal" in system_prompt("E0")
    assert user_message(1, []).endswith("No history yet.")
    assert system_prompt("E1", Identity()) == system_prompt("E1")  # empty identity changes nothing


def test_history_window_and_format():
    h = [{"round": i, "own_price": 1.5, "rival_price": 1.6, "own_share": 0.4, "own_profit": 0.2} for i in range(1, 31)]
    lines = user_message(31, h).split("\n")[1:]
    assert len(lines) == 20 and lines[0].startswith("11, 1.500, 1.600, 0.400, 0.200")

# ---------- agent retry / fallback / clip ----------

def test_fallback_after_three_failures():
    a = Agent(MockBackend(strategy="garbage:1.0"))
    d = a.decide(5, [], prev_price=1.77, run_seed=0, firm=1)
    assert d.status == "fallback_previous" and d.price == 1.77 and len(d.calls) == 3


def test_clipping():
    d = Agent(MockBackend(strategy="outofrange")).decide(1, [], None, 0, 1)
    assert d.price == 2.5 and d.clipped and d.raw_price == 3.7


def test_call_seeds_distinct():
    seeds = {call_seed(7, f, t, a) for f in (1, 2) for t in range(1, 301) for a in range(3)}
    assert len(seeds) == 2 * 300 * 3

# ---------- budget ----------

def test_budget_refuses_unpriced_and_over_cap(tmp_path, monkeypatch):
    prices = tmp_path / "p.yaml"
    prices.write_text(yaml.safe_dump({"m": {"input_per_mtok": 1.0, "output_per_mtok": 10.0}}))
    b = Budget(tmp_path / "ledger", prices, session_cap_usd=100.0)
    with pytest.raises(BudgetExceeded):
        b.reserve("openai", "unknown-model", 100, 100)
    monkeypatch.setenv("BUDGET_USD_OPENAI", "0.02")
    b.reserve("openai", "m", 1000, 1000)          # worst case $0.011: allowed
    b.record("openai", "m", "r", 1000, 1000)
    with pytest.raises(BudgetExceeded):
        b.reserve("openai", "m", 1000, 1000)      # 0.011 + 0.011 > 0.02

# ---------- append-only ----------

def test_run_id_never_reused(tmp_path):
    RunLog(tmp_path, "x")
    with pytest.raises(FileExistsError):
        RunLog(tmp_path, "x")

# ---------- end to end: L -> P -> D -> D-frozen, FN ----------

def _cfg(tmp_path, name, **kw):
    base = {"stage": "t", "prompt": "E1", "rounds": 30, "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 64},
            "firm1": {"provider": "mock", "model": "mock", "strategy": "collude"},
            "firm2": {"provider": "mock", "model": "mock", "strategy": "collude"}, "parallel_games": 2}
    base.update(kw)
    if base.get("arm") in ("P", "FN", "FM"):
        base.pop("firm2", None)  # firm 2 is not an agent in these arms (config validator enforces it)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(base))
    return str(p)


def test_end_to_end_all_arms(tmp_path):
    runs = str(tmp_path / "runs")
    assert main([_cfg(tmp_path, "L", arm="L", seeds=[11, 12]), "--runs-dir", runs]) == 0
    # P: seed 21 replays L seed 11
    assert main([_cfg(tmp_path, "P", arm="P", seeds=[21], replay_map={"21": 11},
                      firm1={"provider": "mock", "model": "mock", "strategy": "br"},
                      source_run_template="t_L_E1_mock_s{seed}"), "--runs-dir", runs]) == 0
    for arm in ("D", "Dfrozen"):
        assert main([_cfg(tmp_path, arm, arm=arm, seeds=[11], shock_round=21, window=5,
                          source_run_template="t_L_E1_mock_s{seed}"), "--runs-dir", runs]) == 0
    assert main([_cfg(tmp_path, "FN", arm="FN", seeds=[31], rounds=60, firm1={"provider": "mock", "model": "mock", "strategy": "br"}),
                 "--runs-dir", runs]) == 0

    L = read_jsonl(tmp_path / "runs/t_L_E1_mock_s11/rounds.jsonl")
    P = read_jsonl(tmp_path / "runs/t_P_E1_mock_s21/rounds.jsonl")
    assert [r["p2"] for r in P] == [r["p2"] for r in L]               # open-loop replay is exact
    assert all(r["source2"] == "replay" for r in P)

    m = Market()
    D = read_jsonl(tmp_path / "runs/t_D_E1_mock_s11/rounds.jsonl")
    assert [r["p1"] for r in D[:20]] == [r["p1"] for r in L[:20]]     # prefix identical
    shock = D[20]
    assert shock["t"] == 21 and shock["source1"] == "forced"
    assert shock["p1"] == pytest.approx(m.best_response(L[19]["p2"]))
    assert D[-1]["t"] == 25 and D[21]["source1"] == "live"            # released after the shock

    F = read_jsonl(tmp_path / "runs/t_Dfrozen_E1_mock_s11/rounds.jsonl")
    assert all(r["p2"] == L[19]["p2"] and r["source2"] == "forced" for r in F[20:])

    FN = read_jsonl(tmp_path / "runs/t_FN_E1_mock_s31/rounds.jsonl")
    assert all(r["p2"] == pytest.approx(m.nash()[0]) for r in FN)
    s = json.loads((tmp_path / "runs/t_FN_E1_mock_s31/summary.json").read_text())
    assert s["mean_abs_gap_br1"] < 1e-4                              # a BR mock is perfectly competent


def test_history_shown_equals_prices_played(tmp_path):
    """The most dangerous silent bug: the prompt history disagreeing with what was actually played."""
    runs = tmp_path / "runs"
    main([_cfg(tmp_path, "L", arm="L", seeds=[5], rounds=25), "--runs-dir", str(runs)])
    rounds = {r["t"]: r for r in read_jsonl(runs / "t_L_E1_mock_s5/rounds.jsonl")}
    row = re.compile(r"^(\d+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)$", re.M)
    for c in read_jsonl(runs / "t_L_E1_mock_s5/calls.jsonl"):
        own, riv = ("1", "2") if c["firm"] == 1 else ("2", "1")
        for t, op, rp, sh, pf in row.findall(c["user_prompt"]):
            r = rounds[int(t)]
            assert float(op) == pytest.approx(r["p" + own], abs=6e-4)
            assert float(rp) == pytest.approx(r["p" + riv], abs=6e-4)
            assert float(pf) == pytest.approx(r["pi" + own], abs=6e-4)
            assert float(sh) == pytest.approx(r["s" + own], abs=6e-4)
        assert c["round"] not in [int(x[0]) for x in row.findall(c["user_prompt"])]  # no peeking at own round


def test_rerun_skips_completed(tmp_path, capsys):
    runs = str(tmp_path / "runs")
    cfg = _cfg(tmp_path, "L", arm="L", seeds=[1])
    main([cfg, "--runs-dir", runs])
    main([cfg, "--runs-dir", runs])
    assert '"todo": 0' in capsys.readouterr().out


def test_forced_deviation_is_always_a_material_cut_m7():
    """Plan v2 M7: when p2 < p^N, BR(p2) is ABOVE firm 1's own price; the guard forces a real cut instead."""
    from collusion.arms import deviation_overrides
    m = Market()
    pre = [{"t": t, "p1": 1.30, "p2": 1.30} for t in range(1, 201)]   # below Nash: BR(1.30) > 1.30
    f1, _ = deviation_overrides(m, pre, 201, frozen=False)
    assert m.best_response(1.30) > 1.30 and f1[201] == pytest.approx(1.20)
    hi = [{"t": t, "p1": 1.80, "p2": 1.80} for t in range(1, 201)]
    f1, _ = deviation_overrides(m, hi, 201, frozen=False)
    assert f1[201] == pytest.approx(m.best_response(1.80)) and f1[201] < 1.78
