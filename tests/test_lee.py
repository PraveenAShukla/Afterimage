"""Lee replication package (src/lee): market, Figure-1 prompts, parser, mock end-to-end, randomised openings."""
import json

import pytest
import yaml

from lee.backend import ChatBackend
from lee.market import NMarket
from lee.parse import parse_reply
from lee.prompts import system_prompt, user_message
from lee.run import main
from lee.validate import validate_run
from collusion.runlog import read_jsonl


def test_market_matches_lee_table1():
    b = NMarket().benchmarks()
    assert b["pN"] == pytest.approx(1.33, abs=0.005) and b["piN"] == pytest.approx(0.0815, abs=5e-5)
    assert b["pM"] == pytest.approx(2.05, abs=0.005) and b["piM"] == pytest.approx(0.2011, abs=5e-5)
    m = NMarket()
    assert m.quantities([b["pN"]] * 4)[1] == pytest.approx(0.017, abs=5e-4)
    assert m.best_response(0, [b["pN"]] * 4) == pytest.approx(b["pN"], abs=1e-9)


def test_figure1_prompts():
    s = system_prompt(0)
    assert s.startswith("You are the pricing manager of Firm A. Your goal is to maximize the firm’s cumulative profit")
    assert "Marginal cost is 1.00 per unit. Prices must be between 0.00 and 5.00." in s
    assert "(round | prices of firms A, B, C, D | your quantity | your profit)" in s
    assert s.endswith('"insight": "<at most 40 words: what you learned>"}')
    u1 = user_message(1, None, None, [], 0)
    assert u1 == "Round 1.\nPrevious plan: (none)\nPrevious insight: (none)\nHistory (last 20 rounds):\n(no history yet)"
    h = [{"round": 7, "prices": [2.15, 2.1, 2.2, 2.1], "q": [0.18, 0.2, 0.17, 0.2], "pi": [0.21, 0.22, 0.2, 0.22]}]
    assert user_message(8, "Hold", "Stable", h, 0).endswith("7 | 2.15 2.10 2.20 2.10 | 0.18 | 0.21")


def test_parser():
    assert parse_reply('x {"price": 9, "plan": "a"} y {"price": 2.1, "plan": "p", "insight": "i"}')["price"] == 2.1
    r = parse_reply('{"price": 7.5, "plan": "p", "insight": "i"}')
    assert r["price"] == 5.0 and r["clipped"]
    assert parse_reply('blah "price": 1.9 truncated')["status"] == "price_only"
    assert parse_reply("")["status"] == "empty" and parse_reply("no json")["status"] == "no_price"


def test_backend_refuses_temperature_and_missing_key(monkeypatch):
    with pytest.raises(ValueError):
        ChatBackend("openai", "gpt-5-nano", {"temperature": 0.7})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    c = ChatBackend("deepseek", "deepseek-flash").complete("s", "u")
    assert c.error and "DEEPSEEK_API_KEY" in c.error


def _cfg(tmp_path, name, **kw):
    c = {"stage": "lt", "phase": "rep", "cell": "mockavg", "firm": {"provider": "mock", "model": "mock", "strategy": "avg"},
         "n": 4, "rounds": 60, "seeds": [1, 2], "max_output_tokens": 200, "parallel_games": 2}
    c.update(kw)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(c))
    return str(p)


def test_end_to_end_cold_start_and_randomised_opening(tmp_path):
    runs = tmp_path / "runs"
    assert main([_cfg(tmp_path, "cold"), "--runs-dir", str(runs)]) == 0
    cold = sorted(runs.glob("lt_rep_mockavg_s*"))
    assert len(cold) == 2
    for d in cold:
        assert validate_run(d) == []
        s = json.loads((d / "summary.json").read_text())
        assert s["status"] == "complete" and s["n_rounds_played"] == 60 and "delta" in s
    op = {"x": 1.80, "source_template": "lt_rep_mockavg_s{seed}", "pool": [1, 2], "perm_seed": 7, "window": [11, 30]}
    assert main([_cfg(tmp_path, "open", phase="disc", opening=op, seeds=[5, 6]), "--runs-dir", str(runs)]) == 0
    ds = sorted(runs.glob("lt_disc_mockavg_x1.800_s*"))
    assert len(ds) == 2
    for d in ds:
        assert validate_run(d) == []
        R = read_jsonl(d / "rounds.jsonl")
        pre = [r for r in R if r["is_prefix"]]
        assert len(pre) == 20 and sum(sum(r["prices"]) for r in pre) / 80 == pytest.approx(1.80)
        assert all(abs(p - 1.80) < 1e-6 for p in R[-1]["prices"])  # the averager keeps the opening level
        c0 = read_jsonl(d / "calls.jsonl")[0]
        assert c0["round"] == 21 and "Previous plan: (none)" in c0["user_prompt"] and "\n1 | " in c0["user_prompt"]
    # a tampered opening is caught
    man = json.loads((ds[0] / "manifest.json").read_text())
    man["opening"]["sha256"] = "0" * 64
    (ds[0] / "manifest.json").write_text(json.dumps(man))
    assert any("opening" in e for e in validate_run(ds[0]))


def test_paid_cell_refused_without_approval(tmp_path):
    p = _cfg(tmp_path, "paid", firm={"provider": "openai", "model": "gpt-5-nano", "extra": {"reasoning_effort": "minimal"}})
    assert main([p, "--runs-dir", str(tmp_path / "runs")]) == 2
