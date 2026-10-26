"""Every spend guard must be shown to fire: no approval, approval cap, 10-minute tripwire, kill switch, rate limit."""
import pytest
import yaml

from collusion.budget import Budget, BudgetExceeded
from collusion.monitor import main as monitor_main
from collusion.ratelimit import Limiter, RateLimitConfigError
from collusion.run import main as run_main


def _budget(tmp_path, monkeypatch, cap=None, provider_cap="50", per10="50"):
    prices = tmp_path / "p.yaml"
    prices.write_text(yaml.safe_dump({"m": {"input_per_mtok": 1.0, "output_per_mtok": 10.0}}))
    monkeypatch.setenv("BUDGET_USD_OPENAI", provider_cap)
    monkeypatch.setenv("MAX_USD_PER_10MIN", per10)
    return Budget(tmp_path / "runs" / "_ledger", prices, session_cap_usd=cap)


def test_paid_call_refused_without_approval(tmp_path, monkeypatch):
    with pytest.raises(BudgetExceeded, match="approved spend"):
        _budget(tmp_path, monkeypatch, cap=None).reserve("openai", "m", 1000, 1000)


def test_session_approval_cap(tmp_path, monkeypatch):
    b = _budget(tmp_path, monkeypatch, cap=0.02)
    r = b.reserve("openai", "m", 1000, 1000)          # worst $0.011
    b.record("openai", "m", "x", 1000, 1000, r)
    with pytest.raises(BudgetExceeded, match="approved spend for this run"):
        b.reserve("openai", "m", 1000, 1000)          # 0.011 + 0.011 > 0.02


def test_ten_minute_tripwire(tmp_path, monkeypatch):
    b = _budget(tmp_path, monkeypatch, cap=10, per10="0.015")
    b.record("openai", "m", "x", 1000, 1000, b.reserve("openai", "m", 1000, 1000))
    with pytest.raises(BudgetExceeded, match="tripwire"):
        b.reserve("openai", "m", 1000, 1000)


def test_kill_switch_blocks_paid_calls(tmp_path, monkeypatch):
    b = _budget(tmp_path, monkeypatch, cap=10)
    assert monitor_main(["--runs-dir", str(tmp_path / "runs"), "--stop"]) == 0
    with pytest.raises(BudgetExceeded, match="kill switch"):
        b.reserve("openai", "m", 10, 10)
    monitor_main(["--runs-dir", str(tmp_path / "runs"), "--resume"])
    b.reserve("openai", "m", 10, 10)


def test_kill_switch_aborts_running_free_game(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "STOP").write_text("x")
    cfg = {"stage": "t", "arm": "L", "prompt": "E1", "rounds": 5, "seeds": [1],
           "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 64},
           "firm1": {"provider": "mock", "model": "mock", "strategy": "br"},
           "firm2": {"provider": "mock", "model": "mock", "strategy": "br"}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert run_main([str(p), "--runs-dir", str(runs)]) == 1
    assert "kill switch" in (runs / "t_L_E1_mock_s1/summary.json").read_text()


def test_paid_run_refused_without_approve_flag(tmp_path, capsys, monkeypatch):
    from collusion.backends.http import DEFAULT_ANTHROPIC_KEY_ENV
    monkeypatch.setenv(DEFAULT_ANTHROPIC_KEY_ENV, "test-dummy-not-a-key")   # refused before any call; no real key needed
    cfg = {"stage": "t", "arm": "FN", "prompt": "E1", "rounds": 5, "seeds": [1], "est_output_tokens": 500,
           "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 1024},
           "firm1": {"provider": "anthropic", "model": "claude-haiku-5-5"}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert run_main([str(p), "--runs-dir", str(tmp_path / "runs")]) == 2
    out = capsys.readouterr()
    assert "REFUSED" in out.err and "usd_expected" in out.out and "usd_worst_no_retries" in out.out


def test_rate_limiter_blocks_until_window_frees():
    t = [0.0]
    slept = []
    lim = Limiter("openai", rpm=2, tpm=10_000, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    lim.acquire(100)
    lim.acquire(100)
    lim.acquire(100)                                  # third request in one minute must wait ~60 s
    assert sum(slept) >= 59.9


def test_rate_limiter_token_budget_and_oversize():
    t = [0.0]
    lim = Limiter("openai", rpm=100, tpm=1000, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    ev = lim.acquire(800)
    lim.settle(ev, 100)                               # actual usage frees room
    lim.acquire(800)
    with pytest.raises(RateLimitConfigError):
        lim.acquire(5000)


def test_request_quota_guard(tmp_path, monkeypatch):
    b = _budget(tmp_path, monkeypatch, cap=10)
    monkeypatch.setenv("MAX_REQUESTS_OPENAI", "2")
    for _ in range(2):
        b.record("openai", "m", "x", 10, 10, b.reserve("openai", "m", 10, 10))
    with pytest.raises(BudgetExceeded, match="quota guard"):
        b.reserve("openai", "m", 10, 10)


def test_min_interval_spaces_call_starts():
    from collusion.ratelimit import Limiter
    t = [0.0]

    def sl(d):
        t[0] += d
    lim = Limiter("x", 1000, 10**9, clock=lambda: t[0], sleep=sl, min_interval_s=0.4)
    starts = []
    for _ in range(4):
        lim.acquire(10)
        starts.append(round(t[0], 2))
    assert starts == [0.0, 0.4, 0.8, 1.2]


def test_incremental_ledger_sees_other_processes_and_stays_fast(tmp_path, monkeypatch):
    """Two Budget objects on one ledger (like two processes): each must count the other's spend toward the provider
    cap after incremental reading, and a reserve on a large ledger must not re-parse everything."""
    import json
    import time
    from collusion.budget import Budget, BudgetExceeded
    prices = tmp_path / "p.yaml"
    prices.write_text("m: {input_per_mtok: 1000000, output_per_mtok: 0}\n")   # $1 per input token: easy arithmetic
    monkeypatch.setenv("BUDGET_USD_OPENAI", "5")
    monkeypatch.setenv("MAX_USD_PER_10MIN", "1000")
    led = tmp_path / "_ledger"
    a, b = Budget(led, prices, session_cap_usd=100), Budget(led, prices, session_cap_usd=100)
    ra = a.reserve("openai", "m", 2, 0)
    a.record("openai", "m", "r", 2, 0, ra)                 # A spent $2
    rb = b.reserve("openai", "m", 2, 0)                    # B sees A's $2: 2 + 2 <= 5
    with pytest.raises(BudgetExceeded):
        a.reserve("openai", "m", 2, 0)                     # A must see B's open $2 reservation: 2 + 2 + 2 > 5
    b.record("openai", "m", "r", 2, 0, rb)
    # speed: 40k pre-existing lines, then one reserve
    with open(led / "spend.jsonl", "a") as f:
        for i in range(40000):
            f.write(json.dumps({"type": "settle", "id": f"x{i}", "ts": 0, "provider": "anthropic", "model": "m", "usd": 0.0}) + "\n")
    c = Budget(led, prices, session_cap_usd=100)
    t0 = time.time()
    monkeypatch.setenv("BUDGET_USD_ANTHROPIC", "1")
    c.reserve("anthropic", "m", 0, 0)                      # first call parses once
    t1 = time.time()
    for _ in range(20):
        c.reserve("anthropic", "m", 0, 0)                  # later calls read only new lines
    t2 = time.time()
    assert (t2 - t1) / 20 < (t1 - t0) + 0.05
