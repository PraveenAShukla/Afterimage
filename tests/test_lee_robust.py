"""Robustness of the Lee replication package before any paid run."""
import json
from pathlib import Path

import pytest
import yaml

import lee.backend as lb
from lee.backend import PROVIDERS, ChatBackend, base_url
from lee.parse import parse_reply
from lee.run import load, main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text, status, price", [
    ('{"price": 2.1, "plan": "p", "insight": "i"}', "ok", 2.1),
    ('```json\n{"price": 2.1, "plan": "p", "insight": "i"}\n```', "ok", 2.1),           # fenced JSON
    ('{"price": "2.10", "plan": "p", "insight": "i"}', "ok", 2.1),                       # price as a string
    ('Thinking... {"price": 9} then {"price": 1.8, "plan": "p", "insight": "i"}', "ok", 1.8),  # last object wins
    ('{"price": true, "plan": "p"}', "no_price", None),                                  # bool is not a price
    ('{"price": "nan", "plan": "p"}', "no_price", None),                                 # NaN rejected
    ('{"price": "inf", "plan": "p"}', "no_price", None),                                 # inf rejected
    ('{"price": "$2.10", "plan": "p"}', "no_price", None),                               # currency sign: retry, not guess
    ('{"price": -3, "plan": "p", "insight": "i"}', "ok", 0.0),                           # clipped to pmin, reported
    ('"price": 2.4 and the rest was cut', "price_only", 2.4),                            # truncated JSON
    ("", "empty", None),
])
def test_parser_edge_cases(text, status, price):
    r = parse_reply(text)
    assert r["status"] == status and r["price"] == price


def test_providers_and_base_url_override(monkeypatch):
    for p in ("openai", "anthropic", "google", "deepseek", "qwen"):
        assert PROVIDERS[p][1].endswith("_API_KEY")  # the log scrubber redacts names ending in KEY
    assert base_url("qwen").startswith("https://dashscope-intl.aliyuncs.com")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://some-other-tool-proxy.example/v1")  # must be ignored
    assert base_url("anthropic") == "https://api.anthropic.com/v1"
    monkeypatch.setenv("LEE_QWEN_BASE_URL", "https://ws123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/")
    assert ChatBackend("qwen", "qwen3.5-flash").base_url == "https://ws123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    with pytest.raises(ValueError):
        ChatBackend("qwen", "qwen3.5-flash", {"temperature": 1.0})


def test_transport_errors_and_odd_responses(monkeypatch):
    """Errors become Completion.error (never an exception); empty choices / missing usage do not crash."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    b = ChatBackend("deepseek", "deepseek-flash", {"thinking": {"type": "disabled"}, "max_tokens": 600})
    sent = {}

    def fake(url, headers, body, timeout, attempts, js=None, err=None):
        sent.update(body)
        return js, err, 1, 0.1
    monkeypatch.setattr(lb, "_post_with_retries", lambda *a: fake(*a, err="HTTP 500: boom"))
    c = b.complete("s", "u")
    assert c.error == "HTTP 500: boom" and c.text == ""
    assert sent["thinking"] == {"type": "disabled"} and "temperature" not in sent
    monkeypatch.setattr(lb, "_post_with_retries", lambda *a: fake(*a, js={"choices": [], "model": "x"}))
    c = b.complete("s", "u")
    assert c.text == "" and c.error is None and c.input_tokens is None
    monkeypatch.setattr(lb, "_post_with_retries", lambda *a: fake(*a, js={"choices": [{"message": {"content": '{"price": 2}'},
                                                                                       "finish_reason": "length"}]}))
    assert b.complete("s", "u").finish_reason == "length"


def _cfg(tmp_path, name, **kw):
    c = {"stage": "lt", "phase": "rep", "cell": "mockavg", "firm": {"provider": "mock", "model": "mock", "strategy": "avg"},
         "n": 4, "rounds": 40, "seeds": [1], "max_output_tokens": 200, "parallel_games": 1}
    c.update(kw)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(c))
    return str(p)


def test_partial_run_is_quarantined_not_appended(tmp_path):
    runs = tmp_path / "runs"
    (runs / "lt_rep_mockavg_s1").mkdir(parents=True)
    (runs / "lt_rep_mockavg_s1" / "rounds.jsonl").write_text('{"t": 1}\n')  # a crashed run's leftovers
    assert main([_cfg(tmp_path, "c"), "--runs-dir", str(runs)]) == 0
    q = list((runs / "_aborted").iterdir())
    assert len(q) == 1 and q[0].name.startswith("lt_rep_mockavg_s1__")
    assert json.loads((runs / "lt_rep_mockavg_s1" / "summary.json").read_text())["status"] == "complete"


def test_r1_refuses_missing_or_invalid_sources(tmp_path):
    runs = tmp_path / "runs"
    op = {"x": 1.80, "source_template": "lt_rep_mockavg_s{seed}", "pool": [1], "perm_seed": 7, "window": [11, 30]}
    with pytest.raises(RuntimeError, match="not a complete, validated R0 run"):
        main([_cfg(tmp_path, "o", phase="conf", opening=op, seeds=[5]), "--runs-dir", str(runs)])


def test_real_lee_configs_are_consistent():
    """Every configs/lee file loads; models have verified prices; no temperature; thinking is off where needed."""
    prices = yaml.safe_load((ROOT / "configs" / "prices.yaml").read_text())
    files = sorted((ROOT / "configs" / "lee").glob("*.yaml"))
    assert len(files) == 20
    seeds = {}
    for f in files:
        c = load(str(f))
        firm = c["firms"][0]
        assert all(x == firm for x in c["firms"]) and len(c["firms"]) == 4
        assert "temperature" not in firm["extra"] and firm["extra"].get("max_tokens", firm["extra"].get("max_completion_tokens")) == 600
        assert "UNVERIFIED" not in prices[firm["model"]]["verified"], firm["model"]
        if firm["provider"] == "deepseek":
            assert firm["extra"]["thinking"] == {"type": "disabled"}
        if "qwen" in firm["model"]:  # thinking must be off (probe: default hits the cap with no price)
            assert firm["extra"].get("enable_thinking") is False or firm["extra"].get("reasoning", {}).get("enabled") is False
        if c.get("opening"):
            assert c["opening"]["source_template"].startswith(f"lee_rep_{c['cell']}_s")
        for s in c["seeds"]:
            assert s not in seeds, f"seed {s} reused in {f.name} and {seeds[s]}"
            seeds[s] = f.name
