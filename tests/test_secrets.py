"""The API-key layer must be shown to FIRE: a key that reaches a model output, error or prompt never hits disk."""
import json
import os
import subprocess

from collusion.agent import Agent
from collusion.backends import Completion, MockBackend
from collusion.runlog import RunLog
from collusion.secrets import Redactor, env_file_problems, scan_text

FAKE_ANT = "sk-ant-" + "api03-" + "A" * 40          # built at runtime so no literal key sits in the repo
FAKE_OAI = "sk-proj-" + "b" * 40


def test_patterns_detected():
    assert "anthropic" in scan_text(f"key={FAKE_ANT}")
    assert "openai" in scan_text(f"Bearer {FAKE_OAI}")
    assert scan_text("price is 1.85, sk-ish words") == []


def test_literal_env_value_redacted(monkeypatch):
    monkeypatch.setenv("MY_SERVICE_TOKEN", "plainsecret123")
    r = Redactor()
    assert r.redact("echo plainsecret123 here") == "echo [REDACTED:MY_SERVICE_TOKEN] here" and r.hits == 1


class LeakyBackend(MockBackend):
    """A model/server that echoes a key in its output (e.g. a 401 error body or prompt injection)."""
    def complete(self, system, user, sampling, seed):
        return Completion(text=f'my key is {os.environ["ANTHROPIC_API_KEY"]} {{"price": 1.7}}', finish_reason="stop",
                          input_tokens=10, output_tokens=10, latency_s=0.01, model_returned=self.model)


def test_key_in_model_output_never_reaches_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANT)
    log = RunLog(tmp_path, "r1")
    d = Agent(LeakyBackend()).decide(1, [], None, 0, 1)
    for c in d.calls:
        log.call(c)
    log.debug("debug " + FAKE_ANT)
    log.finish({"note": FAKE_ANT})
    blob = "".join(p.read_text() for p in (tmp_path / "r1").iterdir() if p.is_file())
    assert FAKE_ANT not in blob and "[REDACTED:ANTHROPIC_API_KEY]" in blob
    events = [json.loads(line) for line in (tmp_path / "r1/events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "secret_redacted" for e in events)


def test_env_file_problems(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".env").write_text("X=1\n")
    os.chmod(tmp_path / ".env", 0o644)
    probs = env_file_problems(tmp_path)
    assert any("not git-ignored" in p for p in probs) and any("chmod 600" in p for p in probs)
    (tmp_path / ".gitignore").write_text(".env\n")
    os.chmod(tmp_path / ".env", 0o600)
    assert env_file_problems(tmp_path) == []


def test_anthropic_key_env_selection_and_naming(monkeypatch):
    """A second Anthropic key is picked by variable NAME; names must end in _API_KEY so the scrubber redacts them."""
    import pytest
    from collusion.backends.http import AnthropicBackend
    from collusion.secrets import SECRET_ENV
    assert SECRET_ENV.search("ANTHROPIC_API_KEY") and not SECRET_ENV.search("ANTHROPIC_API_KEY_NOEL")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-" + "x" * 30)
    b = AnthropicBackend("claude-haiku-5-5", api_key_env="ANTHROPIC_API_KEY")
    assert b.api_key_env == "ANTHROPIC_API_KEY"
    with pytest.raises(ValueError):
        AnthropicBackend("claude-haiku-5-5", api_key_env="ANTHROPIC_API_KEY_NOEL")  # would not be redacted
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError):
        AnthropicBackend("claude-haiku-5-5", api_key_env="ANTHROPIC_API_KEY")
