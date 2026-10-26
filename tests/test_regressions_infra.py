"""Regression tests: each bug found in the second adversarial review (backends, preflight, secrets, Slurm) must stay fixed."""
import json
import os
import shutil
import subprocess
import sys

import httpx
import pytest
import yaml

from collusion import preflight, runlog
from collusion.backends import Sampling
from collusion.backends import http as H
from collusion.backends.base import Completion
from collusion.budget import Budget
from collusion.run import main
from collusion.runlog import REPO

SAMP = Sampling()


class FakeResp:
    def __init__(self, status=200, text="", js=None, headers=None):
        self.status_code, self.text, self._js, self.headers = status, text, js, headers or {}

    def json(self):
        if self._js is None:
            return json.loads(self.text)
        return self._js


# ---------------------------------------------------------------- http.py
@pytest.mark.parametrize("resp", [FakeResp(200, text="<html>bad gateway</html>"),
                                  FakeResp(200, js={"choices": []}),
                                  FakeResp(200, js={"error": {"message": "overloaded"}})])
def test_C1_backend_never_raises_on_malformed_200(monkeypatch, resp):
    """Backend.complete must 'never raise for transport errors: return Completion(error=...)'."""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: resp)
    c = H.VLLMBackend("m", base_url="http://x", max_attempts=1).complete("s", "u", SAMP, 1)
    assert c.error


def test_C2_retry_after_zero_still_backs_off(monkeypatch):
    """Retry-After: 0 (some proxies send it) -> immediate hammering with no backoff at all."""
    sleeps = []
    monkeypatch.setattr(H.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp(429, text="slow", headers={"retry-after": "0"}))
    H._post_with_retries("http://x", {}, {}, 1, 4)
    assert all(s > 0 for s in sleeps) and len(sleeps) == 3


def test_C3_retry_after_is_capped(monkeypatch):
    """An (hostile or buggy) Retry-After: 86400 would park a 24 h GPU job in sleep()."""
    sleeps = []
    monkeypatch.setattr(H.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp(429, text="slow", headers={"retry-after": "86400"}))
    H._post_with_retries("http://x", {}, {}, 1, 2)
    assert max(sleeps) <= 600


def test_C4_anthropic_retries_exactly_max_attempts_and_records_them(monkeypatch):
    import anthropic
    import httpx as _httpx
    from collusion.backends.http import AnthropicBackend
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x" * 10)
    monkeypatch.setattr("collusion.backends.http.time.sleep", lambda s: None)
    be = AnthropicBackend("claude-haiku-5-5", max_attempts=4)
    assert be.client.max_retries == 0                       # SDK retries off: we count attempts ourselves
    calls = []
    def boom(**kw):
        calls.append(1)
        req = _httpx.Request("POST", "https://x")
        raise anthropic.InternalServerError("down", response=_httpx.Response(529, request=req), body=None)
    monkeypatch.setattr(be.client.messages, "create", boom)
    c = be.complete("s", "u", Sampling(), 0)
    assert len(calls) == 4 and c.transport_attempts == 4 and c.error


# ---------------------------------------------------------------- preflight.py
class _FakeVLLM:
    base_url = "http://x"

    def __init__(self, outs):
        self.outs, self.i = outs, 0

    def server_info(self):
        return {"/v1/models": {"data": [{"id": "m"}]}, "/version": {"version": "0"}}

    def complete(self, s, u, sampling, seed):
        o = self.outs[self.i]
        self.i += 1
        return o


def test_C5_preflight_seed_check_does_not_pass_on_errors(monkeypatch):
    """Server dies after the first probe: calls 2 and 3 return error with text ''. Old code reports
    'different seed -> different output: ok' (PASS) because '' != a.text."""
    ok = Completion(text='{"price": 1.5}', output_tokens=5, latency_s=1.0, finish_reason="stop")
    err = Completion(text="", error="ConnectError: refused")
    monkeypatch.setattr(preflight, "make_backend", lambda spec: _FakeVLLM([ok, err, err]))
    r = preflight.Report()
    preflight.check_vllm(r, {"provider": "vllm", "model": "m"}, SAMP, "E1")
    bad = [row for row in r.rows if row[0] == "PASS" and "seed" in row[1]]
    assert not bad, bad


def test_C6_code_version_notices_dirty_serve_slurm_scripts(tmp_path, monkeypatch):
    """code_version only looks at src analysis configs tests pyproject; an edited serve/start_vllm.sh,
    slurm/*.sbatch or scripts/ (they decide the vLLM flags actually used!) leaves dirty=False."""
    g = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q"); g("config", "user.email", "[email removed]"); g("config", "user.name", "t")
    for d in ("serve", "slurm", "scripts", "src"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "f.sh").write_text("x\n")
    g("add", "-A"); g("commit", "-qm", "c")
    monkeypatch.setattr(runlog, "REPO", tmp_path)
    assert runlog.code_version()["dirty"] is False
    (tmp_path / "serve/f.sh").write_text("changed --max-model-len 99999\n")
    assert runlog.code_version()["dirty"] is True


def test_C7_paid_probe_closes_its_reservation(tmp_path, monkeypatch):
    """check_paid reserves, then record() is called WITHOUT the reservation id: the worst-case reservation
    stays open forever and is counted against the cap on top of the real spend."""
    prices = tmp_path / "p.yaml"
    prices.write_text(yaml.safe_dump({"gpt-x": {"input_per_mtok": 5.0, "output_per_mtok": 15.0}}))
    monkeypatch.setenv("BUDGET_USD_OPENAI", "10")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-not-a-key")
    ok = Completion(text='{"price": 1.5}', input_tokens=100, output_tokens=10, model_returned="gpt-x", finish_reason="stop")
    monkeypatch.setattr(preflight, "make_backend", lambda spec: _FakeVLLM([ok]))
    b = Budget(tmp_path / "led", prices, session_cap_usd=100.0)
    r = preflight.Report()
    preflight.check_paid(r, {"provider": "openai", "model": "gpt-x"}, SAMP, "E1", b, probe=True)
    with open(b.path) as f:
        _, open_res = b._entries(f)
    assert open_res == {}, f"phantom open reservation(s): {open_res}"


# ---------------------------------------------------------------- scan_secrets.py
def _key():  # built at runtime so this file itself never contains a key-shaped literal
    return "sk-ant-" + "A1b2C3d4E5" * 3


def _scan_repo(tmp_path, env=None):
    g = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q"); g("config", "user.email", "[email removed]"); g("config", "user.name", "t")
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO / "scripts/scan_secrets.py", tmp_path / "scripts/scan_secrets.py")
    os.symlink(REPO / "src", tmp_path / "src")
    return g


def _run_scan(tmp_path, mode, env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.endswith(("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(tmp_path / "scripts/scan_secrets.py"), mode], cwd=tmp_path,
                          capture_output=True, text=True, env=env)


def test_C8_scan_does_not_skip_tests_dir(tmp_path):
    g = _scan_repo(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/t.py").write_text(f'KEY = "{_key()}"\n')
    g("add", "tests/t.py")
    assert _run_scan(tmp_path, "--staged").returncode == 1


def test_C9_scan_catches_renamed_and_modified_file(tmp_path):
    g = _scan_repo(tmp_path)
    (tmp_path / "a.txt").write_text("".join(f"line {i}\n" for i in range(40)))
    g("add", "a.txt"); g("commit", "-qm", "a")
    g("mv", "a.txt", "b.txt")
    with open(tmp_path / "b.txt", "a") as f:
        f.write(_key() + "\n")
    g("add", "b.txt")
    assert _run_scan(tmp_path, "--staged").returncode == 1  # --diff-filter=ACM drops R


def test_C10_scan_handles_filenames_with_spaces(tmp_path):
    g = _scan_repo(tmp_path)
    (tmp_path / "my notes.txt").write_text(_key() + "\n")
    g("add", "my notes.txt")
    assert _run_scan(tmp_path, "--staged").returncode == 1  # .split() cuts the name in two -> nothing read


def test_C11_prepush_scans_committed_content_not_working_tree(tmp_path):
    """pre-push runs --all, which reads the WORKING TREE. A key that is committed but locally edited out
    (uncommitted) is pushed and the scan says clean."""
    g = _scan_repo(tmp_path)
    (tmp_path / "n.txt").write_text(_key() + "\n")
    g("add", "n.txt"); g("commit", "-qm", "oops")
    (tmp_path / "n.txt").write_text("clean\n")
    assert _run_scan(tmp_path, "--all").returncode == 1


def test_C12_missing_denylist_is_not_silently_clean(tmp_path):
    """internal/private/ is git-ignored, so a fresh clone / the cluster has no denylist: the private-name check is
    then OFF while the script prints 'clean (... 0 private patterns)' and exits 0."""
    _scan_repo(tmp_path)
    r = _run_scan(tmp_path, "--all")
    assert r.returncode != 0 or "WARN" in r.stdout.upper()


def test_C13_scan_checks_dotenv_literals(tmp_path):
    """scan_text() compares against secret env vars, but the hook never loads .env, so a non-key-shaped
    secret (custom token) in .env is not known to the scan."""
    g = _scan_repo(tmp_path)
    (tmp_path / ".env").write_text("PROXY_TOKEN=zq93kd02jf84hs71\n")
    (tmp_path / "notes.md").write_text("curl -H 'X: zq93kd02jf84hs71'\n")
    g("add", "notes.md")
    assert _run_scan(tmp_path, "--staged").returncode == 1


# ---------------------------------------------------------------- shell
def _stubs(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    (b / "vllm").write_text("#!/bin/sh\nexec sleep 30\n")
    (b / "curl").write_text("#!/bin/sh\nexit 0\n")
    for f in b.iterdir():
        f.chmod(0o755)
    return b


def test_C14_vllm_log_goes_to_LOGDIR_and_refuses_without_it(tmp_path):
    b = _stubs(tmp_path)
    logs = tmp_path / "logs"; logs.mkdir()
    script = REPO / "serve/start_vllm.sh"
    env = {"PATH": f"{b}:/usr/bin:/bin"}
    subprocess.run(["bash", "-c", f'PID=$({script} m 1 8000 {logs}); kill $PID'], cwd=tmp_path, env=env,
                   check=True, capture_output=True, timeout=60)
    assert list(logs.glob("vllm_*.log")) and not list(tmp_path.glob("vllm_*.log"))
    r = subprocess.run(["bash", str(script), "m", "1", "8000"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "LOGDIR" in r.stderr                  # no LOGDIR -> refuse, never log into the repo
    sb = (REPO / "slurm/run_stage.sbatch").read_text()
    assert "set -a; source serve/site.env; set +a" in sb


def test_C15_failed_server_start_aborts_job():
    """`PIDS+=("$(serve/start_vllm.sh ...)")` under set -e: does a failing start abort the script?"""
    r = subprocess.run(["bash", "-c", 'set -euo pipefail; PIDS=(); PIDS+=("$(false)"); echo CONTINUED'], capture_output=True, text=True)
    assert "CONTINUED" not in r.stdout


# ---------------------------------------------------------------- run.py glue
def _mock_cfg(tmp_path, **kw):
    base = {"stage": "t", "prompt": "E1", "rounds": 12, "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 64},
            "arm": "L", "seeds": [5], "firm1": {"provider": "mock", "model": "mock", "strategy": "collude"},
            "firm2": {"provider": "mock", "model": "mock", "strategy": "collude"}, "parallel_games": 1}
    base.update(kw)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(base))
    return str(p)


def test_C16_rounds_override_run_is_not_resumed_as_full_run(tmp_path):
    """A smoke run with --rounds 3 (same run_id) has summary.json, so the real 12-round run is silently SKIPPED
    as 'already complete' and the stage ends up with 3-round games."""
    cfg, runs = _mock_cfg(tmp_path), tmp_path / "runs"
    assert main([cfg, "--runs-dir", str(runs), "--rounds", "3"]) == 0
    main([cfg, "--runs-dir", str(runs)])
    rounds = [json.loads(l) for l in (runs / "t_L_E1_mock_s5" / "rounds.jsonl").read_text().splitlines()]
    assert len(rounds) == 12


@pytest.mark.parametrize("bad", ["-3", "5000"])
def test_C17_rounds_override_is_validated(tmp_path, bad):
    """--rounds bypasses the [1, 1000] check that load_config enforces."""
    cfg = _mock_cfg(tmp_path)
    from collusion.config import ConfigError
    with pytest.raises(ConfigError, match="outside"):
        main([cfg, "--runs-dir", str(tmp_path / "r"), "--rounds", bad, "--dry-run"])
