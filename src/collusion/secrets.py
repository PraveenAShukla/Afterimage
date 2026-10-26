"""API-key hygiene for a repo that will be public.

Rules enforced here:
  1. Keys live only in the environment / a git-ignored, owner-only (0600) .env. Never in configs or code.
  2. Every line written to any run log passes through `redact()` before it touches disk: the literal values of all
     secret env vars, and anything shaped like a known key format, become "[REDACTED:<name>]".
  3. `scan_text()` is the same detector, used by scripts/check.sh to block commits/pushes containing a key.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

# Shapes of real keys (deliberately broad; false positives only cost a redaction).
KEY_PATTERNS = {
    "anthropic": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai": re.compile(r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}"),
    "huggingface": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "github": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "aws": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$", re.I)


def secret_values() -> dict[str, str]:
    """Literal values of secret-looking env vars (len >= 8 so we never redact 'true' or '0')."""
    return {k: v for k, v in os.environ.items() if SECRET_ENV.search(k) and v and len(v) >= 8}


class Redactor:
    def __init__(self):
        self.literals = sorted(secret_values().items(), key=lambda kv: -len(kv[1]))
        self.hits = 0

    def redact(self, text: str) -> str:
        out = text
        for name, val in self.literals:
            if val in out:
                out = out.replace(val, f"[REDACTED:{name}]")
        for name, pat in KEY_PATTERNS.items():
            out = pat.sub(f"[REDACTED:{name}]", out)
        if out != text:
            self.hits += 1
        return out


def scan_text(text: str) -> list[str]:
    """Names of key formats found in text (for commit/push scanning)."""
    found = [name for name, pat in KEY_PATTERNS.items() if pat.search(text)]
    found += [f"literal:{name}" for name, val in secret_values().items() if val in text]
    return found


def env_file_problems(repo: Path) -> list[str]:
    """Problems with .env handling; empty list == fine. Used by preflight."""
    probs = []
    env = repo / ".env"
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"], cwd=repo, capture_output=True).returncode == 0
    if tracked:
        probs.append(".env is TRACKED by git: remove it from the index and rotate every key in it")
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=repo).returncode == 0
    if not ignored:
        probs.append(".env is not git-ignored")
    if env.exists():
        mode = stat.S_IMODE(env.stat().st_mode)
        if mode & 0o077:
            probs.append(f".env permissions {oct(mode)} are readable by others: run `chmod 600 .env`")
    return probs
