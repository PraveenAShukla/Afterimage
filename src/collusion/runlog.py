"""Append-only run directories. A run_id that exists is never written into again.

runs/<run_id>/
  manifest.json   config, seeds, code version (git sha + dirty diff hash), host, package versions, server info
  calls.jsonl     one line per agent call (every retry attempt included)
  rounds.jsonl    one line per round: prices, shares, profits, BRs, flags
  events.jsonl    anomalies: parse failures, clipping, truncation, fallbacks, invariant violations
  summary.json    written last; its presence == run completed cleanly
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from .secrets import Redactor  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def code_version() -> dict:
    """git sha + whether the tree is dirty INCLUDING untracked files. No commit at all -> sha None, dirty True."""
    sha = _git("rev-parse", "--verify", "-q", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all", "--", "src", "analysis", "configs", "tests", "serve", "slurm", "scripts", "pyproject.toml")
    diff = _git("diff", "HEAD") if sha else ""
    return {"git_sha": sha or None, "dirty": (not sha) or bool(status),
            "dirty_files": status.splitlines()[:50] if status else [],
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}


def env_info() -> dict:
    pk = {}
    for name in ("numpy", "httpx", "pyyaml", "matplotlib", "vllm", "torch", "transformers"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    # Privacy: no hostnames, cluster names, job ids or paths in logs; only hardware needed for transparency.
    # The hash still lets us tell runs from the same machine apart.
    host_hash = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:10]
    return {"host_hash": host_hash, "platform": platform.system(), "python": sys.version.split()[0],
            "packages": pk, "gpus": _gpus(), "slurm": bool(os.environ.get("SLURM_JOB_ID"))}


def _gpus() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:  # noqa: BLE001 - no GPU on this machine (client side / mock runs)
        return []


class RunLog:
    def __init__(self, runs_dir: Path, run_id: str, resume: bool = False):
        self.dir = Path(runs_dir) / run_id
        if resume:  # continue an interrupted run IN PLACE: append only, never rewrite a line
            if not (self.dir / "manifest.json").exists() or (self.dir / "summary.json").exists():
                raise RuntimeError(f"{run_id}: can only resume a started, unfinished run")
        else:
            self.dir.mkdir(parents=True, exist_ok=False)  # refuses to reuse a run_id: never overwrite
        self.run_id = run_id
        self._files = {k: open(self.dir / f"{k}.jsonl", "a", buffering=1) for k in ("calls", "rounds", "events")}
        self.redactor = Redactor()  # every byte written below passes through it
        self._lock = threading.RLock()

    def write_manifest(self, manifest: dict) -> None:
        manifest = {**manifest, "run_id": self.run_id, "created_unix": time.time(),
                    "code": code_version(), "env": env_info()}
        (self.dir / "manifest.json").write_text(self.redactor.redact(json.dumps(manifest, indent=2, default=str)))

    def _w(self, kind: str, rec: dict) -> None:
        line = json.dumps({"run_id": self.run_id, "ts": time.time(), **rec}, default=str)
        safe = self.redactor.redact(line)
        with self._lock:  # many threads may log at once (one-step map): never interleave lines
            if kind not in self._files:  # extra streams (e.g. samples) are opened on first use
                self._files[kind] = open(self.dir / f"{kind}.jsonl", "a", buffering=1)
            f = self._files[kind]
            f.write(safe + "\n")
            f.flush()
            os.fsync(f.fileno())  # a crash loses at most the in-flight line
        if safe != line and kind != "events":
            self._w("events", {"event": "secret_redacted", "in": kind})

    def call(self, rec: dict) -> None: self._w("calls", rec)
    def round(self, rec: dict) -> None: self._w("rounds", rec)
    def event(self, kind: str, **rec) -> None: self._w("events", {"event": kind, **rec})

    def debug(self, msg: str, echo: bool = False) -> None:
        """Human-readable debug log (debug.log), optionally echoed to stderr with the run id."""
        line = self.redactor.redact(f"[{time.strftime('%H:%M:%S')}] {msg}")
        with open(self.dir / "debug.log", "a") as f:
            f.write(line + "\n")
        if echo:
            print(f"[{self.run_id}] {line}", file=sys.stderr, flush=True)

    def heartbeat(self, state: dict) -> None:
        """Atomic progress file for `watch cat runs/*/heartbeat.json` style monitoring."""
        tmp = self.dir / ".heartbeat.tmp"
        tmp.write_text(self.redactor.redact(json.dumps({"ts": time.time(), **state})))
        os.replace(tmp, self.dir / "heartbeat.json")

    def finish(self, summary: dict) -> None:
        for f in self._files.values():
            f.close()
        (self.dir / "summary.json").write_text(self.redactor.redact(json.dumps(summary, indent=2, default=str)))


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_run(runs_dir: Path, run_id: str) -> dict:
    d = Path(runs_dir) / run_id
    return {"manifest": json.loads((d / "manifest.json").read_text()),
            "rounds": read_jsonl(d / "rounds.jsonl"),
            "summary": json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None}
