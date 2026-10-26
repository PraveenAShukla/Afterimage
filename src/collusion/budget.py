"""Hard spend caps for paid APIs, enforced BEFORE each call and shared across processes.

- Prices per model live in configs/prices.yaml (USD per 1M tokens). A paid model with no price entry is
  refused: we never run blind.
- Caps come from the environment: BUDGET_USD_ANTHROPIC, BUDGET_USD_OPENAI (default 0 = paid calls refused).
- Every paid call appends a line to runs/_ledger/spend.jsonl under an exclusive file lock, so parallel games
  and parallel processes cannot jointly overshoot.
- Pre-call check uses the WORST case (max_tokens of output), so a cap is never exceeded by the call that crosses it.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from pathlib import Path

import yaml

FREE_PROVIDERS = {"vllm", "mock"}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, ledger_dir: Path, prices_path: Path, session_cap_usd: float | None = None):
        """session_cap_usd: the spend the team APPROVED for this invocation (run.py --approve-usd). Paid calls are refused
        without it. It is enforced on top of the provider caps, the rolling 10-minute tripwire and the STOP file."""
        self.session_id = uuid.uuid4().hex[:12]
        self.session_cap = session_cap_usd
        self.stop_file = Path(ledger_dir).parent / "STOP"
        self.path = Path(ledger_dir) / "spend.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.prices = yaml.safe_load(Path(prices_path).read_text()) or {}
        self._recs: list[dict] = []   # parsed ledger, read incrementally (append-only file, written under a lock)
        self._off = 0                 # byte offset already parsed

    def cap(self, provider: str) -> float:
        return float(os.environ.get(f"BUDGET_USD_{provider.upper()}", "0"))

    def price(self, model: str) -> dict:
        if model not in self.prices:
            raise BudgetExceeded(f"No price for paid model {model!r} in prices.yaml; refusing to call it.")
        return self.prices[model]

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self.price(model)
        return (input_tokens * p["input_per_mtok"] + output_tokens * p["output_per_mtok"]) / 1e6

    def _read_new(self, f) -> list[dict]:
        """Parse only the lines appended since the last call (caller holds the lock). Re-reading the whole ledger on every
        call made long runs CPU-bound (Lee R0); the ledger is append-only, so incremental reading is exact."""
        f.seek(self._off)
        chunk = f.read()
        self._recs.extend(json.loads(x) for x in chunk.splitlines() if x.strip())
        self._off = f.tell()
        return self._recs

    def _entries(self, f) -> tuple[dict, dict]:
        """(settled usd per provider, OPEN reservations id -> (provider, worst usd)) from the ledger."""
        settled: dict[str, float] = {}
        open_res: dict[str, tuple[str, float]] = {}
        for line in f:
            r = line if isinstance(line, dict) else json.loads(line)
            kind = r.get("type", "settle")
            if kind == "reserve":
                open_res[r["id"]] = (r["provider"], r["usd_worst"])
            else:
                settled[r["provider"]] = settled.get(r["provider"], 0.0) + r["usd"]
                open_res.pop(r.get("id"), None)
        return settled, open_res

    def spent(self, provider: str) -> float:
        with open(self.path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            return self._entries(f)[0].get(provider, 0.0)

    def reserve(self, provider: str, model: str, est_input_tokens: int, max_output_tokens: int) -> str | None:
        """Atomically: refuse unless settled spend + ALL open reservations + this call's worst case fit under the cap;
        otherwise write a reservation line. Concurrent callers (threads or processes) serialise on the file lock,
        so they can never jointly overshoot. Returns the reservation id to pass to record()."""
        if provider in FREE_PROVIDERS:
            return None
        worst = self.cost(model, est_input_tokens, max_output_tokens)
        with open(self.path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if self.stop_file.exists():
                raise BudgetExceeded(f"kill switch: {self.stop_file} exists (remove it to allow paid calls)")
            if self.session_cap is None:
                raise BudgetExceeded("paid call without an approved spend: run with --approve-usd <amount> (agree the amount first)")
            recs = self._read_new(f)
            settled, open_res = self._entries(recs)
            committed = settled.get(provider, 0.0) + sum(w for p, w in open_res.values() if p == provider)
            if committed + worst > self.cap(provider):
                raise BudgetExceeded(f"{provider}: committed ${committed:.4f} + worst-case ${worst:.4f} > cap ${self.cap(provider):.2f}")
            closed = {r.get("id") for r in recs if r.get("type", "settle") == "settle"}
            mine = sum(r["usd"] for r in recs if r.get("type") == "settle" and r.get("session") == self.session_id)
            mine += sum(r["usd_worst"] for r in recs if r.get("type") == "reserve" and r.get("session") == self.session_id
                        and r["id"] not in closed)
            if mine + worst > self.session_cap:
                raise BudgetExceeded(f"approved spend for this run: ${mine:.4f} committed + ${worst:.4f} > ${self.session_cap:.2f}")
            now = time.time()
            cap_n = os.environ.get(f"MAX_REQUESTS_{provider.upper()}")  # provider request quota per rolling window
            if cap_n:
                hours = float(os.environ.get(f"REQUEST_WINDOW_HOURS_{provider.upper()}", "24"))
                n = sum(1 for r in recs if r.get("type") == "reserve" and r["provider"] == provider and now - r["ts"] < hours * 3600)
                if n + 1 > int(cap_n):
                    raise BudgetExceeded(f"{provider}: {n} requests in the last {hours:g} h; quota guard is {cap_n} "
                                         f"(the provider's own limit is higher; we stop first so a game never hits 429s)")
            window = float(os.environ.get("MAX_USD_PER_10MIN", "5"))
            recent = sum(r["usd"] for r in recs if r.get("type", "settle") == "settle" and now - r["ts"] < 600)
            recent += sum(r["usd_worst"] for r in recs if r.get("type") == "reserve" and r["id"] not in closed and now - r["ts"] < 600)
            if recent + worst > window:
                raise BudgetExceeded(f"spend-rate tripwire: ${recent:.2f} in the last 10 min + ${worst:.4f} > ${window:.2f}")
            rid = uuid.uuid4().hex
            f.seek(0, 2)
            f.write(json.dumps({"type": "reserve", "id": rid, "ts": time.time(), "provider": provider, "session": self.session_id,
                                "model": model, "usd_worst": worst}) + "\n")
            return rid

    def record(self, provider: str, model: str, run_id: str, input_tokens: int | None, output_tokens: int | None,
               reservation: str | None = None) -> float:
        """Settle a reservation with the actual cost (closes it)."""
        if provider in FREE_PROVIDERS:
            return 0.0
        usd = self.cost(model, input_tokens or 0, output_tokens or 0)
        line = json.dumps({"type": "settle", "id": reservation, "ts": time.time(), "provider": provider, "model": model,
                           "session": self.session_id,
                           "run_id": run_id, "input_tokens": input_tokens, "output_tokens": output_tokens, "usd": usd})
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line + "\n")
        return usd
