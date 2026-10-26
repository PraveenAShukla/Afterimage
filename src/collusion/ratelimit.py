"""Strict per-provider rate limiter (requests/minute and output tokens/minute), shared by all threads in a process.

Every paid call blocks in acquire() until it fits inside a sliding 60-second window. Limits come from the env:
RATE_RPM_<PROVIDER>, RATE_OUTPUT_TPM_<PROVIDER>. A paid provider with no limits set is refused (fail closed).
The worst case (max_tokens) is booked up front and corrected to the actual output after the call, so bursts of
parallel games can never exceed the limit.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque

FREE = {"vllm", "mock"}


class RateLimitConfigError(RuntimeError):
    pass


class Limiter:
    def __init__(self, provider: str, rpm: int, tpm: int, clock=time.monotonic, sleep=time.sleep, min_interval_s: float = 0.0):
        self.provider, self.rpm, self.tpm = provider, rpm, tpm
        self.min_interval_s, self.last = min_interval_s, None  # spacing between call starts (burst limits, e.g. IFM ~3/s)
        self.clock, self.sleep = clock, sleep
        self.events: deque[list] = deque()   # [timestamp, booked_output_tokens]
        self.lock = threading.Lock()
        self.waited_s = 0.0

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60.0:
            self.events.popleft()

    def acquire(self, est_output_tokens: int) -> list:
        if est_output_tokens > self.tpm:
            raise RateLimitConfigError(f"{self.provider}: one call books {est_output_tokens} tokens > TPM {self.tpm}")
        while True:
            with self.lock:
                now = self.clock()
                self._prune(now)
                used = sum(e[1] for e in self.events)
                gap = 0.0 if self.last is None else self.min_interval_s - (now - self.last)
                if len(self.events) < self.rpm and used + est_output_tokens <= self.tpm and gap <= 1e-6:
                    ev = [now, est_output_tokens]
                    self.events.append(ev)
                    self.last = now
                    return ev
                wait = max(0.01, gap) if gap > 1e-6 and len(self.events) < self.rpm else (max(0.05, 60.0 - (now - self.events[0][0])) if self.events else 0.05)
            self.waited_s += wait
            self.sleep(min(wait, 5.0))

    def settle(self, ev: list, actual_output_tokens: int | None) -> None:
        with self.lock:
            ev[1] = actual_output_tokens if actual_output_tokens is not None else ev[1]


_limiters: dict[str, Limiter] = {}
_reg_lock = threading.Lock()


def get_limiter(provider: str) -> Limiter | None:
    if provider in FREE:
        return None
    with _reg_lock:
        if provider not in _limiters:
            rpm = os.environ.get(f"RATE_RPM_{provider.upper()}")
            tpm = os.environ.get(f"RATE_OUTPUT_TPM_{provider.upper()}")
            if not rpm or not tpm:
                raise RateLimitConfigError(f"set RATE_RPM_{provider.upper()} and RATE_OUTPUT_TPM_{provider.upper()} in .env")
            gap = float(os.environ.get(f"RATE_MIN_INTERVAL_S_{provider.upper()}", "0"))
            _limiters[provider] = Limiter(provider, int(rpm), int(tpm), min_interval_s=gap)
        return _limiters[provider]
