"""Live run-health checks, evaluated every round. A run that trips a hard check is ABORTED: it keeps its logs, gets
summary.json with status="aborted" plus the reason, and analysis excludes it. Soft checks only emit warnings.

Why: a run with 30% fallbacks has prices frozen by the harness, not chosen by the model. That looks exactly like
stable collusion, which is the very effect we measure.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

DEFAULTS = {"window": 20, "max_fallback_rate": 0.10, "max_truncation_rate": 0.25,
            "max_consecutive_transport_errors": 5, "frozen_price_rounds": 30}


class RunAborted(RuntimeError):
    pass


@dataclass
class HealthMonitor:
    cfg: dict = field(default_factory=dict)

    def __post_init__(self):
        self.c = {**DEFAULTS, **(self.cfg or {})}
        w = self.c["window"]
        self.fallbacks = deque(maxlen=w)     # per decision: 1 if fallback
        self.truncs = deque(maxlen=w * 3)    # per call: 1 if truncated
        self.consec_transport = 0
        self.prices = deque(maxlen=self.c["frozen_price_rounds"])
        self.warned_frozen = False

    def observe_decision(self, status: str, calls: list[dict]) -> None:
        if not calls:  # scripted / fixed / replay: nothing to monitor
            return
        self.fallbacks.append(int(status.startswith("fallback")))
        for c in calls:
            self.truncs.append(int(bool(c["truncated"])))
            self.consec_transport = self.consec_transport + 1 if c["transport_error"] else 0

    stop_file = None  # set by the runner: runs/STOP aborts every running game at its next round

    def check(self, t: int, p1: float, p2: float) -> list[str]:
        """Returns soft warnings; raises RunAborted on a hard failure."""
        c = self.c
        if self.stop_file is not None and self.stop_file.exists():
            raise RunAborted(f"round {t}: kill switch {self.stop_file} present")
        if self.consec_transport >= c["max_consecutive_transport_errors"]:
            raise RunAborted(f"round {t}: {self.consec_transport} consecutive transport errors (server down?)")
        if len(self.fallbacks) == self.fallbacks.maxlen:
            r = sum(self.fallbacks) / len(self.fallbacks)
            if r > c["max_fallback_rate"]:
                raise RunAborted(f"round {t}: fallback rate {r:.2f} > {c['max_fallback_rate']} over last {len(self.fallbacks)} decisions")
        if len(self.truncs) >= 2 * c["window"]:
            r = sum(self.truncs) / len(self.truncs)
            if r > c["max_truncation_rate"]:
                raise RunAborted(f"round {t}: truncation rate {r:.2f} > {c['max_truncation_rate']} (max_tokens too small?)")
        warns = []
        self.prices.append((round(p1, 6), round(p2, 6)))
        if len(self.prices) == self.prices.maxlen and len(set(self.prices)) == 1 and not self.warned_frozen:
            self.warned_frozen = True
            warns.append(f"round {t}: both prices identical for {len(self.prices)} rounds (genuine, or a stuck harness?)")
        return warns
