"""Firm policies. An arm is just a pair of policies (+ an optional replayed prefix for D / D-frozen).

Every policy answers act(t, history, prev_price) -> Decision. Non-LLM policies return a Decision with no
calls, and a `source` tag so rounds.jsonl says exactly where each price came from.
"""
from __future__ import annotations

from dataclasses import dataclass

from .agent import Agent, Decision
from .budget import Budget
from .market import Market


class Policy:
    source = "?"

    def act(self, t: int, history: list[dict], prev_price: float | None, run_seed: int, firm: int,
            budget: Budget | None, run_id: str) -> Decision:
        raise NotImplementedError


@dataclass
class Live(Policy):
    agent: Agent
    source = "live"

    def act(self, t, history, prev_price, run_seed, firm, budget, run_id):
        return self.agent.decide(t, history, prev_price, run_seed, firm, budget, run_id)


@dataclass
class Fixed(Policy):
    """FN / FM arms: firm 2 at a constant price."""
    price: float
    source = "fixed"

    def act(self, *a, **k):
        return Decision(self.price, self.price, "fixed", False)


@dataclass
class Replay(Policy):
    """P arm: open-loop playback of a recorded price sequence (index t-1). Ignores the opponent entirely."""
    prices: list[float]
    source = "replay"

    def act(self, t, *a, **k):
        p = self.prices[t - 1]
        return Decision(p, p, "replay", False)


@dataclass
class Scripted(Policy):
    """Overrides `inner` on specific rounds (forced deviation, frozen control); delegates otherwise."""
    inner: Policy | None
    overrides: dict[int, float]
    source = "scripted"

    def act(self, t, history, prev_price, run_seed, firm, budget, run_id):
        if t in self.overrides:
            p = self.overrides[t]
            return Decision(p, p, "forced", False)
        if self.inner is None:
            raise RuntimeError(f"Scripted policy has no inner policy for round {t}")
        return self.inner.act(t, history, prev_price, run_seed, firm, budget, run_id)

    def source_at(self, t: int) -> str:
        return "forced" if t in self.overrides else getattr(self.inner, "source", "?")


@dataclass
class Impulse(Policy):
    """Arm IMP (plan v2 addendum N3): at rounds t0 .. t0+hold-1 play own last pre-shock price + delta (clipped to
    [1, 2.5]); delegates to `inner` (live) otherwise. The forced price is decided at runtime from p_own(t0 - 1)."""
    inner: Policy
    t0: int
    delta: float
    hold: int = 1
    _base: float | None = None
    source = "impulse"

    def act(self, t, history, prev_price, run_seed, firm, budget, run_id):
        if self.t0 <= t < self.t0 + self.hold:
            if t == self.t0:
                self._base = prev_price
            p = min(2.5, max(1.0, self._base + self.delta))
            return Decision(p, p, "impulse", False)
        return self.inner.act(t, history, prev_price, run_seed, firm, budget, run_id)

    def source_at(self, t: int) -> str:
        return "impulse" if self.t0 <= t < self.t0 + self.hold else getattr(self.inner, "source", "?")


def deviation_overrides(market: Market, prefix: list[dict], shock_round: int, frozen: bool,
                        window: int = 10, placebo: bool = False) -> tuple[dict[int, float], dict[int, float]]:
    """D: firm 1 plays BR(p2 at shock_round-1) in shock_round, then is released.
    D-frozen: additionally, firm 2 is held at its (shock_round-1) price for shock_round..shock_round+window-1.
    `prefix` = the stored L run's round records for rounds 1..shock_round-1."""
    last = prefix[-1]
    if last["t"] != shock_round - 1 or [r["t"] for r in prefix] != list(range(1, shock_round)):
        raise RuntimeError(f"prefix must be rounds 1..{shock_round - 1} with no gaps; got ..{last['t']}")
    if placebo:  # P2: same branch point, NO forced price: the counterfactual / noise floor for D
        return {}, {}
    br = market.best_response(last["p2"])
    # M7 guard: if BR(p2) is not below firm 1's own last price (e.g. p2 < p^N), forcing BR would be a RAISE, not
    # a cut. Then force a material cut of 0.10 instead (floor 1.0); the rule used is visible from forced1 vs p1.
    f1 = {shock_round: br if br < last["p1"] - 0.02 else max(1.0, last["p1"] - 0.10)}
    f2 = {t: last["p2"] for t in range(shock_round, shock_round + window)} if frozen else {}
    return f1, f2
