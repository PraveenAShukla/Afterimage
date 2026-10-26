"""Logit Bertrand duopoly (Calvano et al. 2020 parameterisation). Pure functions, no I/O.

s_i  = exp((a - p_i)/mu) / (exp((a - p_1)/mu) + exp((a - p_2)/mu) + exp(a0/mu))
pi_i = (p_i - c_i) * s_i,   c_1 = c, c_2 = c2 (defaults to c: the symmetric spec market, unchanged)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Market:
    a: float = 2.0
    a0: float = 0.0
    c: float = 1.0
    mu: float = 0.25
    c2: float | None = None  # firm 2's marginal cost (plan v2 M6); None = same as firm 1

    def cost(self, firm: int) -> float:
        return self.c if firm == 1 or self.c2 is None else self.c2

    def shares(self, p1: float, p2: float) -> tuple[float, float, float]:
        """Returns (s1, s2, s0) where s0 is the outside-good share. Numerically stable."""
        z = [(self.a - p1) / self.mu, (self.a - p2) / self.mu, self.a0 / self.mu]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        tot = sum(e)
        return e[0] / tot, e[1] / tot, e[2] / tot

    def profits(self, p1: float, p2: float) -> tuple[float, float]:
        s1, s2, _ = self.shares(p1, p2)
        return (p1 - self.cost(1)) * s1, (p2 - self.cost(2)) * s2

    def clear(self, p1: float, p2: float) -> dict:
        """Everything the market says about one round, for logging."""
        s1, s2, s0 = self.shares(p1, p2)
        pi1, pi2 = (p1 - self.cost(1)) * s1, (p2 - self.cost(2)) * s2
        return {"p1": p1, "p2": p2, "s1": s1, "s2": s2, "s0": s0, "pi1": pi1, "pi2": pi2}

    def best_response(self, p_rival: float, lo: float = 1.0, hi: float = 3.0, tol: float = 1e-12, firm: int = 1) -> float:
        """Static BR of `firm`: solve (p - c_firm)(1 - s_i(p, p_rival)) = mu by bisection on [lo, hi].
        The LHS is strictly increasing in p (from 0 at p = c_firm), so the root is unique."""
        c = self.cost(firm)
        lo = max(lo, c)
        f = lambda p: (p - c) * (1.0 - self.shares(p, p_rival)[0]) - self.mu  # noqa: E731
        assert f(lo) < 0 < f(hi), f"BR root not bracketed for p_rival={p_rival}"
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    def nash(self) -> tuple[float, float]:
        """Symmetric Nash price and per-firm profit: fixed point of BR."""
        p = 1.5
        for _ in range(500):
            p_new = self.best_response(p)
            if abs(p_new - p) < 1e-13:
                break
            p = p_new
        return p, self.profits(p, p)[0]

    def monopoly(self) -> tuple[float, float]:
        """Symmetric joint-profit-maximising price and per-firm profit (golden-section on [c, c+3])."""
        g = (math.sqrt(5) - 1) / 2
        lo, hi = self.c, self.c + 3.0
        joint = lambda p: sum(self.profits(p, p))
        for _ in range(300):
            x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
            if joint(x1) < joint(x2):
                lo = x1
            else:
                hi = x2
        p = 0.5 * (lo + hi)
        return p, self.profits(p, p)[0]

    def nash_asym(self) -> tuple[float, float, float, float]:
        """Nash with per-firm costs: the fixed point of (BR_1(p2), BR_2(p1)). Returns (p1, p2, pi1, pi2)."""
        p1 = p2 = 1.5
        for _ in range(1000):
            n1, n2 = self.best_response(p2, firm=1), self.best_response(p1, firm=2)
            if abs(n1 - p1) < 1e-13 and abs(n2 - p2) < 1e-13:
                break
            p1, p2 = n1, n2
        return (p1, p2, *self.profits(p1, p2))

    def benchmarks(self) -> dict:
        pN, piN = self.nash()
        pM, piM = self.monopoly()
        out = {"pN": pN, "piN": piN, "pM": pM, "piM": piM, **{f"param_{k}": v for k, v in asdict(self).items() if v is not None}}
        if self.c2 is not None:  # asymmetric costs: report the asymmetric Nash too (pN/pM above use firm 1's cost)
            out.update(dict(zip(("pN1_asym", "pN2_asym", "piN1_asym", "piN2_asym"), self.nash_asym(), strict=True)))
        return out


def collusion_index(mean_profit: float, piN: float, piM: float) -> float:
    """Delta = (mean pi - pi^N) / (pi^M - pi^N). 0 = Nash, 1 = joint monopoly."""
    return (mean_profit - piN) / (piM - piN)
