"""n-firm logit Bertrand market (Lee Eq. 1, Calvano parameterisation). Pure functions.

q_i  = exp((a - p_i)/mu) / (sum_j exp((a - p_j)/mu) + exp(a0/mu))
pi_i = (p_i - c) * q_i
Lee: n = 4, a = 2, a0 = 0, c = 1, mu = 0.25 -> Nash 1.33 (pi 0.0815), monopoly 2.05 (pi 0.2011).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NMarket:
    n: int = 4
    a: float = 2.0
    a0: float = 0.0
    c: float = 1.0
    mu: float = 0.25

    def quantities(self, prices: list[float]) -> tuple[list[float], float]:
        """(q_1..q_n, outside share). Numerically stable softmax."""
        if len(prices) != self.n:
            raise ValueError(f"need {self.n} prices, got {len(prices)}")
        z = [(self.a - p) / self.mu for p in prices] + [self.a0 / self.mu]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        tot = sum(e)
        return [x / tot for x in e[:-1]], e[-1] / tot

    def profits(self, prices: list[float]) -> list[float]:
        q, _ = self.quantities(prices)
        return [(p - self.c) * qi for p, qi in zip(prices, q, strict=True)]

    def clear(self, prices: list[float]) -> dict:
        q, q0 = self.quantities(prices)
        return {"prices": list(prices), "q": q, "q0": q0, "pi": [(p - self.c) * qi for p, qi in zip(prices, q, strict=True)]}

    def best_response(self, i: int, prices: list[float], lo: float = 0.0, hi: float = 5.0, tol: float = 1e-12) -> float:
        """Firm i's static BR to the others' prices: solve (p - c)(1 - q_i) = mu (unique root above c)."""
        def f(p: float) -> float:
            pp = list(prices)
            pp[i] = p
            return (p - self.c) * (1.0 - self.quantities(pp)[0][i]) - self.mu
        lo = max(lo, self.c)
        if f(hi) < 0:
            return hi
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
        p = 1.5
        for _ in range(1000):
            q = self.best_response(0, [p] * self.n)
            if abs(q - p) < 1e-13:
                break
            p = q
        return p, self.profits([p] * self.n)[0]

    def monopoly(self) -> tuple[float, float]:
        g = (math.sqrt(5) - 1) / 2
        lo, hi = self.c, self.c + 3.0
        joint = lambda p: sum(self.profits([p] * self.n))  # noqa: E731
        for _ in range(300):
            x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
            if joint(x1) < joint(x2):
                lo = x1
            else:
                hi = x2
        p = 0.5 * (lo + hi)
        return p, self.profits([p] * self.n)[0]

    def benchmarks(self) -> dict:
        pN, piN = self.nash()
        pM, piM = self.monopoly()
        return {"pN": pN, "piN": piN, "pM": pM, "piM": piM, "n": self.n, "a": self.a, "a0": self.a0, "c": self.c, "mu": self.mu}


def delta(mean_profit: float, piN: float, piM: float) -> float:
    return (mean_profit - piN) / (piM - piN)
