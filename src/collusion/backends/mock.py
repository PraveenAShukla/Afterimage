"""Deterministic fake LLMs for testing the whole harness with zero GPUs and zero dollars.

Strategies read the user message (so they exercise the real history formatting) and emit text shaped like
a real model's, including <think> blocks and garbage, so parser/retry/fallback paths get hit.
"""
from __future__ import annotations

import random
import re
import time

from ..market import Market
from .base import Backend, Completion, Sampling

_ROW = re.compile(r"^(\d+), ([\d.]+), ([\d.]+)(?:, ([\d.]+), ([\d.]+))?(?:, [\d.]+)?$", re.M)  # E1-PO rows: 3 cols


_HINT = re.compile(r"highest at a price of (\d\.\d{3})")


def _last_row(user: str) -> tuple[float, float] | None:
    rows = _ROW.findall(user)
    if not rows:
        return None
    _, own, rival, _, _ = rows[-1]
    return float(own), float(rival)


class MockBackend(Backend):
    """strategy:
      br        - best-responds to rival's last price (competitive benchmark)
      collude   - starts at 1.8, matches rival upward, punishes cuts with Nash for 3 rounds
      fixed:<p> - always p
      garbage:<q> - like 'br' but with prob q emits no JSON (tests retries/fallback)
      outofrange - emits 3.7 (tests clipping)
    Scripted CONTROLS for the S1 opening experiment (same pipeline as the LLMs):
      averager      - (own + rival) / 2 of the last round: slope 1 in the opening, midpoint convergence
      grim[:<p>]    - prices p (default 1.80) while every visible rival price >= p - 0.005, else Nash
      hillclimb[:<s>] - profit hill-climber on its own last two rounds, step s (default 0.02)
      partial[:<w>] - partial matcher: w * rival_last + (1 - w) * Nash (default w = 0.5): slope w in the opening
      tft           - tit-for-tat: the rival's last price
      focal[:<p>]   - always p (default 1.80), whatever the history
      noisybr[:<s>] - BR(rival_last) + N(0, s) (default s = 0.03), seeded per call
    Mechanism controls (plan v2 addendum N2):
      match         - the rival's last price (same as tft)
      cutmatch      - min(own last, rival last): matches cuts, ignores raises (the kinked-demand rule)
      raisematch    - max(own last, rival last): the mirror image
      winstay       - move to the rival's last price if own share fell last round, else stay
      satisfice[:g] - move to BR(rival last) only if the static profit gain exceeds g (default 0.025), else stay
      step:x:t0:d   - plays x, and x + d from round t0 on (a scripted rival)
    """
    provider = "mock"

    def __init__(self, model: str = "mock", strategy: str = "br", latency_s: float = 0.0):
        super().__init__(model)
        self.strategy = strategy
        self.latency_s = latency_s
        self.m = Market()
        self._punish: dict[int, int] = {}

    def _price(self, user: str, rng: random.Random) -> float:
        last = _last_row(user)
        s = self.strategy
        if s.startswith("fixed:"):
            return float(s.split(":")[1])
        if s == "outofrange":
            return 3.7
        if s in ("br",) or s.startswith("garbage"):
            return 1.6 if last is None else self.m.best_response(last[1])
        if s == "averager":
            return 1.6 if last is None else (last[0] + last[1]) / 2
        if s.startswith("grim"):
            trig = float(s.split(":")[1]) if ":" in s else 1.80
            rows = _ROW.findall(user)
            if any(float(r[2]) < trig - 0.005 for r in rows):
                return self.m.nash()[0]
            return trig
        if s.startswith("partial"):
            w = float(s.split(":")[1]) if ":" in s else 0.5
            return 1.6 if last is None else w * last[1] + (1 - w) * self.m.nash()[0]
        if s == "hint":  # lever-1 mock: follows a br_hint line when shown, else plays the rival's last price
            mh = _HINT.search(user)
            if mh:
                return float(mh.group(1))
            return 1.6 if last is None else last[1]
        if s in ("tft", "match"):
            return 1.6 if last is None else last[1]
        if s == "cutmatch":
            return 1.6 if last is None else min(last[0], last[1])
        if s == "raisematch":
            return 1.6 if last is None else max(last[0], last[1])
        if s == "winstay":
            rows = _ROW.findall(user)
            if len(rows) < 2 or not rows[-1][3] or not rows[-2][3]:
                return 1.6 if last is None else last[0]
            return last[1] if float(rows[-1][3]) < float(rows[-2][3]) - 1e-12 else last[0]
        if s.startswith("satisfice"):
            g = float(s.split(":")[1]) if ":" in s else 0.025
            if last is None:
                return 1.6
            own, riv = last
            br = self.m.best_response(riv)
            return br if self.m.profits(br, riv)[0] - self.m.profits(own, riv)[0] > g else own
        if s.startswith("step:"):
            _, x, t0, d = s.split(":")
            t = int(re.match(r"Round (\d+)\.", user).group(1))
            return float(x) + (float(d) if t >= int(t0) else 0.0)
        if s.startswith("focal"):
            return float(s.split(":")[1]) if ":" in s else 1.80
        if s.startswith("noisybr"):
            sd = float(s.split(":")[1]) if ":" in s else 0.03
            return 1.6 if last is None else self.m.best_response(last[1]) + rng.gauss(0, sd)
        if s.startswith("hillclimb"):
            step = float(s.split(":")[1]) if ":" in s else 0.02
            rows = _ROW.findall(user)
            if len(rows) < 2:
                return 1.6 if last is None else last[0]
            (_, o0, _, _, pi0), (_, o1, _, _, pi1) = rows[-2], rows[-1]
            dp, dpi = float(o1) - float(o0), float(pi1) - float(pi0)
            if abs(dp) < 1e-9:
                direction = 1 if rng.random() < 0.5 else -1
            else:
                direction = (1 if dp > 0 else -1) * (1 if dpi >= 0 else -1)
            return float(o1) + step * direction
        if s == "smart":  # colludes with a reacting rival, best-responds to a constant (fixed) one: passes both gates
            rows = _ROW.findall(user)
            if len(rows) >= 3 and len({r[2] for r in rows[-3:]}) == 1:
                return self.m.best_response(float(rows[-1][2]))
            s = "collude"
        if s == "collude":
            if last is None:
                return 1.8
            own, rival = last
            if rival < own - 0.05:  # rival cut: punish at Nash
                return self.m.nash()[0]
            return min(1.925, max(own, rival) + 0.01 * rng.random())
        raise ValueError(f"unknown mock strategy {s}")

    def complete(self, system: str, user: str, sampling: Sampling, seed: int) -> Completion:
        """A strategy may carry '~<sd>' (e.g. 'averager~0.02'): Gaussian noise added to its price (held-out tests)."""
        t0 = time.perf_counter()
        if self.latency_s:
            time.sleep(self.latency_s)
        rng = random.Random(seed)
        if system.startswith("You are a market supervisor") and '"review"' in system:  # v2 review: fixed label pattern
            u = rng.random()
            text = ("Looking at prices." if u < 0.15 else
                    'Steady. {"review": "concern noted"}' if u < 0.55 else 'Fine. {"review": "no concern"}')
            return Completion(text=text, finish_reason="stop", input_tokens=len(user) // 4, output_tokens=12,
                              latency_s=time.perf_counter() - t0, model_returned=self.model)
        if system.startswith("You are a market supervisor"):  # supervision arms: a canned note (fails 1 in 5 calls)
            text = "Looking at prices." if rng.random() < 0.2 else \
                f'Prices are steady. {{"note": "Round-{seed % 97} check: prices are watched closely."}}'
            return Completion(text=text, finish_reason="stop", input_tokens=len(user) // 4, output_tokens=20,
                              latency_s=time.perf_counter() - t0, model_returned=self.model)
        if self.strategy.startswith("garbage") and rng.random() < float(self.strategy.split(":")[1]):
            text = "<think>hmm</think>I will price somewhere reasonable."
        else:
            base, _, sd = self.strategy.partition("~")
            if sd:
                self.strategy = base
                try:
                    p = self._price(user, rng) + rng.gauss(0, float(sd))
                finally:
                    self.strategy = f"{base}~{sd}"
            else:
                p = self._price(user, rng)
            text = f'<think>Rival history considered. Draft {{"price": 9.99}}</think>Final answer.\n{{"price": {p:.4f}}}'
        return Completion(
            text=text, finish_reason="stop", input_tokens=len(system + user) // 4,
            output_tokens=len(text) // 4, latency_s=time.perf_counter() - t0, model_returned=self.model,
        )
