"""One Lee-style game: n firms decide simultaneously from their own view (own plan + insight + the shared 20-round
table with their own quantity and profit), then the market clears. An optional injected opening (prefix rows) is
re-cleared through this market code and never re-queried. Every call and round is logged (append-only RunLog).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from collusion.agent import call_seed
from collusion.budget import Budget, BudgetExceeded
from collusion.ratelimit import get_limiter

from .market import NMarket, delta
from .parse import parse_reply
from .prompts import system_prompt, user_message

MAX_RETRIES = 2  # 3 attempts per decision, as in the duopoly harness
LAST = 50


class LeeAgent:
    def __init__(self, backend, me: int, n: int, max_output_tokens: int, pmin: float = 0.0, pmax: float = 5.0, cost: float = 1.0):
        self.backend, self.me, self.max_out = backend, me, max_output_tokens
        self.pmin, self.pmax = pmin, pmax
        self.system = system_prompt(me, n, cost, pmin, pmax)
        self.plan: str | None = None
        self.insight: str | None = None

    def decide(self, t: int, history: list[dict], run_seed: int, budget: Budget | None, run_id: str) -> tuple[float, str, list[dict]]:
        user = user_message(t, self.plan, self.insight, history, self.me)
        calls = []
        for attempt in range(MAX_RETRIES + 1):
            res = budget.reserve(self.backend.provider, self.backend.model, (len(self.system) + len(user)) // 3, self.max_out) \
                if budget is not None else None
            lim = get_limiter(self.backend.provider) if budget is not None else None
            ev = lim.acquire(self.max_out) if lim else None
            c = self.backend.complete(self.system, user) if self.backend.provider != "mock" else \
                self.backend.complete(self.system, user, seed=call_seed(run_seed, self.me, t, attempt))
            if lim:
                lim.settle(ev, c.output_tokens)
            usd = budget.record(self.backend.provider, self.backend.model, run_id, c.input_tokens, c.output_tokens, res) if budget else 0.0
            r = parse_reply(c.text, self.pmin, self.pmax)
            calls.append({"attempt": attempt, "firm": self.me, "model": self.backend.model, "provider": self.backend.provider,
                          "system_prompt": self.system, "user_prompt": user, "raw_output": c.text, "reasoning": c.reasoning,
                          "parse": r, "finish_reason": c.finish_reason, "input_tokens": c.input_tokens,
                          "output_tokens": c.output_tokens, "latency_s": c.latency_s, "model_returned": c.model_returned,
                          "request_id": c.request_id, "transport_error": c.error, "usd": usd, "backend_extra": c.extra})
            if r["status"] in ("ok", "price_only") and not c.error:
                if r["status"] == "ok":  # memory updates only from a well-formed reply
                    self.plan, self.insight = r["plan"], r["insight"]
                return r["price"], ("ok" if attempt == 0 else "ok_after_retry") + ("" if r["status"] == "ok" else "_price_only"), calls
        prev = history[-1]["prices"][self.me] if history else None
        return (prev if prev is not None else 2.0), ("fallback_previous" if prev is not None else "fallback_initial"), calls


def play(log, market: NMarket, agents: list[LeeAgent], n_rounds: int, run_seed: int, prefix: list[list[float]] | None = None,
         budget: Budget | None = None, max_fallback_rate: float = 0.10) -> dict:
    hist: list[dict] = []
    for t, prices in enumerate(prefix or [], start=1):
        rec = {"t": t, "round": t, **market.clear(prices), "is_prefix": True}
        hist.append(rec)
        log.round(rec)
    start, stats, status, t0 = len(hist) + 1, {"calls": 0, "fallbacks": 0, "usd": 0.0, "clipped": 0}, "complete", time.time()
    pool = ThreadPoolExecutor(max_workers=len(agents))
    try:
        for t in range(start, n_rounds + 1):
            futs = [pool.submit(a.decide, t, hist, run_seed, budget, log.run_id) for a in agents]
            outs = []
            for f in futs:
                try:
                    outs.append(f.result())
                except BudgetExceeded:
                    status = "budget_stop"
                    outs.append(None)
            for o in outs:
                for c in (o[2] if o else []):
                    log.call({"round": t, **c})
                    stats["calls"] += 1
                    stats["usd"] += c["usd"] or 0.0
                    stats["clipped"] += bool(c["parse"].get("clipped"))
            if status == "budget_stop":
                log.event("budget_stop", round=t)
                break
            prices = [o[0] for o in outs]
            sts = [o[1] for o in outs]
            stats["fallbacks"] += sum(s.startswith("fallback") for s in sts)
            rec = {"t": t, "round": t, **market.clear(prices), "statuses": sts, "is_prefix": False}
            hist.append(rec)
            log.round(rec)
            log.heartbeat({"round": t, "of": n_rounds, "elapsed_s": round(time.time() - t0, 1), "usd": round(stats["usd"], 4)})
            live = t - start + 1
            if live >= 20 and stats["fallbacks"] / (live * len(agents)) > max_fallback_rate:
                status = "aborted"
                log.event("abort", detail=f"fallback rate {stats['fallbacks'] / (live * len(agents)):.2f} > {max_fallback_rate}")
                break
    finally:
        pool.shutdown(wait=True)
    return summarise(hist, market, stats, status)


def summarise(hist: list[dict], market: NMarket, stats: dict, status: str) -> dict:
    b = market.benchmarks()
    live = [h for h in hist if not h["is_prefix"]]
    last = hist[-LAST:]
    out = {"status": status, "n_rounds_played": len(hist), "n_live": len(live), **stats, "benchmarks": b}
    if len(last) == LAST:
        mpi = sum(sum(h["pi"]) / len(h["pi"]) for h in last) / LAST
        out["delta"] = delta(mpi, b["piN"], b["piM"])
        out["mean_price_last50"] = sum(sum(h["prices"]) / len(h["prices"]) for h in last) / LAST
        moves = sum(any(abs(a - c) > 0.005 for a, c in zip(h["prices"], g["prices"], strict=True)) for h, g in zip(last[1:], last[:-1], strict=True))
        out["converged_lee"] = moves <= 5  # Lee: at most 5 of the last 50 periods with a price change > 0.005
    return out
