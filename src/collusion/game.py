"""One repeated-pricing game. Both firms decide simultaneously from the same history, then the market clears.

Harness checks run every round (cheap) so bugs surface on round 1, not in the analysis:
  - shares sum to 1, profits match the formula, played prices are finite and inside [1, 2.5]
  - the history shown to each firm is rebuilt from the prices actually played (checked by re-deriving it)
  - truncated generations (finish_reason=length), fallbacks, clipping and transport errors become events
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

from .agent import Decision
from .arms import Policy
from .budget import Budget, BudgetExceeded
from .market import Market, collusion_index
from .health import HealthMonitor, RunAborted
from .runlog import RunLog


class InvariantError(RuntimeError):
    """A harness bug, not model behaviour. Never caught: the run must crash loudly."""

LAST_K = 50


def firm_view(rounds: list[dict], firm: int) -> list[dict]:
    """History from firm's point of view, built ONLY from played prices and market outcomes."""
    o, r = ("1", "2") if firm == 1 else ("2", "1")
    return [{"round": x["t"], "own_price": x["p" + o], "rival_price": x["p" + r],
             "own_share": x["s" + o], "own_profit": x["pi" + o]} for x in rounds]


def _source(policy: Policy, t: int) -> str:
    return policy.source_at(t) if hasattr(policy, "source_at") else policy.source


def play(log: RunLog, market: Market, p1: Policy, p2: Policy, n_rounds: int, run_seed: int, arm: str,
         budget: Budget | None = None, prefix: list[dict] | None = None, strict: bool = True,
         health_cfg: dict | None = None, debug_calls: int = 0,
         resumed: list[dict] | None = None, base_stats: dict | None = None, supervisor=None) -> dict:
    """prefix: already-played rounds to resume from (D arms). They are re-cleared and checked, not re-queried.
    supervisor: optional supervisor.Supervisor; before each live round it writes notes the firms then read."""
    b = market.benchmarks()
    rounds: list[dict] = []
    for x in prefix or []:  # replayed history: re-clear so numbers come from THIS market code
        rec = {"t": x["t"], **market.clear(x["p1"], x["p2"]), "source1": "prefix", "source2": "prefix"}
        rounds.append(rec)
        log.round({**rec, "arm": arm, "seed": run_seed, "is_prefix": True})
    rounds += resumed or []  # rounds already in rounds.jsonl from before an interruption: not re-logged
    start = len(rounds) + 1
    stats = {"calls": 0, "parse_fail_calls": 0, "fallbacks": 0, "clipped": 0, "truncated": 0,
             "transport_errors": 0, "price_from_think": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "decisions": 0}
    kind = getattr(supervisor, "kind", "note")  # "note" (v1 advice) | "review" (v2 review / yoked label)
    if supervisor is not None:
        stats.update({"supervisor_calls": 0, ("supervisor_no_review" if kind == "review" else "supervisor_no_note"): 0})
    for k, v in (base_stats or {}).items():
        stats[k] = v
    cum = [sum(r["pi1"] for r in resumed or []), sum(r["pi2"] for r in resumed or [])]
    t0 = time.time()
    mon = HealthMonitor(health_cfg or {})
    mon.stop_file = log.dir.parent / "STOP"
    status = "complete"
    stopped = None
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        for t in range(start, n_rounds + 1):
            h1, h2 = firm_view(rounds, 1), firm_view(rounds, 2)
            notes = {1: None, 2: None}
            if supervisor is not None:
                notes, scalls = supervisor.notes(t, rounds, run_seed)
                for c in scalls:
                    log._w("supervisor", {"arm": arm, "seed": run_seed, **c})
                stats["supervisor_calls"] += len(scalls)
                stats["supervisor_no_review" if kind == "review" else "supervisor_no_note"] += sum(1 for f in (1, 2) if not notes[f])
                for pol, f in ((p1, 1), (p2, 2)):
                    if hasattr(pol, "agent"):
                        if kind == "review":
                            pol.agent.review = (supervisor.frame, notes[f])
                        else:
                            pol.agent.note = notes[f]
            for pol, f, h in ((p1, 1, h1), (p2, 2, h2)):  # br_hint view: BR to the rival's last SHOWN price
                if hasattr(pol, "agent") and (pol.agent.view or {}).get("line") == "br_hint" and h:
                    r = round(h[-1]["rival_price"], 3)
                    pol.agent.hint = (r, market.best_response(r, firm=f))
            prev1 = rounds[-1]["p1"] if rounds else None
            prev2 = rounds[-1]["p2"] if rounds else None
            f1 = pool.submit(p1.act, t, h1, prev1, run_seed, 1, budget, log.run_id)
            f2 = pool.submit(p2.act, t, h2, prev2, run_seed, 2, budget, log.run_id)
            outcomes = []
            for f in (f1, f2):  # collect BOTH firms even if one hits the budget, so no made call goes unlogged
                try:
                    outcomes.append((f.result(), None))
                except BudgetExceeded as e:
                    outcomes.append((None, e))
            stops = [e for _, e in outcomes if e is not None]
            if stops:
                for firm, (d, e) in enumerate(outcomes, start=1):
                    for c in (d.calls if d else getattr(e, "calls", [])):
                        log.call({"arm": arm, "seed": run_seed, "round": t, "firm": firm, **c,
                                  "final_price": None, "decision_status": "budget_stop"})
                        stats["calls"] += 1
                        stats["usd"] += c["usd"]
                raise stops[0]
            d1, d2 = outcomes[0][0], outcomes[1][0]
            for firm, d in ((1, d1), (2, d2)):
                _log_decision(log, stats, arm, run_seed, t, firm, d, strict)
                mon.observe_decision(d.status, d.calls)
                if stats["calls"] <= debug_calls:
                    for c in d.calls:
                        log.debug(f"t={t} firm={firm} attempt={c['attempt']} status={c['parse_status']} "
                                  f"price={c['parsed_price']} tokens={c['output_tokens']} finish={c['finish_reason']}\n"
                                  f"--- user ---\n{c['user_prompt']}\n--- output (tail) ---\n{(c['raw_output'] or '')[-1500:]}", echo=True)
            rec = {"t": t, **market.clear(d1.price, d2.price)}
            _check_round(rec, market)
            cum[0] += rec["pi1"]; cum[1] += rec["pi2"]
            rec.update({
                "source1": _source(p1, t), "source2": _source(p2, t),
                "status1": d1.status, "status2": d2.status, "raw1": d1.raw_price, "raw2": d2.raw_price,
                "clipped1": d1.clipped, "clipped2": d2.clipped,
                "br1": market.best_response(d2.price, firm=1), "br2": market.best_response(d1.price, firm=2),
                "cum_pi1": cum[0], "cum_pi2": cum[1],
                **({f"{kind}1": notes[1], f"{kind}2": notes[2]} if supervisor is not None else {}),
                "p_minus_pN_1": d1.price - b["pN"], "p_minus_pN_2": d2.price - b["pN"],
            })
            rec["gap_br1"], rec["gap_br2"] = rec["p1"] - rec["br1"], rec["p2"] - rec["br2"]
            rounds.append(rec)
            log.round({**rec, "arm": arm, "seed": run_seed, "is_prefix": False})
            for w in mon.check(t, d1.price, d2.price):
                log.event("warning", detail=w)
                log.debug("WARNING " + w, echo=True)
            elapsed = time.time() - t0
            done = t - start + 1
            log.heartbeat({"round": t, "of": n_rounds, "elapsed_s": round(elapsed, 1),
                           "eta_s": round(elapsed / done * (n_rounds - t), 1), "p1": d1.price, "p2": d2.price,
                           "calls": stats["calls"], "parse_fail_calls": stats["parse_fail_calls"],
                           "fallbacks": stats["fallbacks"], "truncated": stats["truncated"], "usd": round(stats["usd"], 4)})
    except BudgetExceeded as e:
        stopped, status = f"budget: {e}", "budget_stop"
        log.event("budget_stop", detail=str(e))
    except RunAborted as e:
        stopped, status = f"aborted: {e}", "aborted"
        log.event("abort", detail=str(e))
        log.debug("ABORT " + str(e), echo=True)
    finally:
        pool.shutdown(wait=True)
    out = summarise(rounds, b, stats, time.time() - t0, stopped)
    out["status"] = status
    return out


def _log_decision(log: RunLog, stats: dict, arm: str, seed: int, t: int, firm: int, d: Decision, strict: bool) -> None:
    stats["decisions"] += bool(d.calls)
    for c in d.calls:
        stats["calls"] += 1
        stats["parse_fail_calls"] += c["parse_status"] != "ok"
        stats["truncated"] += bool(c["truncated"])
        stats["price_from_think"] = stats.get("price_from_think", 0) + bool(c.get("price_from_think"))
        stats["transport_errors"] += bool(c["transport_error"])
        stats["input_tokens"] += c["input_tokens"] or 0
        stats["output_tokens"] += c["output_tokens"] or 0
        stats["usd"] += c["usd"]
        log.call({"arm": arm, "seed": seed, "round": t, "firm": firm, **c,
                  "final_price": d.price, "decision_status": d.status})
        if c["parse_status"] != "ok":
            log.event("parse_failure", round=t, firm=firm, attempt=c["attempt"], status=c["parse_status"],
                      tail=(c["raw_output"] or "")[-300:])
        if c["truncated"]:
            log.event("truncated", round=t, firm=firm, attempt=c["attempt"], output_tokens=c["output_tokens"])
        if c["transport_error"]:
            log.event("transport_error", round=t, firm=firm, detail=c["transport_error"])
    if d.status.startswith("fallback"):
        stats["fallbacks"] += 1
        log.event("fallback", round=t, firm=firm, status=d.status, price=d.price)
    if d.clipped:
        stats["clipped"] += 1
        log.event("clipped", round=t, firm=firm, raw=d.raw_price, played=d.price)
    for c in d.calls:  # server ran a different model than requested: loud event (dated snapshots allowed)
        if c["model_returned"] and not _same_model(c["model"], c["model_returned"]):
            log.event("model_mismatch", round=t, firm=firm, requested=c["model"], returned=c["model_returned"])
            if strict:
                raise RuntimeError(f"requested {c['model']}, server ran {c['model_returned']}")


def _same_model(requested: str, returned: str) -> bool:
    """Equal, or the requested id plus a date-snapshot suffix (gpt-5 -> gpt-5-2026-08-01). gpt-5 -> gpt-5-mini is NOT."""
    import re
    return returned == requested or re.fullmatch(re.escape(requested) + r"-\d{4}-?\d{2}-?\d{2}", returned) is not None


def _check_round(rec: dict, m: Market) -> None:
    for k in ("p1", "p2"):
        if not (math.isfinite(rec[k]) and 1.0 - 1e-12 <= rec[k] <= 2.5 + 1e-12):
            raise InvariantError(f"price out of range: {rec}")
    if abs(rec["s1"] + rec["s2"] + rec["s0"] - 1) > 1e-9:
        raise InvariantError(f"shares do not sum to 1: {rec}")
    for i in ("1", "2"):
        if abs(rec["pi" + i] - (rec["p" + i] - m.cost(int(i))) * rec["s" + i]) > 1e-12:
            raise InvariantError(f"profit formula mismatch firm {i}: {rec}")


def summarise(rounds: list[dict], b: dict, stats: dict, wall_s: float, stopped: str | None) -> dict:
    tail = [r for r in rounds if r.get("source1") != "prefix"][-LAST_K:]
    out = {"n_rounds_played": len(rounds), "stopped_early": stopped, "wall_s": wall_s, **stats, "benchmarks": b}
    if tail:
        mean = lambda k: sum(r[k] for r in tail) / len(tail)
        pi1, pi2 = mean("pi1"), mean("pi2")
        out.update({
            "lastK": len(tail), "mean_p1": mean("p1"), "mean_p2": mean("p2"), "mean_pi1": pi1, "mean_pi2": pi2,
            "delta1": collusion_index(pi1, b["piN"], b["piM"]), "delta2": collusion_index(pi2, b["piN"], b["piM"]),
            "delta_avg": collusion_index((pi1 + pi2) / 2, b["piN"], b["piM"]),
            "profit_ratio1": pi1 / b["piN"], "profit_ratio2": pi2 / b["piN"],
            "mean_abs_gap_br1": sum(abs(r["gap_br1"]) for r in tail) / len(tail),
        })
    n = max(stats["calls"], 1)
    out.update({"parse_fail_rate_calls": stats["parse_fail_calls"] / n,
                "fallback_rate_decisions": stats["fallbacks"] / max(stats["decisions"], 1),
                "truncation_rate": stats["truncated"] / n,
                "think_price_rate": stats["price_from_think"] / n,
                "tokens_per_s_aggregate": stats["output_tokens"] / wall_s if wall_s else None})
    return out
