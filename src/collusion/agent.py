"""A pricing agent: backend + prompt + sampling + the spec's parse/retry/fallback/clip rules.

decide() never raises on model misbehaviour. Every attempt (including retries) is returned so the runner
can log one JSONL line per call, as docs/DESIGN.md 3.6 requires.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

from .backends import Backend, Sampling
from .budget import Budget, BudgetExceeded
from .ratelimit import get_limiter
from .parse import STATUS_OK, clip, parse_price, price_in_think
from .prompts import Identity, system_prompt, user_message

MAX_PARSE_RETRIES = 2  # spec: retry up to 2 times, then repeat previous price


def call_seed(run_seed: int, firm: int, t: int, attempt: int) -> int:
    """Deterministic, collision-resistant per-call seed. Same run seed -> same call seeds, independent of
    thread scheduling (this is what makes vLLM runs reproducible)."""
    h = hashlib.sha256(f"{run_seed}|{firm}|{t}|{attempt}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


@dataclass
class Decision:
    price: float              # price actually played (after fallback and clipping)
    raw_price: float | None   # parsed number before clipping (None if every attempt failed)
    status: str               # ok | ok_after_retry | fallback_previous | fallback_initial
    clipped: bool
    calls: list[dict] = field(default_factory=list)


@dataclass
class Agent:
    backend: Backend
    prompt_variant: str = "E1"
    sampling: Sampling = field(default_factory=Sampling)
    identity: Identity | None = None
    view: dict | None = None  # prompt control variant (label / anchor); None = spec prompt
    cost: float = 1.0  # this firm's marginal cost, stated in its system prompt (plan v2 M6)
    supervision: str | None = None  # "shared" / "individual" monitoring line in the system prompt (SUPERVISION)
    note: str | None = None  # supervisor's latest note to this firm; set by game.play before each round
    review: tuple | None = None  # (frame, label) from a review/yoked supervisor (v2); set by game.play before each round
    pressure: str | None = None  # "gain" / "loss" personal-pressure line in the system prompt (PRESSURE)
    hint: tuple | None = None  # (r, b) for the br_hint view line; set by game.play before each round
    initial_fallback_price: float = 1.4729  # only used if round 1 fails entirely (logged as fallback_initial)

    def __post_init__(self):
        self.system = system_prompt(self.prompt_variant, self.identity, self.cost, self.supervision, self.pressure)

    def decide(self, t: int, history: list[dict], prev_price: float | None, run_seed: int, firm: int,
               budget: Budget | None = None, run_id: str = "") -> Decision:
        user = user_message(t, history, self.view, self.note, self.review, self.hint)
        calls = []
        for attempt in range(MAX_PARSE_RETRIES + 1):
            seed = call_seed(run_seed, firm, t, attempt)
            res = None
            if budget is not None:
                try:
                    res = budget.reserve(self.backend.provider, self.backend.model,
                                         (len(self.system) + len(user)) // 3, self.sampling.max_tokens)
                except BudgetExceeded as e:
                    e.calls = calls  # the attempts already made this decision must still be logged
                    raise
            lim = get_limiter(self.backend.provider)  # strict RPM / output-TPM window for paid providers
            ev = lim.acquire(self.sampling.max_tokens) if lim else None
            c = self.backend.complete(self.system, user, self.sampling, seed)
            if lim:
                lim.settle(ev, c.output_tokens)
            usd = budget.record(self.backend.provider, self.backend.model, run_id, c.input_tokens, c.output_tokens, res) if budget else 0.0
            price, status = parse_price(c.text)
            if c.error:
                status = "transport_error"
            calls.append({
                "attempt": attempt, "call_seed": seed, "model": self.backend.model, "provider": self.backend.provider, "system_prompt": self.system, "user_prompt": user,
                "raw_output": c.text, "reasoning": c.reasoning, "parsed_price": price, "parse_status": status,
                "price_from_think": price_in_think(c.text) if status == STATUS_OK else None,
                "finish_reason": c.finish_reason, "truncated": c.finish_reason in ("length", "max_tokens"),
                "input_tokens": c.input_tokens, "output_tokens": c.output_tokens, "latency_s": c.latency_s,
                "tokens_per_s": (c.output_tokens / c.latency_s) if c.output_tokens and c.latency_s else None,
                "model_returned": c.model_returned, "request_id": c.request_id, "transport_error": c.error,
                "transport_attempts": c.transport_attempts, "usd": usd, "backend_extra": c.extra,
                "sampling": asdict(self.sampling), "prompt_variant": self.prompt_variant,
                "identity": asdict(self.identity) if self.identity else None,
            })
            if status == STATUS_OK:
                played, was_clipped = clip(price)
                return Decision(played, price, "ok" if attempt == 0 else "ok_after_retry", was_clipped, calls)
        if prev_price is not None:
            return Decision(prev_price, None, "fallback_previous", False, calls)
        return Decision(self.initial_fallback_price, None, "fallback_initial", False, calls)
