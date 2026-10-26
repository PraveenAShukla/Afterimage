"""One OpenAI-compatible chat backend for all four providers, as Lee did ("All four are called through the
OpenAI-compatible chat completions interface"). Temperature is NOT sent (Lee: provider default). The per-model
reasoning setting and token limit go in `extra` (from the run config), so nothing is hidden in code.
Keys come only from the environment (names end in _API_KEY, so the log scrubber redacts them); they are never logged.
"""
from __future__ import annotations

import os

from collusion.backends.base import Completion
from collusion.backends.http import _post_with_retries

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),   # Anthropic's OpenAI-SDK compatibility
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_LEE_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "ifm": ("https://api.ifm.ai/v1", "IFM_K2_API_KEY"),   # IFM K2 (OpenAI-compatible); see docs/K2_API.md
    # Qwen (our addition; not in Lee): Alibaba Model Studio, international OpenAI-compatible endpoint. Thinking is ON by
    # default for qwen3.5-flash; configs send enable_thinking: false (verified, Model Studio docs).
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),  # fallback route; not Lee's direct providers
}


def base_url(provider: str) -> str:
    """LEE_<PROVIDER>_BASE_URL overrides the default endpoint. A project-specific name on purpose: generic names such as
    ANTHROPIC_BASE_URL are often set by other tools and can silently redirect every Claude call (a 404, caught by
    the probe)."""
    return os.environ.get(f"LEE_{provider.upper()}_BASE_URL", PROVIDERS[provider][0]).rstrip("/")


class ChatBackend:
    def __init__(self, provider: str, model: str, extra: dict | None = None, timeout_s: float = 120, max_attempts: int = 6):
        if provider not in PROVIDERS:
            raise ValueError(f"provider {provider!r} not in {sorted(PROVIDERS)}")
        self.provider, self.model, self.extra = provider, model, dict(extra or {})
        if "temperature" in self.extra:
            raise ValueError("Lee does not send temperature: remove it from extra")
        self.base_url, self.key_env = base_url(provider), PROVIDERS[provider][1]
        self.timeout_s, self.max_attempts = timeout_s, max_attempts

    def complete(self, system: str, user: str) -> Completion:
        key = os.environ.get(self.key_env)
        if not key:
            return Completion(text="", error=f"{self.key_env} not set")
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                **self.extra}
        js, err, n, lat = _post_with_retries(f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {key}"},
                                             body, self.timeout_s, self.max_attempts)
        if js is None:
            return Completion(text="", error=err, transport_attempts=n, latency_s=lat)
        ch = (js.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        u = js.get("usage") or {}
        return Completion(text=msg.get("content") or "", reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                          finish_reason=ch.get("finish_reason"), input_tokens=u.get("prompt_tokens"),
                          output_tokens=u.get("completion_tokens"), latency_s=lat, model_returned=js.get("model"),
                          request_id=js.get("id"), transport_attempts=n, extra={"sent": {k: v for k, v in body.items() if k != "messages"}})


class MockChat:
    """Scripted Lee-style agent for tests: strategies 'hold' (repeat own last price), 'match_min' (min of all last
    prices), 'avg' (market mean), 'br' (static BR), 'open:<p>' (opens at p, then avg). Replies in Lee's JSON format."""
    provider = "mock"

    def __init__(self, model: str = "mock", strategy: str = "avg", market=None):
        from .market import NMarket
        self.model, self.strategy, self.m = model, strategy, market or NMarket()

    def complete(self, system: str, user: str, seed: int = 0) -> Completion:
        import re
        me = "ABCDEFGH".index(re.search(r"Firm ([A-H])\.", system).group(1))
        rows = re.findall(r"^\d+ \| ([\d. ]+) \| (-?[\d.]+) \| (-?[\d.]+)$", user, re.M)
        last = [float(x) for x in rows[-1][0].split()] if rows else None
        s = self.strategy
        if last is None:
            p = float(s.split(":")[1]) if s.startswith("open:") else 2.0
        elif s == "hold":
            p = last[me]
        elif s == "match_min":
            p = min(last)
        elif s == "br":
            p = self.m.best_response(me, last)
        else:
            p = sum(last) / len(last)
        txt = f'{{"price": {p:.4f}, "plan": "keep {s}", "insight": "round seen"}}'
        return Completion(text=txt, finish_reason="stop", input_tokens=len(system + user) // 4, output_tokens=len(txt) // 4,
                          latency_s=0.0, model_returned=self.model)
