"""Real backends over plain HTTP (httpx). One retry policy for all of them.

- VLLMBackend:      OpenAI-compatible /v1/chat/completions served by `vllm serve --generation-config vllm`.
- OpenAIBackend:    api.openai.com chat completions.
- AnthropicBackend: api.anthropic.com Messages API.

Keys come from the environment only (loaded from .env by collusion.env). They are never logged.
"""
from __future__ import annotations

import os
import random
import time

import httpx

from .base import Backend, Completion, Sampling

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _post_with_retries(url: str, headers: dict, body: dict, timeout_s: float, max_attempts: int) -> tuple[dict | None, str | None, int, float]:
    """Returns (json, error, attempts, latency_of_successful_call). Exponential backoff with jitter."""
    err = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=timeout_s)
            lat = time.perf_counter() - t0
            if r.status_code == 200:
                try:
                    return r.json(), None, attempt, lat
                except ValueError:  # e.g. a proxy's HTML page with status 200: treat as transient
                    err, wait = f"HTTP 200 with non-JSON body: {r.text[:200]!r}", None
                    if attempt < max_attempts:
                        time.sleep(_backoff(attempt, None))
                    continue
            err = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code not in RETRYABLE:
                return None, err, attempt, lat  # 400/401/403/404: retrying cannot help
            retry_after = r.headers.get("retry-after")
            wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else None
        except (httpx.TimeoutException, httpx.TransportError) as e:
            err, wait = f"{type(e).__name__}: {e}", None
        if attempt < max_attempts:
            time.sleep(_backoff(attempt, wait))
    return None, err, max_attempts, 0.0


MAX_WAIT_S = 120.0


def _backoff(attempt: int, retry_after: float | None) -> float:
    """Exponential backoff with jitter; a server's Retry-After can lengthen it but never shorten it to 0,
    and nothing waits longer than MAX_WAIT_S (a huge Retry-After must not park a 24 h GPU job in sleep)."""
    base = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
    return min(MAX_WAIT_S, max(base, retry_after or 0.0))


def _malformed(js, err: Exception, n: int, sent: dict | None = None) -> Completion:
    return Completion(text="", error=f"malformed response ({type(err).__name__}: {err}): {str(js)[:300]}",
                      transport_attempts=n, extra={"sent_params": sent} if sent else {})


class VLLMBackend(Backend):
    provider = "vllm"

    def __init__(self, model: str, base_url: str | None = None, timeout_s: float = 600, max_attempts: int = 6,
                 extra_body: dict | None = None):
        super().__init__(model)
        self.extra_body = extra_body or {}  # e.g. {"chat_template_kwargs": {"enable_thinking": False}}; logged per call
        url = os.path.expandvars(base_url) if base_url else os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        if "$" in url:
            raise ValueError(f"base_url {base_url!r} references an unset environment variable")
        self.base_url = url.rstrip("/")
        self.timeout_s, self.max_attempts = timeout_s, max_attempts

    def complete(self, system: str, user: str, sampling: Sampling, seed: int) -> Completion:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": sampling.temperature, "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens, "seed": seed, **self.extra_body,
        }
        js, err, n, lat = _post_with_retries(f"{self.base_url}/v1/chat/completions", self._headers(), body, self.timeout_s, self.max_attempts)
        if js is None:
            return Completion(text="", error=err, transport_attempts=n)
        try:
            ch = js["choices"][0]
            msg = ch["message"]
        except (KeyError, IndexError, TypeError) as e:  # {"error": ...} with 200, empty choices, ...
            return _malformed(js, e, n)
        # If vLLM runs with a reasoning parser, thinking arrives in reasoning_content; we keep both and
        # build `text` as the full output so "parse the LAST JSON in the output" holds either way.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        content = msg.get("content") or ""
        text = f"<think>{reasoning}</think>{content}" if reasoning else content
        u = js.get("usage") or {}
        return Completion(
            text=text, reasoning=reasoning, finish_reason=ch.get("finish_reason"),
            input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"), latency_s=lat,
            model_returned=js.get("model"), request_id=js.get("id"), transport_attempts=n,
        )

    def _headers(self) -> dict:
        return {}

    def server_info(self) -> dict:
        """vLLM version + served models, saved in each run manifest."""
        info = {}
        for path in ("/version", "/v1/models"):
            try:
                info[path] = _scrub(httpx.get(self.base_url + path, timeout=10).json())
            except Exception as e:  # noqa: BLE001 - manifest must not crash the run
                info[path] = f"unavailable: {e}"
        return info


def _scrub(x):
    """Drop filesystem paths (vLLM reports the weight directory as `root`) so logs never reveal where we ran."""
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items() if k not in ("root", "parent")}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x


class IFMBackend(VLLMBackend):
    """IFM's hosted K2 (docs/K2_API.md): the same OpenAI-compatible body as vLLM, plus Bearer auth from IFM_K2_API_KEY."""
    provider = "ifm"

    def __init__(self, model: str, base_url: str | None = None, timeout_s: float = 600, max_attempts: int = 6,
                 extra_body: dict | None = None, api_key_env: str = "IFM_K2_API_KEY"):
        super().__init__(model, base_url or "https://api.ifm.ai", timeout_s, max_attempts, extra_body)
        if not api_key_env.endswith("_API_KEY"):
            raise ValueError("api_key_env must end in _API_KEY (the log scrubber redacts names ending in KEY)")
        if not os.environ.get(api_key_env):
            raise ValueError(f"{api_key_env} is not set")
        self.api_key_env = api_key_env  # the variable NAME is recorded; the value never is

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {os.environ[self.api_key_env]}"}

    def server_info(self) -> dict:
        return {"provider": "ifm", "base_url": self.base_url, "api_key_env": self.api_key_env}


class OpenAIBackend(Backend):
    """Chat Completions, shaped by a live parameter probe on gpt-6 models:
    `max_tokens` is rejected (we send max_completion_tokens); top_p is rejected (never sent unless send_top_p);
    temperature != 1 is accepted ONLY with reasoning_effort="none"; seed is accepted (best effort, not a guarantee).
    So with reasoning_effort="none" (default) we can match the spec's T=0.7; with reasoning on, temperature is not
    sent and the call is recorded as sampling_ignored. What was actually sent is logged as sent_params."""
    provider = "openai"

    def __init__(self, model: str, reasoning_effort: str | None = "none", send_top_p: bool = False,
                 verbosity: str | None = None, timeout_s: float = 300, max_attempts: int = 6):
        super().__init__(model)
        self.reasoning_effort, self.send_top_p, self.verbosity = reasoning_effort, send_top_p, verbosity
        self.timeout_s, self.max_attempts = timeout_s, max_attempts

    def complete(self, system: str, user: str, sampling: Sampling, seed: int) -> Completion:
        body: dict = {"model": self.model, "seed": seed, "max_completion_tokens": sampling.max_tokens,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        sampling_ignored = self.reasoning_effort not in (None, "none")
        if not sampling_ignored:
            body["temperature"] = sampling.temperature
        if self.send_top_p:
            body["top_p"] = sampling.top_p
        if self.verbosity:
            body["verbosity"] = self.verbosity
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        js, err, n, lat = _post_with_retries("https://api.openai.com/v1/chat/completions", headers, body, self.timeout_s, self.max_attempts)
        sent = {k: v for k, v in body.items() if k != "messages"}
        extra = {"sent_params": sent, "sampling_ignored": sampling_ignored}
        if js is None:
            return Completion(text="", error=err, transport_attempts=n, extra=extra)
        try:
            ch = js["choices"][0]
            ch["message"]
        except (KeyError, IndexError, TypeError) as e:
            return _malformed(js, e, n, sent)
        u = js.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        return Completion(
            text=ch["message"].get("content") or "", finish_reason=ch.get("finish_reason"),
            input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"), latency_s=lat,
            model_returned=js.get("model"), request_id=js.get("id"), transport_attempts=n,
            extra={**extra, "reasoning_tokens": details.get("reasoning_tokens"),
                   "system_fingerprint": js.get("system_fingerprint")},
        )


DEFAULT_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicBackend(Backend):
    """Official `anthropic` SDK. Current Claude models (Opus/Sonnet/Haiku 5.5) REJECT temperature/top_p and have
    always-on thinking; there is no seed. So sampling params are NOT sent (sampling.temperature is ignored and
    this is recorded per call), depth is set with `effort`, and runs are not bit-reproducible.
    Server-side refusal fallbacks are deliberately NOT enabled: a fallback would silently swap the model
    mid-game and contaminate the arm. A refusal is logged as finish_reason='refusal' and handled as a parse failure.
    The SDK's own retries are OFF (max_retries=0): we retry here so transport_attempts and latency are exact."""
    provider = "anthropic"

    THINKING = {"adaptive": {"type": "adaptive", "display": "summarized"},   # default; summary text is logged
                "disabled": {"type": "disabled"},                            # Haiku 5.5 only, effort <= high
                "between_tools": {"type": "between_tools"}}                  # Sonnet 5.5's "off"

    def __init__(self, model: str, effort: str = "low", thinking: str = "adaptive", timeout_s: float = 300,
                 max_attempts: int = 6, api_key_env: str = DEFAULT_ANTHROPIC_KEY_ENV):
        super().__init__(model)
        import anthropic  # lazy: only needed for API arms
        if thinking not in self.THINKING:
            raise ValueError(f"thinking must be one of {sorted(self.THINKING)}")
        if not api_key_env.startswith("ANTHROPIC") or not api_key_env.endswith("_API_KEY"):
            raise ValueError("api_key_env must be ANTHROPIC*_API_KEY (the log scrubber redacts names ending in KEY)")
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"{api_key_env} is not set")
        self.api_key_env = api_key_env  # the variable NAME is recorded; the value never is
        self.client = anthropic.Anthropic(api_key=key, timeout=timeout_s, max_retries=0)
        self.effort, self.thinking, self.max_attempts = effort, thinking, max_attempts

    def complete(self, system: str, user: str, sampling: Sampling, seed: int) -> Completion:
        import anthropic
        sent = {"model": self.model, "max_tokens": sampling.max_tokens,
                "output_config": {"effort": self.effort}, "thinking": self.THINKING[self.thinking]}
        err = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            wait = None
            try:
                r = self.client.messages.create(system=system, messages=[{"role": "user", "content": user}], **sent)
                lat = time.perf_counter() - t0
                break
            except anthropic.APIStatusError as e:
                err = f"{type(e).__name__} {e.status_code}: {str(e)[:300]}"
                if e.status_code not in RETRYABLE:
                    return Completion(text="", error=err, transport_attempts=attempt,
                                      extra={"sent_params": sent, "seed_supported": False})
                ra = e.response.headers.get("retry-after") if e.response is not None else None
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else None
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                err = f"{type(e).__name__}: {str(e)[:300]}"
            if attempt < self.max_attempts:
                time.sleep(_backoff(attempt, wait))
        else:
            return Completion(text="", error=err, transport_attempts=self.max_attempts,
                              extra={"sent_params": sent, "seed_supported": False})
        text = "".join(b.text for b in r.content if b.type == "text")
        thinking = "".join(getattr(b, "thinking", "") or "" for b in r.content if b.type == "thinking") or None
        return Completion(
            text=text, reasoning=thinking, finish_reason=r.stop_reason, input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens, latency_s=lat, model_returned=r.model,
            request_id=r._request_id, transport_attempts=attempt,
            extra={"sent_params": sent, "seed_supported": False, "sampling_ignored": True,
                   "stop_details": r.stop_details.model_dump() if getattr(r, "stop_details", None) else None},
        )
