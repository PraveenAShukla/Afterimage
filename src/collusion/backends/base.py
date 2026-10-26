"""Backend interface. Every backend returns a Completion with everything we can log about the call."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Sampling:
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2048


@dataclass
class Completion:
    text: str                       # full raw output (incl. <think> for R1 if the server returns it inline)
    reasoning: str | None = None    # separate reasoning channel, if the API exposes one
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_s: float | None = None
    model_returned: str | None = None   # what the server says it ran (catches wrong-model bugs)
    request_id: str | None = None
    error: str | None = None            # transport error after all retries; text is "" then
    transport_attempts: int = 1
    extra: dict = field(default_factory=dict)


class Backend(ABC):
    provider: str  # "vllm" | "anthropic" | "openai" | "mock"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(self, system: str, user: str, sampling: Sampling, seed: int) -> Completion:
        """One chat call. Must never raise for transport errors: return Completion(error=...)."""

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model}
