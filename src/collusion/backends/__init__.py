from .base import Backend, Completion, Sampling
from .mock import MockBackend
from .http import AnthropicBackend, IFMBackend, OpenAIBackend, VLLMBackend


def make_backend(spec: dict) -> Backend:
    """spec: {provider: vllm|openai|anthropic|mock, model: ..., **kwargs}"""
    spec = dict(spec)
    provider = spec.pop("provider")
    cls = {"vllm": VLLMBackend, "openai": OpenAIBackend, "anthropic": AnthropicBackend, "mock": MockBackend, "ifm": IFMBackend}[provider]
    return cls(**spec)


__all__ = ["Backend", "Completion", "Sampling", "make_backend",
           "MockBackend", "VLLMBackend", "IFMBackend", "OpenAIBackend", "AnthropicBackend"]
