from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retracemem.providers.base import BaseLLMProvider


@dataclass
class CapUsage:
    outbound_network_calls: int = 0
    tokens_from_outbound_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "outbound_network_calls": self.outbound_network_calls,
            "tokens_from_outbound_calls": self.tokens_from_outbound_calls,
        }


class CappedProviderWrapper(BaseLLMProvider):
    def __init__(self, inner: BaseLLMProvider, max_calls: int, max_tokens: int) -> None:
        self.inner = inner
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.usage = CapUsage()

    @property
    def calls_made(self) -> int:
        return self.usage.outbound_network_calls

    @property
    def tokens_used(self) -> int:
        return self.usage.tokens_from_outbound_calls

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if self.usage.outbound_network_calls >= self.max_calls:
            raise RuntimeError(f"Hard call cap of {self.max_calls} reached.")
        if self.usage.tokens_from_outbound_calls >= self.max_tokens:
            raise RuntimeError(f"Hard token cap of {self.max_tokens} reached.")
        trace = self.inner.generate(*args, **kwargs)
        self.usage.outbound_network_calls += 1
        self.usage.tokens_from_outbound_calls += trace.total_tokens
        if self.usage.tokens_from_outbound_calls >= self.max_tokens:
            raise RuntimeError(f"Hard token cap of {self.max_tokens} reached during call.")
        return trace

    def to_dict(self) -> dict[str, int]:
        return {
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            **self.usage.to_dict(),
        }
