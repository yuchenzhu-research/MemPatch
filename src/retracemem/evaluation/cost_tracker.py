from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostTracker:
    tokens: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def add_tokens(self, key: str, count: int) -> None:
        if count < 0:
            raise ValueError("token count must be non-negative")
        self.tokens[key] = self.tokens.get(key, 0) + count

    def add_call(self, key: str, count: int = 1) -> None:
        if count < 0:
            raise ValueError("call count must be non-negative")
        self.calls[key] = self.calls.get(key, 0) + count

    def merge(self, other: "CostTracker") -> None:
        for key, count in other.tokens.items():
            self.add_tokens(key, count)
        for key, count in other.calls.items():
            self.add_call(key, count)

    def total_tokens(self) -> int:
        return sum(self.tokens.values())

    def total_calls(self) -> int:
        return sum(self.calls.values())

    def reset(self) -> None:
        self.tokens.clear()
        self.calls.clear()

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {"tokens": dict(self.tokens), "calls": dict(self.calls)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CostTracker":
        tracker = cls()
        for key, count in dict(payload.get("tokens", {})).items():
            tracker.add_tokens(str(key), int(count))
        for key, count in dict(payload.get("calls", {})).items():
            tracker.add_call(str(key), int(count))
        return tracker
