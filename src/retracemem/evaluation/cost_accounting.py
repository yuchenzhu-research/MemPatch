from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class CostAccounting:
    latency_ms: float = 0.0
    tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0})
    error_counts: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    configurations: list[dict[str, Any]] = field(default_factory=list)

    def record_call(
        self,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        status: str,
        is_cache_hit: bool,
        config: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> None:
        if is_cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        self.latency_ms += latency_ms
        self.tokens["prompt"] = self.tokens.get("prompt", 0) + prompt_tokens
        self.tokens["completion"] = self.tokens.get("completion", 0) + completion_tokens
        self.tokens["total"] = self.tokens.get("total", 0) + prompt_tokens + completion_tokens
        
        # Increment general calls by status
        self.calls[status] = self.calls.get(status, 0) + 1
        self.calls["total"] = self.calls.get("total", 0) + 1
        
        if status != "success" and error_type:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
        if config:
            self.configurations.append(config)

    def reset(self) -> None:
        self.latency_ms = 0.0
        self.tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.error_counts.clear()
        self.calls.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.configurations.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "tokens": dict(self.tokens),
            "error_counts": dict(self.error_counts),
            "calls": dict(self.calls),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "configurations": list(self.configurations),
        }
