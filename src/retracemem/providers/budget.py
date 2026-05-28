from __future__ import annotations

from typing import Any


class GlobalBudget:
    def __init__(self, max_calls: int, max_tokens: int) -> None:
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls = 0
        self.tokens = 0
        self.calls_by_stage: dict[str, int] = {}
        self.tokens_by_stage: dict[str, int] = {}

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"Global call cap reached: {self.calls}/{self.max_calls}")
        if self.tokens >= self.max_tokens:
            raise RuntimeError(f"Global token cap reached: {self.tokens}/{self.max_tokens}")

    def record(self, total_tokens: int, stage: str) -> None:
        self.calls += 1
        self.tokens += total_tokens
        self.calls_by_stage[stage] = self.calls_by_stage.get(stage, 0) + 1
        self.tokens_by_stage[stage] = self.tokens_by_stage.get(stage, 0) + total_tokens

    def summary(self) -> dict[str, Any]:
        return {
            "total_calls": self.calls,
            "total_tokens": self.tokens,
            "max_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            "calls_by_stage": dict(self.calls_by_stage),
            "tokens_by_stage": dict(self.tokens_by_stage),
        }


class BudgetWrappedProvider:
    def __init__(self, inner: Any, budget: GlobalBudget, stage: str) -> None:
        self.inner = inner
        self.budget = budget
        self.stage = stage

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.budget.check()
        trace = self.inner.generate(*args, **kwargs)
        self.budget.record(trace.total_tokens, self.stage)
        return trace
