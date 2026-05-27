from __future__ import annotations

from typing import Any

from retracemem.evaluation.cost_tracker import CostTracker
from retracemem.evaluation.jsonl import records_to_jsonable
from retracemem.schemas import EvaluationRecord


def evaluation_record_from_backend_output(
    *,
    query_id: str,
    method: str,
    retrieved: list[dict[str, Any]] | None = None,
    answer: str | None = None,
    cost: CostTracker | dict[str, Any] | None = None,
    latency_ms: int | None = None,
    candidate_beliefs: list[dict[str, Any]] | None = None,
    authorized_basis: list[dict[str, Any]] | None = None,
    blocked_beliefs: list[dict[str, Any]] | None = None,
) -> EvaluationRecord:
    """Build the shared JSONL evaluation schema from backend outputs."""

    tokens, calls = _cost_parts(cost)
    return EvaluationRecord(
        query_id=query_id,
        method=method,
        retrieved_evidence=records_to_jsonable(retrieved or []),
        candidate_beliefs=records_to_jsonable(candidate_beliefs or []),
        authorized_basis=records_to_jsonable(authorized_basis or retrieved or []),
        blocked_beliefs=records_to_jsonable(blocked_beliefs or []),
        answer=answer,
        tokens=tokens,
        calls=calls,
        latency_ms=latency_ms,
    )


def _cost_parts(cost: CostTracker | dict[str, Any] | None) -> tuple[dict[str, int], dict[str, int]]:
    if cost is None:
        return {}, {}
    if isinstance(cost, CostTracker):
        return dict(cost.tokens), dict(cost.calls)

    tokens = {str(key): int(value) for key, value in dict(cost.get("tokens", {})).items()}
    calls = {str(key): int(value) for key, value in dict(cost.get("calls", {})).items()}
    return tokens, calls
