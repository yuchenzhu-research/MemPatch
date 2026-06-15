"""Model-visible context construction for the AAAI-27 experiment campaign."""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(value: Any) -> list[str]:
    return TOKEN_RE.findall(str(value).lower())


def _query_text(view: dict[str, Any]) -> str:
    parts = [
        view.get("workflow_context", ""),
        view.get("black_box_task", ""),
        view.get("memory_state_task", ""),
        view.get("evidence_retrieval_task", ""),
        view.get("diagnostic_task", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def _events(view: dict[str, Any]) -> list[dict[str, Any]]:
    return list((view.get("public_input") or {}).get("event_trace") or [])


def _event_text(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(key, ""))
        for key in ("text", "actor_role", "event_type", "trust_level", "visibility_scope")
    )


def _with_events(
    view: dict[str, Any],
    events: list[dict[str, Any]],
    instruction: str,
) -> dict[str, Any]:
    result = copy.deepcopy(view)
    result.setdefault("public_input", {})["event_trace"] = events
    result["context_policy"] = instruction
    return result


def full_context(view: dict[str, Any]) -> dict[str, Any]:
    events = sorted(
        _events(view),
        key=lambda event: (
            event.get("timestamp_order", 0),
            str(event.get("event_id", "")),
        ),
    )
    return _with_events(
        view,
        events,
        "Use the complete event trace in timestamp order. Resolve conflicts before answering.",
    )


def lexical_rag(view: dict[str, Any], k: int = 8) -> dict[str, Any]:
    query_counts = Counter(_tokens(_query_text(view)))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, event in enumerate(_events(view)):
        event_counts = Counter(_tokens(_event_text(event)))
        overlap = sum(min(count, event_counts[token]) for token, count in query_counts.items())
        normalization = math.sqrt(max(sum(event_counts.values()), 1))
        score = overlap / normalization
        scored.append((score, -index, event))
    selected = [item[2] for item in sorted(scored, reverse=True)[:k]]
    selected.sort(key=lambda event: (event.get("timestamp_order", 0), str(event.get("event_id", ""))))
    return _with_events(
        view,
        selected,
        f"Use the top-{k} events selected by deterministic lexical retrieval.",
    )


def time_aware_rag(view: dict[str, Any], k: int = 8) -> dict[str, Any]:
    events = _events(view)
    query_counts = Counter(_tokens(_query_text(view)))
    max_order = max((int(event.get("timestamp_order", 0)) for event in events), default=0)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, event in enumerate(events):
        event_counts = Counter(_tokens(_event_text(event)))
        lexical = sum(min(count, event_counts[token]) for token, count in query_counts.items())
        order = int(event.get("timestamp_order", 0))
        recency = (order + 1) / (max_order + 1 if max_order >= 0 else 1)
        score = lexical + 0.75 * recency
        scored.append((score, -index, event))
    selected = [item[2] for item in sorted(scored, reverse=True)[:k]]
    selected.sort(key=lambda event: (event.get("timestamp_order", 0), str(event.get("event_id", ""))))
    return _with_events(
        view,
        selected,
        f"Use the top-{k} events selected by lexical relevance with an explicit recency prior.",
    )


def summary_memory(view: dict[str, Any]) -> dict[str, Any]:
    events = sorted(
        _events(view),
        key=lambda event: (event.get("timestamp_order", 0), str(event.get("event_id", ""))),
    )
    summary = [
        {
            "event_id": event.get("event_id"),
            "timestamp_order": event.get("timestamp_order"),
            "actor_role": event.get("actor_role"),
            "event_type": event.get("event_type"),
            "related_memory_ids": event.get("related_memory_ids", []),
            "summary": str(event.get("text", "")).strip(),
        }
        for event in events
    ]
    result = _with_events(
        view,
        [],
        "Use the deterministic chronological memory summary instead of the raw event trace.",
    )
    result["memory_summary"] = summary
    return result


def build_method_view(
    method: str,
    view: dict[str, Any],
    retrieval_k: int,
) -> dict[str, Any]:
    if method == "frozen_direct":
        result = copy.deepcopy(view)
        result["context_policy"] = (
            "Answer directly from the frozen public input without retrieval or memory revision."
        )
        return result
    if method == "full_context":
        return full_context(view)
    if method == "lexical_rag":
        return lexical_rag(view, retrieval_k)
    if method == "time_aware_rag":
        return time_aware_rag(view, retrieval_k)
    if method == "summary_memory":
        return summary_memory(view)
    raise ValueError(f"Unknown context method: {method}")
