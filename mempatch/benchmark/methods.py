"""Model-visible context construction for the MemPatch experiment campaign."""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import Counter
from typing import Any

from mempatch.benchmark.method_names import normalize_method_name

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
HASH_DIM = 256


def _tokens(value: Any) -> list[str]:
    return TOKEN_RE.findall(str(value).lower())


def _query_text(view: dict[str, Any]) -> str:
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [text for item in value.values() for text in strings(item)]
        if isinstance(value, list):
            return [text for item in value for text in strings(item)]
        return []

    parts = strings(view.get("workflow_context", ""))
    for key in (
        "black_box_task",
        "memory_state_task",
        "evidence_retrieval_task",
        "diagnostic_task",
    ):
        parts.extend(strings(view.get(key)))
    return " ".join(parts)


def _events(view: dict[str, Any]) -> list[dict[str, Any]]:
    public_input = view.get("public_input") or {}
    return list(public_input.get("event_trace", public_input.get("events", [])) or [])


def _event_text(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(key, ""))
        for key in ("text", "content", "actor_role", "event_type", "trust_level", "visibility_scope")
    )


def _hash_embedding(text: str, *, dim: int = HASH_DIM) -> list[float]:
    """Small local embedding fallback for dense_rag_json.

    This is intentionally dependency-free and deterministic.  It is not meant
    to be a state-of-the-art retriever; it gives the final method a real dense
    cosine-retrieval path without proprietary APIs or model downloads.
    """
    vector = [0.0] * dim
    tokens = _tokens(text)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = -1.0 if digest[4] % 2 else 1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _with_events(
    view: dict[str, Any],
    events: list[dict[str, Any]],
    instruction: str,
    *,
    method: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(view)
    result.setdefault("public_input", {})["event_trace"] = events
    result.setdefault("public_input", {})["events"] = events
    result["context_policy"] = instruction
    if method is not None:
        result["retrieval_metadata"] = {
            "method": method,
            "retrieved_event_count": len(events),
            "retrieved_event_ids": [
                str(event.get("event_id"))
                for event in events
                if isinstance(event, dict) and event.get("event_id")
            ],
        }
    return result


def _effective_retrieval_k(k: int, event_count: int) -> int:
    if event_count <= 1:
        return event_count
    return max(1, min(k, event_count - 1))


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
        method="full_context_json",
    )


def lexical_rag(view: dict[str, Any], k: int = 3) -> dict[str, Any]:
    events = _events(view)
    k = _effective_retrieval_k(k, len(events))
    query_counts = Counter(_tokens(_query_text(view)))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, event in enumerate(events):
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
        method="bm25_rag_json",
    )


def dense_rag(view: dict[str, Any], k: int = 3) -> dict[str, Any]:
    events = _events(view)
    k = _effective_retrieval_k(k, len(events))
    query_vector = _hash_embedding(_query_text(view))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, event in enumerate(events):
        event_vector = _hash_embedding(_event_text(event))
        scored.append((_cosine(query_vector, event_vector), -index, event))
    selected = [item[2] for item in sorted(scored, reverse=True)[:k]]
    selected.sort(key=lambda event: (event.get("timestamp_order", 0), str(event.get("event_id", ""))))
    return _with_events(
        view,
        selected,
        f"Use the top-{k} events selected by deterministic local dense hash retrieval.",
        method="dense_rag_json",
    )


def time_aware_rag(view: dict[str, Any], k: int = 3) -> dict[str, Any]:
    events = _events(view)
    k = _effective_retrieval_k(k, len(events))
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
        method="time_aware_rag_json",
    )


def summary_memory(view: dict[str, Any], max_chars: int = 2400) -> dict[str, Any]:
    events = sorted(
        _events(view),
        key=lambda event: (event.get("timestamp_order", 0), str(event.get("event_id", ""))),
    )
    lines = []
    for event in events:
        text = " ".join(str(event.get("text") or event.get("content") or "").split())
        if len(text) > 180:
            text = text[:177] + "..."
        lines.append(
            f"- t{event.get('timestamp_order', '?')} "
            f"[{event.get('event_id', '?')}] {text}"
        )
    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    result = _with_events(
        view,
        [],
        "Use the deterministic chronological memory summary instead of the raw event trace.",
        method="summary_memory_json",
    )
    result["memory_summary"] = {
        "format": "chronological_extractive_summary",
        "text": summary,
    }
    return result


def build_method_view(
    method: str,
    view: dict[str, Any],
    retrieval_k: int,
) -> dict[str, Any]:
    final_method = normalize_method_name(method)
    if final_method == "direct_json":
        result = copy.deepcopy(view)
        result["context_policy"] = (
            "Answer directly from the frozen public input without retrieval or memory revision."
        )
        result["retrieval_metadata"] = {
            "method": "direct_json",
            "retrieved_event_count": len(_events(result)),
            "retrieved_event_ids": [
                str(event.get("event_id"))
                for event in _events(result)
                if isinstance(event, dict) and event.get("event_id")
            ],
        }
        return result
    if final_method == "full_context_json":
        return full_context(view)
    if final_method == "bm25_rag_json":
        return lexical_rag(view, retrieval_k)
    if final_method == "dense_rag_json":
        return dense_rag(view, retrieval_k)
    if final_method == "time_aware_rag_json":
        return time_aware_rag(view, retrieval_k)
    if final_method == "summary_memory_json":
        return summary_memory(view)
    raise ValueError(f"Unknown context method: {method}")
