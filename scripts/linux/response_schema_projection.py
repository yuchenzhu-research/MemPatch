"""Public-only projection of model output into the benchmark response schema."""

from __future__ import annotations

from typing import Any

from benchmark.general_taxonomy import (
    DECISIONS,
    PRIMARY_FAILURE_MODES,
    PRIMARY_MEMORY_STATUSES,
)
from benchmark.scorers_general import normalize_failure_mode


def _default_memory_status(decision: str) -> str:
    return {
        "refuse_due_to_policy": "should_not_store",
        "escalate": "blocked",
        "ask_clarification": "unresolved",
        "mark_unresolved": "unresolved",
        "use_current_memory": "current",
    }[decision]


def _is_visible_distractor(memory: dict[str, Any]) -> bool:
    return bool(memory.get("is_distractor")) or str(memory.get("text") or "").startswith(
        "Distractor info:"
    )


def project_response_schema(
    raw_response: Any,
    scenario_public_view: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return a complete response using only model output and public IDs.

    The projection never reads hidden gold. It preserves valid model fields,
    removes invented IDs, and fills missing fields with conservative defaults
    derived from the model's own decision and visible memory metadata.
    """
    raw = raw_response if isinstance(raw_response, dict) else {}
    repairs: list[str] = []

    decision = raw.get("decision")
    if decision not in DECISIONS:
        decision = "ask_clarification"
        repairs.append("decision")

    public_input = scenario_public_view.get("public_input") or {}
    memories = [m for m in public_input.get("initial_memory") or [] if isinstance(m, dict)]
    visible_memory_ids = [str(m["memory_id"]) for m in memories if m.get("memory_id")]
    raw_state = raw.get("memory_state") if isinstance(raw.get("memory_state"), dict) else {}
    default_status = _default_memory_status(decision)
    memory_state: dict[str, str] = {}
    for memory in memories:
        memory_id = memory.get("memory_id")
        if not memory_id:
            continue
        memory_id = str(memory_id)
        status = raw_state.get(memory_id)
        if status not in PRIMARY_MEMORY_STATUSES:
            status = "out_of_scope" if _is_visible_distractor(memory) else default_status
            repairs.append(f"memory_state:{memory_id}")
        memory_state[memory_id] = status
    if set(raw_state) - set(visible_memory_ids):
        repairs.append("memory_state:unknown_ids")

    event_ids = {
        str(event["event_id"])
        for event in public_input.get("event_trace") or []
        if isinstance(event, dict) and event.get("event_id")
    }
    evidence = raw.get("evidence_event_ids")
    if isinstance(evidence, str):
        evidence = [evidence]
        repairs.append("evidence_event_ids:type")
    if not isinstance(evidence, list):
        evidence = []
        repairs.append("evidence_event_ids:type")
    filtered_evidence = list(
        dict.fromkeys(value for value in evidence if isinstance(value, str) and value in event_ids)
    )
    if filtered_evidence != evidence:
        repairs.append("evidence_event_ids:unknown_ids")

    diagnosis = normalize_failure_mode(raw.get("failure_diagnosis"))
    if diagnosis not in PRIMARY_FAILURE_MODES:
        diagnosis = {
            "refuse_due_to_policy": "policy_violation",
            "escalate": "conflict_collapse",
            "mark_unresolved": "conflict_collapse",
        }.get(decision, "memory_hallucination")
        repairs.append("failure_diagnosis")

    answer = raw.get("answer")
    if not isinstance(answer, str):
        answer = ""
        repairs.append("answer")

    return (
        {
            "answer": answer,
            "decision": decision,
            "memory_state": memory_state,
            "evidence_event_ids": filtered_evidence,
            "failure_diagnosis": diagnosis,
        },
        list(dict.fromkeys(repairs)),
    )
