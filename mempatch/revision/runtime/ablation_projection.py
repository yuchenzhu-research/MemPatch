"""Direct typed-action projection used only for the no-DPA ablation."""

from __future__ import annotations

from typing import Any

from mempatch.benchmark.general_taxonomy import FAILURE_MODES, MEMORY_OPERATIONS
from mempatch.dpa.methods.contracts import SharedCandidateView
from mempatch.revision.runtime.dpa_runtime import ParseResult


def _raw_field(raw_response: Any, key: str) -> Any:
    if not isinstance(raw_response, dict):
        return None
    nested = raw_response.get("response")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return raw_response.get(key)


def _item_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("content") or "")


def project_actions_without_dpa(
    *,
    view: SharedCandidateView,
    parse_result: ParseResult,
    raw_response: dict[str, Any] | None,
    scenario_public_view: dict[str, Any],
) -> dict[str, Any]:
    """Project parsed actions without gate admission or defeat-path authorization."""
    public_input = scenario_public_view.get("public_input") or {}
    memories = public_input.get("initial_memory") or public_input.get("initial_memories") or []
    memory_state = {
        str(memory["memory_id"]): (
            "out_of_scope"
            if _item_text(memory).startswith("Distractor info:")
            else "current"
        )
        for memory in memories
        if isinstance(memory, dict) and memory.get("memory_id")
    }
    dependent_beliefs: dict[str, list[str]] = {}
    for belief_id, dependencies in view.dependency_edges_by_belief:
        for dependency in dependencies:
            dependent_beliefs.setdefault(dependency.condition_id, []).append(belief_id)

    evidence_ids: list[str] = []
    for action in parse_result.actions:
        evidence_ids.extend(action.evidence_ids)
        if action.action_type == "BLOCKS" and action.target_condition_id:
            for belief_id in dependent_beliefs.get(action.target_condition_id, []):
                memory_state[belief_id] = "blocked"
        elif action.action_type == "RELEASES" and action.target_condition_id:
            for belief_id in dependent_beliefs.get(action.target_condition_id, []):
                memory_state[belief_id] = "current"
        elif action.action_type == "UNCERTAIN" and action.target_belief_id in memory_state:
            memory_state[action.target_belief_id] = "unresolved"
        elif action.action_type in {"REAFFIRMS", "SUPERSEDES"}:
            if action.target_belief_id in memory_state:
                memory_state[action.target_belief_id] = "current"

    raw_state = _raw_field(raw_response, "memory_state")
    if isinstance(raw_state, dict):
        for memory_id, status in raw_state.items():
            if (
                isinstance(status, str)
                and status in {"out_of_scope", "should_not_store", "outdated", "deleted", "restored"}
                and str(memory_id) in memory_state
            ):
                memory_state[str(memory_id)] = str(status)

    raw_decision = _raw_field(raw_response, "decision")
    if "should_not_store" in memory_state.values():
        decision = "refuse_due_to_policy"
    elif "unresolved" in memory_state.values():
        decision = "ask_clarification" if raw_decision == "ask_clarification" else "mark_unresolved"
    elif "blocked" in memory_state.values():
        decision = "ask_clarification" if raw_decision == "ask_clarification" else "escalate"
    else:
        decision = "use_current_memory"

    diagnosis = _raw_field(raw_response, "failure_diagnosis")
    if not isinstance(diagnosis, str) or diagnosis not in FAILURE_MODES:
        diagnosis = "memory_hallucination"
    memory_operation = _raw_field(raw_response, "memory_operation")
    if not isinstance(memory_operation, str) or memory_operation not in MEMORY_OPERATIONS:
        action_types = {action.action_type for action in parse_result.actions}
        if "should_not_store" in memory_state.values():
            memory_operation = "BLOCK"
        elif "deleted" in memory_state.values():
            memory_operation = "DELETE_OR_FORGET"
        elif "restored" in memory_state.values():
            memory_operation = "RESTORE_OR_RELEASE"
        elif "out_of_scope" in memory_state.values():
            memory_operation = "RESTRICT_SCOPE"
        elif "unresolved" in memory_state.values():
            memory_operation = "MARK_UNRESOLVED"
        elif "blocked" in memory_state.values():
            memory_operation = "ESCALATE"
        elif "RELEASES" in action_types:
            memory_operation = "RESTORE_OR_RELEASE"
        elif "BLOCKS" in action_types:
            memory_operation = "BLOCK"
        elif "UNCERTAIN" in action_types:
            memory_operation = "MARK_UNRESOLVED"
        elif "SUPERSEDES" in action_types:
            memory_operation = "REVISE"
        else:
            memory_operation = "PRESERVE"
    answer = _raw_field(raw_response, "answer")
    followup_answer = _raw_field(raw_response, "followup_answer")
    if not isinstance(followup_answer, str):
        followup_answer = answer if isinstance(answer, str) else ""
    return {
        "answer": answer if isinstance(answer, str) else "",
        "decision": decision,
        "memory_operation": memory_operation,
        "memory_state": memory_state,
        "evidence_event_ids": list(dict.fromkeys(evidence_ids)),
        "failure_diagnosis": diagnosis,
        "followup_answer": followup_answer,
    }
