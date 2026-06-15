"""Direct typed-action projection used only for the no-DPA ablation."""

from __future__ import annotations

from typing import Any

from benchmark.general_taxonomy import PRIMARY_FAILURE_MODES
from mempatch.dpa.methods.contracts import SharedCandidateView
from mempatch.revision.runtime.dpa_runtime import ParseResult


def _raw_field(raw_response: Any, key: str) -> Any:
    if not isinstance(raw_response, dict):
        return None
    nested = raw_response.get("response")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return raw_response.get(key)


def project_actions_without_dpa(
    *,
    view: SharedCandidateView,
    parse_result: ParseResult,
    raw_response: dict[str, Any] | None,
    scenario_public_view: dict[str, Any],
) -> dict[str, Any]:
    """Project parsed actions without gate admission or defeat-path authorization."""
    memories = (scenario_public_view.get("public_input") or {}).get("initial_memory") or []
    memory_state = {
        str(memory["memory_id"]): (
            "out_of_scope"
            if str(memory.get("text") or "").startswith("Distractor info:")
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
                and status in {"out_of_scope", "should_not_store"}
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
    if not isinstance(diagnosis, str) or diagnosis not in PRIMARY_FAILURE_MODES:
        diagnosis = "memory_hallucination"
    answer = _raw_field(raw_response, "answer")
    return {
        "answer": answer if isinstance(answer, str) else "",
        "decision": decision,
        "memory_state": memory_state,
        "evidence_event_ids": list(dict.fromkeys(evidence_ids)),
        "failure_diagnosis": diagnosis,
    }
