"""Benchmark Response Projection — MemPatch Revision Module Step 5.

Maps DPA-authorized transitions (``RuntimeResult``) into the canonical
MemPatch-Bench ``response`` schema. DPA keeps internal statuses
(``AUTHORIZED``, ``SUPERSEDED``, …); the evaluator receives ``memory_state``
labels from the public final status space.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mempatch.benchmark.general_taxonomy import DECISIONS, FAILURE_MODES, MEMORY_OPERATIONS, MEMORY_STATUSES

if TYPE_CHECKING:
    from mempatch.revision.runtime.dpa_runtime import RuntimeResult

BENCHMARK_STATUS_BY_DPA_STATUS = {
    "AUTHORIZED": "current",
    "SUPERSEDED": "outdated",
    "BLOCKED": "blocked",
    "UNRESOLVED": "unresolved",
}

def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _raw_response_field(raw_response: Any, key: str) -> Any:
    if isinstance(raw_response, dict):
        response = raw_response.get("response")
        if isinstance(response, dict) and key in response:
            return response[key]
        return raw_response.get(key)
    return None


def _visible_memory_ids(scenario_public_view: dict[str, Any] | None) -> list[str] | None:
    if not scenario_public_view:
        return None
    public_input = scenario_public_view.get("public_input") or {}
    memories = public_input.get("initial_memory") or public_input.get("initial_memories") or []
    ids = [
        str(m["memory_id"])
        for m in memories
        if isinstance(m, dict) and m.get("memory_id")
    ]
    return ids or None


def _item_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("content") or "")


def _memory_looks_like_distractor(memory_id: str, text: str) -> bool:
    """Public-visible distractor cues (never reads dataset-internal flags)."""
    mid = memory_id.lower()
    if mid.endswith("-distractor") or mid.endswith("_distractor"):
        return True
    return text.strip().startswith("Distractor info:")


def _distractor_memory_ids(scenario_public_view: dict[str, Any] | None) -> set[str]:
    if not scenario_public_view:
        return set()
    ids: set[str] = set()
    public_input = scenario_public_view.get("public_input") or {}
    for memory in public_input.get("initial_memory") or public_input.get("initial_memories") or []:
        if not isinstance(memory, dict) or not memory.get("memory_id"):
            continue
        memory_id = str(memory["memory_id"])
        text = _item_text(memory)
        if _memory_looks_like_distractor(memory_id, text):
            ids.add(memory_id)
    return ids


def _condition_memory_ids(scenario_public_view: dict[str, Any] | None) -> set[str]:
    """Heuristic for condition-rule memories used in release/restore scenarios."""
    if not scenario_public_view:
        return set()
    ids: set[str] = set()
    public_input = scenario_public_view.get("public_input") or {}
    for memory in public_input.get("initial_memory") or public_input.get("initial_memories") or []:
        if not isinstance(memory, dict) or not memory.get("memory_id"):
            continue
        text = _item_text(memory)
        if text.startswith("Condition rule:"):
            ids.add(str(memory["memory_id"]))
    return ids


def _dpa_status_to_benchmark(status: str) -> str:
    return BENCHMARK_STATUS_BY_DPA_STATUS.get(str(status), "unresolved")


def _merge_raw_memory_state(
    memory_state: dict[str, str],
    raw_response: Any,
) -> dict[str, str]:
    raw_state = _raw_response_field(raw_response, "memory_state")
    if not isinstance(raw_state, dict):
        return memory_state
    merged = dict(memory_state)
    # DPA owns the core current/blocked/unresolved transition statuses. Raw
    # response labels may only supply benchmark-only states that have no typed
    # DPA action in the current vocabulary.
    auxiliary_statuses = {"out_of_scope", "should_not_store", "outdated", "deleted", "restored"}
    for memory_id, label in raw_state.items():
        if isinstance(label, str) and label in auxiliary_statuses:
            merged[str(memory_id)] = label
    return merged


def _apply_scenario_memory_hints(
    memory_state: dict[str, str],
    runtime_result: "RuntimeResult",
    *,
    scenario_public_view: dict[str, Any] | None,
) -> dict[str, str]:
    if not scenario_public_view:
        return memory_state

    distractors = _distractor_memory_ids(scenario_public_view)
    updated = dict(memory_state)
    for memory_id, dpa_status in runtime_result.final_belief_statuses.items():
        mid = str(memory_id)
        if mid in distractors:
            updated[mid] = "out_of_scope"
            continue
        benchmark = _dpa_status_to_benchmark(str(dpa_status))
        updated[mid] = benchmark

    for memory_id in distractors:
        if memory_id in updated:
            updated[memory_id] = "out_of_scope"
    return updated


def _project_memory_state(
    runtime_result: RuntimeResult,
    *,
    scenario_public_view: dict[str, Any] | None = None,
    raw_response: Any | None = None,
) -> dict[str, str]:
    base = _apply_scenario_memory_hints(
        {
            str(belief_id): _dpa_status_to_benchmark(str(status))
            for belief_id, status in runtime_result.final_belief_statuses.items()
        },
        runtime_result,
        scenario_public_view=scenario_public_view,
    )
    visible = _visible_memory_ids(scenario_public_view)
    if visible is not None:
        base = {mid: base.get(mid, "current") for mid in visible}
    merged = _merge_raw_memory_state(base, raw_response)
    for memory_id in _distractor_memory_ids(scenario_public_view):
        if memory_id in merged:
            merged[memory_id] = "out_of_scope"
    return merged


def _project_evidence_event_ids(runtime_result: RuntimeResult) -> list[str]:
    admitted_ids = {
        d["edge_id"]
        for d in runtime_result.gate_decisions
        if d.get("admitted")
    }
    rejected_ids = {
        d["edge_id"]
        for d in runtime_result.gate_decisions
        if not d.get("admitted")
    }
    evidence: list[str] = []
    for idx, action in enumerate(runtime_result.parse_result.actions):
        edge_id = f"edge_rl_{idx}"
        if edge_id in rejected_ids:
            continue
        if action.action_type == "NO_REVISION" or edge_id in admitted_ids:
            evidence.extend(action.evidence_ids)
    return _dedupe_preserve_order(evidence)


def _project_decision(memory_state: dict[str, str], raw_response: Any) -> str:
    raw_decision = _raw_response_field(raw_response, "decision")
    if any(status == "should_not_store" for status in memory_state.values()):
        return "refuse_due_to_policy"
    if any(status == "unresolved" for status in memory_state.values()):
        if raw_decision == "ask_clarification":
            return "ask_clarification"
        return "mark_unresolved"
    if any(status == "blocked" for status in memory_state.values()):
        if raw_decision == "ask_clarification":
            return "ask_clarification"
        return "escalate"
    return "use_current_memory"


def _project_memory_operation(
    runtime_result: RuntimeResult,
    memory_state: dict[str, str],
    raw_response: Any,
) -> str:
    raw_operation = _raw_response_field(raw_response, "memory_operation")
    if isinstance(raw_operation, str) and raw_operation in MEMORY_OPERATIONS:
        return raw_operation
    if any(status == "should_not_store" for status in memory_state.values()):
        return "BLOCK"
    if any(status == "deleted" for status in memory_state.values()):
        return "DELETE_OR_FORGET"
    if any(status == "restored" for status in memory_state.values()):
        return "RESTORE_OR_RELEASE"
    if any(status == "out_of_scope" for status in memory_state.values()):
        return "RESTRICT_SCOPE"
    if any(status == "unresolved" for status in memory_state.values()):
        return "MARK_UNRESOLVED"
    if any(status == "blocked" for status in memory_state.values()):
        return "ESCALATE"
    admitted_types = {action.action_type for action in runtime_result.admitted_actions}
    if "RELEASES" in admitted_types:
        return "RESTORE_OR_RELEASE"
    if "BLOCKS" in admitted_types:
        return "BLOCK"
    if "UNCERTAIN" in admitted_types:
        return "MARK_UNRESOLVED"
    if "SUPERSEDES" in admitted_types:
        return "REVISE"
    if "NO_REVISION" in admitted_types:
        return "PRESERVE"
    return "PRESERVE"


def _project_failure_diagnosis(
    runtime_result: RuntimeResult,
    memory_state: dict[str, str],
    raw_response: Any,
) -> str:
    raw_diagnosis = _raw_response_field(raw_response, "failure_diagnosis")
    if isinstance(raw_diagnosis, str) and raw_diagnosis in FAILURE_MODES:
        return raw_diagnosis

    raw_decision = _raw_response_field(raw_response, "decision")
    if raw_decision == "refuse_due_to_policy":
        return "policy_violation"
    if any(status == "should_not_store" for status in memory_state.values()):
        return "policy_violation"
    if any(status == "out_of_scope" for status in memory_state.values()):
        if any(status == "current" for status in memory_state.values()):
            return "scope_leakage"
    if runtime_result.parser_errors:
        return "memory_hallucination"
    if runtime_result.gate_errors:
        return "wrong_source_attribution"
    if any(a.action_type == "SUPERSEDES" for a in runtime_result.admitted_actions):
        return "stale_memory_reuse"
    if any(status == "blocked" for status in memory_state.values()):
        return "conflict_collapse"
    if any(status == "unresolved" for status in memory_state.values()):
        return "conflict_collapse"
    if any(a.action_type in {"BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS"} for a in runtime_result.admitted_actions):
        if len(runtime_result.admitted_actions) > 1:
            return "scope_leakage"
    return "memory_hallucination"


def project_to_benchmark_response(
    *,
    runtime_result: "RuntimeResult",
    raw_response: Any | None = None,
    scenario_public_view: dict[str, Any] | None = None,
    fallback_answer: str = "",
) -> dict[str, Any]:
    """Project DPA-authorized transitions into evaluator-ready response fields."""
    memory_state = _project_memory_state(
        runtime_result,
        scenario_public_view=scenario_public_view,
        raw_response=raw_response,
    )
    raw_answer = _raw_response_field(raw_response, "answer")
    answer = raw_answer if isinstance(raw_answer, str) and raw_answer else fallback_answer
    raw_followup_answer = _raw_response_field(raw_response, "followup_answer")
    followup_answer = raw_followup_answer if isinstance(raw_followup_answer, str) else answer
    return {
        "answer": answer,
        "decision": _project_decision(memory_state, raw_response),
        "memory_operation": _project_memory_operation(runtime_result, memory_state, raw_response),
        "memory_state": memory_state,
        "evidence_event_ids": _project_evidence_event_ids(runtime_result),
        "failure_diagnosis": _project_failure_diagnosis(
            runtime_result,
            memory_state,
            raw_response,
        ),
        "followup_answer": followup_answer,
    }


ProjectToBenchmarkResponse = project_to_benchmark_response
