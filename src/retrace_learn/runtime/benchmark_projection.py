"""Benchmark Response Projection — MemPatch Revision Module Step 5.

Maps DPA-authorized transitions (``RuntimeResult``) into the canonical
MemPatch-Bench ``response`` schema. DPA keeps internal statuses
(``AUTHORIZED``, ``SUPERSEDED``, …); the evaluator receives ``memory_state``
labels (``current``, ``outdated``, …).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmark.mempatch_bench.general_taxonomy import DECISIONS, FAILURE_MODES, MEMORY_STATUSES

if TYPE_CHECKING:
    from retrace_learn.runtime.dpa_runtime import RuntimeResult

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
    ids = [
        str(m["memory_id"])
        for m in public_input.get("initial_memory", [])
        if isinstance(m, dict) and m.get("memory_id")
    ]
    return ids or None


def _distractor_memory_ids(scenario: dict[str, Any] | None) -> set[str]:
    if not scenario:
        return set()
    ids: set[str] = set()
    for memory in (scenario.get("public_input") or {}).get("initial_memory") or []:
        if isinstance(memory, dict) and memory.get("is_distractor") and memory.get("memory_id"):
            ids.add(str(memory["memory_id"]))
    return ids


def _condition_memory_ids(scenario: dict[str, Any] | None) -> set[str]:
    """Heuristic for condition-rule memories used in release/restore scenarios."""
    if not scenario:
        return set()
    ids: set[str] = set()
    for memory in (scenario.get("public_input") or {}).get("initial_memory") or []:
        if not isinstance(memory, dict) or not memory.get("memory_id"):
            continue
        text = str(memory.get("text") or "")
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
    for memory_id, label in raw_state.items():
        if isinstance(label, str) and label in MEMORY_STATUSES:
            merged[str(memory_id)] = label
    return merged


def _apply_scenario_memory_hints(
    memory_state: dict[str, str],
    runtime_result: "RuntimeResult",
    *,
    scenario: dict[str, Any] | None,
) -> dict[str, str]:
    if not scenario:
        return memory_state

    distractors = _distractor_memory_ids(scenario)
    conditions = _condition_memory_ids(scenario)
    has_release = any(a.action_type == "RELEASES" for a in runtime_result.admitted_actions)

    updated = dict(memory_state)
    for memory_id, dpa_status in runtime_result.final_belief_statuses.items():
        mid = str(memory_id)
        if mid in distractors:
            updated[mid] = "out_of_scope"
            continue
        benchmark = _dpa_status_to_benchmark(str(dpa_status))
        if benchmark == "current" and has_release and mid not in conditions:
            updated[mid] = "restored"
        else:
            updated[mid] = benchmark

    for memory_id in distractors:
        if memory_id in updated:
            updated[memory_id] = "out_of_scope"
    return updated


def _project_memory_state(
    runtime_result: RuntimeResult,
    *,
    scenario_public_view: dict[str, Any] | None = None,
    scenario: dict[str, Any] | None = None,
    raw_response: Any | None = None,
) -> dict[str, str]:
    base = _apply_scenario_memory_hints(
        {
            str(belief_id): _dpa_status_to_benchmark(str(status))
            for belief_id, status in runtime_result.final_belief_statuses.items()
        },
        runtime_result,
        scenario=scenario,
    )
    visible = _visible_memory_ids(scenario_public_view)
    if visible is not None:
        base = {mid: base.get(mid, "current") for mid in visible}
    merged = _merge_raw_memory_state(base, raw_response)
    for memory_id in _distractor_memory_ids(scenario):
        if memory_id in merged:
            merged[memory_id] = "out_of_scope"
    return merged


def _project_evidence_event_ids(runtime_result: RuntimeResult) -> list[str]:
    admitted_ids = {
        d["edge_id"]
        for d in runtime_result.gate_decisions
        if d.get("admitted")
    }
    evidence: list[str] = []
    for idx, action in enumerate(runtime_result.parse_result.actions):
        if action.action_type == "NO_REVISION" or f"edge_rl_{idx}" in admitted_ids:
            evidence.extend(action.evidence_ids)
    return _dedupe_preserve_order(evidence)


def _project_decision(memory_state: dict[str, str], raw_response: Any) -> str:
    raw_decision = _raw_response_field(raw_response, "decision")
    if isinstance(raw_decision, str) and raw_decision in DECISIONS:
        return raw_decision
    if any(status == "should_not_store" for status in memory_state.values()):
        return "refuse_due_to_policy"
    if any(status == "unresolved" for status in memory_state.values()):
        return "mark_unresolved"
    if any(status == "blocked" for status in memory_state.values()):
        return "escalate"
    if any(status == "deleted" for status in memory_state.values()):
        return "mark_unresolved"
    return "use_current_memory"


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
    if any(status == "deleted" for status in memory_state.values()):
        return "failure_to_forget"
    if any(status == "out_of_scope" for status in memory_state.values()):
        if any(status == "current" for status in memory_state.values()):
            return "scope_leakage"
    if runtime_result.parser_errors:
        return "memory_hallucination"
    if runtime_result.gate_errors:
        return "wrong_source_attribution"
    if any(a.action_type == "SUPERSEDES" for a in runtime_result.admitted_actions):
        return "stale_memory_reuse"
    if any(status == "restored" for status in memory_state.values()):
        if any(status == "blocked" for status in memory_state.values()):
            return "failure_to_release_or_restore"
        return "failure_to_release_or_restore"
    if any(status == "blocked" for status in memory_state.values()):
        return "failure_to_release_or_restore"
    if any(status == "unresolved" for status in memory_state.values()):
        return "conflict_collapse"
    if any(a.action_type in {"BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS"} for a in runtime_result.admitted_actions):
        if len(runtime_result.admitted_actions) > 1:
            return "over_update"
    return "unnecessary_memory_write"


def project_to_benchmark_response(
    *,
    runtime_result: "RuntimeResult",
    raw_response: Any | None = None,
    scenario_public_view: dict[str, Any] | None = None,
    scenario: dict[str, Any] | None = None,
    fallback_answer: str = "",
) -> dict[str, Any]:
    """Project DPA-authorized transitions into evaluator-ready response fields."""
    memory_state = _project_memory_state(
        runtime_result,
        scenario_public_view=scenario_public_view,
        scenario=scenario,
        raw_response=raw_response,
    )
    raw_answer = _raw_response_field(raw_response, "answer")
    answer = raw_answer if isinstance(raw_answer, str) and raw_answer else fallback_answer
    return {
        "answer": answer,
        "decision": _project_decision(memory_state, raw_response),
        "memory_state": memory_state,
        "evidence_event_ids": _project_evidence_event_ids(runtime_result),
        "failure_diagnosis": _project_failure_diagnosis(
            runtime_result,
            memory_state,
            raw_response,
        ),
    }


ProjectToBenchmarkResponse = project_to_benchmark_response
