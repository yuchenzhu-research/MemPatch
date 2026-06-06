"""Benchmark Response Projection — MemPatch Revision Module Step 5.

Maps DPA-authorized transitions (``RuntimeResult``) into the canonical
MemPatch-Bench ``response`` schema. DPA keeps internal statuses
(``AUTHORIZED``, ``SUPERSEDED``, …); the evaluator receives ``memory_state``
labels (``current``, ``outdated``, …).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmark.retrace_bench.general_taxonomy import DECISIONS, FAILURE_MODES

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


def _project_memory_state(
    runtime_result: RuntimeResult,
    *,
    scenario_public_view: dict[str, Any] | None = None,
) -> dict[str, str]:
    memory_state: dict[str, str] = {}
    for belief_id, status in runtime_result.final_belief_statuses.items():
        memory_state[str(belief_id)] = BENCHMARK_STATUS_BY_DPA_STATUS.get(
            str(status),
            "unresolved",
        )
    visible = _visible_memory_ids(scenario_public_view)
    if visible is not None:
        filtered = {mid: memory_state.get(mid, "current") for mid in visible}
        return filtered
    return memory_state


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
    if any(status == "unresolved" for status in memory_state.values()):
        return "mark_unresolved"
    if any(status == "blocked" for status in memory_state.values()):
        return "escalate"
    return "use_current_memory"


def _project_failure_diagnosis(
    runtime_result: RuntimeResult,
    memory_state: dict[str, str],
    raw_response: Any,
) -> str:
    raw_diagnosis = _raw_response_field(raw_response, "failure_diagnosis")
    if isinstance(raw_diagnosis, str) and raw_diagnosis in FAILURE_MODES:
        return raw_diagnosis
    if runtime_result.parser_errors:
        return "memory_hallucination"
    if runtime_result.gate_errors:
        return "wrong_source_attribution"
    if any(a.action_type == "SUPERSEDES" for a in runtime_result.admitted_actions):
        return "stale_memory_reuse"
    if any(status == "blocked" for status in memory_state.values()):
        return "failure_to_release_or_restore"
    if any(status == "unresolved" for status in memory_state.values()):
        return "conflict_collapse"
    return "unnecessary_memory_write"


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
