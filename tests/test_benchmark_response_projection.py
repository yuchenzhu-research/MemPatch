from __future__ import annotations

from retrace_learn.runtime.dpa_runtime import (
    ParseResult,
    RuntimeResult,
    project_to_benchmark_response,
)
from retrace_learn.schemas import RevisionAction


def test_project_to_benchmark_response_maps_dpa_statuses_and_evidence() -> None:
    actions = (
        RevisionAction(
            action_type="SUPERSEDES",
            target_belief_id="m1",
            replacement_belief_id="m1_new",
            evidence_ids=("e2",),
        ),
        RevisionAction(
            action_type="NO_REVISION",
            evidence_ids=("e3",),
        ),
    )
    runtime_result = RuntimeResult(
        final_belief_statuses={
            "m1": "SUPERSEDED",
            "m2": "AUTHORIZED",
            "m3": "BLOCKED",
            "m4": "UNRESOLVED",
        },
        authorized_belief_ids=("m2",),
        excluded_belief_ids=("m1", "m3", "m4"),
        gate_decisions=[
            {"edge_id": "edge_rl_0", "edge_type": "SUPERSEDES", "admitted": True}
        ],
        defeat_paths=[],
        audit_trace={},
        parse_result=ParseResult(
            valid_json=True,
            schema_valid=True,
            actions=actions,
        ),
    )

    response = project_to_benchmark_response(runtime_result)

    assert response["memory_state"] == {
        "m1": "outdated",
        "m2": "current",
        "m3": "blocked",
        "m4": "unresolved",
    }
    assert response["evidence_event_ids"] == ["e2", "e3"]
    assert response["decision"] == "mark_unresolved"
    assert response["failure_diagnosis"] == "stale_memory_reuse"
    assert response["answer"] == ""


def test_project_to_benchmark_response_preserves_valid_raw_fields() -> None:
    runtime_result = RuntimeResult(
        final_belief_statuses={"m1": "AUTHORIZED"},
        authorized_belief_ids=("m1",),
        excluded_belief_ids=(),
        gate_decisions=[],
        defeat_paths=[],
        audit_trace={},
        parse_result=ParseResult(
            valid_json=True,
            schema_valid=True,
            actions=(
                RevisionAction(action_type="NO_REVISION", evidence_ids=("e1",)),
            ),
        ),
    )

    response = project_to_benchmark_response(
        runtime_result,
        {
            "response": {
                "decision": "use_current_memory",
                "failure_diagnosis": "unnecessary_memory_write",
                "answer": "Use the current memory.",
            }
        },
    )

    assert response == {
        "decision": "use_current_memory",
        "memory_state": {"m1": "current"},
        "evidence_event_ids": ["e1"],
        "failure_diagnosis": "unnecessary_memory_write",
        "answer": "Use the current memory.",
    }
