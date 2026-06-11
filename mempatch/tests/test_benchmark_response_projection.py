from __future__ import annotations

from benchmark.api import evaluate_predictions
from benchmark.general_taxonomy import (
    DECISIONS,
    FAILURE_MODES,
    PRIMARY_FAILURE_MODES,
    PRIMARY_MEMORY_STATUSES,
)
from benchmark.public_view import public_scenario_view
from mempatch.revision.runtime.benchmark_projection import project_to_benchmark_response
from mempatch.revision.runtime.dpa_runtime import ParseResult, RuntimeResult
from mempatch.revision.runtime.revision_module import run_revision_module_on_scenario
from mempatch.revision.schemas import RevisionAction


def _runtime_result(**kwargs: object) -> RuntimeResult:
    defaults = {
        "final_belief_statuses": {},
        "authorized_belief_ids": (),
        "excluded_belief_ids": (),
        "gate_decisions": [],
        "defeat_paths": [],
        "audit_trace": {},
        "parse_result": ParseResult(valid_json=True, schema_valid=True, actions=()),
    }
    defaults.update(kwargs)
    return RuntimeResult(**defaults)  # type: ignore[arg-type]


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
    runtime_result = _runtime_result(
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
        parse_result=ParseResult(
            valid_json=True,
            schema_valid=True,
            actions=actions,
        ),
    )

    response = project_to_benchmark_response(
        runtime_result=runtime_result,
        scenario_public_view={
            "public_input": {
                "initial_memory": [
                    {"memory_id": "m1"},
                    {"memory_id": "m2"},
                    {"memory_id": "m3"},
                    {"memory_id": "m4"},
                ]
            }
        },
    )

    assert response["memory_state"] == {
        "m1": "current",
        "m2": "current",
        "m3": "blocked",
        "m4": "unresolved",
    }
    assert response["evidence_event_ids"] == ["e2", "e3"]
    assert response["decision"] == "mark_unresolved"
    assert response["failure_diagnosis"] == "stale_memory_reuse"
    assert response["answer"] == ""


def test_project_to_benchmark_response_maps_reserved_raw_diagnosis_to_primary() -> None:
    runtime_result = _runtime_result(
        final_belief_statuses={"m1": "AUTHORIZED"},
        authorized_belief_ids=("m1",),
        parse_result=ParseResult(
            valid_json=True,
            schema_valid=True,
            actions=(
                RevisionAction(action_type="NO_REVISION", evidence_ids=("e1",)),
            ),
        ),
    )

    response = project_to_benchmark_response(
        runtime_result=runtime_result,
        raw_response={
            "response": {
                "decision": "use_current_memory",
                "failure_diagnosis": "unnecessary_memory_write",
                "answer": "Use the current memory.",
            }
        },
        scenario_public_view={
            "public_input": {"initial_memory": [{"memory_id": "m1"}]}
        },
    )

    assert response == {
        "decision": "use_current_memory",
        "memory_state": {"m1": "current"},
        "evidence_event_ids": ["e1"],
        "failure_diagnosis": "memory_hallucination",
        "answer": "Use the current memory.",
    }
    assert response["failure_diagnosis"] in PRIMARY_FAILURE_MODES


def test_project_to_benchmark_response_keeps_release_projection_primary() -> None:
    runtime_result = _runtime_result(
        final_belief_statuses={"m_target": "AUTHORIZED", "m_condition": "AUTHORIZED"},
        authorized_belief_ids=("m_target", "m_condition"),
        gate_decisions=[{"edge_id": "edge_rl_0", "edge_type": "RELEASES", "admitted": True}],
        parse_result=ParseResult(
            valid_json=True,
            schema_valid=True,
            actions=(
                RevisionAction(
                    action_type="RELEASES",
                    target_condition_id="c1",
                    evidence_ids=("e_release",),
                ),
            ),
        ),
    )
    scenario = {
        "scenario_id": "case_restore",
        "public_input": {
            "initial_memory": [
                {"memory_id": "m_target", "text": "Target memory"},
                {"memory_id": "m_condition", "text": "Condition rule: release required"},
                {"memory_id": "m-case-distractor", "text": "Distractor info: other scope"},
            ]
        },
    }

    response = project_to_benchmark_response(
        runtime_result=runtime_result,
        scenario_public_view=public_scenario_view(scenario),
    )

    assert response["memory_state"]["m_target"] == "current"
    assert response["memory_state"]["m_condition"] == "current"
    assert response["memory_state"]["m-case-distractor"] == "out_of_scope"


def test_project_to_benchmark_response_only_accepts_auxiliary_raw_memory_states() -> None:
    runtime_result = _runtime_result(
        final_belief_statuses={"m1": "AUTHORIZED"},
        authorized_belief_ids=("m1",),
        parse_result=ParseResult(
            valid_json=True,
            schema_valid=True,
            actions=(RevisionAction(action_type="NO_REVISION", evidence_ids=("e1",)),),
        ),
    )

    response = project_to_benchmark_response(
        runtime_result=runtime_result,
        raw_response={
            "response": {
                "memory_state": {
                    "m1": "should_not_store",
                    "m2": "blocked",
                    "m3": "deleted",
                },
                "decision": "refuse_due_to_policy",
                "failure_diagnosis": "policy_violation",
            }
        },
        scenario_public_view={
            "public_input": {
                "initial_memory": [
                    {"memory_id": "m1"},
                    {"memory_id": "m2"},
                    {"memory_id": "m3"},
                ]
            }
        },
    )

    assert response["memory_state"]["m1"] == "should_not_store"
    assert response["memory_state"]["m2"] == "current"
    assert response["memory_state"]["m3"] == "current"
    assert response["decision"] == "refuse_due_to_policy"
    assert response["failure_diagnosis"] == "policy_violation"
    for label in response["memory_state"].values():
        assert label in PRIMARY_MEMORY_STATUSES
    assert response["decision"] in DECISIONS
    assert response["failure_diagnosis"] in FAILURE_MODES
    assert response["failure_diagnosis"] in PRIMARY_FAILURE_MODES


def test_revision_module_pipeline_emits_strict_response_fields() -> None:
    scenario = {
        "scenario_id": "case_pipeline_1",
        "public_input": {
            "initial_memory": [{"memory_id": "m1", "text": "Old value"}],
            "event_trace": [
                {"event_id": "e1", "text": "Earlier"},
                {"event_id": "e2", "text": "Latest"},
            ],
        },
        "black_box_task": {"prompt": "What is the current value?"},
    }
    prediction = run_revision_module_on_scenario(scenario)
    response = prediction["response"]
    assert set(response.keys()) == {
        "answer",
        "decision",
        "memory_state",
        "evidence_event_ids",
        "failure_diagnosis",
    }
    assert response["memory_state"] == {"m1": "current"}
    assert response["evidence_event_ids"] == ["e2"]


def test_scorer_treats_empty_memory_gold_as_not_applicable() -> None:
    scenario = {
        "scenario_id": "case_policy_refusal",
        "domain": "software_engineering_agent",
        "primary_failure_mode": "policy_violation",
        "public_input": {
            "initial_memory": [{"memory_id": "m1", "text": "Secret token"}],
            "event_trace": [{"event_id": "e1", "text": "Do not store credentials"}],
        },
        "hidden_gold": {
            "expected_decision": "refuse_due_to_policy",
            "expected_answer": "",
            "expected_memory_state": {},
            "expected_failure_diagnosis": "policy_violation",
            "expected_evidence_event_ids": ["e1"],
            "counterevidence_event_ids": [],
            "rubric": {},
            "decision_aliases": {},
            "stale_or_wrong_answers": [],
        },
    }
    result = evaluate_predictions(
        [scenario],
        [
            {
                "scenario_id": "case_policy_refusal",
                "response": {
                    "answer": "",
                    "decision": "refuse_due_to_policy",
                    "memory_state": {},
                    "evidence_event_ids": ["e1"],
                    "failure_diagnosis": "policy_violation",
                },
            }
        ],
        strict=True,
    )
    assert result["headline_metrics"]["memory_state_accuracy"] == 1.0
