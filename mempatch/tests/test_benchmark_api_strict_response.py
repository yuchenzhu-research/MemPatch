from __future__ import annotations

import pytest

from mempatch.benchmark.api import FAILURE_MODES, MEMORY_STATUSES, evaluate_predictions
from mempatch.benchmark.scorers_general import score_prediction


def _scenario() -> dict:
    return {
        "scenario_id": "case_000001",
        "domain": "software_engineering_agent",
        "primary_failure_mode": "stale_memory_reuse",
        "public_input": {
            "initial_memory": [{"memory_id": "m1", "text": "Old value"}],
            "event_trace": [{"event_id": "e1", "text": "New value"}],
        },
        "hidden_gold": {
            "expected_decision": "use_current_memory",
            "expected_answer": "New value",
            "expected_memory_state": {"m1": "blocked"},
            "expected_failure_diagnosis": "stale_memory_reuse",
            "expected_evidence_event_ids": ["e1"],
            "counterevidence_event_ids": [],
            "rubric": {"must_include": ["New value"]},
            "decision_aliases": {},
            "stale_or_wrong_answers": ["Old value"],
        },
    }


def test_strict_evaluation_requires_full_response_fields() -> None:
    with pytest.raises(ValueError) as excinfo:
        evaluate_predictions(
            [_scenario()],
            [{"scenario_id": "case_000001", "response": {"decision": "use_current_memory"}}],
            strict=True,
        )

    message = str(excinfo.value)
    assert "missing response field 'memory_state'" in message
    assert "missing response field 'evidence_event_ids'" in message
    assert "missing response field 'failure_diagnosis'" in message
    assert "missing response field 'answer'" in message


def test_prediction_expected_fields_do_not_score_as_response_fallback() -> None:
    result = evaluate_predictions(
        [_scenario()],
        [
            {
                "scenario_id": "case_000001",
                "response": {
                    "answer": "Old value",
                    "decision": "use_current_memory",
                    "memory_state": {"m1": "current"},
                    "evidence_event_ids": [],
                    "failure_diagnosis": "under_update",
                    "expected_memory_state": {"m1": "blocked"},
                    "expected_evidence_event_ids": ["e1"],
                    "expected_failure_diagnosis": "stale_memory_reuse",
                },
            }
        ],
        strict=True,
    )

    metrics = result["headline_metrics"]
    assert metrics["memory_state_accuracy"] == 0.0
    assert metrics["evidence_f1"] == 0.0
    assert metrics["failure_diagnosis_accuracy"] == 0.0


def test_invalid_evidence_ids_lower_schema_compliance() -> None:
    result = evaluate_predictions(
        [_scenario()],
        [
            {
                "scenario_id": "case_000001",
                "response": {
                    "answer": "New value",
                    "decision": "use_current_memory",
                    "memory_state": {"m1": "blocked"},
                    "evidence_event_ids": ["e-hallucinated"],
                    "failure_diagnosis": "stale_memory_reuse",
                },
            }
        ],
        strict=False,
    )
    row = result["scored_predictions"][0]
    assert row["metrics"]["response_schema_compliance_rate"] == 0.0
    assert result["headline_metrics"]["response_schema_compliance_rate"] == 0.0
    assert result["headline_metrics"]["joint_revision_success"] == 0.0


def test_empty_response_still_scores_with_schema_violation() -> None:
    result = evaluate_predictions(
        [_scenario()],
        [{"scenario_id": "case_000001", "response": {}}],
        strict=False,
    )
    assert result["count"] == 1
    assert result["headline_metrics"]["response_schema_compliance_rate"] == 0.0


def test_public_api_exposes_final_taxonomy() -> None:
    assert "over_update" in FAILURE_MODES
    assert "failure_to_forget" in FAILURE_MODES
    assert "deleted" in MEMORY_STATUSES
    assert "restored" in MEMORY_STATUSES


def test_strict_evaluation_rejects_unknown_failure_diagnosis() -> None:
    with pytest.raises(ValueError) as excinfo:
        evaluate_predictions(
            [_scenario()],
            [
                {
                    "scenario_id": "case_000001",
                    "response": {
                        "answer": "New value",
                        "decision": "use_current_memory",
                        "memory_state": {"m1": "current"},
                        "evidence_event_ids": ["e1"],
                        "failure_diagnosis": "not_a_failure_mode",
                    },
                }
            ],
            strict=True,
        )

    message = str(excinfo.value)
    assert "invalid failure_diagnosis label 'not_a_failure_mode'" in message


def test_all_final_failure_metrics_are_reported() -> None:
    metrics = score_prediction(
        _scenario(),
        {
            "response": {
                "answer": "New value",
                "decision": "use_current_memory",
                "memory_state": {"m1": "current"},
                "evidence_event_ids": ["e1"],
                "failure_diagnosis": "stale_memory_reuse",
            }
        },
    )

    assert "over_update_rate" in metrics
    assert "unnecessary_memory_write_rate" in metrics
    assert "failure_to_forget_rate" in metrics
    assert "failure_to_release_or_restore_rate" in metrics
