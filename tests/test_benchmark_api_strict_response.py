from __future__ import annotations

import pytest

from benchmark.mempatch_bench.api import evaluate_predictions


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
            "expected_memory_state": {"m1": "outdated"},
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
                    "expected_memory_state": {"m1": "outdated"},
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
