from __future__ import annotations

import json

import pytest

from benchmark.retrace_bench.api import evaluate_predictions, load_predictions, load_scenarios
from benchmark.retrace_bench.public_view import public_scenario_view


def _scenario() -> dict:
    return {
        "scenario_id": "s1",
        "domain": "software_engineering_agent",
        "difficulty": "L2",
        "workflow_context": "Release check",
        "public_input": {
            "event_trace": [
                {"event_id": "e1", "text": "Issue opened", "trust_level": "trusted"},
                {"event_id": "e2", "text": "Fix merged", "trust_level": "verified"},
            ],
            "initial_memory": [{"memory_id": "m1", "text": "old state", "is_distractor": True}],
        },
        "black_box_task": {"prompt": "answer"},
        "hidden_gold": {
            "expected_decision": "use_current_memory",
            "expected_answer": "Fix merged",
            "memory_states": {"m1": "current"},
            "minimal_evidence_event_ids": ["e2"],
            "failure_diagnosis": "stale_memory_reuse",
            "answer_must_include": ["Fix merged"],
        },
    }


def test_evaluate_predictions_scores_valid_prediction() -> None:
    scenario = _scenario()
    prediction = {
        "scenario_id": "s1",
        "response": {
            "decision": "use_current_memory",
            "answer": "Fix merged",
            "memory_state": {"m1": "current"},
            "evidence_event_ids": ["e2"],
            "failure_diagnosis": "stale_memory_reuse",
        },
    }

    result = evaluate_predictions([scenario], [prediction], strict=True)

    assert result["count"] == 1
    assert result["headline_metrics"]["decision_macro_f1"] == 1.0
    assert result["headline_metrics"]["memory_state_accuracy"] == 1.0
    assert result["errors"] == []


def test_evaluate_predictions_rejects_unknown_evidence_id() -> None:
    prediction = {
        "scenario_id": "s1",
        "response": {
            "decision": "use_current_memory",
            "memory_state": {"m1": "current"},
            "evidence_event_ids": ["missing"],
            "failure_diagnosis": "stale_memory_reuse",
            "answer": "Fix merged",
        },
    }

    with pytest.raises(ValueError, match="evidence_event_ids"):
        evaluate_predictions([_scenario()], [prediction], strict=True)


def test_load_jsonl_helpers(tmp_path) -> None:
    data_dir = tmp_path / "split"
    data_dir.mkdir()
    scenarios_path = data_dir / "scenarios.jsonl"
    preds_path = tmp_path / "predictions.jsonl"
    scenarios_path.write_text(json.dumps(_scenario()) + "\n", encoding="utf-8")
    preds_path.write_text(json.dumps({"scenario_id": "s1", "decision": "use_current_memory"}) + "\n", encoding="utf-8")

    assert load_scenarios(data_dir)[0]["scenario_id"] == "s1"
    assert load_predictions(preds_path)[0]["scenario_id"] == "s1"


def test_public_view_strips_internal_fields() -> None:
    view = public_scenario_view(_scenario())

    assert "hidden_gold" not in view
    assert "is_distractor" not in view["public_input"]["initial_memory"][0]


def test_structure_only_prediction_success() -> None:
    scenario = _scenario()
    # structure-only prediction: decision, memory_state, evidence_event_ids, failure_diagnosis correct.
    # answer is absent/empty.
    prediction = {
        "scenario_id": "s1",
        "response": {
            "decision": "use_current_memory",
            "memory_state": {"m1": "current"},
            "evidence_event_ids": ["e2"],
            "failure_diagnosis": "stale_memory_reuse",
        },
    }

    result = evaluate_predictions([scenario], [prediction], strict=True)
    
    assert result["count"] == 1
    # Assert joint_revision_success is 0 because answer is absent (key_fact_matches / answer_state_consistency is 0)
    assert result["headline_metrics"]["joint_revision_success"] == 0.0
    # Assert structural_revision_success is 1.0 (fully correct structure-only prediction)
    assert result["headline_metrics"]["structural_revision_success"] == 1.0


def test_prediction_normalization_and_validation_hardening() -> None:
    scenario = _scenario()
    # decision and memory_state values are one-element lists.
    prediction = {
        "scenario_id": "s1",
        "response": {
            "decision": ["use_current_memory"],
            "memory_state": {"m1": ["current"]},
            "evidence_event_ids": ["e2"],
            "failure_diagnosis": ["stale_memory_reuse"],
        },
    }

    result = evaluate_predictions([scenario], [prediction], strict=True)
    assert result["headline_metrics"]["structural_revision_success"] == 1.0

    # Test validator handles unhashable/invalid types gracefully under strict=False
    invalid_pred = {
        "scenario_id": "s1",
        "response": {
            "decision": {"invalid": "dict"},
            "memory_state": {"m1": ["current", "invalid_length_two"]},
            "evidence_event_ids": [["unhashable_list"]],
            "failure_diagnosis": {"invalid": "dict"},
        },
    }
    result_invalid = evaluate_predictions([scenario], [invalid_pred], strict=False)
    assert len(result_invalid["errors"]) > 0
    # verify that evaluation did not crash and returned scores (likely 0 for structural success)
    assert result_invalid["headline_metrics"]["structural_revision_success"] == 0.0

    # Verify that strict=True raises ValueError (due to validation errors) rather than TypeError
    with pytest.raises(ValueError) as excinfo:
        evaluate_predictions([scenario], [invalid_pred], strict=True)
    assert "TypeError" not in str(excinfo.value)
    assert "invalid decision" in str(excinfo.value)

