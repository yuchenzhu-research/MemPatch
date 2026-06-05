from __future__ import annotations

import json

import pytest

from benchmark.retrace_bench.api import evaluate_predictions, load_predictions, load_scenarios
from benchmark.retrace_bench.public_view import public_scenario_view
from benchmark.retrace_bench.scorers_general import normalize_failure_mode


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
            "expected_memory_state": {"m1": "current"},
            "expected_evidence_event_ids": ["e2"],
            "expected_failure_diagnosis": "stale_memory_reuse",
            "counterevidence_event_ids": [],
            "rubric": {"must_include": ["Fix merged"]},
            "stale_or_wrong_answers": [],
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
    # Structure-only prediction: decision, memory_state, evidence_event_ids,
    # failure_diagnosis correct; answer is absent.
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
    assert result["headline_metrics"]["joint_revision_success"] == 0.0
    assert result["headline_metrics"]["structural_revision_success"] == 1.0


def test_one_element_list_prediction_normalization() -> None:
    scenario = _scenario()
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


def test_unhashable_invalid_values_return_validation_errors() -> None:
    scenario = _scenario()
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
    assert result_invalid["headline_metrics"]["structural_revision_success"] == 0.0

    with pytest.raises(ValueError) as excinfo:
        evaluate_predictions([scenario], [invalid_pred], strict=True)
    assert "TypeError" not in str(excinfo.value)
    assert "invalid decision" in str(excinfo.value)


def test_normalize_failure_mode_handles_unhashable_values() -> None:
    assert normalize_failure_mode(["stale_memory_reuse"]) == "stale_memory_reuse"
    assert normalize_failure_mode({"invalid": "dict"})
    assert normalize_failure_mode([{"invalid": "dict"}])
