from __future__ import annotations

import json

import pytest

from benchmark.retrace_bench.api import evaluate_predictions, load_predictions, load_scenarios
from benchmark.retrace_bench.public_view import public_scenario_view
from benchmark.retrace_bench.utils.contamination import check_contamination


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


def test_contamination_guard_rejects_benchmark_training_path() -> None:
    with pytest.raises(RuntimeError):
        check_contamination("data/retrace_bench/main/scenarios.jsonl")

    check_contamination("data/retrace_learn/v1/boundary_audit/minimal.jsonl")
