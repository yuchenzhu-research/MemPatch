"""Focused tests for the public ReTrace-Bench scoring API."""

from __future__ import annotations

import pytest

from benchmark.retrace_bench.api import (
    HEADLINE_METRICS,
    evaluate_predictions,
    normalize_prediction,
)


def _scenario(sid: str, decision: str = "use_current_memory") -> dict:
    return {
        "scenario_id": sid,
        "domain": "software_engineering_agent",
        "primary_failure_mode": "stale_memory_reuse",
        "public_input": {"event_trace": [{"event_id": "e1"}, {"event_id": "e2"}]},
        "hidden_gold": {
            "expected_answer": "follow the verified path",
            "expected_decision": decision,
            "expected_evidence_event_ids": ["e1"],
            "expected_memory_state": {"m1": "current"},
            "expected_failure_diagnosis": "stale_memory_reuse",
        },
    }


def _good_response(decision: str = "use_current_memory") -> dict:
    return {
        "answer": "follow the verified path",
        "decision": decision,
        "memory_state": {"m1": "current"},
        "evidence_event_ids": ["e1"],
        "failure_diagnosis": "stale_memory_reuse",
    }


def test_evaluates_tiny_prediction_set():
    scenarios = [_scenario("s1"), _scenario("s2")]
    predictions = [
        {"scenario_id": "s1", "response": _good_response()},
        {"scenario_id": "s2", **_good_response()},  # flat form
    ]
    result = evaluate_predictions(scenarios, predictions, strict=True)
    assert result["count"] == 2
    assert result["errors"] == []
    assert result["all_metrics"]["memory_state_accuracy"] == 1.0
    assert result["all_metrics"]["evidence_f1"] == 1.0


def test_headline_metrics_present_in_output():
    scenarios = [_scenario("s1")]
    predictions = [{"scenario_id": "s1", "response": _good_response()}]
    result = evaluate_predictions(scenarios, predictions, strict=True)
    assert set(HEADLINE_METRICS).issubset(result["headline_metrics"].keys())


def test_missing_prediction_strict_raises_else_warns():
    scenarios = [_scenario("s1"), _scenario("s2")]
    predictions = [{"scenario_id": "s1", "response": _good_response()}]
    with pytest.raises(ValueError):
        evaluate_predictions(scenarios, predictions, strict=True)
    result = evaluate_predictions(scenarios, predictions, strict=False)
    assert result["count"] == 1
    assert any("missing prediction" in e for e in result["errors"])


def test_duplicate_scenario_id_behavior():
    scenarios = [_scenario("s1")]
    predictions = [
        {"scenario_id": "s1", "response": _good_response()},
        {"scenario_id": "s1", "response": _good_response()},
    ]
    with pytest.raises(ValueError):
        evaluate_predictions(scenarios, predictions, strict=True)
    result = evaluate_predictions(scenarios, predictions, strict=False)
    assert result["count"] == 1
    assert any("duplicate prediction" in e for e in result["errors"])


def test_extra_prediction_behavior():
    scenarios = [_scenario("s1")]
    predictions = [
        {"scenario_id": "s1", "response": _good_response()},
        {"scenario_id": "ghost", "response": _good_response()},
    ]
    with pytest.raises(ValueError):
        evaluate_predictions(scenarios, predictions, strict=True)
    result = evaluate_predictions(scenarios, predictions, strict=False)
    assert any("no matching scenario" in e for e in result["errors"])


def test_invalid_decision_label_raises_in_strict_mode():
    scenarios = [_scenario("s1")]
    bad = _good_response()
    bad["decision"] = "not_a_real_label"
    predictions = [{"scenario_id": "s1", "response": bad}]
    with pytest.raises(ValueError):
        evaluate_predictions(scenarios, predictions, strict=True)
    result = evaluate_predictions(scenarios, predictions, strict=False)
    assert any("invalid decision label" in e for e in result["errors"])
    assert result["count"] == 1  # still scored in non-strict mode


def test_invalid_memory_state_and_evidence_raise_in_strict_mode():
    scenarios = [_scenario("s1")]
    bad_state = _good_response()
    bad_state["memory_state"] = {"m1": "bogus_status"}
    with pytest.raises(ValueError):
        evaluate_predictions(scenarios, [{"scenario_id": "s1", "response": bad_state}], strict=True)

    bad_ev = _good_response()
    bad_ev["evidence_event_ids"] = ["e_does_not_exist"]
    with pytest.raises(ValueError):
        evaluate_predictions(scenarios, [{"scenario_id": "s1", "response": bad_ev}], strict=True)


def test_normalize_prediction_canonical_and_flat():
    canonical = normalize_prediction({"scenario_id": "s1", "response": _good_response()})
    flat = normalize_prediction({"scenario_id": "s1", **_good_response()})
    assert canonical["scenario_id"] == "s1"
    assert canonical["response"]["decision"] == "use_current_memory"
    assert flat["response"]["decision"] == "use_current_memory"
    assert flat["response"]["evidence_event_ids"] == ["e1"]
