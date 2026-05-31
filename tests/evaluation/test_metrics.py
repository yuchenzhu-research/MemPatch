import pytest

from retracemem.evaluation.metrics import evaluate_predictions, aggregate_metrics


def test_evaluate_predictions_perfect():
    predictions = [
        {
            "parser_result": {"valid_json": True},
            "sampled_actions": [
                {"action_type": "SUPERSEDES", "target_belief_id": "b1", "replacement_belief_id": "b2", "evidence_ids": ["ev1"]}
            ],
            "gate_decisions": [{"admitted": True}],
            "defeat_paths": [{"belief_id": "b1", "path_type": "DIRECT_SUPERSEDE"}],
            "dpa_final_statuses": {"b1": "SUPERSEDED", "b2": "AUTHORIZED"},
        }
    ]

    gold_final_statuses = {"b1": "SUPERSEDED", "b2": "AUTHORIZED"}
    gold_actions = [
        {"action_type": "SUPERSEDES", "target_belief_id": "b1", "replacement_belief_id": "b2", "evidence_ids": ["ev1"]}
    ]

    res = evaluate_predictions(
        predictions=predictions,
        gold_final_statuses=gold_final_statuses,
        gold_actions=gold_actions,
        valid_belief_ids={"b1", "b2"},
        valid_condition_ids=set(),
        valid_evidence_ids={"ev1"},
    )

    assert res["final_status_accuracy"] == 1.0
    assert res["action_type_accuracy"] == 1.0
    assert res["target_grounding"] == 1.0
    assert res["evidence_grounding"] == 1.0
    assert res["gate_rejection_rate"] == 0.0
    assert res["parser_error_rate"] == 0.0
    assert res["stale_propagation_rate"] == 0.0
    assert res["audit_completeness"] == 1.0


def test_evaluate_predictions_failures():
    predictions = [
        {
            "parser_result": {"valid_json": False},
            "sampled_actions": [],
            "gate_decisions": [],
            "defeat_paths": [],
            "dpa_final_statuses": {"b1": "AUTHORIZED"},
        }
    ]
    # Gold status of b1 is SUPERSEDED, but pred is AUTHORIZED (under_update / stale_propagation)
    gold_final_statuses = {"b1": "SUPERSEDED"}

    res = evaluate_predictions(
        predictions=predictions,
        gold_final_statuses=gold_final_statuses,
        gold_actions=[],
        valid_belief_ids={"b1"},
    )

    assert res["parser_error_rate"] == 1.0
    assert res["under_update_rate"] == 1.0
    assert res["stale_propagation_rate"] == 1.0
    assert res["final_status_accuracy"] == 0.0


def test_aggregate_metrics():
    ep1 = {"final_status_accuracy": 1.0, "parser_error_rate": 0.0}
    ep2 = {"final_status_accuracy": 0.5, "parser_error_rate": 1.0}

    agg = aggregate_metrics([ep1, ep2])
    assert agg["final_status_accuracy"] == 0.75
    assert agg["parser_error_rate"] == 0.5
