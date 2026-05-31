import csv
import json

import pytest

from retracemem.evaluation.metrics import (
    aggregate_metrics,
    evaluate_predictions,
    write_matrix_outputs,
)


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


def test_write_matrix_outputs(tmp_path):
    out_json = tmp_path / "sub" / "metrics.json"
    aggregated = {
        "oracle_proposer": {"final_status_accuracy": 1.0, "parser_error_rate": 0.0},
        "directjudge_mock": {"final_status_accuracy": 0.9, "parser_error_rate": 0.0},
    }
    rows = [
        {"method": "oracle_proposer", "example_id": "ex_0", "final_status_accuracy": 1.0},
        {"method": "directjudge_mock", "example_id": "ex_0", "final_status_accuracy": 0.9},
    ]

    paths = write_matrix_outputs(str(out_json), aggregated, rows)

    # JSON metrics summary round-trips
    with open(paths["json"], "r", encoding="utf-8") as f:
        assert json.load(f) == aggregated

    # CSV summary has a header + one row per method
    with open(paths["csv"], "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0][0] == "method"
    assert {r[0] for r in reader[1:]} == {"oracle_proposer", "directjudge_mock"}

    # JSONL predictions has one line per prediction row
    with open(paths["predictions"], "r", encoding="utf-8") as f:
        pred_lines = [json.loads(line) for line in f if line.strip()]
    assert len(pred_lines) == 2
    assert pred_lines[0]["method"] == "oracle_proposer"
