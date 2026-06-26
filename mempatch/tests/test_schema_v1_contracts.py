from __future__ import annotations

import pytest

from mempatch.benchmark.release import label_row, public_row
from mempatch.benchmark.schema import (
    Prediction,
    PrivateLabel,
    PublicScenario,
    ScoreRecord,
    audit_public_private_pair,
    validate_public_scenario,
)
from mempatch.tests.test_benchmark_v14_kernel import _raw


def test_public_row_rejects_forbidden_fields() -> None:
    row = public_row(_raw())
    row["expected_decision"] = "use_current_memory"
    errors = validate_public_scenario(row)
    assert errors
    assert any("forbidden" in error for error in errors)


def test_private_label_accepts_expected_fields() -> None:
    label = PrivateLabel.from_dict(label_row(_raw(), "main_test_synthetic"))
    assert label.scenario_id == "case-1"
    assert label.expected_memory_operation == "RESTRICT_SCOPE"
    assert label.failure_mode == "scope_leakage"


def test_expected_memory_states_list_roundtrip_and_dynamic_dict_rejected() -> None:
    label = label_row(_raw(), "main_test_synthetic")
    parsed = PrivateLabel.from_dict(label)
    assert parsed.to_dict()["expected_memory_states"] == [{"memory_id": "mem-1", "status": "out_of_scope"}]
    bad = dict(label)
    bad["expected_memory_states"] = {"mem-1": "out_of_scope"}
    with pytest.raises(ValueError, match="not a dynamic dict"):
        PrivateLabel.from_dict(bad)


def test_prediction_roundtrip() -> None:
    row = {
        "scenario_id": "case-1",
        "model": "tiny",
        "method": "direct_json",
        "split": "main_test_synthetic",
        "raw_response": "{}",
        "parsed": {
            "answer": "a",
            "decision": "use_current_memory",
            "memory_operation": "PRESERVE",
            "memory_state": [{"memory_id": "mem-1", "status": "current"}],
            "evidence_event_ids": ["ev-1"],
            "failure_diagnosis": "stale_memory_reuse",
            "followup_answer": "a",
        },
        "input_tokens": 10,
        "output_tokens": 2,
        "latency_sec": 0.5,
        "retrieved_event_count": 1,
    }
    assert Prediction.from_dict(row).to_dict() == row


def test_score_record_roundtrip() -> None:
    row = {
        "scenario_id": "case-1",
        "split": "main_test_synthetic",
        "model": "tiny",
        "method": "direct_json",
        "schema_valid": True,
        "parse_failed": False,
        "exact_state_map": True,
        "contract_valid_state_success": True,
        "decision_correct": True,
        "decision_macro_f1_class": "use_current_memory",
        "memory_state_accuracy": 1.0,
        "evidence_precision": 1.0,
        "evidence_recall": 1.0,
        "evidence_f1": 1.0,
        "diagnosis_correct": True,
        "strict_joint": True,
        "unsafe_reuse": False,
        "downstream_contamination": False,
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "latency_sec": 0.5,
    }
    assert ScoreRecord.from_dict(row).to_dict() == row


def test_public_private_id_matching_and_missing_evidence_audit() -> None:
    public = public_row(_raw())
    label = label_row(_raw(), "main_test_synthetic")
    assert audit_public_private_pair(public, label) == []
    bad = dict(label)
    bad["scenario_id"] = "other"
    bad["expected_evidence_event_ids"] = ["missing-event"]
    errors = audit_public_private_pair(public, bad)
    assert any("scenario_id mismatch" in error for error in errors)
    assert any("missing from public events" in error for error in errors)


def test_public_scenario_contract_roundtrip() -> None:
    row = public_row(_raw())
    assert PublicScenario.from_dict(row).to_dict() == row
