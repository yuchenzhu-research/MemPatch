from __future__ import annotations

from typing import Any
from benchmark.retrace_bench.scorers_general import aggregate_metrics


def test_majority_class_baseline_balanced_metrics():
    # 70% use_current_memory, 30% escalate
    # Predict use_current_memory for everything
    rows = []
    # 7 rows expected use_current_memory, predicted use_current_memory
    for i in range(7):
        rows.append({
            "expected_decision": "use_current_memory",
            "domain": "customer_support_crm",
            "primary_failure_mode": "stale_memory_reuse",
            "response": {"decision": "use_current_memory"},
            "metrics": {
                "black_box_decision_accuracy": 1.0,
                "memory_state_accuracy": 1.0,
                "evidence_f1": 1.0,
                "failure_diagnosis_accuracy": 1.0,
                "stale_reuse_rate": 0.0,
            }
        })
    # 3 rows expected escalate, predicted use_current_memory
    for i in range(3):
        rows.append({
            "expected_decision": "escalate",
            "domain": "customer_support_crm",
            "primary_failure_mode": "scope_leakage",
            "response": {"decision": "use_current_memory"},
            "metrics": {
                "black_box_decision_accuracy": 0.0,
                "memory_state_accuracy": 1.0,
                "evidence_f1": 1.0,
                "failure_diagnosis_accuracy": 1.0,
                "stale_reuse_rate": 0.0,
            }
        })

    result = aggregate_metrics(rows)
    metrics = result["metrics"]

    # Raw accuracy should be 0.70
    assert metrics["black_box_decision_accuracy"] == 0.70
    # Balanced accuracy: (recall("use_current_memory") + recall("escalate")) / 2
    # recall("use_current_memory") = 7/7 = 1.0
    # recall("escalate") = 0/3 = 0.0
    # Balanced accuracy = 0.50
    assert metrics["decision_balanced_accuracy"] == 0.50
    # Macro F1:
    # class use_current_memory: precision = 7/10 = 0.7, recall = 1.0. F1 = 2 * 0.7 * 1 / 1.7 = 1.4 / 1.7 ≈ 0.8235
    # class escalate: precision = 0.0, recall = 0.0. F1 = 0.0
    # Macro F1 = (0.8235 + 0) / 2 ≈ 0.41176
    assert abs(metrics["decision_macro_f1"] - 7/17) < 1e-4


def test_non_answer_decision_accuracy():
    # Mix of use_current_memory, escalate, and ask_clarification
    rows = [
        # expected use_current_memory, correct
        {
            "expected_decision": "use_current_memory",
            "response": {"decision": "use_current_memory"},
            "metrics": {"black_box_decision_accuracy": 1.0}
        },
        # expected escalate, correct
        {
            "expected_decision": "escalate",
            "response": {"decision": "escalate"},
            "metrics": {"black_box_decision_accuracy": 1.0}
        },
        # expected ask_clarification, incorrect
        {
            "expected_decision": "ask_clarification",
            "response": {"decision": "use_current_memory"},
            "metrics": {"black_box_decision_accuracy": 0.0}
        }
    ]
    result = aggregate_metrics(rows)
    metrics = result["metrics"]

    # Total non-answer expected count: 2 (escalate, ask_clarification)
    # Correct non-answer: 1 (escalate)
    # non_answer_decision_accuracy: 1/2 = 0.50
    assert metrics["non_answer_decision_accuracy"] == 0.50
    # use_current_memory_accuracy: 1/1 = 1.0
    assert metrics["use_current_memory_accuracy"] == 1.0


def test_per_failure_mode_and_domain_breakdowns():
    rows = [
        {
            "expected_decision": "use_current_memory",
            "domain": "dom_a",
            "primary_failure_mode": "mode_x",
            "response": {"decision": "use_current_memory"},
            "metrics": {
                "black_box_decision_accuracy": 1.0,
                "memory_state_accuracy": 0.8,
                "evidence_f1": 0.9,
                "failure_diagnosis_accuracy": 1.0,
                "stale_reuse_rate": 0.0,
            }
        },
        {
            "expected_decision": "use_current_memory",
            "domain": "dom_b",
            "primary_failure_mode": "mode_y",
            "response": {"decision": "use_current_memory"},
            "metrics": {
                "black_box_decision_accuracy": 0.0,
                "memory_state_accuracy": 0.6,
                "evidence_f1": 0.4,
                "failure_diagnosis_accuracy": 0.0,
                "stale_reuse_rate": 1.0,
            }
        }
    ]
    result = aggregate_metrics(rows)

    # Check failure modes
    assert "mode_x" in result["per_failure_mode"]
    assert "mode_y" in result["per_failure_mode"]
    assert result["per_failure_mode"]["mode_x"]["count"] == 1
    assert result["per_failure_mode"]["mode_x"]["black_box_decision_accuracy"] == 1.0
    assert result["per_failure_mode"]["mode_y"]["stale_reuse_rate"] == 1.0

    # Check domains
    assert "dom_a" in result["per_domain"]
    assert "dom_b" in result["per_domain"]
    assert result["per_domain"]["dom_a"]["count"] == 1
    assert result["per_domain"]["dom_a"]["memory_state_accuracy"] == 0.8
    assert "stale_reuse_rate" not in result["per_domain"]["dom_a"]  # Not requested in per-domain
