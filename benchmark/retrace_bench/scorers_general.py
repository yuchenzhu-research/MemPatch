"""Scoring utilities for the general English ReTrace-Bench release."""

from __future__ import annotations

from collections import Counter
from typing import Any

from benchmark.retrace_bench.general_taxonomy import FAILURE_MODES


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _f1(predicted: list[str], expected: list[str]) -> float:
    p = set(predicted or [])
    e = set(expected or [])
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    tp = len(p & e)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(e)
    return 2 * precision * recall / (precision + recall)


def score_prediction(scenario: dict[str, Any], prediction: dict[str, Any]) -> dict[str, float]:
    gold = scenario["hidden_gold"]
    response = prediction.get("response", prediction)
    expected_state = gold.get("expected_memory_state", {})
    predicted_state = response.get("memory_state", response.get("expected_memory_state", {})) or {}
    state_total = len(expected_state) or 1
    state_correct = sum(1 for mid, status in expected_state.items() if predicted_state.get(mid) == status)
    expected_decision = gold.get("expected_decision")
    predicted_decision = response.get("decision", response.get("expected_decision"))
    expected_diag = gold.get("expected_failure_diagnosis")
    predicted_diag = response.get("failure_diagnosis", response.get("expected_failure_diagnosis"))

    metrics = {
        "answer_accuracy": float(_norm(response.get("answer")) == _norm(gold.get("expected_answer"))),
        "decision_accuracy": float((expected_decision or "") == (predicted_decision or "")),
        "memory_state_accuracy": state_correct / state_total,
        "evidence_f1": _f1(response.get("evidence_event_ids", []), gold.get("expected_evidence_event_ids", [])),
        "failure_diagnosis_accuracy": float(expected_diag == predicted_diag),
    }
    for mode in FAILURE_MODES:
        metrics[f"{mode}_rate"] = float(scenario.get("primary_failure_mode") == mode and predicted_diag == mode)
    metrics["stale_reuse_rate"] = float(_norm(response.get("answer")) in {_norm(v) for v in gold.get("stale_or_wrong_answers", [])})
    metrics["under_update_rate"] = float(predicted_diag == "under_update")
    metrics["over_update_rate"] = float(predicted_diag == "over_update")
    metrics["scope_leakage_rate"] = float(predicted_diag == "scope_leakage")
    metrics["policy_violation_rate"] = float(predicted_diag == "policy_violation")
    return metrics


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    count = 0
    for row in rows:
        for key, value in row.get("metrics", {}).items():
            totals[key] += float(value)
        count += 1
    return {
        "count": count,
        "metrics": {key: value / count for key, value in sorted(totals.items())} if count else {},
    }

