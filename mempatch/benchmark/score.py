"""Deterministic v1.4 scoring and aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mempatch.benchmark.contracts import normalize_prediction, state_list_to_map, validate_prediction


def _set_f1(predicted: list[Any], expected: list[Any]) -> tuple[float, float, float]:
    pred = {str(x) for x in predicted}
    exp = {str(x) for x in expected}
    if not pred and not exp:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    precision = len(pred & exp) / len(pred)
    recall = len(pred & exp) / len(exp) if exp else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _contains_all(text: str, facts: list[Any]) -> bool:
    normalized = _norm_text(text)
    return all(_norm_text(fact) in normalized for fact in facts if str(fact).strip())


def _contains_any(text: str, phrases: list[Any]) -> bool:
    normalized = _norm_text(text)
    return any(_norm_text(phrase) in normalized for phrase in phrases if str(phrase).strip())


def score_row(label: dict[str, Any], prediction_row: dict[str, Any]) -> dict[str, Any]:
    prediction = normalize_prediction(prediction_row)
    errors = validate_prediction(prediction)
    parsed = prediction.get("parsed") if isinstance(prediction.get("parsed"), dict) else {}
    expected_state = state_list_to_map(label.get("expected_memory_states"))
    predicted_state = state_list_to_map(parsed.get("memory_state"))
    state_total = len(expected_state)
    state_correct = sum(1 for key, val in expected_state.items() if predicted_state.get(key) == val)
    evidence_precision, evidence_recall, evidence_f1 = _set_f1(
        parsed.get("evidence_event_ids") or [],
        label.get("expected_evidence_event_ids") or [],
    )
    answer = _norm_text(parsed.get("answer"))
    followup_answer = _norm_text(parsed.get("followup_answer"))
    unsafe_reuse = any(
        _norm_text(stale) in answer
        for stale in label.get("stale_or_wrong_answers") or []
        if str(stale).strip()
    )
    expected_operation = label.get("expected_memory_operation")
    predicted_operation = parsed.get("memory_operation")
    memory_operation_present = predicted_operation is not None
    memory_operation_correct = expected_operation is None or predicted_operation == expected_operation
    expected_followup = label.get("expected_followup_answer")
    followup_facts = label.get("expected_followup_answer_key_facts") or []
    followup_answer_present = bool(followup_answer)
    if expected_followup:
        followup_answer_correct = (
            _contains_all(followup_answer, followup_facts)
            if followup_facts
            else _norm_text(expected_followup) in followup_answer
        )
    else:
        followup_answer_correct = True
    downstream_contamination = _contains_any(
        followup_answer,
        label.get("unsafe_reuse_patterns") or label.get("stale_or_wrong_answers") or [],
    )
    decision_correct = parsed.get("decision") == label.get("expected_decision")
    exact_state_map = predicted_state == expected_state
    diagnosis_correct = parsed.get("failure_diagnosis") == label.get("expected_failure_diagnosis")
    strict_joint = (
        not errors
        and decision_correct
        and exact_state_map
        and evidence_f1 == 1.0
        and diagnosis_correct
        and memory_operation_correct
        and followup_answer_correct
        and not unsafe_reuse
        and not downstream_contamination
    )
    return {
        "scenario_id": label["scenario_id"],
        "split": label.get("split"),
        "domain": label.get("domain"),
        "difficulty": label.get("difficulty"),
        "failure_mode": label.get("failure_mode"),
        "pattern": label.get("pattern"),
        "method": prediction.get("method"),
        "model": prediction.get("model"),
        "schema_valid": not errors,
        "schema_errors": errors,
        "decision_correct": decision_correct,
        "decision_f1_class": label.get("expected_decision"),
        "memory_operation_present": memory_operation_present,
        "memory_operation_correct": memory_operation_correct,
        "memory_operation_f1_class": expected_operation,
        "exact_state_map": exact_state_map,
        "memory_state_accuracy": 1.0 if state_total == 0 else state_correct / state_total,
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_f1": evidence_f1,
        "diagnosis_correct": diagnosis_correct,
        "followup_answer_present": followup_answer_present,
        "followup_answer_correct": followup_answer_correct,
        "strict_joint": strict_joint,
        "unsafe_reuse": unsafe_reuse,
        "downstream_contamination": downstream_contamination,
    }


def aggregate_scores(rows: list[dict[str, Any]], group_by: list[str] | None = None) -> list[dict[str, Any]]:
    fields = (
        "schema_valid",
        "decision_correct",
        "memory_operation_present",
        "memory_operation_correct",
        "exact_state_map",
        "memory_state_accuracy",
        "evidence_f1",
        "diagnosis_correct",
        "followup_answer_present",
        "followup_answer_correct",
        "strict_joint",
        "unsafe_reuse",
        "downstream_contamination",
    )
    group_by = group_by or []
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("method"), row.get("model"), *(row.get(field) for field in group_by))
        buckets[key].append(row)
    out: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items(), key=lambda item: str(item[0])):
        method, model, *group_values = key
        item = {"method": method, "model": model, "n": len(bucket)}
        for field, value in zip(group_by, group_values):
            item[field] = value
        for field in fields:
            item[field] = sum(float(row.get(field, 0.0)) for row in bucket) / len(bucket)
        out.append(item)
    return out
