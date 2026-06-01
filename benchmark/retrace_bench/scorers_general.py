"""Scoring utilities for the general English ReTrace-Bench release."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from benchmark.retrace_bench.general_taxonomy import FAILURE_MODES


DIAGNOSIS_ALIASES = {
    "stale_memory_reuse": ("stale", "outdated", "obsolete", "superseded", "earlier instruction", "old note"),
    "under_update": ("under update", "failed to update", "not incorporate", "missed update"),
    "over_update": ("over update", "overwrote", "over-applied", "too broad"),
    "conflict_collapse": ("conflict", "incompatible", "unresolved", "collapse"),
    "scope_leakage": ("scope", "out of scope", "cross-scope", "wrong workspace"),
    "policy_violation": ("policy", "private", "credential", "secure", "refuse"),
    "wrong_source_attribution": ("source", "attribution", "wrong source", "misattributed"),
    "memory_hallucination": ("hallucination", "unsupported", "false premise", "missing fact"),
    "unnecessary_memory_write": ("unnecessary", "should not store", "no need to store"),
    "failure_to_forget": ("forget", "deleted", "remove obsolete"),
    "failure_to_release_or_restore": ("restore", "release", "cleared", "temporary block"),
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_failure_mode(value: Any) -> str:
    text = _norm(value)
    if text in FAILURE_MODES:
        return text
    text = text.replace("-", "_").replace(" ", "_")
    if text in FAILURE_MODES:
        return text
    raw = _norm(value)
    for mode, aliases in DIAGNOSIS_ALIASES.items():
        if any(alias in raw for alias in aliases):
            return mode
    return raw


def answer_matches(predicted: Any, expected: Any) -> bool:
    pred = _norm(predicted)
    exp = _norm(expected)
    if pred == exp:
        return True
    return bool(exp and exp in pred)


def key_fact_matches(predicted: Any, expected: Any) -> bool:
    """Lightweight non-LLM key-fact match for synthetic workflow answers.

    This is deliberately not a semantic judge. It checks whether the answer
    preserves the scenario-specific anchors (IDs and update/action keywords)
    that make the black-box action correct. Open-ended answer quality should be
    judged by a separate LLM-as-judge, not by exact string equality.
    """
    pred = _norm(predicted)
    exp = _norm(expected)
    if not exp:
        return False
    if exp in pred:
        return True
    expected_ids = set(re.findall(r"\b(?:c|emp|proj)-[a-z0-9-]+\b", exp))
    if expected_ids and not expected_ids.issubset(set(re.findall(r"\b(?:c|emp|proj)-[a-z0-9-]+\b", pred))):
        return False
    keyword_groups = (
        ("updated", "new", "current"),
        ("earlier", "old", "obsolete"),
        ("restore", "restored", "release", "cleared"),
        ("remove", "deleted", "forget"),
        ("private", "credential", "secure", "policy"),
        ("unresolved", "incompatible", "conflict"),
        ("scope", "workspace"),
    )
    exp_groups = [group for group in keyword_groups if any(word in exp for word in group)]
    if exp_groups:
        return all(any(word in pred for word in group) for group in exp_groups)
    return bool(expected_ids and expected_ids.issubset(set(re.findall(r"\b(?:c|emp|proj)-[a-z0-9-]+\b", pred))))


def decision_matches(predicted: Any, expected: Any) -> bool:
    pred = _norm(predicted)
    exp = _norm(expected)
    if pred == exp:
        return True
    return bool(exp and exp in pred)


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
    predicted_diag = normalize_failure_mode(response.get("failure_diagnosis", response.get("expected_failure_diagnosis")))

    metrics = {
        # Exact text is retained only as a diagnostic; it is too strict for
        # open-ended language and should not be a headline metric.
        "answer_exact_match": float(answer_matches(response.get("answer"), gold.get("expected_answer"))),
        "answer_key_fact_accuracy": float(key_fact_matches(response.get("answer"), gold.get("expected_answer"))),
        "black_box_decision_accuracy": float(decision_matches(predicted_decision, expected_decision)),
        "answer_accuracy": float(key_fact_matches(response.get("answer"), gold.get("expected_answer"))),
        "decision_accuracy": float(decision_matches(predicted_decision, expected_decision)),
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
