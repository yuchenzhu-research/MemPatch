"""Scoring utilities for the general English ReTrace-Bench release."""

from __future__ import annotations

from collections import Counter
from typing import Any

from benchmark.retrace_bench.general_taxonomy import (
    DECISIONS,
    FAILURE_MODES,
    NON_ANSWER_DECISIONS,
)


# Paper-facing headline metrics. These are what ReTrace-Bench reports as the
# primary scores in tables; they are computed in aggregate_metrics() (not as
# per-example raw signals). decision_macro_f1 is the primary decision metric
# because it is robust to the dominant use_current_memory class.
HEADLINE_METRICS = (
    "decision_macro_f1",
    "non_answer_decision_accuracy",
    "memory_state_accuracy",
    "evidence_f1",
    "failure_diagnosis_accuracy",
    "stale_reuse_rate",
    "joint_revision_success",
    "minimal_evidence_exact_match",
    "evidence_precision",
    "overcitation_rate",
    "counterevidence_recall",
    "answer_state_consistency",
    "scope_authority_accuracy",
    "latest_event_shortcut_failure_rate",
)

# Auxiliary / diagnostic raw signals. Reported for completeness but not as the
# headline numbers. In particular black_box_decision_accuracy can be dominated
# by the majority use_current_memory class, so it is auxiliary, not headline.
AUXILIARY_METRICS = (
    "black_box_decision_accuracy",
    "decision_balanced_accuracy",
    "use_current_memory_accuracy",
    "answer_key_fact_accuracy",
    "answer_exact_match",
    "format_failure_rate",
)


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


_PUNCT = ".,;:!?()[]{}\"'`"


def _toks(value: Any) -> set[str]:
    """Punctuation-robust token set used for overlap / F1 comparisons."""
    return {tok.strip(_PUNCT) for tok in _norm(value).split() if tok.strip(_PUNCT)}


def normalize_failure_mode(value: Any) -> str:
    # Some models emit failure_diagnosis as a single-element list/tuple
    # (e.g. ["under_update"]) rather than a bare enum string. Unwrap it so the
    # exact-match path below applies; strings are unaffected (backward compatible).
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
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


def answer_exact_match(predicted: Any, expected: Any) -> bool:
    """Strict normalized equality. Diagnostic only; too strict for open text."""
    exp = _norm(expected)
    return bool(exp) and _norm(predicted) == exp


def token_f1_match(predicted: Any, expected: Any, min_recall: float = 0.80, min_f1: float = 0.60) -> bool:
    """Token-level recall+F1 gate.

    Crucially this does NOT pass just because ``expected`` appears as a
    substring of ``predicted``: a massive retrieve-all answer that stuffs the
    expected string into unrelated text drives F1 down (the prediction token
    set dwarfs the expected one) and fails the F1 threshold.
    """
    pred_toks = _toks(predicted)
    exp_toks = _toks(expected)
    if not pred_toks or not exp_toks:
        return False
    tp = len(pred_toks & exp_toks)
    recall = tp / len(exp_toks)
    f1 = 2 * tp / (len(pred_toks) + len(exp_toks))
    return recall >= min_recall and f1 >= min_f1


def _overlap_hit(predicted: Any, phrases: Any, min_overlap: float = 0.75) -> bool:
    """True if any phrase's tokens are mostly present in ``predicted``."""
    pred_toks = _toks(predicted)
    if not pred_toks:
        return False
    for phrase in phrases or []:
        phrase_toks = _toks(phrase)
        if phrase_toks and len(pred_toks & phrase_toks) / len(phrase_toks) >= min_overlap:
            return True
    return False


def key_fact_matches(predicted: Any, expected: Any, rubric: dict[str, Any] | None = None) -> bool:
    """Rubric-first, otherwise token-F1 key-fact match for workflow answers.

    Scoring order:
    * ``rubric.must_not_include`` present and hit  -> fail (forbidden stale /
      scope / policy facts must never appear).
    * ``rubric.must_include`` present              -> require every required
      fact (by token overlap), never substring stuffing.
    * otherwise                                    -> token-level F1 against the
      expected answer (recall >= 0.80 and F1 >= 0.60).

    This deliberately removes the previous ``expected in predicted`` substring
    shortcut, which let retrieve-all style answers score near-perfect.
    """
    pred = _norm(predicted)
    if not pred:
        return False
    rubric = rubric or {}
    must_not_include = rubric.get("must_not_include")
    if must_not_include and _overlap_hit(predicted, must_not_include):
        return False
    must_include = rubric.get("must_include")
    if must_include:
        return all(_overlap_hit(predicted, [required], min_overlap=0.80) for required in must_include)
    return token_f1_match(predicted, expected)


def decision_matches(predicted: Any, expected: Any, aliases: Any = None) -> bool:
    """Strict normalized equality for enum-like decisions.

    Substring matching is intentionally NOT used: ``reject_refund`` must not
    match ``do_not_reject_refund`` or a sentence that merely contains it.
    Aliases are honored ONLY when explicitly supplied via ``decision_aliases``.
    """
    pred = _norm(predicted)
    exp = _norm(expected)
    if not exp:
        return False
    if pred == exp:
        return True
    if isinstance(aliases, dict):
        accepted = aliases.get(expected) or aliases.get(exp) or []
        if isinstance(accepted, str):
            accepted = [accepted]
        return any(pred == _norm(alias) for alias in accepted)
    return False


def _is_stale_reuse(
    pred_answer: Any,
    stale_list: Any,
    min_overlap: float = 0.75,
    expected_answer: Any = None,
) -> bool:
    """Detect paraphrased reuse of a stale/wrong answer by token overlap.

    When ``expected_answer`` is supplied we compare against each stale answer's
    *distinctive* tokens (those not shared with the correct answer) so that a
    correct answer which happens to be lexically close to a stale one is not
    flagged.
    """
    pred_toks = _toks(pred_answer)
    if not pred_toks:
        return False
    exp_toks = _toks(expected_answer) if expected_answer else set()
    for stale in stale_list or []:
        stale_toks = _toks(stale)
        distinctive = (stale_toks - exp_toks) or stale_toks
        if distinctive and len(pred_toks & distinctive) / len(distinctive) >= min_overlap:
            return True
    return False


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

    rubric = gold.get("rubric", {}) or {}
    answer = response.get("answer")
    expected_answer = gold.get("expected_answer")
    decision_aliases = gold.get("decision_aliases") or rubric.get("decision_aliases") or scenario.get("decision_aliases")

    # Optional adversarial anchors (gold first, then rubric). Absent in the
    # current sample but consumed when datasets provide them.
    def _anchors(key: str) -> list[Any]:
        return list(gold.get(key) or rubric.get(key) or [])

    stale_anchors = _anchors("stale_anchors")
    scope_anchors = _anchors("scope_leakage_anchors")
    policy_anchors = _anchors("policy_violation_anchors")
    must_not_include = list(rubric.get("must_not_include") or [])

    stale_anchor_hit = _overlap_hit(answer, stale_anchors)
    scope_anchor_hit = _overlap_hit(answer, scope_anchors)
    policy_anchor_hit = _overlap_hit(answer, policy_anchors)
    forbidden_fact_hits = sum(1 for phrase in must_not_include if _overlap_hit(answer, [phrase]))

    stale_reuse = (
        _is_stale_reuse(answer, gold.get("stale_or_wrong_answers", []), expected_answer=expected_answer)
        or stale_anchor_hit
    )

    decision_ok = decision_matches(predicted_decision, expected_decision, decision_aliases)
    memory_ok = (state_correct / state_total) >= 1.0
    key_fact_ok = key_fact_matches(answer, expected_answer, rubric)

    pred_ev = set(response.get("evidence_event_ids", []) or [])
    gold_ev = set(gold.get("expected_evidence_event_ids", []) or gold.get("minimal_evidence_event_ids", []) or [])
    
    if not gold_ev and not pred_ev:
        evidence_precision = 1.0
    elif not pred_ev:
        evidence_precision = 0.0
    else:
        evidence_precision = len(pred_ev & gold_ev) / len(pred_ev)

    overcitation_rate = len(pred_ev - gold_ev) / len(pred_ev) if pred_ev else 0.0

    gold_counter = set(gold.get("counterevidence_event_ids", []) or [])
    if not gold_counter:
        counterevidence_recall = 1.0
    elif not pred_ev:
        counterevidence_recall = 0.0
    else:
        counterevidence_recall = len(pred_ev & gold_counter) / len(gold_counter)

    answer_state_consistency = float(decision_ok and key_fact_ok and memory_ok)
    evidence_f1 = _f1(response.get("evidence_event_ids", []), list(gold_ev))

    joint_revision_success = float(
        decision_ok
        and memory_ok
        and evidence_f1 >= 1.0
        and answer_state_consistency >= 1.0
        and float(stale_reuse) == 0.0
    )

    meta = scenario.get("metadata", {}) or {}
    is_scope_auth = (
        meta.get("authority_conflict", False)
        or meta.get("scope_collision", False)
        or meta.get("version_or_release_chain", False)
        or meta.get("branch_scope_conflict", False)
        or scenario.get("primary_failure_mode") in {"scope_leakage", "wrong_source_attribution", "policy_violation"}
    )
    scope_authority_accuracy = float(decision_ok) if is_scope_auth else 1.0

    events = scenario.get("public_input", {}).get("event_trace", [])
    sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
    latest_event_id = sorted_events[-1].get("event_id") if sorted_events else None
    
    is_shortcut_candidate = False
    if latest_event_id and latest_event_id not in gold_ev:
        is_shortcut_candidate = (not decision_ok) and (latest_event_id in pred_ev)
    latest_event_shortcut_failure_rate = float(is_shortcut_candidate)

    metrics = {
        # Per-example raw signals. Paper-facing headline decision metrics
        # (decision_macro_f1, non_answer_decision_accuracy, ...) are computed in
        # aggregate_metrics(); see HEADLINE_METRICS / AUXILIARY_METRICS. Note
        # that black_box_decision_accuracy is an auxiliary raw signal, not the
        # headline decision metric, because it is dominated by the majority
        # use_current_memory class.
        "black_box_decision_accuracy": float(decision_ok),
        "memory_state_accuracy": state_correct / state_total,
        "evidence_f1": evidence_f1,
        "failure_diagnosis_accuracy": float(expected_diag == predicted_diag),
        "stale_reuse_rate": float(stale_reuse),
        # New AAAI headline metrics
        "joint_revision_success": joint_revision_success,
        "minimal_evidence_exact_match": float(pred_ev == gold_ev),
        "evidence_precision": evidence_precision,
        "overcitation_rate": overcitation_rate,
        "counterevidence_recall": counterevidence_recall,
        "answer_state_consistency": answer_state_consistency,
        "scope_authority_accuracy": scope_authority_accuracy,
        "latest_event_shortcut_failure_rate": latest_event_shortcut_failure_rate,
        # Secondary / diagnostic metrics (not headline).
        "answer_key_fact_accuracy": float(key_fact_ok),
        # Exact text is retained only as a diagnostic; it is too strict for
        # open-ended language and should not be a headline metric.
        "answer_exact_match": float(answer_exact_match(answer, expected_answer)),
    }
    # NOTE: legacy ``answer_accuracy`` and ``decision_accuracy`` were duplicates
    # of ``answer_key_fact_accuracy`` and ``black_box_decision_accuracy`` and
    # have been removed as headline metrics.
    for mode in FAILURE_MODES:
        metrics[f"{mode}_rate"] = float(scenario.get("primary_failure_mode") == mode and predicted_diag == mode)
    metrics["under_update_rate"] = float(predicted_diag == "under_update")
    metrics["over_update_rate"] = float(predicted_diag == "over_update")
    metrics["scope_leakage_rate"] = float(predicted_diag == "scope_leakage")
    metrics["policy_violation_rate"] = float(predicted_diag == "policy_violation")
    # Format failure: no parseable decision (e.g. LLM emitted invalid JSON).
    metrics["format_failure_rate"] = float(predicted_decision is None)
    metrics["stale_anchor_hit_rate"] = float(stale_anchor_hit)
    metrics["scope_leakage_anchor_hit_rate"] = float(scope_anchor_hit)
    metrics["policy_violation_anchor_hit_rate"] = float(policy_anchor_hit)
    metrics["forbidden_fact_hits"] = float(forbidden_fact_hits)
    return metrics


# Backward-compatible alias of the centralized decision label space defined in
# general_taxonomy.DECISIONS. Kept as a set for existing membership checks.
ALL_DECISIONS = set(DECISIONS)


def canonicalize_decision(val: Any, expected_decision: Any, aliases: Any = None) -> str:
    if decision_matches(val, expected_decision, aliases):
        return expected_decision
    for dec in ALL_DECISIONS:
        if decision_matches(val, dec, aliases):
            return dec
    return _norm(val)


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    count = 0
    for row in rows:
        for key, value in row.get("metrics", {}).items():
            totals[key] += float(value)
        count += 1

    metrics_dict = {key: value / count for key, value in sorted(totals.items())} if count else {}

    valid_rows = [r for r in rows if r.get("expected_decision") is not None]

    decision_macro_f1 = 0.0
    decision_balanced_accuracy = 0.0
    non_answer_decision_accuracy = 0.0
    use_current_memory_accuracy = 0.0
    per_decision_counts = {}
    per_decision_accuracy = {}

    if valid_rows:
        exp_list = []
        pred_list = []
        for r in valid_rows:
            exp_dec = r["expected_decision"]
            aliases = r.get("decision_aliases")
            pred_raw = r.get("response", {}).get("decision", r.get("response", {}).get("expected_decision"))

            exp_c = canonicalize_decision(exp_dec, exp_dec, aliases)
            pred_c = canonicalize_decision(pred_raw, exp_dec, aliases)
            exp_list.append(exp_c)
            pred_list.append(pred_c)

        expected_classes = sorted(list(set(exp_list)))

        recalls = []
        f1s = []
        for c in expected_classes:
            tp = sum(1 for p, e in zip(pred_list, exp_list) if e == c and p == c)
            fp = sum(1 for p, e in zip(pred_list, exp_list) if e != c and p == c)
            fn = sum(1 for p, e in zip(pred_list, exp_list) if e == c and p != c)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            recalls.append(recall)
            f1s.append(f1)

            per_decision_counts[c] = tp + fn
            per_decision_accuracy[c] = recall

        if expected_classes:
            decision_balanced_accuracy = sum(recalls) / len(expected_classes)
            decision_macro_f1 = sum(f1s) / len(expected_classes)

        non_answer_classes = set(NON_ANSWER_DECISIONS)
        non_answer_indices = [i for i, e in enumerate(exp_list) if e in non_answer_classes]
        if non_answer_indices:
            non_answer_correct = sum(1 for i in non_answer_indices if pred_list[i] == exp_list[i])
            non_answer_decision_accuracy = non_answer_correct / len(non_answer_indices)
        else:
            non_answer_decision_accuracy = 0.0

        use_current_indices = [i for i, e in enumerate(exp_list) if e == "use_current_memory"]
        if use_current_indices:
            use_current_correct = sum(1 for i in use_current_indices if pred_list[i] == exp_list[i])
            use_current_memory_accuracy = use_current_correct / len(use_current_indices)
        else:
            use_current_memory_accuracy = 0.0

    metrics_dict["decision_macro_f1"] = decision_macro_f1
    metrics_dict["decision_balanced_accuracy"] = decision_balanced_accuracy
    metrics_dict["non_answer_decision_accuracy"] = non_answer_decision_accuracy
    metrics_dict["use_current_memory_accuracy"] = use_current_memory_accuracy
    metrics_dict["per_decision_counts"] = per_decision_counts
    metrics_dict["per_decision_accuracy"] = per_decision_accuracy

    observed_modes = sorted(list(set(r.get("primary_failure_mode") for r in rows if r.get("primary_failure_mode") is not None)))
    per_failure_mode = {}
    for mode in observed_modes:
        mode_rows = [r for r in rows if r.get("primary_failure_mode") == mode]
        per_failure_mode[mode] = {
            "count": len(mode_rows),
            "black_box_decision_accuracy": sum(r.get("metrics", {}).get("black_box_decision_accuracy", 0.0) for r in mode_rows) / len(mode_rows),
            "memory_state_accuracy": sum(r.get("metrics", {}).get("memory_state_accuracy", 0.0) for r in mode_rows) / len(mode_rows),
            "evidence_f1": sum(r.get("metrics", {}).get("evidence_f1", 0.0) for r in mode_rows) / len(mode_rows),
            "failure_diagnosis_accuracy": sum(r.get("metrics", {}).get("failure_diagnosis_accuracy", 0.0) for r in mode_rows) / len(mode_rows),
            "stale_reuse_rate": sum(r.get("metrics", {}).get("stale_reuse_rate", 0.0) for r in mode_rows) / len(mode_rows),
        }

    observed_domains = sorted(list(set(r.get("domain") for r in rows if r.get("domain") is not None)))
    per_domain = {}
    for domain in observed_domains:
        domain_rows = [r for r in rows if r.get("domain") == domain]
        per_domain[domain] = {
            "count": len(domain_rows),
            "black_box_decision_accuracy": sum(r.get("metrics", {}).get("black_box_decision_accuracy", 0.0) for r in domain_rows) / len(domain_rows),
            "memory_state_accuracy": sum(r.get("metrics", {}).get("memory_state_accuracy", 0.0) for r in domain_rows) / len(domain_rows),
            "evidence_f1": sum(r.get("metrics", {}).get("evidence_f1", 0.0) for r in domain_rows) / len(domain_rows),
            "failure_diagnosis_accuracy": sum(r.get("metrics", {}).get("failure_diagnosis_accuracy", 0.0) for r in domain_rows) / len(domain_rows),
        }

    # Grouped views so paper-facing scripts know which numbers are headline vs.
    # auxiliary without having to hard-code metric names. ``metrics`` is kept as
    # the flat all-metrics dict for backward compatibility with existing
    # consumers (e.g. scripts/run_retrace_bench_ablation.py:load_metrics).
    headline_metrics = {k: metrics_dict[k] for k in HEADLINE_METRICS if k in metrics_dict}
    auxiliary_metrics = {k: metrics_dict[k] for k in AUXILIARY_METRICS if k in metrics_dict}

    return {
        "count": count,
        "metrics": metrics_dict,
        "headline_metrics": headline_metrics,
        "auxiliary_metrics": auxiliary_metrics,
        "all_metrics": metrics_dict,
        "per_decision_counts": per_decision_counts,
        "per_decision_accuracy": per_decision_accuracy,
        "per_failure_mode": per_failure_mode,
        "per_domain": per_domain,
    }

