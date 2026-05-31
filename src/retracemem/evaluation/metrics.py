"""Unified metrics evaluation module for ReTrace-Learn-Full (Protocol A & B).
"""
from __future__ import annotations

from typing import Any, Dict, List


def calculate_rates(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    gold_final_statuses: dict[str, str],
    gold_actions: list[dict[str, Any]] | None = None,
    valid_belief_ids: set[str] | None = None,
    valid_condition_ids: set[str] | None = None,
    valid_evidence_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compute all unified evaluation metrics for a single episode run."""
    n_preds = len(predictions)
    valid_json_count = sum(1 for p in predictions if p.get("parser_result", {}).get("valid_json", True))
    valid_json_rate = calculate_rates(valid_json_count, n_preds)
    parser_error_rate = 1.0 - valid_json_rate

    # Gather actions and gate proposals
    all_actions = []
    gate_proposals = []
    defeat_paths = []
    dpa_final_statuses = {}
    for p in predictions:
        all_actions.extend(p.get("sampled_actions", []))
        gate_proposals.extend(p.get("gate_decisions", []))
        defeat_paths.extend(p.get("defeat_paths", []))
        dpa_final_statuses.update(p.get("dpa_final_statuses", {}))

    # Action type accuracy
    action_type_correct = 0
    total_actions = len(all_actions)
    if gold_actions:
        for idx, act in enumerate(all_actions):
            act_type = act.get("action_type")
            # Simple position match or target match
            matched = False
            for g_act in gold_actions:
                if (
                    g_act.get("action_type") == act_type
                    and g_act.get("target_belief_id") == act.get("target_belief_id")
                    and g_act.get("target_condition_id") == act.get("target_condition_id")
                ):
                    action_type_correct += 1
                    matched = True
                    break
            if not matched and idx < len(gold_actions):
                if gold_actions[idx].get("action_type") == act_type:
                    action_type_correct += 1
        action_type_accuracy = calculate_rates(action_type_correct, max(total_actions, len(gold_actions)))
    else:
        action_type_accuracy = 1.0 if total_actions == 0 else 0.0

    # Grounding accuracy
    grounded_targets = 0
    grounded_evidence = 0
    revision_actions = [a for a in all_actions if a.get("action_type") != "NO_REVISION"]
    n_rev = len(revision_actions)
    
    for a in revision_actions:
        target_ok = True
        if valid_belief_ids and a.get("target_belief_id") is not None:
            target_ok = target_ok and a["target_belief_id"] in valid_belief_ids
        if valid_belief_ids and a.get("replacement_belief_id") is not None:
            target_ok = target_ok and a["replacement_belief_id"] in valid_belief_ids
        if valid_condition_ids and a.get("target_condition_id") is not None:
            target_ok = target_ok and a["target_condition_id"] in valid_condition_ids
        if target_ok:
            grounded_targets += 1
        if valid_evidence_ids and a.get("evidence_ids"):
            if all(ev in valid_evidence_ids for ev in a["evidence_ids"]):
                grounded_evidence += 1

    target_grounding = calculate_rates(grounded_targets, n_rev)
    evidence_grounding = calculate_rates(grounded_evidence, n_rev)

    # Gate rejection rate
    total_proposals = len(gate_proposals)
    admitted_proposals = sum(1 for g in gate_proposals if g.get("admitted", True))
    gate_rejection_rate = calculate_rates(total_proposals - admitted_proposals, total_proposals)

    # Status Correctness and over/under updates
    gold_keys = list(gold_final_statuses.keys())
    correct_statuses = sum(
        1
        for bid in gold_keys
        if dpa_final_statuses.get(bid) == gold_final_statuses[bid]
    )
    final_status_accuracy = calculate_rates(correct_statuses, len(gold_keys))

    over_updates = 0
    under_updates = 0
    stale_propagations = 0
    uncertainty_errors = 0
    for bid in gold_keys:
        gold = gold_final_statuses[bid]
        pred = dpa_final_statuses.get(bid, "UNRESOLVED")
        if gold == "AUTHORIZED" and pred != "AUTHORIZED":
            over_updates += 1
        if gold != "AUTHORIZED" and pred == "AUTHORIZED":
            under_updates += 1
            if gold in ("SUPERSEDED", "BLOCKED"):
                stale_propagations += 1
        if (gold == "UNRESOLVED" and pred != "UNRESOLVED") or (gold != "UNRESOLVED" and pred == "UNRESOLVED"):
            uncertainty_errors += 1

    n_gold = len(gold_keys)
    over_update_rate = calculate_rates(over_updates, n_gold)
    under_update_rate = calculate_rates(under_updates, n_gold)
    stale_propagation_rate = calculate_rates(stale_propagations, n_gold)
    uncertainty_error_rate = calculate_rates(uncertainty_errors, n_gold)

    # NO_REVISION Overuse
    pred_no_rev_count = sum(1 for a in all_actions if a.get("action_type") == "NO_REVISION")
    gold_no_rev_count = sum(1 for a in gold_actions or [] if a.get("action_type") == "NO_REVISION")
    if pred_no_rev_count > gold_no_rev_count:
        no_revision_overuse = calculate_rates(pred_no_rev_count - gold_no_rev_count, max(1, pred_no_rev_count))
    else:
        no_revision_overuse = 0.0

    # Audit completeness
    excluded_beliefs = [bid for bid, status in dpa_final_statuses.items() if status != "AUTHORIZED"]
    audit_documented = 0
    for bid in excluded_beliefs:
        if any(dp.get("belief_id") == bid for dp in defeat_paths):
            audit_documented += 1
    audit_completeness = calculate_rates(audit_documented, len(excluded_beliefs))

    return {
        "valid_json_rate": valid_json_rate,
        "parser_error_rate": parser_error_rate,
        "action_type_accuracy": action_type_accuracy,
        "target_grounding": target_grounding,
        "evidence_grounding": evidence_grounding,
        "gate_rejection_rate": gate_rejection_rate,
        "final_status_accuracy": final_status_accuracy,
        "over_update_rate": over_update_rate,
        "under_update_rate": under_update_rate,
        "stale_propagation_rate": stale_propagation_rate,
        "uncertainty_error_rate": uncertainty_error_rate,
        "no_revision_overuse": no_revision_overuse,
        "audit_completeness": audit_completeness,
    }


def aggregate_metrics(episode_results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall macro averages across multiple episodes."""
    if not episode_results:
        return {}
    keys = episode_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in episode_results if r[k] is not None]
        agg[k] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return agg
