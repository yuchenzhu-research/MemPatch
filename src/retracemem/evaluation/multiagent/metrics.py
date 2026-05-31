"""Shared evaluation metrics for the multi-agent A/B/C pipeline.

Pure functions: grounding checks, Stage A action metrics, the action->final-status
comparability map, and the aggregate metric computation. No I/O, no proposers.
"""
from __future__ import annotations

from typing import Any

from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateSubmission,
    FixedCandidateInputEpisode,
    FixedCandidateGoldRecord,
    TypedRevisionTarget,
)


_STATUS_MAP_A_TO_COMPARABLE = {
    "AUTHORIZED": "USABLE",
    "BLOCKED": "NOT_USABLE",
    "SUPERSEDED": "NOT_USABLE",
    "UNRESOLVED": "UNCERTAIN",
}


def check_grounding_error_stage_a(action: dict[str, Any], sub: FixedCandidateSubmission) -> bool:
    """Return True if there is a grounding error in Stage A action."""
    action_type = action.get("action_type")
    if action_type == "NO_REVISION":
        return False
        
    target_belief_id = action.get("target_belief_id")
    target_condition_id = action.get("target_condition_id")
    replacement_belief_id = action.get("replacement_belief_id")
    evidence_ids = action.get("evidence_ids") or []

    # Grounding check for target belief/condition
    valid_belief_ids = {b.belief_id for b in sub.candidate_beliefs}
    valid_replacement_ids = {b.belief_id for b in sub.candidate_replacement_beliefs}
    valid_condition_ids = {
        c.condition_id
        for _, conds in sub.candidate_conditions_by_belief
        for c in conds
    }
    valid_evidence_ids = {e.evidence_id for e in sub.evidence_context} | {sub.new_evidence_id}

    if action_type in ("SUPERSEDES", "UNCERTAIN", "REAFFIRMS"):
        if not target_belief_id or target_belief_id not in valid_belief_ids:
            return True
        if action_type == "SUPERSEDES":
            if not replacement_belief_id or replacement_belief_id not in valid_replacement_ids:
                return True
    elif action_type in ("BLOCKS", "RELEASES"):
        if not target_condition_id or target_condition_id not in valid_condition_ids:
            return True
            
    for eid in evidence_ids:
        if eid not in valid_evidence_ids:
            return True

    return False


def check_grounding_error_stage_b(verdict: dict[str, Any], valid_belief_ids: set[str]) -> bool:
    """Return True if Stage B verdict contains an invalid raw belief_id."""
    bid = verdict.get("raw_belief_id")
    return bid not in valid_belief_ids


def compute_stage_a_action_metrics(
    pred_actions: list[dict[str, Any]],
    gold_actions: tuple[TypedRevisionTarget, ...],
    sub: FixedCandidateSubmission,
) -> dict[str, float]:
    """Compute fine-grained metrics for Stage A proposal action match."""
    valid_json = 1.0  # Assumes parsed if we reached here
    
    # Grounding checks
    target_grounding_correct = 0
    total_pred = len(pred_actions)
    for act in pred_actions:
        if not check_grounding_error_stage_a(act, sub):
            target_grounding_correct += 1
    target_grounding = target_grounding_correct / total_pred if total_pred > 0 else 1.0

    # Match exact actions
    def canonical_action_tuple(a: dict[str, Any] | TypedRevisionTarget) -> tuple:
        if isinstance(a, dict):
            evs = tuple(sorted(a.get("evidence_ids") or []))
            return (
                a.get("action_type"),
                a.get("target_belief_id"),
                a.get("target_condition_id"),
                a.get("replacement_belief_id"),
                evs,
            )
        else:
            evs = tuple(sorted(a.evidence_ids))
            return (
                a.action_type,
                a.target_belief_id,
                a.target_condition_id,
                a.replacement_belief_id,
                evs,
            )

    pred_tuples = {canonical_action_tuple(a) for a in pred_actions}
    gold_tuples = {canonical_action_tuple(a) for a in gold_actions}
    exact_action_match = 1.0 if pred_tuples == gold_tuples else 0.0

    # Action type and evidence grounding
    action_type_match_correct = 0
    evidence_grounding_correct = 0
    matched_count = 0

    for pred_act in pred_actions:
        pred_target = pred_act.get("target_belief_id") or pred_act.get("target_condition_id")
        pred_action_type = pred_act.get("action_type")
        
        # Match NO_REVISION
        if pred_action_type == "NO_REVISION":
            for gold_act in gold_actions:
                if gold_act.action_type == "NO_REVISION":
                    matched_count += 1
                    action_type_match_correct += 1
                    pred_evs = set(pred_act.get("evidence_ids") or [])
                    gold_evs = set(gold_act.evidence_ids)
                    if pred_evs == gold_evs:
                         evidence_grounding_correct += 1
                    break
            continue

        if not pred_target:
            continue
        # Find matching gold action by target
        for gold_act in gold_actions:
            gold_target = gold_act.target_belief_id or gold_act.target_condition_id
            if pred_target == gold_target:
                matched_count += 1
                if pred_act.get("action_type") == gold_act.action_type:
                    action_type_match_correct += 1
                
                pred_evs = set(pred_act.get("evidence_ids") or [])
                gold_evs = set(gold_act.evidence_ids)
                if pred_evs == gold_evs:
                    evidence_grounding_correct += 1
                break

    if not pred_actions and not gold_actions:
        action_type_match = 1.0
        evidence_grounding = 1.0
    else:
        action_type_match = action_type_match_correct / max(len(pred_actions), len(gold_actions))
        evidence_grounding = evidence_grounding_correct / max(len(pred_actions), len(gold_actions))

    gold_is_no_rev = not gold_actions or any(a.action_type == "NO_REVISION" for a in gold_actions)
    pred_is_no_rev = not pred_actions or any(a.get("action_type") == "NO_REVISION" for a in pred_actions)
    no_revision_match = 1.0 if gold_is_no_rev == pred_is_no_rev else 0.0

    return {
        "valid_json": valid_json,
        "action_type_match": action_type_match,
        "target_grounding": target_grounding,
        "evidence_grounding": evidence_grounding,
        "exact_action_match": exact_action_match,
        "no_revision_match": no_revision_match,
    }



def compute_eval_metrics(
    processed_cases: list[tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]],
    stage_a_parsed_rows: list[dict[str, Any]],
    stage_b_parsed_rows: list[dict[str, Any]],
    stage_a_raw_rows: list[dict[str, Any]],
    stage_b_raw_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Computes all fine-grained and aggregate evaluation metrics."""
    # Build maps of final results for easy indexing
    stage_a_results_map = {r["episode_id"]: r for r in stage_a_parsed_rows}
    stage_b_results_map = {r["episode_id"]: r for r in stage_b_parsed_rows}
    stage_a_raw_map = {r["episode_id"]: r for r in stage_a_raw_rows}
    stage_b_raw_map = {r["episode_id"]: r for r in stage_b_raw_rows}

    # Case Breakdown Table Setup
    failure_breakdown_rows = []
    episode_accuracies = []
    episode_em_count = 0

    # Overall counters for global metrics
    stage_metrics = {
        "stage_a": {
            "correct_beliefs": 0,
            "total_beliefs": 0,
            "over_updates": 0,
            "not_usable_total": 0,
            "under_updates": 0,
            "usable_total": 0,
            "uncertainty_errors": 0,
            "grounding_errors": 0,
            "valid_outputs": 0,
            "total_outputs": 0,
        },
        "stage_b_strict": {
            "correct_beliefs": 0,
            "total_beliefs": 0,
            "over_updates": 0,
            "not_usable_total": 0,
            "under_updates": 0,
            "usable_total": 0,
            "uncertainty_errors": 0,
        },
        "stage_b_canonicalized": {
            "correct_beliefs": 0,
            "total_beliefs": 0,
            "over_updates": 0,
            "not_usable_total": 0,
            "under_updates": 0,
            "usable_total": 0,
            "uncertainty_errors": 0,
        },
        "stage_b_common": {
            "grounding_errors": 0,
            "valid_outputs": 0,
            "total_outputs": 0,
        }
    }

    # Stage A Action-level metrics counters
    action_metrics_counters = {
        "valid_json": [],
        "action_type_match": [],
        "target_grounding": [],
        "evidence_grounding": [],
        "exact_action_match": [],
        "no_revision_match": [],
        "false_no_revision": [],
        "multi_action_recall": [],
        "parser_error": [],
        "first_pass_valid_json": [],
        "first_pass_parser_error": [],
        "repair_triggered": [],
        "repair_success": [],
    }

    # Stage B Canonicalization counters
    total_stage_b_verdicts = 0
    canonicalized_stage_b_verdicts = 0
    fuzzy_stage_b_verdicts = 0

    for ep, gold in processed_cases:
        ep_id = ep.episode_id

        gold_statuses = gold.gold_snapshot.belief_statuses
        
        # Pull Results
        res_a = stage_a_results_map.get(ep_id, {})
        res_b = stage_b_results_map.get(ep_id, {})
        raw_a = stage_a_raw_map.get(ep_id, {})
        raw_b = stage_b_raw_map.get(ep_id, {})

        pred_a_statuses = res_a.get("final_belief_statuses", {})
        strict_pred_b_statuses = res_b.get("strict_final_belief_statuses", {})
        canonical_pred_b_statuses = res_b.get("canonicalized_final_belief_statuses", {})

        # Compute Action Metrics for Stage A only per submission
        for s_parsed in res_a.get("submissions", []):
            sub_id = s_parsed["submission_id"]
            orig_sub = next(s for s in ep.submissions if s.submission_id == sub_id)
            pred_actions_for_sub = s_parsed.get("actions", [])
            gold_actions_for_sub = tuple(t for t in gold.gold_typed_targets if t.submission_id == sub_id)
            
            act_metrics = compute_stage_a_action_metrics(pred_actions_for_sub, gold_actions_for_sub, orig_sub)
            for k, v in act_metrics.items():
                action_metrics_counters[k].append(v)

            # Calculate new metrics
            gold_is_no_rev = not gold_actions_for_sub or any(a.action_type == "NO_REVISION" for a in gold_actions_for_sub)
            pred_is_no_rev = not pred_actions_for_sub or any(a.get("action_type") == "NO_REVISION" for a in pred_actions_for_sub)
            
            is_false_no_rev = 1.0 if (not gold_is_no_rev and pred_is_no_rev) else 0.0
            action_metrics_counters["false_no_revision"].append(is_false_no_rev)
            
            gold_real_acts = [a for a in gold_actions_for_sub if a.action_type != "NO_REVISION"]
            if len(gold_real_acts) > 1:
                recalled = 0
                for ga in gold_real_acts:
                    ga_target = ga.target_belief_id or ga.target_condition_id
                    for pa in pred_actions_for_sub:
                        pa_target = pa.get("target_belief_id") or pa.get("target_condition_id")
                        if pa_target == ga_target and pa.get("action_type") == ga.action_type:
                            recalled += 1
                            break
                recall_val = recalled / len(gold_real_acts)
                action_metrics_counters["multi_action_recall"].append(recall_val)
                
            has_pe = 1.0 if s_parsed.get("parse_error") is not None else 0.0
            action_metrics_counters["parser_error"].append(has_pe)

            first_pass_valid = s_parsed.get("first_pass_valid_json")
            if first_pass_valid is not None:
                action_metrics_counters["first_pass_valid_json"].append(1.0 if first_pass_valid else 0.0)
                action_metrics_counters["first_pass_parser_error"].append(1.0 if not first_pass_valid else 0.0)
            
            repair_triggered = s_parsed.get("repair_triggered")
            if repair_triggered is not None:
                action_metrics_counters["repair_triggered"].append(1.0 if repair_triggered else 0.0)
                
            repair_success = s_parsed.get("repair_success")
            if repair_success is not None:
                action_metrics_counters["repair_success"].append(1.0 if repair_success else 0.0)

        # Grounding errors Stage A
        has_grounding_err_a = False
        for s_parsed in res_a.get("submissions", []):
            sub_id = s_parsed["submission_id"]
            orig_sub = next(s for s in ep.submissions if s.submission_id == sub_id)
            for act in s_parsed.get("actions", []):
                if check_grounding_error_stage_a(act, orig_sub):
                    has_grounding_err_a = True
                    break

        # Grounding errors Stage B
        has_grounding_err_b = False
        for s_parsed in res_b.get("submissions", []):
            sub_id_b = s_parsed["submission_id"]
            orig_sub = next(s for s in ep.submissions if s.submission_id == sub_id_b)
            valid_belief_ids = {b.belief_id for b in orig_sub.candidate_beliefs} | {
                b.belief_id for b in orig_sub.candidate_replacement_beliefs
            }
            for verd in s_parsed.get("verdicts", []):
                if check_grounding_error_stage_b(verd, valid_belief_ids):
                    has_grounding_err_b = True
                    break

        # Collect Stage B canonicalization stats
        for s_parsed in res_b.get("submissions", []):
            for verd in s_parsed.get("verdicts", []):
                total_stage_b_verdicts += 1
                if verd.get("canonicalization_applied"):
                    canonicalized_stage_b_verdicts += 1
                if verd.get("canonicalization_type") == "fuzzy":
                    fuzzy_stage_b_verdicts += 1

        # Output Validity (Parse Errors)
        has_parse_error_a = any(s.get("parse_error") is not None for s in raw_a.get("submissions", []))
        has_parse_error_b = any(s.get("parse_error") is not None for s in raw_b.get("submissions", []))

        stage_metrics["stage_a"]["total_outputs"] += len(ep.submissions)
        stage_metrics["stage_a"]["valid_outputs"] += sum(1 for s in raw_a.get("submissions", []) if s.get("parse_error") is None)
        
        stage_metrics["stage_b_common"]["total_outputs"] += len(ep.submissions)
        stage_metrics["stage_b_common"]["valid_outputs"] += sum(1 for s in raw_b.get("submissions", []) if s.get("parse_error") is None)

        if has_grounding_err_a:
            stage_metrics["stage_a"]["grounding_errors"] += 1
        if has_grounding_err_b:
            stage_metrics["stage_b_common"]["grounding_errors"] += 1

        # Evaluate final statuses
        ep_correct_a = 0
        ep_correct_b_strict = 0
        ep_correct_b_canonical = 0
        total_beliefs_in_ep = len(gold_statuses)

        for bid, gold_status in gold_statuses.items():
            gold_comp = _STATUS_MAP_A_TO_COMPARABLE.get(gold_status, "UNCERTAIN")
            
            # Stage A Mapping
            actual_a_raw = pred_a_statuses.get(bid, "UNRESOLVED")
            actual_a_comp = _STATUS_MAP_A_TO_COMPARABLE.get(actual_a_raw, "UNCERTAIN")

            # Stage B Mapping (strict & canonicalized)
            actual_b_strict = strict_pred_b_statuses.get(bid, "UNCERTAIN")
            actual_b_canonical = canonical_pred_b_statuses.get(bid, "UNCERTAIN")

            # Final status accuracy counters for Stage A
            stage_metrics["stage_a"]["total_beliefs"] += 1
            if actual_a_comp == gold_comp:
                stage_metrics["stage_a"]["correct_beliefs"] += 1
                ep_correct_a += 1

            # Strict Stage B
            stage_metrics["stage_b_strict"]["total_beliefs"] += 1
            if actual_b_strict == gold_comp:
                stage_metrics["stage_b_strict"]["correct_beliefs"] += 1
                ep_correct_b_strict += 1

            # Canonicalized Stage B
            stage_metrics["stage_b_canonicalized"]["total_beliefs"] += 1
            if actual_b_canonical == gold_comp:
                stage_metrics["stage_b_canonicalized"]["correct_beliefs"] += 1
                ep_correct_b_canonical += 1

            # Over update (Stale propagation)
            if gold_comp == "NOT_USABLE":
                stage_metrics["stage_a"]["not_usable_total"] += 1
                if actual_a_comp == "USABLE":
                    stage_metrics["stage_a"]["over_updates"] += 1

                stage_metrics["stage_b_strict"]["not_usable_total"] += 1
                if actual_b_strict == "USABLE":
                    stage_metrics["stage_b_strict"]["over_updates"] += 1

                stage_metrics["stage_b_canonicalized"]["not_usable_total"] += 1
                if actual_b_canonical == "USABLE":
                    stage_metrics["stage_b_canonicalized"]["over_updates"] += 1

            # Under update
            if gold_comp == "USABLE":
                stage_metrics["stage_a"]["usable_total"] += 1
                if actual_a_comp != "USABLE":
                    stage_metrics["stage_a"]["under_updates"] += 1

                stage_metrics["stage_b_strict"]["usable_total"] += 1
                if actual_b_strict != "USABLE":
                    stage_metrics["stage_b_strict"]["under_updates"] += 1

                stage_metrics["stage_b_canonicalized"]["usable_total"] += 1
                if actual_b_canonical != "USABLE":
                    stage_metrics["stage_b_canonicalized"]["under_updates"] += 1

            # Uncertainty error rate
            if gold_comp == "UNCERTAIN":
                if actual_a_comp != "UNCERTAIN":
                    stage_metrics["stage_a"]["uncertainty_errors"] += 1
                if actual_b_strict != "UNCERTAIN":
                    stage_metrics["stage_b_strict"]["uncertainty_errors"] += 1
                if actual_b_canonical != "UNCERTAIN":
                    stage_metrics["stage_b_canonicalized"]["uncertainty_errors"] += 1
            else:
                if actual_a_comp == "UNCERTAIN":
                    stage_metrics["stage_a"]["uncertainty_errors"] += 1
                if actual_b_strict == "UNCERTAIN":
                    stage_metrics["stage_b_strict"]["uncertainty_errors"] += 1
                if actual_b_canonical == "UNCERTAIN":
                    stage_metrics["stage_b_canonicalized"]["uncertainty_errors"] += 1

        # Accuracy calculations for episode DPA match
        ep_acc = ep_correct_a / total_beliefs_in_ep if total_beliefs_in_ep > 0 else 1.0
        episode_accuracies.append(ep_acc)
        if ep_correct_a == total_beliefs_in_ep:
            episode_em_count += 1

        # Failure breakdown CSV row
        failure_breakdown_rows.append({
            "episode_id": ep_id,
            "failure_type": ep.failure_type_public_or_controlled,
            "domain": ep.domain,
            "total_beliefs": total_beliefs_in_ep,
            "correct_beliefs_a": ep_correct_a,
            "correct_beliefs_b_strict": ep_correct_b_strict,
            "correct_beliefs_b_canonicalized": ep_correct_b_canonical,
            "accuracy_a": ep_correct_a / total_beliefs_in_ep if total_beliefs_in_ep > 0 else 1.0,
            "accuracy_b_strict": ep_correct_b_strict / total_beliefs_in_ep if total_beliefs_in_ep > 0 else 1.0,
            "accuracy_b_canonicalized": ep_correct_b_canonical / total_beliefs_in_ep if total_beliefs_in_ep > 0 else 1.0,
            "has_parse_error_a": has_parse_error_a,
            "has_parse_error_b": has_parse_error_b,
            "has_grounding_error_a": has_grounding_err_a,
            "has_grounding_error_b": has_grounding_err_b,
        })

    # Global aggregate metric calculation
    def calc_rate(num, den):
        return num / den if den > 0 else 0.0

    global_metrics = {
        "stage_a": {
            "final_status_accuracy": calc_rate(stage_metrics["stage_a"]["correct_beliefs"], stage_metrics["stage_a"]["total_beliefs"]),
            "dpa_final_status_accuracy": calc_rate(stage_metrics["stage_a"]["correct_beliefs"], stage_metrics["stage_a"]["total_beliefs"]),
            "macro_final_status_accuracy": sum(episode_accuracies) / len(episode_accuracies) if episode_accuracies else 0.0,
            "episode_exact_match_rate": episode_em_count / len(processed_cases) if processed_cases else 0.0,
            "over_update_rate": calc_rate(stage_metrics["stage_a"]["over_updates"], stage_metrics["stage_a"]["not_usable_total"]),
            "stale_propagation_rate": calc_rate(stage_metrics["stage_a"]["over_updates"], stage_metrics["stage_a"]["not_usable_total"]),
            "under_update_rate": calc_rate(stage_metrics["stage_a"]["under_updates"], stage_metrics["stage_a"]["usable_total"]),
            "uncertainty_error_rate": calc_rate(stage_metrics["stage_a"]["uncertainty_errors"], stage_metrics["stage_a"]["total_beliefs"]),
            "grounding_error_rate": calc_rate(stage_metrics["stage_a"]["grounding_errors"], len(processed_cases)),
            "valid_output_rate": calc_rate(stage_metrics["stage_a"]["valid_outputs"], stage_metrics["stage_a"]["total_outputs"]),
            # Stage A specific metrics
            "valid_json": sum(action_metrics_counters["valid_json"]) / len(action_metrics_counters["valid_json"]) if action_metrics_counters["valid_json"] else 0.0,
            "action_type_match": sum(action_metrics_counters["action_type_match"]) / len(action_metrics_counters["action_type_match"]) if action_metrics_counters["action_type_match"] else 0.0,
            "target_grounding": sum(action_metrics_counters["target_grounding"]) / len(action_metrics_counters["target_grounding"]) if action_metrics_counters["target_grounding"] else 0.0,
            "evidence_grounding": sum(action_metrics_counters["evidence_grounding"]) / len(action_metrics_counters["evidence_grounding"]) if action_metrics_counters["evidence_grounding"] else 0.0,
            "exact_action_match": sum(action_metrics_counters["exact_action_match"]) / len(action_metrics_counters["exact_action_match"]) if action_metrics_counters["exact_action_match"] else 0.0,
            "no_revision_match": sum(action_metrics_counters["no_revision_match"]) / len(action_metrics_counters["no_revision_match"]) if action_metrics_counters["no_revision_match"] else 0.0,
            "false_no_revision_rate": sum(action_metrics_counters["false_no_revision"]) / len(action_metrics_counters["false_no_revision"]) if action_metrics_counters["false_no_revision"] else 0.0,
            "multi_action_recall": sum(action_metrics_counters["multi_action_recall"]) / len(action_metrics_counters["multi_action_recall"]) if action_metrics_counters["multi_action_recall"] else 0.0,
            "parser_error_rate": sum(action_metrics_counters["parser_error"]) / len(action_metrics_counters["parser_error"]) if action_metrics_counters["parser_error"] else 0.0,
            "first_pass_valid_json_rate": sum(action_metrics_counters["first_pass_valid_json"]) / len(action_metrics_counters["first_pass_valid_json"]) if action_metrics_counters["first_pass_valid_json"] else 1.0,
            "first_pass_parser_error_rate": sum(action_metrics_counters["first_pass_parser_error"]) / len(action_metrics_counters["first_pass_parser_error"]) if action_metrics_counters["first_pass_parser_error"] else 0.0,
            "repair_attempt_rate": sum(action_metrics_counters["repair_triggered"]) / len(action_metrics_counters["repair_triggered"]) if action_metrics_counters["repair_triggered"] else 0.0,
            "repair_success_rate": sum(action_metrics_counters["repair_success"]) / len(action_metrics_counters["repair_success"]) if action_metrics_counters["repair_success"] else 0.0,
        },
        "stage_b": {
            "final_status_accuracy": calc_rate(stage_metrics["stage_b_canonicalized"]["correct_beliefs"], stage_metrics["stage_b_canonicalized"]["total_beliefs"]),
            "strict_final_status_accuracy": calc_rate(stage_metrics["stage_b_strict"]["correct_beliefs"], stage_metrics["stage_b_strict"]["total_beliefs"]),
            "canonicalized_final_status_accuracy": calc_rate(stage_metrics["stage_b_canonicalized"]["correct_beliefs"], stage_metrics["stage_b_canonicalized"]["total_beliefs"]),
            "canonicalization_rate": calc_rate(canonicalized_stage_b_verdicts, total_stage_b_verdicts),
            "fuzzy_canonicalization_rate": calc_rate(fuzzy_stage_b_verdicts, total_stage_b_verdicts),
            "over_update_rate": calc_rate(stage_metrics["stage_b_canonicalized"]["over_updates"], stage_metrics["stage_b_canonicalized"]["not_usable_total"]),
            "stale_propagation_rate": calc_rate(stage_metrics["stage_b_canonicalized"]["over_updates"], stage_metrics["stage_b_canonicalized"]["not_usable_total"]),
            "under_update_rate": calc_rate(stage_metrics["stage_b_canonicalized"]["under_updates"], stage_metrics["stage_b_canonicalized"]["usable_total"]),
            "uncertainty_error_rate": calc_rate(stage_metrics["stage_b_canonicalized"]["uncertainty_errors"], stage_metrics["stage_b_canonicalized"]["total_beliefs"]),
            "grounding_error_rate": calc_rate(stage_metrics["stage_b_common"]["grounding_errors"], len(processed_cases)),
            "valid_output_rate": calc_rate(stage_metrics["stage_b_common"]["valid_outputs"], stage_metrics["stage_b_common"]["total_outputs"]),
        }
    }
    return global_metrics, failure_breakdown_rows

