from typing import Any, Dict, List
from benchmark.retrace_bench.schemas import Scenario, Prediction, EvaluationResult
from benchmark.retrace_bench.taxonomy import ProbeType, FinalStatus, RevisionFamily


def calculate_metrics(
    scenarios: List[Scenario],
    predictions: List[Prediction]
) -> Dict[str, Any]:
    pred_map = {(p.scenario_id, p.query_id): p for p in predictions}
    scen_map = {s.scenario_id: s for s in scenarios}

    total_queries = 0
    correct_queries = 0

    probe_stats = {
        ProbeType.STATE_RESOLUTION: {"total": 0, "correct": 0},
        ProbeType.PREMISE_RESISTANCE: {"total": 0, "correct": 0},
        ProbeType.POLICY_ADAPTATION: {"total": 0, "correct": 0},
        ProbeType.AUDIT_LOCALIZATION: {"total": 0, "correct": 0}
    }

    # Tracking for custom rates
    stale_propagation_total = 0
    stale_propagation_success = 0

    over_update_total = 0
    over_update_count = 0

    under_update_total = 0
    under_update_count = 0

    status_match_total = 0
    status_match_correct = 0

    for scen in scenarios:
        primary_belief_id = None
        for entry in scen.memory_snapshot:
            if entry.entry_type == "belief":
                primary_belief_id = entry.entry_id
                break

        for q in scen.probe_queries:
            pred = pred_map.get((scen.scenario_id, q.query_id))
            if not pred:
                continue

            total_queries += 1
            is_correct = (pred.predicted_answer.strip().upper() == q.gold_answer.strip().upper())
            
            probe_stats[q.probe_type]["total"] += 1
            if is_correct:
                probe_stats[q.probe_type]["correct"] += 1
                correct_queries += 1

            # Check predicted statuses if available
            if pred.predicted_final_statuses and primary_belief_id:
                pred_status = pred.predicted_final_statuses.get(primary_belief_id)
                gold_status = scen.gold_final_statuses.get(primary_belief_id)

                if pred_status:
                    status_match_total += 1
                    if pred_status == gold_status:
                        status_match_correct += 1

                    # stale propagation: correctly identify BLOCKED when family is blocks
                    if scen.revision_family == RevisionFamily.BLOCKS:
                        stale_propagation_total += 1
                        if pred_status == FinalStatus.BLOCKED:
                            stale_propagation_success += 1

                    # over update: family is no_revision or reaffirms but predicted status is superseded/blocked/unresolved
                    if scen.revision_family in (RevisionFamily.NO_REVISION, RevisionFamily.REAFFIRMS):
                        over_update_total += 1
                        if pred_status in (FinalStatus.SUPERSEDED, FinalStatus.BLOCKED, FinalStatus.UNRESOLVED):
                            over_update_count += 1

                    # under update: family is supersedes/blocks but predicted status remains AUTHORIZED
                    if scen.revision_family in (RevisionFamily.SUPERSEDES, RevisionFamily.BLOCKS):
                        under_update_total += 1
                        if pred_status == FinalStatus.AUTHORIZED:
                            under_update_count += 1

    # Safe division helpers
    def pct(num, den):
        return (num / den) if den > 0 else 0.0

    return {
        "overall_accuracy": pct(correct_queries, total_queries),
        "state_resolution_accuracy": pct(probe_stats[ProbeType.STATE_RESOLUTION]["correct"], probe_stats[ProbeType.STATE_RESOLUTION]["total"]),
        "premise_resistance_accuracy": pct(probe_stats[ProbeType.PREMISE_RESISTANCE]["correct"], probe_stats[ProbeType.PREMISE_RESISTANCE]["total"]),
        "policy_adaptation_accuracy": pct(probe_stats[ProbeType.POLICY_ADAPTATION]["correct"], probe_stats[ProbeType.POLICY_ADAPTATION]["total"]),
        "audit_localization_score": pct(probe_stats[ProbeType.AUDIT_LOCALIZATION]["correct"], probe_stats[ProbeType.AUDIT_LOCALIZATION]["total"]),
        "final_status_accuracy": pct(status_match_correct, status_match_total),
        "stale_propagation_rate": pct(stale_propagation_success, stale_propagation_total),
        "over_update_rate": pct(over_update_count, over_update_total),
        "under_update_rate": pct(under_update_count, under_update_total),
        "total_evaluated_queries": total_queries
    }
