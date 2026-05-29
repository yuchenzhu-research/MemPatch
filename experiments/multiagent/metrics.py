from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
from experiments.multiagent.contracts import (
    MultiAgentMemoryEpisode,
    EpisodeMethodResult,
    FixedCandidateGoldRecord,
    FixedCandidateEpisodeMethodResult,
    DownstreamTask,
)


def compute_episode_metrics(
    episode: MultiAgentMemoryEpisode,
    result: EpisodeMethodResult,
) -> Dict[str, float]:
    """Calculate granular metrics for a single episode method run."""
    gold = episode.gold_snapshot.belief_statuses
    actual = result.final_belief_statuses

    # 1. Authorization Accuracy (overall accuracy of status matches)
    correct_auth = 0
    total_auth = len(gold)
    for bid, exp_status in gold.items():
        if actual.get(bid) == exp_status:
            correct_auth += 1
    auth_accuracy = correct_auth / total_auth if total_auth > 0 else 1.0

    # 2. Stale Propagation Error Rate
    stale_errors = 0
    total_stale_checks = 0
    for bid, exp_status in gold.items():
        if exp_status in ("SUPERSEDED", "BLOCKED"):
            total_stale_checks += 1
            if actual.get(bid) == "AUTHORIZED":
                stale_errors += 1
    stale_propagation_error = stale_errors / total_stale_checks if total_stale_checks > 0 else 0.0

    # 3. Scope Expansion Error Rate
    scope_errors = 0
    total_scope_checks = 0
    for bid, exp_status in gold.items():
        if exp_status == "AUTHORIZED":
            total_scope_checks += 1
            if actual.get(bid) in ("SUPERSEDED", "BLOCKED"):
                scope_errors += 1
    scope_expansion_error = scope_errors / total_scope_checks if total_scope_checks > 0 else 0.0

    # 4. Conflict Resolution Accuracy
    conflict_correct = 0
    total_conflict_checks = 0
    if episode.failure_type in ("cross_agent_conflict", "direct_supersession"):
        for bid, exp_status in gold.items():
            total_conflict_checks += 1
            if actual.get(bid) == exp_status:
                conflict_correct += 1
    conflict_accuracy = conflict_correct / total_conflict_checks if total_conflict_checks > 0 else 1.0

    # 5. Recovery Accuracy
    recovery_correct = 0
    total_recovery_checks = 0
    if episode.failure_type == "temporary_blocker_recovery":
        for bid, exp_status in gold.items():
            if exp_status == "AUTHORIZED":
                total_recovery_checks += 1
                if actual.get(bid) == "AUTHORIZED":
                    recovery_correct += 1
    recovery_accuracy = recovery_correct / total_recovery_checks if total_recovery_checks > 0 else 1.0

    # 6. Uncertainty Handling Accuracy
    uncertain_correct = 0
    total_uncertain_checks = 0
    if episode.failure_type == "ambiguous_update":
        for bid, exp_status in gold.items():
            if exp_status == "UNRESOLVED":
                total_uncertain_checks += 1
                if actual.get(bid) == "UNRESOLVED":
                    uncertain_correct += 1
    uncertainty_accuracy = uncertain_correct / total_uncertain_checks if total_uncertain_checks > 0 else 1.0

    # 7. Protected Belief Preservation
    protected_correct = 0
    total_protected = 0
    for task in episode.downstream_tasks:
        for bid in task.protected_belief_ids:
            total_protected += 1
            # If the protected belief remains AUTHORIZED, it is preserved
            if actual.get(bid) == "AUTHORIZED":
                protected_correct += 1
    protected_preservation = protected_correct / total_protected if total_protected > 0 else 1.0

    # 8. Trace Availability Rate
    trace_events = sum(1 for ev in result.revision_events if ev.trace_available)
    total_events = len(result.revision_events)
    trace_rate = trace_events / total_events if total_events > 0 else 0.0

    return {
        "authorization_accuracy": auth_accuracy,
        "stale_propagation_error_rate": stale_propagation_error,
        "scope_expansion_error_rate": scope_expansion_error,
        "conflict_resolution_accuracy": conflict_accuracy,
        "recovery_accuracy": recovery_accuracy,
        "uncertainty_handling_accuracy": uncertainty_accuracy,
        "protected_belief_preservation": protected_preservation,
        "trace_availability_rate": trace_rate,
    }


def aggregate_metrics(
    results: List[Tuple[MultiAgentMemoryEpisode, EpisodeMethodResult]]
) -> Dict[str, Any]:
    """Aggregate metrics by method, domain, and failure_type."""
    # Nesting structure: grp_key -> metric_name -> list of values
    aggregated = defaultdict(lambda: defaultdict(list))

    for ep, res in results:
        ep_metrics = compute_episode_metrics(ep, res)
        num_subagents = len(ep.subagent_roles)
        conflict_density = ep.stress_factors.get("conflict_density", 0.0)
        delay_depth = ep.stress_factors.get("delay_depth", 0)

        keys_to_log = {
            "overall": "all",
            "method_name": res.method_name,
            "domain": ep.domain,
            "failure_type": ep.failure_type,
            "number_of_subagents": num_subagents,
            "conflict_density": conflict_density,
            "delay_depth": delay_depth,
        }

        # For each grouping category, accumulate metric values
        for category, key in keys_to_log.items():
            for m_name, val in ep_metrics.items():
                # We group everything by method name first, so we compare methods under different slices
                grp_key = f"{res.method_name}__{category}__{key}"
                aggregated[grp_key][m_name].append(val)

    # Calculate averages
    summary = {}
    for grp_key, metrics in aggregated.items():
        summary[grp_key] = {}
        for m_name, vals in metrics.items():
            summary[grp_key][m_name] = sum(vals) / len(vals) if vals else 0.0

    return summary


# ===========================================================================
# Packet 4B: Fixed-Candidate Metrics
# ===========================================================================


def compute_fixed_candidate_metrics(
    gold_record: FixedCandidateGoldRecord,
    downstream_tasks: Tuple[DownstreamTask, ...],
    result: FixedCandidateEpisodeMethodResult,
) -> Dict[str, float]:
    """Calculate granular metrics for a single fixed-candidate episode method run."""
    gold = gold_record.gold_snapshot.belief_statuses
    actual = result.final_belief_statuses

    # 1. Authorization Accuracy
    correct_auth = 0
    total_auth = len(gold)
    for bid, exp_status in gold.items():
        if actual.get(bid) == exp_status:
            correct_auth += 1
    auth_accuracy = correct_auth / total_auth if total_auth > 0 else 1.0

    # 2. Stale Propagation Error Rate
    stale_errors = 0
    total_stale_checks = 0
    for bid, exp_status in gold.items():
        if exp_status in ("SUPERSEDED", "BLOCKED"):
            total_stale_checks += 1
            if actual.get(bid) == "AUTHORIZED":
                stale_errors += 1
    stale_propagation_error = stale_errors / total_stale_checks if total_stale_checks > 0 else 0.0

    # 3. Scope Expansion Error Rate
    scope_errors = 0
    total_scope_checks = 0
    for bid, exp_status in gold.items():
        if exp_status == "AUTHORIZED":
            total_scope_checks += 1
            if actual.get(bid) in ("SUPERSEDED", "BLOCKED"):
                scope_errors += 1
    scope_expansion_error = scope_errors / total_scope_checks if total_scope_checks > 0 else 0.0

    # 4. Conflict Resolution Accuracy
    conflict_correct = 0
    total_conflict_checks = 0
    if gold_record.failure_type in ("cross_agent_conflict", "direct_supersession"):
        for bid, exp_status in gold.items():
            total_conflict_checks += 1
            if actual.get(bid) == exp_status:
                conflict_correct += 1
    conflict_accuracy = conflict_correct / total_conflict_checks if total_conflict_checks > 0 else 1.0

    # 5. Recovery Accuracy
    recovery_correct = 0
    total_recovery_checks = 0
    if gold_record.failure_type == "temporary_blocker_recovery":
        for bid, exp_status in gold.items():
            if exp_status == "AUTHORIZED":
                total_recovery_checks += 1
                if actual.get(bid) == "AUTHORIZED":
                    recovery_correct += 1
    recovery_accuracy = recovery_correct / total_recovery_checks if total_recovery_checks > 0 else 1.0

    # 6. Uncertainty Handling Accuracy
    uncertain_correct = 0
    total_uncertain_checks = 0
    if gold_record.failure_type == "ambiguous_update":
        for bid, exp_status in gold.items():
            if exp_status == "UNRESOLVED":
                total_uncertain_checks += 1
                if actual.get(bid) == "UNRESOLVED":
                    uncertain_correct += 1
    uncertainty_accuracy = uncertain_correct / total_uncertain_checks if total_uncertain_checks > 0 else 1.0

    # 7. Protected Belief Preservation
    protected_correct = 0
    total_protected = 0
    for task in downstream_tasks:
        for bid in task.protected_belief_ids:
            total_protected += 1
            if actual.get(bid) == "AUTHORIZED":
                protected_correct += 1
    protected_preservation = protected_correct / total_protected if total_protected > 0 else 1.0

    # 8. Decision Count (how many explicit decisions were recorded)
    decision_count = float(len(result.decisions))

    return {
        "authorization_accuracy": auth_accuracy,
        "stale_propagation_error_rate": stale_propagation_error,
        "scope_expansion_error_rate": scope_expansion_error,
        "conflict_resolution_accuracy": conflict_accuracy,
        "recovery_accuracy": recovery_accuracy,
        "uncertainty_handling_accuracy": uncertainty_accuracy,
        "protected_belief_preservation": protected_preservation,
        "decision_count": decision_count,
    }


def aggregate_fixed_candidate_metrics(
    results: List[Tuple[FixedCandidateGoldRecord, FixedCandidateInputEpisode, FixedCandidateEpisodeMethodResult]]
) -> Dict[str, Any]:
    """Aggregate metrics for fixed-candidate evaluation runs."""
    aggregated = defaultdict(lambda: defaultdict(list))

    for gold, ep, res in results:
        ep_metrics = compute_fixed_candidate_metrics(gold, ep.downstream_tasks, res)
        conflict_density = ep.stress_factors.get("conflict_density", 0.0)
        delay_depth = ep.stress_factors.get("delay_depth", 0)

        keys_to_log = {
            "overall": "all",
            "method_name": res.method_name,
            "domain": ep.domain,
            "failure_type": gold.failure_type,
            "protocol_mode": res.protocol_mode,
            "conflict_density": conflict_density,
            "delay_depth": delay_depth,
        }

        for category, key in keys_to_log.items():
            for m_name, val in ep_metrics.items():
                grp_key = f"{res.method_name}__{category}__{key}"
                aggregated[grp_key][m_name].append(val)

    summary = {}
    for grp_key, metrics in aggregated.items():
        summary[grp_key] = {}
        for m_name, vals in metrics.items():
            summary[grp_key][m_name] = sum(vals) / len(vals) if vals else 0.0

    return summary

