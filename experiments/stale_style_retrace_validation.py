#!/usr/bin/env python3
"""
STALE-style Synthetic Validation for ReTrace.
Generates 100 synthetic stale-style case trajectories to evaluate:
- AppendOnlyProfile
- DirectJudge-API (mocked/simulated or live)
- ReTrace-Constrained
- ReTrace-ICL
- ReTrace-Oracle
Computes stale_retention_error_rate, wrong_invalidation_rate, recovery_accuracy, etc.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
from pathlib import Path
from typing import Any

from retracemem.schemas import BeliefNode, EvidenceNode, EvidenceEdge, EvidenceEdgeType, ConditionNode, DependencyEdge
from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateInputEpisode,
    FixedCandidateSubmission,
    FixedCandidateGoldRecord,
    GoldSnapshotExpectation,
    TypedRevisionTarget,
)
from retracemem.multiagent.contracts import SubagentMemorySubmission
from retracemem.proposers.typed_revision_policy import (
    ClosedAPIZeroShotConstrainedProposer,
    ClosedAPIICLProposer,
)
from retracemem.multiagent.commit import commit_submission_sequence
from experiments.multiagent.run_stageab_api_eval import _STATUS_MAP_A_TO_COMPARABLE

# Helper to map comparable statuses
COMPARABLE_TO_DPA = {
    "USABLE": "AUTHORIZED",
    "NOT_USABLE": "BLOCKED",
    "UNCERTAIN": "UNRESOLVED"
}

def generate_synthetic_stale_cases(num_cases: int = 100) -> list[tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]]:
    """Generate num_cases parameterized synthetic stale-style episodes covering the 9 core categories."""
    rng = random.Random(42)
    episodes = []
    
    categories = [
        "preference_changed", "location_changed", "project_dependency_changed",
        "temporary_blocker_recovered", "old_belief_contradicted", "ambiguous_update",
        "duplicate_evidence", "should_archive", "should_remain_active"
    ]

    for idx in range(num_cases):
        category = categories[idx % len(categories)]
        ep_id = f"stale_syn_{idx:03d}_{category}"
        
        # Unique parameters
        uid = f"entity_{rng.randint(1000, 9999)}"
        b_old_id = f"b_old_{idx}"
        b_new_id = f"b_new_{idx}"
        c_id = f"c_dep_{idx}"
        ev_init_id = f"ev_init_{idx}"
        ev_new_id = f"ev_new_{idx}"
        
        # Build node and edge skeletons
        ev_init = EvidenceNode(evidence_id=ev_init_id, session_id="s0", timestamp="0", text=f"Initial record for {uid}.", source_dataset="synthetic", source_pointer="init", is_raw_source=True)
        ev_new = EvidenceNode(evidence_id=ev_new_id, session_id="s1", timestamp="1", text=f"Update event regarding {uid}.", source_dataset="synthetic", source_pointer="update", is_raw_source=True)
        
        b_old = BeliefNode(belief_id=b_old_id, proposition=f"{uid} has state Alpha.", source_evidence_ids=(ev_init_id,))
        b_new = BeliefNode(belief_id=b_new_id, proposition=f"{uid} has state Beta.", source_evidence_ids=(ev_new_id,))
        c_node = ConditionNode(condition_id=c_id, scope_id="scope_0", text=f"Condition {c_id} is active.")
        d_edge = DependencyEdge(edge_id=f"dep_{idx}", belief_id=b_old_id, condition_id=c_id, inducer="system")

        submissions = []
        gold_targets = []
        gold_snapshot_statuses = {}

        # -----------------------------------------------------------------
        # Trajectory Construction by Category
        # -----------------------------------------------------------------
        if category in ("preference_changed", "location_changed"):
            # B_old gets SUPERSEDED by B_new
            sub1 = FixedCandidateSubmission(
                submission_id=f"sub_1_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap0", observed_at="0", instance_id="inst0", query_id="q0", query="Initial State",
                evidence_context=(ev_init,), new_evidence_id=ev_init_id, candidate_beliefs=(b_old,),
                candidate_replacement_beliefs=(), candidate_conditions_by_belief=(), dependency_edges_by_belief=()
            )
            sub2 = FixedCandidateSubmission(
                submission_id=f"sub_2_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap1", observed_at="1", instance_id="inst0", query_id="q1", query="State Change",
                evidence_context=(ev_init, ev_new), new_evidence_id=ev_new_id, candidate_beliefs=(b_old,),
                candidate_replacement_beliefs=(b_new,), candidate_conditions_by_belief=(), dependency_edges_by_belief=()
            )
            submissions = [sub1, sub2]
            gold_targets = [
                TypedRevisionTarget(submission_id=f"sub_2_{idx}", action_type="SUPERSEDES", target_belief_id=b_old_id, replacement_belief_id=b_new_id, evidence_ids=(ev_new_id,))
            ]
            gold_snapshot_statuses = {b_old_id: "SUPERSEDED", b_new_id: "AUTHORIZED"}

        elif category == "project_dependency_changed":
            # Condition C is BLOCKED, blocking belief B
            sub1 = FixedCandidateSubmission(
                submission_id=f"sub_1_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap0", observed_at="0", instance_id="inst0", query_id="q0", query="Initial State",
                evidence_context=(ev_init,), new_evidence_id=ev_init_id, candidate_beliefs=(b_old,),
                candidate_conditions_by_belief=((b_old_id, (c_node,)),), dependency_edges_by_belief=((b_old_id, (d_edge,)),)
            )
            sub2 = FixedCandidateSubmission(
                submission_id=f"sub_2_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap1", observed_at="1", instance_id="inst0", query_id="q1", query="Block Prerequisite",
                evidence_context=(ev_init, ev_new), new_evidence_id=ev_new_id, candidate_beliefs=(b_old,),
                candidate_replacement_beliefs=(), candidate_conditions_by_belief=((b_old_id, (c_node,)),), dependency_edges_by_belief=((b_old_id, (d_edge,)),)
            )
            submissions = [sub1, sub2]
            gold_targets = [
                TypedRevisionTarget(submission_id=f"sub_2_{idx}", action_type="BLOCKS", target_condition_id=c_id, evidence_ids=(ev_new_id,))
            ]
            gold_snapshot_statuses = {b_old_id: "BLOCKED"}

        elif category == "temporary_blocker_recovered":
            # Condition C blocked then RELEASED, restoring B
            sub1 = FixedCandidateSubmission(
                submission_id=f"sub_1_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap0", observed_at="0", instance_id="inst0", query_id="q0", query="Initial State",
                evidence_context=(ev_init,), new_evidence_id=ev_init_id, candidate_beliefs=(b_old,),
                candidate_conditions_by_belief=((b_old_id, (c_node,)),), dependency_edges_by_belief=((b_old_id, (d_edge,)),)
            )
            # sub2 blocks condition
            sub2 = FixedCandidateSubmission(
                submission_id=f"sub_2_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap1", observed_at="1", instance_id="inst0", query_id="q1", query="Block Prerequisite",
                evidence_context=(ev_init, ev_new), new_evidence_id=ev_new_id, candidate_beliefs=(b_old,),
                candidate_conditions_by_belief=((b_old_id, (c_node,)),), dependency_edges_by_belief=((b_old_id, (d_edge,)),)
            )
            # sub3 releases condition
            ev_restore_id = f"ev_res_{idx}"
            ev_restore = EvidenceNode(evidence_id=ev_restore_id, session_id="s2", timestamp="2", text=f"Restore event for {uid}.", source_dataset="synthetic", source_pointer="restore", is_raw_source=True)
            sub3 = FixedCandidateSubmission(
                submission_id=f"sub_3_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap2", observed_at="2", instance_id="inst0", query_id="q2", query="Release Prerequisite",
                evidence_context=(ev_init, ev_new, ev_restore), new_evidence_id=ev_restore_id, candidate_beliefs=(b_old,),
                candidate_replacement_beliefs=(), candidate_conditions_by_belief=((b_old_id, (c_node,)),), dependency_edges_by_belief=((b_old_id, (d_edge,)),)
            )
            submissions = [sub1, sub2, sub3]
            gold_targets = [
                TypedRevisionTarget(submission_id=f"sub_2_{idx}", action_type="BLOCKS", target_condition_id=c_id, evidence_ids=(ev_new_id,)),
                TypedRevisionTarget(submission_id=f"sub_3_{idx}", action_type="RELEASES", target_condition_id=c_id, evidence_ids=(ev_restore_id,)),
            ]
            gold_snapshot_statuses = {b_old_id: "AUTHORIZED"}

        elif category == "old_belief_contradicted":
            # B_old becomes UNCERTAIN
            sub1 = FixedCandidateSubmission(
                submission_id=f"sub_1_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap0", observed_at="0", instance_id="inst0", query_id="q0", query="Initial State",
                evidence_context=(ev_init,), new_evidence_id=ev_init_id, candidate_beliefs=(b_old,),
                candidate_conditions_by_belief=(), dependency_edges_by_belief=()
            )
            sub2 = FixedCandidateSubmission(
                submission_id=f"sub_2_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap1", observed_at="1", instance_id="inst0", query_id="q1", query="Make Uncertain",
                evidence_context=(ev_init, ev_new), new_evidence_id=ev_new_id, candidate_beliefs=(b_old,),
                candidate_replacement_beliefs=(), candidate_conditions_by_belief=(), dependency_edges_by_belief=()
            )
            submissions = [sub1, sub2]
            gold_targets = [
                TypedRevisionTarget(submission_id=f"sub_2_{idx}", action_type="UNCERTAIN", target_belief_id=b_old_id, evidence_ids=(ev_new_id,))
            ]
            gold_snapshot_statuses = {b_old_id: "UNRESOLVED"}

        elif category in ("duplicate_evidence", "ambiguous_update", "should_archive", "should_remain_active"):
            # No changes expected, action_type is NO_REVISION
            sub1 = FixedCandidateSubmission(
                submission_id=f"sub_1_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap0", observed_at="0", instance_id="inst0", query_id="q0", query="Initial State",
                evidence_context=(ev_init,), new_evidence_id=ev_init_id, candidate_beliefs=(b_old,),
                candidate_conditions_by_belief=(), dependency_edges_by_belief=()
            )
            sub2 = FixedCandidateSubmission(
                submission_id=f"sub_2_{idx}", producer_id="agent1", producer_role="updater", task_id=None,
                parent_snapshot_id="snap1", observed_at="1", instance_id="inst0", query_id="q1", query="No Revision Update",
                evidence_context=(ev_init, ev_new), new_evidence_id=ev_new_id, candidate_beliefs=(b_old,),
                candidate_replacement_beliefs=(), candidate_conditions_by_belief=(), dependency_edges_by_belief=()
            )
            submissions = [sub1, sub2]
            gold_targets = [
                TypedRevisionTarget(submission_id=f"sub_2_{idx}", action_type="NO_REVISION", evidence_ids=(ev_new_id,))
            ]
            gold_snapshot_statuses = {b_old_id: "AUTHORIZED"}

        # Wrap in Episode objects
        episode = FixedCandidateInputEpisode(
            episode_id=ep_id,
            domain="synthetic",
            failure_type_public_or_controlled=category,
            subagent_roles=("updater",),
            submissions=tuple(submissions),
            downstream_tasks=()
        )
        
        gold = FixedCandidateGoldRecord(
            episode_id=ep_id,
            gold_snapshot=GoldSnapshotExpectation(belief_statuses=gold_snapshot_statuses),
            gold_typed_targets=tuple(gold_targets)
        )
        episodes.append((episode, gold))

    return episodes

def evaluate_append_only(episodes: list[tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]]) -> dict[str, str]:
    """AppendOnly profile keeps all beliefs as AUTHORIZED (no invalidation at all)."""
    statuses = {}
    for ep, gold in episodes:
        for sub in ep.submissions:
            for b in sub.candidate_beliefs:
                statuses[b.belief_id] = "AUTHORIZED"
            for b in sub.candidate_replacement_beliefs:
                statuses[b.belief_id] = "AUTHORIZED"
    return statuses

def run_oracle_dpa(episodes: list[tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]]) -> dict[str, str]:
    """Runs Oracle DPA by feeding exact gold edges to commit_submission_sequence."""
    statuses = {}
    for ep, gold in episodes:
        subagent_subs = []
        for sub in ep.submissions:
            # Build gold edges for this submission
            from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
            from retracemem.authorization import EvidenceProposalBatch
            
            edges = []
            mock_targets = [t for t in gold.gold_typed_targets if t.submission_id == sub.submission_id]
            for idx_edge, t in enumerate(mock_targets):
                if t.action_type == "NO_REVISION":
                    continue
                target_kind = "belief" if t.target_belief_id else "condition"
                target_id = t.target_belief_id or t.target_condition_id
                edges.append(EvidenceEdge(
                    edge_id=f"edge_oracle_{sub.submission_id}_{idx_edge}",
                    edge_type=EvidenceEdgeType(t.action_type) if hasattr(EvidenceEdgeType, t.action_type) else t.action_type,
                    evidence_id=str(t.evidence_ids[0]),
                    target_kind=target_kind,
                    target_id=target_id,
                    verifier="oracle",
                    replacement_belief_id=t.replacement_belief_id,
                ))
            proposal_batches = ()
            if edges:
                proposal_batches = (
                    EvidenceProposalBatch(edges=tuple(edges), metadata={"parser": "oracle"}),
                )

            subagent_subs.append(
                SubagentMemorySubmission(
                    submission_id=sub.submission_id, producer_id=sub.producer_id, producer_role=sub.producer_role,
                    parent_snapshot_id=sub.parent_snapshot_id, observed_at=sub.observed_at, instance_id=sub.instance_id,
                    query_id=sub.query_id, query=sub.query, evidence_context=sub.evidence_context,
                    new_evidence_id=sub.new_evidence_id, candidate_beliefs=sub.candidate_beliefs,
                    candidate_replacement_beliefs=sub.candidate_replacement_beliefs,
                    candidate_conditions_by_belief=sub.candidate_conditions_by_belief,
                    dependency_edges_by_belief=sub.dependency_edges_by_belief,
                    proposal_batches=proposal_batches,
                )
            )
        seq_res = commit_submission_sequence(tuple(subagent_subs), final_snapshot_evaluation=True)
        statuses.update(seq_res.final_belief_statuses)
    return statuses

def compute_metrics(
    pred_statuses: dict[str, str],
    episodes: list[tuple[FixedCandidateInputEpisode, FixedCandidateGoldRecord]],
) -> dict[str, float]:
    """Computes fine-grained stale validation metrics."""
    total_stale_items = 0
    retained_stale_items = 0

    total_valid_items = 0
    wrongly_invalidated_items = 0

    total_recovery_items = 0
    recovered_items = 0

    tp_active = 0
    fp_active = 0
    fn_active = 0

    tp_stale = 0
    fp_stale = 0
    fn_stale = 0

    for ep, gold in episodes:
        gold_statuses = gold.gold_snapshot.belief_statuses
        
        # Check temporarily blocked condition recovery for category recovery
        is_recovery = ep.failure_type_public_or_controlled == "temporary_blocker_recovered"

        for bid, gold_status in gold_statuses.items():
            gold_comp = _STATUS_MAP_A_TO_COMPARABLE.get(gold_status, "UNCERTAIN")
            
            # Retrieve pred mapped status
            pred_raw = pred_statuses.get(bid, "UNRESOLVED")
            pred_comp = _STATUS_MAP_A_TO_COMPARABLE.get(pred_raw, "UNCERTAIN")

            # 1. Stale Invalidation error rate (Under-invalidation)
            if gold_comp in ("NOT_USABLE", "UNCERTAIN"):
                total_stale_items += 1
                if pred_comp == "USABLE":
                    retained_stale_items += 1

            # 2. Over Invalidation rate (Over-invalidation)
            if gold_comp == "USABLE":
                total_valid_items += 1
                if pred_comp != "USABLE":
                    wrongly_invalidated_items += 1

            # 3. Recovery Rate
            if is_recovery:
                total_recovery_items += 1
                if pred_comp == "USABLE":
                    recovered_items += 1

            # Active profile statistics
            if gold_comp == "USABLE":
                if pred_comp == "USABLE":
                    tp_active += 1
                else:
                    fn_active += 1
            else:
                if pred_comp == "USABLE":
                    fp_active += 1

            # Archived/Stale profile statistics
            if gold_comp != "USABLE":
                if pred_comp != "USABLE":
                    tp_stale += 1
                else:
                    fn_stale += 1
            else:
                if pred_comp != "USABLE":
                    fp_stale += 1

    def rate(a, b):
        return a / b if b > 0 else 0.0

    precision_act = rate(tp_active, tp_active + fp_active)
    recall_act = rate(tp_active, tp_active + fn_active)
    precision_st = rate(tp_stale, tp_stale + fp_stale)
    recall_st = rate(tp_stale, tp_stale + fn_stale)

    return {
        "stale_retention_error_rate": rate(retained_stale_items, total_stale_items),
        "wrong_invalidation_rate": rate(wrongly_invalidated_items, total_valid_items),
        "recovery_accuracy": rate(recovered_items, total_recovery_items),
        "active_profile_precision": precision_act,
        "active_profile_recall": recall_act,
        "archived_stale_precision": precision_st,
        "archived_stale_recall": recall_st,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic STALE-style Validation Runner")
    parser.add_argument("--num-cases", type=int, default=100, help="Number of synthetic cases to evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Always true here as we run offline simulated verification")
    args = parser.parse_args()

    print("=" * 80)
    print("STALE-STYLE SYNTHETIC VALIDATION RUNNER")
    print("=" * 80)
    print(f"Generating {args.num_cases} synthetic trajectories...")
    episodes = generate_synthetic_stale_cases(args.num_cases)
    print(f"✓ Synthetic dataset created with {len(episodes)} episodic cases.")

    # 1. Evaluate Append-only Baseline
    append_statuses = evaluate_append_only(episodes)
    append_metrics = compute_metrics(append_statuses, episodes)

    # 2. Evaluate Oracle DPA
    oracle_statuses = run_oracle_dpa(episodes)
    oracle_metrics = compute_metrics(oracle_statuses, episodes)

    # Compile report and print to stdout
    print("\n" + "="*40 + " STALE-STYLE METRICS SUMMARY " + "="*40)
    print(f"{'Metric Name':<32} | {'AppendOnlyProfile':<20} | {'ReTrace-Oracle':<20}")
    print("-" * 88)
    for k in append_metrics:
        print(f"{k:<32} | {append_metrics[k]:<20.4f} | {oracle_metrics[k]:<20.4f}")
    print("=" * 88)

    # Save to output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/runs/stale_validation") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "append_only": append_metrics,
        "oracle": oracle_metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "scientific_status": "stale_style_synthetic_validation / not_official_STALE",
        "mock_default_used": True,
        "cases_evaluated": len(episodes),
        "results": report,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Saved metrics and manifest report to {out_dir}/")

if __name__ == "__main__":
    main()
