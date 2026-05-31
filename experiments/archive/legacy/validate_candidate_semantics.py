from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from retracemem.authorization import EvidenceProposalBatch
from retracemem.multiagent.contracts import SubagentMemorySubmission
from retracemem.multiagent.commit import commit_submission_sequence
from retracemem.evaluation.multiagent.data.dev_expansion import generate_expanded_episodes

def validate_episode(episode: Any, gold_record: Any) -> Tuple[bool, Dict[str, Any]]:
    # Map gold targets by submission
    targets_by_sub: Dict[str, List[Any]] = {}
    for t in gold_record.gold_typed_targets:
        targets_by_sub.setdefault(t.submission_id, []).append(t)
        
    subagent_subs = []
    for sub in episode.submissions:
        sub_targets = targets_by_sub.get(sub.submission_id, [])
        edges = []
        for idx, t in enumerate(sub_targets):
            if t.action_type == "NO_REVISION":
                continue
            target_kind = "belief" if t.target_belief_id else "condition"
            target_id = t.target_belief_id or t.target_condition_id
            
            # Construct EvidenceEdge
            edge_id = f"edge_gold_val_{episode.episode_id}_{sub.submission_id}_{idx}"
            edge = EvidenceEdge(
                edge_id=edge_id,
                edge_type=EvidenceEdgeType(t.action_type) if hasattr(EvidenceEdgeType, t.action_type) else t.action_type,
                evidence_id=str(t.evidence_ids[0]),
                target_kind=target_kind,
                target_id=target_id,
                verifier="gold_consistency_validator",
                replacement_belief_id=t.replacement_belief_id,
                rationale=t.rationale,
            )
            edges.append(edge)
            
        if edges:
            batches = (
                EvidenceProposalBatch(
                    edges=tuple(edges),
                    metadata={"validator": "gold_consistency"},
                ),
            )
        else:
            batches = ()
            
        subagent_sub = SubagentMemorySubmission(
            submission_id=sub.submission_id,
            producer_id=sub.producer_id,
            producer_role=sub.producer_role,
            parent_snapshot_id=sub.parent_snapshot_id,
            observed_at=sub.observed_at,
            instance_id=sub.instance_id,
            query_id=sub.query_id,
            query=sub.query,
            evidence_context=sub.evidence_context,
            new_evidence_id=sub.new_evidence_id,
            candidate_beliefs=sub.candidate_beliefs,
            candidate_replacement_beliefs=sub.candidate_replacement_beliefs,
            candidate_conditions_by_belief=sub.candidate_conditions_by_belief,
            dependency_edges_by_belief=sub.dependency_edges_by_belief,
            proposal_batches=batches,
            task_id=sub.task_id,
            metadata=sub.metadata,
        )
        subagent_subs.append(subagent_sub)
        
    # Replay sequence using canonical commit
    seq_res = commit_submission_sequence(tuple(subagent_subs), final_snapshot_evaluation=True)
    active_statuses = seq_res.final_belief_statuses
    
    expected_statuses = gold_record.gold_snapshot.belief_statuses
    
    # Check for mismatches
    mismatches: Dict[str, Dict[str, str]] = {}
    
    # Check all expected keys
    for bid, expected in expected_statuses.items():
        actual = active_statuses.get(bid)
        if actual != expected:
            mismatches[bid] = {"expected": expected, "actual": str(actual)}
            
    # Check if active has unexpected keys (that were candidate beliefs but not defined in gold_statuses)
    # Get all candidate belief ids that were introduced
    candidate_bids = set()
    for sub in episode.submissions:
        for b in sub.candidate_beliefs:
            candidate_bids.add(b.belief_id)
        for b in sub.candidate_replacement_beliefs:
            candidate_bids.add(b.belief_id)
            
    for bid in candidate_bids:
        if bid not in expected_statuses:
            # Under DPA, any candidate belief that is active will have some status.
            # If the gold record did not state it, check if DPA calculated it as something other than AUTHORIZED/etc.
            # Usually all candidate beliefs must have their expected gold status defined.
            actual = active_statuses.get(bid)
            if actual is not None:
                mismatches[bid] = {"expected": "UNDEFINED_IN_GOLD", "actual": actual}

    is_pass = len(mismatches) == 0
    detail = {
        "episode_id": episode.episode_id,
        "failure_type": episode.failure_type_public_or_controlled,
        "domain": episode.domain,
        "passed": is_pass,
        "mismatches": mismatches,
        "expected_statuses": expected_statuses,
        "actual_statuses": active_statuses,
    }
    return is_pass, detail

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Stage C candidate gold consistency (Packet 4F-B)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="development_candidate",
        help="Filter episodes by split"
    )
    parser.add_argument(
        "--require-all-pass",
        action="store_true",
        help="Exit with error code 1 if any candidate validation fails"
    )
    args = parser.parse_args()
    
    episodes = generate_expanded_episodes()
    filtered = [pair for pair in episodes if pair[0].split == args.split]
    
    print(f"[*] Validating {len(filtered)} episodes for split '{args.split}'...")
    
    results = {}
    failures = []
    passed_count = 0
    failed_count = 0
    
    for episode, gold in filtered:
        is_pass, detail = validate_episode(episode, gold)
        results[episode.episode_id] = detail
        if is_pass:
            passed_count += 1
        else:
            failed_count += 1
            failures.append(detail)
            
    print(f"[+] Validation complete. Passed: {passed_count}, Failed: {failed_count}")
    
    os.makedirs("outputs", exist_ok=True)
    
    # Save outputs
    with open("outputs/stagec_dev_semantic_validation_70.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    with open("outputs/stagec_dev_semantic_validation_failures.jsonl", "w", encoding="utf-8") as f:
        for fail in failures:
            f.write(json.dumps(fail) + "\n")
            
    if failed_count > 0:
        print("[!] Validation Failures detected:")
        for fail in failures:
            print(f"  - {fail['episode_id']} ({fail['failure_type']}):")
            for bid, details in fail['mismatches'].items():
                print(f"    * Belief '{bid}': expected={details['expected']}, actual={details['actual']}")
        if args.require_all_pass:
            print("Error: require-all-pass is set. Exiting with error.")
            sys.exit(1)
    else:
        print("[+] All candidates passed executable semantic validation!")

if __name__ == "__main__":
    main()
