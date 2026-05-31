from __future__ import annotations

import json
import os
import sys
import datetime
from typing import Any, Dict, List, Tuple

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src"))

from retracemem.evaluation.multiagent.data.stagec_dataset import build_stagec_dataset
from retracemem.proposers.typed_revision_policy import PromptTypedRevisionPolicy
from retracemem.evaluation.multiagent.data.episodes_fc_dev import get_fc_dev_episodes
from retracemem.multiagent.commit import commit_submission_sequence
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from retracemem.authorization import EvidenceProposalBatch


def generate_ablation_mock_response(targets: tuple[Any, ...], new_evidence_id: str, allowed_actions: tuple[str, ...]) -> str:
    items = []
    for t in targets:
        if t.action_type not in allowed_actions:
            continue
        if t.action_type == "NO_REVISION":
            items.append({
                "action_type": "NO_REVISION",
                "target_belief_id": None,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "rationale": t.rationale or "No evidence-grounded revision is warranted.",
                "evidence_ids": list(t.evidence_ids) if t.evidence_ids else [new_evidence_id],
            })
        else:
            items.append({
                "action_type": t.action_type,
                "target_belief_id": t.target_belief_id,
                "target_condition_id": t.target_condition_id,
                "replacement_belief_id": t.replacement_belief_id,
                "rationale": t.rationale or "Ablated mock replay rationale",
                "evidence_ids": list(t.evidence_ids),
            })
    # If no actions are mock-generated but NO_REVISION was not explicitly ablated, fallback to NO_REVISION
    if not items and "NO_REVISION" in allowed_actions:
        items.append({
            "action_type": "NO_REVISION",
            "target_belief_id": None,
            "target_condition_id": None,
            "replacement_belief_id": None,
            "rationale": "Fallback NO_REVISION due to ablation.",
            "evidence_ids": [new_evidence_id],
        })
    return json.dumps(items, indent=2)


def run_ablation_experiment(variant_name: str, allowed_actions: tuple[str, ...]) -> Dict[str, Any]:
    examples = build_stagec_dataset()
    episodes = get_fc_dev_episodes()
    examples_map = {ex.example_id: ex for ex in examples}
    
    policy = PromptTypedRevisionPolicy(allowed_actions=allowed_actions)
    
    total_episodes = len(episodes)
    unrepresentable_count = 0
    correct_episodes = 0
    parser_valid_count = 0
    total_proposals = 0
    
    results = []

    for ep, gold, artifact in episodes:
        subagent_subs = []
        is_unrepresentable = False
        
        for sub in ep.submissions:
            example_id = f"ex_{ep.episode_id}_{sub.submission_id}"
            ex = examples_map.get(example_id)
            if not ex:
                continue
                
            total_proposals += 1
            # Determine if this example is representable under current allowed_actions
            has_unallowed = any(t.action_type not in allowed_actions for t in ex.targets)
            if has_unallowed:
                is_unrepresentable = True
                
            # Build user messages (just to ensure generation runs)
            policy.build_messages(sub)
            mock_response = generate_ablation_mock_response(ex.targets, sub.new_evidence_id, allowed_actions)
            
            policy_out = policy.parse_response(
                mock_response,
                example_id=example_id,
                submission=sub,
            )
            
            if policy_out.parsing_valid:
                parser_valid_count += 1
                
            # Convert targets to SubagentMemorySubmission with proposal batches
            from retracemem.multiagent.contracts import SubagentMemorySubmission
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
                proposal_batches=policy_out.proposal_batches,
                task_id=sub.task_id,
                metadata=sub.metadata,
            )
            subagent_subs.append(subagent_sub)
            
        if is_unrepresentable:
            unrepresentable_count += 1
            
        # Execute the committed sequence
        exec_result = commit_submission_sequence(tuple(subagent_subs), final_snapshot_evaluation=True)
        final_belief_statuses = exec_result.final_belief_statuses
        
        # Verify status accuracy against gold expected statuses
        match = True
        for bid, expected_status in gold.gold_snapshot.belief_statuses.items():
            actual = final_belief_statuses.get(bid)
            if actual != expected_status:
                match = False
                break
                
        if match:
            correct_episodes += 1
            
        results.append({
            "episode_id": ep.episode_id,
            "failure_type": gold.failure_type or ep.failure_type_public_or_controlled,
            "is_unrepresentable": is_unrepresentable,
            "success": match,
        })
        
    accuracy = correct_episodes / total_episodes if total_episodes else 0.0
    unrepresentable_rate = unrepresentable_count / total_episodes if total_episodes else 0.0
    parser_validity_rate = parser_valid_count / total_proposals if total_proposals else 0.0
    
    return {
        "variant": variant_name,
        "allowed_actions": allowed_actions,
        "accuracy": accuracy,
        "unrepresentable_case_rate": unrepresentable_rate,
        "parser_validity_rate": parser_validity_rate,
        "correct_episodes": correct_episodes,
        "total_episodes": total_episodes,
        "results": results,
    }


def main() -> None:
    print("=" * 70)
    print("ACTION VOCABULARY ABLATION EVALUATION RUNNER")
    print("=" * 70)
    
    variants = {
        "FullCore": ("SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"),
        "w/o_BLOCKS": ("SUPERSEDES", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"),
        "w/o_RELEASES": ("SUPERSEDES", "BLOCKS", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"),
        "w/o_SUPERSEDES": ("BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS", "NO_REVISION"),
        "w/o_UNCERTAIN": ("SUPERSEDES", "BLOCKS", "RELEASES", "REAFFIRMS", "NO_REVISION"),
        "w/o_REAFFIRMS": ("SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "NO_REVISION"),
        "w/o_NO_REVISION": ("SUPERSEDES", "BLOCKS", "RELEASES", "UNCERTAIN", "REAFFIRMS"),
    }
    
    heatmap_data = []
    summary_report = {}
    
    for name, allowed in variants.items():
        print(f"[*] Evaluating ablation variant: {name} ...")
        res = run_ablation_experiment(name, allowed)
        summary_report[name] = {
            "accuracy": f"{res['correct_episodes']}/{res['total_episodes']} ({res['accuracy']:.2%})",
            "unrepresentable_rate": f"{res['unrepresentable_case_rate']:.2%}",
            "parser_validity": f"{res['parser_validity_rate']:.2%}",
        }
        
        # Build Heatmap delta data relative to FullCore
        # Collect success status per failure type
        by_fail_type: Dict[str, List[bool]] = {}
        for r in res["results"]:
            by_fail_type.setdefault(r["failure_type"], []).append(r["success"])
            
        summary_by_fail = {ft: sum(s) / len(s) for ft, s in by_fail_type.items()}
        summary_report[name]["by_failure_type"] = summary_by_fail
        
    # Calculate Deltas relative to FullCore
    full_core_fail = summary_report["FullCore"]["by_failure_type"]
    for name in variants:
        if name == "FullCore":
            continue
        v_fail = summary_report[name]["by_failure_type"]
        for ft in full_core_fail:
            delta = v_fail.get(ft, 0.0) - full_core_fail[ft]
            heatmap_data.append({
                "removed_action": name,
                "failure_type": ft,
                "delta": delta,
            })
            
    print("\nAction Ablation Experiment Summary:")
    print(json.dumps(summary_report, indent=2))
    
    # Save outputs
    output_dir = "outputs/ablation_studies"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "action_ablation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_report,
            "heatmap": heatmap_data,
        }, f, indent=2, ensure_ascii=False)
        
    print(f"\n[+] Saved ablation report to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
