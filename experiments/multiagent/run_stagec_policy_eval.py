from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from typing import Any, Dict, List, Tuple
from experiments.multiagent.stagec_dataset import build_stagec_dataset
from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
from experiments.multiagent.episodes_fc_dev import get_fc_dev_episodes
from experiments.multiagent.metrics import (
    compute_fixed_candidate_metrics,
    aggregate_fixed_candidate_metrics,
)
from retracemem.multiagent.contracts import (
    SubagentMemorySubmission,
    SharedMemorySnapshotResult,
)
from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    FixedCandidateEpisodeMethodResult,
    MethodDecisionRecord,
    ExperimentRunManifest,
)
from retracemem.multiagent.commit import commit_submission_sequence
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from retracemem.authorization import EvidenceProposalBatch


def get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def generate_mock_llm_response(targets) -> str:
    """Generate a valid mock JSON response based on gold targets."""
    items = []
    for t in targets:
        if t.action_type == "NO_REVISION":
            continue
        items.append({
            "action_type": t.action_type,
            "target_belief_id": t.target_belief_id,
            "target_condition_id": t.target_condition_id,
            "replacement_belief_id": t.replacement_belief_id,
            "rationale": t.rationale or "Oracle replay rationale",
            "evidence_ids": list(t.evidence_ids),
        })
    return json.dumps(items, indent=2)


def run_stagec_policy_eval(mode: str) -> Dict[str, Any]:
    run_id = f"stagec_eval_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    examples = build_stagec_dataset()
    episodes = get_fc_dev_episodes()
    
    # Map examples by example_id
    examples_map = {ex.example_id: ex for ex in examples}
    
    policy = PromptTypedRevisionPolicy()
    
    results_rows: List[Dict[str, Any]] = []
    run_results = []
    
    # Track episodes evaluated
    for ep, gold, artifact in episodes:
        subagent_subs = []
        for sub in ep.submissions:
            example_id = f"ex_{ep.episode_id}_{sub.submission_id}"
            ex = examples_map.get(example_id)
            if not ex:
                continue
                
            # Simulate LLM call: build user messages and use mock fixture response
            messages = policy.build_messages(sub)
            mock_response = generate_mock_llm_response(ex.targets)
            
            # Parse responses
            policy_out = policy.parse_response(
                mock_response,
                example_id=example_id,
                submission_id=sub.submission_id,
            )
            
            # Extract proposal batches
            proposal_batches = policy_out.proposal_batches
            
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
                proposal_batches=proposal_batches,
                task_id=sub.task_id,
                metadata=sub.metadata,
            )
            subagent_subs.append(subagent_sub)
            
        # Commit the sequence through cumulative snapshot evaluation
        seq_res = commit_submission_sequence(tuple(subagent_subs), final_snapshot_evaluation=True)
        active_statuses = seq_res.final_belief_statuses
        
        # Build Decisions
        decisions = []
        for bid, status in active_statuses.items():
            dec = "AUTHORIZE"
            if status == "BLOCKED":
                dec = "REJECT"
            elif status == "SUPERSEDED":
                dec = "SUPERSEDE"
            elif status == "UNRESOLVED":
                dec = "DEFER"
                
            decisions.append(
                MethodDecisionRecord(
                    belief_id=bid,
                    decision=dec,
                    rationale=f"PromptPolicyFixture: DPA state {status}",
                )
            )
            
        res = FixedCandidateEpisodeMethodResult(
            episode_id=ep.episode_id,
            domain=ep.domain,
            failure_type=gold.failure_type or ep.failure_type_public_or_controlled,
            method_name="PromptPolicy_Fixture",
            protocol_mode=ep.protocol_mode,
            proposal_source="offline_fixture_response",
            final_belief_statuses=active_statuses,
            decisions=tuple(decisions),
            task_predictions={}, # calculated during metrics evaluation
            metadata={},
        )
        
        run_results.append((gold, ep, res))
        
        ep_metrics = compute_fixed_candidate_metrics(gold, ep.downstream_tasks, res)
        
        conflict_density = ep.stress_factors.get("conflict_density", 0.0)
        delay_depth = ep.stress_factors.get("delay_depth", 0)
        num_subagents = len(set(s.producer_id for s in ep.submissions))
        num_submissions = len(ep.submissions)
        role_diversity = len(set(ep.subagent_roles))
        recovery_present = gold.failure_type == "temporary_blocker_recovery"
        
        for m_name, m_val in ep_metrics.items():
            results_rows.append({
                "run_id": run_id,
                "episode_id": ep.episode_id,
                "domain": ep.domain,
                "failure_type": gold.failure_type or ep.failure_type_public_or_controlled,
                "protocol_mode": ep.protocol_mode,
                "scientific_status": "pipeline_validation_only",
                "split": ep.split,
                "method_name": "PromptPolicy_Fixture",
                "backbone_model": None,
                "proposal_source": "offline_fixture_response",
                "candidate_source": "fixed_candidate",
                "number_of_subagents": num_subagents,
                "number_of_submissions": num_submissions,
                "role_diversity": role_diversity,
                "conflict_density": conflict_density,
                "delay_depth": delay_depth,
                "recovery_present": recovery_present,
                "metric_name": m_name,
                "metric_value": m_val,
                "trace_available": True,
                "calls": 0,
                "tokens": 0,
                "latency_ms": 0.0,
                "policy_variant": "prompt_fixture",
                "checkpoint_id": None,
                "training_split": "development_only",
                "training_step": 0,
                "training_examples_seen": 0,
                "reward_variant": None,
                "authorization_reward": 0.0,
                "downstream_task_reward": 0.0,
                "scope_expansion_penalty": 0.0,
                "stale_penalty": 0.0,
                "total_reward": 0.0,
            })
            
    aggregated = aggregate_fixed_candidate_metrics(run_results)
    
    manifest = ExperimentRunManifest(
        run_id=run_id,
        split="development_only",
        methods=("PromptPolicy_Fixture",),
        episode_ids=tuple(ep.episode_id for ep, _, _ in episodes),
        model_config={"api": "offline_replay", "protocol_mode": "oracle_edge_replay"},
        prompt_hashes={"prompt_revision_instruction": "hash_placeholder"},
        code_commit_sha=get_git_commit_sha(),
        created_at=datetime.datetime.now().isoformat(),
        mode=mode,
    )
    
    os.makedirs("outputs", exist_ok=True)
    jsonl_path = "outputs/stagec_policy_dev_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r) + "\n")
            
    summary_path = "outputs/stagec_policy_dev_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
        
    manifest_dict = manifest.to_dict()
    manifest_dict.update({
        "dataset_version": "fc_dev_v1",
        "episode_factory_hash": "hash_placeholder",
        "method_config": {},
        "random_seed": 42,
        "notes": "Stage C prompt policy offline fixture evaluation."
    })
    manifest_path = "outputs/stagec_policy_dev_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)
        
    details_path = "outputs/stagec_policy_dev_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump([res.to_dict() for _, _, res in run_results], f, indent=2)
        
    return {
        "results_count": len(results_rows),
        "aggregated": aggregated,
        "jsonl_path": jsonl_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "details_path": details_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage C Prompt Policy Offline Fixture Evaluation (Packet 4D)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline_fixture",
        choices=["offline_fixture"],
        help="Evaluation execution mode",
    )
    args = parser.parse_args()
    
    res = run_stagec_policy_eval(args.mode)
    print("Stage C Prompt Policy Offline Fixture Evaluation completed successfully.")
    print(f"Results: {res['jsonl_path']}, {res['summary_path']}, {res['manifest_path']}")
    print(f"Total metric rows: {res['results_count']}")
    
    # Print accuracy
    for key, vals in sorted(res["aggregated"].items()):
        if "__overall__all" in key:
            acc = vals.get("authorization_accuracy", "N/A")
            print(f"  Stage C policy overall accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
