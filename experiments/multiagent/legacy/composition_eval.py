from __future__ import annotations

import json
import os
import sys
import datetime
from typing import Any, Dict, List, Tuple

# Ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src"))

from experiments.multiagent.stagec_dataset import build_stagec_dataset
from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
from experiments.multiagent.episodes_fc_dev import get_fc_dev_episodes
from experiments.multiagent.methods import DirectJudgeReplayMethod
from retracemem.multiagent.contracts import SubagentMemorySubmission
from retracemem.multiagent.commit import commit_submission_sequence


def generate_mock_response(targets: tuple[Any, ...], new_evidence_id: str, limit_one: bool = False) -> str:
    active_targets = targets
    if limit_one and targets:
        active_targets = targets[:1]
        
    items = []
    for t in active_targets:
        if t.action_type == "NO_REVISION":
            items.append({
                "action_type": "NO_REVISION",
                "target_belief_id": None,
                "target_condition_id": None,
                "replacement_belief_id": None,
                "rationale": t.rationale or "Mock rationale",
                "evidence_ids": list(t.evidence_ids) if t.evidence_ids else [new_evidence_id],
            })
        else:
            items.append({
                "action_type": t.action_type,
                "target_belief_id": t.target_belief_id,
                "target_condition_id": t.target_condition_id,
                "replacement_belief_id": t.replacement_belief_id,
                "rationale": t.rationale or "Mock rationale",
                "evidence_ids": list(t.evidence_ids),
            })
            
    if not items:
        items.append({
            "action_type": "NO_REVISION",
            "target_belief_id": None,
            "target_condition_id": None,
            "replacement_belief_id": None,
            "rationale": "No actions proposed.",
            "evidence_ids": [new_evidence_id],
        })
    return json.dumps(items, indent=2)


def run_composition_experiment() -> Dict[str, Any]:
    examples = build_stagec_dataset()
    episodes = get_fc_dev_episodes()
    examples_map = {ex.example_id: ex for ex in examples}
    
    policy = PromptTypedRevisionPolicy()
    dj_method = DirectJudgeReplayMethod()
    
    episode_records = []
    
    for ep, gold, artifact in episodes:
        gold_actions_list = []
        for sub in ep.submissions:
            example_id = f"ex_{ep.episode_id}_{sub.submission_id}"
            ex = examples_map.get(example_id)
            if ex:
                gold_actions_list.extend(ex.targets)
                
        gold_action_count = len(gold_actions_list)
        requires_multi = gold_action_count > 1
        
        # Determine pattern
        patterns = sorted([t.action_type for t in gold_actions_list if t.action_type != "NO_REVISION"])
        pattern = "+".join(patterns) if patterns else "NO_REVISION"
        
        # 1. MultiAction ReTrace
        ma_subs = []
        for sub in ep.submissions:
            example_id = f"ex_{ep.episode_id}_{sub.submission_id}"
            ex = examples_map.get(example_id)
            mock_res = generate_mock_response(ex.targets if ex else (), sub.new_evidence_id, limit_one=False)
            policy_out = policy.parse_response(mock_res, example_id=example_id, submission=sub)
            
            ma_subs.append(SubagentMemorySubmission(
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
            ))
        ma_exec = commit_submission_sequence(tuple(ma_subs), final_snapshot_evaluation=True)
        ma_match = True
        for bid, exp in gold.gold_snapshot.belief_statuses.items():
            if ma_exec.final_belief_statuses.get(bid) != exp:
                ma_match = False
                break
                
        # 2. SingleAction ReTrace
        sa_subs = []
        for sub in ep.submissions:
            example_id = f"ex_{ep.episode_id}_{sub.submission_id}"
            ex = examples_map.get(example_id)
            mock_res = generate_mock_response(ex.targets if ex else (), sub.new_evidence_id, limit_one=True)
            policy_out = policy.parse_response(mock_res, example_id=example_id, submission=sub)
            
            sa_subs.append(SubagentMemorySubmission(
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
            ))
        sa_exec = commit_submission_sequence(tuple(sa_subs), final_snapshot_evaluation=True)
        sa_match = True
        for bid, exp in gold.gold_snapshot.belief_statuses.items():
            if sa_exec.final_belief_statuses.get(bid) != exp:
                sa_match = False
                break
                
        # 3. DirectJudge Replay
        dj_res = dj_method.run_fixed_episode(ep, artifact)
        dj_match = True
        for bid, exp in gold.gold_snapshot.belief_statuses.items():
            if dj_res.final_belief_statuses.get(bid) != exp:
                dj_match = False
                break
                
        episode_records.append({
            "episode_id": ep.episode_id,
            "gold_action_count": gold_action_count,
            "requires_multi_action": requires_multi,
            "action_composition_pattern": pattern,
            "multi_action_success": ma_match,
            "single_action_success": sa_match,
            "direct_judge_success": dj_match,
        })
        
    # Group results by gold_action_count (1, 2, 3+)
    groups = {
        "1": {"multi": [], "single": [], "dj": []},
        "2": {"multi": [], "single": [], "dj": []},
        "3+": {"multi": [], "single": [], "dj": []},
    }
    
    for r in episode_records:
        cnt = r["gold_action_count"]
        g_key = "1" if cnt == 1 else ("2" if cnt == 2 else "3+")
        groups[g_key]["multi"].append(r["multi_action_success"])
        groups[g_key]["single"].append(r["single_action_success"])
        groups[g_key]["dj"].append(r["direct_judge_success"])
        
    plot_ready_data = []
    for g_key, data in groups.items():
        ma_acc = sum(data["multi"]) / len(data["multi"]) if data["multi"] else 0.0
        sa_acc = sum(data["single"]) / len(data["single"]) if data["single"] else 0.0
        dj_acc = sum(data["dj"]) / len(data["dj"]) if data["dj"] else 0.0
        plot_ready_data.append({
            "gold_action_count": g_key,
            "multi_action_accuracy": ma_acc,
            "single_action_accuracy": sa_acc,
            "direct_judge_accuracy": dj_acc,
            "sample_size": len(data["multi"]),
        })
        
    return {
        "plot_data": plot_ready_data,
        "records": episode_records,
    }


def main() -> None:
    print("=" * 70)
    print("MULTI-ACTION COMPOSITION EVALUATION")
    print("=" * 70)
    
    res = run_composition_experiment()
    print("\nLine Plot Ready Data (Accuracies by Action Count):")
    print(json.dumps(res["plot_data"], indent=2))
    
    output_dir = "outputs/ablation_studies"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "multi_action_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
        
    print(f"\n[+] Saved multi-action report to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
