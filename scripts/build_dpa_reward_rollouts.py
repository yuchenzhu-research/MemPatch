#!/usr/bin/env python3
"""Execute rollouts and compile RL Rollout examples with DPA-in-the-loop rewards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retracemem.evaluation.raw_dialogue.generator import SyntheticDialogueGenerator
from retracemem.training.reward import score_rollout
from retracemem.authorization import authorize, EvidenceProposalBatch
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType, EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
from retracemem.methods.contracts import SharedCandidateView


def _run_view_and_actions(view: SharedCandidateView, actions: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    edges = []
    for idx, a in enumerate(actions):
        if a["action_type"] == "NO_REVISION":
            continue
        kind = "belief" if a.get("target_belief_id") else "condition"
        tid = a.get("target_belief_id") or a.get("target_condition_id")
        edges.append(
            EvidenceEdge(
                edge_id=f"edge_rollout_{idx}",
                edge_type=EvidenceEdgeType(a["action_type"]),
                evidence_id=a["evidence_ids"][0],
                target_kind=kind,
                target_id=tid,
                verifier="rollout_verifier",
                replacement_belief_id=a.get("replacement_belief_id"),
            )
        )
    proposal = EvidenceProposalBatch(edges=tuple(edges))
    res = authorize(view, (proposal,))
    return res.trace["fine_grained_statuses"], res.trace["defeat_paths"], res.trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPA reward rollouts.")
    parser.add_argument("--in-file", type=str, default="outputs/raw_dialogue_synth.jsonl", help="Input dataset path.")
    parser.add_argument("--out-file", type=str, default="outputs/dpa_rl_rollouts.jsonl", help="Output rollouts JSONL path.")
    args = parser.parse_args()

    if not os.path.exists(args.in_file):
        print(f"Error: input file {args.in_file} does not exist. Run scripts/build_raw_dialogue_synth.py first.")
        sys.exit(1)

    rollouts = []
    with open(args.in_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            ex_id = data["example_id"]
            gold_actions = data["metadata"]["gold_actions"]
            gold_final_statuses = data["metadata"]["gold_final_statuses"]
            g = data["gold_graph"]
            new_ev_id = data["metadata"]["new_evidence_id"]

            # Reconstruct View to execute authorize
            evidence_context = []
            new_evidence_node = None
            for ev in g["evidence_nodes"]:
                node = EvidenceNode(
                    evidence_id=ev["evidence_id"],
                    session_id=ev["session_id"],
                    timestamp=ev.get("timestamp"),
                    text=ev["text"],
                    source_dataset="synth",
                    source_pointer="p",
                )
                evidence_context.append(node)
                if ev["evidence_id"] == new_ev_id:
                    new_evidence_node = node
            if new_evidence_node is None:
                new_evidence_node = EvidenceNode(
                    evidence_id=new_ev_id,
                    session_id="sess_synth",
                    timestamp="2026-06-01T00:00:00Z",
                    text="Trigger",
                    source_dataset="synth",
                    source_pointer="p",
                )
                evidence_context.append(new_evidence_node)

            candidate_beliefs = [
                BeliefNode(
                    belief_id=b["belief_id"],
                    proposition=b["proposition"],
                    source_evidence_ids=tuple(b.get("source_evidence_ids", ())),
                ) for b in g["belief_nodes"]
            ]
            candidate_replacement_beliefs = [
                BeliefNode(
                    belief_id=b["belief_id"],
                    proposition=b["proposition"],
                    source_evidence_ids=tuple(b.get("source_evidence_ids", ())),
                ) for b in g["candidate_replacement_beliefs"]
            ]

            cond_by_id = {
                c["condition_id"]: ConditionNode(
                    condition_id=c["condition_id"],
                    scope_id=c.get("scope_id", "global"),
                    text=c["text"],
                ) for c in g["condition_nodes"]
            }

            conditions_by_belief_dict = {}
            deps_by_belief_dict = {}
            for edge in g["dependency_edges"]:
                bid = edge["belief_id"]
                cid = edge["condition_id"]
                dep = DependencyEdge(
                    edge_id=edge["edge_id"],
                    belief_id=bid,
                    condition_id=cid,
                    inducer="synth",
                )
                deps_by_belief_dict.setdefault(bid, []).append(dep)
                cond = cond_by_id.get(cid)
                if cond:
                    conditions_by_belief_dict.setdefault(bid, []).append(cond)

            view = SharedCandidateView(
                instance_id=ex_id,
                query_id="query",
                query="query",
                new_evidence=new_evidence_node,
                evidence_context=tuple(evidence_context),
                candidate_beliefs=tuple(candidate_beliefs),
                candidate_replacement_beliefs=tuple(candidate_replacement_beliefs),
                candidate_conditions_by_belief=tuple((bid, tuple(conds)) for bid, conds in conditions_by_belief_dict.items()),
                dependency_edges_by_belief=tuple((bid, tuple(deps)) for bid, deps in deps_by_belief_dict.items()),
            )

            # Define valid IDs for score calculations
            valid_beliefs = {b.belief_id for b in candidate_beliefs} | {b.belief_id for b in candidate_replacement_beliefs}
            valid_conditions = set(cond_by_id.keys())
            valid_evidences = {e.evidence_id for e in evidence_context}

            # ----------------------------------------------------
            # Case 1: Gold Rollout (Should score high / full marks)
            # ----------------------------------------------------
            pred_statuses, dpaths, trace = _run_view_and_actions(view, gold_actions)
            breakdown_gold = score_rollout(
                actions=gold_actions,
                valid_json=True,
                valid_vocabulary=True,
                dpa_final_statuses=pred_statuses,
                gold_final_statuses=gold_final_statuses,
                valid_belief_ids=valid_beliefs,
                valid_condition_ids=valid_conditions,
                valid_evidence_ids=valid_evidences,
                defeat_paths=dpaths,
                gold_actions=gold_actions,
            )

            rollout_gold = {
                "example_id": f"{ex_id}_gold",
                "prompt_input": "propose actions",
                "sampled_actions": gold_actions,
                "parser_result": {"valid_json": True, "schema_valid": True},
                "gate_decisions": trace.get("edge_proposals", []),
                "dpa_final_statuses": pred_statuses,
                "gold_final_statuses": gold_final_statuses,
                "reward_breakdown": breakdown_gold.reward_breakdown(),
                "total_reward": breakdown_gold.total_reward,
                "failure_category": breakdown_gold.failure_category,
                "audit_trace": trace,
            }
            rollouts.append(rollout_gold)

            # ----------------------------------------------------
            # Case 2: Perturbed Rollout (Drop one action to trigger penalty)
            # ----------------------------------------------------
            if len(gold_actions) > 1:
                perturbed_actions = gold_actions[:-1]
                pred_statuses_p, dpaths_p, trace_p = _run_view_and_actions(view, perturbed_actions)
                breakdown_p = score_rollout(
                    actions=perturbed_actions,
                    valid_json=True,
                    valid_vocabulary=True,
                    dpa_final_statuses=pred_statuses_p,
                    gold_final_statuses=gold_final_statuses,
                    valid_belief_ids=valid_beliefs,
                    valid_condition_ids=valid_conditions,
                    valid_evidence_ids=valid_evidences,
                    defeat_paths=dpaths_p,
                    gold_actions=gold_actions,
                )

                rollout_p = {
                    "example_id": f"{ex_id}_perturbed",
                    "prompt_input": "propose actions",
                    "sampled_actions": perturbed_actions,
                    "parser_result": {"valid_json": True, "schema_valid": True},
                    "gate_decisions": trace_p.get("edge_proposals", []),
                    "dpa_final_statuses": pred_statuses_p,
                    "gold_final_statuses": gold_final_statuses,
                    "reward_breakdown": breakdown_p.reward_breakdown(),
                    "total_reward": breakdown_p.total_reward,
                    "failure_category": breakdown_p.failure_category,
                    "audit_trace": trace_p,
                }
                rollouts.append(rollout_p)

    # Ensure out dir exists
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully compiled {len(rollouts)} rollout instances at {args.out_file}.")


if __name__ == "__main__":
    main()
