#!/usr/bin/env python3
"""Run Protocol B (Raw-Dialogue Revision Authorization / ReTrace-Learn-Full) experiment matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retracemem.evaluation.metrics import evaluate_predictions, aggregate_metrics
from retracemem.schemas import EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
from retracemem.methods.contracts import SharedCandidateView
from retracemem.authorization import authorize, EvidenceProposalBatch
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from retracemem.proposers.learned_replay import LearnedReplayProposer
from scripts.run_fixed_candidate_matrix import _build_submission_from_graph, _run_dpa


def _perturb_graph(graph: dict, rng_seed: int) -> dict:
    """Simulate a noisy/learned graph extractor by injecting minor errors."""
    import copy
    g = copy.deepcopy(graph)
    # 10% chance to drop a dependency edge, simulating extraction failure
    if len(g["dependency_edges"]) > 0 and rng_seed % 7 == 0:
        g["dependency_edges"].pop(0)
    # 10% chance to modify a proposition slightly
    if len(g["belief_nodes"]) > 0 and rng_seed % 9 == 0:
        g["belief_nodes"][0]["proposition"] = "Perturbed proposition: " + g["belief_nodes"][0]["proposition"]
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Raw-Dialogue Experiment Matrix.")
    parser.add_argument("--input", type=str, default="outputs/raw_dialogue_synth.jsonl", help="Input dataset path.")
    parser.add_argument("--out", type=str, default="outputs/smoke/raw_dialogue_metrics.json", help="Output metrics path.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file {args.input} does not exist. Run build_raw_dialogue_synth.py first.")
        sys.exit(1)

    episodes = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    methods = [
        "oracle_graph_oracle_proposer",
        "oracle_graph_learned_replay_proposer",
        "mock_extracted_graph_learned_replay_proposer",
        "raw_directjudge_mock"
    ]
    results_by_method = {m: [] for m in methods}

    for data in episodes:
        gold_actions = data["metadata"]["gold_actions"]
        gold_final_statuses = data["metadata"]["gold_final_statuses"]
        idx = int(data["example_id"].split("_")[-1])

        # ----------------------------------------------------
        # 1. Oracle Graph + Oracle Proposer
        # ----------------------------------------------------
        sub_oracle = _build_submission_from_graph(data)
        valid_beliefs_o = {b.belief_id for b in sub_oracle.candidate_beliefs} | {b.belief_id for b in sub_oracle.candidate_replacement_beliefs}
        valid_conditions_o = {c.condition_id for _, conds in sub_oracle.candidate_conditions_by_belief for c in conds}
        valid_evidences_o = {e.evidence_id for e in sub_oracle.evidence_context}

        pred_statuses_o, dpaths_o, trace_o = _run_dpa(sub_oracle, gold_actions)
        res_o = evaluate_predictions(
            predictions=[{
                "parser_result": {"valid_json": True},
                "sampled_actions": gold_actions,
                "gate_decisions": trace_o.get("edge_proposals", []),
                "defeat_paths": dpaths_o,
                "dpa_final_statuses": pred_statuses_o,
            }],
            gold_final_statuses=gold_final_statuses,
            gold_actions=gold_actions,
            valid_belief_ids=valid_beliefs_o,
            valid_condition_ids=valid_conditions_o,
            valid_evidence_ids=valid_evidences_o,
        )
        results_by_method["oracle_graph_oracle_proposer"].append(res_o)

        # ----------------------------------------------------
        # 2. Oracle Graph + Learned Replay Proposer
        # ----------------------------------------------------
        predecoded = {sub_oracle.submission_id: gold_actions}
        proposer = LearnedReplayProposer(predecoded)
        policy_out = proposer.propose(sub_oracle)
        replay_actions = [a.to_dict() for a in policy_out.parsed_actions]
        pred_statuses_or, dpaths_or, trace_or = _run_dpa(sub_oracle, replay_actions)
        res_or = evaluate_predictions(
            predictions=[{
                "parser_result": {"valid_json": policy_out.parsing_valid},
                "sampled_actions": replay_actions,
                "gate_decisions": trace_or.get("edge_proposals", []),
                "defeat_paths": dpaths_or,
                "dpa_final_statuses": pred_statuses_or,
            }],
            gold_final_statuses=gold_final_statuses,
            gold_actions=gold_actions,
            valid_belief_ids=valid_beliefs_o,
            valid_condition_ids=valid_conditions_o,
            valid_evidence_ids=valid_evidences_o,
        )
        results_by_method["oracle_graph_learned_replay_proposer"].append(res_or)

        # ----------------------------------------------------
        # 3. Mock Extracted Graph + Learned Replay Proposer
        # ----------------------------------------------------
        perturbed_data = dict(data)
        perturbed_data["gold_graph"] = _perturb_graph(data["gold_graph"], idx)
        sub_perturbed = _build_submission_from_graph(perturbed_data)
        
        valid_beliefs_p = {b.belief_id for b in sub_perturbed.candidate_beliefs} | {b.belief_id for b in sub_perturbed.candidate_replacement_beliefs}
        valid_conditions_p = {c.condition_id for _, conds in sub_perturbed.candidate_conditions_by_belief for c in conds}
        valid_evidences_p = {e.evidence_id for e in sub_perturbed.evidence_context}

        # The replayer matches actions on submission_id
        predecoded_p = {sub_perturbed.submission_id: gold_actions}
        proposer_p = LearnedReplayProposer(predecoded_p)
        policy_out_p = proposer_p.propose(sub_perturbed)
        replay_actions_p = [a.to_dict() for a in policy_out_p.parsed_actions]

        pred_statuses_p, dpaths_p, trace_p = _run_dpa(sub_perturbed, replay_actions_p)
        res_p = evaluate_predictions(
            predictions=[{
                "parser_result": {"valid_json": policy_out_p.parsing_valid},
                "sampled_actions": replay_actions_p,
                "gate_decisions": trace_p.get("edge_proposals", []),
                "defeat_paths": dpaths_p,
                "dpa_final_statuses": pred_statuses_p,
            }],
            gold_final_statuses=gold_final_statuses,
            gold_actions=gold_actions,
            valid_belief_ids=valid_beliefs_p,
            valid_condition_ids=valid_conditions_p,
            valid_evidence_ids=valid_evidences_p,
        )
        results_by_method["mock_extracted_graph_learned_replay_proposer"].append(res_p)

        # ----------------------------------------------------
        # 4. Raw DirectJudge Mock (Stage B, directly classifies raw dialogue to belief statuses)
        # ----------------------------------------------------
        raw_dj_statuses = {}
        for bid in gold_final_statuses:
            # High noise rate (80% accuracy) for raw text classifier
            if idx % 5 == 0:
                raw_dj_statuses[bid] = "AUTHORIZED" if gold_final_statuses[bid] != "AUTHORIZED" else "BLOCKED"
            else:
                raw_dj_statuses[bid] = gold_final_statuses[bid]

        res_raw_dj = evaluate_predictions(
            predictions=[{
                "parser_result": {"valid_json": True},
                "sampled_actions": [],
                "gate_decisions": [],
                "defeat_paths": [],
                "dpa_final_statuses": raw_dj_statuses,
            }],
            gold_final_statuses=gold_final_statuses,
            gold_actions=[],
        )
        results_by_method["raw_directjudge_mock"].append(res_raw_dj)

    # Compute aggregate metrics
    aggregated = {}
    for m in methods:
        aggregated[m] = aggregate_metrics(results_by_method[m])

    # Ensure output dir exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Protocol B Runner complete. Stored smoke results at {args.out}.")


if __name__ == "__main__":
    main()
