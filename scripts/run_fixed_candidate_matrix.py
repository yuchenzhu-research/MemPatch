#!/usr/bin/env python3
"""Run Protocol A (Fixed-Candidate Revision Control) experiment matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retracemem.evaluation.raw_dialogue.generator import SyntheticDialogueGenerator
from retracemem.evaluation.metrics import evaluate_predictions, aggregate_metrics, write_matrix_outputs
from retracemem.schemas import EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
from retracemem.methods.contracts import SharedCandidateView
from retracemem.authorization import authorize, EvidenceProposalBatch
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from retracemem.proposers.learned_replay import LearnedReplayProposer
from retracemem.evaluation.multiagent.contracts import FixedCandidateSubmission


def _build_submission_from_graph(data: dict) -> FixedCandidateSubmission:
    g = data["gold_graph"]
    new_ev_id = data["metadata"]["new_evidence_id"]
    
    evidence_context = []
    new_ev_node = None
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
            new_ev_node = node
            
    if new_ev_node is None:
        new_ev_node = EvidenceNode(
            evidence_id=new_ev_id,
            session_id="sess_synth",
            timestamp="2026-06-01T00:00:00Z",
            text="Trigger",
            source_dataset="synth",
            source_pointer="p",
        )
        evidence_context.append(new_ev_node)

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

    return FixedCandidateSubmission(
        submission_id=data["example_id"],
        producer_id="synth",
        producer_role="synth",
        task_id=None,
        parent_snapshot_id="snap_0",
        observed_at="2026-06-01T00:00:00Z",
        instance_id="inst",
        query_id="query",
        query="what is the status?",
        evidence_context=tuple(evidence_context),
        new_evidence_id=new_ev_id,
        candidate_beliefs=tuple(candidate_beliefs),
        candidate_replacement_beliefs=tuple(candidate_replacement_beliefs),
        candidate_conditions_by_belief=tuple((bid, tuple(conds)) for bid, conds in conditions_by_belief_dict.items()),
        dependency_edges_by_belief=tuple((bid, tuple(deps)) for bid, deps in deps_by_belief_dict.items()),
    )


def _run_dpa(sub: FixedCandidateSubmission, actions: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    evidence_context = list(sub.evidence_context)
    new_ev_node = next((e for e in evidence_context if e.evidence_id == sub.new_evidence_id), None)
    
    view = SharedCandidateView(
        instance_id=sub.instance_id,
        query_id=sub.query_id,
        query=sub.query,
        new_evidence=new_ev_node,
        evidence_context=tuple(evidence_context),
        candidate_beliefs=sub.candidate_beliefs,
        candidate_replacement_beliefs=sub.candidate_replacement_beliefs,
        candidate_conditions_by_belief=sub.candidate_conditions_by_belief,
        dependency_edges_by_belief=sub.dependency_edges_by_belief,
    )

    edges = []
    for idx, a in enumerate(actions):
        if a["action_type"] == "NO_REVISION":
            continue
        kind = "belief" if a.get("target_belief_id") else "condition"
        tid = a.get("target_belief_id") or a.get("target_condition_id")
        edges.append(
            EvidenceEdge(
                edge_id=f"edge_fixed_{idx}",
                edge_type=EvidenceEdgeType(a["action_type"]),
                evidence_id=a["evidence_ids"][0],
                target_kind=kind,
                target_id=tid,
                verifier="fixed_verifier",
                replacement_belief_id=a.get("replacement_belief_id"),
            )
        )
    proposal = EvidenceProposalBatch(edges=tuple(edges))
    res = authorize(view, (proposal,))
    return res.trace["fine_grained_statuses"], res.trace["defeat_paths"], res.trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fixed-Candidate Experiment Matrix.")
    parser.add_argument("--input", type=str, default="outputs/raw_dialogue_synth.jsonl", help="Input dataset path.")
    parser.add_argument("--out", type=str, default="outputs/smoke/fixed_candidate_metrics.json", help="Output metrics path.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file {args.input} does not exist. Run build_raw_dialogue_synth.py first.")
        sys.exit(1)

    episodes = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    methods = ["oracle_proposer", "stage_a_replay_or_mock", "stage_c_learned_replay", "structured_directjudge_mock"]
    results_by_method = {m: [] for m in methods}
    prediction_rows: list[dict[str, Any]] = []

    def _record(method: str, data: dict, actions: list, statuses: dict, metrics: dict) -> None:
        prediction_rows.append({
            "method": method,
            "example_id": data["example_id"],
            "case_family": data.get("metadata", {}).get("case_family") or data.get("case_family"),
            "predicted_actions": actions,
            "predicted_final_statuses": statuses,
            "metrics": metrics,
        })

    for data in episodes:
        sub = _build_submission_from_graph(data)
        gold_actions = data["metadata"]["gold_actions"]
        gold_final_statuses = data["metadata"]["gold_final_statuses"]
        
        valid_beliefs = {b.belief_id for b in sub.candidate_beliefs} | {b.belief_id for b in sub.candidate_replacement_beliefs}
        valid_conditions = {c.condition_id for _, conds in sub.candidate_conditions_by_belief for c in conds}
        valid_evidences = {e.evidence_id for e in sub.evidence_context}

        # 1. Oracle Proposer
        pred_statuses, dpaths, trace = _run_dpa(sub, gold_actions)
        res_oracle = evaluate_predictions(
            predictions=[{
                "parser_result": {"valid_json": True},
                "sampled_actions": gold_actions,
                "gate_decisions": trace.get("edge_proposals", []),
                "defeat_paths": dpaths,
                "dpa_final_statuses": pred_statuses,
            }],
            gold_final_statuses=gold_final_statuses,
            gold_actions=gold_actions,
            valid_belief_ids=valid_beliefs,
            valid_condition_ids=valid_conditions,
            valid_evidence_ids=valid_evidences,
        )
        results_by_method["oracle_proposer"].append(res_oracle)
        _record("oracle_proposer", data, gold_actions, pred_statuses, res_oracle)

        # 2. Stage A Replay (Replays gold actions with a tiny chance of omitting one action to simulate LLM variance)
        stage_a_actions = list(gold_actions)
        if len(stage_a_actions) > 1 and int(data["example_id"].split("_")[-1]) % 5 == 0:
            stage_a_actions = stage_a_actions[:-1]
        pred_statuses_a, dpaths_a, trace_a = _run_dpa(sub, stage_a_actions)
        res_a = evaluate_predictions(
            predictions=[{
                "parser_result": {"valid_json": True},
                "sampled_actions": stage_a_actions,
                "gate_decisions": trace_a.get("edge_proposals", []),
                "defeat_paths": dpaths_a,
                "dpa_final_statuses": pred_statuses_a,
            }],
            gold_final_statuses=gold_final_statuses,
            gold_actions=gold_actions,
            valid_belief_ids=valid_beliefs,
            valid_condition_ids=valid_conditions,
            valid_evidence_ids=valid_evidences,
        )
        results_by_method["stage_a_replay_or_mock"].append(res_a)
        _record("stage_a_replay_or_mock", data, stage_a_actions, pred_statuses_a, res_a)

        # 3. Stage C Learned Replay (Runs through the LearnedReplayProposer class)
        predecoded = {sub.submission_id: gold_actions}
        proposer = LearnedReplayProposer(predecoded)
        policy_out = proposer.propose(sub)
        
        # Format proposer output actions
        replay_actions = [a.to_dict() for a in policy_out.parsed_actions]
        pred_statuses_c, dpaths_c, trace_c = _run_dpa(sub, replay_actions)
        res_c = evaluate_predictions(
            predictions=[{
                "parser_result": {"valid_json": policy_out.parsing_valid},
                "sampled_actions": replay_actions,
                "gate_decisions": trace_c.get("edge_proposals", []),
                "defeat_paths": dpaths_c,
                "dpa_final_statuses": pred_statuses_c,
            }],
            gold_final_statuses=gold_final_statuses,
            gold_actions=gold_actions,
            valid_belief_ids=valid_beliefs,
            valid_condition_ids=valid_conditions,
            valid_evidence_ids=valid_evidences,
        )
        results_by_method["stage_c_learned_replay"].append(res_c)
        _record("stage_c_learned_replay", data, replay_actions, pred_statuses_c, res_c)

        # 4. Structured DirectJudge Mock (DirectJudge baseline, directly predicts final statuses, bypasses Core DPA)
        # DirectJudge directly outputs statuses with slight noise (90% accuracy)
        dj_statuses = {}
        for bid in gold_final_statuses:
            if int(data["example_id"].split("_")[-1]) % 10 == 0:
                # Inject noise: change status
                dj_statuses[bid] = "AUTHORIZED" if gold_final_statuses[bid] != "AUTHORIZED" else "SUPERSEDED"
            else:
                dj_statuses[bid] = gold_final_statuses[bid]

        res_dj = evaluate_predictions(
            predictions=[{
                "parser_result": {"valid_json": True},
                "sampled_actions": [],
                "gate_decisions": [],
                "defeat_paths": [],
                "dpa_final_statuses": dj_statuses,
            }],
            gold_final_statuses=gold_final_statuses,
            gold_actions=[],
        )
        results_by_method["structured_directjudge_mock"].append(res_dj)
        _record("structured_directjudge_mock", data, [], dj_statuses, res_dj)

    # Compute aggregate metrics
    aggregated = {}
    for m in methods:
        aggregated[m] = aggregate_metrics(results_by_method[m])

    paths = write_matrix_outputs(args.out, aggregated, prediction_rows)

    print(
        "Protocol A Runner complete. "
        f"metrics(json)={paths['json']} metrics(csv)={paths['csv']} predictions={paths['predictions']}"
    )


if __name__ == "__main__":
    main()
