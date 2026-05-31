"""Shared ReTrace revision pipeline: typed proposal -> commit -> DPA.

Stage A and Stage C both call run_retrace_variant_on_episode with a proposer.
The final commit is deterministic via commit_submission_sequence -> authorize(...).
"""
from __future__ import annotations

from typing import Any

from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from retracemem.authorization import EvidenceProposalBatch
from retracemem.multiagent.commit import commit_submission_sequence
from retracemem.multiagent.contracts import SubagentMemorySubmission
from retracemem.evaluation.multiagent.contracts import (
    FixedCandidateInputEpisode,
    FixedCandidateGoldRecord,
    TypedRevisionTarget,
    ProposalPolicyOutput,
)


def run_retrace_variant_on_episode(
    episode: FixedCandidateInputEpisode,
    gold: FixedCandidateGoldRecord,
    proposer: Any,
    mock: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Runs Stage A (Decomposition flow) on a single episode."""
    subagent_subs = []
    ep_a_raw_outputs = []
    ep_a_parsed_actions = []

    for sub in episode.submissions:
        if mock:
            # Generate dynamic mock targets based on gold labels
            mock_targets = []
            for target in gold.gold_typed_targets:
                if target.submission_id == sub.submission_id:
                    mock_targets.append(target)
            if not mock_targets:
                from retracemem.evaluation.multiagent.contracts import TypedRevisionTarget
                mock_targets.append(TypedRevisionTarget(
                    submission_id=sub.submission_id,
                    action_type="NO_REVISION",
                    target_belief_id=None,
                    target_condition_id=None,
                    replacement_belief_id=None,
                    rationale="Mock correct target (NO_REVISION)",
                    evidence_ids=(sub.new_evidence_id,)
                ))
            
            from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
            from retracemem.authorization import EvidenceProposalBatch
            from retracemem.evaluation.multiagent.contracts import ProposalPolicyOutput
            
            edges = []
            for idx_edge, t in enumerate(mock_targets):
                if t.action_type == "NO_REVISION":
                    continue
                target_kind = "belief" if t.target_belief_id else "condition"
                target_id = t.target_belief_id or t.target_condition_id
                edges.append(EvidenceEdge(
                    edge_id=f"edge_policy_ex_{sub.submission_id}_{idx_edge}",
                    edge_type=EvidenceEdgeType(t.action_type) if hasattr(EvidenceEdgeType, t.action_type) else t.action_type,
                    evidence_id=str(t.evidence_ids[0]),
                    target_kind=target_kind,
                    target_id=target_id,
                    verifier="stagec_policy",
                    replacement_belief_id=t.replacement_belief_id,
                    rationale=t.rationale,
                ))
            proposal_batches = ()
            if edges:
                proposal_batches = (
                    EvidenceProposalBatch(
                        edges=tuple(edges),
                        metadata={"parser": "PromptTypedRevisionPolicy"},
                    ),
                )
            proposal_output = ProposalPolicyOutput(
                example_id=f"ex_{sub.submission_id}",
                submission_id=sub.submission_id,
                policy_variant="mock_gold",
                proposal_batches=proposal_batches,
                parsing_valid=True,
                errors=(),
                parsed_actions=tuple(mock_targets),
                metadata={"prompt": "Mock Prompt", "raw_response": "Mock Response"},
            )
        else:
            proposal_output = proposer.propose(sub)

        raw_response = proposal_output.metadata.get("raw_response", "")
        full_prompt = proposal_output.metadata.get("prompt", "")
        parse_err = "\n".join(proposal_output.errors) if proposal_output.errors else None

        parsed_actions = []
        if proposal_output.parsing_valid:
            for t in proposal_output.parsed_actions:
                parsed_actions.append({
                    "action_type": t.action_type,
                    "target_belief_id": t.target_belief_id,
                    "target_condition_id": t.target_condition_id,
                    "replacement_belief_id": t.replacement_belief_id,
                    "rationale": t.rationale,
                    "evidence_ids": list(t.evidence_ids),
                })

        proposal_batches = proposal_output.proposal_batches
        proposal_edges = []
        for batch in proposal_batches:
            for edge in batch.edges:
                proposal_edges.append({
                    "edge_id": edge.edge_id,
                    "edge_type": edge.edge_type.value if hasattr(edge.edge_type, "value") else str(edge.edge_type),
                    "evidence_id": edge.evidence_id,
                    "target_kind": edge.target_kind,
                    "target_id": edge.target_id,
                    "replacement_belief_id": edge.replacement_belief_id,
                    "rationale": edge.rationale,
                })

        # Build SubagentMemorySubmission with predicted proposal batches
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

        ep_a_raw_outputs.append({
            "submission_id": sub.submission_id,
            "prompt": full_prompt,
            "raw_response": raw_response,
            "parse_error": parse_err,
        })
        ep_a_parsed_actions.append({
            "submission_id": sub.submission_id,
            "actions": parsed_actions,
            "proposal_edges": proposal_edges,
            "parse_error": parse_err,
            "decision_audit": proposal_output.metadata.get("decision_audit"),
            "first_pass_valid_json": proposal_output.metadata.get("first_pass_valid_json"),
            "first_pass_parser_error": proposal_output.metadata.get("first_pass_parser_error"),
            "repair_triggered": proposal_output.metadata.get("repair_triggered"),
            "repair_success": proposal_output.metadata.get("repair_success"),
            # Prompt-variant / conflict-aware visibility (debug only; not used by
            # metrics). Surfaced from ProposalPolicyOutput.metadata so conflict
            # diagnostics appear in stage_a_parsed.jsonl.
            "prompt_variant": proposal_output.metadata.get("prompt_variant"),
            "conflict_warning_triggered": proposal_output.metadata.get("conflict_warning_triggered"),
            "conflict_established_belief_ids": proposal_output.metadata.get("conflict_established_belief_ids"),
            "conflict_new_belief_ids": proposal_output.metadata.get("conflict_new_belief_ids"),
        })

    # Run Stage A sequence commit to get final DPA statuses
    seq_res = commit_submission_sequence(tuple(subagent_subs), final_snapshot_evaluation=True)
    final_dpa_statuses = seq_res.final_belief_statuses

    trace_dict = {
        "dpa_trace": seq_res.trace,
        "final_belief_statuses": final_dpa_statuses,
    }
    return ep_a_raw_outputs, ep_a_parsed_actions, final_dpa_statuses, trace_dict
