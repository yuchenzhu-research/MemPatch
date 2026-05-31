"""Predecoded action replay proposer for Stage C.

Allows loading frozen/predecoded revision actions into the evaluation harness.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from retracemem.authorization import EvidenceProposalBatch
from retracemem.schemas import EvidenceEdge, EvidenceEdgeType
from retracemem.evaluation.multiagent.contracts import (
    ApprovedRevisionExemplar,
    FixedCandidateSubmission,
    ProposalPolicyOutput,
    TypedRevisionTarget,
    TypedRevisionProposer,
)


class LearnedReplayProposer(TypedRevisionProposer):
    """Replay proposer that loads pre-decoded outputs for evaluation."""

    proposer_name = "learned_replay"
    policy_variant = "learned_replay"
    provider_kind = "replay"
    model_id = "frozen"

    def __init__(self, predecoded_actions_by_submission: dict[str, list[dict[str, Any]]]) -> None:
        self.predecoded = predecoded_actions_by_submission

    def propose(
        self,
        submission: FixedCandidateSubmission,
        *,
        exemplars: tuple[ApprovedRevisionExemplar, ...] = (),
    ) -> ProposalPolicyOutput:
        sub_id = submission.submission_id
        action_dicts = self.predecoded.get(sub_id, [])

        parsed_actions = []
        edges = []
        errors = []

        try:
            for idx, a in enumerate(action_dicts):
                action_type = a["action_type"]
                target_belief_id = a.get("target_belief_id")
                target_condition_id = a.get("target_condition_id")
                replacement_belief_id = a.get("replacement_belief_id")
                rationale = a.get("rationale", "Replay proposer")
                evidence_ids = tuple(a.get("evidence_ids", [submission.new_evidence_id]))

                target = TypedRevisionTarget(
                    submission_id=sub_id,
                    action_type=action_type,
                    target_belief_id=target_belief_id,
                    target_condition_id=target_condition_id,
                    replacement_belief_id=replacement_belief_id,
                    rationale=rationale,
                    evidence_ids=evidence_ids,
                )
                parsed_actions.append(target)

                if action_type == "NO_REVISION":
                    continue

                target_kind = "belief" if target_belief_id else "condition"
                target_id = target_belief_id or target_condition_id

                edge = EvidenceEdge(
                    edge_id=f"edge_replay_{sub_id}_{idx}",
                    edge_type=EvidenceEdgeType(action_type),
                    evidence_id=str(evidence_ids[0]),
                    target_kind=target_kind,
                    target_id=target_id,
                    verifier="replay_verifier",
                    replacement_belief_id=replacement_belief_id,
                    rationale=rationale,
                )
                edges.append(edge)

        except Exception as exc:
            errors.append(f"Replay processing failed: {exc}")

        proposal_batches = ()
        if not errors and edges:
            proposal_batches = (
                EvidenceProposalBatch(
                    edges=tuple(edges),
                    metadata={"proposer": "LearnedReplayProposer"},
                ),
            )

        return ProposalPolicyOutput(
            example_id=f"ex_{sub_id}",
            submission_id=sub_id,
            policy_variant=self.policy_variant,
            proposal_batches=proposal_batches,
            backbone_model=self.model_id,
            parsing_valid=len(errors) == 0,
            errors=tuple(errors),
            parsed_actions=tuple(parsed_actions),
        )
