from __future__ import annotations

from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction, RelationType


class PromptRelationVerifier:
    """Placeholder prompt verifier.

    This keeps the interface stable while API-backed prompting is wired in.
    """

    def verify(
        self,
        new_evidence: EpisodicEvidence,
        candidate_belief: Belief,
        context: dict[str, object] | None = None,
    ) -> RelationPrediction:
        del context
        return RelationPrediction(
            relation=RelationType.UNCERTAIN,
            evidence_id=new_evidence.id,
            belief_id=candidate_belief.id,
            rationale="Prompt verifier is not configured yet.",
            confidence=0.0,
        )
