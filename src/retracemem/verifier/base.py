from __future__ import annotations

from typing import Protocol

from retracemem.schemas import Belief, EpisodicEvidence, RelationPrediction


class RelationVerifier(Protocol):
    def verify(
        self,
        new_evidence: EpisodicEvidence,
        candidate_belief: Belief,
        context: dict[str, object] | None = None,
    ) -> RelationPrediction:
        ...
