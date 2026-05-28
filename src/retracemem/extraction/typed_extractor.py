from __future__ import annotations

from typing import Protocol
from retracemem.schemas import BeliefNode, EvidenceNode


class TypedBeliefExtractor(Protocol):
    """Protocol for extracting BeliefNodes from an EvidenceNode."""

    def extract(
        self,
        evidence: EvidenceNode,
        scope_id: str,
    ) -> list[BeliefNode]:
        """Extract BeliefNodes from the given EvidenceNode."""
        ...


class ManualTypedBeliefExtractor:
    """Development-only deterministic fixture for typed belief extraction.

    This class extracts predefined beliefs mapped to specific evidence IDs
    strictly for testing the DPA pipeline.
    """

    def __init__(self, mappings: dict[str, list[BeliefNode]] | None = None) -> None:
        self.mappings = mappings or {}

    def extract(
        self,
        evidence: EvidenceNode,
        scope_id: str,
    ) -> list[BeliefNode]:
        if not scope_id:
            raise ValueError("scope_id is required and cannot be empty")

        beliefs = self.mappings.get(evidence.evidence_id, [])
        for belief in beliefs:
            # Grounding check: evidence.evidence_id must be in belief.source_evidence_ids.
            if evidence.evidence_id not in belief.source_evidence_ids:
                raise ValueError(
                    f"Grounding violation: Belief {belief.belief_id} does not list "
                    f"source evidence {evidence.evidence_id} in source_evidence_ids {belief.source_evidence_ids}"
                )

        return beliefs
