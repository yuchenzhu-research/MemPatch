from __future__ import annotations

from retracemem.extraction.base import BeliefExtractor
from retracemem.schemas import Belief, EpisodicEvidence


class ManualFixtureExtractor(BeliefExtractor):
    """An extractor that returns pre-configured mock beliefs for specific evidence IDs."""

    def __init__(self, fixtures: dict[str, list[Belief]] | None = None) -> None:
        self.fixtures = fixtures or {}

    def register(self, evidence_id: str, beliefs: list[Belief]) -> None:
        self.fixtures[evidence_id] = beliefs

    def extract(self, evidence: EpisodicEvidence) -> list[Belief]:
        # Copy to avoid side-effects
        beliefs = self.fixtures.get(evidence.id, [])
        return [
            Belief(
                id=b.id,
                proposition=b.proposition,
                supported_by=list(b.supported_by),
                status=b.status,
                metadata=dict(b.metadata),
            )
            for b in beliefs
        ]
