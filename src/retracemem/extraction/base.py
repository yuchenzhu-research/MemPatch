from __future__ import annotations

from typing import Protocol

from retracemem.schemas import Belief, EpisodicEvidence


class BeliefExtractor(Protocol):
    """Protocol for extracting beliefs from episodic evidence."""

    def extract(self, evidence: EpisodicEvidence) -> list[Belief]:
        """Extracts atomic beliefs from a piece of episodic evidence."""
        ...
