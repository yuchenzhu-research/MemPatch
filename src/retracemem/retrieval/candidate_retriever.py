from __future__ import annotations

from typing import Protocol

from retracemem.schemas import Belief, EpisodicEvidence


class CandidateRelationRetriever(Protocol):
    """Protocol for filtering prior beliefs relevant to a new piece of evidence."""

    def retrieve_candidates(
        self,
        new_evidence: EpisodicEvidence,
        all_beliefs: list[Belief],
    ) -> list[Belief]:
        """Filters all beliefs down to a candidate subset relevant to the new evidence."""
        ...


class MockCandidateRetriever:
    """Mock retriever that returns pre-configured candidate beliefs for specific evidence IDs."""

    def __init__(self, mapping: dict[str, list[Belief]] | None = None) -> None:
        self.mapping = mapping or {}

    def register(self, evidence_id: str, candidates: list[Belief]) -> None:
        self.mapping[evidence_id] = candidates

    def retrieve_candidates(
        self,
        new_evidence: EpisodicEvidence,
        all_beliefs: list[Belief],
    ) -> list[Belief]:
        if new_evidence.id in self.mapping:
            # Only return candidates that actually exist in the belief list
            exist_ids = {b.id for b in all_beliefs}
            return [b for b in self.mapping[new_evidence.id] if b.id in exist_ids]
        return all_beliefs


class SimpleOverlapRetriever:
    """A baseline candidate retriever based on word token overlap."""

    def __init__(self, stopwords: set[str] | None = None) -> None:
        self.stopwords = stopwords or {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "user",
        }

    def retrieve_candidates(
        self,
        new_evidence: EpisodicEvidence,
        all_beliefs: list[Belief],
    ) -> list[Belief]:
        # Normalize and filter words
        ev_text = new_evidence.text.lower()
        # Remove common punctuation
        for char in ".,!?()[]{}":
            ev_text = ev_text.replace(char, " ")
        ev_words = set(ev_text.split()) - self.stopwords

        if not ev_words:
            return all_beliefs

        candidates = []
        for belief in all_beliefs:
            b_text = belief.proposition.lower()
            for char in ".,!?()[]{}":
                b_text = b_text.replace(char, " ")
            b_words = set(b_text.split()) - self.stopwords
            if ev_words.intersection(b_words):
                candidates.append(belief)
        return candidates
