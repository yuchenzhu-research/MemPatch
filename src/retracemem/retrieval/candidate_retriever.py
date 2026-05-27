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
            exist_ids = {b.id for b in all_beliefs}
            return [b for b in self.mapping[new_evidence.id] if b.id in exist_ids]
        return all_beliefs


class SimpleOverlapRetriever:
    """A highly optimized candidate retriever based on word token overlap with caching."""

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
        # In-memory cache for normalized belief words to prevent repetitive O(N) string processing
        self._belief_words_cache: dict[str, set[str]] = {}

    def retrieve_candidates(
        self,
        new_evidence: EpisodicEvidence,
        all_beliefs: list[Belief],
    ) -> list[Belief]:
        ev_text = new_evidence.text.lower()
        for char in ".,!?()[]{}":
            ev_text = ev_text.replace(char, " ")
        ev_words = set(ev_text.split()) - self.stopwords

        if not ev_words:
            # If evidence has no keywords, return empty list to avoid verifying everything
            return []

        candidates_with_score = []
        for belief in all_beliefs:
            # Retrieve or calculate normalized words from cache
            if belief.id not in self._belief_words_cache:
                b_text = belief.proposition.lower()
                for char in ".,!?()[]{}":
                    b_text = b_text.replace(char, " ")
                b_words = set(b_text.split()) - self.stopwords
                self._belief_words_cache[belief.id] = b_words

            b_words = self._belief_words_cache[belief.id]
            overlap = ev_words.intersection(b_words)
            if overlap:
                candidates_with_score.append((len(overlap), belief))

        # Sort candidates by overlap size descending, and limit to top 5
        candidates_with_score.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in candidates_with_score[:5]]
