from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from retracemem.schemas import BeliefNode, ConditionNode, EvidenceNode
from retracemem.memory.belief_store import BeliefStore


@dataclass(frozen=True)
class ImpactCandidate:
    belief: BeliefNode
    conditions: tuple[ConditionNode, ...] = ()


class ImpactCandidateRetriever(Protocol):
    """Protocol for retrieving prior beliefs and their conditions affected by new evidence."""

    def retrieve_impacts(
        self,
        new_evidence: EvidenceNode,
        prior_beliefs: tuple[BeliefNode, ...],
        store: BeliefStore,
        limit: int = 10,
    ) -> list[ImpactCandidate]:
        ...


class QueryBeliefRetriever(Protocol):
    """Protocol for retrieving query-relevant beliefs from a set of beliefs."""

    def retrieve_for_query(
        self,
        query: str,
        beliefs: tuple[BeliefNode, ...],
        limit: int = 10,
    ) -> list[BeliefNode]:
        ...


class ManualImpactCandidateRetriever:
    """Development-only deterministic fixture for impact candidate retrieval.

    Allows manually configuring which prior belief ids are impacted by which evidence.
    It resolves their associated conditions from the BeliefStore dynamically.
    Forbidden for paper main-method retrieval implementations.
    """

    def __init__(self, impact_map: dict[str, list[str]] | None = None) -> None:
        # Maps evidence_id -> list of belief_ids that are candidates for verification
        self.impact_map = impact_map or {}

    def retrieve_impacts(
        self,
        new_evidence: EvidenceNode,
        prior_beliefs: tuple[BeliefNode, ...],
        store: BeliefStore,
        limit: int = 10,
    ) -> list[ImpactCandidate]:
        impacted_ids = self.impact_map.get(new_evidence.evidence_id, [])
        candidates: list[ImpactCandidate] = []
        
        belief_dict = {b.belief_id: b for b in prior_beliefs}
        
        for bid in impacted_ids[:limit]:
            if bid in belief_dict:
                belief = belief_dict[bid]
                # Gather conditions through existing DependencyEdges in store
                # Find dependency edges for this belief
                dep_edges = store.dependencies_of(bid)
                conditions: list[ConditionNode] = []
                for edge in dep_edges:
                    cond = store.get_condition(edge.condition_id)
                    conditions.append(cond)
                candidates.append(ImpactCandidate(belief=belief, conditions=tuple(conditions)))
                
        return candidates


class ManualQueryBeliefRetriever:
    """Development-only deterministic fixture for query belief retrieval.

    Allows manually mapping query text to specific belief ids.
    Forbidden for paper main-method retrieval implementations.
    """

    def __init__(self, query_map: dict[str, list[str]] | None = None) -> None:
        # Maps query -> list of belief_ids
        self.query_map = query_map or {}

    def retrieve_for_query(
        self,
        query: str,
        beliefs: tuple[BeliefNode, ...],
        limit: int = 10,
    ) -> list[BeliefNode]:
        target_ids = self.query_map.get(query, [])
        belief_dict = {b.belief_id: b for b in beliefs}
        
        results: list[BeliefNode] = []
        for bid in target_ids[:limit]:
            if bid in belief_dict:
                results.append(belief_dict[bid])
        return results


class OverlapImpactCandidateRetriever:
    """Production-capable overlap candidate retriever for impact prior beliefs.

    Matches prior beliefs based on token overlap with the new evidence.
    """

    def __init__(self, stopwords: set[str] | None = None) -> None:
        self.stopwords = stopwords or {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "with", "by", "user", "i", "my",
            "me", "you", "your", "he", "she", "it", "we", "they", "this", "that",
        }

    def _tokenize(self, text: str) -> set[str]:
        text_lower = text.lower()
        for char in ".,!?()[]{}'\":;-/":
            text_lower = text_lower.replace(char, " ")
        return set(text_lower.split()) - self.stopwords

    def retrieve_impacts(
        self,
        new_evidence: EvidenceNode,
        prior_beliefs: tuple[BeliefNode, ...],
        store: BeliefStore,
        limit: int = 10,
    ) -> list[ImpactCandidate]:
        ev_words = self._tokenize(new_evidence.text)
        if not ev_words:
            return []

        candidates_with_score = []
        for belief in prior_beliefs:
            b_words = self._tokenize(belief.proposition)
            overlap = ev_words.intersection(b_words)
            if overlap:
                score = len(overlap)
                dep_edges = store.dependencies_of(belief.belief_id)
                conditions = []
                for edge in dep_edges:
                    if store.has_condition(edge.condition_id):
                        conditions.append(store.get_condition(edge.condition_id))
                candidates_with_score.append(
                    (score, ImpactCandidate(belief=belief, conditions=tuple(conditions)))
                )

        candidates_with_score.sort(key=lambda x: x[0], reverse=True)
        return [candidate for _, candidate in candidates_with_score[:limit]]


class OverlapQueryBeliefRetriever:
    """Production-capable query-time belief retriever based on token overlap."""

    def __init__(self, stopwords: set[str] | None = None) -> None:
        self.stopwords = stopwords or {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "with", "by", "user", "i", "my",
            "me", "you", "your", "he", "she", "it", "we", "they", "this", "that",
        }

    def _tokenize(self, text: str) -> set[str]:
        text_lower = text.lower()
        for char in ".,!?()[]{}'\":;-/":
            text_lower = text_lower.replace(char, " ")
        return set(text_lower.split()) - self.stopwords

    def retrieve_for_query(
        self,
        query: str,
        beliefs: tuple[BeliefNode, ...],
        limit: int = 10,
    ) -> list[BeliefNode]:
        q_words = self._tokenize(query)
        if not q_words:
            return list(beliefs[:limit])

        candidates_with_score = []
        for belief in beliefs:
            b_words = self._tokenize(belief.proposition)
            overlap = q_words.intersection(b_words)
            score = len(overlap)
            candidates_with_score.append((score, belief))

        # Sort: first by overlap score descending, then by belief ID for determinism
        candidates_with_score.sort(key=lambda x: (-x[0], x[1].belief_id))
        return [b for _, b in candidates_with_score[:limit]]

