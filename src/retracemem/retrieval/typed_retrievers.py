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
                    try:
                        cond = store.get_condition(edge.condition_id)
                        conditions.append(cond)
                    except KeyError:
                        # If condition is not yet stored, ignore or fail.
                        # Per DPA design, dependency edges require condition nodes to exist.
                        pass
                candidates.append(ImpactCandidate(belief=belief, conditions=tuple(conditions)))
                
        return candidates


class ManualQueryBeliefRetriever:
    """Development-only deterministic fixture for query belief retrieval.

    Allows manually mapping query text to specific belief ids.
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
