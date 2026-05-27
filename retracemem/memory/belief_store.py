from __future__ import annotations

from retracemem.schemas import Belief, RelationPrediction


class BeliefStore:
    """Open-text belief nodes plus local relation predictions."""

    def __init__(self) -> None:
        self._beliefs: dict[str, Belief] = {}
        self._relations: list[RelationPrediction] = []

    def add_belief(self, belief: Belief) -> None:
        if not belief.id:
            raise ValueError("belief id is required")
        if not belief.proposition:
            raise ValueError("belief proposition is required")
        if belief.id in self._beliefs:
            raise ValueError(f"belief already exists: {belief.id}")
        self._beliefs[belief.id] = belief

    def get_belief(self, belief_id: str) -> Belief:
        if belief_id not in self._beliefs:
            raise KeyError(f"unknown belief: {belief_id}")
        return self._beliefs[belief_id]

    def all_beliefs(self) -> list[Belief]:
        return list(self._beliefs.values())

    def add_relation(self, relation: RelationPrediction) -> None:
        self._relations.append(relation)

    def relations_for_belief(self, belief_id: str) -> list[RelationPrediction]:
        return [
            relation
            for relation in self._relations
            if relation.belief_id == belief_id or relation.target_belief_id == belief_id
        ]

    def all_relations(self) -> list[RelationPrediction]:
        return list(self._relations)

    def has_belief(self, belief_id: str) -> bool:
        return belief_id in self._beliefs

    def __len__(self) -> int:
        return len(self._beliefs)
