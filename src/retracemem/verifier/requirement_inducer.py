from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from retracemem.schemas import BeliefNode, DependencyEdge, EvidenceNode


class ManualRequirementInducer:
    """Return pre-specified typed requirement edges for development fixtures."""

    name = "manual_requirement_inducer"

    def __init__(
        self,
        edges: tuple[DependencyEdge, ...]
        | dict[str, Iterable[DependencyEdge]]
        | None = None,
    ) -> None:
        self._edges_by_belief: dict[str, list[DependencyEdge]] = {}
        if isinstance(edges, dict):
            for belief_id, belief_edges in edges.items():
                for edge in belief_edges:
                    if edge.belief_id != belief_id:
                        raise ValueError(
                            f"manual edge {edge.edge_id} belongs to {edge.belief_id}, "
                            f"not fixture key {belief_id}"
                        )
                    self.register(edge)
        else:
            for edge in edges or ():
                self.register(edge)

    def register(self, edge: DependencyEdge) -> None:
        if edge.edge_type != "REQUIRES":
            raise ValueError("ManualRequirementInducer accepts only REQUIRES dependency edges")
        self._edges_by_belief.setdefault(edge.belief_id, []).append(edge)

    def induce_requirements(
        self,
        belief: BeliefNode,
        evidence_context: tuple[EvidenceNode, ...],
    ) -> list[DependencyEdge]:
        del evidence_context
        return list(self._edges_by_belief.get(belief.belief_id, ()))


class HeuristicRequirementInducer:
    """Deterministic offline requirement inducer for typed DPA smoke tests."""

    name = "heuristic_requirement_inducer"

    _MOBILITY_TERMS = (
        "bike",
        "bicycle",
        "cycling",
        "commute",
        "drive",
        "driving",
        "run",
        "running",
        "tennis",
        "stairs",
        "hiking",
    )
    _SCHEDULE_TERMS = ("available", "availability", "call", "calls", "schedule", "meeting")

    def induce_requirements(
        self,
        belief: BeliefNode,
        evidence_context: tuple[EvidenceNode, ...],
    ) -> list[DependencyEdge]:
        proposition = _normalize(belief.proposition)
        supporting_ids = tuple(e.evidence_id for e in evidence_context)
        edges: list[DependencyEdge] = []

        if _contains_any(proposition, self._MOBILITY_TERMS):
            edges.append(
                self._edge(
                    belief=belief,
                    condition_id=self._condition_id(belief, "mobility"),
                    supporting_ids=supporting_ids,
                    rationale="Mobility-related beliefs require current mobility ability.",
                )
            )

        if _contains_any(proposition, self._SCHEDULE_TERMS):
            edges.append(
                self._edge(
                    belief=belief,
                    condition_id=self._condition_id(belief, "availability"),
                    supporting_ids=supporting_ids,
                    rationale="Scheduling beliefs require current availability.",
                )
            )

        return edges

    def _edge(
        self,
        *,
        belief: BeliefNode,
        condition_id: str,
        supporting_ids: tuple[str, ...],
        rationale: str,
    ) -> DependencyEdge:
        return DependencyEdge(
            edge_id=f"dep:{belief.belief_id}:{condition_id}",
            belief_id=belief.belief_id,
            condition_id=condition_id,
            inducer=self.name,
            edge_type="REQUIRES",
            supporting_evidence_ids=supporting_ids or belief.source_evidence_ids,
            confidence=0.55,
            rationale=rationale,
        )

    @staticmethod
    def _condition_id(belief: BeliefNode, suffix: str) -> str:
        scope_id = str(belief.metadata.get("scope_id") or "global")
        return f"condition:{scope_id}:{suffix}"


def clone_dependency_edge(edge: DependencyEdge, **updates: object) -> DependencyEdge:
    """Copy helper for tests and fixture construction."""

    return replace(edge, **updates)


def _normalize(text: str) -> str:
    return f" {re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()} "


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
