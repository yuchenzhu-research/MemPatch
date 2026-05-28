from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge, EvidenceNode
from retracemem.verifier.contracts import RequirementProposal


class ManualRequirementInducer:
    """Return pre-specified typed requirement edges for development fixtures."""

    name = "manual_requirement_inducer"

    def __init__(
        self,
        proposals: tuple[RequirementProposal, ...]
        | dict[str, Iterable[RequirementProposal]]
        | None = None,
    ) -> None:
        self._proposals_by_belief: dict[str, list[RequirementProposal]] = {}
        if isinstance(proposals, dict):
            for belief_id, belief_proposals in proposals.items():
                for proposal in belief_proposals:
                    if proposal.dependency_edge.belief_id != belief_id:
                        raise ValueError(
                            f"manual proposal {proposal.dependency_edge.edge_id} belongs to {proposal.dependency_edge.belief_id}, "
                            f"not fixture key {belief_id}"
                        )
                    self.register(proposal)
        else:
            for proposal in proposals or ():
                self.register(proposal)

    def register(self, proposal: RequirementProposal) -> None:
        if proposal.dependency_edge.edge_type != "REQUIRES":
            raise ValueError("ManualRequirementInducer accepts only REQUIRES dependency edges")
        self._proposals_by_belief.setdefault(proposal.dependency_edge.belief_id, []).append(proposal)

    def induce_requirements(
        self,
        belief: BeliefNode,
        evidence_context: tuple[EvidenceNode, ...],
    ) -> list[RequirementProposal]:
        del evidence_context
        proposals = self._proposals_by_belief.get(belief.belief_id, [])
        for proposal in proposals:
            if proposal.dependency_edge.condition_id != proposal.condition.condition_id:
                raise ValueError(
                    f"Mismatched condition_id: dependency_edge has '{proposal.dependency_edge.condition_id}', "
                    f"but condition has '{proposal.condition.condition_id}'"
                )
            if proposal.dependency_edge.belief_id != belief.belief_id:
                raise ValueError(
                    f"Mismatched belief_id: dependency_edge has '{proposal.dependency_edge.belief_id}', "
                    f"but requested belief has '{belief.belief_id}'"
                )
        return list(proposals)


class HeuristicRequirementInducer:
    """Development-only deterministic fixture for requirement induction.

    This class serves as a development-only deterministic fixture for offline
    validation and smoke testing, not for production use.
    """

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
    ) -> list[RequirementProposal]:
        proposition = _normalize(belief.proposition)
        supporting_ids = tuple(e.evidence_id for e in evidence_context)
        proposals: list[RequirementProposal] = []

        scope_id = belief.metadata.get("scope_id")
        if scope_id is None:
            raise ValueError("scope_id is missing in belief.metadata")
        scope_id = str(scope_id)

        if _contains_any(proposition, self._MOBILITY_TERMS):
            cond_id = f"condition:{scope_id}:mobility"
            condition = ConditionNode(
                condition_id=cond_id,
                scope_id=scope_id,
                text="Mobility-related beliefs require current mobility ability.",
            )
            edge = DependencyEdge(
                edge_id=f"dep:{belief.belief_id}:{cond_id}",
                belief_id=belief.belief_id,
                condition_id=cond_id,
                inducer=self.name,
                edge_type="REQUIRES",
                supporting_evidence_ids=supporting_ids or belief.source_evidence_ids,
                confidence=0.55,
                rationale="Mobility-related beliefs require current mobility ability.",
            )
            proposals.append(RequirementProposal(condition=condition, dependency_edge=edge))

        if _contains_any(proposition, self._SCHEDULE_TERMS):
            cond_id = f"condition:{scope_id}:availability"
            condition = ConditionNode(
                condition_id=cond_id,
                scope_id=scope_id,
                text="Scheduling beliefs require current availability.",
            )
            edge = DependencyEdge(
                edge_id=f"dep:{belief.belief_id}:{cond_id}",
                belief_id=belief.belief_id,
                condition_id=cond_id,
                inducer=self.name,
                edge_type="REQUIRES",
                supporting_evidence_ids=supporting_ids or belief.source_evidence_ids,
                confidence=0.55,
                rationale="Scheduling beliefs require current availability.",
            )
            proposals.append(RequirementProposal(condition=condition, dependency_edge=edge))

        return proposals


def clone_dependency_edge(edge: DependencyEdge, **updates: object) -> DependencyEdge:
    """Copy helper for tests and fixture construction."""

    return replace(edge, **updates)


def _normalize(text: str) -> str:
    return f" {re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()} "


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
