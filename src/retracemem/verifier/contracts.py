from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceNode,
)


@dataclass(frozen=True)
class RequirementProposal:
    condition: ConditionNode
    dependency_edge: DependencyEdge


class RequirementInducer(Protocol):
    """Infer prerequisite dependencies for a belief.

    Inducers may propose only ``DependencyEdge`` objects with
    ``edge_type == "REQUIRES"``. They do not decide authorization.
    """

    def induce_requirements(
        self,
        belief: BeliefNode,
        evidence_context: tuple[EvidenceNode, ...],
    ) -> list[RequirementProposal]:
        ...


class EvidenceEdgeVerifier(Protocol):
    """Propose typed local evidence-update edges for DPA.

    Verifiers receive candidate condition context and temporal context, and
    return only typed local edges. They do not decide authorization.
    """

    def verify_edges(
        self,
        new_evidence: EvidenceNode,
        candidate_belief: BeliefNode,
        candidate_conditions: tuple[ConditionNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> list[EvidenceEdge]:
        ...
