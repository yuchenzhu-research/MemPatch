from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from retracemem.retrieval.typed_retrievers import ImpactCandidate
from retracemem.schemas import BeliefNode, EvidenceEdge, EvidenceNode
from retracemem.verifier.contracts import BatchedEvidenceEdgeVerifier, EvidenceEdgeVerifier


@dataclass(frozen=True)
class EvidenceEdgeProposalResult:
    edges: tuple[EvidenceEdge, ...]
    verifier_calls: int
    batch_count: int
    execution_mode: str
    latency_ms: float


class EvidenceEdgeProposalStrategy(Protocol):
    def propose_edges(
        self,
        *,
        new_evidence: EvidenceNode,
        impact_candidates: tuple[ImpactCandidate, ...],
        candidate_replacement_beliefs: tuple[BeliefNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> EvidenceEdgeProposalResult:
        ...


class PerBeliefEvidenceEdgeProposalStrategy:
    def __init__(self, verifier: EvidenceEdgeVerifier) -> None:
        self.verifier = verifier

    def propose_edges(
        self,
        *,
        new_evidence: EvidenceNode,
        impact_candidates: tuple[ImpactCandidate, ...],
        candidate_replacement_beliefs: tuple[BeliefNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> EvidenceEdgeProposalResult:
        started = time.perf_counter()
        edges: list[EvidenceEdge] = []
        for candidate in impact_candidates:
            edges.extend(self.verifier.verify_edges(
                new_evidence=new_evidence,
                candidate_belief=candidate.belief,
                candidate_replacement_beliefs=candidate_replacement_beliefs,
                candidate_conditions=candidate.conditions,
                temporal_context=temporal_context,
            ))
        return EvidenceEdgeProposalResult(
            edges=tuple(edges),
            verifier_calls=len(impact_candidates),
            batch_count=len(impact_candidates),
            execution_mode="per_belief",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


class BatchedEvidenceEdgeProposalStrategy:
    def __init__(self, verifier: BatchedEvidenceEdgeVerifier, max_batch_beliefs: int = 8) -> None:
        if max_batch_beliefs < 1:
            raise ValueError("max_batch_beliefs must be >= 1")
        self.verifier = verifier
        self.max_batch_beliefs = max_batch_beliefs

    def propose_edges(
        self,
        *,
        new_evidence: EvidenceNode,
        impact_candidates: tuple[ImpactCandidate, ...],
        candidate_replacement_beliefs: tuple[BeliefNode, ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> EvidenceEdgeProposalResult:
        started = time.perf_counter()
        edges: list[EvidenceEdge] = []
        batch_count = 0
        for batch in self._batches(impact_candidates):
            batch_count += 1
            raw_edges = self.verifier.verify_edges_batch(
                new_evidence=new_evidence,
                candidate_beliefs=tuple(candidate.belief for candidate in batch),
                candidate_replacement_beliefs=candidate_replacement_beliefs,
                candidate_conditions_by_belief=tuple(
                    (candidate.belief.belief_id, candidate.conditions)
                    for candidate in batch
                ),
                temporal_context=temporal_context,
            )
            edges.extend(tuple(getattr(raw_edges, "proposed_edges", raw_edges)))
        return EvidenceEdgeProposalResult(
            edges=tuple(edges),
            verifier_calls=batch_count,
            batch_count=batch_count,
            execution_mode="batched",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _batches(self, impact_candidates: tuple[ImpactCandidate, ...]) -> tuple[tuple[ImpactCandidate, ...], ...]:
        return tuple(
            impact_candidates[index:index + self.max_batch_beliefs]
            for index in range(0, len(impact_candidates), self.max_batch_beliefs)
        )
