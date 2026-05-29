from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
)
from retracemem.methods.contracts import SharedCandidateView
from retracemem.multiagent.contracts import SubagentMemorySubmission
from experiments.stale_adapter import assert_no_evaluation_leakage


@dataclass(frozen=True)
class CupMemRevisionCandidate:
    submission_id: str
    producer_id: str
    producer_role: str
    parent_snapshot_id: str
    observed_at: str
    instance_id: str
    query_id: str
    query: str
    evidence_context: tuple[EvidenceNode, ...]
    new_evidence_id: str
    candidate_beliefs: tuple[BeliefNode, ...]
    candidate_replacement_beliefs: tuple[BeliefNode, ...] = ()
    candidate_conditions_by_belief: tuple[tuple[str, tuple[ConditionNode, ...]], ...] = ()
    dependency_edges_by_belief: tuple[tuple[str, tuple[DependencyEdge, ...]], ...] = ()
    upstream_trace: dict[str, Any] = field(default_factory=dict)


def map_cupmem_candidate_to_subagent_submission(
    candidate: CupMemRevisionCandidate,
    *,
    proposal_batches: tuple[Any, ...] = (),
) -> SubagentMemorySubmission:
    """Truthful minimal adapter mapping a CupMemRevisionCandidate into a SubagentMemorySubmission."""
    # Ensure no gold fields are present in any of the candidate inputs
    assert_no_evaluation_leakage(candidate)
    
    return SubagentMemorySubmission(
        submission_id=candidate.submission_id,
        producer_id=candidate.producer_id,
        producer_role=candidate.producer_role,
        parent_snapshot_id=candidate.parent_snapshot_id,
        observed_at=candidate.observed_at,
        instance_id=candidate.instance_id,
        query_id=candidate.query_id,
        query=candidate.query,
        evidence_context=candidate.evidence_context,
        new_evidence_id=candidate.new_evidence_id,
        candidate_beliefs=candidate.candidate_beliefs,
        candidate_replacement_beliefs=candidate.candidate_replacement_beliefs,
        candidate_conditions_by_belief=candidate.candidate_conditions_by_belief,
        dependency_edges_by_belief=candidate.dependency_edges_by_belief,
        proposal_batches=proposal_batches,
        task_id=candidate.upstream_trace.get("task_id"),
        metadata=candidate.upstream_trace,
    )
