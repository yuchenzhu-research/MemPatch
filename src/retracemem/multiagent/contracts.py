from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
)
from retracemem.authorization import AuthorizationResult, EvidenceProposalBatch


@dataclass(frozen=True)
class SubagentMemorySubmission:
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
    proposal_batches: tuple[EvidenceProposalBatch, ...] = ()
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SharedMemoryCommitResult:
    submission_id: str
    producer_id: str
    producer_role: str
    parent_snapshot_id: str
    next_snapshot_id: str
    authorization_result: AuthorizationResult
    commit_trace: dict[str, Any]


@dataclass(frozen=True)
class SharedMemorySnapshotResult:
    initial_snapshot_id: str
    final_snapshot_id: str
    submission_results: tuple[SharedMemoryCommitResult, ...]
    final_belief_statuses: dict[str, str]
    final_authorized_belief_ids: tuple[str, ...]
    final_excluded_belief_ids: tuple[str, ...]
    trace: dict[str, Any]
