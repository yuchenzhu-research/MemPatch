from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
from retracemem.schemas import BeliefNode, EvidenceNode, EvidenceEdge
from retracemem.multiagent.contracts import SubagentMemorySubmission


@dataclass(frozen=True)
class DownstreamTask:
    task_id: str
    query: str
    expected_answer_or_action: str | None = None
    relevant_belief_ids: Tuple[str, ...] = ()
    protected_belief_ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "expected_answer_or_action": self.expected_answer_or_action,
            "relevant_belief_ids": list(self.relevant_belief_ids),
            "protected_belief_ids": list(self.protected_belief_ids),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GoldSnapshotExpectation:
    belief_statuses: Dict[str, str]
    required_authorized_belief_ids: Tuple[str, ...] = ()
    forbidden_authorized_belief_ids: Tuple[str, ...] = ()
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_statuses": self.belief_statuses,
            "required_authorized_belief_ids": list(self.required_authorized_belief_ids),
            "forbidden_authorized_belief_ids": list(self.forbidden_authorized_belief_ids),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class MultiAgentMemoryEpisode:
    episode_id: str
    domain: str
    failure_type: str
    subagent_roles: Tuple[str, ...]
    submissions: Tuple[SubagentMemorySubmission, ...]
    downstream_tasks: Tuple[DownstreamTask, ...]
    gold_snapshot: GoldSnapshotExpectation
    stress_factors: Dict[str, Any] = field(default_factory=dict)
    split: str = "development_only"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "failure_type": self.failure_type,
            "subagent_roles": list(self.subagent_roles),
            "submissions": [self._submission_to_dict(s) for s in self.submissions],
            "downstream_tasks": [t.to_dict() for t in self.downstream_tasks],
            "gold_snapshot": self.gold_snapshot.to_dict(),
            "stress_factors": self.stress_factors,
            "split": self.split,
            "metadata": self.metadata,
        }

    @staticmethod
    def _submission_to_dict(s: SubagentMemorySubmission) -> Dict[str, Any]:
        return {
            "submission_id": s.submission_id,
            "producer_id": s.producer_id,
            "producer_role": s.producer_role,
            "parent_snapshot_id": s.parent_snapshot_id,
            "observed_at": s.observed_at,
            "instance_id": s.instance_id,
            "query_id": s.query_id,
            "query": s.query,
            "new_evidence_id": s.new_evidence_id,
            "task_id": s.task_id,
            "metadata": s.metadata,
        }


@dataclass(frozen=True)
class RevisionEventResult:
    submission_id: str
    producer_id: str
    producer_role: str
    parent_snapshot_id: str
    next_snapshot_id: str | None
    belief_statuses: Dict[str, str]
    trace_available: bool
    latency_ms: float | None = None
    calls: int | None = None
    tokens: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "producer_id": self.producer_id,
            "producer_role": self.producer_role,
            "parent_snapshot_id": self.parent_snapshot_id,
            "next_snapshot_id": self.next_snapshot_id,
            "belief_statuses": self.belief_statuses,
            "trace_available": self.trace_available,
            "latency_ms": self.latency_ms,
            "calls": self.calls,
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class EpisodeMethodResult:
    episode_id: str
    domain: str
    failure_type: str
    method_name: str
    final_belief_statuses: Dict[str, str]
    revision_events: Tuple[RevisionEventResult, ...]
    task_predictions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "failure_type": self.failure_type,
            "method_name": self.method_name,
            "final_belief_statuses": self.final_belief_statuses,
            "revision_events": [ev.to_dict() for ev in self.revision_events],
            "task_predictions": self.task_predictions,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ExperimentRunManifest:
    run_id: str
    split: str
    methods: Tuple[str, ...]
    episode_ids: Tuple[str, ...]
    model_config: Dict[str, Any]
    prompt_hashes: Dict[str, str]
    code_commit_sha: str
    created_at: str
    mode: str  # offline_replay / smoke_live / official_frozen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "split": self.split,
            "methods": list(self.methods),
            "episode_ids": list(self.episode_ids),
            "model_config": self.model_config,
            "prompt_hashes": self.prompt_hashes,
            "code_commit_sha": self.code_commit_sha,
            "created_at": self.created_at,
            "mode": self.mode,
        }


# ===========================================================================
# Packet 4B: Fixed-Candidate Evaluation Contracts
# ===========================================================================


@dataclass(frozen=True)
class FixedCandidateSubmission:
    """A submission with fixed candidate beliefs and edges.

    Unlike SubagentMemorySubmission, this is purely an evaluation input:
    the candidate beliefs and candidate edges are given, and the method
    decides which to authorize.
    """
    submission_id: str
    producer_id: str
    producer_role: str
    timestamp: str
    evidence_context: Tuple[EvidenceNode, ...]
    candidate_beliefs: Tuple[BeliefNode, ...]
    candidate_edges: Tuple[EvidenceEdge, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "producer_id": self.producer_id,
            "producer_role": self.producer_role,
            "timestamp": self.timestamp,
            "evidence_context": [
                {"evidence_id": e.evidence_id, "text": e.text, "timestamp": e.timestamp}
                for e in self.evidence_context
            ],
            "candidate_beliefs": [
                {"belief_id": b.belief_id, "proposition": b.proposition}
                for b in self.candidate_beliefs
            ],
            "candidate_edges": [
                {
                    "edge_id": e.edge_id,
                    "edge_type": e.edge_type.value,
                    "target_id": e.target_id,
                    "replacement_belief_id": e.replacement_belief_id,
                }
                for e in self.candidate_edges
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MethodDecisionRecord:
    """Records the decision a method made for a specific belief.

    Separates model outputs from episode inputs so we can compare
    methods on identical candidate sets.
    """
    belief_id: str
    decision: str  # AUTHORIZE, REJECT, SUPERSEDE, DEFER
    replacement_belief_id: str | None = None
    confidence: float | None = None
    rationale: str = ""
    model_call_trace_id: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "decision": self.decision,
            "replacement_belief_id": self.replacement_belief_id,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "model_call_trace_id": self.model_call_trace_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FixedCandidateEpisode:
    """An evaluation episode with fixed candidate inputs.

    All methods receive exactly the same candidate beliefs and edges,
    enabling controlled comparison.
    """
    episode_id: str
    domain: str
    failure_type: str
    subagent_roles: Tuple[str, ...]
    submissions: Tuple[FixedCandidateSubmission, ...]
    downstream_tasks: Tuple[DownstreamTask, ...]
    gold_snapshot: GoldSnapshotExpectation
    # Replay fixture: pre-authored decisions for offline replay methods
    replay_decisions: Tuple[MethodDecisionRecord, ...] = ()
    stress_factors: Dict[str, Any] = field(default_factory=dict)
    protocol_mode: str = "oracle_edge_replay"
    proposal_source: str = "hand_authored_development"
    split: str = "development_only"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "failure_type": self.failure_type,
            "subagent_roles": list(self.subagent_roles),
            "submissions": [s.to_dict() for s in self.submissions],
            "downstream_tasks": [t.to_dict() for t in self.downstream_tasks],
            "gold_snapshot": self.gold_snapshot.to_dict(),
            "replay_decisions": [d.to_dict() for d in self.replay_decisions],
            "stress_factors": self.stress_factors,
            "protocol_mode": self.protocol_mode,
            "proposal_source": self.proposal_source,
            "split": self.split,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FixedCandidateEpisodeMethodResult:
    """Result of running a method on a FixedCandidateEpisode."""
    episode_id: str
    domain: str
    failure_type: str
    method_name: str
    protocol_mode: str
    proposal_source: str
    final_belief_statuses: Dict[str, str]
    decisions: Tuple[MethodDecisionRecord, ...]
    task_predictions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "failure_type": self.failure_type,
            "method_name": self.method_name,
            "protocol_mode": self.protocol_mode,
            "proposal_source": self.proposal_source,
            "final_belief_statuses": self.final_belief_statuses,
            "decisions": [d.to_dict() for d in self.decisions],
            "task_predictions": self.task_predictions,
            "metadata": self.metadata,
        }

