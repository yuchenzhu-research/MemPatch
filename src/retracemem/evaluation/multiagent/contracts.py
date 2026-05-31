from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
from retracemem.schemas import BeliefNode, EvidenceNode, EvidenceEdge, ConditionNode, DependencyEdge
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
# Packet 4C: Stage C and Repaired E1 Evaluation Contracts
# ===========================================================================

@dataclass(frozen=True)
class FixedCandidateSubmission:
    """A submission with fixed candidate beliefs but without edges.
    
    This is purely a fair evaluation input where candidate edges/verdicts
    are stripped.
    """
    submission_id: str
    producer_id: str
    producer_role: str
    task_id: str | None
    parent_snapshot_id: str
    observed_at: str
    instance_id: str
    query_id: str
    query: str

    # Method-visible semantic input only
    evidence_context: Tuple[EvidenceNode, ...]
    new_evidence_id: str
    candidate_beliefs: Tuple[BeliefNode, ...]
    candidate_replacement_beliefs: Tuple[BeliefNode, ...] = ()
    candidate_conditions_by_belief: Tuple[Tuple[str, Tuple[ConditionNode, ...]], ...] = ()
    dependency_edges_by_belief: Tuple[Tuple[str, Tuple[DependencyEdge, ...]], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "producer_id": self.producer_id,
            "producer_role": self.producer_role,
            "task_id": self.task_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "observed_at": self.observed_at,
            "instance_id": self.instance_id,
            "query_id": self.query_id,
            "query": self.query,
            "evidence_context": [
                {"evidence_id": e.evidence_id, "text": e.text, "timestamp": e.timestamp}
                for e in self.evidence_context
            ],
            "new_evidence_id": self.new_evidence_id,
            "candidate_beliefs": [
                {"belief_id": b.belief_id, "proposition": b.proposition}
                for b in self.candidate_beliefs
            ],
            "candidate_replacement_beliefs": [
                {"belief_id": b.belief_id, "proposition": b.proposition}
                for b in self.candidate_replacement_beliefs
            ],
            "candidate_conditions_by_belief": [
                (bid, [
                    {"condition_id": c.condition_id, "scope_id": c.scope_id, "text": c.text}
                    for c in conds
                ])
                for bid, conds in self.candidate_conditions_by_belief
            ],
            "dependency_edges_by_belief": [
                (bid, [
                    {"edge_id": d.edge_id, "belief_id": d.belief_id, "condition_id": d.condition_id}
                    for d in deps
                ])
                for bid, deps in self.dependency_edges_by_belief
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MethodDecisionRecord:
    """Records the decision a method made for a specific belief."""
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
class FixedCandidateInputEpisode:
    """An evaluation episode with fixed candidate inputs, stripped of gold/replay data."""
    episode_id: str
    domain: str
    failure_type_public_or_controlled: str
    subagent_roles: Tuple[str, ...]
    submissions: Tuple[FixedCandidateSubmission, ...]
    downstream_tasks: Tuple[DownstreamTask, ...]
    stress_factors: Dict[str, Any] = field(default_factory=dict)
    split: str = "development_only"
    protocol_mode: str = "fixed_candidate_revision"
    proposal_source: str = "hand_authored_development"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "domain": self.domain,
            "failure_type_public_or_controlled": self.failure_type_public_or_controlled,
            "subagent_roles": list(self.subagent_roles),
            "submissions": [s.to_dict() for s in self.submissions],
            "downstream_tasks": [t.to_dict() for t in self.downstream_tasks],
            "stress_factors": self.stress_factors,
            "split": self.split,
            "protocol_mode": self.protocol_mode,
            "proposal_source": self.proposal_source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TypedRevisionTarget:
    """Evaluation label: targeted revision action for training/gold scoring."""
    submission_id: str
    action_type: str  # SUPERSEDES / BLOCKS / RELEASES / UNCERTAIN / REAFFIRMS / NO_REVISION
    target_belief_id: str | None = None
    target_condition_id: str | None = None
    replacement_belief_id: str | None = None
    rationale: str = ""
    evidence_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "action_type": self.action_type,
            "target_belief_id": self.target_belief_id,
            "target_condition_id": self.target_condition_id,
            "replacement_belief_id": self.replacement_belief_id,
            "rationale": self.rationale,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class FixedCandidateGoldRecord:
    """Evaluator-only sidecar holding gold expectations for an episode."""
    episode_id: str
    gold_snapshot: GoldSnapshotExpectation
    gold_typed_targets: Tuple[TypedRevisionTarget, ...] = ()
    failure_type: str | None = None
    representable_by_core_actions: bool = True
    minimum_core_action_count: int | None = None
    requires_multi_action: bool = False
    missing_extension: str = "NONE"
    missing_extension_notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "gold_snapshot": self.gold_snapshot.to_dict(),
            "gold_typed_targets": [t.to_dict() for t in self.gold_typed_targets],
            "failure_type": self.failure_type,
            "representable_by_core_actions": self.representable_by_core_actions,
            "minimum_core_action_count": self.minimum_core_action_count,
            "requires_multi_action": self.requires_multi_action,
            "missing_extension": self.missing_extension,
            "missing_extension_notes": self.missing_extension_notes,
            "metadata": self.metadata,
        }


from retracemem import EvidenceProposalBatch

def _proposal_batch_to_dict(b: EvidenceProposalBatch) -> Dict[str, Any]:
    return {
        "model_call_trace_id": b.model_call_trace_id,
        "source_belief_id": b.source_belief_id,
        "edges": [
            {
                "edge_id": e.edge_id,
                "edge_type": e.edge_type.value if hasattr(e.edge_type, "value") else str(e.edge_type),
                "evidence_id": e.evidence_id,
                "target_kind": e.target_kind,
                "target_id": e.target_id,
                "verifier": e.verifier,
                "replacement_belief_id": e.replacement_belief_id,
            }
            for e in b.edges
        ],
        "metadata": b.metadata,
    }


@dataclass(frozen=True)
class MethodDecisionArtifact:
    """Evaluator sidecar that supplies replayed decisions/proposals to baseline methods."""
    episode_id: str
    method_name: str
    protocol_mode: str
    proposal_source: str       # oracle_replay / prompt / sft / reward_refined
    backbone_model: str | None
    typed_proposal_batches_by_submission: Tuple[
        Tuple[str, Tuple[EvidenceProposalBatch, ...]], ...
    ] = ()
    direct_verdicts_by_submission: Tuple[
        Tuple[str, Tuple[MethodDecisionRecord, ...]], ...
    ] = ()
    model_call_trace_ids: Tuple[str, ...] = ()
    prompt_hash: str | None = None
    checkpoint_id: str | None = None
    calls: int = 0
    tokens: int | None = None
    latency_ms: float | None = None
    scientific_status: str = "pipeline_validation_only"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "method_name": self.method_name,
            "protocol_mode": self.protocol_mode,
            "proposal_source": self.proposal_source,
            "backbone_model": self.backbone_model,
            "typed_proposal_batches_by_submission": [
                (sub_id, [_proposal_batch_to_dict(b) for b in batches])
                for sub_id, batches in self.typed_proposal_batches_by_submission
            ],
            "direct_verdicts_by_submission": [
                (sub_id, [v.to_dict() for v in verdicts])
                for sub_id, verdicts in self.direct_verdicts_by_submission
            ],
            "model_call_trace_ids": list(self.model_call_trace_ids),
            "prompt_hash": self.prompt_hash,
            "checkpoint_id": self.checkpoint_id,
            "calls": self.calls,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "scientific_status": self.scientific_status,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FixedCandidateEpisodeMethodResult:
    """Result of running a method on a FixedCandidateInputEpisode."""
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


# ===========================================================================
# Stage C Training Contracts
# ===========================================================================

@dataclass(frozen=True)
class StageCTrainingExample:
    """Input-label pair used for SFT or alignment training of the revision policy."""
    example_id: str
    episode_id: str
    submission_id: str
    method_visible_input: FixedCandidateSubmission
    targets: Tuple[TypedRevisionTarget, ...]
    split: str                   # train / dev / frozen_test
    domain: str
    failure_type: str
    label_source: str            # human_authored / human_verified_generated
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "episode_id": self.episode_id,
            "submission_id": self.submission_id,
            "method_visible_input": self.method_visible_input.to_dict(),
            "targets": [t.to_dict() for t in self.targets],
            "split": self.split,
            "domain": self.domain,
            "failure_type": self.failure_type,
            "label_source": self.label_source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ProposalPolicyOutput:
    """Revision action batches generated by a policy variant."""
    example_id: str
    submission_id: str
    policy_variant: str          # prompt / sft / reward_refined / oracle_replay
    proposal_batches: Tuple[EvidenceProposalBatch, ...]
    backbone_model: str | None = None
    checkpoint_id: str | None = None
    prompt_hash: str | None = None
    parsing_valid: bool = True
    errors: Tuple[str, ...] = ()
    calls: int = 0
    tokens: int | None = None
    latency_ms: float | None = None
    parsed_actions: Tuple[TypedRevisionTarget, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "submission_id": self.submission_id,
            "policy_variant": self.policy_variant,
            "proposal_batches": [_proposal_batch_to_dict(b) for b in self.proposal_batches],
            "backbone_model": self.backbone_model,
            "checkpoint_id": self.checkpoint_id,
            "prompt_hash": self.prompt_hash,
            "parsing_valid": self.parsing_valid,
            "errors": list(self.errors),
            "calls": self.calls,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "parsed_actions": [a.to_dict() for a in self.parsed_actions],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RewardBreakdown:
    """Academic reward components used for Stage C optimization/analysis."""
    authorization_reward: float
    downstream_task_reward: float
    stale_penalty: float
    scope_expansion_penalty: float
    conflict_penalty: float
    recovery_reward: float
    uncertainty_reward: float
    total_reward: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authorization_reward": self.authorization_reward,
            "downstream_task_reward": self.downstream_task_reward,
            "stale_penalty": self.stale_penalty,
            "scope_expansion_penalty": self.scope_expansion_penalty,
            "conflict_penalty": self.conflict_penalty,
            "recovery_reward": self.recovery_reward,
            "uncertainty_reward": self.uncertainty_reward,
            "total_reward": self.total_reward,
        }


from typing import Protocol

class TypedRevisionPolicy(Protocol):
    """Protocol for Stage C learned models proposing typed edges."""
    policy_variant: str
    def propose(self, example_or_submission: FixedCandidateSubmission) -> ProposalPolicyOutput:
        ...


@dataclass(frozen=True)
class ApprovedRevisionExemplar:
    exemplar_id: str
    source_episode_id: str
    domain: str
    failure_type: str
    method_visible_input: FixedCandidateSubmission
    approved_typed_actions: tuple[TypedRevisionTarget, ...]
    reviewer: str
    review_manifest_hash: str
    training_or_icl_eligibility: str


class TypedRevisionProposer(Protocol):
    proposer_name: str
    policy_variant: str
    provider_kind: str | None
    model_id: str | None

    def propose(
        self,
        submission: FixedCandidateSubmission,
        *,
        exemplars: tuple[ApprovedRevisionExemplar, ...] = (),
    ) -> ProposalPolicyOutput:
        ...


