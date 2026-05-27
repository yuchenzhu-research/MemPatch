from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BeliefStatus(str, Enum):
    HISTORICAL = "historical"
    AUTHORIZED = "authorized"
    BLOCKED = "blocked"
    UNRESOLVED = "unresolved"


class RelationType(str, Enum):
    SUPPORT = "SUPPORT"
    SUPERSEDE = "SUPERSEDE"
    BLOCK = "BLOCK"
    CONDITION = "CONDITION"
    NONE = "NONE"
    UNCERTAIN = "UNCERTAIN"
    REQUIRED_BY = "REQUIRED_BY"


@dataclass(frozen=True)
class EpisodicEvidence:
    id: str
    timestamp: str
    text: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Belief:
    id: str
    proposition: str
    supported_by: list[str] = field(default_factory=list)
    status: BeliefStatus = BeliefStatus.AUTHORIZED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelationPrediction:
    relation: RelationType
    evidence_id: str | None = None
    belief_id: str | None = None
    target_belief_id: str | None = None
    condition: str | None = None
    rationale: str | None = None
    span: str | None = None
    confidence: float | None = None
    valid_from: str | None = None
    valid_until: str | None = None


@dataclass(frozen=True)
class AuthorizationDecision:
    belief_id: str
    authorized: bool
    reason: str
    justification_path: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationRecord:
    query_id: str
    method: str
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    candidate_beliefs: list[dict[str, Any]] = field(default_factory=list)
    authorized_basis: list[dict[str, Any]] = field(default_factory=list)
    blocked_beliefs: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    tokens: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    latency_ms: int | None = None


# --- Phase 1: 11 New Schema Records ---

@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    session_id: str
    timestamp: str | None
    text: str
    source_dataset: str  # stale | memora | manual_audit | other
    source_pointer: str
    is_raw_source: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefRecord:
    belief_id: str
    proposition: str
    source_evidence_ids: list[str] = field(default_factory=list)
    source_span: str | None = None
    timestamp: str | None = None
    extractor_version: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionRecord:
    belief_id: str
    condition: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RelationLabel(str, Enum):
    SUPPORT = "SUPPORT"
    SUPERSEDE = "SUPERSEDE"
    BLOCK = "BLOCK"
    CONDITION = "CONDITION"
    NONE = "NONE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class RelationPredictionRecord:
    relation: RelationLabel | str
    evidence_id: str | None = None
    belief_id: str | None = None
    target_belief_id: str | None = None
    condition: str | None = None
    rationale: str | None = None
    span: str | None = None
    confidence: float | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AuthorizationStatus(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    BLOCKED = "BLOCKED"
    SUPERSEDED = "SUPERSEDED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class AuthorizationRecord:
    belief_id: str
    authorization_status: AuthorizationStatus | str
    reason: str
    justification_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DefeatPathType(str, Enum):
    DIRECT_SUPERSEDE = "DIRECT_SUPERSEDE"
    PREREQUISITE_BLOCK = "PREREQUISITE_BLOCK"
    UNRESOLVED_UNCERTAIN = "UNRESOLVED_UNCERTAIN"


@dataclass(frozen=True)
class DefeatPathRecord:
    path_id: str
    path_type: DefeatPathType | str
    source_belief_id: str
    target_belief_id: str | None = None
    evidence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    query_text: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodTraceRecord:
    example_id: str
    method_name: str
    query: str
    upstream_commit: str | None = None
    candidate_evidence_ids: list[str] = field(default_factory=list)
    decision_payload: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None
    model_config_id: str | None = None
    token_counts: dict[str, int] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreRecord:
    example_id: str
    benchmark: str
    method_name: str
    official_scores: dict[str, Any] = field(default_factory=dict)
    local_diagnostics: dict[str, Any] = field(default_factory=dict)
    evaluator_version: str | None = None
    run_manifest_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunManifest:
    run_manifest_id: str
    method_name: str
    model_config_id: str
    timestamp: str
    upstream_commit: str | None = None
    output_path: str | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCallTrace:
    call_id: str
    provider: str
    model_id: str
    model_revision_or_api_version: str | None = None
    prompt_template_hash: str | None = None
    response_schema_version: str | None = None
    parser_version: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    input_hash: str | None = None
    condition_context_hash: str | None = None
    temporal_context_hash: str | None = None
    status: str = "success"  # success | failure | parse_error
    response: str | None = None
    parsed_output: Any | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retries: int = 0
    error_message: str | None = None
    eligible_for_replay: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canonical Typed Graph Schemas (Refactor Wave 0).
#
# These are the only schemas that new ReTrace runtime code (Waves 1+) is
# allowed to import. The legacy `EpisodicEvidence`, `Belief`,
# `RelationPrediction`, `AuthorizationDecision`, `EvaluationRecord`, and the
# Phase-1 `*Record` dataclasses above remain in-tree for one transition wave
# so existing tests still import; they must not be used by new runtime code.
#
# See `docs/refactor_plan_defeat_path.md` for the binding amendments
# A1-A10 that govern these definitions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceNode:
    """Immutable raw evidence atom."""

    evidence_id: str
    session_id: str
    timestamp: str | None
    text: str
    source_dataset: str
    source_pointer: str
    is_raw_source: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeliefNode:
    """Open-text belief proposition derived from evidence."""

    belief_id: str
    proposition: str
    source_evidence_ids: tuple[str, ...] = ()
    source_span: str | None = None
    extractor_version: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionNode:
    """Open-text prerequisite required for current use of a belief.

    Per amendment A7, identity is namespaced by `scope_id` (typically the
    user id) so identical strings across users do not merge.
    """

    condition_id: str
    scope_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencyEdge:
    """`belief --REQUIRES--> condition` dependency edge.

    Per amendment A8, provenance is first-class:
    `inducer`, `supporting_evidence_ids`, `model_call_trace_id`,
    `confidence`, and `rationale` are not metadata.
    """

    edge_id: str
    belief_id: str
    condition_id: str
    inducer: str
    edge_type: str = "REQUIRES"
    supporting_evidence_ids: tuple[str, ...] = ()
    model_call_trace_id: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceEdgeType(str, Enum):
    BLOCKS = "BLOCKS"            # target_kind = "condition"
    RELEASES = "RELEASES"        # target_kind = "condition"
    SUPERSEDES = "SUPERSEDES"    # target_kind = "belief"; replacement_belief_id required
    REAFFIRMS = "REAFFIRMS"      # target_kind = "belief"
    UNCERTAIN = "UNCERTAIN"      # target_kind = "belief"


@dataclass(frozen=True)
class EvidenceEdge:
    """`evidence --(edge_type)--> (condition | belief)` update edge.

    Per amendment A1, `SUPERSEDES` edges must populate
    `replacement_belief_id` so the typed graph preserves both the defeated
    prior belief and the current replacement.
    """

    edge_id: str
    edge_type: EvidenceEdgeType
    evidence_id: str
    target_kind: str
    target_id: str
    verifier: str
    replacement_belief_id: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    span: str | None = None
    model_call_trace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DefeatPath:
    """A fully auditable accepted defeat path produced by DPA.

    Per amendment A5, an authorized belief produces an
    `AuthorizationTrace` whose `accepted_defeat_path` is `None`; only
    actually defeating paths are represented here.
    """

    path_id: str
    path_type: DefeatPathType
    target_belief_id: str
    supporting_dependency_edge_ids: tuple[str, ...] = ()
    supporting_evidence_edge_ids: tuple[str, ...] = ()
    replacement_belief_id: str | None = None
    as_of_time: str | None = None
    as_of_evidence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthorizationTrace:
    """Final authorization decision for a single belief.

    Produced exclusively by `DefeatPathAuthorizationAlgorithm` (Wave 1).
    Verifiers may not produce this object directly.
    """

    trace_id: str
    belief_id: str
    status: AuthorizationStatus
    accepted_defeat_path: DefeatPath | None = None
    considered_defeat_paths: tuple[DefeatPath, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    query_id: str | None = None
    as_of_time: str | None = None
    as_of_evidence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

