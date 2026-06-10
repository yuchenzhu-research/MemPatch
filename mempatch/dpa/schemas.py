from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AuthorizationStatus(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    BLOCKED = "BLOCKED"
    SUPERSEDED = "SUPERSEDED"
    UNRESOLVED = "UNRESOLVED"


class DefeatPathType(str, Enum):
    DIRECT_SUPERSEDE = "DIRECT_SUPERSEDE"
    PREREQUISITE_BLOCK = "PREREQUISITE_BLOCK"
    UNRESOLVED_UNCERTAIN = "UNRESOLVED_UNCERTAIN"


class EvidenceEdgeType(str, Enum):
    BLOCKS = "BLOCKS"            # target_kind = "condition"
    RELEASES = "RELEASES"        # target_kind = "condition"
    SUPERSEDES = "SUPERSEDES"    # target_kind = "belief"; replacement_belief_id required
    REAFFIRMS = "REAFFIRMS"      # target_kind = "belief"
    UNCERTAIN = "UNCERTAIN"      # target_kind = "belief"


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
    """Open-text prerequisite required for current use of a belief."""

    condition_id: str
    scope_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencyEdge:
    """`belief --REQUIRES--> condition` dependency edge."""

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


@dataclass(frozen=True)
class EvidenceEdge:
    """`evidence --(edge_type)--> (condition | belief)` update edge."""

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
    """A fully auditable accepted defeat path produced by DPA."""

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
    """Final authorization decision for a single belief."""

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


# ===========================================================================
# Surviving Executable Dependencies (Provider, Cache, and Scaffolding layer)
# ===========================================================================

@dataclass(frozen=True)
class ModelCallTrace:
    """Surviving executable dependency: caches prompt completions & token counts."""

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
    status: str = "success"
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


@dataclass(frozen=True)
class EvaluationRecord:
    """Surviving executable dependency: records evaluation runs."""

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
