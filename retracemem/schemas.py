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
