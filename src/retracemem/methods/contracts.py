"""Shared controlled-comparison contracts for Stage A/B attribution.

These dataclasses define the fixed authorization input shared by Stage A
(ReTrace-LLM) and Stage B (DirectJudge-LLM) in the primary controlled
comparison track. They are method-local and do not modify schemas.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from retracemem.schemas import BeliefNode, ConditionNode, EvidenceNode


@dataclass(frozen=True)
class SharedCandidateView:
    """Exact fixed authorization input shared by Stage A and Stage B.

    In the primary controlled comparison, both methods receive this
    identical view. Neither method performs its own extraction or retrieval.
    """

    instance_id: str
    query_id: str
    query: str
    evidence_context: tuple[EvidenceNode, ...]
    candidate_beliefs: tuple[BeliefNode, ...]
    candidate_replacement_beliefs: tuple[BeliefNode, ...]
    candidate_conditions_by_belief: dict[str, tuple[ConditionNode, ...]] = field(
        default_factory=dict,
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        belief_ids = [b.belief_id for b in self.candidate_beliefs]
        if len(belief_ids) != len(set(belief_ids)):
            raise ValueError("candidate_beliefs contains duplicate belief_ids")
        replacement_ids = [b.belief_id for b in self.candidate_replacement_beliefs]
        if len(replacement_ids) != len(set(replacement_ids)):
            raise ValueError("candidate_replacement_beliefs contains duplicate belief_ids")
        valid_belief_ids = set(belief_ids)
        for key, conds in self.candidate_conditions_by_belief.items():
            if key not in valid_belief_ids:
                raise ValueError(
                    f"candidate_conditions_by_belief key '{key}' is not a candidate belief id"
                )
            cond_ids = [c.condition_id for c in conds]
            if len(cond_ids) != len(set(cond_ids)):
                raise ValueError(
                    f"Duplicate condition_ids for belief '{key}'"
                )


class DirectUsabilityStatus(str, Enum):
    """Verdict categories for DirectJudge-LLM.

    These are not DPA edge types. They represent a direct final usability
    judgment that bypasses the typed-edge decomposition entirely.
    """

    USABLE = "USABLE"
    NOT_USABLE = "NOT_USABLE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class DirectUsabilityVerdict:
    """A single DirectJudge-LLM verdict for one candidate belief.

    This is not an EvidenceEdge. It contains no typed-edge or defeat-path
    semantics. It records a direct model judgment of current usability.
    """

    belief_id: str
    status: DirectUsabilityStatus
    rationale: str
    model_call_trace_id: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControlledMethodResult:
    """Unified output from either Stage A or Stage B in the controlled track.

    Allows side-by-side comparison of authorization decisions using
    identical candidate inputs.
    """

    method_name: str
    instance_id: str
    query_id: str
    authorized_belief_ids: tuple[str, ...]
    excluded_belief_ids: tuple[str, ...] = ()
    verdicts: tuple[DirectUsabilityVerdict, ...] = ()
    model_call_trace_ids: tuple[str, ...] = ()
    cost: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
