"""Shared controlled-comparison contracts for Stage A/B attribution.

These dataclasses define the fixed authorization input shared by Stage A
(ReTrace-LLM) and Stage B (DirectJudge-LLM) in the primary controlled
comparison track. They are method-local and do not modify schemas.py.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from retracemem.schemas import BeliefNode, ConditionNode, DependencyEdge, EvidenceNode


def _compute_view_fingerprint(
    instance_id: str,
    query_id: str,
    query: str,
    evidence_context: tuple[EvidenceNode, ...],
    new_evidence: EvidenceNode | None,
    candidate_beliefs: tuple[BeliefNode, ...],
    candidate_replacement_beliefs: tuple[BeliefNode, ...],
    candidate_conditions_by_belief: tuple[tuple[str, tuple[ConditionNode, ...]], ...],
    dependency_edges_by_belief: tuple[tuple[str, tuple[DependencyEdge, ...]], ...],
) -> str:
    """Compute a deterministic fingerprint from all fixed controlled inputs."""
    parts: list[str] = [
        f"instance:{instance_id}",
        f"query_id:{query_id}",
        f"query:{query}",
    ]
    for ev in evidence_context:
        parts.append(f"ev:{ev.evidence_id}:{ev.text}")
    if new_evidence is not None:
        parts.append(f"new_ev:{new_evidence.evidence_id}")
    for b in candidate_beliefs:
        parts.append(f"cb:{b.belief_id}:{b.proposition}")
    for b in candidate_replacement_beliefs:
        parts.append(f"cr:{b.belief_id}:{b.proposition}")
    for bid, conds in candidate_conditions_by_belief:
        for c in conds:
            parts.append(f"cond:{bid}:{c.condition_id}:{c.text}")
    for bid, deps in dependency_edges_by_belief:
        for d in deps:
            parts.append(f"dep:{bid}:{d.edge_id}:{d.condition_id}")
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    candidate_conditions_by_belief: tuple[tuple[str, tuple[ConditionNode, ...]], ...] = ()
    dependency_edges_by_belief: tuple[tuple[str, tuple[DependencyEdge, ...]], ...] = ()
    new_evidence: EvidenceNode | None = None
    view_fingerprint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        belief_ids = [b.belief_id for b in self.candidate_beliefs]
        if len(belief_ids) != len(set(belief_ids)):
            raise ValueError("candidate_beliefs contains duplicate belief_ids")
        replacement_ids = [b.belief_id for b in self.candidate_replacement_beliefs]
        if len(replacement_ids) != len(set(replacement_ids)):
            raise ValueError("candidate_replacement_beliefs contains duplicate belief_ids")
        valid_belief_ids = set(belief_ids)

        # Validate conditions
        seen_condition_ids_by_belief: dict[str, set[str]] = {}
        for key, conds in self.candidate_conditions_by_belief:
            if key not in valid_belief_ids:
                raise ValueError(
                    f"candidate_conditions_by_belief key '{key}' is not a candidate belief id"
                )
            cond_ids = [c.condition_id for c in conds]
            if len(cond_ids) != len(set(cond_ids)):
                raise ValueError(
                    f"Duplicate condition_ids for belief '{key}'"
                )
            seen_condition_ids_by_belief[key] = set(cond_ids)

        # Validate dependency edges
        seen_dep_edge_ids: set[str] = set()
        for key, deps in self.dependency_edges_by_belief:
            if key not in valid_belief_ids:
                raise ValueError(
                    f"dependency_edges_by_belief key '{key}' is not a candidate belief id"
                )
            belief_conditions = seen_condition_ids_by_belief.get(key, set())
            for dep in deps:
                if dep.edge_type != "REQUIRES":
                    raise ValueError(
                        f"Dependency edge '{dep.edge_id}' has edge_type '{dep.edge_type}'; "
                        f"only 'REQUIRES' is permitted"
                    )
                if dep.belief_id != key:
                    raise ValueError(
                        f"Dependency edge '{dep.edge_id}' targets belief '{dep.belief_id}' "
                        f"but is mapped under '{key}'"
                    )
                if dep.condition_id not in belief_conditions:
                    raise ValueError(
                        f"Dependency edge '{dep.edge_id}' refers to condition "
                        f"'{dep.condition_id}' not supplied for belief '{key}'"
                    )
                if dep.edge_id in seen_dep_edge_ids:
                    raise ValueError(
                        f"Duplicate dependency edge_id '{dep.edge_id}'"
                    )
                seen_dep_edge_ids.add(dep.edge_id)

        # Validate new_evidence
        ev_ids = {e.evidence_id for e in self.evidence_context}
        if self.new_evidence is not None:
            if self.new_evidence.evidence_id not in ev_ids:
                raise ValueError(
                    f"new_evidence '{self.new_evidence.evidence_id}' must appear in evidence_context"
                )

        # Compute fingerprint
        fp = _compute_view_fingerprint(
            instance_id=self.instance_id,
            query_id=self.query_id,
            query=self.query,
            evidence_context=self.evidence_context,
            new_evidence=self.new_evidence,
            candidate_beliefs=self.candidate_beliefs,
            candidate_replacement_beliefs=self.candidate_replacement_beliefs,
            candidate_conditions_by_belief=self.candidate_conditions_by_belief,
            dependency_edges_by_belief=self.dependency_edges_by_belief,
        )
        object.__setattr__(self, "view_fingerprint", fp)


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
