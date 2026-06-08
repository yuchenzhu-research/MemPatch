"""Shared revision-view contracts for controlled method comparison.

These dataclasses define the fixed authorization input shared by baseline
configurations (typed-action proposer vs direct status judge) in the primary
controlled comparison. They are method-local and do not modify schemas.py.

Metadata policy:
    ``metadata`` fields on SharedCandidateView, EvidenceNode, BeliefNode,
    ConditionNode, DependencyEdge are NON-SEMANTIC diagnostic attachments.
    Baseline executors MUST NOT consume them during method execution and MUST
    NOT include them in ``view_fingerprint``.
"""
from __future__ import annotations

import hashlib
import json as _json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mempatch_dpa.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceNode,
)

_FINGERPRINT_SCHEMA_VERSION = "v1"


def _compute_view_fingerprint(
    instance_id: str,
    query_id: str,
    query: str,
    evidence_context: tuple[EvidenceNode, ...],
    new_evidence: EvidenceNode,
    candidate_beliefs: tuple[BeliefNode, ...],
    candidate_replacement_beliefs: tuple[BeliefNode, ...],
    candidate_conditions_by_belief: tuple[tuple[str, tuple[ConditionNode, ...]], ...],
    dependency_edges_by_belief: tuple[tuple[str, tuple[DependencyEdge, ...]], ...],
) -> str:
    """Compute a deterministic canonical fingerprint over all first-class
    controlled-input fields. Uses versioned JSON serialization with sorted
    keys. Metadata fields are explicitly excluded (non-semantic).
    """
    def _ev(e: EvidenceNode) -> dict[str, Any]:
        return {
            "evidence_id": e.evidence_id,
            "session_id": e.session_id,
            "timestamp": e.timestamp,
            "text": e.text,
            "source_dataset": e.source_dataset,
            "source_pointer": e.source_pointer,
            "is_raw_source": e.is_raw_source,
        }

    def _belief(b: BeliefNode) -> dict[str, Any]:
        return {
            "belief_id": b.belief_id,
            "proposition": b.proposition,
            "source_evidence_ids": list(b.source_evidence_ids),
            "source_span": b.source_span,
            "extractor_version": b.extractor_version,
            "confidence": b.confidence,
        }

    def _cond(c: ConditionNode) -> dict[str, Any]:
        return {
            "condition_id": c.condition_id,
            "scope_id": c.scope_id,
            "text": c.text,
        }

    def _dep(d: DependencyEdge) -> dict[str, Any]:
        return {
            "edge_id": d.edge_id,
            "edge_type": d.edge_type,
            "belief_id": d.belief_id,
            "condition_id": d.condition_id,
            "inducer": d.inducer,
            "supporting_evidence_ids": list(d.supporting_evidence_ids),
            "model_call_trace_id": d.model_call_trace_id,
            "confidence": d.confidence,
            "rationale": d.rationale,
        }

    payload: dict[str, Any] = {
        "_schema_version": _FINGERPRINT_SCHEMA_VERSION,
        "instance_id": instance_id,
        "query_id": query_id,
        "query": query,
        "evidence_context": [_ev(e) for e in evidence_context],
        "new_evidence": _ev(new_evidence),
        "candidate_beliefs": [_belief(b) for b in candidate_beliefs],
        "candidate_replacement_beliefs": [_belief(b) for b in candidate_replacement_beliefs],
        "candidate_conditions_by_belief": [
            {"belief_id": bid, "conditions": [_cond(c) for c in conds]}
            for bid, conds in candidate_conditions_by_belief
        ],
        "dependency_edges_by_belief": [
            {"belief_id": bid, "edges": [_dep(d) for d in deps]}
            for bid, deps in dependency_edges_by_belief
        ],
    }
    canonical = _json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SharedCandidateView:
    """Bounded revision view: scenario context + new evidence for authorization.

    In controlled comparison, all methods receive this identical view.
    Neither method performs its own Scenario View Builder extraction.

    ``metadata`` is non-semantic. Baseline executors MUST NOT read it.
    It is not included in ``view_fingerprint``.
    """

    instance_id: str
    query_id: str
    query: str
    evidence_context: tuple[EvidenceNode, ...]
    new_evidence: EvidenceNode
    candidate_beliefs: tuple[BeliefNode, ...]
    candidate_replacement_beliefs: tuple[BeliefNode, ...]
    candidate_conditions_by_belief: tuple[tuple[str, tuple[ConditionNode, ...]], ...] = ()
    dependency_edges_by_belief: tuple[tuple[str, tuple[DependencyEdge, ...]], ...] = ()
    view_fingerprint: str = field(init=False, default="")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # --- Evidence context invariants ---
        ev_ids_list = [e.evidence_id for e in self.evidence_context]
        if len(ev_ids_list) != len(set(ev_ids_list)):
            raise ValueError("evidence_context contains duplicate evidence_ids")

        # new_evidence must appear in evidence_context by identity
        found = False
        for e in self.evidence_context:
            if e.evidence_id == self.new_evidence.evidence_id:
                if e is not self.new_evidence and e != self.new_evidence:
                    raise ValueError(
                        f"evidence_context contains evidence_id "
                        f"'{self.new_evidence.evidence_id}' with a different "
                        f"payload than new_evidence"
                    )
                found = True
                break
        if not found:
            raise ValueError(
                f"new_evidence '{self.new_evidence.evidence_id}' must appear "
                f"in evidence_context"
            )

        # --- Belief id invariants ---
        belief_ids = [b.belief_id for b in self.candidate_beliefs]
        if len(belief_ids) != len(set(belief_ids)):
            raise ValueError("candidate_beliefs contains duplicate belief_ids")
        replacement_ids = [b.belief_id for b in self.candidate_replacement_beliefs]
        if len(replacement_ids) != len(set(replacement_ids)):
            raise ValueError(
                "candidate_replacement_beliefs contains duplicate belief_ids"
            )
        overlap = set(belief_ids) & set(replacement_ids)
        if overlap:
            raise ValueError(
                f"candidate belief ids and replacement belief ids overlap: "
                f"{sorted(overlap)}"
            )
        valid_belief_ids = set(belief_ids)

        # --- Conditions invariants ---
        seen_condition_keys: set[str] = set()
        seen_condition_ids_by_belief: dict[str, set[str]] = {}
        all_condition_nodes: dict[str, ConditionNode] = {}
        for key, conds in self.candidate_conditions_by_belief:
            if key in seen_condition_keys:
                raise ValueError(
                    f"candidate_conditions_by_belief contains repeated key "
                    f"'{key}'"
                )
            seen_condition_keys.add(key)
            if key not in valid_belief_ids:
                raise ValueError(
                    f"candidate_conditions_by_belief key '{key}' is not a "
                    f"candidate belief id"
                )
            cond_ids = [c.condition_id for c in conds]
            if len(cond_ids) != len(set(cond_ids)):
                raise ValueError(
                    f"Duplicate condition_ids for belief '{key}'"
                )
            for c in conds:
                existing = all_condition_nodes.get(c.condition_id)
                if existing is not None and existing != c:
                    raise ValueError(
                        f"ConditionNode '{c.condition_id}' appears with "
                        f"conflicting payloads across belief groups"
                    )
                all_condition_nodes[c.condition_id] = c
            seen_condition_ids_by_belief[key] = set(cond_ids)

        # --- Dependency edge invariants ---
        seen_dep_keys: set[str] = set()
        seen_dep_edge_ids: set[str] = set()
        for key, deps in self.dependency_edges_by_belief:
            if key in seen_dep_keys:
                raise ValueError(
                    f"dependency_edges_by_belief contains repeated key '{key}'"
                )
            seen_dep_keys.add(key)
            if key not in valid_belief_ids:
                raise ValueError(
                    f"dependency_edges_by_belief key '{key}' is not a "
                    f"candidate belief id"
                )
            belief_conditions = seen_condition_ids_by_belief.get(key, set())
            for dep in deps:
                if dep.edge_type != "REQUIRES":
                    raise ValueError(
                        f"Dependency edge '{dep.edge_id}' has edge_type "
                        f"'{dep.edge_type}'; only 'REQUIRES' is permitted"
                    )
                if dep.belief_id != key:
                    raise ValueError(
                        f"Dependency edge '{dep.edge_id}' targets belief "
                        f"'{dep.belief_id}' but is mapped under '{key}'"
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

        # --- Compute fingerprint (derived, not caller-settable) ---
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


@dataclass(frozen=True)
class EdgePredictionBatch:
    """Traced output from a single PromptEvidenceEdgeVerifier invocation.

    Preserves the model_call_trace_id even when zero edges are predicted,
    enabling full auditability of every verifier invocation.
    """

    proposed_edges: tuple[EvidenceEdge, ...]
    model_call_trace_id: str
    prompt_version: str
    model_id: str
    provider: str
    model_revision_or_api_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
