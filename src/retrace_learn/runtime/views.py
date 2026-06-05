"""Bridge between MemPatch Revision Module views/actions and DPA projection.

This module is the only place that translates the dict-shaped training data
into the canonical ``retracemem`` dataclasses consumed by ``authorize(...)``.
Keeping the translation here means the learned components never hand-roll
``EvidenceEdge`` / ``SharedCandidateView`` construction, so the deterministic
kernel stays the single source of truth.
"""
from __future__ import annotations

from typing import Any

from retracemem.authorization import EvidenceProposalBatch
from retracemem.methods.contracts import SharedCandidateView
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceNode,
)

from retrace_learn.schemas import RevisionAction

DEFAULT_SESSION = "session_0"
DEFAULT_DATASET = "retrace_learn_synthetic"


def evidence_node_from_dict(d: dict[str, Any]) -> EvidenceNode:
    return EvidenceNode(
        evidence_id=d["evidence_id"],
        session_id=d.get("session_id", DEFAULT_SESSION),
        timestamp=d.get("timestamp"),
        text=d.get("text", ""),
        source_dataset=d.get("source_dataset", DEFAULT_DATASET),
        source_pointer=d.get("source_pointer", d["evidence_id"]),
        is_raw_source=d.get("is_raw_source", True),
        metadata=d.get("metadata", {}),
    )


def belief_node_from_dict(d: dict[str, Any]) -> BeliefNode:
    return BeliefNode(
        belief_id=d["belief_id"],
        proposition=d.get("proposition", ""),
        source_evidence_ids=tuple(d.get("source_evidence_ids", ()) or ()),
        source_span=d.get("source_span"),
        extractor_version=d.get("extractor_version"),
        confidence=d.get("confidence"),
        metadata=d.get("metadata", {}),
    )


def condition_node_from_dict(d: dict[str, Any]) -> ConditionNode:
    return ConditionNode(
        condition_id=d["condition_id"],
        scope_id=d.get("scope_id", "scope_0"),
        text=d.get("text", ""),
        metadata=d.get("metadata", {}),
    )


def dependency_edge_from_dict(d: dict[str, Any]) -> DependencyEdge:
    return DependencyEdge(
        edge_id=d["edge_id"],
        belief_id=d["belief_id"],
        condition_id=d["condition_id"],
        inducer=d.get("inducer", "retrace_learn_extractor"),
        edge_type=d.get("edge_type", "REQUIRES"),
        supporting_evidence_ids=tuple(d.get("supporting_evidence_ids", ()) or ()),
        confidence=d.get("confidence"),
        rationale=d.get("rationale"),
        metadata=d.get("metadata", {}),
    )


def build_view(
    *,
    instance_id: str,
    query_id: str,
    query: str,
    evidence_context: list[dict[str, Any]],
    new_evidence_id: str,
    candidate_beliefs: list[dict[str, Any]],
    candidate_replacement_beliefs: list[dict[str, Any]],
    candidate_conditions_by_belief: dict[str, list[dict[str, Any]]],
    dependency_edges_by_belief: dict[str, list[dict[str, Any]]],
) -> SharedCandidateView:
    """Assemble a :class:`SharedCandidateView` from dict-shaped graph fields."""
    evidence_nodes = [evidence_node_from_dict(e) for e in evidence_context]
    matches = [e for e in evidence_nodes if e.evidence_id == new_evidence_id]
    if len(matches) != 1:
        raise ValueError(
            f"new_evidence_id '{new_evidence_id}' must match exactly one evidence node "
            f"(found {len(matches)})"
        )
    new_ev = matches[0]

    conditions_by_belief = tuple(
        (bid, tuple(condition_node_from_dict(c) for c in conds))
        for bid, conds in candidate_conditions_by_belief.items()
    )
    deps_by_belief = tuple(
        (bid, tuple(dependency_edge_from_dict(d) for d in deps))
        for bid, deps in dependency_edges_by_belief.items()
    )

    return SharedCandidateView(
        instance_id=instance_id,
        query_id=query_id,
        query=query,
        evidence_context=tuple(evidence_nodes),
        new_evidence=new_ev,
        candidate_beliefs=tuple(belief_node_from_dict(b) for b in candidate_beliefs),
        candidate_replacement_beliefs=tuple(
            belief_node_from_dict(b) for b in candidate_replacement_beliefs
        ),
        candidate_conditions_by_belief=conditions_by_belief,
        dependency_edges_by_belief=deps_by_belief,
    )


def actions_to_proposal_batches(
    actions: list[RevisionAction],
    *,
    verifier: str = "retrace_learn_proposer",
    edge_id_prefix: str = "edge_rl",
    model_call_trace_id: str | None = None,
) -> tuple[EvidenceProposalBatch, ...]:
    """Convert validated typed actions into runtime ``EvidenceProposalBatch``es.

    ``NO_REVISION`` actions produce no edges (they are pure no-ops). The
    resulting edges are still subject to the ``RevisionGate`` inside
    ``authorize(...)`` — this function performs no admission itself.
    """
    edges: list[EvidenceEdge] = []
    for idx, action in enumerate(actions):
        edge_type = action.evidence_edge_type
        if edge_type is None:  # NO_REVISION
            continue
        if action.target_condition_id is not None:
            target_kind = "condition"
            target_id = action.target_condition_id
        else:
            target_kind = "belief"
            target_id = action.target_belief_id
        evidence_id = action.evidence_ids[0] if action.evidence_ids else ""
        edges.append(
            EvidenceEdge(
                edge_id=f"{edge_id_prefix}_{idx}",
                edge_type=edge_type,
                evidence_id=evidence_id,
                target_kind=target_kind,
                target_id=target_id or "",
                verifier=verifier,
                replacement_belief_id=action.replacement_belief_id,
                rationale=action.rationale or None,
                metadata={},
            )
        )
    if not edges:
        return ()
    return (
        EvidenceProposalBatch(
            edges=tuple(edges),
            model_call_trace_id=model_call_trace_id,
            metadata={"proposer": "retrace_learn"},
        ),
    )
