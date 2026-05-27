from __future__ import annotations

from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)


def evidence(evidence_id: str, text: str, metadata: dict[str, object] | None = None) -> EvidenceNode:
    return EvidenceNode(
        evidence_id=evidence_id,
        session_id="session_1",
        timestamp="2026-05-28T09:00:00Z",
        text=text,
        source_dataset="manual_audit",
        source_pointer=f"fixture:{evidence_id}",
        metadata=dict(metadata or {}),
    )


def belief(belief_id: str, proposition: str) -> BeliefNode:
    return BeliefNode(
        belief_id=belief_id,
        proposition=proposition,
        source_evidence_ids=("support_1",),
        metadata={"scope_id": "user_1"},
    )


def condition(condition_id: str, text: str) -> ConditionNode:
    return ConditionNode(condition_id=condition_id, scope_id="user_1", text=text)


def dependency(edge_id: str = "dep_1") -> DependencyEdge:
    return DependencyEdge(
        edge_id=edge_id,
        belief_id="belief_bike",
        condition_id="condition:user_1:mobility",
        inducer="manual_fixture",
        edge_type="REQUIRES",
        supporting_evidence_ids=("support_1",),
        confidence=1.0,
        rationale="Manual fixture dependency.",
    )


def evidence_edge(edge_type: EvidenceEdgeType, target_kind: str, target_id: str) -> EvidenceEdge:
    return EvidenceEdge(
        edge_id=f"edge_{edge_type.value}",
        edge_type=edge_type,
        evidence_id="evidence_1",
        target_kind=target_kind,
        target_id=target_id,
        verifier="manual_fixture",
        replacement_belief_id="belief_replacement" if edge_type == EvidenceEdgeType.SUPERSEDES else None,
        confidence=1.0,
        rationale="Manual fixture edge.",
    )
