"""Deterministic typed-graph fixture builders shared across gate-unit tests.

These helpers build `EpisodicEvidence` ledger entries and typed graph nodes /
edges by hand. They never invoke a verifier or extractor, and they never
read from boundary-audit / STALE / Memora fixtures. Every value is local to
the test, so the tests can be re-ordered and re-run deterministically.
"""

from __future__ import annotations

from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceNode,
    EvidenceEdge,
    EvidenceEdgeType,
)


def make_evidence(evidence_id: str, timestamp: str, text: str = "") -> EvidenceNode:
    """Build an EvidenceNode with a deterministic timestamp."""
    return EvidenceNode(
        evidence_id=evidence_id,
        session_id="session_default",
        timestamp=timestamp,
        text=text or f"evidence {evidence_id}",
        source_dataset="manual_audit",
        source_pointer="gate_unit_fixture",
    )


def make_belief(
    belief_id: str,
    proposition: str,
    source_evidence_ids: tuple[str, ...] = (),
) -> BeliefNode:
    return BeliefNode(
        belief_id=belief_id,
        proposition=proposition,
        source_evidence_ids=source_evidence_ids,
        extractor_version="gate_unit_fixture",
    )


def make_condition(condition_id: str, text: str, scope_id: str = "scope_default") -> ConditionNode:
    return ConditionNode(
        condition_id=condition_id,
        scope_id=scope_id,
        text=text,
    )


def make_dependency(
    edge_id: str,
    belief_id: str,
    condition_id: str,
    supporting_evidence_ids: tuple[str, ...] = (),
    rationale: str | None = None,
) -> DependencyEdge:
    return DependencyEdge(
        edge_id=edge_id,
        belief_id=belief_id,
        condition_id=condition_id,
        inducer="manual_fixture",
        supporting_evidence_ids=supporting_evidence_ids,
        rationale=rationale,
    )


def make_evidence_edge(
    edge_id: str,
    edge_type: EvidenceEdgeType,
    evidence_id: str,
    target_kind: str,
    target_id: str,
    replacement_belief_id: str | None = None,
    rationale: str | None = None,
) -> EvidenceEdge:
    return EvidenceEdge(
        edge_id=edge_id,
        edge_type=edge_type,
        evidence_id=evidence_id,
        target_kind=target_kind,
        target_id=target_id,
        verifier="manual_fixture",
        replacement_belief_id=replacement_belief_id,
        rationale=rationale,
    )


def build_world(
    *,
    evidences: list[EvidenceNode],
    beliefs: list[BeliefNode],
    conditions: list[ConditionNode] | None = None,
    dependency_edges: list[DependencyEdge] | None = None,
    evidence_edges: list[EvidenceEdge] | None = None,
) -> tuple[BeliefStore, EpisodeLedger]:
    """Assemble a typed-graph store and an evidence ledger from raw lists.

    Insertion order follows the list order, which fixes the ledger-index
    component of the temporal recency key.
    """
    ledger = EpisodeLedger()
    for ev in evidences:
        ledger.append(ev)

    store = BeliefStore()
    for belief in beliefs:
        store.add_belief(belief)
    for condition in conditions or ():
        store.add_condition(condition)
    for dep in dependency_edges or ():
        store.add_dependency_edge(dep)
    for edge in evidence_edges or ():
        store.add_evidence_edge(edge)

    return store, ledger
