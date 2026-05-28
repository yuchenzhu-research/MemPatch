from __future__ import annotations

import pytest
from retracemem.generation.basis_builder import BasisBuilder
from retracemem.retrieval.typed_retrievers import ManualQueryBeliefRetriever
from retracemem.memory.belief_store import BeliefStore
from retracemem.memory.episode_ledger import EpisodeLedger
from retracemem.tms.authorization import DefeatPathAuthorizationAlgorithm
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)


def test_basis_builder_returns_expected_json_structure():
    # Setup store and ledger
    store = BeliefStore()
    ledger = EpisodeLedger()

    # Evidence
    ev = EvidenceNode(
        evidence_id="ev_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice resides in SF.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    ledger.append(ev)

    # Beliefs
    b1 = BeliefNode(
        belief_id="belief_1",
        proposition="Alice resides in SF.",
        source_evidence_ids=("ev_1",),
    )
    b2 = BeliefNode(
        belief_id="belief_2",
        proposition="Alice rides a bicycle.",
        source_evidence_ids=("ev_1",),
    )
    store.add_belief(b1)
    store.add_belief(b2)

    # DPA and Retriever
    engine = DefeatPathAuthorizationAlgorithm(store, ledger)
    retriever = ManualQueryBeliefRetriever(query_map={
        "where does Alice live?": ["belief_1", "belief_2"]
    })

    builder = BasisBuilder(retriever, engine)
    res = builder.build(
        query="where does Alice live?",
        beliefs=(b1, b2),
        limit=10,
        query_id="q_1",
    )

    # Assert JSON-compatible output dictionary structure
    assert res["query_id"] == "q_1"
    assert res["retrieved_belief_ids"] == ["belief_1", "belief_2"]
    assert len(res["authorized_basis"]) == 2
    assert res["authorized_basis"][0]["belief_id"] == "belief_1"
    assert res["authorized_basis"][0]["proposition"] == "Alice resides in SF."
    assert res["authorized_basis"][0]["source_evidence_ids"] == ["ev_1"]
    assert res["authorized_basis"][0]["authorization_status"] == "AUTHORIZED"
    assert res["excluded"] == []


def test_basis_builder_excludes_blocked_and_superseded_beliefs():
    store = BeliefStore()
    ledger = EpisodeLedger()

    # Evidences
    ev1 = EvidenceNode(
        evidence_id="ev_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice commutes by bicycle.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    ev2 = EvidenceNode(
        evidence_id="ev_2",
        session_id="session_1",
        timestamp="2026-05-28T01:00:00Z",
        text="Alice broke her leg.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    ledger.append(ev1)
    ledger.append(ev2)

    # Beliefs
    b1 = BeliefNode(
        belief_id="belief_1",
        proposition="Alice commutes by bicycle.",
        source_evidence_ids=("ev_1",),
    )
    store.add_belief(b1)

    # Conditions and Dependencies
    c1 = ConditionNode(
        condition_id="cond_1",
        scope_id="user_alice",
        text="Alice has sufficient mobility.",
    )
    store.add_condition(c1)

    dep = DependencyEdge(
        edge_id="dep_1",
        belief_id="belief_1",
        condition_id="cond_1",
        inducer="manual",
    )
    store.add_dependency_edge(dep)

    # Blocker edge
    block_edge = EvidenceEdge(
        edge_id="block_1",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="ev_2",
        target_kind="condition",
        target_id="cond_1",
        verifier="manual",
    )
    store.add_evidence_edge(block_edge)

    engine = DefeatPathAuthorizationAlgorithm(store, ledger)
    retriever = ManualQueryBeliefRetriever(query_map={
        "how does Alice commute?": ["belief_1"]
    })

    builder = BasisBuilder(retriever, engine)
    res = builder.build(
        query="how does Alice commute?",
        beliefs=(b1,),
        limit=10,
        query_id="q_2",
    )

    assert res["query_id"] == "q_2"
    assert res["retrieved_belief_ids"] == ["belief_1"]
    assert res["authorized_basis"] == []
    assert len(res["excluded"]) == 1
    assert res["excluded"][0]["belief_id"] == "belief_1"
    assert res["excluded"][0]["status"] == "BLOCKED"
    assert res["excluded"][0]["accepted_defeat_path"] is not None
    assert res["excluded"][0]["accepted_defeat_path"]["path_type"] == "PREREQUISITE_BLOCK"
    assert res["excluded"][0]["accepted_defeat_path"]["target_belief_id"] == "belief_1"


def test_basis_builder_surfaces_replacement_for_superseded_belief():
    store = BeliefStore()
    ledger = EpisodeLedger()

    # Evidences
    ev1 = EvidenceNode(
        evidence_id="ev_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice lives in SF.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    ev2 = EvidenceNode(
        evidence_id="ev_2",
        session_id="session_1",
        timestamp="2026-05-28T01:00:00Z",
        text="Alice moved to NY.",
        source_dataset="manual_audit",
        source_pointer="test",
    )
    ledger.append(ev1)
    ledger.append(ev2)

    # Beliefs
    b1 = BeliefNode(
        belief_id="belief_1",
        proposition="Alice lives in SF.",
        source_evidence_ids=("ev_1",),
    )
    b2 = BeliefNode(
        belief_id="belief_2",
        proposition="Alice lives in NY.",
        source_evidence_ids=("ev_2",),
    )
    store.add_belief(b1)
    store.add_belief(b2)

    # Supersedes edge
    super_edge = EvidenceEdge(
        edge_id="super_1",
        edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="ev_2",
        target_kind="belief",
        target_id="belief_1",
        verifier="manual",
        replacement_belief_id="belief_2",
    )
    store.add_evidence_edge(super_edge)

    engine = DefeatPathAuthorizationAlgorithm(store, ledger)
    # The retriever only finds the old belief
    retriever = ManualQueryBeliefRetriever(query_map={
        "where does Alice live?": ["belief_1"]
    })

    builder = BasisBuilder(retriever, engine)
    res = builder.build(
        query="where does Alice live?",
        beliefs=(b1, b2),
        limit=10,
        query_id="q_3",
    )

    assert res["query_id"] == "q_3"
    assert res["retrieved_belief_ids"] == ["belief_1"]
    
    # Authorized basis should contain the replacement belief NY, since it's authorized.
    assert len(res["authorized_basis"]) == 1
    assert res["authorized_basis"][0]["belief_id"] == "belief_2"
    assert res["authorized_basis"][0]["proposition"] == "Alice lives in NY."

    # Excluded should contain the old belief SF
    assert len(res["excluded"]) == 1
    assert res["excluded"][0]["belief_id"] == "belief_1"
    assert res["excluded"][0]["status"] == "SUPERSEDED"
    assert res["excluded"][0]["accepted_defeat_path"]["replacement_belief_id"] == "belief_2"
