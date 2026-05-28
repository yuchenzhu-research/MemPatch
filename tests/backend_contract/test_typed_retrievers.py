from __future__ import annotations

from retracemem.retrieval.typed_retrievers import (
    ManualImpactCandidateRetriever,
    ManualQueryBeliefRetriever,
    ImpactCandidate,
)
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceNode,
)
from retracemem.memory.belief_store import BeliefStore


def test_write_time_retriever_consumes_new_evidence_and_prior_beliefs():
    store = BeliefStore()
    
    # Store beliefs and condition
    belief_1 = BeliefNode(belief_id="belief_1", proposition="Alice in SF", source_evidence_ids=("ev_0",))
    belief_2 = BeliefNode(belief_id="belief_2", proposition="Bob in NY", source_evidence_ids=("ev_0",))
    store.add_belief(belief_1)
    store.add_belief(belief_2)
    
    cond_1 = ConditionNode(condition_id="cond_1", scope_id="user_alice", text="Alice must have keys")
    store.add_condition(cond_1)
    
    dep_edge = DependencyEdge(
        edge_id="dep_1",
        belief_id="belief_1",
        condition_id="cond_1",
        inducer="manual",
    )
    store.add_dependency_edge(dep_edge)

    new_evidence = EvidenceNode(
        evidence_id="ev_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice lost her keys.",
        source_dataset="manual_audit",
        source_pointer="test",
    )

    # ManualImpactCandidateRetriever maps ev_1 -> belief_1
    retriever = ManualImpactCandidateRetriever(impact_map={"ev_1": ["belief_1"]})
    prior_beliefs = (belief_1, belief_2)
    
    candidates = retriever.retrieve_impacts(
        new_evidence=new_evidence,
        prior_beliefs=prior_beliefs,
        store=store,
    )
    
    assert len(candidates) == 1
    assert candidates[0].belief == belief_1
    assert candidates[0].conditions == (cond_1,)


def test_query_time_retriever_consumes_query():
    belief_1 = BeliefNode(belief_id="belief_1", proposition="Alice in SF", source_evidence_ids=("ev_0",))
    belief_2 = BeliefNode(belief_id="belief_2", proposition="Bob in NY", source_evidence_ids=("ev_0",))
    
    retriever = ManualQueryBeliefRetriever(query_map={"Where does Alice live?": ["belief_1"]})
    
    results = retriever.retrieve_for_query(
        query="Where does Alice live?",
        beliefs=(belief_1, belief_2),
    )
    
    assert results == [belief_1]


def test_irrelevant_beliefs_not_returned():
    belief_1 = BeliefNode(belief_id="belief_1", proposition="Alice in SF", source_evidence_ids=("ev_0",))
    belief_2 = BeliefNode(belief_id="belief_2", proposition="Bob in NY", source_evidence_ids=("ev_0",))
    
    retriever = ManualQueryBeliefRetriever(query_map={"Where does Alice live?": ["belief_1"]})
    
    # query for Bob should not return belief_1
    results = retriever.retrieve_for_query(
        query="Where does Bob live?",
        beliefs=(belief_1, belief_2),
    )
    
    assert results == []


def test_no_legacy_imports():
    # Verify no legacy imports in typed_retrievers
    with open("src/retracemem/retrieval/typed_retrievers.py", "r") as f:
        content = f.read()
    
    assert "RelationPrediction" not in content
    assert "EpisodicEvidence" not in content
    assert "Belief " not in content
    assert "CandidateRelationRetriever" not in content
