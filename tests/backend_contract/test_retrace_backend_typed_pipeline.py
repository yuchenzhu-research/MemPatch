from __future__ import annotations

import pytest
from retracemem.backends.retrace_backend import ReTraceBackend
from retracemem.pipeline import ReTracePipeline
from retracemem.schemas import (
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
    EvidenceNode,
)
from retracemem.extraction.typed_extractor import ManualTypedBeliefExtractor
from retracemem.verifier.requirement_inducer import ManualRequirementInducer
from retracemem.verifier.contracts import RequirementProposal
from retracemem.verifier.evidence_edge_verifier import ManualEvidenceEdgeVerifier
from retracemem.retrieval.typed_retrievers import (
    ManualImpactCandidateRetriever,
    ManualQueryBeliefRetriever,
)


class _FailingClient:
    def __init__(self) -> None:
        self.called = False

    def generate(self, **kwargs):
        del kwargs
        self.called = True
        raise AssertionError("Wave 2 backend must not call external clients")


def test_pipeline_blocker_and_release_flow():
    user_id = "user_test"

    # Setup Manual Extractor
    # Session 1: Commute
    b_commute = BeliefNode(
        belief_id="belief_commute",
        proposition="Alice usually commutes by bicycle.",
        source_evidence_ids=("session_1",),
    )
    # Session 2: Broken leg
    # Session 3: Recovered
    extractor = ManualTypedBeliefExtractor({
        "session_1": [b_commute],
        "session_2": [],
        "session_3": [],
    })

    # Setup Manual Inducer
    c_mobility = ConditionNode(
        condition_id="cond_mobility",
        scope_id=user_id,
        text="Alice has sufficient mobility.",
    )
    dep_commute = DependencyEdge(
        edge_id="dep_commute",
        belief_id="belief_commute",
        condition_id="cond_mobility",
        inducer="manual",
    )
    inducer = ManualRequirementInducer([
        RequirementProposal(condition=c_mobility, dependency_edge=dep_commute)
    ])

    # Setup Manual Impact Retriever
    impact_retriever = ManualImpactCandidateRetriever({
        "session_1": [],
        "session_2": ["belief_commute"],
        "session_3": ["belief_commute"],
    })

    # Setup Manual Edge Verifier
    edge_block = EvidenceEdge(
        edge_id="edge_block_mobility",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="session_2",
        target_kind="condition",
        target_id="cond_mobility",
        verifier="manual",
    )
    edge_release = EvidenceEdge(
        edge_id="edge_release_mobility",
        edge_type=EvidenceEdgeType.RELEASES,
        evidence_id="session_3",
        target_kind="condition",
        target_id="cond_mobility",
        verifier="manual",
    )
    edge_verifier = ManualEvidenceEdgeVerifier()
    edge_verifier.register(edge_block, belief_id="belief_commute")
    edge_verifier.register(edge_release, belief_id="belief_commute")

    # Setup Query Retriever
    query_retriever = ManualQueryBeliefRetriever({
        "how does Alice commute?": ["belief_commute"]
    })

    # Instantiate Pipeline
    pipeline = ReTracePipeline(
        extractor=extractor,
        inducer=inducer,
        edge_verifier=edge_verifier,
        impact_retriever=impact_retriever,
        query_retriever=query_retriever,
    )

    pipeline.reset_user(user_id)

    # 1. Ingest Session 1: Commute belief added
    ev1 = EvidenceNode(
        evidence_id="session_1",
        session_id="session_1",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice usually commutes by bicycle.",
        source_dataset="manual",
        source_pointer="test",
    )
    pipeline.ingest_evidence(user_id, ev1)

    # Verify belief commutes is authorized
    basis = pipeline.authorized_basis(user_id, "how does Alice commute?")
    assert len(basis) == 1
    assert basis[0]["belief_id"] == "belief_commute"

    # 2. Ingest Session 2: Broken leg blocks mobility
    ev2 = EvidenceNode(
        evidence_id="session_2",
        session_id="session_2",
        timestamp="2026-05-28T01:00:00Z",
        text="Alice broke her leg.",
        source_dataset="manual",
        source_pointer="test",
    )
    admitted = pipeline.ingest_evidence(user_id, ev2)
    assert len(admitted) == 1
    assert admitted[0].edge_id == "edge_block_mobility"

    # Verify belief is now blocked / excluded
    basis = pipeline.authorized_basis(user_id, "how does Alice commute?")
    assert len(basis) == 0
    blocked_result = pipeline.backend.search(user_id, "how does Alice commute?")
    assert blocked_result["authorized_basis"] == []
    assert len(blocked_result["excluded"]) == 1
    assert blocked_result["excluded"][0]["belief_id"] == "belief_commute"
    assert blocked_result["excluded"][0]["status"] == "BLOCKED"
    assert blocked_result["excluded"][0]["accepted_defeat_path"]["path_type"] == "PREREQUISITE_BLOCK"

    # Verify EvaluationRecord blocked/excluded structures
    record = pipeline.answer(user_id, "how does Alice commute?")
    assert record.authorized_basis == []
    assert len(record.blocked_beliefs) == 1
    assert record.blocked_beliefs[0]["belief_id"] == "belief_commute"
    assert record.blocked_beliefs[0]["reason"] == "BLOCKED"

    # 3. Ingest Session 3: Recovered releases mobility blocker
    ev3 = EvidenceNode(
        evidence_id="session_3",
        session_id="session_3",
        timestamp="2026-05-28T02:00:00Z",
        text="Alice recovered completely.",
        source_dataset="manual",
        source_pointer="test",
    )
    admitted = pipeline.ingest_evidence(user_id, ev3)
    assert len(admitted) == 1
    assert admitted[0].edge_id == "edge_release_mobility"

    # Verify belief commutes is authorized again
    basis = pipeline.authorized_basis(user_id, "how does Alice commute?")
    assert len(basis) == 1
    assert basis[0]["belief_id"] == "belief_commute"


def test_pipeline_supersedes_flow():
    user_id = "user_test_2"

    # Setup Manual Extractor
    # Session 4: Lives in NYC
    b_nyc = BeliefNode(
        belief_id="belief_nyc",
        proposition="Alice lives in NYC.",
        source_evidence_ids=("session_4",),
    )
    # Session 5: Moves to Chicago
    b_chicago = BeliefNode(
        belief_id="belief_chicago",
        proposition="Alice lives in Chicago.",
        source_evidence_ids=("session_5",),
    )
    extractor = ManualTypedBeliefExtractor({
        "session_4": [b_nyc],
        "session_5": [b_chicago],
    })

    # Setup Manual Inducer (none needed)
    inducer = ManualRequirementInducer([])

    # Setup Manual Impact Retriever
    impact_retriever = ManualImpactCandidateRetriever({
        "session_4": [],
        "session_5": ["belief_nyc"],
    })

    # Setup Manual Edge Verifier
    edge_super = EvidenceEdge(
        edge_id="edge_super_nyc",
        edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="session_5",
        target_kind="belief",
        target_id="belief_nyc",
        verifier="manual",
        replacement_belief_id="belief_chicago",
    )
    edge_verifier = ManualEvidenceEdgeVerifier()
    edge_verifier.register(edge_super, belief_id="belief_nyc")

    # Setup Query Retriever (only queries old belief)
    query_retriever = ManualQueryBeliefRetriever({
        "where does Alice live?": ["belief_nyc"]
    })

    # Instantiate Pipeline
    pipeline = ReTracePipeline(
        extractor=extractor,
        inducer=inducer,
        edge_verifier=edge_verifier,
        impact_retriever=impact_retriever,
        query_retriever=query_retriever,
    )

    pipeline.reset_user(user_id)

    # 1. Ingest Session 4: NYC belief added
    ev4 = EvidenceNode(
        evidence_id="session_4",
        session_id="session_4",
        timestamp="2026-05-28T00:00:00Z",
        text="Alice resides in NYC.",
        source_dataset="manual",
        source_pointer="test",
    )
    pipeline.ingest_evidence(user_id, ev4)

    # Verify NYC is authorized
    basis = pipeline.authorized_basis(user_id, "where does Alice live?")
    assert len(basis) == 1
    assert basis[0]["belief_id"] == "belief_nyc"

    # 2. Ingest Session 5: Move to Chicago supersedes NYC
    ev5 = EvidenceNode(
        evidence_id="session_5",
        session_id="session_5",
        timestamp="2026-05-28T01:00:00Z",
        text="Alice moved to Chicago.",
        source_dataset="manual",
        source_pointer="test",
    )
    admitted = pipeline.ingest_evidence(user_id, ev5)
    assert len(admitted) == 1
    assert admitted[0].edge_id == "edge_super_nyc"

    # Verify Chicago is surfaced as replacement when querying NYC
    basis = pipeline.authorized_basis(user_id, "where does Alice live?")
    assert len(basis) == 1
    assert basis[0]["belief_id"] == "belief_chicago"
    assert basis[0]["proposition"] == "Alice lives in Chicago."

    search_result = pipeline.backend.search(user_id, "where does Alice live?")
    assert search_result["authorized_basis"] == basis
    assert len(search_result["excluded"]) == 1
    assert search_result["excluded"][0]["belief_id"] == "belief_nyc"
    assert search_result["excluded"][0]["status"] == "SUPERSEDED"
    assert search_result["excluded"][0]["accepted_defeat_path"]["replacement_belief_id"] == "belief_chicago"


def test_backend_search_retains_unresolved_excluded_trace():
    user_id = "user_test_3"
    belief = BeliefNode(
        belief_id="belief_schedule",
        proposition="Alice attends morning standup.",
        source_evidence_ids=("session_6",),
    )
    extractor = ManualTypedBeliefExtractor({"session_6": [belief], "session_7": []})
    edge_uncertain = EvidenceEdge(
        edge_id="edge_uncertain_schedule",
        edge_type=EvidenceEdgeType.UNCERTAIN,
        evidence_id="session_7",
        target_kind="belief",
        target_id="belief_schedule",
        verifier="manual",
    )
    edge_verifier = ManualEvidenceEdgeVerifier()
    edge_verifier.register(edge_uncertain, belief_id="belief_schedule")
    pipeline = ReTracePipeline(
        extractor=extractor,
        inducer=ManualRequirementInducer([]),
        edge_verifier=edge_verifier,
        impact_retriever=ManualImpactCandidateRetriever({"session_7": ["belief_schedule"]}),
        query_retriever=ManualQueryBeliefRetriever({"what is Alice schedule?": ["belief_schedule"]}),
    )
    pipeline.reset_user(user_id)
    pipeline.ingest_evidence(
        user_id,
        EvidenceNode(
            evidence_id="session_6",
            session_id="session_6",
            timestamp="2026-05-28T00:00:00Z",
            text="Alice attends morning standup.",
            source_dataset="manual",
            source_pointer="test",
        ),
    )
    pipeline.ingest_evidence(
        user_id,
        EvidenceNode(
            evidence_id="session_7",
            session_id="session_7",
            timestamp="2026-05-28T01:00:00Z",
            text="Alice may have changed standup plans.",
            source_dataset="manual",
            source_pointer="test",
        ),
    )

    assert pipeline.authorized_basis(user_id, "what is Alice schedule?") == []
    result = pipeline.backend.search(user_id, "what is Alice schedule?")
    assert result["retrieved_belief_ids"] == ["belief_schedule"]
    assert result["authorized_basis"] == []
    assert len(result["excluded"]) == 1
    assert result["excluded"][0]["status"] == "UNRESOLVED"
    assert result["excluded"][0]["accepted_defeat_path"]["path_type"] == "UNRESOLVED_UNCERTAIN"


def test_backend_answer_is_offline_only_and_rejects_client():
    client = _FailingClient()
    with pytest.raises(ValueError, match="API-backed answer generation belongs to a later Stage A wrapper"):
        ReTraceBackend.for_development_fixture(client=client)
    assert client.called is False

    backend = ReTraceBackend.for_development_fixture()
    answer = backend.answer(
        "user",
        "query",
        {"authorized_basis": [{"proposition": "offline fact"}], "excluded": []},
    )
    assert answer == "Query: query\nAuthorized basis:\noffline fact"


def test_canonical_backend_requires_explicit_components():
    with pytest.raises(ValueError, match="requires explicit typed components"):
        ReTraceBackend()


def test_unsupported_backend_ablation_flags_fail_loudly():
    with pytest.raises(ValueError, match="does not support disable_ledger"):
        ReTraceBackend.for_development_fixture(disable_ledger=True)
    with pytest.raises(ValueError, match="does not support disable_ledger"):
        ReTraceBackend.for_development_fixture(disable_gate=True)
    with pytest.raises(ValueError, match="does not support disable_ledger"):
        ReTraceBackend.for_development_fixture(disable_temporal=True)
