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
from retracemem.evaluation.jsonl import write_jsonl, read_jsonl


class ManualBatchedEvidenceEdgeVerifier:
    def __init__(self, edges_by_belief: dict[str, list[EvidenceEdge]]) -> None:
        self.edges_by_belief = edges_by_belief
        self.calls: list[tuple[str, ...]] = []

    def verify_edges_batch(
        self,
        new_evidence: EvidenceNode,
        candidate_beliefs: tuple[BeliefNode, ...],
        candidate_replacement_beliefs: tuple[BeliefNode, ...],
        candidate_conditions_by_belief: tuple[tuple[str, tuple[ConditionNode, ...]], ...],
        temporal_context: tuple[EvidenceNode, ...],
    ) -> list[EvidenceEdge]:
        del new_evidence, candidate_replacement_beliefs, candidate_conditions_by_belief, temporal_context
        batch_ids = tuple(belief.belief_id for belief in candidate_beliefs)
        self.calls.append(batch_ids)
        edges: list[EvidenceEdge] = []
        for belief_id in batch_ids:
            edges.extend(self.edges_by_belief.get(belief_id, []))
        return edges


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
    pipeline = ReTracePipeline.for_development_fixture(
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
    pipeline = ReTracePipeline.for_development_fixture(
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
    pipeline = ReTracePipeline.for_development_fixture(
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


def test_pipeline_requires_explicit_backend_or_all_components():
    with pytest.raises(ValueError, match="requires either an explicit backend or all five typed components"):
        ReTracePipeline()
    with pytest.raises(ValueError, match="Missing: inducer, edge_verifier, impact_retriever, query_retriever"):
        ReTracePipeline(extractor=ManualTypedBeliefExtractor({}))


def test_pipeline_answer_blocked_beliefs_are_query_conditioned():
    user_id = "user_qcond"
    b_bike = BeliefNode(
        belief_id="belief_bike",
        proposition="Alice commutes by bicycle.",
        source_evidence_ids=("s1",),
    )
    b_food = BeliefNode(
        belief_id="belief_food",
        proposition="Alice likes sushi.",
        source_evidence_ids=("s1",),
    )
    extractor = ManualTypedBeliefExtractor({"s1": [b_bike, b_food], "s2": []})
    c_mobility = ConditionNode(condition_id="cond_mob", scope_id=user_id, text="mobility")
    dep = DependencyEdge(edge_id="dep_mob", belief_id="belief_bike", condition_id="cond_mob", inducer="manual")
    inducer = ManualRequirementInducer([RequirementProposal(condition=c_mobility, dependency_edge=dep)])
    edge_block = EvidenceEdge(
        edge_id="edge_block_mob",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="s2",
        target_kind="condition",
        target_id="cond_mob",
        verifier="manual",
    )
    edge_verifier = ManualEvidenceEdgeVerifier()
    edge_verifier.register(edge_block, belief_id="belief_bike")
    impact_retriever = ManualImpactCandidateRetriever({"s2": ["belief_bike"]})
    query_retriever = ManualQueryBeliefRetriever({
        "how does Alice commute?": ["belief_bike"],
        "what food does Alice like?": ["belief_food"],
    })
    pipeline = ReTracePipeline.for_development_fixture(
        extractor=extractor,
        inducer=inducer,
        edge_verifier=edge_verifier,
        impact_retriever=impact_retriever,
        query_retriever=query_retriever,
    )
    pipeline.reset_user(user_id)
    pipeline.ingest_evidence(
        user_id,
        EvidenceNode(evidence_id="s1", session_id="s1", timestamp="2026-01-01T00:00:00Z",
                     text="Alice commutes by bicycle and likes sushi.",
                     source_dataset="manual", source_pointer="test"),
    )
    pipeline.ingest_evidence(
        user_id,
        EvidenceNode(evidence_id="s2", session_id="s2", timestamp="2026-01-02T00:00:00Z",
                     text="Alice broke her leg.",
                     source_dataset="manual", source_pointer="test"),
    )

    commute_record = pipeline.answer(user_id, "how does Alice commute?")
    assert commute_record.authorized_basis == []
    assert len(commute_record.blocked_beliefs) == 1
    assert commute_record.blocked_beliefs[0]["belief_id"] == "belief_bike"

    food_record = pipeline.answer(user_id, "what food does Alice like?")
    assert len(food_record.authorized_basis) == 1
    assert food_record.authorized_basis[0]["belief_id"] == "belief_food"
    assert food_record.blocked_beliefs == []


def test_pipeline_answer_record_is_jsonl_compatible(tmp_path):
    user_id = "user_jsonl"
    b = BeliefNode(
        belief_id="belief_food",
        proposition="Alice likes Thai food.",
        source_evidence_ids=("s1",),
    )
    extractor = ManualTypedBeliefExtractor({"s1": [b]})
    pipeline = ReTracePipeline.for_development_fixture(
        extractor=extractor,
        inducer=ManualRequirementInducer([]),
        edge_verifier=ManualEvidenceEdgeVerifier(),
        impact_retriever=ManualImpactCandidateRetriever({}),
        query_retriever=ManualQueryBeliefRetriever({"What food does Alice like?": ["belief_food"]}),
    )
    pipeline.reset_user(user_id)
    pipeline.ingest_evidence(
        user_id,
        EvidenceNode(evidence_id="s1", session_id="s1", timestamp="2026-01-01T00:00:00Z",
                     text="Alice likes Thai food.",
                     source_dataset="manual", source_pointer="test"),
    )
    record = pipeline.answer(user_id, "What food does Alice like?")
    output_path = tmp_path / "answers.jsonl"
    write_jsonl([record], output_path)
    loaded = read_jsonl(output_path)
    assert loaded[0]["query_id"] == f"{user_id}:What food does Alice like?"
    assert loaded[0]["method"] == "retrace_pipeline"
    assert loaded[0]["authorized_basis"][0]["belief_id"] == "belief_food"
    assert loaded[0]["blocked_beliefs"] == []


def test_backend_batched_ingestion_one_batch_within_k():
    user_id = "user_batch_one"
    b1 = BeliefNode(belief_id="b1", proposition="Alice bikes.", source_evidence_ids=("s1",))
    b2 = BeliefNode(belief_id="b2", proposition="Alice runs.", source_evidence_ids=("s1",))
    edge1 = EvidenceEdge(
        edge_id="edge_b1_uncertain", edge_type=EvidenceEdgeType.UNCERTAIN,
        evidence_id="s2", target_kind="belief", target_id="b1", verifier="manual_batch",
    )
    edge2 = EvidenceEdge(
        edge_id="edge_b2_uncertain", edge_type=EvidenceEdgeType.UNCERTAIN,
        evidence_id="s2", target_kind="belief", target_id="b2", verifier="manual_batch",
    )
    batched = ManualBatchedEvidenceEdgeVerifier({"b1": [edge1], "b2": [edge2]})
    backend = ReTraceBackend.for_development_fixture(
        extractor=ManualTypedBeliefExtractor({"s1": [b1, b2], "s2": []}),
        inducer=ManualRequirementInducer([]),
        edge_verifier=ManualEvidenceEdgeVerifier(),
        batched_edge_verifier=batched,
        impact_retriever=ManualImpactCandidateRetriever({"s2": ["b1", "b2"]}),
        query_retriever=ManualQueryBeliefRetriever({}),
        max_batch_beliefs=8,
    )
    backend.reset_user(user_id)
    backend.ingest_evidence(user_id, EvidenceNode(evidence_id="s1", session_id="s1", timestamp=None,
                                                  text="Alice bikes and runs.", source_dataset="manual", source_pointer="test"))
    admitted = backend.ingest_evidence(user_id, EvidenceNode(evidence_id="s2", session_id="s2", timestamp=None,
                                                             text="Alice may have changed activity.", source_dataset="manual", source_pointer="test"))
    assert {edge.edge_id for edge in admitted} == {"edge_b1_uncertain", "edge_b2_uncertain"}
    assert batched.calls == [("b1", "b2")]
    assert backend.last_ingest_stats["execution_mode"] == "batched"
    assert backend.last_ingest_stats["batch_count"] == 1
    assert backend.last_ingest_stats["verifier_calls"] == 1


def test_backend_batched_ingestion_exceeding_k_splits_deterministically():
    user_id = "user_batch_split"
    beliefs = tuple(
        BeliefNode(belief_id=f"b{i}", proposition=f"Belief {i}.", source_evidence_ids=("s1",))
        for i in range(5)
    )
    batched = ManualBatchedEvidenceEdgeVerifier({})
    backend = ReTraceBackend.for_development_fixture(
        extractor=ManualTypedBeliefExtractor({"s1": list(beliefs), "s2": []}),
        inducer=ManualRequirementInducer([]),
        edge_verifier=ManualEvidenceEdgeVerifier(),
        batched_edge_verifier=batched,
        impact_retriever=ManualImpactCandidateRetriever({"s2": [b.belief_id for b in beliefs]}),
        query_retriever=ManualQueryBeliefRetriever({}),
        max_batch_beliefs=2,
    )
    backend.reset_user(user_id)
    backend.ingest_evidence(user_id, EvidenceNode(evidence_id="s1", session_id="s1", timestamp=None,
                                                  text="Initial beliefs.", source_dataset="manual", source_pointer="test"))
    backend.ingest_evidence(user_id, EvidenceNode(evidence_id="s2", session_id="s2", timestamp=None,
                                                  text="Update.", source_dataset="manual", source_pointer="test"))
    assert batched.calls == [("b0", "b1"), ("b2", "b3"), ("b4",)]
    assert backend.last_ingest_stats["batch_count"] == 3
    assert backend.last_ingest_stats["verifier_calls"] == 3


def test_backend_batched_ingestion_does_not_include_unretrieved_global_beliefs():
    user_id = "user_batch_scope"
    b1 = BeliefNode(belief_id="b1", proposition="Impacted.", source_evidence_ids=("s1",))
    b2 = BeliefNode(belief_id="b2", proposition="Unrelated.", source_evidence_ids=("s1",))
    batched = ManualBatchedEvidenceEdgeVerifier({})
    backend = ReTraceBackend.for_development_fixture(
        extractor=ManualTypedBeliefExtractor({"s1": [b1, b2], "s2": []}),
        inducer=ManualRequirementInducer([]),
        edge_verifier=ManualEvidenceEdgeVerifier(),
        batched_edge_verifier=batched,
        impact_retriever=ManualImpactCandidateRetriever({"s2": ["b1"]}),
        query_retriever=ManualQueryBeliefRetriever({}),
    )
    backend.reset_user(user_id)
    backend.ingest_evidence(user_id, EvidenceNode(evidence_id="s1", session_id="s1", timestamp=None,
                                                  text="Initial.", source_dataset="manual", source_pointer="test"))
    backend.ingest_evidence(user_id, EvidenceNode(evidence_id="s2", session_id="s2", timestamp=None,
                                                  text="Update.", source_dataset="manual", source_pointer="test"))
    assert batched.calls == [("b1",)]
    assert backend.last_ingest_stats["candidate_count"] == 1


def test_backend_per_belief_and_batched_paths_same_query_result():
    user_id = "user_batch_equiv"
    b = BeliefNode(belief_id="belief_schedule", proposition="Alice attends standup.", source_evidence_ids=("s1",))
    edge = EvidenceEdge(
        edge_id="edge_uncertain_schedule", edge_type=EvidenceEdgeType.UNCERTAIN,
        evidence_id="s2", target_kind="belief", target_id="belief_schedule", verifier="manual",
    )
    query_retriever = ManualQueryBeliefRetriever({"what is schedule?": ["belief_schedule"]})

    per_edge_verifier = ManualEvidenceEdgeVerifier()
    per_edge_verifier.register(edge, belief_id="belief_schedule")
    per_backend = ReTraceBackend.for_development_fixture(
        extractor=ManualTypedBeliefExtractor({"s1": [b], "s2": []}),
        inducer=ManualRequirementInducer([]),
        edge_verifier=per_edge_verifier,
        impact_retriever=ManualImpactCandidateRetriever({"s2": ["belief_schedule"]}),
        query_retriever=query_retriever,
    )

    batch_backend = ReTraceBackend.for_development_fixture(
        extractor=ManualTypedBeliefExtractor({"s1": [b], "s2": []}),
        inducer=ManualRequirementInducer([]),
        edge_verifier=ManualEvidenceEdgeVerifier(),
        batched_edge_verifier=ManualBatchedEvidenceEdgeVerifier({"belief_schedule": [edge]}),
        impact_retriever=ManualImpactCandidateRetriever({"s2": ["belief_schedule"]}),
        query_retriever=query_retriever,
    )

    for backend in (per_backend, batch_backend):
        backend.reset_user(user_id)
        backend.ingest_evidence(user_id, EvidenceNode(evidence_id="s1", session_id="s1", timestamp=None,
                                                      text="Alice attends standup.", source_dataset="manual", source_pointer="test"))
        backend.ingest_evidence(user_id, EvidenceNode(evidence_id="s2", session_id="s2", timestamp=None,
                                                      text="Schedule may have changed.", source_dataset="manual", source_pointer="test"))

    assert per_backend.search(user_id, "what is schedule?") == batch_backend.search(user_id, "what is schedule?")
