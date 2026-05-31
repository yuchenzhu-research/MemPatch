from __future__ import annotations

import pytest
from typing import Any
from retracemem.schemas import (
    EvidenceNode,
    EvidenceEdge,
    BeliefNode,
    EvidenceEdgeType,
    ConditionNode,
    DependencyEdge,
)
from retracemem.methods.contracts import SharedCandidateView
from experiments.archive.stale_adapter import (
    split_stale_record,
    assert_no_evaluation_leakage,
    iter_chronological_sessions,
    StaleWriteHistory,
    StaleProbeTask,
    StaleGoldRecord,
)
from retracemem.authorization import authorize, EvidenceProposalBatch


@pytest.fixture
def sample_raw_record() -> dict[str, Any]:
    return {
        "uid": "test_uid_123",
        "M_old": "User lives in Seattle.",
        "M_new": "User lives in Denver.",
        "explanation": "User relocated to Denver.",
        "probing_queries": {
            "dim1_query": "Does the user live in Seattle?",
            "dim2_query": "Recommend Seattle activities.",
            "dim3_query": "Where is user located?"
        },
        "relevant_session_index": [49],
        "timestamps": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        "haystack_session": [
            [
                {"role": "user", "content": "I live in Seattle."},
                {"role": "assistant", "content": "Got it!"}
            ],
            [
                {"role": "user", "content": "I am moving to Denver."},
                {"role": "assistant", "content": "Safe travels!"}
            ]
        ],
        "type": "T1"
    }


def test_split_stale_record(sample_raw_record):
    history, probes, gold = split_stale_record(sample_raw_record)
    
    assert history.uid == "test_uid_123"
    assert len(history.sessions) == 2
    assert history.sessions[0][0].role == "user"
    assert history.sessions[0][0].content == "I live in Seattle."
    assert history.timestamps == ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z")
    
    assert len(probes) == 3
    assert {p.dimension for p in probes} == {"dim1_query", "dim2_query", "dim3_query"}
    assert probes[0].uid == "test_uid_123"
    
    assert gold.uid == "test_uid_123"
    assert gold.m_old == "User lives in Seattle."
    assert gold.m_new == "User lives in Denver."
    assert gold.explanation == "User relocated to Denver."
    assert gold.relevant_session_index == (49,)
    assert gold.conflict_type == "T1"


def test_leakage_assertion():
    assert_no_evaluation_leakage({"clean_key": "clean_val"})
    
    with pytest.raises(ValueError, match="Evaluation leakage detected"):
        assert_no_evaluation_leakage({"M_old": "leak"})
        
    with pytest.raises(ValueError, match="Evaluation leakage detected"):
        assert_no_evaluation_leakage({"type": "leak"})
        
    with pytest.raises(ValueError, match="Evaluation leakage detected"):
        bad_history = StaleWriteHistory(
            uid="uid",
            sessions=((),),
            timestamps=(),
        )
        object.__setattr__(bad_history, "m_old", "leak")
        assert_no_evaluation_leakage(bad_history)


def test_iter_chronological_sessions(sample_raw_record):
    history, _, _ = split_stale_record(sample_raw_record)
    nodes = iter_chronological_sessions(history)
    assert len(nodes) == 2
    assert isinstance(nodes[0], EvidenceNode)
    assert nodes[0].evidence_id == "e_0"
    assert nodes[0].session_id == "s_0"
    assert nodes[0].timestamp == "2026-01-01T00:00:00Z"
    assert nodes[0].text == "user: I live in Seattle.\nassistant: Got it!"
    assert nodes[0].source_dataset == "stale"
    assert nodes[0].source_pointer == "stale://test_uid_123/0"


def test_frozen_probes_binding(sample_raw_record):
    history, probes, _ = split_stale_record(sample_raw_record)
    snapshot_id = "snap_789"
    bound_probes = [
        StaleProbeTask(
            uid=p.uid,
            dimension=p.dimension,
            query=p.query,
            memory_snapshot_id=snapshot_id
        )
        for p in probes
    ]
    for bp in bound_probes:
        assert bp.memory_snapshot_id == "snap_789"


def test_authorize_with_proposals():
    ev_node = EvidenceNode(
        evidence_id="e_0",
        session_id="s_0",
        timestamp="2026-01-01T00:00:00Z",
        text="user: I live in Seattle.",
        source_dataset="test",
        source_pointer="test_ptr",
    )
    b_node = BeliefNode(
        belief_id="b_0",
        proposition="User lives in Seattle.",
        source_evidence_ids=("e_0",),
    )
    view = SharedCandidateView(
        instance_id="inst_1",
        query_id="q_1",
        query="Does the user live in Seattle?",
        evidence_context=(ev_node,),
        new_evidence=ev_node,
        candidate_beliefs=(b_node,),
        candidate_replacement_beliefs=(),
    )
    res = authorize(view, proposal_batches=())
    assert "b_0" in res.authorized_belief_ids
    
    res_zero = authorize(
        view,
        proposal_batches=(
            EvidenceProposalBatch(edges=(), model_call_trace_id="trace_zero"),
        )
    )
    assert "b_0" in res_zero.authorized_belief_ids
    assert "trace_zero" in res_zero.trace["model_call_trace_ids"]


def test_map_cupmem_candidate_to_subagent_submission():
    from experiments.archive.cupmem_adapter import CupMemRevisionCandidate, map_cupmem_candidate_to_subagent_submission
    import pytest

    ev_old = EvidenceNode(
        evidence_id="e_old", session_id="s_0", timestamp="2026-01-01T00:00:00Z",
        text="User commutes by bicycle.", source_dataset="test", source_pointer="ptr"
    )
    ev_new = EvidenceNode(
        evidence_id="e_new", session_id="s_1", timestamp="2026-01-02T00:00:00Z",
        text="User broke their leg.", source_dataset="test", source_pointer="ptr"
    )
    b_bike = BeliefNode(belief_id="b_bike", proposition="User commutes by bicycle.", source_evidence_ids=("e_old",))

    candidate = CupMemRevisionCandidate(
        submission_id="sub_1",
        producer_id="subagent_1",
        producer_role="role_1",
        parent_snapshot_id="snap_0",
        observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_123",
        query_id="q_999",
        query="Where does user live?",
        evidence_context=(ev_old, ev_new),
        new_evidence_id="e_new",
        candidate_beliefs=(b_bike,),
        candidate_replacement_beliefs=(),
        upstream_trace={"task_id": "task_1", "extra": "info"}
    )

    submission = map_cupmem_candidate_to_subagent_submission(candidate)
    assert submission.submission_id == "sub_1"
    assert submission.producer_id == "subagent_1"
    assert submission.producer_role == "role_1"
    assert submission.parent_snapshot_id == "snap_0"
    assert submission.observed_at == "2026-05-29T10:00:00Z"
    assert submission.instance_id == "inst_123"
    assert submission.query_id == "q_999"
    assert submission.query == "Where does user live?"
    assert submission.evidence_context == (ev_old, ev_new)
    assert submission.new_evidence_id == "e_new"
    assert submission.candidate_beliefs == (b_bike,)
    assert submission.task_id == "task_1"
    assert submission.metadata == {"task_id": "task_1", "extra": "info"}

    # Test leakage prevention in CupMemRevisionCandidate upstream_trace
    leaky_candidate = CupMemRevisionCandidate(
        submission_id="sub_1",
        producer_id="subagent_1",
        producer_role="role_1",
        parent_snapshot_id="snap_0",
        observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_123",
        query_id="q_999",
        query="Where does user live?",
        evidence_context=(ev_old, ev_new),
        new_evidence_id="e_new",
        candidate_beliefs=(b_bike,),
        upstream_trace={"M_old": "some gold memory"}
    )
    with pytest.raises(ValueError, match="Evaluation leakage detected"):
        map_cupmem_candidate_to_subagent_submission(leaky_candidate)


def test_revision_safety_diagnostic_cases():
    ev_old = EvidenceNode(
        evidence_id="e_old", session_id="s_0", timestamp="2026-01-01T00:00:00Z",
        text="User commutes by bicycle.", source_dataset="test", source_pointer="ptr"
    )
    ev_new = EvidenceNode(
        evidence_id="e_new", session_id="s_1", timestamp="2026-01-02T00:00:00Z",
        text="User broke their leg.", source_dataset="test", source_pointer="ptr"
    )
    b_bike = BeliefNode(belief_id="b_bike", proposition="User commutes by bicycle.", source_evidence_ids=("e_old",))
    c_phys = ConditionNode(condition_id="c_phys", scope_id="user1", text="User is physically able.")
    dep = DependencyEdge(edge_id="dep1", belief_id="b_bike", condition_id="c_phys", inducer="test")
    
    view = SharedCandidateView(
        instance_id="inst_diag",
        query_id="q_diag",
        query="How does user commute?",
        evidence_context=(ev_old, ev_new),
        new_evidence=ev_new,
        candidate_beliefs=(b_bike,),
        candidate_replacement_beliefs=(),
        candidate_conditions_by_belief=(("b_bike", (c_phys,)),),
        dependency_edges_by_belief=(("b_bike", (dep,)),)
    )
    
    res_protected = authorize(view, proposal_batches=())
    assert "b_bike" in res_protected.authorized_belief_ids
    
    block_edge = EvidenceEdge(
        edge_id="edge_block",
        edge_type=EvidenceEdgeType.BLOCKS,
        evidence_id="e_new",
        target_kind="condition",
        target_id="c_phys",
        verifier="test"
    )
    res_block = authorize(
        view,
        proposal_batches=(
            EvidenceProposalBatch(edges=(block_edge,), model_call_trace_id="trace_block"),
        )
    )
    assert "b_bike" in res_block.excluded_belief_ids
    assert res_block.trace["fine_grained_statuses"]["b_bike"] == "BLOCKED"
