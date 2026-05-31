from __future__ import annotations

import pytest
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
    EvidenceEdge,
    EvidenceEdgeType,
)
from retracemem.authorization import EvidenceProposalBatch
from retracemem.multiagent.contracts import SubagentMemorySubmission, SharedMemoryCommitResult
from retracemem.multiagent.commit import commit_subagent_submission, order_subagent_submissions, commit_submission_sequence


def test_commit_subagent_submission_validation():
    ev_init = EvidenceNode(
        evidence_id="e_init", session_id="sess_0", timestamp="2026-05-29T10:00:00Z",
        text="User lives in Seattle.", source_dataset="stale_dev", source_pointer="dev://1"
    )
    b_location = BeliefNode(
        belief_id="b_location", proposition="User lives in Seattle.", source_evidence_ids=("e_init",)
    )

    # 1. Invalid new_evidence_id (does not exist in context)
    submission_invalid_new_ev = SubagentMemorySubmission(
        submission_id="sub_test_01",
        producer_id="agent_1",
        producer_role="test_role",
        parent_snapshot_id="snap_root",
        observed_at="2026-05-29T11:00:00Z",
        instance_id="inst_1",
        query_id="q_1",
        query="What is the location?",
        evidence_context=(ev_init,),
        new_evidence_id="e_missing",
        candidate_beliefs=(b_location,)
    )

    with pytest.raises(ValueError, match="new_evidence_id 'e_missing' must match exactly one"):
        commit_subagent_submission(submission_invalid_new_ev)

    # 2. Missing referenced source evidence in belief
    b_invalid_ref = BeliefNode(
        belief_id="b_invalid", proposition="Invalid belief.", source_evidence_ids=("e_unrepresented",)
    )
    submission_missing_evidence = SubagentMemorySubmission(
        submission_id="sub_test_02",
        producer_id="agent_1",
        producer_role="test_role",
        parent_snapshot_id="snap_root",
        observed_at="2026-05-29T11:00:00Z",
        instance_id="inst_1",
        query_id="q_1",
        query="What is the location?",
        evidence_context=(ev_init,),
        new_evidence_id="e_init",
        candidate_beliefs=(b_invalid_ref,)
    )

    with pytest.raises(ValueError, match="references source evidence 'e_unrepresented' not present in evidence_context"):
        commit_subagent_submission(submission_missing_evidence)


def test_deterministic_trace_and_ordering():
    ev_1 = EvidenceNode(
        evidence_id="e_1", session_id="s_1", timestamp="2026-05-29T10:00:00Z",
        text="A", source_dataset="t", source_pointer="p"
    )
    b_1 = BeliefNode(belief_id="b_1", proposition="A", source_evidence_ids=("e_1",))
    
    sub1 = SubagentMemorySubmission(
        submission_id="sub_1", producer_id="agent_1", producer_role="role",
        parent_snapshot_id="snap_0", observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_1", query_id="q_1", query="Q",
        evidence_context=(ev_1,), new_evidence_id="e_1", candidate_beliefs=(b_1,)
    )

    sub2 = SubagentMemorySubmission(
        submission_id="sub_2", producer_id="agent_2", producer_role="role",
        parent_snapshot_id="snap_0", observed_at="2026-05-29T09:00:00Z",
        instance_id="inst_1", query_id="q_1", query="Q",
        evidence_context=(ev_1,), new_evidence_id="e_1", candidate_beliefs=(b_1,)
    )

    # Test deterministic ordering
    ordered = order_subagent_submissions((sub1, sub2))
    assert ordered[0].submission_id == "sub_2"
    assert ordered[1].submission_id == "sub_1"

    # Test commit deterministic next_snapshot_id
    res1 = commit_subagent_submission(sub1)
    res2 = commit_subagent_submission(sub1)
    assert res1.next_snapshot_id == res2.next_snapshot_id
    assert "producer_id" in res1.commit_trace
    assert res1.commit_trace["producer_id"] == "agent_1"


def test_commit_submission_sequence_basic():
    ev_1 = EvidenceNode(
        evidence_id="e_1", session_id="s_1", timestamp="2026-05-29T10:00:00Z",
        text="Deploy in us-east-1.", source_dataset="t", source_pointer="p"
    )
    ev_2 = EvidenceNode(
        evidence_id="e_2", session_id="s_1", timestamp="2026-05-29T11:00:00Z",
        text="Deploy changed to eu-west-1.", source_dataset="t", source_pointer="p"
    )
    b_east = BeliefNode(belief_id="b_east", proposition="Deploy in us-east-1.", source_evidence_ids=("e_1",))
    b_west = BeliefNode(belief_id="b_west", proposition="Deploy in eu-west-1.", source_evidence_ids=("e_2",))

    edge_super = EvidenceEdge(
        edge_id="edge_super", edge_type=EvidenceEdgeType.SUPERSEDES,
        evidence_id="e_2", target_kind="belief", target_id="b_east",
        verifier="deployer", replacement_belief_id="b_west"
    )

    sub1 = SubagentMemorySubmission(
        submission_id="sub_1", producer_id="agent_1", producer_role="role",
        parent_snapshot_id="snap_0", observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_1", query_id="q_1", query="Q",
        evidence_context=(ev_1,), new_evidence_id="e_1",
        candidate_beliefs=(b_east,)
    )

    sub2 = SubagentMemorySubmission(
        submission_id="sub_2", producer_id="agent_1", producer_role="role",
        parent_snapshot_id="snap_1", observed_at="2026-05-29T11:00:00Z",
        instance_id="inst_1", query_id="q_1", query="Q",
        evidence_context=(ev_1, ev_2), new_evidence_id="e_2",
        candidate_beliefs=(),
        candidate_replacement_beliefs=(b_west,),
        proposal_batches=(EvidenceProposalBatch(edges=(edge_super,)),)
    )

    # Execute sequence commit
    seq_res = commit_submission_sequence((sub1, sub2), final_snapshot_evaluation=True)

    assert seq_res.initial_snapshot_id == "snap_0"
    assert seq_res.final_snapshot_id is not None
    assert len(seq_res.submission_results) == 2
    assert seq_res.final_belief_statuses.get("b_east") == "SUPERSEDED"
    assert seq_res.final_belief_statuses.get("b_west") == "AUTHORIZED"
    assert seq_res.final_authorized_belief_ids == ("b_west",)
    assert seq_res.final_excluded_belief_ids == ("b_east",)
    assert seq_res.trace["number_of_submissions"] == 2
