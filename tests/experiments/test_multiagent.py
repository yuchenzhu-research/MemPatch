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
from retracemem.multiagent.commit import commit_subagent_submission, order_subagent_submissions
from experiments.cupmem_adapter import CupMemRevisionCandidate, map_cupmem_candidate_to_subagent_submission


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


def test_cupmem_candidate_mapping():
    ev_1 = EvidenceNode(
        evidence_id="e_1", session_id="s_1", timestamp="2026-05-29T10:00:00Z",
        text="A", source_dataset="t", source_pointer="p"
    )
    b_1 = BeliefNode(belief_id="b_1", proposition="A", source_evidence_ids=("e_1",))

    candidate = CupMemRevisionCandidate(
        submission_id="sub_c_1",
        producer_id="agent_c",
        producer_role="role_c",
        parent_snapshot_id="snap_0",
        observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_1",
        query_id="q_1",
        query="Q",
        evidence_context=(ev_1,),
        new_evidence_id="e_1",
        candidate_beliefs=(b_1,),
        upstream_trace={"task_id": "task_cup"}
    )

    sub = map_cupmem_candidate_to_subagent_submission(candidate)
    assert sub.submission_id == "sub_c_1"
    assert sub.task_id == "task_cup"
    assert sub.metadata == {"task_id": "task_cup"}
