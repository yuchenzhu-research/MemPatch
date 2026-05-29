from __future__ import annotations

import pytest
from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
from experiments.multiagent.contracts import FixedCandidateSubmission

@pytest.fixture
def fake_submission() -> FixedCandidateSubmission:
    from retracemem.schemas import EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
    ev1 = EvidenceNode(
        evidence_id="ev_old", session_id="s1", timestamp="2026-05-29T09:00:00Z",
        text="Old version 1.4", source_dataset="t", source_pointer="p"
    )
    ev2 = EvidenceNode(
        evidence_id="ev_new", session_id="s1", timestamp="2026-05-29T10:00:00Z",
        text="Migrated to 2.0", source_dataset="t", source_pointer="p"
    )
    b = BeliefNode(belief_id="b1", proposition="Version 1.4", source_evidence_ids=("ev_old",))
    b_rep = BeliefNode(belief_id="b2", proposition="Version 2.0", source_evidence_ids=("ev_new",))
    
    return FixedCandidateSubmission(
        submission_id="sub_test_02", producer_id="agent_1", producer_role="role",
        task_id="t1", parent_snapshot_id="snap_0", observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_1", query_id="q_1", query="Q",
        evidence_context=(ev1, ev2), new_evidence_id="ev_new",
        candidate_beliefs=(b,),
        candidate_replacement_beliefs=(b_rep,),
    )


def test_parse_response_empty_evidence_ids(fake_submission):
    policy = PromptTypedRevisionPolicy()
    empty_ev_json = """
    [
      {
        "action_type": "SUPERSEDES",
        "target_belief_id": "b1",
        "replacement_belief_id": "b2",
        "evidence_ids": []
      }
    ]
    """
    out = policy.parse_response(empty_ev_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "requires a non-empty evidence_ids array" in out.errors[0]


def test_parse_response_missing_new_evidence_id(fake_submission):
    policy = PromptTypedRevisionPolicy()
    # evidence_ids has ev_old but lacks new_evidence_id "ev_new"
    missing_new_json = """
    [
      {
        "action_type": "SUPERSEDES",
        "target_belief_id": "b1",
        "replacement_belief_id": "b2",
        "evidence_ids": ["ev_old"]
      }
    ]
    """
    out = policy.parse_response(missing_new_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "must explicitly include submission's new_evidence_id" in out.errors[0]


def test_parse_response_valid_grounding(fake_submission):
    policy = PromptTypedRevisionPolicy()
    valid_json = """
    [
      {
        "action_type": "SUPERSEDES",
        "target_belief_id": "b1",
        "replacement_belief_id": "b2",
        "evidence_ids": ["ev_new", "ev_old"]
      }
    ]
    """
    out = policy.parse_response(valid_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is True
    assert len(out.errors) == 0
    assert len(out.proposal_batches) == 1
    assert out.proposal_batches[0].edges[0].evidence_id == "ev_new"


def test_parse_response_no_revision_grounding(fake_submission):
    policy = PromptTypedRevisionPolicy()
    
    # 1. NO_REVISION missing ev_new is invalid
    invalid_no_rev = """
    [
      {
        "action_type": "NO_REVISION",
        "evidence_ids": ["ev_old"]
      }
    ]
    """
    out1 = policy.parse_response(invalid_no_rev, example_id="ex_1", submission=fake_submission)
    assert out1.parsing_valid is False
    assert "must explicitly include" in out1.errors[0]

    # 2. NO_REVISION containing ev_new is valid
    valid_no_rev = """
    [
      {
        "action_type": "NO_REVISION",
        "evidence_ids": ["ev_new"]
      }
    ]
    """
    out2 = policy.parse_response(valid_no_rev, example_id="ex_1", submission=fake_submission)
    assert out2.parsing_valid is True
    assert len(out2.errors) == 0
    assert out2.proposal_batches == ()
    assert out2.parsed_actions[0].action_type == "NO_REVISION"
