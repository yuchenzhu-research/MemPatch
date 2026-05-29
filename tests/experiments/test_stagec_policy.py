from __future__ import annotations

import pytest
from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
from experiments.multiagent.contracts import FixedCandidateSubmission

@pytest.fixture
def fake_submission() -> FixedCandidateSubmission:
    from retracemem.schemas import EvidenceNode, BeliefNode
    ev = EvidenceNode(
        evidence_id="ev_1", session_id="s1", timestamp="2026-05-29T10:00:00Z",
        text="Deploy us-east-1", source_dataset="t", source_pointer="p"
    )
    b = BeliefNode(belief_id="b1", proposition="Deploy us-east-1", source_evidence_ids=("ev_1",))
    return FixedCandidateSubmission(
        submission_id="sub_test_01", producer_id="agent_1", producer_role="role",
        task_id="t1", parent_snapshot_id="snap_0", observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_1", query_id="q_1", query="Q",
        evidence_context=(ev,), new_evidence_id="ev_1", candidate_beliefs=(b,)
    )


def test_build_messages(fake_submission):
    policy = PromptTypedRevisionPolicy()
    messages = policy.build_messages(fake_submission)
    
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "You are the ReTrace Stage C revision policy" in messages[0]["content"]
    assert "sub_test_01" in messages[1]["content"]


def test_parse_response_supersedes(fake_submission):
    policy = PromptTypedRevisionPolicy()
    valid_json = """
    [
      {
        "action_type": "SUPERSEDES",
        "target_belief_id": "b_old",
        "replacement_belief_id": "b_new",
        "rationale": "Superseding old info.",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(valid_json, example_id="ex_1", submission_id=fake_submission.submission_id)
    assert out.parsing_valid is True
    assert len(out.errors) == 0
    assert len(out.proposal_batches) == 1
    edge = out.proposal_batches[0].edges[0]
    assert edge.edge_type.value == "SUPERSEDES"
    assert edge.target_id == "b_old"
    assert edge.replacement_belief_id == "b_new"
    assert edge.evidence_id == "ev_1"


def test_parse_response_blocks(fake_submission):
    policy = PromptTypedRevisionPolicy()
    valid_json = """
    [
      {
        "action_type": "BLOCKS",
        "target_condition_id": "c_blocked",
        "rationale": "Blocking condition.",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(valid_json, example_id="ex_1", submission_id=fake_submission.submission_id)
    assert out.parsing_valid is True
    assert len(out.errors) == 0
    assert len(out.proposal_batches) == 1
    edge = out.proposal_batches[0].edges[0]
    assert edge.edge_type.value == "BLOCKS"
    assert edge.target_kind == "condition"
    assert edge.target_id == "c_blocked"
    assert edge.evidence_id == "ev_1"


def test_parse_response_no_revision(fake_submission):
    policy = PromptTypedRevisionPolicy()
    valid_json = "[]"
    out = policy.parse_response(valid_json, example_id="ex_1", submission_id=fake_submission.submission_id)
    assert out.parsing_valid is True
    assert len(out.errors) == 0
    assert out.proposal_batches == ()


def test_parse_response_malformed_json(fake_submission):
    policy = PromptTypedRevisionPolicy()
    malformed = "[ { 'action_type': 'BLOCKS' } "  # Missing braces and brackets
    out = policy.parse_response(malformed, example_id="ex_1", submission_id=fake_submission.submission_id)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "Parsing failed" in out.errors[0]
    assert out.proposal_batches == ()


def test_parse_response_invalid_action(fake_submission):
    policy = PromptTypedRevisionPolicy()
    # AUTHORIZED is an evaluation status, not a Stage C proposal action!
    invalid_action_json = """
    [
      {
        "action_type": "AUTHORIZED",
        "target_belief_id": "b1",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(invalid_action_json, example_id="ex_1", submission_id=fake_submission.submission_id)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "not in the canonical vocabulary" in out.errors[0]


def test_parse_response_missing_fields(fake_submission):
    policy = PromptTypedRevisionPolicy()
    # Missing evidence_ids
    missing_evidence = """
    [
      {
        "action_type": "BLOCKS",
        "target_condition_id": "c1"
      }
    ]
    """
    out = policy.parse_response(missing_evidence, example_id="ex_1", submission_id=fake_submission.submission_id)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "requires a non-empty list of evidence_ids" in out.errors[0]
