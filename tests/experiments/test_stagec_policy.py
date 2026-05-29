from __future__ import annotations

import pytest
from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
from experiments.multiagent.contracts import FixedCandidateSubmission

@pytest.fixture
def fake_submission() -> FixedCandidateSubmission:
    from retracemem.schemas import EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
    ev = EvidenceNode(
        evidence_id="ev_1", session_id="s1", timestamp="2026-05-29T10:00:00Z",
        text="Deploy us-east-1", source_dataset="t", source_pointer="p"
    )
    b = BeliefNode(belief_id="b1", proposition="Deploy us-east-1", source_evidence_ids=("ev_1",))
    b_rep = BeliefNode(belief_id="b2", proposition="Deploy us-west-2", source_evidence_ids=("ev_1",))
    c = ConditionNode(condition_id="c1", scope_id="scope1", text="Server is active")
    dep = DependencyEdge(edge_id="d1", belief_id="b1", condition_id="c1", inducer="test")
    
    return FixedCandidateSubmission(
        submission_id="sub_test_01", producer_id="agent_1", producer_role="role",
        task_id="t1", parent_snapshot_id="snap_0", observed_at="2026-05-29T10:00:00Z",
        instance_id="inst_1", query_id="q_1", query="Q",
        evidence_context=(ev,), new_evidence_id="ev_1",
        candidate_beliefs=(b,),
        candidate_replacement_beliefs=(b_rep,),
        candidate_conditions_by_belief=((b.belief_id, (c,)),),
        dependency_edges_by_belief=((b.belief_id, (dep,)),),
    )


def test_build_messages(fake_submission):
    policy = PromptTypedRevisionPolicy()
    messages = policy.build_messages(fake_submission)
    
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "You are the ReTrace Stage C revision policy" in messages[0]["content"]
    assert "sub_test_01" in messages[1]["content"]
    # Check that visible condition text and dependency requires anchors are present in the user prompt
    assert "Server is active" in messages[1]["content"]
    assert "b1 --REQUIRES--> c1" in messages[1]["content"]


def test_parse_response_supersedes(fake_submission):
    policy = PromptTypedRevisionPolicy()
    valid_json = """
    [
      {
        "action_type": "SUPERSEDES",
        "target_belief_id": "b1",
        "replacement_belief_id": "b2",
        "rationale": "Superseding old info.",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(valid_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is True
    assert len(out.errors) == 0
    assert len(out.proposal_batches) == 1
    edge = out.proposal_batches[0].edges[0]
    assert edge.edge_type.value == "SUPERSEDES"
    assert edge.target_id == "b1"
    assert edge.replacement_belief_id == "b2"
    assert edge.evidence_id == "ev_1"


def test_parse_response_blocks(fake_submission):
    policy = PromptTypedRevisionPolicy()
    valid_json = """
    [
      {
        "action_type": "BLOCKS",
        "target_condition_id": "c1",
        "rationale": "Blocking condition.",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(valid_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is True
    assert len(out.errors) == 0
    assert len(out.proposal_batches) == 1
    edge = out.proposal_batches[0].edges[0]
    assert edge.edge_type.value == "BLOCKS"
    assert edge.target_kind == "condition"
    assert edge.target_id == "c1"
    assert edge.evidence_id == "ev_1"


def test_parse_response_no_revision(fake_submission):
    policy = PromptTypedRevisionPolicy()
    valid_json = """
    [
      {
        "action_type": "NO_REVISION",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(valid_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is True
    assert len(out.errors) == 0
    assert out.proposal_batches == ()
    assert len(out.parsed_actions) == 1
    assert out.parsed_actions[0].action_type == "NO_REVISION"


def test_parse_response_empty_is_rejected(fake_submission):
    policy = PromptTypedRevisionPolicy()
    out = policy.parse_response("[]", example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "Parsed action array is empty" in out.errors[0]


def test_parse_response_malformed_json(fake_submission):
    policy = PromptTypedRevisionPolicy()
    malformed = "[ { 'action_type': 'BLOCKS' } "  # Missing braces and brackets
    out = policy.parse_response(malformed, example_id="ex_1", submission=fake_submission)
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
    out = policy.parse_response(invalid_action_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "is a final DPA status" in out.errors[0]


def test_parse_response_out_of_bounds_ids(fake_submission):
    policy = PromptTypedRevisionPolicy()
    # Target belief ID b_unknown is not in candidate beliefs
    oob_json = """
    [
      {
        "action_type": "UNCERTAIN",
        "target_belief_id": "b_unknown",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(oob_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "is not in candidate beliefs" in out.errors[0]


def test_parse_response_duplicate_deduplication(fake_submission):
    policy = PromptTypedRevisionPolicy()
    dup_json = """
    [
      {
        "action_type": "BLOCKS",
        "target_condition_id": "c1",
        "evidence_ids": ["ev_1"]
      },
      {
        "action_type": "BLOCKS",
        "target_condition_id": "c1",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(dup_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is True
    assert len(out.errors) == 0
    # Duplicate should be removed, leaving exactly 1 edge in batch
    assert len(out.proposal_batches[0].edges) == 1
    assert out.metadata.get("has_duplicates") is True


def test_parse_response_incompatible_actions(fake_submission):
    policy = PromptTypedRevisionPolicy()
    # Condition c1 is both BLOCKS and RELEASES
    incompatible_json = """
    [
      {
        "action_type": "BLOCKS",
        "target_condition_id": "c1",
        "evidence_ids": ["ev_1"]
      },
      {
        "action_type": "RELEASES",
        "target_condition_id": "c1",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(incompatible_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "Conflict: Condition 'c1' is both BLOCKED and RELEASED." in out.errors[0]


def test_allowed_actions_filtering(fake_submission):
    # Only allow SUPERSEDES and NO_REVISION
    policy = PromptTypedRevisionPolicy(allowed_actions=("SUPERSEDES", "NO_REVISION"))
    
    # Try to parse BLOCKS (which is not allowed)
    valid_blocks_json = """
    [
      {
        "action_type": "BLOCKS",
        "target_condition_id": "c1",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    out = policy.parse_response(valid_blocks_json, example_id="ex_1", submission=fake_submission)
    assert out.parsing_valid is False
    assert len(out.errors) == 1
    assert "not allowed in current vocabulary configuration" in out.errors[0]


def test_icl_proposer_retrieval_and_propose(fake_submission):
    from experiments.multiagent.stagec_policy import ClosedAPIICLProposer
    from experiments.multiagent.contracts import ApprovedRevisionExemplar
    from retracemem.providers.base import MockLLMProvider
    
    valid_json = """
    [
      {
        "action_type": "NO_REVISION",
        "target_belief_id": null,
        "target_condition_id": null,
        "replacement_belief_id": null,
        "rationale": "No revision needed",
        "evidence_ids": ["ev_1"]
      }
    ]
    """
    mock_client = MockLLMProvider(default_response=valid_json)
    proposer = ClosedAPIICLProposer(
        provider_kind="mock",
        model_id="mock-model",
        client=mock_client,
        top_k=1,
    )
    
    exemplar = ApprovedRevisionExemplar(
        exemplar_id="ex_1",
        source_episode_id="ep_1",
        domain="software_engineering",
        failure_type="direct_supersession",
        method_visible_input=fake_submission,
        approved_typed_actions=(),
        reviewer="Test Reviewer",
        review_manifest_hash="dummy_hash",
        training_or_icl_eligibility="eligible",
    )
    
    selected = proposer.retrieve_exemplars(fake_submission, (exemplar,))
    assert len(selected) == 1
    assert selected[0].exemplar_id == "ex_1"
    
    # Propose should run and return proposal policy output
    out = proposer.propose(fake_submission, exemplars=(exemplar,))
    assert out.parsing_valid is True
    assert len(out.parsed_actions) == 1
    assert out.parsed_actions[0].action_type == "NO_REVISION"
