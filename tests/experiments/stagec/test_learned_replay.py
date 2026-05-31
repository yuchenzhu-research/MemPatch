import pytest

from retracemem.proposers.learned_replay import LearnedReplayProposer
from retracemem.evaluation.multiagent.contracts import FixedCandidateSubmission
from retracemem.schemas import EvidenceNode, BeliefNode, ConditionNode, DependencyEdge
from retracemem.authorization import authorize


def test_learned_replay_proposer_normal():
    # 1. Prepare candidate data
    ev_new = EvidenceNode(
        evidence_id="ev_new",
        session_id="sess_1",
        timestamp="2026-06-01T00:01:00Z",
        text="MySQL 8.0 upgrade success.",
        source_dataset="test",
        source_pointer="p",
    )
    b_old = BeliefNode(
        belief_id="b_old",
        proposition="Service database runs on MySQL 5.7",
        source_evidence_ids=("ev_old",),
    )
    b_new = BeliefNode(
        belief_id="b_new",
        proposition="Service database runs on MySQL 8.0",
        source_evidence_ids=("ev_new",),
    )

    submission = FixedCandidateSubmission(
        submission_id="sub_test_1",
        producer_id="migrator",
        producer_role="migrator",
        task_id=None,
        parent_snapshot_id="snap_0",
        observed_at="2026-06-01T00:01:00Z",
        instance_id="inst_1",
        query_id="q1",
        query="what db?",
        evidence_context=(ev_new,),
        new_evidence_id="ev_new",
        candidate_beliefs=(b_old,),
        candidate_replacement_beliefs=(b_new,),
    )

    # Predecoded dictionary matching actions
    predecoded = {
        "sub_test_1": [
            {
                "action_type": "SUPERSEDES",
                "target_belief_id": "b_old",
                "replacement_belief_id": "b_new",
                "evidence_ids": ["ev_new"],
                "rationale": "new db upgrade",
            }
        ]
    }

    proposer = LearnedReplayProposer(predecoded)
    output = proposer.propose(submission)

    assert output.parsing_valid
    assert len(output.errors) == 0
    assert len(output.parsed_actions) == 1
    assert output.parsed_actions[0].action_type == "SUPERSEDES"
    assert len(output.proposal_batches) == 1
    assert len(output.proposal_batches[0].edges) == 1
    assert output.proposal_batches[0].edges[0].edge_type.value == "SUPERSEDES"


def test_learned_replay_proposer_no_revision():
    submission = FixedCandidateSubmission(
        submission_id="sub_test_no_rev",
        producer_id="sentry",
        producer_role="sentry",
        task_id=None,
        parent_snapshot_id="snap_0",
        observed_at="2026-06-01T00:01:00Z",
        instance_id="inst_1",
        query_id="q1",
        query="what db?",
        evidence_context=(),
        new_evidence_id="ev_new",
        candidate_beliefs=(),
        candidate_replacement_beliefs=(),
    )

    predecoded = {
        "sub_test_no_rev": [
            {
                "action_type": "NO_REVISION",
                "target_belief_id": None,
                "replacement_belief_id": None,
                "evidence_ids": ["ev_new"],
            }
        ]
    }

    proposer = LearnedReplayProposer(predecoded)
    output = proposer.propose(submission)

    assert output.parsing_valid
    assert len(output.errors) == 0
    assert len(output.parsed_actions) == 1
    assert output.parsed_actions[0].action_type == "NO_REVISION"
    assert len(output.proposal_batches) == 0  # NO_REVISION creates no edges
