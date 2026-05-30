from __future__ import annotations

import json
import pytest
from retracemem.schemas import (
    EvidenceNode,
    BeliefNode,
    ConditionNode,
    DependencyEdge,
)
from experiments.multiagent.contracts import (
    FixedCandidateSubmission,
    TypedRevisionTarget,
    FixedCandidateGoldRecord,
    FixedCandidateInputEpisode,
    GoldSnapshotExpectation,
)
from experiments.multiagent.run_stageab_api_eval import (
    rename_episode_and_gold,
    render_direct_judge_prompt,
    parse_direct_judge_response,
    compute_stage_a_action_metrics,
    check_grounding_error_stage_a,
    check_grounding_error_stage_b,
)


@pytest.fixture
def mock_episode_and_gold():
    ev = EvidenceNode("ev_1", "sess_1", "2026-05-30T00:00:00Z", "Some evidence", "dataset", "pointer")
    b = BeliefNode("b_1", "Proposition 1", ("ev_1",))
    b2 = BeliefNode("b_2", "Proposition 2", ("ev_1",))
    c = ConditionNode("c_1", "scope_1", "Condition 1")
    dep = DependencyEdge("dep_1", "b_1", "c_1", "system")

    sub = FixedCandidateSubmission(
        submission_id="sub_1",
        producer_id="writer",
        producer_role="writer",
        task_id="task_1",
        parent_snapshot_id="snapshot_init",
        observed_at="2026-05-30T00:00:00Z",
        instance_id="inst_1",
        query_id="q_1",
        query="Check status?",
        evidence_context=(ev,),
        new_evidence_id="ev_1",
        candidate_beliefs=(b,),
        candidate_replacement_beliefs=(b2,),
        candidate_conditions_by_belief=(("b_1", (c,)),),
        dependency_edges_by_belief=(("b_1", (dep,)),),
    )

    episode = FixedCandidateInputEpisode(
        episode_id="ep_test",
        domain="software_engineering",
        failure_type_public_or_controlled="stale_propagation",
        subagent_roles=("writer",),
        submissions=(sub,),
        downstream_tasks=(),
    )

    gold_snapshot = GoldSnapshotExpectation(
        belief_statuses={"b_1": "SUPERSEDED"}
    )
    gold = FixedCandidateGoldRecord(
        episode_id="ep_test",
        gold_snapshot=gold_snapshot,
        gold_typed_targets=(
            TypedRevisionTarget("sub_1", "SUPERSEDES", target_belief_id="b_1", replacement_belief_id="b_2", evidence_ids=("ev_1",)),
        ),
        failure_type="stale_propagation",
    )

    return episode, gold


def test_rename_episode_and_gold(mock_episode_and_gold):
    ep, gold = mock_episode_and_gold
    ep_renamed, gold_renamed = rename_episode_and_gold(ep, gold)

    assert ep_renamed.episode_id == "ep_test__heldout_base"
    assert gold_renamed.episode_id == "ep_test__heldout_base"
    assert ep_renamed.submissions[0].submission_id == "sub_1"  # Not altered unless containing ep_test
    # Since belief_id and condition_id don't contain old_ns, they remain intact.
    # But if they did, they would be renamed.


def test_render_direct_judge_prompt(mock_episode_and_gold):
    ep, _ = mock_episode_and_gold
    sub = ep.submissions[0]
    template = "Query: {query}\nNew Evidence: {new_evidence_text}\nCandidate Beliefs:\n{candidate_beliefs}"
    rendered = render_direct_judge_prompt(template, sub)

    assert "Query: Check status?" in rendered
    assert "New Evidence: Some evidence" in rendered
    assert "b_1" in rendered


def test_parse_direct_judge_response():
    valid_belief_ids = {"b_1", "b_2"}
    response = """
    {
      "verdicts": [
        {"belief_id": "b_1", "status": "USABLE", "rationale": "ok", "confidence": 0.9},
        {"belief_id": "b_2", "status": "NOT_USABLE", "rationale": "stale", "confidence": 0.8}
      ]
    }
    """
    verdicts = parse_direct_judge_response(response, valid_belief_ids)
    assert len(verdicts) == 2
    assert verdicts[0]["belief_id"] == "b_1"
    assert verdicts[0]["status"] == "USABLE"
    assert verdicts[1]["belief_id"] == "b_2"
    assert verdicts[1]["status"] == "NOT_USABLE"


def test_parse_direct_judge_response_errors():
    valid_belief_ids = {"b_1"}
    response_missing = '{"verdicts": []}'
    with pytest.raises(ValueError, match="omitted verdicts"):
        parse_direct_judge_response(response_missing, valid_belief_ids)

    response_invalid_id = '{"verdicts": [{"belief_id": "b_unknown", "status": "USABLE"}]}'
    with pytest.raises(ValueError, match="references unknown belief ID"):
        parse_direct_judge_response(response_invalid_id, valid_belief_ids)


def test_check_grounding_error_stage_a(mock_episode_and_gold):
    ep, _ = mock_episode_and_gold
    sub = ep.submissions[0]

    # Correct action
    act_ok = {
        "action_type": "BLOCKS",
        "target_condition_id": "c_1",
        "evidence_ids": ["ev_1"]
    }
    assert check_grounding_error_stage_a(act_ok, sub) is False

    # Grounding error: invalid condition id
    act_err = {
        "action_type": "BLOCKS",
        "target_condition_id": "c_unknown",
        "evidence_ids": ["ev_1"]
    }
    assert check_grounding_error_stage_a(act_err, sub) is True


def test_check_grounding_error_stage_b():
    valid_belief_ids = {"b_1"}
    assert check_grounding_error_stage_b({"belief_id": "b_1"}, valid_belief_ids) is False
    assert check_grounding_error_stage_b({"belief_id": "b_2"}, valid_belief_ids) is True


def test_compute_stage_a_action_metrics(mock_episode_and_gold):
    ep, gold = mock_episode_and_gold
    sub = ep.submissions[0]

    pred_actions = [
        {
            "action_type": "SUPERSEDES",
            "target_belief_id": "b_1",
            "replacement_belief_id": "b_2",
            "evidence_ids": ["ev_1"],
        }
    ]

    metrics = compute_stage_a_action_metrics(pred_actions, gold.gold_typed_targets, sub)
    assert metrics["valid_json"] == 1.0
    assert metrics["exact_action_match"] == 1.0
    assert metrics["action_type_match"] == 1.0
    assert metrics["target_grounding"] == 1.0
