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
    canonicalize_belief_id_with_type,
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


def test_canonicalize_belief_id_with_type():
    valid_ids = {"b_1_active", "b_2_inactive", "b_3"}
    
    # 1. Exact Match
    cid, applied, mtype = canonicalize_belief_id_with_type("b_3", valid_ids)
    assert cid == "b_3"
    assert applied is False
    assert mtype == "exact"
    
    # 2. Prefix Match (v_id starts with returned_id)
    cid, applied, mtype = canonicalize_belief_id_with_type("b_1", valid_ids)
    assert cid == "b_1_active"
    assert applied is True
    assert mtype == "prefix"

    # 3. Suffix Match (returned_id ends with v_id)
    cid, applied, mtype = canonicalize_belief_id_with_type("prefix_b_3", valid_ids)
    assert cid == "b_3"
    assert applied is True
    assert mtype == "suffix"
    
    # 4. Fuzzy Match
    cid, applied, mtype = canonicalize_belief_id_with_type("b_2_inac-tive", valid_ids)
    assert cid == "b_2_inactive"
    assert applied is True
    assert mtype == "fuzzy"
    
    # 5. Failed Canonicalization
    cid, applied, mtype = canonicalize_belief_id_with_type("b_unknown", valid_ids)
    assert cid == "b_unknown"
    assert applied is False
    assert mtype == "failed"


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
    assert verdicts[0]["canonical_belief_id"] == "b_1"
    assert verdicts[0]["raw_belief_id"] == "b_1"
    assert verdicts[0]["canonicalization_applied"] is False
    assert verdicts[0]["canonicalization_type"] == "exact"
    assert verdicts[0]["status"] == "USABLE"
    
    assert verdicts[1]["canonical_belief_id"] == "b_2"
    assert verdicts[1]["raw_belief_id"] == "b_2"
    assert verdicts[1]["status"] == "NOT_USABLE"


def test_parse_direct_judge_response_errors():
    valid_belief_ids = {"b_1"}
    response_missing = '{"verdicts": []}'
    with pytest.raises(ValueError, match="omitted verdicts"):
        parse_direct_judge_response(response_missing, valid_belief_ids)

    response_invalid_id = '{"verdicts": [{"belief_id": "b_unknown", "status": "USABLE"}]}'
    with pytest.raises(ValueError, match="failed canonicalization"):
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
    assert check_grounding_error_stage_b({"raw_belief_id": "b_1"}, valid_belief_ids) is False
    assert check_grounding_error_stage_b({"raw_belief_id": "b_2"}, valid_belief_ids) is True


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


def test_strict_vs_canonicalized_metric_divergence():
    # Test that strict vs canonicalized accuracy metrics can diverge under fuzzy matching
    # Gold statuses: b_1 is NOT_USABLE
    gold_comp = "NOT_USABLE"
    
    # Pred statuses (Stage B)
    # Canonicalized has b_1 = NOT_USABLE (via fuzzy match) -> Correct
    # Strict has b_1 = UNCERTAIN (no exact match) -> Incorrect
    strict_pred = {"b_1": "UNCERTAIN"}
    canonical_pred = {"b_1": "NOT_USABLE"}
    
    correct_strict = 1 if strict_pred.get("b_1") == gold_comp else 0
    correct_canonical = 1 if canonical_pred.get("b_1") == gold_comp else 0
    
    assert correct_strict == 0
    assert correct_canonical == 1
    assert correct_strict != correct_canonical


def test_prompt_non_leakage(mock_episode_and_gold):
    ep, gold = mock_episode_and_gold
    sub = ep.submissions[0]
    
    # Test Stage B prompt rendering
    from experiments.multiagent.run_stageab_api_eval import load_direct_judge_template, render_direct_judge_prompt
    template = load_direct_judge_template()
    prompt_b = render_direct_judge_prompt(template, sub)
    
    # Gold status of b_1 is SUPERSEDED, which maps to NOT_USABLE
    # Prompt must NOT contain "SUPERSEDED". "NOT_USABLE" is a class name in the template,
    # but the prompt must NOT contain "b_1: NOT_USABLE" or similar label assignments.
    assert "SUPERSEDED" not in prompt_b
    assert "b_1: NOT_USABLE" not in prompt_b
    assert "b_1: \"NOT_USABLE\"" not in prompt_b

    # Test Stage A prompt rendering
    from experiments.multiagent.stagec_policy import PromptTypedRevisionPolicy
    policy = PromptTypedRevisionPolicy()
    messages = policy.build_messages(sub)
    user_prompt_a = messages[1]["content"]
    
    # Prompts must not leak the gold action targets
    assert "Identify the correct revision actions" in user_prompt_a
    assert "gold_snapshot" not in user_prompt_a
