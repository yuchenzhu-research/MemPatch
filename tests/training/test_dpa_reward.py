import pytest

from retracemem.training.reward import score_rollout, classify_failure


def test_perfect_gold_score():
    gold_actions = [
        {"action_type": "SUPERSEDES", "target_belief_id": "b1", "replacement_belief_id": "b2", "evidence_ids": ["ev1"]},
        {"action_type": "BLOCKS", "target_condition_id": "c1", "evidence_ids": ["ev2"]},
    ]
    dpa_statuses = {
        "b1": "SUPERSEDED",
        "b2": "AUTHORIZED",
    }
    defeat_paths = [
        {"belief_id": "b1", "path_type": "DIRECT_SUPERSEDE"}
    ]

    breakdown = score_rollout(
        actions=gold_actions,
        valid_json=True,
        valid_vocabulary=True,
        dpa_final_statuses=dpa_statuses,
        gold_final_statuses=dpa_statuses,
        valid_belief_ids={"b1", "b2"},
        valid_condition_ids={"c1"},
        valid_evidence_ids={"ev1", "ev2"},
        defeat_paths=defeat_paths,
        gold_actions=gold_actions,
    )

    assert breakdown.total_reward > 0
    assert breakdown.failure_category == "NONE"
    assert breakdown.final_status_reward == 1.0
    assert breakdown.valid_json_reward == 1.0
    assert breakdown.valid_vocabulary_reward == 1.0
    assert breakdown.minimal_sufficient_action_set_reward == 1.0
    assert breakdown.audit_completeness_reward == 1.0


def test_parser_failure_penalty():
    gold_actions = [
        {"action_type": "SUPERSEDES", "target_belief_id": "b1", "replacement_belief_id": "b2", "evidence_ids": ["ev1"]}
    ]
    dpa_statuses = {
        "b1": "SUPERSEDED",
        "b2": "AUTHORIZED",
    }
    defeat_paths = [
        {"belief_id": "b1", "path_type": "DIRECT_SUPERSEDE"}
    ]

    breakdown = score_rollout(
        actions=gold_actions,
        valid_json=False,
        valid_vocabulary=True,
        dpa_final_statuses=dpa_statuses,
        gold_final_statuses=dpa_statuses,
        valid_belief_ids={"b1", "b2"},
        valid_condition_ids=set(),
        valid_evidence_ids={"ev1"},
        defeat_paths=defeat_paths,
        gold_actions=gold_actions,
    )

    assert breakdown.parser_error_penalty == 1.0
    assert breakdown.failure_category == "PARSER_ERROR"


def test_over_update_penalty():
    gold_actions = [
        {"action_type": "SUPERSEDES", "target_belief_id": "b1", "replacement_belief_id": "b2", "evidence_ids": ["ev1"]}
    ]
    # Gold status of b1 is AUTHORIZED, but predicted is SUPERSEDED (over-update)
    gold_statuses = {"b1": "AUTHORIZED"}
    pred_statuses = {"b1": "SUPERSEDED"}

    breakdown = score_rollout(
        actions=gold_actions,
        valid_json=True,
        valid_vocabulary=True,
        dpa_final_statuses=pred_statuses,
        gold_final_statuses=gold_statuses,
        valid_belief_ids={"b1", "b2"},
        valid_condition_ids=set(),
        valid_evidence_ids={"ev1"},
        defeat_paths=[],
        gold_actions=gold_actions,
    )

    assert breakdown.over_update_penalty > 0
    assert breakdown.failure_category == "OVER_UPDATE"


def test_stale_propagation_penalty():
    gold_actions = [
        {"action_type": "SUPERSEDES", "target_belief_id": "b1", "replacement_belief_id": "b2", "evidence_ids": ["ev1"]}
    ]
    # Gold status of b1 is SUPERSEDED, but predicted is AUTHORIZED (stale-propagation)
    gold_statuses = {"b1": "SUPERSEDED"}
    pred_statuses = {"b1": "AUTHORIZED"}

    breakdown = score_rollout(
        actions=[],
        valid_json=True,
        valid_vocabulary=True,
        dpa_final_statuses=pred_statuses,
        gold_final_statuses=gold_statuses,
        valid_belief_ids={"b1", "b2"},
        valid_condition_ids=set(),
        valid_evidence_ids={"ev1"},
        defeat_paths=[],
        gold_actions=gold_actions,
    )

    assert breakdown.stale_propagation_penalty > 0
    assert breakdown.failure_category == "STALE_PROPAGATION"
    assert breakdown.no_stale_propagation_reward == 0.0


def test_action_minimality_and_audit_completeness():
    gold_actions = [
        {"action_type": "BLOCKS", "target_condition_id": "c1", "evidence_ids": ["ev2"]},
    ]
    # Model generates two actions, which is redundant (non-minimal)
    actions = [
        {"action_type": "BLOCKS", "target_condition_id": "c1", "evidence_ids": ["ev2"]},
        {"action_type": "BLOCKS", "target_condition_id": "c1", "evidence_ids": ["ev2"]},
    ]
    pred_statuses = {"b1": "BLOCKED"}
    # b1 is blocked but there is no defeat path documented in audit (incomplete audit)
    defeat_paths = []

    breakdown = score_rollout(
        actions=actions,
        valid_json=True,
        valid_vocabulary=True,
        dpa_final_statuses=pred_statuses,
        gold_final_statuses=pred_statuses,
        valid_belief_ids={"b1"},
        valid_condition_ids={"c1"},
        valid_evidence_ids={"ev2"},
        defeat_paths=defeat_paths,
        gold_actions=gold_actions,
    )

    # minimal_sufficient_action_set_reward should be 0.5 because len(gold)=1 and len(actions)=2
    assert breakdown.minimal_sufficient_action_set_reward == 0.5
    # audit_completeness_reward should be 0.0 because b1 is blocked but not in defeat_paths
    assert breakdown.audit_completeness_reward == 0.0
