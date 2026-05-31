"""DPA-in-the-loop reward tests."""
from __future__ import annotations

from retrace_learn.schemas import RevisionAction
from retrace_learn.data.build_synthetic_raw_dialogue import get_smoke_episode
from retrace_learn.runtime.dpa_runtime import run_from_text
from retrace_learn.runtime.learned_proposer import actions_to_json
from retrace_learn.runtime.reward import compute_reward_for_view


def _run(view, completion, gold_statuses, gold_actions):
    result = run_from_text(view, completion)
    return result, compute_reward_for_view(
        view, result, gold_statuses, gold_actions=gold_actions
    )


def test_gold_rollout_is_clean_and_highest():
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()
    _, b = _run(view, actions_to_json(ep.gold_actions), gold, list(ep.gold_actions))
    assert b.failure_category == "NONE"
    assert b.final_status_reward == 1.0
    assert b.valid_json_reward == 1.0
    assert b.parser_error_penalty == 0.0
    assert b.stale_propagation_penalty == 0.0


def test_parser_error_fail_closed():
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()
    result, b = _run(view, "definitely not json", gold, list(ep.gold_actions))
    assert b.failure_category == "PARSER_ERROR"
    assert b.parser_error_penalty == 1.0
    assert result.parse_result.valid_json is False
    # Fail-closed: no revisions admitted, no stale propagation introduced.
    assert result.final_belief_statuses["b_old"] == "AUTHORIZED"


def test_dropping_release_lowers_reward_vs_gold():
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()
    _, gold_b = _run(view, actions_to_json(ep.gold_actions), gold, list(ep.gold_actions))
    dropped = [a for a in ep.gold_actions if a.action_type != "RELEASES"]
    _, drop_b = _run(view, actions_to_json(dropped), gold, list(ep.gold_actions))
    assert drop_b.total_reward < gold_b.total_reward
    assert drop_b.final_status_reward < 1.0


def test_invalid_target_penalized():
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()
    bad = [
        RevisionAction(
            action_type="SUPERSEDES",
            target_belief_id="ghost",
            replacement_belief_id="phantom",
            evidence_ids=("ev2",),
        )
    ]
    _, b = _run(view, actions_to_json(bad), gold, list(ep.gold_actions))
    assert b.invalid_target_penalty > 0.0
    assert b.failure_category == "INVALID_TARGET"


def test_spurious_uncertain_penalized():
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()
    spurious = list(ep.gold_actions) + [
        RevisionAction(action_type="UNCERTAIN", target_belief_id="b_dep", evidence_ids=("ev3",))
    ]
    _, b = _run(view, actions_to_json(spurious), gold, list(ep.gold_actions))
    assert b.spurious_uncertain_penalty > 0.0
