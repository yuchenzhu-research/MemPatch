"""Tests for structured engine error contracts."""
from __future__ import annotations

import json

from retrace_learn.runtime.engine_errors import (
    EngineError,
    EngineStage,
    ErrorSeverity,
    PARSER_INVALID_JSON,
    PARSER_ITEM_NOT_OBJECT,
    PARSER_SCHEMA_VIOLATION,
)
from retrace_learn.schemas import RevisionAction
from retrace_learn.data.build_synthetic_raw_dialogue import get_smoke_episode
from retrace_learn.runtime.dpa_runtime import ParseResult, RuntimeResult, parse_actions, run_from_text
from retrace_learn.runtime.learned_proposer import actions_to_json
from retrace_learn.runtime.reward import (
    RewardWeights,
    compute_reward,
    compute_reward_for_view,
)


# ---------------------------------------------------------------------------
# EngineError construction and serialization
# ---------------------------------------------------------------------------


def test_engine_error_construction():
    err = EngineError(
        stage=EngineStage.PARSER,
        code=PARSER_INVALID_JSON,
        message="bad json",
    )
    assert err.stage == EngineStage.PARSER
    assert err.code == PARSER_INVALID_JSON
    assert err.severity == ErrorSeverity.ERROR
    assert err.fail_closed is True
    assert err.action_index is None
    assert err.evidence_ids == ()


def test_engine_error_to_dict():
    err = EngineError(
        stage=EngineStage.REVISION_GATE,
        code="GATE_UNKNOWN_BELIEF",
        message="belief not found",
        severity=ErrorSeverity.WARNING,
        fail_closed=False,
        belief_id="b1",
        action_index=2,
    )
    d = err.to_dict()
    assert d["stage"] == "REVISION_GATE"
    assert d["severity"] == "WARNING"
    assert d["belief_id"] == "b1"
    assert d["action_index"] == 2
    assert d["fail_closed"] is False


# ---------------------------------------------------------------------------
# Parser errors propagated to ParseResult
# ---------------------------------------------------------------------------


def test_parse_result_carries_errors_on_json_failure():
    result = parse_actions("definitely not valid json {{{{")
    assert result.valid_json is False
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.stage == EngineStage.PARSER
    assert err.code == PARSER_INVALID_JSON


def test_parse_result_carries_errors_on_schema_violation():
    # Valid JSON array but invalid action schema (missing action_type).
    raw = json.dumps([{"target_belief_id": "b1"}])
    result = parse_actions(raw)
    assert result.valid_json is True
    assert result.schema_valid is False
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.stage == EngineStage.PARSER
    assert err.code == PARSER_SCHEMA_VIOLATION


def test_parse_result_carries_errors_on_item_not_object():
    raw = json.dumps(["not_an_object", 42])
    result = parse_actions(raw)
    assert result.valid_json is True
    assert result.schema_valid is False
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.stage == EngineStage.PARSER
    assert err.code == PARSER_ITEM_NOT_OBJECT
    assert err.action_index == 0


def test_parse_result_no_errors_on_valid():
    ep = get_smoke_episode()
    raw = actions_to_json(ep.gold_actions)
    result = parse_actions(raw)
    assert result.valid_json is True
    assert result.schema_valid is True
    assert result.errors == ()


def test_parse_result_to_dict_includes_errors():
    result = parse_actions("not json")
    d = result.to_dict()
    assert "errors" in d
    assert len(d["errors"]) == 1
    assert d["errors"][0]["stage"] == "PARSER"


# ---------------------------------------------------------------------------
# RuntimeResult aggregates gate rejection errors
# ---------------------------------------------------------------------------


def test_runtime_result_aggregates_parser_errors():
    """Invalid JSON -> parser error surfaces in RuntimeResult.engine_errors."""
    ep = get_smoke_episode()
    view = ep.build_view()
    result = run_from_text(view, "not json at all")
    assert len(result.engine_errors) >= 1
    parser_errs = [e for e in result.engine_errors if e.stage == EngineStage.PARSER]
    assert len(parser_errs) == 1
    assert parser_errs[0].code == PARSER_INVALID_JSON


def test_runtime_result_gate_rejection_errors():
    """Gate-rejected edges produce REVISION_GATE errors in engine_errors."""
    ep = get_smoke_episode()
    view = ep.build_view()
    # Use an ungrounded SUPERSEDES to trigger a gate rejection.
    bad_actions = [
        RevisionAction(
            action_type="SUPERSEDES",
            target_belief_id="ghost_belief",
            replacement_belief_id="phantom_belief",
            evidence_ids=("ev2",),
        )
    ]
    result = run_from_text(view, actions_to_json(bad_actions))
    gate_errs = [e for e in result.engine_errors if e.stage == EngineStage.REVISION_GATE]
    # The gate should reject the SUPERSEDES with an unknown target.
    # If admitted (kernel may differ), gate_errs could be empty — that's OK.
    # What matters is the wiring: if any gate_decision has admitted=False,
    # there must be a corresponding EngineError.
    rejected = [gd for gd in result.gate_decisions if not gd.get("admitted", True)]
    assert len(gate_errs) == len(rejected)


def test_runtime_result_to_dict_includes_engine_errors():
    ep = get_smoke_episode()
    result = run_from_text(ep.build_view(), "broken json")
    d = result.to_dict()
    assert "engine_errors" in d
    assert isinstance(d["engine_errors"], list)


# ---------------------------------------------------------------------------
# Reward: gate rejection penalty
# ---------------------------------------------------------------------------


def test_reward_gate_rejection_penalty():
    """Gate rejection errors should trigger a reward penalty."""
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()

    # Gold rollout: no gate rejections.
    gold_result = run_from_text(view, actions_to_json(ep.gold_actions))
    gold_b = compute_reward_for_view(view, gold_result, gold, gold_actions=list(ep.gold_actions))
    assert gold_b.gate_rejection_penalty == 0.0

    # If we fabricate engine_errors for testing compute_reward directly:
    valid_belief_ids = {b.belief_id for b in view.candidate_beliefs}
    valid_belief_ids |= {b.belief_id for b in view.candidate_replacement_beliefs}
    valid_condition_ids = {
        c.condition_id for _bid, conds in view.candidate_conditions_by_belief for c in conds
    }
    valid_evidence_ids = {e.evidence_id for e in view.evidence_context}

    fake_gate_err = EngineError(
        stage=EngineStage.REVISION_GATE,
        code="GATE_SUPERSEDES_REJECTED",
        message="rejected",
    )
    actions = list(ep.gold_actions)
    parse_result = ParseResult(valid_json=True, schema_valid=True, actions=tuple(actions))
    b = compute_reward(
        actions=actions,
        parse_result=parse_result,
        dpa_final_statuses=gold_result.final_belief_statuses,
        gold_final_statuses=gold,
        valid_belief_ids=valid_belief_ids,
        valid_condition_ids=valid_condition_ids,
        valid_evidence_ids=valid_evidence_ids,
        gold_actions=actions,
        engine_errors=(fake_gate_err,),
    )
    assert b.gate_rejection_penalty > 0.0


# ---------------------------------------------------------------------------
# Reward: NO_REVISION overuse penalty
# ---------------------------------------------------------------------------


def test_reward_no_revision_overuse_penalty():
    """All NO_REVISION when gold expects revisions -> overuse penalty."""
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()
    gold_actions = list(ep.gold_actions)

    # All NO_REVISION actions.
    no_rev_actions = [
        RevisionAction(action_type="NO_REVISION"),
        RevisionAction(action_type="NO_REVISION"),
        RevisionAction(action_type="NO_REVISION"),
    ]
    parse_result = ParseResult(
        valid_json=True, schema_valid=True, actions=tuple(no_rev_actions)
    )

    valid_belief_ids = {b.belief_id for b in view.candidate_beliefs}
    valid_belief_ids |= {b.belief_id for b in view.candidate_replacement_beliefs}
    valid_condition_ids = {
        c.condition_id for _bid, conds in view.candidate_conditions_by_belief for c in conds
    }
    valid_evidence_ids = {e.evidence_id for e in view.evidence_context}

    b = compute_reward(
        actions=no_rev_actions,
        parse_result=parse_result,
        dpa_final_statuses={bid: "AUTHORIZED" for bid in gold},
        gold_final_statuses=gold,
        valid_belief_ids=valid_belief_ids,
        valid_condition_ids=valid_condition_ids,
        valid_evidence_ids=valid_evidence_ids,
        gold_actions=gold_actions,
    )
    # 100% NO_REVISION -> ratio=1.0 > 0.5 -> penalty = (1.0-0.5)*2 = 1.0
    assert b.no_revision_overuse_penalty == 1.0


def test_reward_no_revision_overuse_zero_when_no_gold():
    """No penalty when gold_actions is None (nothing to compare against)."""
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()

    no_rev_actions = [RevisionAction(action_type="NO_REVISION")]
    parse_result = ParseResult(
        valid_json=True, schema_valid=True, actions=tuple(no_rev_actions)
    )

    valid_belief_ids = {b.belief_id for b in view.candidate_beliefs}
    valid_condition_ids = {
        c.condition_id for _bid, conds in view.candidate_conditions_by_belief for c in conds
    }
    valid_evidence_ids = {e.evidence_id for e in view.evidence_context}

    b = compute_reward(
        actions=no_rev_actions,
        parse_result=parse_result,
        dpa_final_statuses={bid: "AUTHORIZED" for bid in gold},
        gold_final_statuses=gold,
        valid_belief_ids=valid_belief_ids,
        valid_condition_ids=valid_condition_ids,
        valid_evidence_ids=valid_evidence_ids,
        gold_actions=None,
    )
    assert b.no_revision_overuse_penalty == 0.0


# ---------------------------------------------------------------------------
# End-to-end: invalid JSON -> errors in RuntimeResult -> reward penalties
# ---------------------------------------------------------------------------


def test_end_to_end_invalid_json_errors_in_result():
    ep = get_smoke_episode()
    view = ep.build_view()
    gold = ep.gold_final_statuses()

    result = run_from_text(view, "{{not json}}")
    assert result.parse_result.valid_json is False
    assert len(result.engine_errors) >= 1

    b = compute_reward_for_view(view, result, gold, gold_actions=list(ep.gold_actions))
    assert b.parser_error_penalty == 1.0
    assert b.failure_category == "PARSER_ERROR"
    # No mutations were admitted, so no gate rejections.
    assert b.gate_rejection_penalty == 0.0
