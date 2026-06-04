"""Tests for hard-split expected_decision scheduling."""

from __future__ import annotations

from benchmark.retrace_bench.generation.hard_plus_blueprints import build_deterministic_scenario
from benchmark.retrace_bench.generation.pattern_spec import (
    HARD_DECISION_TARGET_FRACTIONS,
    build_hard_pattern_decision_plan,
    hard_decision_distribution,
    validate_pattern_semantics,
)


def test_hard150_decision_distribution_within_targets():
    count = 150
    seed = 2027
    dist = hard_decision_distribution(count, seed)
    total = sum(dist.values())
    assert total == count

    shares = {decision: dist[decision] / total for decision in dist}
    assert 0.45 <= shares["use_current_memory"] <= 0.55
    assert 0.15 <= shares["mark_unresolved"] <= 0.25
    assert 0.10 <= shares["ask_clarification"] <= 0.15
    assert 0.08 <= shares["refuse_due_to_policy"] <= 0.12
    assert 0.05 <= shares["escalate"] <= 0.10


def test_hard_plan_covers_required_pattern_decisions():
    plan = build_hard_pattern_decision_plan(150, 2027)
    by_pattern: dict[str, set[str]] = {}
    for pattern, decision in plan:
        by_pattern.setdefault(pattern, set()).add(decision)

    assert {"ask_clarification", "mark_unresolved", "escalate"} <= by_pattern[
        "ci_failed_after_claim"
    ]
    assert by_pattern["security_policy_override"] == {"refuse_due_to_policy"}
    assert "mark_unresolved" in by_pattern["authority_conflict"]
    assert "mark_unresolved" in by_pattern["negative_evidence_required"]
    assert "mark_unresolved" in by_pattern["closed_as_duplicate_not_fixed"]


def test_hard150_scenarios_semantically_valid_with_scheduled_decisions():
    for index in range(150):
        scenario = build_deterministic_scenario(index, "hard", 2027, split_count=150)
        errors = validate_pattern_semantics(scenario, scenario["hidden_gold"])
        assert errors == [], errors


def test_target_fractions_sum_to_one():
    assert abs(sum(HARD_DECISION_TARGET_FRACTIONS.values()) - 1.0) < 1e-9
